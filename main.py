# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import operator
import torch.backends.cudnn as cudnn
import json
# import timm.models

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset, GroupedDataset
from engine import train_one_epoch, evaluate, test
from samplers import RASampler

import models
import utils
import random

# integrate with wandb and hydra
import hydra
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf


class CustomClassifier(torch.nn.Module):
    def __init__(self, model, input_dim, output_dim, 
                    multi_dataset_classes=None, known_data_source=False, model_name=None):
        '''
        Custom classifier with a Norm layer followed by a Linear layer.
        '''
        super().__init__()
        self.backbone = model
        self.model_name = model_name

        self.known_data_source = known_data_source
        self.multi_dataset_classes = multi_dataset_classes  # 一个 list, 包含各数据集 classes 数目.
        
        self.channel_bn = torch.nn.BatchNorm1d(
            input_dim,
            affine=False,
        )
        self.channel_pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(start_dim=1)
        )
        
        self.head = torch.nn.Linear(input_dim, output_dim)
        self.head_dist = torch.nn.Linear(input_dim, output_dim)

    def forward(self, img, dataset_id=None):
        '''dataset_id 这个 input 目前没有被用到; 后续改进时需要思考如何使用它.
        model 应该根据 self.known_data_source 是否为 True 来决定是否能在 forward 中使用 dataset_id 的信息.'''
        # TODO: how to leverage dataset_source in training and inference stage?
        # timm models 的 forward 分为 forward_features 与 forward_head 两部分; 
        # forward_features 返回 backbone 处理好的 representation; forward_head 再把它送入最后的线性分类头输出分类向量.
        pdtype = img.dtype
        feature = self.backbone.forward_features(img).to(pdtype)

        # 1. vit/deit(no distillation): 只有 cls_token
        if self.model_name in {'deit_small_patch16_224', 'deit_base_patch16_224'}:
            outputs = self.channel_bn(torch.squeeze(feature[:, 0, :], dim=1))
            outputs = self.head(outputs)
            return outputs

        # 2. deit(distilled): cls_token + dist_token
        elif self.model_name in {'deit_small_distilled_patch16_224', 'deit_base_distilled_patch16_224'}:
            cls, dist = feature[:, 0], feature[:, 1]
            cls = self.head(cls)
            dist = self.head_dist(dist)
            return (cls + dist) / 2

        # 3. swin-transformer: (N,L,C) + mean(dim=1)
        elif self.model_name in {'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224'}:
            feature = feature.mean(dim=1)
            feature = self.head(feature)
            return feature

        # 4. EfficientNet 类模型 forward_features 输出的形状为 (N, 1280, H, W);
        elif 'efficientnet' in self.model_name:
            outputs = self.channel_pool(feature)
            outputs = self.head(outputs)
            return outputs
        
        else:
            return NotImplementedError


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--known_data_source', action='store_true', dest='known_data_source', default=True)
    parser.add_argument('--unknown_data_source', action='store_false', dest='known_data_source', default=True)
    parser.add_argument('--dataset_list', type=str, nargs='+',
                            default=['10shot_cifar100_20200721', '10shot_country211_20210924', '10shot_food_101_20211007', '10shot_oxford_iiit_pets_20211007', '10shot_stanford_cars_20211007'])

    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=0, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=0, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
        
    parser.add_argument('--src', action='store_true') #simple random crop
    parser.add_argument('--flip', type=float, default=None, metavar='PCT',
                        help='flip image, both VerticalFlip and HorizontalFlip')
    
    parser.add_argument('--rotation', type=int, default=None, metavar='PCT',
                        help='image Rotation')
    


    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    
    # Dataset parameters
    # modif: 已经把默认的 data_path 换为数据集所在目录.
    parser.add_argument('--data-path', default='/remote-home/share/course23/aicourse_dataset_final/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


@hydra.main(version_base=None, config_path="configs/", config_name="test")
def main(args) -> None:
    
    # modif 4: 把 wandb 接入 Hydra config files
    wandb_cfg = OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=args.wandb.setup.entity, project=args.wandb.setup.project, config=wandb_cfg)
    
    # 把原来 main 函数外的 mkdir 操作封装到 main 函数中完成.
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(args.dataset_list)
    utils.init_distributed_mode(args)

    print(OmegaConf.to_yaml(args))  # modif 2
    # print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # args.nb_classes 是所有 dataset 的 classes 数目的总和.
    # build_dataset 返回的 dataset_train 是一个 MultiImageFolder.
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, *_ = build_dataset(is_train=False, args=args)

    # TODO: 搞清楚下面是否只有在 args.distributed = True 的情况下才应该执行;
    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, num_repeats=3
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 用 dataset_train 这个 MultiIMageFolder 来构造 dataloader_train;
    # 不在 dataloader 中声明 batch_size 的话, 默认 batch_size=1.
    # collate_fn = operator.itemgetter(0), 表示每个样本独立返回, 不用合并成一个 tensor(batch)
    if args.known_data_source :
        # 下面第一次定义的 dataloader_train 只是过渡性的, batch_size=1, 是为了
        # 第二次定义中取 samples 方便;
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_sampler=None,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=operator.itemgetter(0), 
        )
        
        # 第二次定义 dataloader_train, 最终返回的其实是一个 IterableDataset, 
        # 把同一个 dataset 中 容量=batch_size 的 images, targets, dataset_ids 分别作为三个列表返回.
        data_loader_train = GroupedDataset(data_loader_train, args.batch_size, len(args.dataset_list))
    else :
        # 若 args.known_data_source = False, 则直接从混合的若干个数据集(即 dataset_train)中, 
        # 任意取出 batch_size 个样本.
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

    data_loader_val_list = []
    dataset_val_total = dataset_val
    # 这样写更好, 不容易引起歧义: for dataset_val in dataset_val_total.dataset_list: 
    for dataset_val in dataset_val.dataset_list:
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(2 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        data_loader_val_list.append(data_loader_val)

    print(f"Creating model: {args.model}")
    
    # Create a model with the timm function; any other model pre-trained under ImageNet-1k is allowed.
    # TODO: 任意用 ImageNet-1k 预训练的模型都可以, 也没有限制预训练的方式; 因此不一定要用 timm 提供的预训练参数;
    # 可以用 supervised-FSL 的方式, 在 imagenet-1k 上有监督地预训练, 之后再利用本任务提供的少量 labeled data 与
    # 较多 unlabeled data 来 fine-tune.
    model = create_model(args.model, num_classes=args.nb_classes, pretrained=True)
    
    # number of classes for each dataset
    multi_dataset_classes = [len(x) for x in dataset_train.classes_list]

    # 1. For vit/deit from timm: 
    if 'deit' in args.model or 'vit' in args.model:
        model = CustomClassifier(
            model, model.embed_dim, args.nb_classes, multi_dataset_classes=multi_dataset_classes, 
            known_data_source=args.known_data_source, model_name=args.model)

    # 2. For swin-transformer/ConvNets from timm:
    else:
        model = CustomClassifier(
            model, model.num_features, args.nb_classes, multi_dataset_classes=multi_dataset_classes, 
            known_data_source=args.known_data_source, model_name=args.model)
    
    model.to(device)

    # ema 是 Exponential Moving Average 的简称;
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # ddp 是 DistributedDataParallel 的简称, 标记是否为分布式并行训练.
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    backbone_params = [x for name, x in model_without_ddp.named_parameters() if 'backbone' in name]
    custom_params = [x for name, x in model_without_ddp.named_parameters() if 'backbone' not in name]

    # use smaller lr for backbone params
    # backbone parameters 采用 0.1*args.lr, 而 custom_params 采用默认的 args.lr.
    params = [
                {'params': backbone_params, 'lr': args.lr * 0.1},
                {'params': custom_params}
    ]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    loss_scaler = NativeScaler()
    
    # create_scheduler 函数根据 args.sched 参数, 返回不同种类的 lr_scheduler;
    # 默认的是 cosine.
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    # 下面从 url 下载 checkpoints, 并且分别把 checkpoint 中的 'model', 'optimizer', 'lr_scheduler' 等都 load 入对应模块.
    # TODO: 搞懂 checkpoint 的下载和 load 逻辑.
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    
    # 在 test_only 的指令下, 会把 pred_all.json 文件生成在 output_dir 目录下.
    if args.test_only:
        # the format of submitted json
        # {
        #   'n_parameters': n_parameters,
        #   'dataset_1': {id_1:pred_1, id_2:pred2, ...},
        #    ...,
        #   'dataset_n': ...,
        # }
        pred_path = str(output_dir) + "/" + "pred_all.json"
        result_list = {}
        result_list['n_parameters'] = n_parameters
        for dataset_id, data_loader_val in enumerate(data_loader_val_list):
            pred_json = test(data_loader_val, model, device, dataset_id, num_classes_list=multi_dataset_classes,
                                know_dataset=args.known_data_source)
            result_list[args.dataset_list[dataset_id]] = pred_json
        with open(pred_path, 'w') as f:
            json.dump(result_list, f)
        return

    # args.eval: 表明只做 evaluation, 不做 training.
    if args.eval:
        for dataset_id, data_loader_val in enumerate(data_loader_val_list):
            test_stats = evaluate(data_loader_val, model, device, dataset_id, args=args)
            print(f"Accuracy of the network on {args.dataset_list[dataset_id]} of {len(dataset_val_total.dataset_list[dataset_id])} "
                    f"test images: {test_stats[f'dataset_{dataset_id}_test_acc1']:.1f}%")

        return

    # 下面才是正常的训练流程; 其中 train / evaluate / test 等函数都在 engine.py 中定义好了.
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        # TODO: Consistent lr now
        # how to use a lr scheduler for better convergence.
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
        
        # 默认每 10 个 epochs 在 validation set 上做一次 evaluate.
        if (epoch + 1) % args.test_interval == 0 or epoch + 1 == args.epochs :
            test_stats_total = {}
            test_stats_list = []
            for dataset_id, data_loader_val in enumerate(data_loader_val_list):
                test_stats = evaluate(data_loader_val, model, device, dataset_id, args=args)
                test_stats_list.append(test_stats)
                print(f"Accuracy of the network on {args.dataset_list[dataset_id]} of {len(dataset_val_total.dataset_list[dataset_id])} test images: {test_stats[f'dataset_{dataset_id}_test_acc1']:.1f}%")
                for k, v in test_stats.items():
                    test_stats_total['dataset_{}_{}'.format(args.dataset_list[dataset_id], k)] = v

            sum_acc = sum([x[f'dataset_{dataset_id}_test_acc1'] for dataset_id, x in enumerate(test_stats_list)])
            if max_accuracy < sum_acc:
                max_accuracy = sum_acc
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'model_ema': get_state_dict(model_ema),
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)
                
            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats_total.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    wandb.finish()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # main(args)
    
    # 用了 Hydra 之后, args 参数由 Hydra 根据 config 文件自动生成.
    main()  # modif 3
