# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import wandb


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, set_training_mode=True, args = None,
                    class_indicator=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 第一个 meter: lr; 规定了 fmt 的格式: 只有每次的精确值; 不像 loss 所采用的默认格式: median (global_avg)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10 # 每 <print_freq> 个 iterations 输出一次实验结果.

    for data in metric_logger.log_every(data_loader, print_freq, header):
        samples, targets, dataset_ids = data
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        with torch.cuda.amp.autocast():
            outputs = model(samples, dataset_ids)
            if class_indicator is not None :
                mask = class_indicator[targets]
                outputs[~mask.bool()] = -1e2 # applying bitwise negation to the binary representation of mask.bool()

            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # 每个 iteration 更新一次 loss 与 lr;
        # 第二个 meter: loss. loss 是由 metric_logger.update() 新创建的, 具体原理是:
        # metric_logger.update(loss=loss_value) 调用了 self.meters[loss].update(loss_value),
        # 这就进一步调用了 self.meters: DefaultDict 的默认 key-value 生成方法: SmoothedValue, 
        # 因此生成的以 loss 为 key 的 meter 用的是 SmoothedValue 类的默认输出格式.
        metric_logger.update(train_loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        batch_size = args.batch_size
        metric_logger.meters['train_acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['train_acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    avg_stat_per_epoch = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    wandb.log(avg_stat_per_epoch)
    return avg_stat_per_epoch


@torch.no_grad()
def evaluate(data_loader, model, device, dataset_id=None, dump_result=False, args = None):
    '''known_dataset_source 与否对 evaluate 没有区别: 因为 evaluation 时仍然知道 target,
    所以正常地用 output 与 target 计算 loss 与 accuracy 即可.
    关键还是在于 model.forward(images, dataset_id), 它们产生了 output, 它们可以根据 dataset_id 来决定 output 的
    维度大小 / 哪些维度强制为0 等额外约束, 使得交叉熵损失进一步减小.'''
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    result_json = {}

    for data in metric_logger.log_every(data_loader, 10, header):
        images, target = data[:2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, dataset_id)
            # print(f'DEBUGGING: output.shape = {output.shape}')

        if dump_result :
            file_ids = data[-1].tolist()
            pred_labels = output.max(-1)[1].tolist()
            for id, pred_id in zip(file_ids, pred_labels) :
                result_json[id] = pred_id
        else :
            # one_hot_target = torch.nn.functional.one_hot(
            #     target, num_classes=args.nb_classes
            # )
            # loss = criterion(output, one_hot_target)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            # metric_logger.update(test_loss=loss.item())
            metric_logger.meters[f'dataset_{dataset_id}_test_loss'].update(loss.item())
            metric_logger.meters[f'dataset_{dataset_id}_test_acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters[f'dataset_{dataset_id}_test_acc5'].update(acc5.item(), n=batch_size)

    if dump_result :
        return result_json
    else :
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # print('* Test_Acc@1 {top1.global_avg:.3f} Test_Acc@5 {top5.global_avg:.3f} Test_loss {losses.global_avg:.3f}'
        #     .format(top1=metric_logger.test_acc1, top5=metric_logger.test_acc5, losses=metric_logger.test_loss))
        print('* Test_Acc@1 {top1.global_avg:.3f} Test_Acc@5 {top5.global_avg:.3f} Test_loss {losses.global_avg:.3f}'
            .format(top1=getattr(metric_logger, f'dataset_{dataset_id}_test_acc1'), 
                    top5=getattr(metric_logger, f'dataset_{dataset_id}_test_acc5'), 
                    losses=getattr(metric_logger, f'dataset_{dataset_id}_test_loss')))

        avg_stat_per_epoch = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        wandb.log(avg_stat_per_epoch)
        return avg_stat_per_epoch


@torch.no_grad()
def test(data_loader, model, device, dataset_id=None, num_classes_list=None, know_dataset=True):
    '''known_dataset_source 与否对 test 有区别, 这决定了我们在给出 prediction(pred_all.json) 的时候怎么取预测值.
    如果 known_dataset_source = False, 那么 output 的维度一定是最大的 645, prediction 的选取也是在这所有 645 个值中选最大;
    如果 known_dataset_source = True, 那么传入的参数 dataset_id 就有用了, 既可以缩小 prediction 的选取范围从而与对应子数据集对应,
    又可以在产生 output 的 forward 过程中作为参数, 指导产生的 output 值.'''
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    result_json = {}

    class_start_id_list = []
    start_id = 0
    for num_classes in num_classes_list:
        class_start_id_list.append(start_id)
        start_id += num_classes

    for data in metric_logger.log_every(data_loader, 10, header):
        images, target = data[:2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, dataset_id)
        file_ids = data[-1].tolist()

        if not know_dataset:
            pred_labels = output.max(-1)[1].tolist()
            # map the concated class_id into original class_id
            pred_labels = [x-class_start_id_list[dataset_id] for x in pred_labels]
        else :
            output = output[:, class_start_id_list[dataset_id]:class_start_id_list[dataset_id]+num_classes_list[dataset_id]]
            pred_labels = output.max(-1)[1].tolist()

        for id, pred_id in zip(file_ids, pred_labels) :
            result_json[id] = pred_id

    return result_json
