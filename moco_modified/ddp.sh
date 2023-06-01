python main_moco.py \
  -a resnet50 \
  --lr 0.015 --epochs 200 \
  --batch-size 128 --moco-k 4096 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --resume 'checkpoint_0004.pth.tar' \
  --mlp --moco-t 0.2 --aug-plus --cos \
  ~