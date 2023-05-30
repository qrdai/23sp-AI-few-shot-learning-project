export RANK=0
export WORLD_SIZE=1

torchrun --nproc_per_node=2 main.py