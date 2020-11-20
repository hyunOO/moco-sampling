#!/bin/zsh

CUDA_VISIBLE_DEVICES=0 python moco_cifar10_demo.py --sample-method kcore --data-ratio 0.7 &
CUDA_VISIBLE_DEVICES=1 python moco_cifar10_demo.py --sample-method kcore --data-ratio 0.5 &
CUDA_VISIBLE_DEVICES=2 python moco_cifar10_demo.py --sample-method kcore --data-ratio 0.3 &

CUDA_VISIBLE_DEVICES=0 python moco_cifar10_demo.py --sample-method high_loss --data-ratio 0.7 &
CUDA_VISIBLE_DEVICES=1 python moco_cifar10_demo.py --sample-method high_loss --data-ratio 0.5 &
CUDA_VISIBLE_DEVICES=2 python moco_cifar10_demo.py --sample-method high_loss --data-ratio 0.3 &

CUDA_VISIBLE_DEVICES=3 python moco_cifar10_demo.py --sample-method low_loss --data-ratio 0.7 &
CUDA_VISIBLE_DEVICES=3 python moco_cifar10_demo.py --sample-method low_loss --data-ratio 0.5 &
CUDA_VISIBLE_DEVICES=2 python moco_cifar10_demo.py --sample-method low_loss --data-ratio 0.3 &

