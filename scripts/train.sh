#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch --main_process_port 29501 --num_processes 6 \
    --config_file ../recipes/accelerate_configs/zero2.yaml ../src/grpo.py \
    --config ../recipes/config_r1.yaml