#!/bin/bash

MODEL_NAME=Qwen2.5-1.5B-Instruct

# VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 nohup trl vllm-serve --model $MODEL_NAME > log.txt &
nohup trl vllm-serve --model $MODEL_NAME --data_parallel_size 1 --tensor_parallel_size 1 > log.txt &
# pkill -9 -u $(whoami) -f python
# tail -f log.txt
