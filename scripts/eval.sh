#!/bin/bash

# https://huggingface.co/docs/lighteval/quicktour

NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=/data1/speech/anhnmt2/R1/scripts/outputs/Qwen-1.5B-GRPO_vllm
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=8192,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=evals/Qwen-1.5B-GRPO_vllm
# SYSTEM_PROMPT="You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
SYSTEM_PROMPT=$'Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>'
# SYSTEM_PROMPT=$'Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>. Think for 512 tokens.'

# reasoning model
CUDA_VISIBLE_DEVICES=1 lighteval vllm $MODEL_ARGS \
    "lighteval|gsm8k_test|0|0" \
    --custom-tasks ./tasks_tests.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --system-prompt "$SYSTEM_PROMPT" \
    --save-details

# base model
# CUDA_VISIBLE_DEVICES=1 lighteval vllm $MODEL_ARGS \
#     "lighteval|gsm8k_test|0|0" \
#     --custom-tasks ./tasks_tests.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR \
#     --save-details