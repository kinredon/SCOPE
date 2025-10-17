#!bin/bash

set -x

PAPER_TABLE=gqa,mmbench_en_dev,mme,pope,scienceqa_img,textvqa_val,seedbench,mmvet
PAPER_TABLE=pope
LOG_DIR=./logs_final
RUN_NAME=SCOPE_LLaVA_7b_token_$1

# for llava-v1.5 13B
# Change the pretrained argument to "liuhaotian/llava-v1.5-13b"
CUDA_VISIBLE_DEVICES=0,1,2,3 ALPHA=1.0 BASELINE=SCOPE SUBSET_RATIO=$1 python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks $PAPER_TABLE \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME