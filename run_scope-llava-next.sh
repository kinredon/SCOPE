#!bin/bash

set -x

# PAPER_TABLE=coco2017_cap_val,flickr30k_test,gqa,mmbench_en_dev,mme,mmmu_val,nocaps_val,ok_vqa_val2014,pope,scienceqa_img,seedbench
# PAPER_TABLE=mmbench,mme,mmmu_val,pope,scienceqa_img,seedbench,gqa,textvqa_val
#-------------------------------------
# for 7b
# PAPER_TABLE=gqa,mmbench_en_dev,mme,scienceqa_img,textvqa_val,mmmu_val
#-------------------------------------
# for 13b
PAPER_TABLE=gqa,mmbench_en_dev,mme,pope,scienceqa_img,textvqa_val,mmmu_val,seedbench
#-------------------------------------
# PAPER_TABLE=mmvet
# PAPER_TABLE=gqa,mmbench_en_dev,mme,pope,scienceqa_img,textvqa_val,seedbench
LOG_DIR=./logs_final
RUN_NAME=scope_llava_next_13b_token_$1
# RUN_NAME=debug_visionzip_submodule_llava_next_7b_token_$1

# for llava-next
CUDA_VISIBLE_DEVICES=0,1,2,3 ALPHA=1.0 BASELINE=VISIONZIPSUB FUNC=FacilityLocation SUBSET_RATIO=$1 python3 -m accelerate.commands.launch \
    --num_processes=4 \
    --main_process_port=29501 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-13b" \
    --tasks $PAPER_TABLE \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME