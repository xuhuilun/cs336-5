#!/bin/bash
# export CUDA_VISIBLE_DEVICES='2,3'
# ================= 配置区 =================
BASE_MODEL="model/Qwen2.5-Math-1.5B"
# TRAIN_DATA="data/gsm8k/train.jsonl"
# VAL_DATA="data/gsm8k/test.jsonl"
# WANDB_PROJECT="cs336-grpo-after-base-length-norm"
TRAIN_DATA="data/math12k/data/train-00000-of-00001.parquet"
VAL_DATA="data/math12k/data/test-00000-of-00001.parquet"
WANDB_PROJECT="cs336-grpo-math12k-after-base-length-norm"

PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_BASE="result/ablation_length_norm"


# 实验参数
BEST_LR=3e-5
N_STEPS=100

# ================= 消融循环 =================
# 1. mask_mean: Token-level (传统 SFT 风格，倾向于生成短回答，梯度被长度稀释)
# 2. mask_normalize: Sentence-level (DeepSeek-R1 风格，鼓励长链条推理 CoT)
for NORM_TYPE in "mask_mean" "mask_normalize"; do

    RUN_NAME="grpo_len_norm_${NORM_TYPE}_lr${BEST_LR}"
    echo "======================================================="
    echo "🚀 启动长度归一化消融实验: $NORM_TYPE"
    echo "======================================================="

    uv run python cs336_alignment/train_grpo.py \
        --model_id "$BASE_MODEL" \
        --train_data_path "$TRAIN_DATA" \
        --test_data_path "$VAL_DATA" \
        --prompt_path "$PROMPT_TEMPLATE" \
        --output_dir "${OUTPUT_BASE}/${RUN_NAME}" \
        --length_norm_type "$NORM_TYPE" \
        --n_grpo_steps "$N_STEPS" \
        --lr "$BEST_LR" \
        --rollout_batch_size 256 \
        --group_size 8 \
        --train_batch_size 256 \
        --gradient_accumulation_steps 128 \
        --loss_type "grpo_clip" \
        --use_std_normalization \
        --device cuda:0 \
        --vllm_device cuda:1 \
        --vllm_gpu_util 0.25 \
        --eval_every_steps 8 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$RUN_NAME"

    if [ $? -ne 0 ]; then
        echo "❌ 实验 $RUN_NAME 失败！"
        exit 1
    fi
    
    sleep 10
done

echo "🎉 长度归一化消融实验全部完成！"