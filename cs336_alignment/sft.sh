export CUDA_VISIBLE_DEVICES='1'
export CUDA_LAUNCH_BLOCKING=1
uv run python cs336_alignment/sft.py \
    --train_path "data/sft-cs336-assign5-datasets/sft-instruct/train.jsonl" \
    --eval_path "data/sft-cs336-assign5-datasets/sft-instruct/test.jsonl" \
    --model_path "model/Qwen2.5-3B" \
    --output_dir "result/sft_qwen_3b_ultraChat_SafetyLlama" \
    --batch_size 32 \
    --micro_batch_size 4 \
    --lr 2e-5 \
    --max_seq_len 512 \
    --save_every_steps 100 \
    --eval_every_steps 100