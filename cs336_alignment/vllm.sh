# export CUDA_VISIBLE_DEVICES='2'
vllm serve model/Qwen2.5-Math-1.5B \
    --served-model-name Qwen2.5-Math-1.5B \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 2000 \
    --host 0.0.0.0 \
    --port 8010

# export CUDA_VISIBLE_DEVICES='2'
# vllm serve model/Qwen2.5-3B \
#     --served-model-name Qwen2.5-3B-Base \
#     --tensor-parallel-size 1 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.2 \
#     --max-model-len 2000 \
#     --host 0.0.0.0 \
#     --port 8011

# export CUDA_VISIBLE_DEVICES='0'
# vllm serve model/Qwen2.5-7B-Instruct \
#     --served-model-name Qwen2.5-7B-Instruct \
#     --tensor-parallel-size 1 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.4 \
#     --max-model-len 5000 \
#     --host 0.0.0.0 \
#     --port 8010

export CUDA_VISIBLE_DEVICES='1'
vllm serve model/Qwen2.5-14B-Instruct \
    --served-model-name Qwen2.5-14B-Instruct \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 5000 \
    --host 0.0.0.0 \
    --port 8010

# export CUDA_VISIBLE_DEVICES='1'
# vllm serve model/Qwen2.5-32B-Instruct \
#     --served-model-name Qwen2.5-32B-Instruct \
#     --tensor-parallel-size 1 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.8 \
#     --max-model-len 5000 \
#     --host 0.0.0.0 \
#     --port 8010

# vllm  serve model/Qwen2.5-72B-Instruct \
#     --served-model-name Qwen2.5-72B-Instruct \
#     --tensor-parallel-size 2 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.95 \
#     --max-model-len 5000 \
#     --host 0.0.0.0 \
#     --port 8010

# export CUDA_VISIBLE_DEVICES='2'
# vllm serve result/sft_qwen_3b_ultraChat_SafetyLlama/checkpoint-3000 \
#     --served-model-name Qwen2.5-3B-Base-SFT \
#     --tensor-parallel-size 1 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.2 \
#     --max-model-len 2000 \
#     --host 0.0.0.0 \
#     --port 8011

# export CUDA_VISIBLE_DEVICES='2'
# vllm serve result/sft_qwen_3b_ultraChat_SafetyLlama/checkpoint-6300 \
#     --served-model-name Qwen2.5-3B-Base \
#     --tensor-parallel-size 1 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.2 \
#     --max-model-len 2000 \
#     --host 0.0.0.0 \
#     --port 8011