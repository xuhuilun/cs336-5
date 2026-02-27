
export OPENAI_BASE_URL="http://localhost:8010/v1"
export OPENAI_API_KEY="empty"
export ALPACA_EVAL_MAX_CONCURRENCY=10
# uv run alpaca_eval --model_outputs 'result/alpaca_Qwen2.5-14B-Instruct_predictions.json' \
# uv run alpaca_eval --model_outputs 'result/alpaca_Qwen2.5-3B-Base-SFT_predictions.json' \
uv run alpaca_eval --model_outputs 'result/alpaca_Qwen2.5-3B-Base_predictions.json' \
  --annotators_config 'scripts/alpaca_eval_vllm_qwen2_5_14b_fn' \
  --base-dir '.'