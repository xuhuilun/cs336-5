import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
VLLM_API_URL = "http://localhost:8010/v1"
MODEL_NAME = "Qwen2.5-3B-Base-SFT"
MODEL_NAME = "Qwen2.5-14B-Instruct"
DATA_PATH = "data/alpaca_eval/alpaca_eval.jsonl" 
# 注意：PDF 要求保存为标准的 JSON Array 格式
# OUTPUT_FILE = f"result/alpaca_{MODEL_NAME}_predictions.json" 
OUTPUT_FILE = f"result/alpaca_{MODEL_NAME}_predictions.json" 
MAX_WORKERS = 100 

client = OpenAI(base_url=VLLM_API_URL, api_key="empty")

# 官方指定的统一 System Prompt (PDF 第2页)
SYSTEM_PROMPT = (
    "Below is a list of conversations between a human and an AI assistant (you).\n"
    "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\".\n"
    "You are a helpful, respectful, and honest assistant.\n"
    "You should always answer as helpfully as possible while ensuring safety.\n"
    "Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
    "Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\n"
    "Your response must be socially responsible, and thus you can reject to answer some controversial topics."
)

# ================= 工具函数 =================

def load_alpaca_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def call_vllm_raw_api(item):
    """
    遵循 PDF 要求的格式构造 Raw Prompt 并调用补全接口
    """
    instruction = item['instruction']
    
    # 严格按照 PDF 要求的格式拼接原始字符串
    # 格式: # Instruction + SystemPrompt
    full_raw_prompt = (
        f"# Instruction\n{SYSTEM_PROMPT}\n\n"
        f"# Query:\n```\n{instruction}\n```\n\n"
        f"# Answer:\n```\n"
    )
    
    try:
        # 使用 completions 接口直接发送原始字符串，不带对话模版干扰
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=full_raw_prompt,
            temperature=0.0, # 官方要求的 Greedy Decoding
            max_tokens=1024,
            # 停止符：防止 Base 模型在输出答案后复读题目
            stop=["# Query:", "```", "\n\n#", "User:", "Assistant:"] 
        )
        gen_text = response.choices[0].text.strip()
        
        # 返回 AlpacaEval 要求的 JSON 字典项
        return {
            "instruction": instruction,
            "output": gen_text,
            "generator": MODEL_NAME,
            "dataset": item.get("dataset", "alpaca_eval")
        }
    except Exception as e:
        return {"error": str(e), "instruction": instruction}

# ================= 主程序 =================

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return
        
    eval_set = load_alpaca_data(DATA_PATH)
    print(f"🚀 Loaded {len(eval_set)} AlpacaEval instructions.")

    all_results = []
    start_time = time.time()

    print(f"正在启动并发评测 (模型: {MODEL_NAME}, 并发: {MAX_WORKERS})...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        futures = {executor.submit(call_vllm_raw_api, item): item for item in eval_set}
        
        for future in tqdm(as_completed(futures), total=len(eval_set), desc="AlpacaEval"):
            res = future.result()
            if "error" not in res:
                all_results.append(res)
            else:
                print(f"\n警告: 请求失败 -> {res['error']}")

    duration = time.time() - start_time
    throughput = len(all_results) / duration if duration > 0 else 0

    # 3. 序列化为 JSON 数组 (AlpacaEval 官方评估器要求必须是一个 JSON list)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        json.dump(all_results, fout, indent=2, ensure_ascii=False)

    print("\n" + "="*40)
    print(f"📊 评测完成!")
    print(f"有效结果数: {len(all_results)}")
    print(f"总耗时: {duration/60:.2f} 分钟")
    print(f"吞吐量 (Throughput): {throughput:.2f} samples/s")
    print(f"结果已保存至: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    main()