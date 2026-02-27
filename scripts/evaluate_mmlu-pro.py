import os
import json
import time
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
VLLM_API_URL = "http://localhost:8011/v1"
DATA_PATH = "data/MMLU-Pro/data/test-00000-of-00001.parquet"
MODEL_NAME = "Qwen2.5-3B-Base"
# OUTPUT_FILE = f"result/mmlu-pro_{MODEL_NAME}_baseline_raw_results.json"
OUTPUT_FILE = f"result/mmlu-pro_{MODEL_NAME}_SFT_raw_results.json"
MAX_WORKERS = 100

client = OpenAI(base_url=VLLM_API_URL, api_key="empty")

# 1. 官方指定的统一 System Prompt (PDF 第2页)
SYSTEM_PROMPT = (
    "Below is a list of conversations between a human and an AI assistant (you).\n"
    "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\".\n"
    "You are a helpful, respectful, and honest assistant.\n"
    "You should always answer as helpfully as possible while ensuring safety.\n"
    "Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
    "Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\n"
    "Your response must be socially responsible, and thus you can reject to answer some controversial topics."
)

# 2. MMLU-Pro 任务特定的提示词模板 (遵循 PDF 第3页风格)
MMLU_PRO_TASK_TEMPLATE = (
    "Answer the following multiple choice question about {subject}. Respond with a single "
    "sentence of the form \"The correct answer is _\", filling the blank with the letter "
    "corresponding to the correct answer (i.e., A, B, C, D, E, F, G, H, I, or J).\n\n"
    "Question: {question}\n"
    "{options_str}\n"
    "Answer:"
)

# ================= 工具函数 =================

def form_options_str(options: list) -> str:
    """格式化选项列表"""
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    return "\n".join([f"{l}. {t}" for l, t in zip(labels, options)])

def parse_mmlu_pro_response(model_response: str) -> str | None:
    """解析输出，提取 A-J 选项"""
    if not model_response: return None
    # 优先匹配题目要求的标准格式: "The correct answer is X"
    match_std = re.search(r"[Tt]he correct answer is\s*([A-J])", model_response)
    if match_std: return match_std.group(1).upper()
    # 备选：匹配文本中出现的第一个 A-J（Base 模型有时直接给字母）
    match_alt = re.search(r"\b([A-J])\b", model_response)
    if match_alt: return match_alt.group(1).upper()
    return None

def call_vllm_raw_api(full_prompt: str) -> str:
    """使用 client.completions 调用 Base 模型，不带对话模版干扰"""
    try:
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=full_prompt,
            temperature=0.0, # Greedy Decoding
            max_tokens=64,   # MMLU 只需要短输出
            top_p=1.0,
            # 停止符：防止模型写完答案后复读题目或开启新一轮 Query
            stop=["# Query:", "```", "\n\n#"] 
        )
        return response.choices[0].text
    except Exception as e:
        return f"ERROR: {str(e)}"

def process_entry(entry: dict):
    """构建完全符合作业 PDF 格式要求的 Raw Prompt"""
    
    # 构建具体的任务内容
    options_formatted = form_options_str(entry['options'])
    instruction_content = MMLU_PRO_TASK_TEMPLATE.format(
        subject=entry['category'],
        question=entry['question'],
        options_str=options_formatted
    )
    
    full_raw_prompt = f"{SYSTEM_PROMPT}\n{instruction_content}"

    raw_output = call_vllm_raw_api(full_raw_prompt)
    
    if "ERROR" in raw_output:
        return {"error": raw_output, "category": entry['category']}
        
    prediction = parse_mmlu_pro_response(raw_output)
    is_correct = (entry["answer"] == prediction)
    
    return {
        "category": entry['category'],
        "gold": entry["answer"],
        "prediction": prediction,
        "output": raw_output,
        "is_correct": is_correct
    }

# ================= 主程序 =================

def main():
    # 1. 加载数据
    print(f"正在加载 MMLU-Pro 数据集: {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    test_entries = df.to_dict('records')
    print(f"成功加载 {len(test_entries)} 条测试数据。")

    # 2. 执行并发评估
    all_results = []
    print(f"正在启动 Base 模型零样本评测 (并发数: {MAX_WORKERS})...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_entry = {executor.submit(process_entry, e): e for e in test_entries}
        for future in tqdm(as_completed(future_to_entry), total=len(test_entries), desc="MMLU-Pro"):
            all_results.append(future.result())

    duration = time.time() - start_time

    # 3. 结果汇总
    valid_results = [r for r in all_results if "error" not in r]
    total_corr = sum(1 for r in valid_results if r["is_correct"])
    overall_acc = total_corr / len(valid_results) if valid_results else 0

    # 4. 打印报告
    cat_stats = {}
    for res in valid_results:
        cat = res['category']
        if cat not in cat_stats: cat_stats[cat] = [0, 0]
        cat_stats[cat][1] += 1
        if res['is_correct']: cat_stats[cat][0] += 1

    print('\n' + '='*60)
    print(f"{'Category':25s} | {'Accuracy':10s} | {'Count'}")
    print('-'*60)
    for cat in sorted(cat_stats.keys()):
        corr, cnt = cat_stats[cat]
        print(f"{cat:25s} | {corr/cnt:10.2%} | {cnt}")
    
    print('='*60)
    print(f"Overall Baseline Accuracy: {overall_acc:.2%}")
    print(f"Throughput: {len(valid_results)/duration:.2f} samples/s")

    # 5. 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"metrics": {"accuracy": overall_acc}, "details": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()