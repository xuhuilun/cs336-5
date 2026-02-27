import os
import json
import time
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
# 1. 请确保终端运行: vllm serve /path/to/model --port 8010 --served-model-name Qwen2.5-3B-Base
VLLM_API_URL = "http://localhost:8010/v1"
MMLU_DATA_DIR = "data/mmlu/test"
MODEL_NAME = "Qwen2.5-3B-Base"
OUTPUT_FILE = f"result/mmlu_{MODEL_NAME}_baseline_results.json"
MAX_WORKERS = 100

client = OpenAI(base_url=VLLM_API_URL, api_key="empty")

# 官方指定的统一 System Prompt
SYSTEM_PROMPT = (
    "Below is a list of conversations between a human and an AI assistant (you).\n"
    "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\".\n"
    "You are a helpful, respectful, and honest assistant.\n"
    "You should always answer as helpfully as possible while ensuring safety.\n"
    "Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
    "Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\n"
    "Your response must be socially responsible, and thus you can reject to answer some controversial topics."
)

# MMLU 任务特定的提示词模板 (遵循 PDF 第 3 页)
MMLU_TASK_TEMPLATE = (
    "Answer the following multiple choice question about {subject}. Respond with a single "
    "sentence of the form \"The correct answer is _\", filling the blank with the letter "
    "corresponding to the correct answer (i.e., A, B, C or D).\n\n"
    "Question: {question}\n"
    "A. {optA}\n"
    "B. {optB}\n"
    "C. {optC}\n"
    "D. {optD}\n"
    "Answer:"
)

# ================= 工具函数 =================

def parse_mmlu_response(model_response: str) -> str | None:
    """解析 MMLU 输出，提取 A/B/C/D"""
    if not model_response: return None
    # 匹配 "The correct answer is X"
    match_std = re.search(r"[Tt]he correct answer is\s*([A-D])", model_response)
    if match_std: return match_std.group(1).upper()
    # 备选：匹配第一个出现的 A/B/C/D
    match_alt = re.search(r"\b([A-D])\b", model_response)
    if match_alt: return match_alt.group(1).upper()
    return None

def load_all_mmlu_tests(data_dir):
    """加载所有 MMLU 测试 CSV 并包装成作业要求的格式"""
    all_items = []
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return []
    
    for file in os.listdir(data_dir):
        if file.endswith("_test.csv"):
            subject = file.replace("_test.csv", "").replace("_", " ")
            path = os.path.join(data_dir, file)
            df = pd.read_csv(path, header=None)
            df.columns = ["question", "A", "B", "C", "D", "answer"]
            for _, row in df.iterrows():
                # 按照作业要求构造 User Query 文本
                user_instruction = MMLU_TASK_TEMPLATE.format(
                    subject=subject, question=row["question"],
                    optA=row["A"], optB=row["B"], optC=row["C"], optD=row["D"]
                )
                
                full_prompt = f"{SYSTEM_PROMPT}\n{user_instruction}"
                
                all_items.append({
                    "full_prompt": full_prompt,
                    "gold": str(row["answer"]).strip().upper(),
                    "subject": subject
                })
    return all_items

def call_api(item):
    """调用 API 运行推理"""
    try:
        # 使用 completions 接口直接发送原始字符串，避免 chat 模版干扰 Base 模型
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=item["full_prompt"], # 修复之前的 NameError
            temperature=0.0,
            max_tokens=64,
            stop=["# Query:", "```", "\n\n#"]
        )
        gen_text = response.choices[0].text
        pred = parse_mmlu_response(gen_text)
        return {
            "subject": item["subject"],
            "gold": item["gold"],
            "pred": pred,
            "output": gen_text,
            "is_correct": (pred == item["gold"])
        }
    except Exception as e:
        return {"error": str(e), "subject": item["subject"]}

# ================= 主程序 =================

def main():
    # 1. 检查连接
    print(f"正在检查 vLLM 连接: {VLLM_API_URL}...")
    try:
        models = client.models.list()
        print(f"连接成功！可用模型: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"❌ 无法连接到 vLLM: {e}\n请确保服务已启动并检查端口。")
        return

    # 2. 加载数据
    all_items = load_all_mmlu_tests(MMLU_DATA_DIR)
    if not all_items: return
    print(f"已加载 {len(all_items)} 条 MMLU 测试题。")

    # 3. 并发推理
    all_results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(call_api, item): item for item in all_items}
        for future in tqdm(as_completed(futures), total=len(all_items), desc="Evaluating MMLU"):
            all_results.append(future.result())

    duration = time.time() - start_time

    # 4. 统计指标
    valid_results = [r for r in all_results if "error" not in r]
    correct_count = sum(1 for r in valid_results if r["is_correct"])
    accuracy = correct_count / len(valid_results) if valid_results else 0
    
    metrics = {
        "accuracy": accuracy,
        "throughput": len(all_results) / duration,
        "total": len(all_results),
        "success": len(valid_results)
    }

    # 4. 打印报告
    # 按学科（subject）统计
    cat_stats = {}
    for res in valid_results:
        cat = res['subject'] # 统一使用 subject，修复之前的 KeyError
        if cat not in cat_stats: cat_stats[cat] = {"correct": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        if res['is_correct']: cat_stats[cat]["correct"] += 1

    # 打印格式化报告
    print('\n' + '='*70)
    print(f"{'Subject':40s} | {'Accuracy':10s} | {'Count'}")
    print('-'*70)
    for cat in sorted(cat_stats.keys()):
        stats = cat_stats[cat]
        acc = stats["correct"] / stats["total"]
        print(f"{cat:40s} | {acc:10.2%} | {stats['total']}")
    print('='*70)

    # 5. 保存

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"metrics": metrics, "details": all_results}, f, indent=2)

    print(f"\n✅ 评测完成！准确率: {accuracy:.2%}, 吞吐量: {metrics['throughput']:.2f} samples/s")

if __name__ == "__main__":
    main()