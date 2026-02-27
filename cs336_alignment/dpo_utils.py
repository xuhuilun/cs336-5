import torch
import torch.nn.functional as F
import random
import gzip
import json
import os
from typing import List, Dict, Tuple

# ==========================================
# 核心工具函数：带掩码的对数概率计算
# ==========================================
def get_log_probs(model, input_ids, labels):
    """
    计算序列中 Response 部分的总对数概率。
    修复了 torch.gather 处理 -100 索引导致的 CUDA 断言错误。
    """
    outputs = model(input_ids)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    
    shift_logits = log_probs[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # --- 关键修复步骤 ---
    # 创建一个安全的标签副本，将 -100 替换为 0 (或其他合法索引)
    # 否则 torch.gather 在遇到 -100 时会触发 device-side assert
    safe_labels = shift_labels.clone()
    mask = (safe_labels != -100)
    safe_labels[safe_labels == -100] = 0 
    
    per_token_log_probs = torch.gather(
        shift_logits, dim=2, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # 使用真正的掩码将刚才替换为 0 的位置的概率清零
    per_token_log_probs = per_token_log_probs * mask
    
    # 返回序列的总对数概率
    return per_token_log_probs.sum(dim=-1)

# ==========================================
# 核心 DPO 损失函数实现
# ==========================================

def compute_dpo_loss(
    model, 
    ref_model, 
    tokenizer, 
    beta, 
    prompt, 
    response_chosen, 
    response_rejected
):
    """
    计算 DPO 损失并返回监控指标。
    """
    device = model.device

    # 定义 SFT 阶段使用的相同模板
    def get_full_text(p, r):
        return (
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{p}\n\n"
            f"### Response:\n{r}"
        )

    def encode_pair(p, r):
        # 1. 编码 Prompt 部分用于计算长度
        prompt_text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{p}\n\n### Response:\n"
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        
        # 2. 编码全量文本 (Prompt + Response)
        full_text = get_full_text(p, r) + tokenizer.eos_token
        full_ids = tokenizer.encode(full_text, add_special_tokens=True)
        
        # 3. 构造 Labels：Prompt 区域填充 -100
        labels = torch.tensor(full_ids).clone()
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = -100
        
        return torch.tensor([full_ids]).to(device), labels.unsqueeze(0).to(device)

    # 准备输入和标签
    chosen_ids, chosen_labels = encode_pair(prompt, response_chosen)
    rejected_ids, rejected_labels = encode_pair(prompt, response_rejected)

    # --- 1. 计算 Policy 模型 (正在训练的模型) 的对数概率 ---
    lp_theta_chosen = get_log_probs(model, chosen_ids, chosen_labels)
    lp_theta_rejected = get_log_probs(model, rejected_ids, rejected_labels)

    # --- 2. 计算 Reference 模型 (冻结模型) 的对数概率 ---
    with torch.no_grad():
        lp_ref_chosen = get_log_probs(ref_model, chosen_ids, chosen_labels)
        lp_ref_rejected = get_log_probs(ref_model, rejected_ids, rejected_labels)

    # --- 3. 计算 DPO 核心公式 ---
    # pi_log_ratio = log(pi_theta(y_w|x) / pi_ref(y_w|x))
    # 公式简化为: (lp_theta_w - lp_ref_w)
    prob_ratio_chosen = lp_theta_chosen - lp_ref_chosen
    prob_ratio_rejected = lp_theta_rejected - lp_ref_rejected
    
    # DPO 目标是最大化此 logits
    dpo_logits = beta * (prob_ratio_chosen - prob_ratio_rejected)
    
    # 损失函数为 -log sigmoid(beta * log_ratio_diff)
    loss = -F.logsigmoid(dpo_logits).mean()
    
    # --- 4. 计算监控指标 (用于 WandB) ---
    with torch.no_grad():
        # 奖励值 (用于观察模型对 chosen/rejected 的评分)
        chosen_reward = beta * prob_ratio_chosen
        rejected_reward = beta * prob_ratio_rejected
        
        # 准确率：chosen 的对数概率提升是否大于 rejected
        # 即使损失在降，也要看准确率是否在升
        accuracy = (chosen_reward > rejected_reward).float().mean()
        margin = (chosen_reward - rejected_reward).mean()

    metrics = {
        "loss": loss.item(),
        "reward_chosen": chosen_reward.item(),
        "reward_rejected": rejected_reward.item(),
        "reward_margin": margin.item(),
        "accuracy": accuracy.item()
    }
    
    return loss, metrics

# ==========================================
# 数据加载与清洗逻辑 (Anthropic HH)
# ==========================================

def load_anthropic_hh_dataset(root_dir: str, split: str = "train") -> List[Dict[str, str]]:
    if split not in ["train", "test"]:
        raise ValueError("split 必须是 'train' 或 'test'")

    file_name = "train.jsonl.gz" if split == "train" else "test.jsonl.gz"
    subsets = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
    combined_data = []
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    for subset in subsets:
        file_path = os.path.join(root_dir, subset, file_name)
        if not os.path.exists(file_path):
            continue
            
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    # 仅保留单轮对话
                    if item['chosen'].count("\n\nHuman:") > 1:
                        continue
                    
                    # 提取指令和回复
                    c_parts = item['chosen'].split("\n\nAssistant:")
                    r_parts = item['rejected'].split("\n\nAssistant:")
                    
                    if len(c_parts) < 2 or len(r_parts) < 2:
                        continue
                    
                    combined_data.append({
                        "instruction": c_parts[0].replace("\n\nHuman:", "").strip(),
                        "chosen": c_parts[1].strip(),
                        "rejected": r_parts[1].strip(),
                        "subset": subset
                    })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
                    
    print(f"✅ [{split.upper()}] 加载完成，共 {len(combined_data)} 条单轮数据。")
    return combined_data

# ==========================================
# 调试与检查
# ==========================================

def inspect_samples(data):
    # 随机抽查
    sample_size = min(len(data), 2)
    samples = random.sample(data, sample_size)
    
    for i, s in enumerate(samples):
        print(f"\n--- 样本 {i+1} [{s['subset']}] ---")
        print(f"Prompt: {s['instruction'][:100]}...")
        print(f"Chosen: {s['chosen'][:50]}...")
        print(f"Rejected: {s['rejected'][:50]}...")

if __name__ == "__main__":
    # 使用示例
    # dataset = load_anthropic_hh_dataset("data/hh-rlhf", split="train")
    # inspect_samples(dataset)
    pass