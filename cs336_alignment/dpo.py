"""
    DPO implementation with comprehensive WandB logging.
    Tracks Loss, Rewards, Margins, and Log-probabilities.
"""
import os
import gzip
import json
import torch
import argparse
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

# --- 全局配置 ---
HH_PATH = "data/hh-rlhf"
OUTPUT_DIR = "result/dpo_qwen_3b_sft"
POLICY_MODEL_PATH  = "result/sft_qwen_3b_ultraChat_SafetyLlama/checkpoint-6300"
REFERENCE_MODEL_PATH = "result/sft_qwen_3b_ultraChat_SafetyLlama/checkpoint-6300"
max_length = 512
train_batch_size = 64
gradient_accumulation_steps = 16
micro_batch_size = train_batch_size // gradient_accumulation_steps
lr = 1e-6
epochs = 1
beta = 0.1
eval_interval = 20
save_interval = 100
policy_device = "cuda:0"
reference_device = "cuda:0"

def load_hh_dataset():
    filenames = ["harmless-base/train.jsonl.gz",
                 "helpful-base/train.jsonl.gz",
                 "helpful-online/train.jsonl.gz",
                 "helpful-rejection-sampled/train.jsonl.gz"]
    all_examples = []
    for filename in filenames:
        file_path = os.path.join(HH_PATH, filename)
        if not os.path.exists(file_path): continue
        with gzip.open(file_path, "rt") as f:
            for line in f:
                data = json.loads(line)
                c_msg = [m for m in data.get("chosen", "").split("\n\n") if m.strip()]
                r_msg = [m for m in data.get("rejected", "").split("\n\n") if m.strip()]
                if len(c_msg) == 2 and len(r_msg) == 2 and c_msg[0] == r_msg[0]:
                    all_examples.append({"instruction": c_msg[0], "chosen": c_msg[1], "rejected": r_msg[1]})
    random.shuffle(all_examples)
    split = int(len(all_examples) * 0.95)
    return all_examples[:split], all_examples[split:]


class HHPreferenceDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)
    
    def _process_tokens(self, prompt_tokens, response_tokens):
        """
        处理单条序列的核心逻辑：拼接 -> 截断 -> 左填充
        """
        # 1. 拼接并从左侧截断（保留序列的末尾，即完整的 Response 和尽可能多的 Prompt）
        full_tokens = (prompt_tokens + response_tokens)[-self.max_length:]
        
        # 2. 计算在截断后的序列中，Prompt 占了多少长度
        # 如果 Response 特别长，导致 Prompt 被完全截断，则 p_len 为 0
        actual_p_len = max(0, len(full_tokens) - len(response_tokens))
        
        # 3. 计算左填充长度
        pad_len = self.max_length - len(full_tokens)
        
        # 4. 构建 Tensor
        input_ids = [self.tokenizer.pad_token_id] * pad_len + full_tokens
        attention_mask = [0] * pad_len + [1] * len(full_tokens)
        
        # 5. 计算 Prompt 在整个 input_ids 序列中的结束位置
        # 这个值用于 compute_response_log_probs 中的 mask
        # 逻辑：左侧填充长度 + 序列中 prompt 实际长度
        prompt_end_index = pad_len + actual_p_len
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            prompt_end_index
        )

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tmpl = "Below is an instruction that describes a task.\n\n### Instruction:\n{}\n\n### Response:\n"
        prompt = tmpl.format(ex["instruction"])
        
        p_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        c_tokens = self.tokenizer.encode(ex["chosen"] + self.tokenizer.eos_token, add_special_tokens=False)
        r_tokens = self.tokenizer.encode(ex["rejected"] + self.tokenizer.eos_token, add_special_tokens=False)

        # 分别处理 chosen 和 rejected
        c_ids, c_mask, c_p_len = self._process_tokens(p_tokens, c_tokens)
        r_ids, r_mask, r_p_len = self._process_tokens(p_tokens, r_tokens)

        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "chosen_prompt_len": c_p_len,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
            "rejected_prompt_len": r_p_len,
        }


def compute_response_log_probs(logits, input_ids, attention_mask, prompt_lengths):
    log_probs = F.log_softmax(logits, dim=-1)
    shift_log_probs = log_probs[:, :-1, :]
    shift_input_ids = input_ids[:, 1:]
    shift_attention_mask = attention_mask[:, 1:]
    token_log_probs = torch.gather(shift_log_probs, dim=2, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
    
    response_mask = torch.zeros_like(shift_attention_mask)
    for i in range(token_log_probs.shape[0]):
        response_mask[i, prompt_lengths[i]-1:] = 1
    
    combined_mask = shift_attention_mask * response_mask
    return (token_log_probs * combined_mask).sum(dim=1), combined_mask.sum(dim=1)

def train_dpo():
    # 1. 初始化 WandB 并记录超参数
    wandb.init(
        project="cs336-dpo",
        name=f"Qwen3B-DPO-beta{beta}-lr{lr}",
        config={
            "policy_model": POLICY_MODEL_PATH,
            "beta": beta,
            "learning_rate": lr,
            "batch_size": train_batch_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_length": max_length,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=policy_device)
    ref_model = AutoModelForCausalLM.from_pretrained(REFERENCE_MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=reference_device)
    ref_model.eval()

    train_ex, val_ex = load_hh_dataset()
    train_loader = DataLoader(HHPreferenceDataset(train_ex, tokenizer, max_length), batch_size=micro_batch_size, shuffle=True)
    val_loader = DataLoader(HHPreferenceDataset(val_ex, tokenizer, max_length), batch_size=micro_batch_size, shuffle=False)

    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=lr)
    
    step = 0
    policy_model.train()

    for epoch in range(epochs):
        # 用于累加一个 Step 内的指标
        step_metrics = {"loss": 0, "chosen_reward": 0, "rejected_reward": 0, "margin": 0, "acc": 0}

        for batch_idx, batch in enumerate(train_loader):
            c_ids, r_ids = batch["chosen_input_ids"].to(policy_device), batch["rejected_input_ids"].to(policy_device)
            c_mask, r_mask = batch["chosen_attention_mask"].to(policy_device), batch["rejected_attention_mask"].to(policy_device)
            c_p_len, r_p_len = batch["chosen_prompt_len"].to(policy_device), batch["rejected_prompt_len"].to(policy_device)

            # 前向传播
            c_logits = policy_model(c_ids, attention_mask=c_mask).logits
            r_logits = policy_model(r_ids, attention_mask=r_mask).logits
            with torch.no_grad():
                c_logits_ref = ref_model(c_ids.to(reference_device), attention_mask=c_mask.to(reference_device)).logits.to(policy_device)
                r_logits_ref = ref_model(r_ids.to(reference_device), attention_mask=r_mask.to(reference_device)).logits.to(policy_device)

            # 计算 Log Prob
            c_lp, _ = compute_response_log_probs(c_logits, c_ids, c_mask, c_p_len)
            c_lp_ref, _ = compute_response_log_probs(c_logits_ref, c_ids, c_mask, c_p_len)
            r_lp, _ = compute_response_log_probs(r_logits, r_ids, r_mask, r_p_len)
            r_lp_ref, _ = compute_response_log_probs(r_logits_ref, r_ids, r_mask, r_p_len)

            # DPO 计算
            c_reward = beta * (c_lp - c_lp_ref)
            r_reward = beta * (r_lp - r_lp_ref)
            loss = -F.logsigmoid(c_reward - r_reward).mean()

            (loss / gradient_accumulation_steps).backward()

            # 指标收集 (detach 以释放显存)
            step_metrics["loss"] += loss.item() / gradient_accumulation_steps
            step_metrics["chosen_reward"] += c_reward.mean().item() / gradient_accumulation_steps
            step_metrics["rejected_reward"] += r_reward.mean().item() / gradient_accumulation_steps
            step_metrics["margin"] += (c_reward - r_reward).mean().item() / gradient_accumulation_steps
            step_metrics["acc"] += (c_reward > r_reward).float().mean().item() / gradient_accumulation_steps

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                # --- 记录训练指标到 WandB ---
                wandb.log({
                    "train/loss": step_metrics["loss"],
                    "train/chosen_reward": step_metrics["chosen_reward"],
                    "train/rejected_reward": step_metrics["rejected_reward"],
                    "train/reward_margin": step_metrics["margin"],
                    "train/accuracy": step_metrics["acc"],
                    "train/lr": optimizer.param_groups[0]["lr"]
                }, step=step)

                if step % 10 == 0:
                    print(f"Step {step} | Margin: {step_metrics['margin']:.4f} | Acc: {step_metrics['acc']:.4f}")

                # 重置累加器
                step_metrics = {k: 0 for k in step_metrics}

                # 评估
                if step % eval_interval == 0:
                    v_loss, v_acc, v_margin = evaluate_dpo(policy_model, ref_model, val_loader, beta, policy_device)
                    wandb.log({
                        "val/loss": v_loss,
                        "val/accuracy": v_acc,
                        "val/reward_margin": v_margin
                    }, step=step)
                    print(f"[Eval] Step {step} | Acc: {v_acc:.4f} | Margin: {v_margin:.4f}")

                if step % save_interval == 0:
                    checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
                    print(f"Saving checkpoint to {checkpoint_path}...")
                    policy_model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)

    wandb.finish()

def evaluate_dpo(policy_model, ref_model, loader, beta, device):
    policy_model.eval()
    losses, accs, margins = [], [], []
    with torch.no_grad():
        for batch in loader:
            c_ids, r_ids = batch["chosen_input_ids"].to(device), batch["rejected_input_ids"].to(device)
            c_mask, r_mask = batch["chosen_attention_mask"].to(device), batch["rejected_attention_mask"].to(device)
            c_p_len, r_p_len = batch["chosen_prompt_len"].to(device), batch["rejected_prompt_len"].to(device)

            c_lp, _ = compute_response_log_probs(policy_model(c_ids, attention_mask=c_mask).logits, c_ids, c_mask, c_p_len)
            c_lp_ref, _ = compute_response_log_probs(ref_model(c_ids, attention_mask=c_mask).logits, c_ids, c_mask, c_p_len)
            r_lp, _ = compute_response_log_probs(policy_model(r_ids, attention_mask=r_mask).logits, r_ids, r_mask, r_p_len)
            r_lp_ref, _ = compute_response_log_probs(ref_model(r_ids, attention_mask=r_mask).logits, r_ids, r_mask, r_p_len)

            c_rew, r_rew = beta * (c_lp - c_lp_ref), beta * (r_lp - r_lp_ref)
            losses.append(-F.logsigmoid(c_rew - r_rew).mean().item())
            accs.append((c_rew > r_rew).float().mean().item())
            margins.append((c_rew - r_rew).mean().item())
            
    policy_model.train()
    return np.mean(losses), np.mean(accs), np.mean(margins)

if __name__ == "__main__":
    train_dpo()