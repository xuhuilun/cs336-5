import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
import argparse
from tqdm import tqdm

from cs336_alignment.sft_dataset import InstructionDataset
from cs336_alignment.sft_utils import compute_entropy 

def main():
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="model/Qwen2.5-Math-1.5B")
    parser.add_argument("--output_dir", type=str, default="result/checkpoints")
    # 核心超参数
    parser.add_argument("--batch_size", type=int, default=32, help="逻辑总 Batch Size")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="物理单步 Batch Size")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    # 监控与保存
    parser.add_argument("--eval_every_steps", type=int, default=100)
    parser.add_argument("--save_every_steps", type=int, default=200)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    
    args = parser.parse_args()

    # --- 1. 初始化监控 ---
    wandb.init(project="cs336-sft-instruct", name=f"lr{args.lr}-bs{args.batch_size}", config=vars(args))

    # --- 2. 加载模型与分词器 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.gradient_checkpointing_enable()

    # --- 3. 数据流水线 ---
    # 使用之前实现的带 -100 Mask 的 Dataset
    train_ds = InstructionDataset(tokenizer, args.train_path, args.max_seq_len, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True)

    eval_loader = None
    if args.eval_path:
        eval_ds = InstructionDataset(tokenizer, args.eval_path, args.max_seq_len, shuffle=False)
        eval_loader = DataLoader(eval_ds, batch_size=args.micro_batch_size, shuffle=False)

    # --- 4. 优化器与学习率调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    grad_accum_steps = max(1, args.batch_size // args.micro_batch_size)
    total_steps = (len(train_loader) // grad_accum_steps) * args.epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    # --- 5. 训练主循环 ---
    print(f"训练: 总计 {total_steps} 更新步 | 累积步数: {grad_accum_steps}")
    progress_bar = tqdm(total=total_steps, desc="SFT Training")
    
    model.train()
    optimizer.zero_grad()
    
    accumulated_loss = 0
    accumulated_entropy = 0
    global_step = 0

    for epoch in range(args.epochs):
        for idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)

            # --- A. 前向传播与原生 Loss 计算 ---
            
            logits = model(input_ids=input_ids).logits
            
            # 核心知识点：使用 PyTorch 集成的交叉熵，忽略 -100
            # 需要将 [B, L, V] 展平为 [B*L, V]，将 [B, L] 展平为 [B*L]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            
            # 梯度累积缩放
            scaled_loss = loss / grad_accum_steps
            
            # --- B. 反向传播 ---
            scaled_loss.backward()

            # 统计指标（不参与计算图）
            accumulated_loss += loss.item()
            with torch.no_grad():
                # 监控 Response Entropy 以防坍缩
                ent_all = compute_entropy(logits)
                active_mask = (labels != -100)
                if active_mask.any():
                    accumulated_entropy += ent_all[active_mask].mean().item()

            # --- C. 参数更新（触发条件：完成一个逻辑 Batch） ---
            if (idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()    # 应用梯度
                scheduler.step()    # 更新学习率（修改 optimizer 内部的 param_groups）
                optimizer.zero_grad() # 清空现场
                
                global_step += 1
                
                # 日志记录
                metrics = {
                    "train/loss": accumulated_loss / grad_accum_steps,
                    "train/response_entropy": accumulated_entropy / grad_accum_steps,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch + (idx / len(train_loader))
                }

                # 周期性验证
                if eval_loader and global_step % args.eval_every_steps == 0:
                    val_loss = run_evaluation(model, eval_loader)
                    metrics["eval/loss"] = val_loss
                    model.train()

                # 周期性保存
                if global_step % args.save_every_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)

                wandb.log(metrics, step=global_step)
                progress_bar.set_postfix({"loss": f"{metrics['train/loss']:.4f}"})
                progress_bar.update(1)
                
                accumulated_loss = 0
                accumulated_entropy = 0

    # 最终保存
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()

def run_evaluation(model, eval_loader):
    model.eval()
    total_loss = 0
    max_batches = 50
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_batches: break
            input_ids, labels = batch["input_ids"].to(model.device), batch["labels"].to(model.device)
            logits = model(input_ids).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            total_loss += loss.item()
            count += 1
    return total_loss / max(1, count)

if __name__ == "__main__":
    main()