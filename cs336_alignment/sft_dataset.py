import json
import gzip
import torch
import random
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length: int, shuffle: bool, apply_masking: bool = True):
        self.tokenizer = tokenizer
        # 为了保证 Shift 时有足够的元素，我们实际需要的块长度是 seq_length + 1
        # 但对外暴露的还是 seq_length
        self.seq_length = seq_length 
        self.apply_masking = apply_masking
        
        dataset_path_str = str(dataset_path)
        
        # 1. 加载数据
        docs = []
        open_fn = gzip.open if dataset_path_str.endswith(".gz") else open
        with open_fn(dataset_path_str, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        # 2. 打乱顺序
        if shuffle:
            random.seed(42)
            random.shuffle(docs)

        # 3. 模版定义 (Alpaca 风格)
        template = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{prompt}\n\n"
            "### Response:\n{response}"
        )
        
        all_token_ids = []
        all_label_ids = []
        eos_token_id = tokenizer.eos_token_id
        
        print("Tokenizing and applying masks...")
        
        # 4. 编码与掩码
        for i, doc in enumerate(docs):
            p = doc.get("prompt", doc.get("instruction", ""))
            r = doc.get("response", doc.get("output", ""))
            
            # 完整文本编码
            formatted_text = template.format(prompt=p, response=r)
            full_tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
            
            # 标签初始化（默认等于 input_ids）
            labels = list(full_tokens)
            
            if self.apply_masking:
                # 仅 Prompt 编码
                prompt_text = template.format(prompt=p, response="")
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                
                # 安全检查：确保前缀完全一致
                if full_tokens[:len(prompt_tokens)] != prompt_tokens:
                    print(f"Warning: Tokenization mismatch at index {i}. Masking might be slightly off.")
                
                # 执行掩码：将 Prompt 部分的标签设为 -100
                mask_len = len(prompt_tokens)
                labels[:mask_len] = [-100] * mask_len
            
            # 将当前文档追加到长链，并在末尾添加 EOS
            all_token_ids.extend(full_tokens) # [xxx].extend([yyy]) -> [xxxyyy]
            all_token_ids.append(eos_token_id) # [xxx].append(x) -> [xxxyyyx]
            
            all_label_ids.extend(labels)
            # EOS 是模型必须学习输出的，所以标签处保留其真实 ID，不能设为 -100
            all_label_ids.append(eos_token_id)
            
        # 转换为 Tensor 以便进行高效操作
        all_token_ids = torch.tensor(all_token_ids, dtype=torch.long)
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)

        # 5. 执行 Packing 与 Shift 逻辑
        print("Packing sequences into fixed chunks...")
        
        # 为了能平移一位（Shift），我们需要切出一个长为 seq_length + 1 的块
        chunk_size = self.seq_length + 1 
        num_chunks = len(all_token_ids) // chunk_size
        
        if num_chunks == 0:
            raise ValueError(f"Dataset is too small! Total tokens ({len(all_token_ids)}) < required chunk size ({chunk_size})")

        # 截断多余的部分并变形
        input_chunks = all_token_ids[:num_chunks * chunk_size].view(num_chunks, chunk_size)
        label_chunks = all_label_ids[:num_chunks * chunk_size].view(num_chunks, chunk_size)

        # 6. 执行 Shift 逻辑：产生最终的输入和标签
        # 输入：取前 seq_length 个 [0 : N-1]
        self.final_input_ids = input_chunks[:, :-1]
        # 标签：取后 seq_length 个 [1 : N]
        self.final_labels = label_chunks[:, 1:]

        # 7. “死区”剔除 (Filtering Invalid Chunks)
        # 检查每个序列的 Labels：如果一整行全都是 -100，说明这 chunk_size 个 token 全是 Prompt
        # 这种数据无法产生梯度，必须丢弃。
        valid_chunk_mask = (self.final_labels != -100).any(dim=1)
        
        total_dropped = num_chunks - valid_chunk_mask.sum().item()
        if total_dropped > 0:
            print(f"Dropped {total_dropped} / {num_chunks} chunks ({total_dropped/num_chunks:.1%}) that contained ONLY masked prompt tokens.")
        else:
            print("All chunks are valid.")

        # 保留有效的数据块
        self.final_input_ids = self.final_input_ids[valid_chunk_mask]
        self.final_labels = self.final_labels[valid_chunk_mask]

    def __len__(self):
        return len(self.final_input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.final_input_ids[i].clone(),
            "labels": self.final_labels[i].clone()
        }