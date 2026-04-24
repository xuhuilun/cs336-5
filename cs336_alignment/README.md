# CS336 Alignment 项目文件阅读指南

本项目是一个完整的LLM对齐（Alignment）研究框架，包含数据预处理、多种对齐算法（SFT、DPO、GRPO、Expert Iteration）的实现与评估。

---

## 阅读顺序总览

```
第一阶段：数据准备（了解输入数据格式）
    ↓
第二阶段：工具模块（理解核心工具函数）
    ↓
第三阶段：基础训练（掌握SFT实现）
    ↓
第四阶段：高级算法（学习DPO/GRPO/EI）
    ↓
第五阶段：评估测试（模型评估与交互）
```

---

## 第一阶段：数据准备（Data Preparation）

### 1. `parse_utils.py` ⭐ 优先阅读
- **作用**：解析模型输出的工具函数
- **核心功能**：
  - `parse_mmlu_response()`: 解析MMLU选择题答案（匹配 "The correct answer is [A-D]"）
  - `parse_gsm8k_response()`: 提取数学问题的最后一个数字答案
- **为什么先读**：这是数据处理的底层工具，理解如何提取模型输出

### 2. `drgrpo_grader.py` ⭐⭐ 重点阅读
- **作用**：数学答案评分器（从开源项目改编）
- **核心功能**：
  - `r1_zero_reward_fn()`: R1-Zero风格的奖励函数（检查格式+答案正确性）
  - `question_only_reward_fn()`: 仅检查答案正确性
  - `grade()`: 综合评分逻辑（支持LaTeX、数值、符号比较）
- **关键概念**：
  - 格式奖励（format_reward）：是否遵循 `<think>...</think> <answer>...</answer>` 格式
  - 答案奖励（answer_reward）：答案是否正确
  - 总奖励（reward）：format * answer
- **为什么重要**：这是所有训练和评估的基础评分逻辑

### 3. `convert_gsm8k.py`
- **作用**：将GSM8K原始数据转换为R1风格格式
- **输入格式**：`{question, answer: "推理过程####数字答案"}`
- **输出格式**：`{question, prompt, response, gold, is_correct, reward_details}`
- **转换逻辑**：
  - 清洗GSM8K的计算标签 `<<...>>`
  - 构造System Prompt（R1风格）
  - 分离推理过程和最终答案

### 4. `convert_hh.py`
- **作用**：加载Anthropic HH-RLHF偏好数据集
- **处理逻辑**：
  - 过滤多轮对话（只保留单轮）
  - 提取instruction、chosen、rejected字段
  - 用于DPO训练

### 5. `clean_sft_data.py`
- **作用**：清洗SFT数据并验证正确性
- **核心操作**：
  - 修复Prompt中的重复标签
  - 使用`r1_zero_reward_fn`验证每条数据
  - 输出清洗后的JSONL文件

### 6. `gen_sft_reason_data.py`
- **作用**：使用DeepSeek-R1 API生成高质量的推理数据
- **功能**：
  - 多线程并发调用API
  - 提取reasoning_content和content
  - 构造训练所需的格式

---

## 第二阶段：工具模块（Utility Modules）

### 7. `sft_dataset.py` ⭐⭐ 重点阅读
- **作用**：SFT数据集实现（支持Packing）
- **核心类**：`InstructionDataset`
- **关键机制**：
  - **Prompt Masking**: 只对Response部分计算loss（Prompt部分设为-100）
  - **Packing**: 将多个短样本拼接成固定长度的序列
  - **Shift逻辑**: 输入取`[0:N-1]`，标签取`[1:N]`
  - **死区剔除**: 过滤掉全是Prompt的chunk
- **适用场景**：标准SFT训练的数据加载

### 8. `sft_utils.py` ⭐⭐ 重点阅读
- **作用**：SFT训练的核心工具函数
- **核心函数**：
  - `tokenize_prompt_and_output()`: 分词、拼接、生成response_mask
  - `compute_entropy()`: 计算next-token预测的熵（监控模型置信度）
  - `get_response_log_probs()`: 获取模型对label的log概率
  - `sft_microbatch_train_step()`: 单次微批次SFT更新
  - `log_generations()`: 使用vLLM生成并记录评估指标
- **关键理解**：
  - Response Mask的作用：只让模型学习生成Response
  - Shift逻辑：因果语言模型的标准训练方式

### 9. `grpo_utils.py` ⭐⭐⭐ 核心阅读
- **作用**：GRPO（Group Relative Policy Optimization）算法实现
- **核心函数**：
  - `compute_group_normalized_rewards()`: 计算组内归一化优势（Advantage）
  - `compute_grpo_clip_loss()`: GRPO Clip损失（PPO风格的比率截断）
  - `compute_grpo_no_clip_loss()`: 无截断的GRPO损失
  - `grpo_microbatch_train_step()`: GRPO微批次训练步
  - `log_generations()`: GRPO专用的生成日志记录
- **关键概念**：
  - **Group Normalization**: 同一问题的多个回答之间做归一化
  - **Advantage**: (reward - mean) / std，避免std过于小造成震荡，也可以（reward - mean）
  - **Clip机制**: 限制新旧策略比率在[0.8, 1.2]之间
  - **长度归一化类型**: mask_mean / mask_normalize / mask_dapo
  - 样本公平/token公平

---

## 第三阶段：基础训练（Basic Training）

### 10. `sft.py` ⭐ 入门首选
- **作用**：最基础的SFT训练脚本
- **特点**：
  - 使用`sft_dataset.py`的Dataset
  - 标准的epoch-based训练
  - 简单的梯度累积实现
  - 监控Response Entropy
- **适合**：理解SFT的基本流程

### 11. `train_sft_step.py`
- **作用**：Step-based SFT训练
- **特点**：
  - 一次性预分词整个数据集（速度快）
  - 使用infinite dataloader逻辑（随机采样）
  - 基于step的评估和保存
  - 支持filter_correct（只训练正确答案）
- **适合**：大规模数据的快速实验

### 12. `train_sft_epoch.py`
- **作用**：Epoch-based SFT训练（带vLLM评估）
- **特点**：
  - 每个epoch shuffle数据
  - 使用vLLM进行定期评估
  - 更细致的熵监控（prompt/response/global）
  - 支持dataset_size限制（子集实验）
- **适合**：需要精确控制epoch数的实验

---

## 第四阶段：高级算法（Advanced Algorithms）

### 13. `dpo.py` ⭐⭐ 偏好学习
- **作用**：Direct Preference Optimization实现
- **核心逻辑**：
  - 加载HH-RLHF偏好数据
  - 同时加载Policy模型和Reference模型
  - 计算chosen/rejected的log-prob差异
  - 优化目标：最大化chosen vs rejected的margin
- **关键监控指标**：
  - chosen_reward / rejected_reward
  - reward_margin
  - accuracy（chosen_reward > rejected_reward的比例）

### 14. `train_grpo.py` ⭐⭐⭐ 核心算法
- **作用**：GRPO完整训练流程
- **核心流程（4个Phase）**：
  1. **Phase 1 - Rollout**: 使用vLLM生成回答（group_size个/问题）
  2. **Phase 2 - Filter & Resample**: 长度过滤+重采样补齐
  3. **Phase 3 - Training**: 计算advantage + GRPO loss + 参数更新
  4. **Phase 4 - Eval & Save**: 定期评估和保存checkpoint
- **关键创新**：
  - 组内归一化（不需要critic模型）
  - 在线生成数据（on-policy）
  - 双卡设计（训练卡+推理卡分离）
- **超参数重点**：
  - `group_size`: 每个问题的采样数（G）
  - `rollout_batch_size`: 每次采样的问题数
  - `train_batch_size`: 训练时的batch size
  - `length_norm_type`: 长度归一化策略

### 15. `train_ei_step.py` / `train_ei_epoch.py` ⭐⭐ Expert Iteration
- **作用**：专家迭代算法（Self-Improvement）
- **核心流程**：
  1. **采样阶段**：用当前模型生成多个回答（rollouts）
  2. **过滤阶段**：用reward_fn筛选正确的回答（expert data）
  3. **训练阶段**：在expert data上做SFT
  4. **迭代**：重复上述过程
- **两个版本的区别**：
  - `train_ei_step.py`: 基于step的动态训练（推荐）
  - `train_ei_epoch.py`: 基于epoch的固定轮数训练
- **关键超参数**：
  - `n_ei_steps`: 外层迭代次数
  - `ei_batch_size` (Db): 每次采样的问题数
  - `rollouts` (G): 每个问题的生成数
  - `epochs_per_ei`: 内层SFT的epoch数

---

## 第五阶段：评估测试（Evaluation & Testing）

### 16. `evaluate_zero_shot.py`
- **作用**：零样本（Zero-shot）模型评估
- **流程**：
  - 加载测试数据（R1风格格式）
  - 使用vLLM批量生成
  - 用`r1_zero_reward_fn`评分
  - 输出accuracy、format_error等指标
- **适用**：评估基础模型或训练后的模型

### 17. `evaluate_all_checkpoints.py`
- **作用**：批量评估所有checkpoint并绘制学习曲线
- **功能**：
  - 自动扫描checkpoint目录
  - 支持按学习率分组
  - 生成CSV结果和可视化图表
- **适用**：超参数搜索（learning rate sweep）

### 18. `chat.py`
- **作用**：交互式聊天测试
- **使用**：连接本地vLLM服务，进行实时对话
- **支持**：加载不同的prompt template

---

## 文件依赖关系图

```
                    ┌─────────────────────────────────────┐
                    │         drgrpo_grader.py            │
                    │    (评分器 - 所有模块的基础)         │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  sft_utils.py │      │  grpo_utils.py   │      │  parse_utils.py  │
│  (SFT工具)    │      │  (GRPO工具)      │      │  (解析工具)      │
└───────┬───────┘      └────────┬─────────┘      └──────────────────┘
        │                       │
        ▼                       ▼
┌───────────────┐      ┌──────────────────┐
│sft_dataset.py │      │  train_grpo.py   │
│ (Dataset类)   │      │   (GRPO训练)     │
└───────┬───────┘      └──────────────────┘
        │
        ▼
┌───────────────┐
│    sft.py     │
└───────────────┘
```

---

## 快速上手路径

### 路径1：理解基础SFT（推荐新手）
```
parse_utils.py → drgrpo_grader.py → sft_utils.py → sft.py
```

### 路径2：深入GRPO（推荐研究）
```
drgrpo_grader.py → grpo_utils.py → train_grpo.py
```

### 路径3：完整项目理解
```
convert_gsm8k.py → sft_utils.py → train_sft_step.py → train_grpo.py → evaluate_zero_shot.py
```

---

## 关键概念速查

| 概念 | 解释 | 所在文件 |
|------|------|----------|
| Response Mask | 标记Prompt和Response位置的掩码 | sft_utils.py |
| Packing | 将短序列拼接成长的固定长度序列 | sft_dataset.py |
| Shift逻辑 | 输入[0:N-1]预测[1:N] | sft_dataset.py |
| Group Normalization | 组内奖励归一化 | grpo_utils.py |
| Advantage | 相对优势分数 | grpo_utils.py |
| Clip机制 | 限制策略更新幅度 | grpo_utils.py |
| Expert Iteration | 生成→过滤→训练的迭代循环 | train_ei_step.py |

---

## 附录：数据格式示例

### R1-Zero风格格式（用于GRPO/EI）
```json
{
  "question": "计算 2+3",
  "prompt": "A conversation...User: 计算 2+3\nAssistant: <think>",
  "gold": "5"
}
```

### SFT训练格式
```json
{
  "prompt": "A conversation...User: 计算 2+3\nAssistant: <think>",
  "response": "2+3=5 </think> <answer>5</answer>",
  "gold": "5",
  "is_correct": true
}
```

### DPO偏好格式
```json
{
  "instruction": "你好",
  "chosen": "你好！有什么可以帮助你？",
  "rejected": "走开。"
}
```
