# Phase 2b：Qwen2.5-VL-7B + GRPO 训练计划

## 一、目标

训练一个本地 Qwen2.5-VL-7B 美学评估模型，替代当前 Evaluator 中的 VLM API 调用（`score_aesthetic`），并作为 MCTS 启发式评分的升级方案。

### 当前瓶颈

| 问题 | 现状 | Phase 2b 解决后 |
|------|------|----------------|
| Evaluator 美学评分 | 每轮 3 次 VLM API（带图），占 token 28% | 本地推理，0 API |
| MCTS 评分区分度 | 启发式规则，同类方案得分相同 | 本地 VLM 看关键帧评估，区分度高 |
| 单次运行 token | ~400k | ~150k（仅 Perceiver） |
| 单次运行耗时 | ~30 分钟 | ~12 分钟 |

---

## 二、模型方案

### 2.1 基座模型

**Qwen2.5-VL-7B-Instruct**
- 参数量：7B
- 视觉能力：原生支持图片输入，理解画面内容和质量
- 开源：Hugging Face 免费下载
- 显存需求：全量推理 ~16GB，LoRA 训练 ~20-24GB（4-bit 量化可降到 16GB）

### 2.2 训练方法：GRPO（Group Relative Policy Optimization）

```
核心思想：
  1. 给模型一组相同输入（编辑前后的帧对比图）
  2. 模型生成 K 个不同的评估回答（不同 temperature 采样）
  3. 用奖励信号对 K 个回答排序
  4. GRPO 损失函数让模型偏好高奖励回答

与传统 RLHF 的区别：
  - RLHF 需要单独训练 reward model → 再用 PPO 对齐
  - GRPO 直接用组内相对排名训练，不需要独立 reward model
  - 更简单、更稳定、数据需求更小
```

### 2.3 微调方式：LoRA

```
LoRA 配置：
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  dropout: 0.05

可训练参数：~10M（占 7B 的 0.14%）
显存节省：全量微调需要 ~50GB，LoRA 只需 ~20GB
4-bit 量化 + LoRA：可降到 ~16GB
```

---

## 三、训练数据

### 3.1 数据来源汇总

| 来源 | 数据量 | 类型 | 状态 |
|------|--------|------|------|
| Phase 2a 合成编辑对 | 10,485 组 | (原始帧, 编辑帧, 质量分) | ✅ 已有 |
| Phase 2a CLIP 特征 | 11,184 张 | CLIP ViT-B-32 特征 | ✅ 已有 |
| Pipeline 运行数据 | ~100 组（当前） | VLM 评分 + 编辑前后帧 | 需要扩充到 500+ |
| KoNViD-1k | 1,200 段视频 | MOS 质量分 | 需要下载 |
| AVA 子集 | ~20,000 张 | 美学评分 1-10 | 可选 |

### 3.2 Phase 2b 需要的额外数据

**核心需求：编辑前后帧对比 + 质量标签**

与 CLIP+MLP 不同，Qwen-VL 的训练输入是**图片 + 文本**，不是特征向量。所以需要：

```
每条训练数据 = {
    "编辑前帧图片": image_before.jpg,
    "编辑后帧图片": image_after.jpg,
    "人类/VLM 偏好标签": "A 比 B 好" 或 "分数 0.72",
    "评估理由"（可选）: "色彩更和谐，但锐化过度..."
}
```

**数据扩充计划**：

```
Step 1：从已有合成数据中筛选
  - 10,485 组合成编辑对已有帧图片和质量分
  - 筛选质量分差距 > 0.2 的偏好对 → ~3,000 组高置信度偏好对
  - 直接可用，不需要额外标注

Step 2：Pipeline 数据收集
  - 用当前系统（MCTS 本地 + Evaluator VLM）跑 30-50 个视频
  - 自动保存每次评估的帧图片 + VLM 美学分数
  - 预计产出 ~500 组真实偏好对
  - Token 消耗：~300k × 50 视频 ≈ 15M token

Step 3：KoNViD-1k 质量数据
  - 从每段视频提取中间帧
  - MOS 分归一化作为质量标签
  - 高 MOS vs 低 MOS 帧 = 偏好对 ~2,000 组

合计：~5,500 组偏好对 + 10,485 组合成三元组
```

### 3.3 训练数据格式

**GRPO 训练需要的格式**（兼容 TRL 库）：

```json
{
    "prompt": [
        {
            "role": "system",
            "content": "你是视频编辑质量评审专家。对比编辑前后的画面，评估编辑效果。\n\n评分标准：\n- 0.5 = 无变化\n- >0.5 = 编辑后更好\n- <0.5 = 编辑后变差\n- 引入伪影/锯齿/过度处理必须给低分\n\n返回 JSON: {\"score\": 0.0-1.0, \"reason\": \"...\"}"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "原始画面:"},
                {"type": "image", "image": "path/to/original.jpg"},
                {"type": "text", "text": "编辑后画面:"},
                {"type": "image", "image": "path/to/edited.jpg"},
                {"type": "text", "text": "请评估编辑效果。"}
            ]
        }
    ],
    "reward": 0.72
}
```

**数据转换脚本需要做的事**：
1. 从 `labels.json` 读取合成编辑对
2. 将质量分转为 reward 信号
3. 将帧图片路径写入 prompt
4. 输出 JSONL 格式的训练文件

---

## 四、训练配置

### 4.1 环境要求

```
硬件：
  GPU：1× A100 40GB（推荐）或 1× A100 80GB
  替代方案：1× RTX 4090 24GB（需要 4-bit 量化）

  云 GPU 推荐平台：
  - AutoDL：A100 40GB ~¥3/小时
  - 阿里云 PAI：A100 按量计费
  - Google Colab Pro：A100 40GB

软件：
  Python >= 3.10
  PyTorch >= 2.1
  transformers >= 4.45
  trl >= 0.12.0
  peft >= 0.13.0
  bitsandbytes >= 0.44.0（4-bit 量化）
  accelerate >= 1.0
  qwen-vl-utils（Qwen-VL 图片处理工具）
```

### 4.2 训练超参数

```yaml
# GRPO 训练配置
model:
  name: "Qwen/Qwen2.5-VL-7B-Instruct"
  torch_dtype: "bfloat16"
  quantization: "4bit"  # 如果显存不够

lora:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  dropout: 0.05

grpo:
  num_generations: 4        # 每组生成 4 个回答
  temperature: 0.7          # 生成多样性
  max_new_tokens: 200       # 回答长度限制
  beta: 0.1                 # KL 散度惩罚系数

training:
  learning_rate: 5e-5
  batch_size: 2              # 每 GPU
  gradient_accumulation: 8   # 有效批大小 = 2 × 8 = 16
  num_epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  save_steps: 100
  eval_steps: 50

optimizer:
  name: "adamw_torch"
  # 使用 8-bit Adam 可进一步节省显存
```

### 4.3 训练时间估算

```
数据量：~5,500 组偏好对
GRPO 每组生成 4 个回答 → 实际前向/反向传播 ~22,000 次

A100 40GB：
  - 每步耗时 ~3-5 秒（含 4 次生成 + 1 次策略更新）
  - 3 epochs × ~350 步 ≈ 1,050 步
  - 总时间：~1-1.5 小时

RTX 4090 24GB（4-bit 量化）：
  - 每步耗时 ~6-10 秒
  - 总时间：~2-3 小时

云 GPU 成本：
  - A100 40GB × 2 小时 × ¥3/小时 ≈ ¥6
  - 加上数据准备和验证 ~¥10-15
```

---

## 五、训练流程

### 5.1 数据准备（本地完成）

```
Step 1：准备合成数据 → GRPO 格式
  输入：training/data/synthetic/labels.json + frames/
  输出：training/data/grpo/synthetic_train.jsonl

Step 2：准备 Pipeline 数据 → GRPO 格式
  输入：training/data/pipeline/*.json
  输出：training/data/grpo/pipeline_train.jsonl

Step 3：合并 + 划分
  合并 → training/data/grpo/train.jsonl (80%)
        training/data/grpo/val.jsonl (20%)

Step 4：打包上传到云 GPU
  tar -czf grpo_training_data.tar.gz training/data/grpo/ training/data/synthetic/frames/
```

### 5.2 云 GPU 训练

```bash
# 1. 安装依赖
pip install torch transformers trl peft bitsandbytes accelerate qwen-vl-utils

# 2. 下载基座模型
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./qwen2.5-vl-7b

# 3. 解压训练数据
tar -xzf grpo_training_data.tar.gz

# 4. 运行训练脚本
python train_grpo.py \
  --model_path ./qwen2.5-vl-7b \
  --train_data training/data/grpo/train.jsonl \
  --val_data training/data/grpo/val.jsonl \
  --output_dir ./grpo_output \
  --lora_rank 16 \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation 8 \
  --learning_rate 5e-5

# 5. 导出 LoRA 权重
python merge_lora.py \
  --base_model ./qwen2.5-vl-7b \
  --lora_path ./grpo_output/checkpoint-best \
  --output_dir ./grpo_merged
```

### 5.3 验证（云 GPU 上）

```
离线验证指标：
  1. Pairwise Accuracy（偏好对预测准确率）→ 目标 > 90%
  2. Kendall τ（与 VLM API 打分的排名相关性）→ 目标 > 0.8
  3. 分数分布分析：确保模型不会给所有输入相同分数

端到端验证：
  1. 用训练好的模型替代 Evaluator VLM 调用
  2. 跑 5 个测试视频
  3. 对比模型版本 vs VLM API 版本的最终输出
```

### 5.4 部署回本地

```
下载 LoRA 权重（或合并后的完整模型）到本地：
  training/models/qwen_vl_grpo_lora/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...

或合并后的完整模型（更大但推理更快）：
  training/models/qwen_vl_grpo_merged/
    ├── config.json
    ├── model-*.safetensors
    └── ...

本地推理选项：
  A. transformers 直接加载（需要 16GB+ 显存）
  B. vLLM 部署（推理速度提升 3-5 倍）
  C. llama.cpp / GGUF 量化（可在 CPU 或 8GB 显卡上运行，但较慢）
```

---

## 六、集成到 VlogAgent

### 6.1 替代 Evaluator 美学评分

```python
# evaluator.py 中新增本地 VLM 评估方法

class Evaluator:
    def __init__(self, config, llm):
        # ... 现有代码 ...
        self.local_vlm = None
        self.local_vlm_path = config.get("local_vlm_path", None)

    def _load_local_vlm(self):
        if self.local_vlm is not None:
            return
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.local_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.local_vlm_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.local_processor = AutoProcessor.from_pretrained(self.local_vlm_path)

    def score_aesthetic(self, original_path, edited_path, scenes=None):
        if self.local_vlm_path:
            return self._score_aesthetic_local(original_path, edited_path, scenes)
        else:
            return self._score_aesthetic_api(original_path, edited_path, scenes)
```

### 6.2 替代 MCTS 启发式评分

当前 MCTS 的 `_simulate` 用启发式规则评分（区分度低）。训练好的 Qwen-VL 可以：
1. 看当前关键帧 + 提议的动作方案
2. 预测执行后的质量变化
3. 比启发式规则更准确

但由于 MCTS 模拟次数较多（50 次），本地 VLM 推理 ~200ms/次，50 次 = ~10 秒，仍然可接受。

### 6.3 配置切换

```yaml
# configs/default.yaml

evaluator:
  # 模型选择（二选一）
  # 方式 1：VLM API（当前）
  use_local_vlm: false

  # 方式 2：本地 Qwen-VL（Phase 2b 训练后）
  # use_local_vlm: true
  # local_vlm_path: "training/models/qwen_vl_grpo_merged"
```

---

## 七、时间线

```
Day 1-2：数据准备
  - Pipeline 数据收集脚本（自动保存帧+评分）
  - 合成数据 → GRPO 格式转换脚本
  - 跑 30-50 个视频积累 Pipeline 数据

Day 3：云 GPU 训练
  - 租用 A100，上传数据
  - GRPO 训练 ~2 小时
  - 离线验证

Day 4：集成 + 端到端验证
  - 下载权重回本地
  - 集成到 Evaluator 和 MCTS
  - 端到端测试 5 个视频

Day 5：优化 + 文档
  - 如有问题调参重训
  - 更新实施计划
  - 性能基准测试
```

---

## 八、风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 本地无 GPU 推理太慢 | 中 | llama.cpp GGUF 量化，或继续用 API |
| 训练数据不够，模型泛化差 | 中 | 增加合成数据多样性 + KoNViD-1k 补充 |
| GRPO 训练不稳定 | 低 | 降低学习率，增加 warmup，用 SFT 预热 |
| 云 GPU 租用麻烦 | 低 | AutoDL 一键租用，Colab 备选 |
| 模型太大本地跑不了 | 中 | GGUF Q4 量化到 ~4GB，CPU 也能跑 |

---

## 九、预期效果

### 训练前（当前）

```
Pipeline 单次运行：
  VLM API 调用：~30 次
  Token 消耗：~400k
  总耗时：~30 分钟
  MCTS 区分度：低（启发式规则）
  Evaluator 美学：依赖 API
```

### 训练后

```
Pipeline 单次运行：
  VLM API 调用：~6 次（仅 Perceiver 观察 + 建议）
  Token 消耗：~150k（降 62%）
  总耗时：~12 分钟（降 60%）
  MCTS 区分度：高（本地 VLM 看帧评估）
  Evaluator 美学：本地推理 <200ms/次
```

### 长期

```
如果 Perceiver 也训练本地模型替代：
  VLM API 调用：0 次
  完全本地化运行
  但 Perceiver 需要更强的模型（生成能力），Phase 3 再考虑
```

---

*本计划最后更新：2026-03-22*
