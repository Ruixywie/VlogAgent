# VlogAgent 多 Agent 反思架构设计

## 一、核心思想

### 1.1 从"流水线"到"多角色协作"

当前架构本质是一条固定流水线：Perceiver → Planner → Executor → Evaluator → 循环。每一步做什么、下一步去哪，都是代码写死的。LLM 只是在每一步中被当作工具调用。

新架构引入三个具有不同职责和决策权的 Agent 角色：

```
Director（导演）：负责 "做什么风格" + "要不要继续编辑"  — 战略层  ← 通用 VLM（不训练）
Editor  （编辑）：负责 "具体怎么做"                    — 战术层  ← 通用 VLM + MCTS（不训练）
Critic  （评审）：负责 "做得怎么样" + "为什么" + "怎么改" — 评估层  ← 训练专用模型
```

它们之间不是固定的顺序调用，而是**两层嵌套循环 + 动态路由**：

```
外层循环（Director 驱动）：每轮产出一个被 accept 的方案，累积到 action_chain
  └── 内层循环（Editor-Critic 驱动）：把一个方案打磨到可接受
```

### 1.2 与 PhotoAgent 的算法区别

| | PhotoAgent | VlogAgent（新架构） |
|--|--|--|
| Agent 数量 | 1 个（全能 Agent） | 3 个（专业化角色） |
| 循环结构 | 单层（Perceive→Plan→Execute→score→loop） | 双层（外层策略 + 内层打磨） |
| 闭环信号 | 数字分数（score > threshold） | 结构化反馈（逐段因果分析 + 改进建议） |
| 搜索策略 | 无约束 MCTS | Reflection-Guided Constrained MCTS |
| 失败处理 | 回退到上一版本 | 因果分析为什么失败 → 针对性调整 |
| 决策路由 | 固定流程 | Critic 动态决定回 Director 还是 Editor |
| 训练策略 | 训练 UGC Reward Model | 只训练 Critic（Director/Editor 用通用 VLM） |

### 1.3 形式化

```
标准闭环（PhotoAgent）：
  πₜ = Perceive(sₜ)                    ← 每轮独立感知
  aₜ = Plan(πₜ)                        ← 无约束 MCTS 搜索
  sₜ₊₁ = Execute(sₜ, aₜ)              ← 执行
  rₜ = Evaluate(sₜ₊₁)                 ← 数字评分
  if rₜ > rₜ₋₁: accept else rollback  ← 数字比较

反思闭环（VlogAgent）：
  G = Director(s₀)                               ← 全局风格策略
  while Director.should_continue():               ← 外层循环
    while True:                                   ← 内层循环
      aₜ = Editor(sₜ, G, Fₜ₋₁)                  ← 约束搜索 + 上轮反馈
      sₜ₊₁ = Execute(sₜ, aₜ)                    ← 执行
      (rₜ, Fₜ) = Critic(s₀, sₜ₊₁, aₜ)          ← 分数 + 结构化反馈
      if Fₜ.route == "accept": break             ← 内层接受
      elif Fₜ.route == "redirect": G = Director(s₀, Fₜ); break  ← 换策略
      else: continue                              ← refine，继续内层
    action_chain.extend(aₜ)                       ← 累积动作
  final_execute(original_video, action_chain)      ← 原视频一次性编码
```

---

## 二、两层循环架构

### 2.1 整体架构图

```
原始视频
  │
  ├── Step 1: 视频分析（本地计算）
  │     场景分割 + 质量检测 + 关键帧提取 + 音频分析
  │     输出: SegmentMetadata + 关键帧
  │
  ├── Step 2: Director → StyleBrief               [通用 VLM, 1 次]
  │     VLM 看全片关键帧 → 制定风格策略
  │
  │  ╔═══════════════ 外层循环（Director 驱动）═══════════════╗
  │  ║                                                        ║
  │  ║   ╔═══════ 内层循环（Editor-Critic 打磨）═══════╗      ║
  │  ║   ║                                              ║      ║
  │  ║   ║  Step 3: Editor(MCTS) → EditPlan            ║      ║
  │  ║   ║    ① 通用 VLM 生成候选动作                    ║      ║
  │  ║   ║    ② Director 约束 + Critic 反馈过滤          ║      ║
  │  ║   ║    ③ MCTS 搜索（CLIP+MLP 本地评估）          ║      ║
  │  ║   ║    ④ 输出 EditPlan（2-3 步动作序列）          ║      ║
  │  ║   ║              │                                ║      ║
  │  ║   ║  Step 4: Executor                            ║      ║
  │  ║   ║    在当前视频上执行 EditPlan                   ║      ║
  │  ║   ║              │                                ║      ║
  │  ║   ║  Step 5: Critic → CriticFeedback             ║      ║
  │  ║   ║    逐段对比 + 因果分析 + 路由决策              ║      ║
  │  ║   ║              │                                ║      ║
  │  ║   ║  路由决策：                                   ║      ║
  │  ║   ║    accept → 跳出内层 ─────────────────→      ║      ║
  │  ║   ║    refine → 回到 Step 3（带反馈微调）         ║      ║
  │  ║   ║    redirect → 跳出内层，回 Step 2 ────→      ║      ║
  │  ║   ║                                              ║      ║
  │  ║   ╚══════════════════════════════════════════════╝      ║
  │  ║                    │                                    ║
  │  ║        accept 的方案追加到 action_chain                  ║
  │  ║        current_video 更新为编辑后结果                    ║
  │  ║                    │                                    ║
  │  ║  Step 6: Director 判断是否继续                           ║
  │  ║    score > 0.85 或无新问题 → 终止外层                    ║
  │  ║    还有改进空间 → 新一轮外层循环                          ║
  │  ║    达到最大轮数 → 终止外层                               ║
  │  ║                                                        ║
  │  ╚════════════════════════════════════════════════════════╝
  │                    │
  │          action_chain = [第1轮方案 + 第2轮方案 + ...]
  │                    │
  ├── Step 7: 在原始视频上一次性执行完整 action_chain
  │     段级编辑：enable='between(t,start,end)' 滤镜仅对目标段生效
  │     滤镜合并为一条 FFmpeg 链（含段级 enable）
  │     stabilize 仅支持全局模式，段级 stabilize 自动跳过
  │     CRF=10 视觉无损单次编码
  │
  ▼
最终输出视频 + 决策日志
```

### 2.2 两层循环的分工

| | 内层循环（Editor-Critic） | 外层循环（Director 驱动） |
|--|--|--|
| **目标** | 把一个方案打磨到 Critic 认可 | 决定要不要继续添加更多编辑 |
| **决策者** | Critic（refine/accept/redirect） | Director（继续/终止） |
| **每轮产出** | 一个被 accept 的 EditPlan | 追加到 action_chain |
| **典型次数** | 1-3 轮（首轮可能直接 accept） | 1-3 轮（视频好的话 1 轮就终止） |
| **终止条件** | Critic accept 或 redirect | Director 判断"够了"或达到最大轮数 |

### 2.3 具体流程示例

```
═══════ Step 2: Director ═══════════════════════
Director → StyleBrief: "清新田园风格，保留暖色调，保留动态感"
action_chain = []

═══════ 外层第 1 轮 ════════════════════════════

  ── 内层第 1 轮 ──
  Editor(MCTS) → [stabilize(seg-1, smoothing=12), color_adjust(seg-1, br=0.1)]
  Executor → 编辑后视频
  Critic → seg-1: degraded（stabilize 裁切 + br 过高）
           route = "refine"
           suggestion: "去掉 stabilize，br 降到 0.05"

  ── 内层第 2 轮 ──
  Editor 收到反馈 → 排除 stabilize，缩小 br 范围
  Editor(MCTS) → [color_adjust(seg-1, br=0.05)]
  Executor → 编辑后视频
  Critic → seg-1: improved（提亮自然）
           route = "accept"

  → 方案 [color_adjust(seg-1, br=0.05)] 追加到 action_chain
  → Director 看结果："seg-2 色彩饱和度偏高，还可以优化" → 继续

═══════ 外层第 2 轮 ════════════════════════════

  ── 内层第 1 轮 ──
  Editor(MCTS) → [color_adjust(seg-2, sat=0.9), auto_color_harmonize]
  Executor → 编辑后视频
  Critic → 全段 improved，段间一致性提升
           route = "accept"

  → 方案追加到 action_chain
  → Director 看结果："视频已经很好了" → 终止

═══════ Step 7: 最终执行 ════════════════════════

action_chain = [
  color_adjust(seg-1, br=0.05),        ← 外层第 1 轮
  color_adjust(seg-2, sat=0.9),        ← 外层第 2 轮
  auto_color_harmonize                 ← 外层第 2 轮
]

原始视频 → 合并滤镜链 → CRF=10 单次编码 → 输出
```

### 2.4 Director 的继续/终止判断

Director 在每轮外层循环结束后做轻量判断：

```
终止条件（满足任一即终止）：
  1. 达到最大外层迭代次数（默认 3）
  2. Critic 的 overall_score > 0.85（视频已足够好）
  3. Critic 没有发现任何 degraded 段且无 global_issues（无需继续）
  4. 连续 2 轮外层循环 Critic score 无提升

继续条件：
  Critic 仍有 global_issues 或 degraded 段 → 还有改进空间 → 继续
```

---

## 三、三个 Agent 的详细设计

### 3.1 Director Agent（导演）— 通用 VLM，不训练

**职责**：
1. 初始化：观察视频全貌 → 制定 StyleBrief
2. 外层循环：判断是否继续编辑
3. redirect 时：根据 Critic 反馈调整风格策略

**为什么不需要训练**：Director 的任务是创意性的（判断视频风格、情绪、方向），通用 LLM 擅长。调用频率低（一次运行 1-3 次），不是瓶颈。

**输出**：StyleBrief（含分阶段规划）

```python
@dataclass
class StyleBrief:
    overall_style: str          # "清新自然" / "电影感"
    color_direction: str        # "保持暖色调" / "偏冷"
    priority: str               # "优先修复暗段曝光"
    constraints: list[str]      # "不要对高速运动段做 stabilize"
    target_mood: str            # "宁静治愈"
    stages: list[StageDecision] # 分阶段规划（Stage-Aware Planning）

# Director 输出示例：
stages = [
    StageDecision(stage="stabilize", scope="skip"),
    StageDecision(stage="denoise", scope="global", direction="轻度降噪，保留纹理"),
    StageDecision(stage="color_correct", scope="per_segment", direction="校正seg-1欠曝", target_segments=["seg-1"]),
    StageDecision(stage="color_grade", scope="global", direction="偏暖电影感色调"),
    StageDecision(stage="sharpen", scope="skip"),
]
```

**Stage-Aware Planning 的价值**：
- 顺序来自 Director 的**专业判断**（不是代码硬编码）
- 每阶段 Agent 自主决策：跳过 / 全局 / 段级
- color_correct（技术校正）和 color_grade（创意调色）分离，对齐专业工作流

### 3.2 Editor Agent（编辑师）— 通用 VLM + 逐阶段视觉预筛选，不训练

**职责**：在 Director 的分阶段规划下，为每个阶段生成候选方案并通过 PIL 模拟 + CLIP+MLP 预筛选最优。

**为什么不需要训练**：三层保障弥补通用 LLM 的不足：
1. Director 的 StyleBrief 约束搜索方向
2. MCTS 搜索弥补单次生成不够好
3. Critic 反馈纠正错误（内层循环打磨）

**内部流程——Reflection-Guided Constrained MCTS**：

```
首轮（无 Critic 反馈）：
  ① 通用 VLM 根据 StyleBrief 生成 8-15 条候选
  ② 排除不符合 Director 约束的候选
  ③ MCTS 搜索（CLIP+MLP 评估）→ Top-K 方案
  ④ 输出最佳 EditPlan

后续轮（有 Critic 反馈）：
  ① 通用 VLM 根据 StyleBrief + Critic 反馈生成候选
  ② 排除 Director 约束外 + Critic 明确否定的候选
  ③ Critic 建议的动作初始 Q₀ = 0.7（引导搜索先验）
  ④ MCTS 在缩小的空间中搜索 → 更精准
```

**与 PhotoAgent MCTS 的算法区别**：

```
PhotoAgent MCTS：
  搜索空间 = 全部候选 × 所有排列
  先验 = 无（Q₀ = 0）
  评估 = VLM API（3-5s/次）

Reflection-Guided Constrained MCTS：
  搜索空间 = Director 约束 ∩ ¬Critic 排除      ← 逐轮缩小
  先验 = Critic 建议 → Q₀ = 0.7                ← 引导搜索方向
  参数范围 = Critic 指定的安全范围               ← 避免过度处理
  评估 = 本地 CLIP+MLP（<10ms/次）             ← 快且免费
```

### 3.3 Critic Agent（评审）— 训练专用模型

**职责**：
1. 对比编辑前后画面 → 逐段因果分析
2. 输出结构化反馈 → 改进建议
3. 路由决策 → accept / refine / redirect

**为什么必须训练**：
- 需要一致性（同样效果每次评分相近）
- 需要结构化输出（逐段 JSON，路由决策）
- 调用频率最高（每次内层循环都调）
- 对齐 PhotoAgent：只训练 Evaluator 的设计哲学

**输出**：CriticFeedback

```python
@dataclass
class SegmentCritic:
    segment_id: str
    verdict: str                    # "improved" / "unchanged" / "degraded"
    reason: str                     # 因果分析
    action_feedback: dict           # 具体到每个动作的反馈

@dataclass
class CriticFeedback:
    overall_score: float            # 0-1 综合分
    segment_feedback: dict[str, SegmentCritic]  # 逐段反馈
    global_issues: list[str]        # 全局问题
    global_positives: list[str]     # 全局优点
    suggestions: list[str]          # 改进建议
    route: str                      # "accept" / "refine" / "redirect"
    route_reason: str               # 路由理由
```

**路由决策逻辑**：

```
overall_score > prev_score + 阈值？
  ├── Yes → "accept"
  └── No → 分析失败原因
         ├── 少数段 degraded → "refine"（回 Editor 微调）
         ├── 参数过度 → "refine"（缩小参数范围）
         ├── 方向错误 → "redirect"（回 Director 换策略）
         └── 连续 2 轮 refine 无提升 → "redirect" 或终止
```

**Critic 的两层实现**：

```
┌───────────────────────────────────────────────┐
│ Critic Agent                                   │
│                                                │
│  Layer 1：CLIP+MLP 快速评分（Phase 2a ✅ 已训练）│
│    用途：Editor 内部 MCTS 的模拟评估             │
│    速度：<10ms/次                               │
│    输出：单一分数 [0,1]                          │
│                                                │
│  Layer 2：Qwen-VL-7B + GRPO（Phase 2b 待训练）  │
│    用途：每轮内层循环的完整评审 + 路由决策         │
│    速度：~200ms/次（本地推理）                    │
│    输出：CriticFeedback（分数+逐段分析+路由）     │
│    过渡期：通用 VLM API                          │
└───────────────────────────────────────────────┘
```

---

## 四、训练策略：只训练 Critic

### 4.1 为什么只训练 Critic

| Agent | 是否训练 | 理由 |
|-------|:---:|------|
| Director | 否 | 创意性任务，通用 LLM 擅长；调用 1-3 次/运行 |
| Editor | 否 | MCTS + Director 约束 + Critic 反馈三层保障 |
| **Critic** | **是** | 需要一致性/结构化输出/高频调用；对齐 PhotoAgent |

### 4.2 Critic 两层训练计划

#### Layer 1：CLIP+MLP（Phase 2a ✅ 已完成）

```
状态：已训练，验证准确率 81.1%
用途：Editor 内部 MCTS 的模拟评估
速度：<10ms/次
部署：MCTSPlanner._simulate()
```

#### Layer 2：Qwen-VL-7B + GRPO（Phase 2b 待训练）

```
状态：待训练（需要云 GPU）
用途：每轮内层循环的完整评审 + 路由决策
速度：~200ms/次（本地推理）
部署：Critic.evaluate()

训练数据：
  - 合成编辑对（10,485 组）→ 自动推导 verdict/reason/route
  - Pipeline 运行数据（~500 组）→ VLM 结构化反馈作为标注
  - 公开数据集（KoNViD-1k 等）→ 质量分作为参考

训练配置：
  基座：Qwen2.5-VL-7B-Instruct
  方式：LoRA (rank=16) + GRPO
  显存：24GB+（4-bit 量化可降到 16GB）
  时间：~2 小时（A100）
  成本：~¥10
```

### 4.3 训练数据来源

| 来源 | 数据量 | 内容 | 成本 |
|------|--------|------|------|
| Phase 2a 合成编辑对 | 10,485 组 | 帧图片 + 质量分 | 已有 |
| 公开数据集 | ~3,000 组 | 帧 + MOS 质量分 | 免费下载 |
| Pipeline 运行 | ~500 组 | VLM 评分 + 帧 + 结构化反馈 | ~15M token |

**结构化反馈的标注**（Phase 2b 特有）：

```
合成数据 → 自动推导：
  brightness=0.3（过高）→ verdict="degraded", reason="过度提亮"
  brightness=0.05（温和）→ verdict="improved", reason="轻微提亮自然"

Pipeline 运行 → VLM 输出直接作为标注：
  Critic VLM 返回的 JSON → 作为 GRPO 训练目标

路由决策 → 从 accept/reject 记录推导：
  score 提升 → route="accept"
  score 下降但部分段 OK → route="refine"
  连续下降 → route="redirect"
```

---

## 五、VLM 调用量对比

### 典型场景（外层 2 轮 × 内层 1-2 轮）

| 阶段 | 当前架构 | 新架构（过渡期） | Critic 训练后 |
|------|:---:|:---:|:---:|
| Director | — | 1 次 | 1 次 |
| Editor 候选生成 | 6 次 | 3 次 | 3 次 |
| MCTS 评估 | 0（本地） | 0（本地） | 0（本地） |
| ToolSelector | ~18 次 | ~6 次 | ~6 次 |
| Critic | ~10 次 | 3 次 | **0（本地）** |
| **合计** | **~34 次** | **~13 次** | **~10 次** |

---

## 六、论文故事线

### 标题

```
VlogAgent: Self-Reflective Multi-Agent Video Enhancement
via Critic-Guided Constrained Planning
```

### 核心贡献

```
1. 问题定义：
   首次将视频美学编辑定义为"时序耦合的多段联合优化"

2. Director-Editor-Critic 多 Agent 协作架构：
   - 双层循环：外层控制整体策略，内层打磨具体方案
   - 动态路由：Critic 因果分析 → refine/redirect/accept
   - 增量编辑：只修有问题的部分，保留已验证的好决策

3. Reflection-Guided Constrained MCTS：
   - Director 约束 + Critic 反馈逐轮压缩搜索空间
   - Critic 建议初始化搜索先验
   - 本地评估模型替代 VLM API
   - 比 PhotoAgent 无约束 MCTS 更高效

4. Critic 训练方案：
   - 两层 Critic：CLIP+MLP 快速评分 + Qwen-VL 结构化评审
   - 合成数据 + 公开数据集预训练 → Pipeline 数据微调
   - 零额外标注成本

5. VlogEdit 评估基准（数据贡献）
```

### 消融实验

| 消融 | 对比 | 证明什么 |
|------|------|---------|
| 单 Agent vs 三 Agent | 当前架构 vs 新架构 | 多角色协作的优越性 |
| 数字闭环 vs 反思闭环 | score 比较 vs Critic 结构化反馈 | 反思让迭代更高效 |
| 无 Director vs 有 Director | 自由编辑 vs 风格约束 | Director 防止方向偏移 |
| 标准 MCTS vs 约束 MCTS | 无约束搜索 vs Director+Critic 约束 | 约束提升搜索效率 |
| 全量重规划 vs 增量修正 | 每轮从头来 vs 只改有问题的 | 增量更快更精准 |
| 外层 1 轮 vs 多轮 | 单次编辑 vs 迭代改进 | 外层循环的价值 |

### 回应 PhotoAgent 审稿人的质疑

| PhotoAgent 被批的问题 | VlogAgent 的回应 |
|---|---|
| "只是拼装现有组件" | 三 Agent 协作 + 反思路由 + 约束 MCTS 是新的算法范式 |
| "580 秒/张图太慢" | 本地 Critic + 约束搜索 + 增量修正 → 目标 5-10 分钟/视频 |
| "Sim-to-Real Gap" | 直接在原视频上搜索，无代理副本 |
| "评估依赖 VLM API" | 训练本地 Critic，两层架构 |
| "人类评估不足" | 从一开始设计用户研究（200+ 票 pairwise） |
| "与简单 baseline 比如何" | 消融实验覆盖 6 组对比 |

---

## 七、现有模块映射

```
当前模块                    → 新架构角色
────────────────────────────────────────────
Perceiver.observe()        → Director.strategize()
Perceiver.suggest()        → Editor.generate_candidates()
MCTSPlanner                → Editor 内部的约束 MCTS
Evaluator.score_aesthetic() → Critic.evaluate()（升级为结构化输出）
Evaluator.evaluate_baseline()→ 保留
agent.py 主循环             → 双层循环协调器
BasicTools / CompoundTools  → 不变
ToolSelector               → 不变
RunLogger                  → 升级（记录 Director/Editor/Critic 交互 + 路由决策）
CLIP+MLP（已训练）          → Critic Layer 1（MCTS 内部评估）
Qwen-VL-7B（待训练）       → Critic Layer 2（结构化评审）
```

---

## 八、实施路线

```
Phase 1（2-3 天）：重构为三 Agent 双层循环
  - Director：从 Perceiver.observe() 拆分出来
  - Editor：Perceiver.suggest() + 约束 MCTS
  - Critic：Evaluator 升级为结构化输出 + 路由决策
  - 双层循环协调器：外层 Director 驱动 + 内层 Editor-Critic 打磨
  - 增量编辑：Editor 只修改 Critic 指出的问题

Phase 2（1-2 天）：集成验证
  - 跑 5-10 个视频端到端验证
  - 对比旧架构 vs 新架构效果和效率
  - 收集消融实验数据

Phase 3（3-5 天）：Critic 训练（Phase 2b）
  - 数据准备 + 云 GPU GRPO 训练
  - 替代 Critic 的 VLM API 调用

Phase 4（3-5 天）：实验 + 论文素材
  - 消融实验（6 组对比）
  - 端到端测试（25-40 个视频）
  - Case study 可视化
  - 用户研究（10-15 人，200+ 票）
```

---

*本设计最后更新：2026-03-22*
