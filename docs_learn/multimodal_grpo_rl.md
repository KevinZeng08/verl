# Multimodal GRPO RL 高层解读

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Hybrid Engine 架构                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────┐                                 │
│                         │   TaskRunner    │                                 │
│                         │   (主控制器)     │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│          ┌───────────────────────┼───────────────────────┐                  │
│          ▼                       ▼                       ▼                  │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│   │    Actor    │         │ Reference   │         │   Rollout   │          │
│   │   (可训练)   │◄───────►│   (冻结)    │◄───────►│   Engine    │          │
│   │   FSDP      │   KL    │   FSDP      │  权重同步 │  SGLang/    │          │
│   │             │   Loss  │  (offload)  │          │   vLLM      │          │
│   └─────────────┘         └─────────────┘         └─────────────┘          │
│          │                                                 │               │
│          │               GRPO 不需要 Critic                 │               │
│          │              (用 group 内相对 reward)             │               │
│          │                                                 │               │
│          └─────────────────┬───────────────────────────────┘               │
│                            ▼                                                │
│                    ┌─────────────┐                                         │
│                    │   Reward    │                                         │
│                    │  Function   │                                         │
│                    │ (规则/模型)  │                                         │
│                    └─────────────┘                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键组件说明**：

| 组件 | 作用 | Multimodal 特殊处理 |
|------|------|---------------------|
| **Actor** | 可训练的策略模型，输出响应 | Vision Encoder 参与梯度计算 |
| **Reference** | 冻结的参考模型，计算 KL 散度 | 同 Actor 结构，权重冻结 |
| **Rollout Engine** | 高效推理引擎，生成响应 | 支持 image + text 输入 |
| **Reward Function** | 评估响应质量 | 可基于图像-文本一致性 |

---

## 二、算法核心：GRPO vs PPO

### 2.1 PPO 传统流程

```
┌──────────────────────────────────────────────────────────────┐
│                        PPO 流程                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Prompt x ──► Actor π_θ ──► Response y ──► Reward r(x,y)   │
│                    │                          │              │
│                    ▼                          ▼              │
│              log π_θ(y|x)              Value V_φ(x,y)       │
│                    │                          │              │
│                    └──────────┬───────────────┘              │
│                               ▼                              │
│                    Advantage = r + γV - V                    │
│                               │                              │
│                               ▼                              │
│                    PPO Loss + Value Loss                     │
│                    (需要同时训练 Actor 和 Critic)            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**PPO 的问题**：
- 需要额外的 Critic 模型，显存开销大
- Critic 训练不稳定，需要 carefully tuned value function

### 2.2 GRPO 改进

```
┌──────────────────────────────────────────────────────────────┐
│                        GRPO 流程                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Prompt x ──► Actor π_θ ──► n个响应 {y_1, ..., y_n}        │
│                              │                               │
│                              ▼                               │
│                    Rewards {r_1, ..., r_n}                  │
│                              │                               │
│                              ▼                               │
│           Group 相对 Advantage (无需 Critic!)               │
│                                                              │
│           A_i = (r_i - mean(r)) / std(r)                    │
│                                                              │
│                              │                               │
│                              ▼                               │
│                    PPO Loss (仅 Actor)                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**GRPO 核心思想**：
- 用同一 prompt 生成的多个响应之间的**相对 reward** 来替代 Critic 估计的 value
- Group 内归一化：`A_i = (r_i - mean(r)) / std(r)`
- 自动适应不同难度的问题

**GRPO 优势**：
1. 无需 Critic：节省显存，减少模型数量
2. 训练稳定：避免了 Critic 训练的不稳定性
3. 简单高效：算法更简洁，超参数更少

---

## 三、Multimodal 的特殊处理

### 3.1 数据流对比

**Text-only RL**：
```
Text Prompt ──► Tokenizer ──► Text Embedding ──► LLM ──► Response
```

**Multimodal RL**：
```
┌─────────────────────────────────────────────────────────────────┐
│                    Multimodal 数据流                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image ──► Vision Encoder ──┐                                  │
│                              │                                  │
│                              ├──► Cross-Attention ──► LLM ──► Response
│                              │                                  │
│  Text   ──► Text Embedding ─┘                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Token 序列差异

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Text Token:  [T1] [T2] [T3] [T4] [T5] ...                      │
│                                                                 │
│  Multimodal Token:                                              │
│  [T1] [IMG_TOK_1] [IMG_TOK_2] ... [IMG_TOK_K] [T2] [T3] ...    │
│         └──────────────────────────────────┘                    │
│                    图像 token 序列 (数量动态变化)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Multimodal 带来的挑战**：
| 挑战 | 原因 | 解决方案 |
|------|------|---------|
| 显存占用大 | Vision Encoder + LLM 双重开销 | Gradient Checkpointing, FSDP |
| 序列长度动态 | 图像 token 数量不固定 | Remove Padding, 动态 batching |
| 训练不稳定 | Vision Encoder 梯度可能爆炸 | LayerNorm, gradient clipping |
| 生成慢 | 多模态 token 更长更复杂 | SGLang/vLLM 高效推理引擎 |
| Reward 设计难 | 需要考虑图像-文本一致性 | 任务相关的 reward 函数 |

### 3.3 Vision Encoder 的训练策略

**关键配置参数**：
```yaml
actor_rollout_ref.actor.freeze_vision_tower: false  # 默认值
```

**代码实现** (`verl/workers/fsdp_workers.py:529-538`)：
```python
if self.config.actor.get("freeze_vision_tower", False):
    vision_tower = get_vl_model_vision_tower(actor_module)
    if vision_tower is not None:
        vision_tower.requires_grad_(False)
        print("[actor model] Vision tower is set to not trainable.")
```

**两种策略对比**：

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **ViT 训练** (默认) | 模型能学习到视觉特征与任务的关联 | 显存开销大，需要更稳定训练 | 视觉理解任务，如 VQA, OCR 等 |
| **ViT 冻结** | 节省显存，训练更稳定 | 可能限制视觉能力提升 | 视觉特征已足够好的任务 |

**Qwen2.5-VL 默认行为**：ViT 是**训练的**（`freeze_vision_tower: false`）

---

## 四、Reward 计算

### 4.1 Multimodal Reward 类型

```
┌─────────────────────────────────────────────────────────────────┐
│                    Reward 计算方式                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 规则-based Reward（本例使用）                               │
│     ┌──────────────────────────────────────────────┐           │
│     │  问题: 根据图像解答几何题                     │           │
│     │  响应: "答案是 45°"                          │           │
│     │  真实答案: 45°                               │           │
│     │  Reward: 1.0 (正确) 或 0.0 (错误)            │           │
│     └──────────────────────────────────────────────┘           │
│                                                                 │
│  2. 模型-based Reward（可选）                                   │
│     ┌──────────────────────────────────────────────┐           │
│     │  需要额外的 Reward Model                     │           │
│     │  输入: (Image, Prompt, Response)             │           │
│     │  输出: Scalar Reward                         │           │
│     └──────────────────────────────────────────────┘           │
│                                                                 │
│  3. Vision Reward（如图像描述质量）                             │
│     ┌──────────────────────────────────────────────┐           │
│     │  CLIP Score / CLIP Similarity                │           │
│     │  评估生成文本与图像的匹配度                   │           │
│     └──────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 配置方式

```yaml
reward:
  custom_reward_function:
    name: compute_score    # 使用自定义 reward 函数
    path: null             # 默认路径或自定义模块路径
```

---

## 五、训练流程单步详解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Single Training Step                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SAMPLE BATCH                                                           │
│     ┌─────────────────────────────────────────────┐                        │
│     │  Batch = {(img_1, prompt_1), ..., (img_B, prompt_B)}                │
│     │  B = train_batch_size / n = 512 / 5 ≈ 102 prompts                  │
│     └─────────────────────────────────────────────┘                        │
│                                                                             │
│  2. ROLLOUT (生成)                                                         │
│     ┌─────────────────────────────────────────────┐                        │
│     │  For each (img, prompt):                    │                        │
│     │      Generate n=5 responses                 │                        │
│     │      → {(img, prompt, y_1), ..., (img, prompt, y_5)}               │
│     └─────────────────────────────────────────────┘                        │
│                                                                             │
│  3. REWARD COMPUTATION                                                     │
│     ┌─────────────────────────────────────────────┐                        │
│     │  For each response:                         │                        │
│     │      r_i = reward_fn(img, prompt, y_i)      │                        │
│     │  → Group rewards: [r_1, ..., r_5]           │                        │
│     └─────────────────────────────────────────────┘                        │
│                                                                             │
│  4. GRPO ADVANTAGE                                                         │
│     ┌─────────────────────────────────────────────┐                        │
│     │  For each group (same prompt):              │                        │
│     │      mean_r = mean([r_1, ..., r_5])         │                        │
│     │      std_r = std([r_1, ..., r_5])           │                        │
│     │      A_i = (r_i - mean_r) / std_r           │                        │
│     └─────────────────────────────────────────────┘                        │
│                                                                             │
│  5. LOG PROBABILITY                                                        │
│     ┌─────────────────────────────────────────────┐                        │
│     │  Actor:      log π_θ(y|img, prompt)         │  (需要梯度)            │
│     │  Old Policy: log π_old(y|img, prompt)       │  (固定，用于 PPO)      │
│     │  Reference:  log π_ref(y|img, prompt)       │  (固定，用于 KL)       │
│     └─────────────────────────────────────────────┘                        │
│                                                                             │
│  6. PPO LOSS + KL LOSS                                                     │
│     ┌─────────────────────────────────────────────┐                        │
│     │  ratio = exp(log_π_θ - log_π_old)           │                        │
│     │  pg_loss = -min(ratio*A, clip(ratio)*A)     │                        │
│     │  kl_loss = KL(π_θ || π_ref)                 │                        │
│     │  loss = pg_loss + 0.01 * kl_loss            │                        │
│     └─────────────────────────────────────────────┘                        │
│                                                                             │
│  7. BACKPROP & UPDATE                                                      │
│     ┌─────────────────────────────────────────────┐                        │
│     │  Update Actor (包括 Vision Encoder)         │                        │
│     │  Sync weights to Rollout Engine             │                        │
│     └─────────────────────────────────────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 六、关键配置参数

```yaml
# 算法配置
algorithm.adv_estimator=grpo                    # 使用 GRPO
actor_rollout_ref.actor.use_kl_loss=True       # 启用 KL 散度约束
actor_rollout_ref.actor.kl_loss_coef=0.01      # KL 损失系数

# Multimodal 配置
data.image_key=images                          # 图像输入字段
actor_rollout_ref.model.use_remove_padding=True # 优化 padding 计算

# Vision Encoder 配置
actor_rollout_ref.actor.freeze_vision_tower=false  # ViT 是否冻结 (默认训练)

# 生成配置
actor_rollout_ref.rollout.n=5                  # 每个 prompt 生成 5 个响应

# 显存优化
actor_rollout_ref.model.enable_gradient_checkpointing=True  # 节省显存
actor_rollout_ref.ref.fsdp_config.param_offload=True       # Reference 模型 offload
```

---

## 七、性能指标参考

### 单步训练时间分解（Qwen2.5-VL-7B, 8 GPU）

| 阶段 | 时间 | 比例 |
|------|------|-----|
| Rollout (生成) | ~23s | 23% |
| Old LogProb | ~12s | 12% |
| Reference LogProb | ~18s | 18% |
| Actor Update | ~41s | 41% |
| Weight Sync | ~6s | 6% |
| **Total** | **~100s** | 100% |

### 吞吐量参考

- Throughput: ~2000 tokens/s
- Avg response length: ~368 tokens
- Memory: ~32-45 GB GPU memory