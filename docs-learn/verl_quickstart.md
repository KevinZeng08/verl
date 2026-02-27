# veRL Quickstart 分析

## 一、启动流程

```
Ray 初始化
    ↓
TaskRunner 启动 (Hydra 配置解析)
    ↓
数据集创建 (train: 7473 samples, val: 1319 samples)
    ↓
Critic Model 初始化 (Qwen2ForTokenClassification + FSDP)
    ↓
Actor Model 初始化 (Qwen2ForCausalLM + FSDP)
    ↓
vLLM Rollout 引擎初始化 (CUDA Graph 捕获)
    ↓
AgentLoopWorker 启动
    ↓
权重同步 (Actor → vLLM)
    ↓
训练循环开始
```

## 二、每个 Step 流程

```
gen → reward → old_log_prob → values → adv → update_critic → update_actor → update_weights
```

## 三、关键 Metrics

### 训练效果指标 (最重要)

| Metric | 说明 | 健康状态 |
|--------|------|----------|
| `critic/score/mean` | 平均奖励分数 | 应逐步上升 |
| `critic/rewards/mean` | 平均 reward | 应逐步上升 |
| `actor/pg_loss` | Policy Gradient Loss | 应趋于稳定/下降 |
| `critic/vf_loss` | Value Function Loss | 应逐步下降 |
| `actor/entropy` | 策略熵 | 过低=收敛过度，过高=探索不足 |
| `actor/ppo_kl` | KL 散度 | 应为较小值 |

### 训练稳定性指标

| Metric | 说明 | 关注点 |
|--------|------|--------|
| `actor/pg_clipfrac` | PPO clip 比例 | 0.1-0.3 合理 |
| `critic/vf_clipfrac` | Value clip 比例 | 监控 critic 稳定性 |
| `critic/vf_explained_var` | Value 预测解释方差 | 应为正值接近 1 |
| `actor/grad_norm` | Actor 梯度范数 | 监控梯度爆炸 |
| `critic/grad_norm` | Critic 梯度范数 | 监控梯度爆炸 |

### 性能指标

| Metric | 说明 |
|--------|------|
| `timing_s/step` | 每 step 总耗时 |
| `timing_s/gen` | 生成耗时 (通常瓶颈) |
| `timing_s/update_critic` | Critic 更新耗时 |
| `timing_s/update_actor` | Actor 更新耗时 |
| `perf/throughput` | tokens/秒 吞吐量 |
| `perf/mfu/actor` | Actor MFU |
| `perf/mfu/critic` | Critic MFU |

### 序列长度指标

| Metric | 说明 |
|--------|------|
| `response_length/mean` | 平均响应长度 |
| `response_length/clip_ratio` | 达到 max_length 比例 |
| `prompt_length/mean` | 平均 prompt 长度 |

### 内存指标

| Metric | 说明 |
|--------|------|
| `perf/max_memory_allocated_gb` | GPU 最大分配内存 |
| `perf/max_memory_reserved_gb` | GPU 最大预留内存 |

## 四、常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `critic/score/mean` 一直很低 | reward function 设计不当 | 检查 reward function |
| `actor/entropy` 快速下降 | 策略收敛过度 | 增加 entropy coefficient |
| `actor/pg_clipfrac` 过高 (>0.5) | 更新步长过大 | 减小 learning rate 或 clip ratio |
| `timing_s/gen` 过长 | 生成瓶颈 | 减小 batch size 或 response length |
| `critic/vf_explained_var` 为负 | Critic 预测太差 | 增加 critic 训练迭代或调整 lr |

## 五、配置要点

### 模型路径配置

```bash
# 必须同时设置 path 和 tokenizer_path
actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
actor_rollout_ref.model.tokenizer_path=Qwen/Qwen2.5-0.5B-Instruct
critic.model.path=Qwen/Qwen2.5-0.5B-Instruct
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `data.train_batch_size` | 训练 batch size |
| `data.max_prompt_length` | 最大 prompt 长度 |
| `data.max_response_length` | 最大响应长度 |
| `actor_rollout_ref.actor.optim.lr` | Actor 学习率 |
| `critic.optim.lr` | Critic 学习率 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | PPO mini batch size |
| `algorithm.kl_ctrl.kl_coef` | KL 惩罚系数 |
| `trainer.total_epochs` | 总训练 epochs |