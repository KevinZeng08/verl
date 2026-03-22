# verl v0.7 Release Blog 总结

> 原文链接：https://verl.readthedocs.io/en/latest/blog/v0.7.html  
> 发布日期：2026 年 1 月 3 日  
> 作者：verl team

---

## 一、整体架构概览

verl 采用 **Hybrid-Controller 架构**（又称 HybridFlow），与 Google Pathways 等异步分片数据流系统共享设计理念。它将强化学习算法（PPO、GRPO、DAPO 等）建模为多阶段、多模型、可并行化的数据流图。

### 两种编程模型的统一

| 层级 | 模式 | 描述 |
|------|------|------|
| 高层编排 | 单控制器 (MPMD) | 单进程 `RLTrainer` 管理全局计算图，负责调度 rollout 生成、触发奖励评分、分发分布式训练任务 |
| 底层执行 | 多控制器 (SPMD) | Model Engine 运行在标准分布式训练模式，Workers 执行相同程序，通过集合通信同步 |

### 核心优势

- **灵活编排**：单控制器设计支持动态管理复杂约束，包括数据依赖、资源分配与模型布局、细粒度异步控制
- **抽象复杂性**：5D 并行（DP、TP、CP、PP、EP）等复杂并行策略封装在 Model Engine 内，用户只需关注 RL 算法逻辑
- **资源隔离与共享**：借助 Ray placement groups，提供 `ResourcePool` 和 `WorkerGroup` 抽象，实现 Actor、Critic、Reward、Rollout 等角色之间的 GPU 灵活共享

### 整体架构分层

```
verl-trainer
  └─ 构建 On-Policy / One-Step-Off-Policy / Fully Async 等 RL 训练流水线

verl-core（核心组件）
  ├─ Model Engine       # 模型训练引擎
  ├─ Rollout Engine     # 推理生成引擎
  ├─ Checkpoint Engine  # 权重同步引擎
  └─ TransferQueue      # 数据传输队列
```

---

## 二、verl-core 核心组件详解

### 2.1 Model Engine（模型引擎）

Model Engine 是 verl 的核心训练引擎，定义了一套支持可插拔后端的抽象接口，以 SPMD 模式运行。

**工作模式：**
- **SFT**：通过 `torchrun` 启动 Workers
- **RL**：通过 `WorkerGroup` API 由单控制器调用

**核心抽象接口：**

```python
class BaseEngine:
    def initialize(self): ...          # 实例化/加载模型、优化器、lr scheduler
    def optimizer_zero_grad(self): ... # 梯度清零
    def optimizer_step(self): ...      # 执行优化步骤
    def lr_scheduler_step(self): ...   # 推进学习率调度
    def forward_backward_batch(self, data, loss_function, forward_only=False): ...
    def get_per_tensor_param(self): ...
    def to(self, device, model, optimizer, grad): ...
```

**支持的后端对比：**

| 后端 | 并行支持 | 适用规模 | 支持模型 | 新模型支持周期 |
|------|----------|----------|----------|----------------|
| FSDP | FSDP + SP | 中等 Dense / 低端 MoE | 所有 Transformer 模型 | Day 0 |
| MCore | DP+TP+PP+EP+CP | 高 | Megatron-Bridge 支持列表 | 数周至数月 |
| VeOmni | FSDP+SP+EP | 中等 | VeOmni 支持列表 | ~1 周 |

> 新增 **VeOmni** 引擎（Alpha 状态）

---

### 2.2 Rollout Engine（推理引擎）

随着 LLM 强化学习从单轮静态任务演进到多轮动态 Agentic 任务，v0.7 **移除了 SPMD rollout 模式**，默认切换为 **Rollout Server 模式**。

**Server 模式优势：**
- LLM Server 以在线服务方式运行，支持 **dynamic batching**，大幅提升多轮对话吞吐
- 无需对推理引擎做侵入式修改，可无缝集成 vLLM、SGLang、TensorRT-LLM 等后端

**新增 AgentLoop 抽象：**

提供可扩展的 `AgentLoopBase`，用于定义自定义 Agentic 任务循环：

| 实现类 | 用途 |
|--------|------|
| `SingleTurnAgentLoop` | 标准单轮任务 |
| `ToolAgentLoop` | 经典 ReAct 多轮工具调用架构 |
| 自定义 | 如 SWEAgentLoop、GUIAgentLoop 等 |

---

### 2.3 TransferQueue（数据传输队列）

**背景问题：**  
原先 `RLTrainer` 同时负责控制流和数据流，成为瓶颈，在多模态训练（图片/视频/音频）或 router replay 等场景下尤为突出。使用 Ray object store 的早期尝试因缺乏 Tensor 优化而性能不佳。

**v0.7 解决方案：**

引入 **TransferQueue（实验性）**，将控制流与数据流解耦：

- `RLTrainer` 只负责分发指令和元数据
- TransferQueue 通过引用传递处理数据传输
- 专门针对 PyTorch Tensor 优化（支持 zero-copy 和 RDMA）
- 可扩展后端：ZeroMQ、NIXL、Ray RDT

> 计划在 **v0.8 成为默认传输方式**

---

### 2.4 Checkpoint Engine（权重同步引擎）

**背景问题：**  
随着 LLM 上下文长度增加和 Agentic 任务复杂化，rollout 的"长尾"问题日益突出。异步训练（解耦 Trainer 与 Rollouter）是解决方案，但引入了**跨节点参数高效同步**的挑战。

**设计：**

提供统一抽象层，用于各训练后端与推理后端之间的权重同步：

```python
class CheckpointEngine(ABC):
    async def send_weights(self, weights: Generator): ...
    async def receive_weights(self) -> Generator: ...
```

**支持的传输后端：**
- **NCCL**：集合通信广播
- **NIXL**：P2P 点对点通信

---

## 三、verl-trainer 训练流水线

基于 verl-core 的四大核心组件，verl-trainer 构建了三类 RL 训练流水线：

### 3.1 On-Policy（同步）

- **特点**：Rollout 与 Training 串行执行，通常共享 GPU（Colocate），严格遵守 on-policy 定义
- **适用场景**：基线实现，优先保证算法正确性而非训练吞吐

### 3.2 One-Step-Off-Policy（异步）

- **特点**：将当前训练步与下一批次的生成并行化，资源隔离，使用上一步参数进行 rollout
- **适用场景**：需要适度效率提升（**20%–40%**）同时保持接近严格 on-policy 稳定性

### 3.3 Fully Async（全异步，解耦与流式）

- **特点**：Trainer 与 Rollouter 完全解耦部署在独立节点，利用流式数据传输、staleness 控制和 partial rollout 最大化吞吐
- **适用场景**：大规模训练（**128+ GPUs**）或复杂推理任务（如长 CoT），生成延迟严重瓶颈时不可或缺

---

## 四、v0.7 详细发布内容

### Model Engine

- Megatron-Bridge 集成，支持 LoRA/PEFT（参见博文：[How We Build Trillion Parameter Reasoning RL with 10% GPUs](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus)）
- 支持 Megatron 后端的实验性 **fp8 训练**
- Megatron 后端新增模型支持：GPT-OSS、Qwen3-Next
- 新模型引擎全面支持（FSDP 和 Megatron 引擎达到**生产就绪**状态）：
  - 使用嵌套 Tensor 的 TensorDict 替代 padding DataProto 进行数据分发
  - 新增类 Tinker API 风格的 `TrainingWorker`
  - VLM 支持（SFT 和 RL Trainer）
  - 基于模型引擎的 Critic 模型
  - 实现 `ActorRolloutRefWorker`，支持单 Worker 内使用不同后端
- 新增 **VeOmni** 引擎（Alpha）

### Rollout Engine

- 移除 SPMD rollout 模式
- 支持 vllm 和 sglang 的 blockwise fp8 rollout；支持 vllm + torchao 的 online quant
- 实验性 router replay 支持（vllm）
- 优化多模态数据获取与预处理，**支持视频输入**
- 版本升级：`vllm==0.12.0`；`sglang==0.5.6`

### Reward（奖励）

- 支持混合奖励场景（生成式、判别式、规则式及其组合）
- 奖励模型重构为 **server 模式**，支持 Colocated 和 Standalone 两种部署方式
- 新增 Reward Manager：
  - **Limited mode**：请求速率控制
  - **Remote mode**：CPU 密集型任务

### Algorithm（算法）

- 新增 [**CISPO**](https://arxiv.org/pdf/2506.13585)：Clipped IS-weight Policy Optimization
- 新增 [**SAPO**](https://arxiv.org/abs/2511.20347)：Soft Adaptive Policy Optimization

### Recipe（配方/应用）

- **[NEW] VLA**：实验性 VLA 模型支持
- **[NEW] [rhymerl](https://arxiv.org/abs/2508.18588)**：利用历史经验加速 LLM 强化学习（History Rhymes: Accelerating LLM RL with RhymeRL）
- TransferQueue：支持多数据分区，优化 Tensor zero-copy 序列化
- One-step-off-policy / Fully async：通过 Checkpoint Engine 优化权重同步（bucket + pipeline 支持）

---

## 五、版本路线图

### v0.8 计划

| 模块 | 计划内容 |
|------|----------|
| Model Engine | 用 TensorDict 替换 DataProto（零 padding 传输）；切换默认到新模型引擎；VeOmni 引擎生产就绪；支持 MTP RL 训练；新模型：DeepSeek V3.2 等 |
| Rollout Engine | 新增 TensorRT-LLM rollout 引擎；vllm worker 与 trainer 进程分离，通过 CUDA IPC 更新权重 |
| TransferQueue | 合并入主分支；优化图像/视频 VLM 训练流水线；优化 router replay 传输 |
| Checkpoint Engine | 新增抽象接口；NCCL 和 NIXL 传输后端；更多传输后端 |

### v0.9 计划

| 模块 | 计划内容 |
|------|----------|
| Trainer | Full async 合并入主分支，基于 verl-core 组件重构 |
| Model Engine | 移除旧版模型引擎（`fsdp_workers.py`、`megatron_workers.py`）；支持 Omni 模型 RL 训练（Qwen3-Omni、BAGEL 等） |
| Rollout Engine | 新增 vllm-omni rollout 引擎 |
| Agentic Recipe | SWEAgent、GUIAgent 等 |

---

## 六、核心设计理念总结

1. **解耦与抽象**：verl-core 四大组件均提供清晰抽象接口，实现可插拔、可扩展
2. **渐进式异步**：从同步 → 一步异步 → 全异步，用户可按需选择效率与正确性的平衡点
3. **数据流优化**：TransferQueue 和 Checkpoint Engine 专门解决大规模 RL 训练中的数据传输瓶颈
4. **Agentic 原生支持**：AgentLoop 抽象使多轮 Agentic 任务成为一等公民
5. **多模态就绪**：从数据传输到推理引擎全面优化多模态（图片、视频）支持
