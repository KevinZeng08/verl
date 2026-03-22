# verl Code Walkthrough：高层运行逻辑与核心接口设计

> 基于 verl v0.7 架构设计，以 Qwen2.5-VL-7B GRPO 训练脚本为切入点
> 源码路径：`verl/`

---

## 目录

1. [全局视角：系统分层与运行入口](#一全局视角系统分层与运行入口)
2. [启动流程：从 Shell 脚本到 Ray Cluster](#二启动流程从-shell-脚本到-ray-cluster)
3. [核心数据结构：DataProto](#三核心数据结构dataproto)
4. [分布式抽象层：单控制器架构](#四分布式抽象层单控制器架构)
5. [Worker 层：角色与实现](#五worker-层角色与实现)
6. [Rollout Engine：AgentLoop 与 LLM Server](#六rollout-engineagentloop-与-llm-server)
7. [Checkpoint Engine：权重同步机制](#七checkpoint-engine权重同步机制)
8. [训练主循环：RayPPOTrainer.fit()](#八训练主循环rayppotrainerfit)
9. [完整数据流图](#九完整数据流图)
10. [关键接口速查表](#十关键接口速查表)
11. [算法-系统协同设计：GRPO 计算流程的系统映射](#十一算法-系统协同设计grpo-计算流程的系统映射)

---

## 一、全局视角：系统分层与运行入口

verl 的架构分为两层，上层 `verl-trainer` 调用下层 `verl-core` 的四大组件来构建 RL 训练流水线：

```
┌─────────────────────────────────────────────────────────┐
│                      verl-trainer                        │
│  RayPPOTrainer.fit() ← 单控制器，驱动整个计算图          │
└───────────────┬─────────────────────────────────────────┘
                │ 调用
┌───────────────▼─────────────────────────────────────────┐
│                       verl-core                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Model Engine │  │Rollout Engine│  │  Checkpoint  │  │
│  │(FSDP/MCore/  │  │(vLLM/SGLang/ │  │   Engine     │  │
│  │  VeOmni)     │  │  TRT-LLM)    │  │(NCCL/NIXL)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                   ┌──────────────────┐                  │
│                   │  TransferQueue   │                  │
│                   │  (实验性, v0.7)  │                  │
│                   └──────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

**关键文件**：

| 文件路径 | 作用 |
|----------|------|
| `verl/trainer/main_ppo.py` | 主入口，Ray 集群初始化，TaskRunner 启动 |
| `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer` 实现，单控制器主循环 |
| `verl/single_controller/ray/base.py` | `RayWorkerGroup`、`ResourcePoolManager` |
| `verl/workers/engine_workers.py` | `ActorRolloutRefWorker`（新引擎）、`TrainingWorker` |
| `verl/workers/fsdp_workers.py` | `ActorRolloutRefWorker`（旧引擎 legacy） |
| `verl/experimental/agent_loop/agent_loop.py` | `AgentLoopBase`、`AgentLoopManager` |
| `verl/checkpoint_engine/base.py` | `CheckpointEngine`、`CheckpointEngineManager` |
| `verl/protocol.py` | `DataProto`，系统统一数据容器 |

---

## 二、启动流程：从 Shell 脚本到 Ray Cluster

### 2.1 入口脚本（Qwen2.5-VL GRPO）

```bash
# examples/grpo_trainer/run_qwen2_5_vl-7b.sh
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.rollout.n=5 \        # 每个 prompt 采样 5 条 response
    actor_rollout_ref.rollout.name=vllm \  # rollout 后端使用 vLLM
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    ...
```

`python3 -m verl.trainer.main_ppo` 通过 `@hydra.main` 装饰器加载配置，直接进入 `main(config)` 函数。

### 2.2 main() → run_ppo() → TaskRunner.run()

```
main(config)
 └─ migrate_legacy_reward_impl(config)    # 迁移旧版 reward 配置
 └─ run_ppo(config)
     └─ ray.init(...)                     # 初始化 Ray 集群（本地或多节点）
     └─ task_runner_class = ray.remote(TaskRunner)
     └─ runner = task_runner_class.remote()
     └─ ray.get(runner.run.remote(config)) # 在 Ray Worker 上执行主逻辑
```

> **设计要点**：`TaskRunner.run()` 本身作为一个 Ray remote actor 运行（`num_cpus=1`），而非直接在 head node 上执行。这确保主逻辑运行在 worker 上，避免占用 head node 资源。

### 2.3 TaskRunner.run() 的职责

```python
# verl/trainer/main_ppo.py - TaskRunner.run()
def run(self, config):
    # 1. 注册各 Role 的 Worker 类
    actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
    self.add_critic_worker(config)               # (PPO 需要)
    self.add_reward_model_resource_pool(config)  # (使用 RM 时)
    self.add_ref_policy_worker(config, ...)      # (使用 KL 时)

    # 2. 加载 tokenizer 和 processor（VLM 需要 processor）
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path)         # Qwen2.5-VL 的图像处理器

    # 3. 构建 ResourcePoolManager
    resource_pool_manager = self.init_resource_pool_mgr(config)

    # 4. 构建 Dataset + Sampler
    train_dataset = create_rl_dataset(...)
    train_sampler = create_rl_sampler(config.data, train_dataset)

    # 5. 初始化 RayPPOTrainer 并开始训练
    trainer = RayPPOTrainer(...)
    trainer.init_workers()  # 创建所有 Ray Worker
    trainer.fit()           # 主训练循环
```

### 2.4 Worker 类注册策略

`TaskRunner` 维护两个字典，决定「谁来干活」和「在哪块 GPU 上」：

```python
self.role_worker_mapping: dict[Role, type[Worker]] = {
    Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    Role.Critic:       ray.remote(CriticWorker),
    Role.RefPolicy:    ray.remote(AsyncActorRolloutRefWorker),
}
self.mapping: dict[Role, str] = {
    Role.ActorRollout: "global_pool",
    Role.Critic:       "global_pool",
    Role.RefPolicy:    "global_pool",
}
```

Worker 实现的选择取决于 `use_legacy_worker_impl` 配置：

| `use_legacy_worker_impl` | Actor/Rollout Worker | 说明 |
|--------------------------|----------------------|------|
| `"disable"` | `engine_workers.ActorRolloutRefWorker` | 新引擎，生产就绪（v0.7） |
| `"auto"` / `"enable"` | `fsdp_workers.AsyncActorRolloutRefWorker` | 旧引擎（legacy） |

---

## 三、核心数据结构：DataProto

`DataProto` 是 verl 中组件间通信的**统一数据容器**，定义在 `verl/protocol.py`。

### 3.1 结构

```python
@dataclass
class DataProto:
    batch: TensorDict          # GPU Tensor 数据（input_ids, attention_mask, logprobs, ...）
    non_tensor_batch: dict     # 非 Tensor 数据（uid, data_source, reward_model ground_truth, ...）
    meta_info: dict            # 元信息（temperature, eos_token_id, global_steps, ...）
```

### 3.2 关键方法

| 方法 | 用途 |
|------|------|
| `DataProto.from_single_dict(batch_dict)` | 从 DataLoader 输出构造 DataProto |
| `a.union(b)` | 合并两个 DataProto（字段取并集），常用于将不同阶段的输出合入主 batch |
| `DataProto.concat(list)` | 沿 batch 维度拼接多个 DataProto |
| `batch.chunk(n)` | 将 batch 分成 n 份，分发给 n 个 agent loop worker |
| `batch.repeat(n, interleave=True)` | 将每条样本重复 n 次（GRPO 中 `rollout.n=5` 时使用） |
| `batch.select_idxs(indices)` | 按索引过滤样本 |
| `batch.pop(batch_keys, non_tensor_batch_keys)` | 弹出指定字段 |

### 3.3 在训练流中的流转

```
DataLoader → DataProto(prompts + non_tensor_batch)
                │
                ├─ .repeat(n=5)              # GRPO: 复制 5 份待生成
                │
                ▼
         AgentLoopManager.generate_sequences()
                │
                ▼
         DataProto(+ responses, attention_mask, rollout_log_probs)
                │
                ├─ .union(old_log_prob)      # 合入重计算的 log_prob
                ├─ .union(ref_log_prob)      # 合入参考策略 log_prob
                ├─ .union(values)            # 合入 Critic 估计的 values
                │
                ▼
         compute_advantage(batch)            # Driver 端计算 advantage（轻量级）
                │
                ▼
         update_critic(batch) → update_actor(batch)
```

---

## 四、分布式抽象层：单控制器架构

### 4.1 核心抽象三角

```
ResourcePool（GPU 资源池）
    └─ RayResourcePool（Ray 实现）
         └─ 通过 placement_group 管理 GPU 节点分配

WorkerGroup（Worker 组，SPMD 执行单元）
    └─ RayWorkerGroup（Ray 实现）
         └─ 持有 N 个 Ray Actor handles
         └─ 自动广播/收集 DataProto（依据 Dispatch 模式）

ResourcePoolManager（资源调度器）
    └─ 持有 resource_pool_spec + role→pool 映射
    └─ 负责创建 RayResourcePool 并检查 GPU 可用性
```

### 4.2 ResourcePoolManager

```python
# verl/single_controller/ray/base.py
class ResourcePoolManager:
    """管理资源池的创建与角色到资源池的映射。"""
    resource_pool_spec: dict[str, list[int]]  # pool_name → [gpus_per_node] * nnodes
    mapping: dict[Role, str]                  # Role → pool_name

# 在 TaskRunner.init_resource_pool_mgr() 中创建：
resource_pool_spec = {
    "global_pool": [8] * 1,     # 1 节点 × 8 GPU（本例）
    # "reward_pool": [4] * 2,   # 如果 RM 有独立资源池
}
resource_pool_manager = ResourcePoolManager(
    resource_pool_spec=resource_pool_spec,
    mapping={
        Role.ActorRollout: "global_pool",
        Role.Critic:       "global_pool",
    }
)
```

### 4.3 RayWorkerGroup 与 Dispatch 机制

`RayWorkerGroup` 封装了对分布式 Worker 的 RPC 调用。Worker 的方法通过 `@register(dispatch_mode=...)` 装饰器声明调用语义：

```python
# verl/single_controller/base/decorator.py
class Dispatch(Enum):
    ONE_TO_ALL         # 控制器广播 → 所有 Worker 执行相同输入
    DP_COMPUTE         # 沿 DP 维度切分输入 → 各 Worker 处理各自的分片
    MEGATRON_COMPUTE   # 按 Megatron DP 维度切分
    # ... 更多模式
```

以 `ActorRolloutRefWorker` 中的 `update_actor` 为例：

```python
# verl/workers/engine_workers.py
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def update_actor(self, data: TensorDict) -> TensorDict:
    """控制器调用时，data 自动按 DP 维度分发到各 GPU；返回值自动收集并合并。"""
    output = self.actor.train_mini_batch(data=data)
    return output.cpu() if output is not None else None
```

调用方（`RayPPOTrainer`）只需：
```python
actor_output = self.actor_rollout_wg.update_actor(batch)
# 底层自动：切分 batch → 广播到所有 GPU → 收集结果 → 合并
```

### 4.4 Colocated Worker 的合并优化

当 Actor、Ref、Rollout 共享同一个 ResourcePool 时，verl 使用 `create_colocated_worker_cls` 将多个角色合并到同一 Ray Actor 中，以减少进程间通信开销：

```python
# verl/trainer/ppo/ray_trainer.py - init_workers()
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = self.ray_worker_group_cls(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
    )
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    all_wg.update(spawn_wg)
```

---

## 五、Worker 层：角色与实现

### 5.1 Role 枚举

```python
# verl/trainer/ppo/utils.py
class Role(Enum):
    Actor          = 0
    Rollout        = 1
    ActorRollout   = 2   # Actor + Rollout 合并（最常用）
    Critic         = 3
    RefPolicy      = 4
    RewardModel    = 5
    ActorRolloutRef = 6  # Actor + Rollout + Ref 三合一（新引擎默认）
    Env            = 7
```

### 5.2 新引擎：ActorRolloutRefWorker（engine_workers.py）

v0.7 新引擎中，`ActorRolloutRefWorker` 是一个组合 Worker，内部聚合了三个子组件：

```python
# verl/workers/engine_workers.py
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    Actor + Rollout + Reference Policy 三合一 Worker
    - actor:   TrainingWorker（FSDP/MCore/VeOmni 训练引擎）
    - rollout: BaseRollout（vLLM/SGLang 推理引擎，Server 模式）
    - ref:     TrainingWorker（参考策略，LoRA 场景下与 actor 共享模型）
    """
    def init_model(self):
        # 1. 构建 Ref TrainingWorker（可选）
        if "ref" in self.role:
            self.ref = TrainingWorker(config=ref_training_config)

        # 2. 构建 Actor TrainingWorker
        if "actor" in self.role:
            self.actor = TrainingWorker(config=actor_training_config)
            self.actor.set_loss_fn(ppo_loss)  # 设置 PPO/GRPO loss

        # 3. 构建 Rollout Engine（vLLM/SGLang Server）
        if "rollout" in self.role:
            rollout_cls = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(config, model_config, device_mesh)

        # 4. 构建 CheckpointEngine（NCCL/NIXL）
        self.checkpoint_engine = CheckpointEngineRegistry.new(backend, ...)
```

**核心方法接口**：

```python
# --- 推理阶段（actor 计算 log_prob）---
@register(dispatch_mode=..., mesh_name="actor")
def compute_log_prob(self, data: TensorDict) -> TensorDict:
    return self.actor.infer_batch(data).cpu()

# --- 参考策略（ref 计算 log_prob）---
@register(dispatch_mode=..., mesh_name="ref")
def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
    return self.ref.infer_batch(data).cpu()

# --- 训练阶段（更新 actor 参数）---
@register(dispatch_mode=..., mesh_name="actor")
def update_actor(self, data: TensorDict) -> TensorDict:
    return self.actor.train_mini_batch(data).cpu()

# --- 权重同步（训练引擎 → rollout server）---
@register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
async def update_weights(self, global_steps: int = None):
    if backend == "naive":          # 同机 colocate：直接内存复制
        ...
    else:                           # 跨节点：通过 CheckpointEngine
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param)
```

### 5.3 TrainingWorker（新引擎核心）

`TrainingWorker` 是 v0.7 新引擎的核心，提供类似 Tinker 风格的 API：

```python
# verl/workers/engine_workers.py
class TrainingWorker(Worker, DistProfilerExtension):
    """
    封装 BaseEngine（FSDP/MCore/VeOmni），对外暴露：
    - reset()           → 调用 engine.initialize()
    - set_loss_fn(fn)   → 设置损失函数（可热更新）
    - infer_batch(data) → 前向推理（forward_only=True）
    - train_mini_batch(data) → 前向+反向+优化步骤
    - to(device, ...)   → 模型/优化器 load/offload
    - get_dispatch_collect() → 获取 DP mesh dispatch/collect 方法
    """
```

### 5.4 旧引擎（Legacy）：fsdp_workers.py

旧引擎中，`ActorRolloutRefWorker` 直接管理 FSDP 模型和 vLLM rollout，无 `TrainingWorker` 抽象层。关键方法包括 `init_model()`、`update_actor()`、`generate_sequences()`、`compute_log_prob()`。

> **v0.7 状态**：旧引擎仍可用，v0.8 计划标记为 deprecated，v0.9 计划移除。

---

## 六、Rollout Engine：AgentLoop 与 LLM Server

### 6.1 v0.7 的重大变化：SPMD → Server 模式

```
# v0.6 及之前（SPMD 模式）：
Worker 进程直接调用 vLLM generate()，静态 batch，无法动态调度

# v0.7（Server 模式）：
LLM Server（vLLM/SGLang）作为独立服务运行
AgentLoopWorker 通过 HTTP/OpenAI API 向 Server 发送请求
支持 dynamic batching，多轮对话，工具调用
```

### 6.2 AgentLoopBase 抽象接口

```python
# verl/experimental/agent_loop/agent_loop.py
class AgentLoopBase(ABC):
    """
    每个 sample 独立运行一个 agent loop，与 LLM server 交互。
    子类实现 run() 方法定义具体的交互逻辑。
    """
    def __init__(self, trainer_config, server_manager, tokenizer, processor,
                 dataset_cls, data_config):
        self.server_manager = server_manager  # 管理 LLM server 连接
        self.tokenizer = tokenizer
        self.processor = processor            # VLM 图像处理器
        ...

    @abstractmethod
    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        """
        核心接口：执行一次完整的 agent 交互循环
        输入：sampling_params（temperature, top_p等）+ dataset 字段（prompt, image等）
        输出：AgentLoopOutput（responses, response_mask, prompt_ids, ...）
        """
        raise NotImplementedError

    async def apply_chat_template(self, messages, tools=None, images=None, videos=None):
        """处理多模态输入（图片/视频），Qwen2.5-VL 场景下自动调用 processor"""
        ...

    async def process_vision_info(self, messages):
        """从 messages 中提取图片/视频，调用 processor 预处理"""
        ...
```

**内置实现**：

| 类 | 用途 | 源文件 |
|----|------|--------|
| `SingleTurnAgentLoop` | 标准单轮生成（默认） | `single_turn_agent_loop.py` |
| `ToolAgentLoop` | ReAct 多轮工具调用 | `tool_agent_loop.py` |
| `SWEAgentLoop` | 软件工程任务（社区贡献） | PR #4080 |

### 6.3 AgentLoopManager：批量调度

```python
# verl/experimental/agent_loop/agent_loop.py
class AgentLoopManager:
    """
    管理多个 AgentLoopWorker（Ray Actor），将 batch 分发到各 worker 并行处理。
    - hybrid 模式：rollout server 与训练引擎共享 GPU（当前默认）
    - standalone 模式：rollout server 使用独立 GPU（One-Step-Off / Fully Async）
    """

    @classmethod
    async def create(cls, config, worker_group, rollout_resource_pool, ...):
        instance = cls(...)
        await instance._initialize_llm_servers()   # 启动 vLLM/SGLang Server
        await instance._init_agent_loop_workers()  # 创建 AgentLoopWorker Ray Actors
        return instance

    async def _initialize_llm_servers(self):
        """
        计算 num_replicas = world_size // (TP * DP * PP)
        为每个 replica 创建 RolloutReplica 对象并初始化 LLM Server
        """
        for replica_rank in range(num_replicas):
            replica = RolloutReplica(replica_rank, ...)
        await asyncio.gather(*[server.init_hybrid(worker_group) for server in self.rollout_replicas])
        # init_hybrid: 在现有 Ray Actor 内部启动 vLLM/SGLang Server 进程

    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        1. 将 prompts 均分给 num_workers 个 AgentLoopWorker
        2. 并发执行各 worker 的 generate_sequences()
        3. concat 所有输出
        """
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(*[
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ])
        return DataProto.concat(outputs)
```

### 6.4 VLM（Qwen2.5-VL）的多模态处理

在 Qwen2.5-VL 场景下，`AgentLoopBase` 的 `apply_chat_template` 会自动处理图像输入：

```python
async def apply_chat_template(self, messages, images=None, videos=None, ...):
    if self.processor is not None:
        # Qwen2.5-VL: 使用 processor 处理图像，生成 vision tokens
        model_inputs = self.processor(
            text=[raw_prompt], images=images, videos=videos,
            return_tensors="pt"
        )
        prompt_ids = model_inputs.pop("input_ids")
    else:
        # 纯文本：使用 tokenizer
        prompt_ids = self.tokenizer.apply_chat_template(messages, ...)
```

---

## 七、Checkpoint Engine：权重同步机制

### 7.1 设计动机

在 colocate 模式下（训练和推理共享 GPU），推理阶段需要释放 GPU 内存给训练引擎使用：

```
训练步骤结束 → checkpoint_manager.sleep_replicas()  → rollout server 释放显存
rollout 阶段 → checkpoint_manager.update_weights()  → rollout server 加载最新权重，唤醒
```

在 disaggregated 模式（Fully Async）下，需要跨节点传输权重：

```
Trainer Node ──[NCCL/NIXL]──→ Rollout Node
  ME (FSDP)                      vLLM/SGLang
  CE (send)                      CE (receive) → cuda ipc → GPU内存
```

### 7.2 CheckpointEngine 抽象接口

```python
# verl/checkpoint_engine/base.py
class CheckpointEngine(ABC):
    """统一的权重传输抽象层"""

    @abstractmethod
    def prepare(self) -> dict:
        """
        每步同步前的准备：
        - 分配权重 bucket
        - [可选] RDMA 注册
        - 返回通信拓扑所需元数据（master ip:port 等）
        """

    @classmethod
    @abstractmethod
    def build_topology(cls, trainer_world_size, rollout_world_size, metadata):
        """
        构建 trainer 和 rollout worker 之间的通信拓扑
        返回 trainer_kwargs 和 rollout_kwargs（传给各 worker 的 init_process_group 参数）
        """

    @abstractmethod
    def init_process_group(self, **kwargs):
        """初始化通信进程组（NCCL communicator 或 NIXL channel）"""

    @abstractmethod
    async def send_weights(self, weights: Generator[tuple[str, Tensor], None, None]):
        """
        在 Trainer 侧调用：流式发送模型权重
        weights 是一个 generator，yield (param_name, tensor) 对
        支持 bucket 化传输（v0.7 新增，减少同步等待）
        """

    @abstractmethod
    async def receive_weights(self) -> Generator[tuple[str, Tensor], None, None]:
        """
        在 Rollout 侧调用：流式接收模型权重
        """
```

**支持的后端**：

| 后端 | 注册名 | 适用场景 |
|------|--------|---------|
| `ColocatedCheckpointEngine` | `"naive"` | 同机 colocate（直接内存引用，零拷贝） |
| `NCCLCheckpointEngine` | `"nccl"` | 多机 disaggregated，集合通信广播 |
| `NIXLCheckpointEngine` | `"nixl"` | 多机 disaggregated，P2P RDMA 点对点 |

### 7.3 CheckpointEngineManager

```python
# verl/checkpoint_engine/base.py
class CheckpointEngineManager:
    """
    协调 trainer（模型引擎侧）和多个 rollout replicas 之间的权重同步

    架构示意图：
    ┌────────────────────┐         ┌──────────────────────────┐
    │   Trainer Workers  │         │   Rollout Replicas       │
    │  ME0──CE──┐        │         │  CE──cuda ipc──vLLM GPU  │
    │  ME1──CE  ├─nccl/nixl──────► │  CE──cuda ipc──vLLM GPU  │
    │  ...      │        │         │  CE──cuda ipc──vLLM GPU  │
    └───────────┴────────┘         └──────────────────────────┘
    """

    async def sleep_replicas(self):
        """colocate 模式：所有 rollout replicas 释放显存（让出给训练引擎）"""
        if self.backend == "naive":
            await asyncio.gather(*[r.sleep() for r in self.replicas])

    async def update_weights(self, global_steps: int = None):
        """触发权重同步：trainer → rollout replicas"""
        if self.backend == "naive":
            # colocate: 直接在 worker 内部完成（调用 ActorRolloutRefWorker.update_weights）
            ray.get(self.trainer.update_weights(global_steps=global_steps))
        else:
            # disaggregated: 需要建立 NCCL/NIXL 通信组
            self.build_process_group(rollout)
            ray.get(
                self.trainer.update_weights(global_steps) +
                rollout.update_weights(global_steps)
            )
```

---

## 八、训练主循环：RayPPOTrainer.fit()

`fit()` 是 verl 的大脑，运行在单控制器（driver process）上，通过 RPC 调用分布式 Worker 来完成 RL 训练。

### 8.1 训练循环总览

```python
# verl/trainer/ppo/ray_trainer.py - RayPPOTrainer.fit()
def fit(self):
    self._load_checkpoint()
    self.checkpoint_manager.update_weights(0)  # 加载初始权重到 rollout

    for epoch in range(total_epochs):
        for batch_dict in self.train_dataloader:
            # ── Step 1: 数据准备 ─────────────────────────────────
            batch = DataProto.from_single_dict(batch_dict)
            batch.non_tensor_batch["uid"] = [uuid4() for ...]  # 为 GRPO advantage 分组用
            gen_batch = batch.repeat(n=rollout.n, interleave=True)  # GRPO: 每条 prompt 扩展 n 份

            # ── Step 2: Rollout 生成 ─────────────────────────────
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
            self.checkpoint_manager.sleep_replicas()  # 释放 rollout 显存

            # ── Step 3: 奖励计算 ─────────────────────────────────
            reward_tensor, reward_extra_infos = extract_reward(batch)
            # reward = rule-based + RM score（如配置了 RM）

            # ── Step 4: 重计算 log_prob（可选 bypass 模式）───────
            old_log_prob = self._compute_old_log_prob(batch)  # 当前策略 log_prob
            batch = batch.union(old_log_prob)

            # ── Step 5: 参考策略 log_prob（KL 惩罚时）───────────
            if self.use_reference_policy:
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

            # ── Step 6: Critic 估值（PPO 需要）──────────────────
            if self.use_critic:
                values = self._compute_values(batch)
                batch = batch.union(values)

            # ── Step 7: Advantage 计算（Driver 端，轻量级）──────
            batch.batch["token_level_scores"] = reward_tensor
            if use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl)  # KL 惩罚作为奖励的一部分
            batch = compute_advantage(batch, adv_estimator="grpo", ...)

            # ── Step 8: Critic 更新 ──────────────────────────────
            if self.use_critic:
                critic_output = self._update_critic(batch)

            # ── Step 9: Actor 更新 ──────────────────────────────
            actor_output = self._update_actor(batch)

            # ── Step 10: 权重同步（训练引擎 → rollout server）───
            self.checkpoint_manager.update_weights(self.global_steps)

            # ── Step 11: 日志、Checkpoint、Validation ──────────
            logger.log(metrics, step=self.global_steps)
```

### 8.2 GRPO 模式下的 Advantage 计算

```python
# verl/trainer/ppo/ray_trainer.py - compute_advantage()
elif adv_estimator == AdvantageEstimator.GRPO:
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=batch.batch["token_level_rewards"],  # shape: [B, T]
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["uid"],  # 用于分组：同一 prompt 的 n 条 response 为一组
        norm_adv_by_std_in_grpo=True,
    )
```

GRPO 的 advantage 计算完全在 driver 端完成，不需要 critic 模型，这也是 GRPO 相比 PPO 节省资源的原因。

### 8.3 关键 Worker 调用方法链

```python
# 各阶段 WorkerGroup 方法调用
self.async_rollout_manager.generate_sequences(gen_batch)   # Rollout: AgentLoopManager
self.actor_rollout_wg.compute_log_prob(batch)              # Actor: 前向计算 log_prob
self.ref_policy_wg.compute_ref_log_prob(batch)             # Ref: 前向计算参考 log_prob
self.critic_wg.compute_values(batch)                       # Critic: 估计 V(s)
self.critic_wg.update_critic(batch)                        # Critic: 更新参数
self.actor_rollout_wg.update_actor(batch)                  # Actor: 更新参数
self.checkpoint_manager.update_weights(global_steps)       # 同步权重到 rollout
```

### 8.4 init_workers() 流程

```python
def init_workers(self):
    # 1. 创建 Ray Placement Groups
    self.resource_pool_manager.create_resource_pool()

    # 2. 构建 colocated worker class（合并同一 pool 内的所有角色）
    for resource_pool, class_dict in self.resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict)
        wg_dict = RayWorkerGroup(resource_pool, worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())  # 按角色拆分 WG
        all_wg.update(spawn_wg)

    # 3. 初始化各角色 WG
    self.actor_rollout_wg = all_wg["actor_rollout"]
    self.actor_rollout_wg.init_model()  # 加载模型权重，启动 vLLM Server

    # 4. 创建 RewardLoopManager（处理 RM 评分）
    self.reward_loop_manager = RewardLoopManager(config, rm_resource_pool)

    # 5. 创建 AgentLoopManager（管理 rollout server 和 agent loop workers）
    self.async_rollout_manager = AgentLoopManager.create(
        config=config,
        worker_group=self.actor_rollout_wg,
        rollout_resource_pool=actor_rollout_resource_pool,
        reward_loop_worker_handles=...,
    )

    # 6. 创建 CheckpointEngineManager（管理权重同步）
    self.checkpoint_manager = CheckpointEngineManager(
        config=checkpoint_engine_config,
        trainer=self.actor_rollout_wg,
        replicas=self.async_rollout_manager.rollout_replicas,
    )

    # 7. 初始状态：sleep rollout（等待第一次权重加载）
    self.checkpoint_manager.sleep_replicas()
```

---

## 九、完整数据流图

以 Qwen2.5-VL GRPO 单训练步为例：

```
训练脚本 (run_qwen2_5_vl-7b.sh)
    │
    ▼
main_ppo.py → ray.init() → TaskRunner.run()
    │
    ▼
RayPPOTrainer.fit() [Driver Process, 单控制器]
    │
    ├─[每个 batch]
    │    │
    │    ▼ 1. 数据加载
    │   DataLoader → DataProto {
    │       batch: {input_ids, attention_mask, pixel_values(图像)}
    │       non_tensor_batch: {uid, reward_model:{ground_truth}, data_source}
    │   }
    │    │
    │    ▼ 2. .repeat(n=5) → 每条 prompt 扩展为 5 份
    │    │
    │    ▼ 3. AgentLoopManager.generate_sequences()
    │   ┌─────────────────────────────────────────────┐
    │   │  AgentLoopWorker[0..N] (Ray Actors, CPU)    │
    │   │    ├─ apply_chat_template(messages, images) │
    │   │    └─ HTTP → vLLM/SGLang Server             │
    │   │         → 生成 responses                    │
    │   │  输出 DataProto += {                        │
    │   │    responses, response_mask,                │
    │   │    rollout_log_probs, attention_mask         │
    │   │  }                                          │
    │   └─────────────────────────────────────────────┘
    │    │
    │    ▼ 4. sleep_replicas()（rollout 释放显存）
    │    │
    │    ▼ 5. extract_reward() → reward_tensor
    │   rule-based: 格式检查、答案匹配等
    │    │
    │    ▼ 6. actor_rollout_wg.compute_log_prob(batch)
    │   [FSDP Workers × 8 GPU, SPMD]
    │   DataProto += {old_log_probs, entropys}
    │    │
    │    ▼ 7. ref_policy_wg.compute_ref_log_prob(batch)
    │   DataProto += {ref_log_prob}
    │    │
    │    ▼ 8. apply_kl_penalty() [Driver 端]
    │   token_level_rewards = scores - β × KL(ref || current)
    │    │
    │    ▼ 9. compute_advantage(adv_estimator=GRPO) [Driver 端]
    │   按 uid 分组 → 组内归一化 → advantages
    │   DataProto += {advantages, returns}
    │    │
    │    ▼ 10. actor_rollout_wg.update_actor(batch)
    │   ┌──────────────────────────────────────────────────┐
    │   │  ActorRolloutRefWorker × 8 GPU                   │
    │   │  TrainingWorker.train_mini_batch():               │
    │   │    for mini_batch in batch.split(ppo_mini_bsz):  │
    │   │        loss = ppo_loss(mini_batch)                │
    │   │        loss.backward()                            │
    │   │        optimizer.step()                           │
    │   └──────────────────────────────────────────────────┘
    │    │
    │    ▼ 11. checkpoint_manager.update_weights()
    │   (naive backend) actor.engine → rollout server 内存
    │    │
    │    ▼ 12. 日志记录 + Validation（每 test_freq 步）
    │
    └─[下一个 batch]
```

---

## 十、关键接口速查表

### DataProto（数据容器）

| 方法 | 签名 | 用途 |
|------|------|------|
| `from_single_dict` | `cls(data: dict) → DataProto` | 从 DataLoader 构造 |
| `union` | `(other: DataProto) → DataProto` | 合并字段（不同阶段输出合并） |
| `concat` | `cls(list[DataProto]) → DataProto` | 沿 batch 轴拼接 |
| `repeat` | `(n: int, interleave: bool) → DataProto` | 每条样本重复 n 次 |
| `chunk` | `(n: int) → list[DataProto]` | 均分给多个 worker |
| `select_idxs` | `(idxs: list[int]) → DataProto` | 按索引过滤 |

### RayPPOTrainer（主控制器）

| 方法 | 职责 |
|------|------|
| `init_workers()` | 创建所有 Ray WorkerGroup，启动 LLM Server |
| `fit()` | 训练主循环 |
| `_compute_old_log_prob()` | 调用 actor 重计算 log_prob |
| `_compute_ref_log_prob()` | 调用 ref policy 计算参考 log_prob |
| `_compute_values()` | 调用 critic 计算 V(s) |
| `_update_actor()` | 调用 actor 更新参数 |
| `_update_critic()` | 调用 critic 更新参数 |
| `_validate()` | 验证集评估 |
| `_save_checkpoint()` | 保存 actor/critic checkpoint |

### ActorRolloutRefWorker（Worker 层，新引擎）

| 方法 | dispatch_mode | 职责 |
|------|--------------|------|
| `init_model()` | ONE_TO_ALL | 加载模型，启动 vLLM Server |
| `compute_log_prob()` | DP_COMPUTE (actor mesh) | 计算策略 log_prob |
| `compute_ref_log_prob()` | DP_COMPUTE (ref mesh) | 计算参考策略 log_prob |
| `update_actor()` | DP_COMPUTE (actor mesh) | 执行 PPO/GRPO 参数更新 |
| `update_weights()` | ONE_TO_ALL (async) | 同步权重到 rollout server |
| `save_checkpoint()` | ONE_TO_ALL | 保存模型权重 |
| `load_checkpoint()` | ONE_TO_ALL | 加载模型权重 |

### CheckpointEngine（权重同步）

| 方法 | 职责 |
|------|------|
| `prepare()` | 分配 bucket，返回通信元数据 |
| `build_topology()` | 构建 trainer-rollout 通信拓扑 |
| `init_process_group()` | 初始化 NCCL/NIXL 进程组 |
| `send_weights(generator)` | Trainer 侧：流式发送权重 |
| `receive_weights()` | Rollout 侧：流式接收权重 |

### AgentLoopBase（Rollout 逻辑）

| 方法 | 职责 |
|------|------|
| `run(sampling_params, **kwargs)` | 单条样本的完整 agent 交互循环（抽象方法） |
| `apply_chat_template(messages, ...)` | 多模态 chat template 处理 |
| `process_vision_info(messages)` | 提取并预处理图片/视频 |

---

---

## 十一、算法-系统协同设计：GRPO 计算流程的系统映射

> 以 **Qwen2.5-VL-7B GRPO** 为例，展示 RL 算法数学公式与 verl 系统执行之间的精确对应关系。
> 核心思想：**算法步骤通过 DataProto 这一"数据总线"流转，每一步的输入/输出字段与系统调用严格绑定。**

---

### 11.1 GRPO 算法回顾

GRPO（Group Relative Policy Optimization）是 DeepSeek-Math 提出的无 Critic PPO 变体，其核心流程如下：

**Step 1：对每条 prompt $q$ 采样 $G$ 条回复**

$$
\{o_1, o_2, \ldots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)
$$

**Step 2：对每条回复计算奖励 $r_i$（规则函数或 Reward Model）**

**Step 3：组内归一化得到优势估计**

$$
A_i = \frac{r_i - \operatorname{mean}(\{r_j\}_{j=1}^G)}{\operatorname{std}(\{r_j\}_{j=1}^G) + \epsilon}
$$

**Step 4：计算 PPO-Clip 策略梯度损失**

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} A_i,\ \operatorname{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon\right) A_i\right)\right]
$$

> **关于 KL 惩罚**：GRPO 原论文本身不包含 KL 惩罚项，KL 是 verl 工程实现中的可选正则化手段，且有两种完全不同的施加方式（详见 11.3 协同点 4），默认均不开启。

---

### 11.2 每个算法步骤的系统映射

下表将 GRPO 的每个数学步骤与 verl 的系统组件、DataProto 字段变化精确对应：

| 算法步骤 | 数学公式 | verl 系统调用 | DataProto 字段变化 | 执行位置 |
|---------|---------|--------------|-------------------|---------|
| **采样** | $o_i \sim \pi_{\theta_{\text{old}}}(q)$ | `async_rollout_manager.generate_sequences()` | `+= {responses, rollout_log_probs, response_mask}` | AgentLoopWorker（CPU Ray Actor）→ vLLM/SGLang Server（GPU） |
| **扩展 batch** | $q \to (q,q,\ldots,q) \times G$ | `batch.repeat(n=G, interleave=True)` | batch size: $B \to B \times G$ | Driver 进程（纯 Python） |
| **奖励计算** | $r_i = \text{RewardFn}(q, o_i)$ | `extract_reward(batch)` | `+= {token_level_scores}` | Driver 进程（调用规则函数/RM） |
| **重算 old log_prob** | $\log \pi_{\theta_{\text{old}}}(o_i \mid q)$ | `actor_rollout_wg.compute_log_prob(batch)` | `+= {old_log_probs, entropys}` | FSDP Training Workers（GPU，SPMD） |
| **参考策略 log_prob** | $\log \pi_{\text{ref}}(o_i \mid q)$ | `ref_policy_wg.compute_ref_log_prob(batch)` | `+= {ref_log_prob}` | FSDP Ref Workers（GPU，SPMD）仅 use_kl_in_reward=True 或 use_kl_loss=True 时 |
| **[可选] KL-in-reward** | $r_i' = r_i - \beta \cdot \text{KL}_{\text{token}}$ | `apply_kl_penalty(batch, kl_ctrl)` | `token_level_scores → token_level_rewards` | Driver 进程（CPU）；**发生在 Step 3 组内归一化之前** |
| **组内归一化** | $A_i = (r_i' - \mu_G)/\sigma_G$ | `compute_advantage(adv_estimator=GRPO)` | `+= {advantages, returns}` | Driver 进程（纯 Python，CPU） |
| **Actor 参数更新** | $\nabla_\theta \mathcal{L}_{\text{GRPO}}$ + [可选] KL Loss | `actor_rollout_wg.update_actor(batch)` | 内部参数更新，`meta_info += {metrics}` | FSDP Training Workers（GPU，SPMD） |
| **权重同步** | $\pi_{\theta_{\text{old}}} \leftarrow \pi_\theta$ | `checkpoint_manager.update_weights()` | rollout server 内存热更新 | CheckpointEngine（NCCL/NIXL） |

---

### 11.3 算法与系统的关键协同点

#### 协同点 1：采样扩展与 uid 分组——"组"概念的系统实现

GRPO 要求同一 prompt 的 $G$ 条回复能被识别为一组，以进行组内统计。这在 verl 中通过两步实现：

```python
# ray_trainer.py: fit() 内
# 步骤 A：为每条 prompt 分配唯一 uid
batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], ...)

# 步骤 B：batch 在 repeat 之前只有 B 条，repeat 后 uid 被复制 G 份
gen_batch_output = gen_batch.repeat(repeat_times=G, interleave=True)
# → batch 中每相邻 G 条样本的 uid 相同，天然构成"组"
```

```python
# ray_trainer.py: compute_advantage() 内
advantages, returns = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=data.batch["token_level_rewards"],  # shape: (B×G, seq_len)
    response_mask=data.batch["response_mask"],
    index=data.non_tensor_batch["uid"],                     # 用 uid 标识分组
)
```

```python
# core_algos.py: compute_grpo_outcome_advantage()
scores = token_level_rewards.sum(dim=-1)                   # (B×G,)：每条回复的总奖励
id2score = defaultdict(list)
for i in range(bsz):
    id2score[index[i]].append(scores[i])                   # 按 uid 聚合同组分数

for idx in id2score:
    scores_tensor = torch.stack(id2score[idx])
    id2mean[idx] = torch.mean(scores_tensor)               # μ_G
    id2std[idx]  = torch.std(scores_tensor)                # σ_G

for i in range(bsz):
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + ε)  # A_i

advantages = scores.unsqueeze(-1) * response_mask          # 广播到 token 维度
```

**协同设计要点**：GRPO 的"组"是纯数学概念，verl 用 `uid` 字段在 DataProto 中编码它，`repeat(interleave=True)` 保证同一 prompt 的 G 份扩展在 batch 中连续存放，`compute_grpo_outcome_advantage` 通过 `uid` 字典聚合完成统计。整个过程在 Driver 进程的 CPU 上完成，无需 GPU 通信。

---

#### 协同点 2：重算 old_log_prob——训练稳定性与系统 overhead 的权衡

GRPO 理论上用 rollout 时的策略参数 $\pi_{\theta_{\text{old}}}$ 计算重要性比值，但 verl 默认**重新计算** `old_log_probs`（而非直接使用 rollout 产生的 `rollout_log_probs`）：

```python
# ray_trainer.py: fit() 内——decoupled mode（默认）
# rollout 阶段：rollout server 产生 rollout_log_probs（可选记录）
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

# 训练阶段：actor FSDP workers 重算 old_log_probs 作为 PPO clip 的锚点
old_log_prob = self._compute_old_log_prob(batch)
# → batch += {old_log_probs}
```

这样设计的原因：rollout server（vLLM）与 training worker（FSDP）可能存在数值差异（如量化/精度），重算确保 $\pi_{\theta_{\text{old}}}$ 与 training engine 完全一致。

verl v0.7 也支持 **bypass mode**（设置 `algorithm.rollout_correction.bypass_mode=true`），直接用 `rollout_log_probs` 作为 `old_log_probs`，节省一次前向计算：

```python
# bypass mode：直接复用 rollout log_probs
if bypass_recomputing_logprobs:
    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
    apply_bypass_mode(batch, rollout_corr_config, policy_loss_config)
    # → batch["old_log_probs"] = batch["rollout_log_probs"]
```

---

#### 协同点 3：Advantage 计算在 Driver 端——轻量操作不下沉到 GPU

GRPO 的优势估计（组内均值/方差计算）是纯 CPU 操作：

```python
# ray_trainer.py: fit() 内
with marked_timer("adv", timing_raw, color="brown"):
    batch = compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,   # 枚举类型，避免字符串拼写错误
        ...
    )
    # → 在 Driver 进程（单 CPU）上完成，毫秒级
```

这与 PPO 的 GAE 形成对比——GAE 需要 Critic 估计的 V(s)，因此要等 `compute_values()` 完成后才能计算 advantage。GRPO **无需 Critic**，advantage 计算完全解耦，是 GRPO 在工程上的核心优势：

```
PPO:  rollout → compute_values (GPU) → GAE (CPU) → update_actor + update_critic
GRPO: rollout → extract_reward (CPU) → GRPO_adv (CPU) → update_actor only
              ↑ 省去了 Critic 前向/反向，减少约 1/3 的 GPU 计算
```

---

#### 协同点 4：两种 KL 惩罚——作用位置、时序与默认配置完全不同

GRPO **原论文本身不包含 KL 惩罚**，它只依赖组内归一化的奖励差异来提供稳定的梯度信号。verl 作为工程框架，额外提供了两种独立的 KL 正则化手段，它们在 `fit()` 循环中的作用位置、数学意义和配置项完全不同：

---

**机制 A：KL-in-Reward**（`algorithm.use_kl_in_reward`，默认 `False`）

在 **Advantage 计算之前**，将 KL 散度从 token 级别的奖励中扣除：

```python
# ray_trainer.py - fit() 内，adv 计算 timer 块中
if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(
        batch,
        kl_ctrl=self.kl_ctrl_in_reward,          # 自适应 KL 控制器
        kl_penalty=self.config.algorithm.kl_penalty
    )
    # 内部实现：
    # kld = KL(old_log_probs || ref_log_prob)  逐 token 计算
    # token_level_rewards = token_level_scores - β × kld
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

# 之后才进行 GRPO 组内归一化：
batch = compute_advantage(batch, adv_estimator=GRPO, ...)
# compute_grpo_outcome_advantage 读取的是 token_level_rewards（已含 KL 扣除）
```

**关键后果**：KL 扣除发生在组内归一化**之前**，因此 KL 惩罚会被一起归一化进 $A_i$，最终影响 policy gradient 的幅度，而非以独立项出现在 loss 里。

---

**机制 B：KL-in-Loss**（`actor_rollout_ref.actor.use_kl_loss`，默认 `False`）

在 **Actor 参数更新内部**，将 KL 散度作为独立的正则化损失叠加到策略梯度损失上：

```python
# dp_actor.py - update_policy() 内，mini-batch 循环中
if self.config.use_kl_loss:
    ref_log_prob = model_inputs["ref_log_prob"]
    kld = kl_penalty(log_prob, ref_log_prob, kl_penalty=self.config.kl_loss_type)
    kl_loss = agg_loss(kld, response_mask, loss_agg_mode)

    # 直接加到总 loss：
    policy_loss = pg_loss + kl_loss * self.config.kl_loss_coef
```

**关键后果**：KL 以独立损失项出现在反向传播的 loss 中，不影响 advantage 计算，相当于在 PPO-Clip objective 旁边加了一个 KL 正则项。

---

**两种机制对比**：

| 维度 | KL-in-Reward（机制 A） | KL-in-Loss（机制 B） |
|-----|----------------------|---------------------|
| 配置项 | `algorithm.use_kl_in_reward` | `actor.use_kl_loss` |
| 默认值 | `False` | `False` |
| 作用位置 | Advantage 计算**前**（Driver CPU） | 反向传播 loss 中（GPU） |
| 数学效果 | $r_i' = r_i - \beta \text{KL}$，被归一化进 $A_i$ | $\mathcal{L} = \mathcal{L}_{\text{pg}} + \lambda \cdot \text{KL}$，独立正则项 |
| 是否需要 ref_log_prob | 是（需要提前算好写入 batch） | 是（在 mini-batch 中实时读取） |
| 典型使用场景 | RLHF 早期训练，防止策略偏离过远 | 细粒度 KL 控制，与 pg_loss 解耦调参 |

**Qwen2.5-VL 示例脚本中的选择**：

```bash
# run_qwen2_5_vl-7b.sh
algorithm.use_kl_in_reward=False    # 机制 A 关闭：reward 不扣 KL，归一化的是纯 task reward

actor_rollout_ref.actor.use_kl_loss=True    # 机制 B 开启：KL 作为独立 loss 项
actor_rollout_ref.actor.kl_loss_coef=0.01   # KL loss 权重 λ=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl  # 低方差 KL 估计
```

这是 verl 官方推荐的 GRPO 配置：**组内归一化使用纯 task reward**（保证 advantage 的语义清晰），**KL 约束通过 loss 项独立施加**（与梯度方向解耦，更易调参）。

---

#### 协同点 5：多模态数据（Qwen2.5-VL）的透明传递

Qwen2.5-VL GRPO 相比文本 GRPO 的额外挑战是图像数据需要全程携带。verl 通过 DataProto 的 `non_tensor_batch` 字段透明传递多模态输入：

```python
# DataProto 字段布局（Qwen2.5-VL GRPO）
DataProto {
    batch: {
        input_ids:        (B×G, seq_len),       # 文本 token
        attention_mask:   (B×G, seq_len),
        pixel_values:     (num_patches, C, H, W),# 图像 patch（可变长）
        image_grid_thw:   (num_images, 3),       # 时序/高/宽信息
        old_log_probs:    (B×G, seq_len),        # 策略 log_prob
        advantages:       (B×G, seq_len),        # GRPO 优势
        token_level_rewards: (B×G, seq_len),     # 带 KL 惩罚的 reward
    },
    non_tensor_batch: {
        uid:             (B×G,),                 # 分组标识
        multi_modal_inputs: list[dict],           # 每条样本的图像元信息
        reward_model: {ground_truth: str},        # 规则 reward 需要的标准答案
    }
}
```

`AgentLoopBase.apply_chat_template()` 和 `process_vision_info()` 在 rollout 阶段将图像预处理好后写入 DataProto，后续所有 worker（actor、ref、critic）直接读取，无需重复预处理。

---

#### 协同点 6：向量化 GRPO（`grpo_vectorized`）——算法优化与系统实现协同

verl v0.7 提供了 `GRPO_VECTORIZED` 变体（`core_algos.py: compute_grpo_vectorized_outcome_advantage`），消除了原始 GRPO 中 Python 字典循环的性能瓶颈：

```python
# 原始 GRPO：Python 字典循环（O(B×G) Python 开销）
for i in range(bsz):
    id2score[index[i]].append(scores[i])
for idx in id2score:
    id2mean[idx] = torch.mean(torch.stack(id2score[idx]))

# 向量化 GRPO：全 Tensor 操作（消除 Python 循环）
g = as_torch_index(index, device=scores.device)          # uid → int 索引
mean_g, std_g, _ = group_mean_std(scores, g, eps=ε)      # scatter_reduce 实现
scalars = (scores - mean_g[g]) / (std_g[g] + ε)          # 全量 Tensor 广播
advantages = scalars.unsqueeze(-1) * response_mask
```

这体现了 verl 的一个设计哲学：**核心的轻量计算（advantage estimation）放在 Driver CPU 上，但通过 Tensor 向量化保证性能不成为瓶颈**。

---

### 11.4 完整的 GRPO 数据流与系统调用时序

```
Driver Process (CPU)                    GPU Workers
──────────────────────────────────────────────────────────────────────
1. batch = DataLoader 取一批 prompt
   batch.non_tensor_batch["uid"] = uuid()  # 每条 prompt 唯一 ID
   batch = batch.repeat(G, interleave=True) # 扩展为 B×G，uid 复制

2.                                      [AgentLoop Workers ← vLLM Server]
   generate_sequences(batch) ──────────► generate G responses per prompt
   ◄──── DataProto += {responses,          rollout_log_probs}

3.  sleep_replicas()                    # rollout server 释放显存

4.  extract_reward(batch)               # 规则函数打分
    token_level_scores = reward_fn(q, o) # shape: (B×G, seq_len)

5.                                      [ActorRollout FSDP Workers]
   compute_log_prob(batch) ────────────► forward pass on (q+o)
   ◄──── DataProto += {old_log_probs,       entropys}

6.                                      [Ref Policy FSDP Workers]  （仅 use_kl_in_reward 或 use_kl_loss=True 时）
   compute_ref_log_prob(batch) ─────────► forward pass
   ◄──── DataProto += {ref_log_prob}

7.  [可选，机制 A] apply_kl_penalty()   # CPU 计算，发生在组内归一化之前
    token_level_rewards = scores - β×KL_token
    ↑ 此时 KL 惩罚被混入 reward，后续归一化时会一起被 normalize
    若 use_kl_in_reward=False：
    token_level_rewards = token_level_scores  # 直接等于原始 reward

8.  compute_advantage(GRPO)             # CPU 纯 Python/Tensor
    scores = token_level_rewards.sum(-1) # (B×G,) 汇总每条回复的（已扣 KL 的）reward
    按 uid 分组：对每组 G 个 scores 计算 μ, σ
    A_i = (r_i' - μ) / (σ + ε)         # 标量优势（r_i' 是已扣 KL 的 reward）
    advantages = A_i * response_mask    # 广播到 token 维度 (B×G, seq_len)

9.                                      [ActorRollout FSDP Workers]
   update_actor(batch) ─────────────────► for mini_batch in batch.split():
                                              ratio = exp(log_π_θ - old_log_probs)
                                              pg_loss = -min(ratio×A, clip(ratio)×A)
                                              [可选，机制 B] kl_loss = KL(π_θ || π_ref) × λ
                                              loss = pg_loss + kl_loss  ← use_kl_loss=True 时
                                              loss.backward(); optimizer.step()
   ◄──── meta_info += {actor/loss, actor/kl_loss, ...}

10. update_weights()                    [CheckpointEngine: NCCL/NIXL]
    FSDP shards ──────────────────────► rollout server 内存热更新
    π_θ_old ← π_θ                       # 权重同步完成，准备下一步 rollout
```

---

### 11.5 算法配置与系统行为的映射关系

在 `run_qwen2_5_vl-7b.sh` 中的配置直接控制上述系统行为：

```bash
# 每条 prompt 采样 G=5 条回复 → batch.repeat(5)
actor_rollout_ref.rollout.n=5

# 使用 GRPO 优势估计 → compute_advantage(adv_estimator=GRPO)
algorithm.adv_estimator=grpo

# KL-in-Reward 关闭：reward 不扣 KL，组内归一化基于纯 task reward
algorithm.use_kl_in_reward=False

# KL-in-Loss 开启：KL 作为独立正则化 loss 叠加到 pg_loss 上
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01    # λ=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl  # 低方差 KL 估计

# PPO clip 范围 → compute_policy_loss(cliprange)
actor_rollout_ref.actor.clip_ratio_high=0.2

# mini-batch 大小 → batch.split(ppo_mini_batch_size)
actor_rollout_ref.actor.ppo_mini_batch_size=128

# 不使用 Critic → use_critic=False，跳过 compute_values/update_critic
trainer.critic_warmup=0   # critic_warmup=0 表示跳过 critic 热身（GRPO 无 critic）
```

---

### 11.6 设计原则总结：为什么这样的映射是最优的

| 设计选择 | 系统实现 | 背后原因 |
|---------|---------|---------|
| **Advantage 在 Driver 端计算** | `compute_advantage()` 在 CPU Python 进程中运行 | 组内统计是轻量操作（O(B×G)），避免额外的 GPU kernel launch 开销 |
| **uid 作为分组键** | `DataProto.non_tensor_batch["uid"]` 贯穿全流程 | 解耦"哪些样本属于同一组"与"batch 内的物理位置"，支持 balance_batch 乱序 |
| **DataProto 作为数据总线** | 每个算法步骤只读/写特定字段，其余字段原样传递 | 算法步骤解耦，任意步骤可替换为新算法（只需保证字段契约） |
| **两种 KL 机制独立可配置** | `use_kl_in_reward`（reward 层） vs `use_kl_loss`（loss 层），默认均关闭 | 前者影响 advantage 归一化的基底，后者是独立正则项；解耦使调参更精细 |
| **GRPO 默认不加 KL** | 两个 KL 配置项默认均为 `False` | GRPO 组内归一化本身已提供足够稳定的梯度信号，KL 是可选的额外约束 |
| **重算 old_log_prob（默认）** | Decoupled mode：训练 engine 重算，非 rollout 产出 | 消除 vLLM/FSDP 数值不一致，保证 PPO ratio 计算精确 |
| **bypass_mode 可选** | `rollout_log_probs` 直接用作 `old_log_probs` | 追求极致吞吐时可接受轻微数值差异，节省一次完整前向 |
| **无 Critic 设计** | GRPO 跳过 `compute_values` 和 `update_critic` | 比 PPO 省约 1/3 GPU 计算，尤其对超大模型意义显著 |

---

## 附录：核心概念关系图

```
                    ┌──────────────────────────────────────────┐
                    │           RayPPOTrainer（单控制器）        │
                    │  .actor_rollout_wg  (RayWorkerGroup)     │
                    │  .critic_wg         (RayWorkerGroup)     │
                    │  .ref_policy_wg     (RayWorkerGroup)     │
                    │  .async_rollout_manager (AgentLoopMgr)   │
                    │  .checkpoint_manager  (CkptEngineMgr)    │
                    └────────────────┬─────────────────────────┘
                                     │ RPC (ray.get / async)
              ┌──────────────────────┼───────────────────────┐
              ▼                      ▼                       ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │  RayWorkerGroup  │   │ AgentLoopManager │   │CkptEngineMgr     │
   │ (actor_rollout)  │   │                  │   │                  │
   │ ┌────────────┐   │   │ AgentLoopWorkers │   │  trainer WG      │
   │ │ActorRollout│   │   │ (Ray Actors)     │   │  rollout replicas│
   │ │RefWorker×8 │   │   │    │             │   └──────────────────┘
   │ │(GPU×8)     │   │   │    ▼             │
   │ │ ┌────────┐ │   │   │ vLLM/SGLang     │
   │ │ │Training│ │   │   │ Server(s)        │
   │ │ │Worker  │ │   └──────────────────────┘
   │ │ │(actor) │ │
   │ │ └────────┘ │
   │ │ ┌────────┐ │
   │ │ │Training│ │
   │ │ │Worker  │ │
   │ │ │(ref)   │ │
   │ │ └────────┘ │
   │ └────────────┘ │
   └────────────────┘
```
