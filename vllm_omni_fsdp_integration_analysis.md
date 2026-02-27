# vLLM-Omni + FSDP Integration in VerL: Implementation Analysis

## Overview

This PR (#5297) introduces **vLLM-Omni** and **VeOmni Engine** integration into the VerL framework, enabling efficient RL training for large language models, vision-language models, and diffusion models. The architecture provides a unified way to combine different training backends with high-performance inference engines.

## Key Components

### 1. **VeOmni Engine** (`veomni/transformer_impl.py`)

The **VeOmniEngine** is a specialized training engine that extends the existing FSDP (Fully Sharded Data Parallel) infrastructure with VeOmni-specific optimizations.

#### Core Features:
- **Only supports FSDP2** (line 82): Enforces modern FSDP implementation for better performance
- **Device Mesh Initialization**: Creates parallel state with configurable:
  - Data Parallel (DP) size
  - Ulysses Sequence Parallel size
  - Expert Parallel (EP) size for MoE models
  - FSDP shard/replicate sizes (lines 101-108)

#### Integration with VeOmni Library:
```python
from veomni.distributed import parallel_state
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.auto import build_foundation_model
from veomni.optim import build_lr_scheduler, build_optimizer
```

#### Flow:
1. **Initialize parallel state** (line 101-108)
2. **Build foundation model** using VeOmni's model builder (line 197-204)
3. **Apply parallel strategies** (line 209-220):
   - Full sharding (ZeRO-3)
   - Mixed precision
   - Gradient checkpointing
   - FSDP offloading
4. **Build optimizer and scheduler** (line 225-227)
5. **Create activation offloading contexts** (line 235-239)

#### Special VL (Vision-Language) Support:
- `OmniSequenceShardCollator`: Handles sequence parallel slicing for vision inputs
- Supports padding for pixel values with custom scales
- Integration with Qwen-VL model types

### 2. **vLLM-Omni Rollout** (`vllm_omni_rollout.py`, `vllm_omni_async_server.py`)

The rollout system enables high-performance asynchronous inference using vLLM-Omni.

#### Architecture:
```
┌─────────────────────────────────────────────────────────────────┐
│                    vLLMOmniReplica                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              vLLMOmniHttpServer (per node)               │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │           AsyncOmni Engine                          │ │   │
│  │  │  ┌─────────────────────────────────────────────┐   │ │   │
│  │  │  │      vLLMOmniColocateWorkerExtension        │   │ │   │
│  │  │  └─────────────────────────────────────────────┘   │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Components:

**vLLMOmniServerAdapter** (`vllm_omni_rollout.py`):
- Acts as client to communicate with vLLM-Omni server via Ray actors
- Uses ZMQ for IPC communication
- Supports both IPC and Shared Memory for weight transfer (lines 74-84)
- Handles distributed execution via `collective_rpc` (line 113)

**vLLMOmniHttpServer** (`vllm_omni_async_server.py`):
- Launches vLLM-Omni HTTP server per node
- Configures:
  - Tensor/Pipeline/Expert Parallelism
  - Data Parallel for MoE models
  - LoRA support
  - CUDA graph capture
  - Sleep mode for efficient GPU memory management

#### Server Launch Flow:
1. **Configure Engine Args** (lines 182-241):
   - Model configuration
   - Parallel settings
   - LoRA settings
   - Compilation config with CUDA graph support
2. **Launch Server** (lines 338-367):
   - Uses `AsyncOmni` from vllm_omni
   - Supports custom pipelines for diffusion models

### 3. **Engine Workers** (`engine_workers.py`)

The `ActorRolloutRefWorker` class combines:
- **Actor**: Main policy model for training
- **Rollout**: Inference engine for sampling
- **Ref**: Reference model for KL divergence computation

#### Initialization Flow (line 482-599):
```python
def init_model(self):
    # 1. Build reference model (if needed)
    # 2. Build actor model with training config
    # 3. Build rollout engine
    # 4. Build checkpoint engine for weight updates
```

#### Weight Update Flow (`update_weights` method, lines 633-694):
```
┌─────────────────────────────────────┐
│  Update Weights from Actor          │
│  to Rollout Engine                  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  1. Resume rollout (wake up)        │
│     if free_cache_engine=True       │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  2. Get per-tensor params from      │
│     actor engine (FSDP/VeOmni)      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  3. Update rollout weights          │
│     via DTensor/HF loading          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  4. Offload actor model to CPU      │
│     to save GPU memory              │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  5. Resume KV cache                 │
└─────────────────────────────────────┘
```

### 4. **Configuration System**

#### VeOmniEngineConfig (`config/engine.py`, lines 227-310):
Key configurations for VeOmni backend:
- `fsdp_size`: FSDP group size
- `ulysses_parallel_size`: Ulysses sequence parallel
- `expert_parallel_size`: Expert parallel for MoE
- `init_device`: Where to initialize model ("cpu", "cuda", "meta", "npu")
- `enable_full_shard`: ZeRO-3 sharding
- `attn_implementation`: Flash attention variants with SP support
- `moe_implementation`: Fused MoE kernels
- `activation_gpu_limit`: For activation offloading

#### DiffusionRolloutConfig (`config/rollout.py`, lines 282-406):
Extended rollout config for diffusion models:
- `image_height/width`: Generated image size
- `num_inference_steps`: Diffusion steps
- `noise_level`: Noise scheduling
- `guidance_scale`: CFG guidance
- `sde_type`: Stochastic differential equation type

### 5. **Training Flow Integration**

The integration into the main training loop (in `main_ppo.py`):

```python
# 1. Select backend based on strategy
if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
    actor_rollout_cls = AsyncActorRolloutRefWorker
elif config.actor_rollout_ref.actor.strategy == "veomni":
    # Uses new TrainingWorker with VeOmniEngine
    from verl.workers.engine_workers import ActorRolloutRefWorker
```

#### Role Selection:
- **VeOmni strategy**: Uses `TrainingWorker` with `VeOmniEngine`
- **FSDP strategy**: Uses `AsyncActorRolloutRefWorker` or `ActorRolloutRefWorker`
- **Megatron strategy**: Uses Megatron-specific workers

### 6. **Model Offloading & Memory Management**

Both FSDP and VeOmni engines support:
- **Parameter offloading**: Keep params on CPU, load to GPU for compute
- **Optimizer offloading**: Keep optimizer states on CPU
- **Activation offloading**: Offload activations during backward pass

#### VeOmni Offloading (`veomni/utils.py`):
```python
def load_veomni_model_to_gpu(module):
    # Handles both FSDP1 and FSDP2 module loading

def offload_veomni_model_to_cpu(module):
    # Offload parameters to CPU

def offload_veomni_optimizer(optimizer):
    # Offload optimizer states
```

## Training Loop Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RayPPOTrainer / RayFlowGRPOTrainer              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Training Loop                                  │   │
│  │                                                                   │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐   │   │
│  │  │  ActorRollout   │    │     Critic      │    │ RewardModel │   │   │
│  │  │  RefWorker      │    │    Worker       │    │   Worker    │   │   │
│  │  └────────┬────────┘    └────────┬────────┘    └─────────────┘   │   │
│  │           │                      │                               │   │
│  │           ▼                      ▼                               │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                      │   │
│  │  │  TrainingWorker │    │  TrainingWorker │                      │   │
│  │  │                 │    │                 │                      │   │
│  │  │  ┌───────────┐  │    │  ┌───────────┐  │                      │   │
│  │  │  │VeOmniEngine│  │    │  │FSDPEngine │  │                      │   │
│  │  │  │  (Actor)  │  │    │  │ (Critic)  │  │                      │   │
│  │  │  └─────┬─────┘  │    │  └─────┬─────┘  │                      │   │
│  │  │        │        │    │        │        │                      │   │
│  │  └────────┼────────┘    └────────┼────────┘                      │   │
│  │           │                      │                               │   │
│  │           ▼                      ▼                               │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                      │   │
│  │  │  vLLMOmniServer │    │  vLLM Server    │                      │   │
│  │  │   (Rollout)     │    │   (Rollout)     │                      │   │
│  │  └─────────────────┘    └─────────────────┘                      │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Commit History Summary

The PR includes ~47 commits covering:

1. **Core Integration**:
   - `[veomni, trainer]` fixes for NPU patch scope
   - `[fsdp,veomni]` FSDPUlyssesShardingManager removal for reentrancy
   - `[vllm_omni]` CFG and code cleanup

2. **FSDP Improvements**:
   - `[fsdp,algo]` NVFP4 QAT support
   - `[fsdp, vllm]` NPU GRPO training scripts for Qwen3-VL
   - Fully async implementation via Engine Workers

3. **Rollout Enhancements**:
   - Async rollout mode as default (sync mode removed)
   - Multi-turn rollout support
   - Router replay for MoE models
   - Sleep mode for GPU memory efficiency

4. **Refactoring**:
   - Reward configuration refactoring
   - Actor strategy KL training support
   - Torchtitan as alternative training engine

## Usage Example

```yaml
# Configuration for VeOmni + vLLM-Omni
trainer:
  n_gpus_per_node: 8
  nnodes: 1

actor_rollout_ref:
  actor:
    strategy: veomni  # Use VeOmniEngine
    engine:
      fsdp_size: -1
      ulysses_parallel_size: 1
      expert_parallel_size: 1
      enable_full_shard: false
      mixed_precision: true
      attn_implementation: flash_attention_2

  rollout:
    name: vllm_omni  # Use vLLM-Omni for rollout
    mode: async
    tensor_model_parallel_size: 2
    data_parallel_size: 1
    enable_sleep_mode: true
    load_format: dummy  # Will be updated from actor weights
```

## Conclusion

This PR brings a powerful integration of:
1. **VeOmni**: A high-performance training backend with FSDP2, supporting advanced features like expert parallelism and specialized attention implementations
2. **vLLM-Omni**: An efficient inference engine for generation and diffusion tasks

The architecture maintains compatibility with existing VerL components while enabling:
- Scalable RL training on hundreds of GPUs
- Support for large vision-language models
- Efficient memory management through offloading and sleep modes
- Flexible backend selection (FSDP, Megatron, VeOmni, Torchtitan)
