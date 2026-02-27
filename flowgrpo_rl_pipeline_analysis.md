# FlowGRPO RL Pipeline Analysis

## Executive Summary

Based on the two entry scripts (`run_flowgrpo.sh` and `run_flowgrpo_async_reward.sh`), this document analyzes how VerL runs a complete RL pipeline for diffusion models (FlowGRPO). The key difference between the two scripts is:

- **run_flowgrpo.sh**: Uses **colocated reward model** (reward model shares GPUs with training through sleep/wake mechanism)
- **run_flowgrpo_async_reward.sh**: Uses **async reward server** (separate vLLM server for reward computation)

## Script Comparison

### Configuration Differences

| Aspect | Colocated (`run_flowgrpo.sh`) | Async Reward (`run_flowgrpo_async_reward.sh`) |
|--------|-------------------------------|-----------------------------------------------|
| Reward Model Enable | `reward.reward_model.enable=True` | `reward.reward_model.enable=False` |
| Reward Router Address | Auto-managed | `localhost:9529` (external server) |
| GPU Allocation | GPUs shared between training and reward | Separate GPU for reward server |
| Memory Management | Sleep/wake mechanism for rollout/reward | External server, no sleep needed |

### Key Configuration Snippets

**Colocated Mode:**
```bash
reward.reward_model.enable=True
reward.reward_model.rollout.name=vllm
```

**Async Reward Mode:**
```bash
# Launch reward server separately
CUDA_VISIBLE_DEVICES=0 vllm serve $reward_model_name --host localhost --port 9529

# Training uses external server
reward.reward_model.enable=False
+reward.reward_model.reward_router_address=$reward_router_address
```

## Complete RL Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FlowGRPO Pipeline Overview                                     │
│                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              RayDiffusionTrainer (Main Process)                           │   │
│  │                              ray_diffusion_trainer.py                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                                    │
│         ┌────────────────────┼────────────────────┐                                             │
│         │                    │                    │                                             │
│         ▼                    ▼                    ▼                                             │
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐                                  │
│  │  ActorRollout │   │   Critic (Opt)  │   │  Reward Loop    │                                  │
│  │  Ref Worker   │   │    Worker       │   │  Manager        │                                  │
│  └──────┬───────┘   └─────────────────┘   └────────┬────────┘                                  │
│         │                                           │                                           │
│         └───────────────────┬───────────────────────┘                                           │
│                             │                                                                   │
│         ┌───────────────────▼───────────────────┐                                               │
│         │      AgentLoopManager                  │                                               │
│         │  (Async Rollout Manager)               │                                               │
│         └───────────────┬───────────────────────┘                                               │
│                         │                                                                       │
│         ┌───────────────▼───────────────────────┐                                               │
│         │   CheckpointEngineManager             │                                               │
│         │   (Weight Sync & Sleep/Wake)          │                                               │
│         └───────────────────────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Pipeline Flow

### 1. Initialization Phase

```python
# 1. Initialize worker groups
self.actor_rollout_wg = RayWorkerGroup(...)  # Actor + Rollout + Ref

# 2. Create Reward Loop Manager (handles reward computation)
self.reward_loop_manager = RewardLoopManager(
    config=self.config,
    rm_resource_pool=resource_pool,  # None if using external reward server
)

# 3. Create Async Rollout Manager (AgentLoopManager)
self.async_rollout_manager = AgentLoopManager(
    config=self.config,
    worker_group=self.actor_rollout_wg,
    rollout_resource_pool=actor_rollout_resource_pool,
    reward_loop_worker_handles=reward_loop_worker_handles,
)

# 4. Create Checkpoint Manager (weight synchronization)
self.checkpoint_manager = CheckpointEngineManager(
    backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
    trainer=self.actor_rollout_wg,
    replicas=self.async_rollout_manager.rollout_replicas,
)

# 5. Sleep replicas to prepare for training
self.checkpoint_manager.sleep_replicas()
```

### 2. Training Loop (`fit` method)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            Training Loop (per batch)                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

Step 1: Generate Sequences
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   Input Batch (DataProto)                                                         │
│   ├── batch["input_ids"]: [bsz, prompt_length]                                   │
│   ├── batch["attention_mask"]: [bsz, prompt_length]                              │
│   └── non_tensor_batch["raw_prompt"], ["multi_modal_inputs"]                     │
│                                                                                  │
│                              │                                                   │
│                              ▼                                                   │
│   ┌──────────────────────────────────────────────────┐                          │
│   │  AgentLoopManager.generate_sequences()           │                          │
│   │                                                   │                          │
│   │  • Launch parallel agent loop workers            │                          │
│   │  • Each worker calls _run_agent_loop()           │                          │
│   │                                                   │                          │
│   │  Agent Loop Types:                                │                          │
│   │  - diffusion_single_turn_agent (for FlowGRPO)   │                          │
│   │  - single_turn_agent (for text generation)      │                          │
│   └──────────────────────────────────────────────────┘                          │
│                              │                                                   │
│                              ▼                                                   │
│   Output Batch (gen_batch_output)                                                 │
│   ├── batch["responses"]: [bsz*n, response_length]  # n=rollout.n               │
│   ├── batch["input_ids"]: [bsz*n, prompt_length + response_length]              │
│   └── batch["attention_mask"]: [bsz*n, seq_length]                               │
│                                                                                  │
│   Note: batch is repeated n times (rollout.n=16) for GRPO                        │
└──────────────────────────────────────────────────────────────────────────────────┘

Step 2: Sleep Rollout Replicas
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   checkpoint_manager.sleep_replicas()                                             │
│                                                                                  │
│   Frees GPU memory from rollout engine                                            │
│   - Weights moved to CPU/disk                                                     │
│   - KV cache cleared                                                              │
│   GPU now available for actor training                                            │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

Step 3: Compute Reward
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   COLOCATED MODE                          ASYNC REWARD MODE                      │
│   ════════════════                        ═══════════════════                    │
│                                                                                  │
│   Step 3a: Wake up reward model         Step 3a: Send HTTP request               │
│   reward_model_manager.wake_up()        to external vLLM server                  │
│                                               │                                   │
│   Step 3b: Compute RM score                     │                                │
│   RewardLoopManager.compute_rm_score()        ▼                                │
│                                           Get score from                        │
│                                           reward_loop_workers                   │
│   Step 3c: Sleep reward model                                                   │
│   reward_model_manager.sleep()                                                  │
│                                                                                  │
│   Output: rm_scores [bsz*n, 1] or [bsz*n, response_length]                      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

Step 4: Compute Log Probabilities
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                       │
│   old_log_prob = _compute_old_log_prob(batch)                                         │
│   ├── ActorRolloutRefWorker.compute_log_prob() via RPC                               │
│   └── Returns: old_log_probs [bsz*n, response_length]                                │
│                                                                                       │
│   ref_log_prob = _compute_ref_log_prob(batch)  (if using KL penalty)                 │
│   ├── ActorRolloutRefWorker.compute_ref_log_prob() via RPC                           │
│   └── Returns: ref_log_probs [bsz*n, response_length]                                │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘

Step 5: Compute Values (if using critic)
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                       │
│   values = _compute_values(batch)                                                     │
│   ├── CriticWorker.compute_values() via RPC                                          │
│   └── Returns: values [bsz*n, response_length]                                       │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘

Step 6: Compute Advantages
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                       │
│   # FlowGRPO specific advantage computation                                           │
│   compute_advantage(batch, adv_estimator="flow_grpo")                                 │
│                                                                                       │
│   Advantages computed per group (n samples from same prompt):                        │
│   advantages = (rewards - mean(rewards)) / std(rewards)                              │
│                                                                                       │
│   Output: batch["advantages"] [bsz*n, response_length]                               │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘

Step 7: Update Critic (if enabled)
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                       │
│   critic_output = _update_critic(batch)                                               │
│   ├── critic_output.value_loss                                                       │
│   └── critic_output.metrics                                                          │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘

Step 8: Update Actor (after warmup)
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                       │
│   actor_output = _update_actor(batch)                                                 │
│                                                                                       │
│   Internally:                                                                         │
│   1. ActorRolloutRefWorker.update_actor() via RPC                                    │
│   2. TrainingWorker.train_mini_batch()                                               │
│   3. VeOmniEngine.forward_backward_batch()                                           │
│   4. Apply PPO loss:                                                                 │
│      loss = -min(ratio * advantages, clipped_ratio * advantages)                     │
│                                                                                       │
│   Output: actor_output with training metrics                                         │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘

Step 9: Update Weights (Trainer → Rollout)
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                       │
│   checkpoint_manager.update_weights()                                                 │
│                                                                                       │
│   For "naive" backend (default):                                                     │
│   1. ActorRolloutRefWorker.update_weights()                                          │
│      ├── actor.engine.get_per_tensor_param()                                         │
│      ├── Load model to GPU, get DTensor weights                                      │
│      ├── rollout.update_weights()  (FSDP → vLLM)                                     │
│      └── actor.engine.to("cpu")  # Free GPU memory                                   │
│                                                                                       │
│   For "nccl/nixl/hccl" backend (disaggregated):                                      │
│   1. Build process group between trainer and rollout                                 │
│   2. Send weights via high-speed interconnect                                        │
│   3. Update rollout worker weights                                                   │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘

Step 10: Validation (periodic)
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                       │
│   _validate()                                                                         │
│   ├── Generate on validation set using val_kwargs                                     │
│   ├── Compute reward scores                                                           │
│   ├── Log sample generations                                                          │
│   └── Return validation metrics                                                       │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## Diffusion-Specific Components (FlowGRPO)

### 1. Diffusion Agent Loop

```python
# diffusion_single_turn_agent (configured in script)
actor_rollout_ref.rollout.agent.default_agent_loop=diffusion_single_turn_agent

# This agent loop:
# 1. Takes image/text prompt as input
# 2. Uses vLLM-Omni to generate images (diffusion model)
# 3. Returns image latents as responses
```

### 2. Custom Pipeline

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.custom_pipeline=verl.utils.vllm_omni.pipelines.QwenImagePipelineWithLogProb
```

This custom pipeline enables:
- Image generation from text prompts
- Log probability computation for RL training
- SDE (Stochastic Differential Equation) sampling

### 3. Diffusion Rollout Configuration

```python
actor_rollout_ref.rollout.name=vllm_omni      # Use vLLM-Omni engine
actor_rollout_ref.rollout.n=16                # 16 samples per prompt (GRPO)
actor_rollout_ref.rollout.guidance_scale=4.0  # CFG scale for diffusion
actor_rollout_ref.rollout.noise_level=1.0     # Noise level for generation
actor_rollout_ref.rollout.sde_window_size=2   # SDE window for sampling
```

### 4. Image Reward Manager

```python
reward.reward_manager.name=image  # Use ImageRewardManager
```

This handles:
- Scoring generated images against ground truth
- Computing perceptual similarity metrics
- Reward shaping for visual quality

## Memory Management Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GPU Memory Timeline                                  │
│                                                                                   │
│  Time    │    Action          │    GPU Memory Usage                              │
│  ────────┼────────────────────┼───────────────────────────────────────────────  │
│  t0      │    Init            │    Actor+Critic+Rollout loaded                  │
│  t1      │    Sleep Replicas  │    Only Actor weights in GPU                    │
│          │                    │    Rollout weights → CPU/Disk                   │
│          │                    │    KV cache → freed                              │
│  t2      │    Rollout Gen     │    Actor + Replicas (wake up)                   │
│          │                    │    → Generate sequences                         │
│  t3      │    Sleep Replicas  │    Free rollout memory for training             │
│  t4      │    Compute Reward  │    Colocated: Wake reward model                 │
│          │                    │    → Compute scores                             │
│          │                    │    → Sleep reward model                         │
│  t5      │    Actor Training  │    Actor training (backward pass)               │
│  t6      │    Update Weights  │    Actor → Rollout weight sync                  │
│  t7      │    Repeat          │    Back to t2                                   │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Two Modes: Detailed Comparison

### Mode 1: Colocated Reward Model (run_flowgrpo.sh)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Colocated Mode                                      │
│                                                                              │
│  GPU 0,1,2,3:                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    ActorRolloutRefWorker                              │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────────┐   │   │
│  │  │  Actor    │  │   Ref     │  │  Rollout  │  │  CheckpointMgr  │   │   │
│  │  │ (Train)   │  │ (Frozen)  │  │ (vLLM)    │  │                 │   │   │
│  │  └─────┬─────┘  └───────────┘  └─────┬─────┘  └─────────────────┘   │   │
│  │        │                              │                               │   │
│  │        └──────────────┬───────────────┘                               │   │
│  │                       │                                               │   │
│  │              ┌────────▼────────┐                                     │   │
│  │              │ Weight Synch    │ (sleep/wake cycle)                  │   │
│  │              └─────────────────┘                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  (Optional) Same GPUs:                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    RewardLoopManager                                  │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │  Reward Model   │ (vLLM) - Shares GPUs via sleep/wake            │   │
│  │  │  (Qwen3-VL)     │                                                │   │
│  │  └─────────────────┘                                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Sequence: Train → Sleep Rollout → Wake Reward → Compute → Sleep Reward →   │
│            Train → Update Weights → Wake Rollout → Repeat                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Mode 2: External Reward Server (run_flowgrpo_async_reward.sh)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Async Reward Mode                                   │
│                                                                              │
│  GPU 0 (Dedicated):                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Reward Server (Separate Process)                   │   │
│  │  vllm serve Qwen3-VL-8B --port 9529                                   │   │
│  │                                                                       │   │
│  │  • Always loaded in GPU                                               │   │
│  │  • HTTP API for scoring                                               │   │
│  │  • No sleep/wake overhead                                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  GPU 1,2,3:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    ActorRolloutRefWorker                              │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                       │   │
│  │  │  Actor    │  │   Ref     │  │  Rollout  │                       │   │
│  │  │ (Train)   │  │ (Frozen)  │  │ (vLLM)    │                       │   │
│  │  └─────┬─────┘  └───────────┘  └─────┬─────┘                       │   │
│  │        │                              │                              │   │
│  │        └──────────────┬───────────────┘                              │   │
│  │                       │                                              │   │
│  │              ┌────────▼────────┐                                    │   │
│  │              │ Weight Synch    │                                    │   │
│  │              └─────────────────┘                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Reward Computation:                                                         │
│  ┌──────────────┐      HTTP POST      ┌──────────────┐                      │
│  │ Reward Loop  │ ─────────────────▶  │ Reward Server│                      │
│  │ Workers      │  (images + prompt)  │ (GPU 0)      │                      │
│  └──────────────┘                     └──────────────┘                      │
│         │                                      │                              │
│         │◀────────── JSON scores ────────────│                              │
│         │                                      │                              │
│                                                                              │
│  Sequence: Train → Sleep Rollout → HTTP to Reward → Train → Update → Repeat │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Source Files

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| Main Entry | `verl/trainer/main_ppo.py` | `main()`, `run_ppo()` |
| Diffusion Trainer | `verl/trainer/ppo/ray_diffusion_trainer.py` | `RayFlowGRPOTrainer.fit()` |
| Standard Trainer | `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer.fit()` |
| Async Rollout | `verl/experimental/agent_loop/agent_loop.py` | `AgentLoopManager.generate_sequences()` |
| Reward Loop | `verl/experimental/reward_loop/reward_loop.py` | `RewardLoopManager.compute_rm_score()` |
| Checkpoint Engine | `verl/checkpoint_engine/base.py` | `CheckpointEngineManager.update_weights()` |
| Worker | `verl/workers/engine_workers.py` | `ActorRolloutRefWorker.update_weights()` |

## Performance Considerations

### Colocated Mode
- **Pros**: Fewer GPUs needed, lower latency for reward computation
- **Cons**: Sleep/wake overhead, GPU context switching cost

### Async Reward Mode
- **Pros**: No sleep/wake overhead, dedicated GPU for consistent reward serving
- **Cons**: Requires additional GPU, HTTP communication overhead

### When to Use Each
- **Colocated**: Resource-constrained environments, smaller reward models
- **Async Reward**: Large reward models, high-throughput requirements, stable latency needs
