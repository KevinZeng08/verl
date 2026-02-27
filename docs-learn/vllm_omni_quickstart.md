# vLLM-Omni + VeOmni Engine Quick Start Guide

## Prerequisites

### Dependencies
```bash
# Core dependencies
pip install veomni>=0.1.0  # VeOmni training library
pip install vllm-omni>=0.1.0  # vLLM-Omni inference engine
pip install ray  # Distributed computing
pip install tensordict  # Tensor operations
pip install hydra-core  # Configuration management

# Optional for specific features
pip install nixl  # For NIXL checkpoint engine
```

## Quick Configuration Examples

### 1. Basic VeOmni + vLLM-Omni for Language Models

```yaml
# conf/veomni_qwen25_7b.yaml
trainer:
  n_gpus_per_node: 8
  nnodes: 1

data:
  tokenizer_path: Qwen/Qwen2.5-7B-Instruct
  train_files: data/train.parquet
  val_files: data/val.parquet
  prompt_key: prompt
  response_key: response

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B-Instruct

  actor:
    strategy: veomni  # Use VeOmniEngine
    engine:
      fsdp_size: -1  # Use all GPUs for FSDP
      ulysses_parallel_size: 1
      expert_parallel_size: 1
      enable_full_shard: false
      mixed_precision: true
      attn_implementation: flash_attention_2
    optim:
      lr: 1e-6
      weight_decay: 0.01
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 4

  rollout:
    name: vllm_omni  # Use vLLM-Omni for rollout
    dtype: bfloat16
    tensor_model_parallel_size: 2
    data_parallel_size: 1
    gpu_memory_utilization: 0.7
    max_num_seqs: 512
    enable_sleep_mode: true  # Enable sleep for memory efficiency
```

### 2. Vision-Language Model (Qwen-VL)

```yaml
# conf/veomni_qwen_vl.yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen2-VL-7B-Instruct
    model_type: vision_language_model  # VLM type

  actor:
    strategy: veomni
    engine:
      ulysses_parallel_size: 2  # Enable sequence parallelism
      attn_implementation: veomni_flash_attention_2_with_sp

  rollout:
    name: vllm_omni
    limit_images: 8  # Max images per prompt
```

### 3. MoE (Mixture of Experts) Model

```yaml
# conf/veomni_qwen_moe.yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen1.5-MoE-A2.7B

  actor:
    strategy: veomni
    engine:
      expert_parallel_size: 4  # Expert parallelism
      enable_full_shard: true  # ZeRO-3 for MoE
      moe_implementation: fused  # Fused MoE kernels

  rollout:
    name: vllm_omni
    enable_expert_parallel: true  # EP in rollout
```

### 4. Diffusion Model (FlowGRPO)

```yaml
# conf/veomni_diffusion.yaml
actor_rollout_ref:
  model:
    path: your-diffusion-model
    model_type: diffusion_model

  actor:
    strategy: veomni

  rollout:
    name: vllm_omni
    mode: async
    image_height: 512
    image_width: 512
    num_inference_steps: 40
    guidance_scale: 4.5
    sde_type: sde
```

## Running Training

### Single Node
```bash
# Basic training
python verl/trainer/main_ppo.py --config-path=conf --config-name=veomni_qwen25_7b

# With command line overrides
python verl/trainer/main_ppo.py \
    --config-path=conf \
    --config-name=veomni_qwen25_7b \
    trainer.n_gpus_per_node=4 \
    actor_rollout_ref.actor.engine.fsdp_size=4
```

### Multi-Node (SLURM)
```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

srun python -m verl.trainer.main_ppo \
    --config-path=conf \
    --config-name=veomni_qwen25_7b \
    trainer.nnodes=4
```

### Multi-Node (Ray)
```bash
# Start Ray cluster on head node
ray start --head --port=6379

# Connect worker nodes
ray start --address="<head-node-ip>:6379"

# Run training
python verl/trainer/main_ppo.py \
    --config-path=conf \
    --config-name=veomni_qwen25_7b \
    trainer.nnodes=4
```

## Memory Optimization Guide

### For Large Models (70B+)
```yaml
actor_rollout_ref:
  actor:
    engine:
      enable_full_shard: true  # ZeRO-3
      param_offload: true  # CPU offload
      optimizer_offload: true  # CPU offload
      activation_gpu_limit: 2.0  # GB

  rollout:
    enable_sleep_mode: true
    gpu_memory_utilization: 0.6
```

### For Limited GPU Memory
```yaml
actor_rollout_ref:
  actor:
    engine:
      fsdp_size: 4  # Smaller FSDP groups
      mixed_precision: true
    ppo_micro_batch_size_per_gpu: 1
    ppo_max_token_len_per_gpu: 4096

  rollout:
    tensor_model_parallel_size: 4
    max_num_seqs: 256
```

## Debugging Tips

### 1. Enable Detailed Logging
```yaml
# Add to config
logging:
  level: DEBUG
  log_gpu_memory: true
```

### 2. Memory Profiling
```bash
# Run with memory profiler
python -m memory_profiler verl/trainer/main_ppo.py ...
```

### 3. Gradient Norm Issues
```yaml
actor_rollout_ref:
  actor:
    optim:
      clip_grad: 1.0  # Adjust based on instability
```

### 4. Check Data Loading
```python
# Add to dataset iteration
from verl.utils import log_gpu_memory_usage
log_gpu_memory_usage("After data loading")
```

## Common Issues and Solutions

### Issue 1: NCCL Communication Error
```
RuntimeError: NCCL operation failed
```
**Solution**:
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Adjust interface name
```

### Issue 2: OOM During Weight Update
```
CUDA out of memory during update_weights
```
**Solution**:
```yaml
actor_rollout_ref:
  rollout:
    free_cache_engine: true
    enable_sleep_mode: true
```

### Issue 3: vLLM-Omni Server Not Starting
```
RuntimeError: vllm_omni_server not found
```
**Solution**:
```bash
# Verify installation
python -c "import vllm_omni; print(vllm_omni.__version__)"

# Check Ray actors
ray status
```

### Issue 4: Checkpoint Loading Failure
```
Error: checkpoint not found or corrupted
```
**Solution**:
```yaml
actor_rollout_ref:
  actor:
    checkpoint:
      path: /path/to/checkpoint
      load_optimizer: false  # Try without optimizer states
```

### Issue 5: Sequence Parallel Mismatch
```
RuntimeError: Sequence length not divisible by sp_size
```
**Solution**:
```yaml
actor_rollout_ref:
  actor:
    engine:
      # Ensure sequences are padded
      basic_modules:
        - "Qwen2VLDecoderLayer"
```

## Performance Tuning

### For Throughput
```yaml
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 8  # Larger batches
    use_dynamic_bsz: true
    engine:
      use_torch_compile: true  # Compile ops

  rollout:
    max_num_seqs: 2048  # Higher concurrency
    enable_chunked_prefill: true
```

### For Latency
```yaml
actor_rollout_ref:
  rollout:
    enforce_eager: true  # No CUDA graph compilation
    max_num_batched_tokens: 4096  # Smaller batches
```

## Monitoring Metrics

### Key Metrics to Watch
1. **MFU (Model FLOPs Utilization)**: Should be >30%
2. **GPU Memory**: Keep <90% utilization
3. **Throughput**: tokens/sec or samples/sec
4. **KL Divergence**: Should not explode (use KL penalty)
5. **Reward Mean**: Should increase over training

### Prometheus Monitoring
```yaml
actor_rollout_ref:
  rollout:
    prometheus:
      enable: true
      port: 9090
      served_model_name: "my-model"
```

## Advanced Features

### LoRA Fine-tuning
```yaml
actor_rollout_ref:
  model:
    lora:
      rank: 64
      alpha: 16
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
      merge: false  # Keep separate for efficient sync
```

### Custom Reward Function
```python
# rewards/my_reward.py
from verl.workers.reward import BaseReward

class MyReward(BaseReward):
    def __call__(self, data):
        # Compute reward
        return reward_score
```

```yaml
reward:
  reward_fn:
    class_path: rewards.my_reward.MyReward
```

### Multi-Turn Rollout
```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: true
      max_assistant_turns: 5
      tool_config_path: conf/tools.yaml
```

## Testing Your Setup

### Quick Sanity Check
```bash
# 1. Check imports
python -c "from verl.workers.engine.veomni import VeOmniEngine; print('OK')"

# 2. Check vLLM-Omni
python -c "from vllm_omni.entrypoints import AsyncOmni; print('OK')"

# 3. Test with small config
python verl/trainer/main_ppo.py \
    --config-path=conf \
    --config-name=test_config \
    data.train_max_samples=10 \
    train.total_epochs=1
```

## Additional Resources

- [VeOmni Documentation](https://github.com/ByteDance-Seed/VeOmni)
- [vLLM-Omni Documentation](https://github.com/your-org/vllm-omni)
- [VerL Documentation](https://verl.readthedocs.io)
