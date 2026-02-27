# vLLM-Omni + FSDP Architecture Diagram

## Complete System Architecture

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                    Main Process (Hydra)                      │
                                    │                     main_ppo.py                              │
                                    │                                                              │
                                    │   ┌─────────────────────────────────────────────────┐      │
                                    │   │           TaskRunner (Ray Actor)                │      │
                                    │   │                                                  │      │
                                    │   │  ┌───────────────────────────────────────────┐  │      │
                                    │   │  │        RayPPOTrainer /                      │  │      │
                                    │   │  │        RayFlowGRPOTrainer                   │  │      │
                                    │   │  │                                              │  │      │
                                    │   │  │  ┌─────────┐ ┌─────────┐ ┌─────────────┐   │  │      │
                                    │   │  │  │ Dataset │ │ Sampler │ │ DataLoader  │   │  │      │
                                    │   │  │  └────┬────┘ └────┬────┘ └──────┬──────┘   │  │      │
                                    │   │  │       └───────────┴─────────────┘          │  │      │
                                    │   │  │                    │                         │  │      │
                                    │   │  └────────────────────┼─────────────────────────┘  │      │
                                    │   └───────────────────────┼────────────────────────────┘      │
                                    └───────────────────────────┼───────────────────────────────────┘
                                                                │
                                          ┌─────────────────────┼─────────────────────┐
                                          │                     │                     │
                    ┌─────────────────────▼─────┐    ┌──────────▼──────────┐   ┌──────▼──────┐
                    │   ResourcePoolManager     │    │  ResourcePoolManager│   │   Reward    │
                    │      global_pool          │    │     reward_pool     │   │   Pool      │
                    └─────────────┬─────────────┘    └──────────┬──────────┘   └─────────────┘
                                  │                             │
          ┌───────────────────────┼──────────────────┐          │
          │                       │                  │          │
┌─────────▼──────────┐  ┌─────────▼──────────┐      │  ┌───────▼────────┐
│  ActorRolloutRef   │  │  ActorRolloutRef   │      │  │ TrainingWorker │
│     Worker         │  │     Worker         │...   │  │   (Critic)     │
└─────────┬──────────┘  └─────────┬──────────┘      │  └────────┬───────┘
          │                       │                  │           │
          │  ┌───────────────────────────────────┐   │  ┌────────▼────────┐
          │  │         Worker Components          │   │  │FSDPEngine/     │
          │  │                                    │   │  │VeOmniEngine     │
          │  │  ┌──────────┐  ┌──────────┐       │   │  └─────────────────┘
          │  │  │  Actor   │  │   Ref    │       │   │
          │  │  │ Training │  │ Training │       │   │
          │  │  │  Worker  │  │  Worker  │       │   │
          │  │  └────┬─────┘  └────┬─────┘       │   │
          │  │       │             │             │   │
          │  │  ┌────▼─────────────▼─────┐       │   │
          │  │  │      Rollout Engine    │       │   │
          │  │  │    (vLLM/vLLM-Omni)    │       │   │
          │  │  └────────────────────────┘       │   │
          │  │                                    │   │
          │  └────────────────────────────────────┘   │
          │                                           │
          ▼                                           ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│      TrainingWorker (Actor)      │  │      TrainingWorker (Ref)        │
│                                  │  │                                  │
│  ┌───────────────────────────┐   │  │  ┌───────────────────────────┐   │
│  │      VeOmniEngine         │   │  │  │      VeOmniEngine         │   │
│  │                           │   │  │  │       (forward_only)      │   │
│  │  ┌─────────────────────┐  │   │  │  │                           │   │
│  │  │  Parallel State     │  │   │  │  │  ┌─────────────────────┐  │   │
│  │  │  - dp_shard_size    │  │   │  │  │  │  Parallel State     │  │   │
│  │  │  - dp_replicate_size│  │   │  │  │  │  - dp_shard_size    │  │   │
│  │  │  - ep_size          │  │   │  │  │  │  - ulysses_size     │  │   │
│  │  │  - ulysses_size     │  │   │  │  │  └─────────────────────┘  │   │
│  │  └─────────────────────┘  │   │  │  │                           │   │
│  │                           │   │  │  │  ┌─────────────────────┐  │   │
│  │  ┌─────────────────────┐  │   │  │  │  │  build_foundation   │  │   │
│  │  │  build_foundation   │  │   │  │  │  │  _model             │  │   │
│  │  │  _model             │  │   │  │  │  └─────────────────────┘  │   │
│  │  │  (from VeOmni)      │  │   │  │  │                           │   │
│  │  └─────────────────────┘  │   │  │  │  ┌─────────────────────┐  │   │
│  │                           │   │  │  │  │  build_optimizer    │  │   │
│  │  ┌─────────────────────┐  │   │  │  │  └─────────────────────┘  │   │
│  │  │  build_parallelize  │  │   │  │  │                           │   │
│  │  │  _model             │  │   │  │  │  ┌─────────────────────┐  │   │
│  │  └─────────────────────┘  │   │  │  │  │  build_lr_scheduler │  │   │
│  │                           │   │  │  │  └─────────────────────┘  │   │
│  │  ┌─────────────────────┐  │   │  │  └───────────────────────────┘   │
│  │  │  build_optimizer    │  │   │  └──────────────────────────────────┘
│  │  └─────────────────────┘  │   │
│  │                           │   │
│  │  ┌─────────────────────┐  │   │
│  │  │  build_lr_scheduler │  │   │
│  │  └─────────────────────┘  │   │
│  │                           │   │
│  │  ┌─────────────────────┐  │   │
│  │  │activation_offloading│  │   │
│  │  │       context       │  │   │
│  │  └─────────────────────┘  │   │
│  │                           │   │
│  │  ┌─────────────────────┐  │   │
│  │  │  FSDPCheckpoint     │  │   │
│  │  │     Manager         │  │   │
│  │  └─────────────────────┘  │   │
│  └───────────────────────────┘   │
└──────────────────────────────────┘
```

## VeOmni Engine Detail

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            VeOmniEngine                                           │
│                      (verl/workers/engine/veomni/transformer_impl.py)             │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                         Initialization Flow                              │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│   1. init_parallel_state()                                                        │
│      ├── dp_size                                                                  │
│      ├── dp_replicate_size                                                        │
│      ├── dp_shard_size (FSDP2)                                                    │
│      ├── ep_size (Expert Parallel)                                                │
│      └── ulysses_size (Sequence Parallel)                                         │
│                              │                                                    │
│                              ▼                                                    │
│   2. build_foundation_model()                                                     │
│      ├── config_path (HF path)                                                    │
│      ├── weights_path                                                             │
│      ├── torch_dtype (float32/bfloat16)                                           │
│      ├── attn_implementation                                                      │
│      └── init_device (cpu/cuda/meta/npu)                                          │
│                              │                                                    │
│                              ▼                                                    │
│   3. build_parallelize_model()                                                    │
│      ├── enable_full_shard (ZeRO-3)                                               │
│      ├── enable_mixed_precision                                                   │
│      ├── enable_gradient_checkpointing                                            │
│      ├── basic_modules (_no_split_modules)                                        │
│      └── enable_forward_prefetch                                                  │
│                              │                                                    │
│                              ▼                                                    │
│   4. build_optimizer()                                                            │
│      └── Type: adamw/adam/adafactor...                                            │
│                              │                                                    │
│                              ▼                                                    │
│   5. build_lr_scheduler()                                                         │
│      └── warmup + decay schedule                                                  │
│                              │                                                    │
│                              ▼                                                    │
│   6. build_activation_offloading_context()                                        │
│      └── GPU memory limit for activations                                         │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │                         Training Mode Context                            │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│   EngineTrainModeCtx:                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                      │
│   │ Load model   │───▶│ Set SP group │───▶│ Call train() │                      │
│   │ to GPU       │    │              │    │              │                      │
│   └──────────────┘    └──────────────┘    └──────────────┘                      │
│                                                   │                               │
│   EngineEvalModeCtx:                              │                               │
│   ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐                      │
│   │ Load model   │───▶│ Set SP group │───▶│ Call eval()  │                      │
│   │ to GPU       │    │              │    │ if needed    │                      │
│   └──────────────┘    └──────────────┘    └──────────────┘                      │
│                                                   │                               │
│                                                   ▼                               │
│   ┌─────────────────────────────────────────────────────────────────────┐       │
│   │                    forward_backward_batch                            │       │
│   │                                                                      │       │
│   │   ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────┐  │       │
│   │   │ prepare │───▶│  model_fwd  │───▶│  model_bwd  │───▶│ clip    │  │       │
│   │   │ batches │    │  _context   │    │  _context   │    │ grad    │  │       │
│   │   └─────────┘    └─────────────┘    └─────────────┘    └─────────┘  │       │
│   │                                                              │       │       │
│   │                                                              ▼       │       │
│   │                                                      ┌───────────┐   │       │
│   │                                                      │ optimize  │   │       │
│   │                                                      │  step     │   │       │
│   │                                                      └───────────┘   │       │
│   └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## vLLM-Omni Rollout Detail

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               vLLM-Omni Rollout Architecture                                     │
│                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              vLLMOmniReplica                                              │    │
│  │                         (verl/workers/rollout/replica.py)                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                                         │
│                    ┌──────────────────┼──────────────────┐                                     │
│                    │                  │                  │                                     │
│                    ▼                  ▼                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         vLLMOmniHttpServer (per node)                                 │    │
│  │                   (verl/workers/rollout/vllm_rollout/vllm_omni_async_server.py)       │    │
│  └─────────────────────────────────────────────────────────────────────────────────────┘    │
│                     │                                                                         │
│                     │  launch_server()                                                        │
│                     ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                           AsyncOmni Engine                                            │    │
│  │                        (from vllm_omni library)                                       │    │
│  │                                                                                       │    │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐   │    │
│  │  │                         Engine Configuration                                     │   │    │
│  │  │                                                                                │   │    │
│  │  │  • tensor_parallel_size     • enable_sleep_mode                                │   │    │
│  │  │  • pipeline_parallel_size   • enable_chunked_prefill                           │   │    │
│  │  │  • data_parallel_size       • enable_prefix_caching                            │   │    │
│  │  │  • expert_parallel_size     • cuda_graph_sizes                                 │   │    │
│  │  │  • dtype (bfloat16/fp16)    • gpu_memory_utilization                           │   │    │
│  │  └───────────────────────────────────────────────────────────────────────────────┘   │    │
│  │                                                                                       │    │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐   │    │
│  │  │                      Worker Extension                                           │   │    │
│  │  │             vLLMOmniColocateWorkerExtension                                     │   │    │
│  │  │                                                                                │   │    │
│  │  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                     │   │    │
│  │  │  │   sleep()    │    │  wake_up()   │    │ update_weights│                    │   │    │
│  │  │  │              │    │              │    │              │                     │   │    │
│  │  │  │ Free GPU     │    │ Load weights │    │ Sync from    │                     │   │    │
│  │  │  │ memory       │    │ from CPU/    │    │ FSDP/        │                     │   │    │
│  │  │  │ for training │    │ checkpoint   │    │ VeOmni       │                     │   │    │
│  │  │  └──────────────┘    └──────────────┘    └──────────────┘                     │   │    │
│  │  └───────────────────────────────────────────────────────────────────────────────┘   │    │
│  │                                                                                       │    │
│  └─────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         vLLMOmniServerAdapter                                         │    │
│  │                    (verl/workers/rollout/vllm_rollout/vllm_omni_rollout.py)           │    │
│  │                                                                                       │    │
│  │  • Acts as client to communicate with vLLM-Omni server                                │    │
│  │  • Uses Ray collective_rpc for distributed communication                              │    │
│  │  • Supports IPC (CUDA) and Shared Memory (NPU) for weight transfer                    │    │
│  │                                                                                       │    │
│  │  Communication Flow:                                                                  │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                            │    │
│  │  │  Training    │    │  collective  │    │  vLLM-Omni   │                            │    │
│  │  │  Worker      │───▶│  _rpc        │───▶│  Server      │                            │    │
│  │  └──────────────┘    └──────────────┘    └──────────────┘                            │    │
│  │         │                                          │                                  │    │
│  │         └──────────────────────────────────────────┘                                  │    │
│  │                         ZMQ / IPC / SHM                                               │    │
│  └─────────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Weight Synchronization Flow

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            Weight Update: Actor → Rollout                                         │
│                                                                                                   │
│   Step 1: Actor Training                                                                          │
│   ┌─────────────────────────┐                                                                     │
│   │   ActorRolloutRefWorker │                                                                     │
│   │                         │                                                                     │
│   │  ┌───────────────────┐  │                                                                     │
│   │  │     Actor         │  │  ◄─── Trains on batch                                              │
│   │  │  (VeOmniEngine)   │  │                                                                     │
│   │  └─────────┬─────────┘  │                                                                     │
│   └────────────┼────────────┘                                                                     │
│                │                                                                                  │
│                │ After training                                                                   │
│                ▼                                                                                  │
│   ┌─────────────────────────┐                                                                     │
│   │      update_weights()   │  ◄─── Called periodically                                          │
│   └────────────┬────────────┘                                                                     │
│                │                                                                                  │
│                │ 1. Resume rollout (wake up engine)                                               │
│                ▼                                                                                  │
│   ┌─────────────────────────┐                                                                     │
│   │   rollout.resume()      │  ◄─── Load weights to GPU, wake from sleep                         │
│   │   (vLLMOmniServer)      │                                                                     │
│   └────────────┬────────────┘                                                                     │
│                │                                                                                  │
│                │ 2. Get tensor parameters from actor                                              │
│                ▼                                                                                  │
│   ┌─────────────────────────┐                                                                     │
│   │ get_per_tensor_param()  │                                                                     │
│   │  ┌───────────────────┐  │                                                                     │
│   │  │ Load model to GPU │  │  ◄─── Offloaded params back to GPU                               │
│   │  │ (if offloaded)    │  │                                                                     │
│   │  └─────────┬─────────┘  │                                                                     │
│   │            │            │                                                                     │
│   │            ▼            │                                                                     │
│   │  ┌───────────────────┐  │                                                                     │
│   │  │ state_dict()      │  │  ◄─── Get FSDP sharded state dict                                 │
│   │  │ Get DTensor/WDT   │  │                                                                     │
│   │  └─────────┬─────────┘  │                                                                     │
│   │            │            │                                                                     │
│   │            ▼            │                                                                     │
│   │  ┌───────────────────┐  │                                                                     │
│   │  │ all_gather (EP)   │  │  ◄─── For MoE models, gather expert weights                        │
│   │  └───────────────────┘  │                                                                     │
│   └────────────┬────────────┘                                                                     │
│                │                                                                                  │
│                │ Returns: param_generator()                                                        │
│                ▼                                                                                  │
│   ┌─────────────────────────┐                                                                     │
│   │  rollout.update_weights │                                                                     │
│   │  ┌───────────────────┐  │                                                                     │
│   │  │ DTensor/HF loader │  │  ◄─── Use DTensor to synchronize across TP ranks                   │
│   │  │ Layer offload     │  │      (or HF weight loader with broadcast)                         │
│   │  └───────────────────┘  │                                                                     │
│   └────────────┬────────────┘                                                                     │
│                │                                                                                  │
│                │ 3. Offload actor model                                                           │
│                ▼                                                                                  │
│   ┌─────────────────────────┐                                                                     │
│   │  engine.to(device="cpu")│  ◄─── Free GPU memory for rollout generation                       │
│   └─────────────────────────┘                                                                     │
│                │                                                                                  │
│                │ 4. Resume KV cache                                                               │
│                ▼                                                                                  │
│   ┌─────────────────────────┐                                                                     │
│   │  rollout.resume(kv_cache│  ◄─── Ready for generation                                         │
│   └─────────────────────────┘                                                                     │
│                                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Sequence Parallel Support (Ulysses)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          Ulysses Sequence Parallel in VeOmni                                      │
│                                                                                                   │
│   Configuration:                                                                                  │
│   • ulysses_parallel_size = 4  (4-way SP)                                                         │
│   • dp_shard_size = 4                                                                           │
│                                                                                                   │
│   Data Flow:                                                                                      │
│                                                                                                   │
│   Input: [batch_size=2, seq_len=4096]                                                             │
│                                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                        OmniSequenceShardCollator                                           │  │
│   │                                                                                            │  │
│   │  1. Padding for SP alignment                                                               │  │
│   │  2. Slice sequence dimension across SP ranks                                               │  │
│   └───────────────────────────────────────────────────────────────────────────────────────────┘  │
│                    │                                                                              │
│                    ▼                                                                              │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                            │  │
│   │  DP Rank 0                           DP Rank 1                                             │  │
│   │  ┌─────────────────────┐            ┌─────────────────────┐                                │  │
│   │  │ SP Group 0          │            │ SP Group 1          │                                │  │
│   │  │ ┌─────┬─────┬─────┐│            │ ┌─────┬─────┬─────┐ │                                │  │
│   │  │ │ SP0 │ SP1 │ SP2 ││            │ │ SP0 │ SP1 │ SP2 │ │                                │  │
│   │  │ │     │     │     ││            │ │     │     │     │ │                                │  │
│   │  │ │1024 │1024 │1024 ││            │ │1024 │1024 │1024 │ │  ...                           │  │
│   │  │ │tokens│    │     ││            │ │tokens│    │     │ │                                │  │
│   │  │ └─────┴─────┴─────┘│            │ └─────┴─────┴─────┘ │                                │  │
│   │  └─────────────────────┘            └─────────────────────┘                                │  │
│   │                                                                                            │  │
│   │  • Each SP rank processes 1024 tokens                                                      │  │
│   │  • AllGather for attention computation                                                      │  │
│   │  • ReduceScatter for gradients                                                            │  │
│   └───────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                   │
│   Vision-Language Special Handling:                                                               │
│   • pixel_values: padding with scale factor (default 4)                                           │
│   • image_mask, video_mask: identification in input_ids                                           │
│   • Custom sequence slicing for multimodal inputs                                                 │
│                                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Expert Parallel (MoE) Support

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           Expert Parallel in VeOmniEngine                                        │
│                                                                                                  │
│   Configuration:                                                                                 │
│   • expert_parallel_size = 4                                                                     │
│   • MoE implementation: "fused" or "eager"                                                       │
│                                                                                                  │
│   Weight Organization for MoE Layers:                                                            │
│                                                                                                  │
│   Global Expert Weights: [num_experts=64, hidden, ffn_hidden]                                   │
│                              │                                                                   │
│                              ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                           EP Group Distribution                                            │  │
│   │                                                                                            │  │
│   │  EP Rank 0: experts [0:16]                                                                 │  │
│   │  EP Rank 1: experts [16:32]                                                                │  │
│   │  EP Rank 2: experts [32:48]                                                                │  │
│   │  EP Rank 3: experts [48:64]                                                                │  │
│   │                                                                                            │  │
│   │  Each rank holds 1/4 of the experts                                                        │  │
│   │  all_gather is used during weight sync to rollout                                           │  │
│   └─────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                  │
│   get_per_tensor_param() Flow:                                                                   │
│                                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │  for name, param in module.state_dict():                                                  │  │
│   │      if is_expert_layer and is_proj:                                                      │  │
│   │          # [16, H, I] local tensor                                                        │  │
│   │          output_shape = [64, H, I]  # multiply by ep_size                                 │  │
│   │          stacked_tensor = empty(output_shape)                                             │  │
│   │          all_gather_into_tensor(stacked_tensor, unsharded_tensor, ep_group)             │  │
│   │          # Now: [64, H, I] full tensor for rollout                                         │  │
│   │          yield from process_func(name, stacked_tensor)                                    │  │
│   └─────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```
