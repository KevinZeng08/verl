# Design Doc: verl 集成 Qwen-Omni RL 训练（基于 vLLM-Omni Rollout）

## 1. 目标与范围

### 1.1 目标

在 verl 框架中支持 Qwen3-Omni 模型的 RL（GRPO）训练，使用 vLLM-Omni 作为 rollout 推理引擎。

### 1.2 范围界定

本设计分两个阶段：

| 阶段 | 范围 | 输入 | 输出 | RL 训练目标 |
|------|------|------|------|-------------|
| Phase 1 | 多模态理解 RL | text + image + audio | text only | Thinker (LLM decoder) |
| Phase 2 (未来) | 全模态生成 RL | text + image + audio | text + audio | Thinker + Talker |

Phase 1 是本文档的核心设计目标。Phase 2 依赖 vLLM-Omni 对 Qwen3-TTS/Talker 的稳定支持（目前仍在开发中）。

### 1.3 为什么用 vLLM-Omni 而不是标准 vLLM

Qwen3-Omni 包含音频 encoder（Whisper-like），标准 vLLM 对其支持有限。vLLM-Omni 原生支持：
- 多模态输入（text + image + audio）的统一推理
- 音频 encoder 的前处理和 feature extraction
- 未来扩展到音频输出（Talker + Code2Wav）的路径

即使 Phase 1 只做文本输出，使用 vLLM-Omni 也能避免后续迁移成本。

## 2. Qwen3-Omni 模型架构分析

### 2.1 模型组成

```
Qwen3-Omni
├── Audio Encoder (Whisper-like)     ← 音频输入 → audio embeddings
├── Vision Encoder (ViT)             ← 图像/视频输入 → visual embeddings
├── Thinker (LLM Decoder, MoE)      ← embeddings → text tokens (核心 RL 训练目标)
├── Talker (Conditional Generation)  ← text tokens → speech codes (Phase 2)
└── Code2Wav (Vocoder)               ← speech codes → waveform audio (Phase 2)
```

### 2.2 Phase 1 训练策略

参考 ROLL 框架的做法：
- **冻结**：Audio Encoder、Vision Encoder（作为 feature extractor）
- **训练**：Thinker（LLM decoder，MoE 架构）
- **不加载**：Talker、Code2Wav（`enable_audio_output=False`）

这使得 Phase 1 本质上等价于一个增强版 VLM 的 RL 训练，复用 verl 现有的 LLM RL pipeline，只需扩展多模态输入处理。

### 2.3 与现有 Qwen3-VL 的差异

| 维度 | Qwen3-VL | Qwen3-Omni |
|------|----------|------------|
| HF model type | `qwen3_vl` / `qwen3_vl_moe` | `qwen3_omni_moe` |
| 输入模态 | text + image + video | text + image + video + audio |
| 额外 encoder | 无 | Audio Encoder |
| LLM 架构 | Dense / MoE | MoE |
| 特殊 token | IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX | + AUDIO_INPUT_INDEX |

## 3. 整体架构设计

### 3.1 Phase 1 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        RayPPOTrainer                            │
│  (复用现有 trainer，无需新建 RayOmniTrainer)                      │
└──────┬──────────────┬───────────────┬──────────────┬────────────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
  ┌─────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │ Rollout │  │  Actor    │  │    Ref    │  │  Reward   │
  │ (vLLM-  │  │ (FSDP/   │  │  (FSDP/  │  │  (rule/   │
  │  Omni)  │  │  VeOmni) │  │  VeOmni) │  │  model)   │
  └─────────┘  └───────────┘  └───────────┘  └───────────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
  多模态输入     Thinker only    Thinker only    文本评分
  text输出       log_prob计算     ref log_prob
```

### 3.2 关键设计决策

**决策 1：复用 RayPPOTrainer，不新建 Trainer**

理由：
- Phase 1 的 Qwen3-Omni RL 本质上是 LLM text generation 的 GRPO，与现有 Qwen3-VL 训练流程一致
- 新建 Trainer（如 PR #5297 的 `RayFlowGRPOTrainer`）会导致大量代码重复和维护负担
- 差异点（多模态输入处理、音频 token embedding）应在 model/rollout 层解决，而非 trainer 层

**决策 2：使用 vLLM-Omni 作为 rollout 引擎**

理由：
- vLLM-Omni 已原生支持 Qwen3-Omni 的多模态推理（text + image + audio 输入）
- verl 已有 `vLLMOmniServerAdapter` 和 `vLLMOmniHttpServer` 基础设施（来自 PR #5297 的扩散模型支持）
- 需要适配：当前 vLLM-Omni rollout 代码面向扩散模型（ImageOutput），需扩展支持 LLM text output

**决策 3：训练引擎优先支持 FSDP/VeOmni，Megatron 后续跟进**

理由：
- Megatron 后端需要 mbridge 注册 `qwen3_omni_moe`，这是外部依赖
- FSDP/VeOmni 可直接加载 HuggingFace 权重，无需额外转换
- VeOmni 已支持 `qwen3_vl_moe`，扩展到 `qwen3_omni_moe` 改动较小

## 4. 详细模块设计

### 4.1 模型注册与 Forward Pass

**文件改动：**
- 新增 `verl/models/transformers/qwen3_omni.py`
- 修改 `verl/models/transformers/__init__.py`（注册新模型）

**设计要点：**

继承现有 Qwen3-VL 的实现，扩展音频输入处理：

```python
# verl/models/transformers/qwen3_omni.py
# 核心思路：复用 qwen3_vl 的 forward，扩展 construct_inputs_embeds 处理 audio tokens

def qwen3_omni_moe_forward(
    self, input_ids, attention_mask, position_ids,
    pixel_values=None, image_grid_thw=None,
    audio_features=None, audio_feature_lengths=None,  # 新增
    **kwargs
):
    # 1. 构建 text embeddings
    inputs_embeds = self.embed_tokens(input_ids)

    # 2. 替换 image/video tokens（复用 qwen3_vl 逻辑）
    if pixel_values is not None:
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        inputs_embeds = merge_visual_tokens(inputs_embeds, image_embeds, input_ids)

    # 3. 替换 audio tokens（新增）
    if audio_features is not None:
        audio_embeds = self.audio_encoder(audio_features, audio_feature_lengths)
        inputs_embeds = merge_audio_tokens(inputs_embeds, audio_embeds, input_ids)

    # 4. 标准 LLM forward
    return original_forward(inputs_embeds, attention_mask, position_ids, **kwargs)
```

**VL_TYPE2INDEX 扩展：**

```python
# 在现有注册表中添加
VL_TYPE2INDEX["qwen3_omni_moe"] = {
    "IMAGE_INPUT_INDEX": 151655,
    "VIDEO_INPUT_INDEX": 151656,
    "AUDIO_INPUT_INDEX": <从 HF config 获取>,  # Qwen3-Omni 的 audio_token_id
}
```

### 4.2 Rollout 层：vLLM-Omni 适配

**核心问题：** 现有 `vLLMOmniHttpServer` 面向扩散模型，`generate()` 返回 `ImageOutput`（latents/timesteps）。Qwen3-Omni 的 rollout 需要返回标准 LLM 输出（token ids + log_probs）。

**方案：扩展 vLLMOmniHttpServer 支持 LLM 模式**

```
现有架构：
  vLLMOmniHttpServer.generate() → ImageOutput (扩散模型专用)

目标架构：
  vLLMOmniHttpServer
  ├── generate()          → ImageOutput  (扩散模型，保持不变)
  └── generate_text()     → TextOutput   (LLM 模式，新增)
```

**文件改动：**
- 修改 `verl/workers/rollout/vllm_rollout/vllm_omni_async_server.py`
- 修改 `verl/workers/rollout/vllm_rollout/vllm_omni_rollout.py`
- 修改 `verl/workers/rollout/replica.py`（新增 TextOutput 或复用现有结构）

**generate_text 核心逻辑：**

```python
async def generate_text(
    self,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    request_id: str,
    image_data: Optional[list[Any]] = None,
    video_data: Optional[list[Any]] = None,
    audio_data: Optional[list[Any]] = None,  # 新增音频输入
) -> dict:
    """Generate text with multi-modal input via vLLM-Omni."""
    multi_modal_data = {}
    if image_data is not None:
        multi_modal_data["image"] = image_data
    if video_data is not None:
        multi_modal_data["video"] = video_data
    if audio_data is not None:
        multi_modal_data["audio"] = audio_data

    # 调用 vLLM-Omni 的标准 LLM generate（非扩散模式）
    generator = self.engine.generate(
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        request_id=request_id,
        multi_modal_data=multi_modal_data,
    )

    final_res = None
    async for output in generator:
        final_res = output

    return {
        "output_ids": final_res.outputs[0].token_ids,
        "log_probs": final_res.outputs[0].logprobs,
    }
```

**关键考虑：**
- vLLM-Omni 的 `AsyncOmni` 引擎同时支持扩散和 LLM 推理，通过请求类型区分
- 音频数据需要在 agent loop 层做预处理（采样率转换、特征提取），或依赖 vLLM-Omni 内部处理
- 权重同步逻辑（`vLLMOmniServerAdapter`）可完全复用，因为只同步 Thinker 权重

### 4.3 训练引擎适配

**Phase 1 优先路径：VeOmni 引擎**

VeOmni 已支持 `qwen3_vl_moe`，扩展到 `qwen3_omni_moe` 需要：

**文件改动：**
- 修改 `verl/workers/engine/veomni/transformer_impl.py`（如需特殊处理音频 encoder 冻结）
- 新增/修改 VeOmni 的模型构建逻辑以识别 `qwen3_omni_moe`

**关键点：**

```python
# 模型加载时的冻结策略
def freeze_non_thinker_modules(model):
    """冻结 Audio Encoder 和 Vision Encoder，只训练 Thinker。"""
    for name, param in model.named_parameters():
        if any(prefix in name for prefix in [
            "audio_encoder", "audio_tower",
            "visual", "vision_tower",
        ]):
            param.requires_grad = False

# FSDP 分片策略
# Audio/Vision Encoder 冻结后不参与 FSDP 分片，减少通信开销
# 只对 Thinker (LLM decoder) 部分做 FSDP wrap
```

**FSDP 分片注意事项：**
- 冻结的 encoder 参数不需要梯度同步，应排除在 FSDP 分片之外或使用 `NO_SHARD` 策略
- MoE 层的 expert parallel 可复用 VeOmni 现有的 EP 支持
- 音频 encoder 的权重较小（~600M），可全量放在每个 rank 上

**Megatron 后端（后续）：**
- 需要在 mbridge 中注册 `qwen3_omni_moe` 的权重映射
- 需要处理 audio encoder 的 PP（pipeline parallel）放置策略
- 依赖 mbridge 上游支持，不在本设计的首期范围内

### 4.4 数据集与 DataLoader

**文件改动：**
- 修改 `verl/utils/dataset/rl_dataset.py`（扩展多模态数据加载）

**数据格式设计：**

```jsonl
{
  "prompt": [
    {"role": "user", "content": [
      {"type": "text", "text": "描述这张图片中的声音场景"},
      {"type": "image", "image": "s3://bucket/image.jpg"},
      {"type": "audio", "audio": "s3://bucket/audio.wav"}
    ]}
  ],
  "reward_model": {"style": "rule", "ground_truth": "..."}
}
```

**关键处理：**
- 图像：复用现有 Qwen3-VL 的图像预处理（smart_resize + pixel_values + image_grid_thw）
- 音频：需要新增音频预处理 pipeline
  - 采样率标准化（16kHz）
  - 特征提取（mel spectrogram 或直接传 raw waveform，取决于 vLLM-Omni 的接口）
  - padding/truncation 到固定长度
- Collate 函数需要处理 batch 内不同样本的模态组合不同（有的只有图像，有的只有音频，有的都有）

```python
# 音频预处理示例
def preprocess_audio(audio_path: str, target_sr: int = 16000) -> dict:
    """加载并预处理音频数据。"""
    import librosa
    waveform, sr = librosa.load(audio_path, sr=target_sr)
    return {
        "audio_data": waveform,
        "audio_length": len(waveform),
    }
```

### 4.5 Reward 设计

Phase 1 支持的 reward 类型（参考 ROLL 的 Qwen3-Omni 配置）：

| Reward 类型 | 适用场景 | 实现方式 |
|-------------|----------|----------|
| 规则 reward | 数学推理、代码生成 | 正则匹配 + 结果验证（verl 已有） |
| VLM reward | 图像理解任务 | 调用 Qwen2.5-VL 等外部模型打分 |
| ASR reward | 语音理解任务 | 调用 Whisper 等 ASR 模型，比较转录结果 |
| 检测 reward | 目标检测任务 | IoU / mAP 计算（参考 ROLL 的 CV detection reward） |

**文件改动：**
- 可复用 `verl/utils/reward_score/` 下的现有 reward 函数
- 如需新增音频相关 reward，在该目录下添加

**关键点：**
- Phase 1 的 reward 全部基于文本输出评估，与现有 VLM RL 完全一致
- 音频输入只影响模型的理解能力，不影响 reward 计算方式
- 多任务混合训练时，不同任务使用不同 reward 函数（通过 dataset 的 `reward_model` 字段路由）

## 5. 配置设计

### 5.1 示例训练配置

```yaml
# examples/qwen3_omni/ppo_omni_trainer.yaml

# 模型配置
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-Omni-30B-A3B-Thinking
    enable_audio_output: false  # Phase 1 不启用音频输出
    freeze_modules:             # 冻结非 Thinker 模块
      - audio_encoder
      - visual

  # Rollout 配置 - 使用 vLLM-Omni
  rollout:
    name: vllm_omni
    mode: async
    tensor_model_parallel_size: 4
    expert_parallel_size: 8
    response_length: 4096
    temperature: 1.0
    top_p: 1.0
    gpu_memory_utilization: 0.85
    # vLLM-Omni 特有配置
    engine_kwargs:
      generation_mode: text     # 区分扩散模式和 LLM 模式
      enable_audio_input: true  # 启用音频输入处理

  # Actor 配置
  actor:
    strategy: fsdp              # 或 veomni
    ppo_micro_batch_size: 4
    ppo_mini_batch_size: 16
    grad_clip: 1.0
    optim:
      lr: 1.0e-6
      weight_decay: 0.01

  # Ref 配置
  ref:
    strategy: fsdp
    log_prob_micro_batch_size: 8

# 算法配置
algorithm:
  adv_estimator: grpo
  kl_ctrl:
    type: fixed
    kl_coef: 0.01
  grpo:
    n: 8                        # 每个 prompt 采样 8 个 response

# Reward 配置
reward_model:
  reward_manager: default
  # 多任务 reward 路由
  reward_fn:
    math: rule_based
    vision: vlm_judge
    audio_understanding: asr_match

# 数据配置
data:
  train_files:
    - path: data/omni_math.parquet
      weight: 0.4
    - path: data/omni_vision.parquet
      weight: 0.3
    - path: data/omni_audio.parquet
      weight: 0.3
  val_files:
    - path: data/omni_val.parquet

# 训练配置
trainer:
  total_epochs: 3
  save_freq: 50
  test_freq: 10
  project_name: qwen3_omni_rl
  logger:
    - wandb
```

### 5.2 配置与现有体系的关系

```
现有配置体系：
  ppo_trainer.yaml              ← LLM RL (Qwen3, Llama, etc.)
  _generated_ppo_veomni_trainer.yaml  ← VeOmni 引擎 LLM RL
  ppo_diffusion_trainer.yaml    ← 扩散模型 RL (PR #5297)

新增：
  ppo_omni_trainer.yaml         ← Qwen3-Omni RL (本设计)
```

关键区别：`ppo_omni_trainer.yaml` 本质上是 `ppo_trainer.yaml` 的变体，只是 rollout 引擎从 `vllm` 改为 `vllm_omni`，并增加了音频输入相关配置。不需要新的 trainer 入口。

## 6. Phase 2 展望：全模态输出 RL

### 6.1 目标

支持 Qwen3-Omni 的语音输出 RL 训练：text + image + audio 输入 → text + audio 输出。

### 6.2 前置依赖

| 依赖项 | 当前状态 | 说明 |
|--------|----------|------|
| vLLM-Omni Talker 推理 | 开发中，有 bug | Issue #1411: 重复音频输出；Issue #1403: RuntimeError |
| vLLM-Omni TTS batched decoding | PR #1426 开发中 | 批量语音生成 |
| Talker 权重同步 | 未实现 | 需要同步 Thinker + Talker 两部分权重 |
| 音频 reward 模型 | 需要设计 | 评估语音质量（MOS、WER、说话人相似度等） |

### 6.3 架构变化

Phase 2 需要的核心改动：

```
Phase 1:
  Rollout: vLLM-Omni → text tokens
  Training: Thinker only (标准 GRPO)

Phase 2:
  Rollout: vLLM-Omni → text tokens + speech codes + audio waveform
  Training: Thinker + Talker (需要联合训练或分阶段训练)
  Reward: 文本 reward + 音频 reward (MOS, WER, speaker similarity)
```

**关键挑战：**

1. **联合 vs 分阶段训练**
   - 联合训练：Thinker 和 Talker 同时更新，梯度需要跨模块传播
   - 分阶段训练：先训练 Thinker（Phase 1），再冻结 Thinker 训练 Talker
   - 建议先实现分阶段训练，降低复杂度

2. **音频 log_prob 计算**
   - Talker 是 conditional generation 模型，其 log_prob 计算方式与 LLM 不同
   - 需要在训练引擎中实现 Talker 的 forward pass 和 log_prob 提取

3. **多模态 reward 聚合**
   - 文本输出和音频输出需要分别评估，然后加权聚合
   - reward 权重的设计是一个开放问题

### 6.4 Talker RL 的算法选择

| 方案 | 描述 | 复杂度 |
|------|------|--------|
| 标准 GRPO on Talker | 把 speech codes 当作 token 序列，直接用 GRPO | 低 |
| FlowGRPO on Talker | 如果 Talker 使用 flow matching，用 FlowGRPO | 中（可复用 PR #5297） |
| 分离 reward | Thinker 用文本 reward，Talker 用音频 reward，独立优化 | 中 |

## 7. 实施计划

### 7.1 Phase 1 任务分解

```
Phase 1: 多模态理解 RL（预计 4-6 周）
│
├── P1.1 模型注册与 Forward Pass (1 周)
│   ├── 新增 verl/models/transformers/qwen3_omni.py
│   ├── 注册 qwen3_omni_moe 到 VL_TYPE2INDEX
│   ├── 实现 audio token merge 逻辑
│   └── 单元测试：forward pass 正确性
│
├── P1.2 vLLM-Omni Rollout 适配 (1-2 周)
│   ├── vLLMOmniHttpServer 新增 generate_text() 方法
│   ├── 支持音频输入数据传递
│   ├── Agent loop 扩展音频预处理
│   └── 集成测试：多模态输入 → 文本输出
│
├── P1.3 训练引擎适配 (1 周)
│   ├── VeOmni/FSDP 引擎支持 qwen3_omni_moe 加载
│   ├── 实现 freeze_non_thinker_modules
│   ├── 验证 FSDP 分片 + MoE EP 正确性
│   └── 单元测试：训练 step 正确性
│
├── P1.4 数据集与配置 (0.5 周)
│   ├── 扩展 rl_dataset.py 支持音频数据加载
│   ├── 新增 ppo_omni_trainer.yaml
│   └── 新增示例数据预处理脚本
│
├── P1.5 端到端集成与验证 (1-2 周)
│   ├── 小规模端到端训练验证（数学 + 视觉 + 音频理解任务）
│   ├── 训练曲线对比（vs ROLL 的 Qwen3-Omni 结果）
│   ├── 性能 benchmark（throughput, GPU utilization）
│   └── 示例脚本 examples/qwen3_omni/
│
└── P1.6 文档与 PR (0.5 周)
    ├── 使用文档
    └── PR 提交与 review
```

### 7.2 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `verl/models/transformers/qwen3_omni.py` | 新增 | 模型 forward pass + monkey patch |
| `verl/models/transformers/__init__.py` | 修改 | 注册 qwen3_omni_moe |
| `verl/workers/rollout/vllm_rollout/vllm_omni_async_server.py` | 修改 | 新增 generate_text() |
| `verl/workers/rollout/vllm_rollout/vllm_omni_rollout.py` | 修改 | 适配 LLM 模式 |
| `verl/workers/engine/veomni/transformer_impl.py` | 修改 | 支持 qwen3_omni_moe |
| `verl/utils/dataset/rl_dataset.py` | 修改 | 音频数据加载 |
| `verl/trainer/config/ppo_omni_trainer.yaml` | 新增 | 训练配置 |
| `verl/trainer/config/model/omni_model.yaml` | 新增 | 模型配置 |
| `examples/qwen3_omni/run_ppo_omni.sh` | 新增 | 启动脚本 |
| `examples/data_preprocess/qwen3_omni.py` | 新增 | 数据预处理 |
| `tests/models/test_qwen3_omni_forward.py` | 新增 | 模型测试 |

### 7.3 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| vLLM-Omni 对 Qwen3-Omni 的多模态推理有 bug（OOM、精度问题） | Rollout 不稳定 | 密切跟进 vLLM-Omni 上游修复；准备 fallback 到纯文本+图像模式 |
| 音频 encoder 冻结后 FSDP 分片行为异常 | 训练失败 | 先用 `NO_SHARD` 策略放置 encoder；必要时将 encoder 移到 CPU |
| Qwen3-Omni HF 权重格式与 VeOmni 加载不兼容 | 无法启动训练 | 编写权重转换脚本；或直接用 FSDP 引擎绕过 VeOmni |
| 音频预处理成为数据加载瓶颈 | 训练吞吐下降 | 离线预处理音频特征，训练时直接加载 tensor |

## 8. 与现有工作的关系

| 现有工作 | 关系 | 复用/冲突 |
|----------|------|-----------|
| PR #5297 (FlowGRPO for QwenImage) | 互补 | 复用 vLLMOmniServerAdapter、vLLMOmniReplica 基础设施 |
| PR #3297 (Qwen-2.5-Omni audio) | 部分重叠 | 可参考其音频预处理逻辑 |
| Issue #5281 (Qwen3-Omni Megatron) | 下游依赖 | Megatron 支持作为后续工作 |
| VeOmni 引擎 (已合入) | 直接复用 | 作为首选训练引擎 |
| Qwen3-VL 支持 (已合入) | 直接复用 | 模型注册、forward pass 的基础 |
