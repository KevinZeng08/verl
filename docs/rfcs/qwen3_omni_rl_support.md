# RFC: Qwen3-Omni Model RL Support

## Abstract

This RFC proposes comprehensive support for Qwen3-Omni model in veRL framework, enabling reinforcement learning training for omni-modal models that can process text, image, video, and audio inputs simultaneously.

## Motivation

Qwen3-Omni represents the next generation of multimodal models with native support for multiple modalities (text, image, video, audio). Enabling RL training for such models in veRL will unlock capabilities for:
- Aligning omni-modal models with human preferences
- Training agent systems with rich multimodal interactions
- Supporting advanced multimodal reasoning tasks

## Goals

1. Integrate Qwen3-Omni rollout using vLLM-Omni framework
2. Adapt training pipeline (FSDP/Megatron backend decision needed)
3. Design RL algorithm and reward model strategy
4. Extend veRL's multimodal data handling to support audio modality
5. Validate end-to-end training pipeline

---

## Phase 1: Rollout Integration

### 1.1 vLLM-Omni Integration

**Overview**: Integrate Qwen3-Omni inference using vLLM-Omni framework for efficient multimodal token generation.

**Tasks**:
- [ ] Research vLLM-Omni API and multimodal input handling
- [ ] Create `OmniServerManager` or extend existing `AsyncLLMServerManager`
- [ ] Implement `OmniReplica` class supporting omni-modal inputs

**Key Components**:

```python
# Proposed: verl/workers/rollout/vllm_rollout/omni_async_server.py

class OmniHttpServer:
    """vLLM-Omni HTTP Server for Qwen3-Omni model."""

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        multi_modal_data: Optional[dict[str, Any]] = None,  # {"image": [...], "video": [...], "audio": [...]}
    ) -> TokenOutput:
        """Generate with omni-modal inputs."""
        pass
```

**Questions to Resolve**:
- What is vLLM-Omni's `multi_modal_data` format for audio?
- Does vLLM-Omni support all modalities in `TokensPrompt`?
- What is the audio tokenization/preprocessing pipeline?

### 1.2 Audio Modality Support

**Audio Input Format** (TBD based on vLLM-Omni):

```python
multi_modal_data = {
    "image": [PIL.Image, ...],           # Existing
    "video": [(torch.Tensor, metadata), ...],  # Existing
    "audio": [???],                       # New - format to be determined
}
```

Potential audio formats:
- Raw waveform: `np.ndarray` or `torch.Tensor` with sample rate
- Audio file paths: `list[str]`
- Pre-processed features: `dict[str, torch.Tensor]`

---

## Phase 2: Training Backend Adaptation

### 2.1 Backend Options

| Backend | Pros | Cons |
|---------|------|------|
| **FSDP** | Native PyTorch, easier integration, well-supported in veRL | May have efficiency limitations for large models |
| **Megatron** | Better efficiency for large-scale training, tensor parallelism | More complex setup, requires Megatron-LM dependency |

### 2.2 Decision Points

- [ ] Benchmark FSDP vs Megatron for Qwen3-Omni throughput
- [ ] Evaluate memory efficiency with multimodal inputs
- [ ] Consider hybrid approach (FSDP for development, Megatron for production)

### 2.3 Audio Data Pipeline for Training

**Processor Integration**:
```python
# Extend processor to handle audio
class OmniProcessor:
    def __call__(
        self,
        text: list[str],
        images: list = None,
        videos: list = None,
        audios: list = None,  # New
        **kwargs
    ):
        pass
```

**Tasks**:
- [ ] Investigate Qwen3-Omni's audio preprocessing requirements
- [ ] Implement audio feature extraction in `AgentLoopWorker._compute_multi_modal_inputs()`
- [ ] Handle audio-specific position IDs if needed

---

## Phase 3: RL Algorithm & Reward Model

### 3.1 Algorithm Options

| Algorithm | Use Case | Complexity |
|-----------|----------|------------|
| **PPO** | General preference alignment, stable training | Medium |
| **GRPO** | Group-based optimization, simpler implementation | Low |
| **DAPO** | Dynamic advantage, better for long sequences | Medium |
| **Reinforce++** | Improved variance reduction | Medium |

**Recommendation**: Start with PPO or GRPO as baseline, evaluate others for specific use cases.

### 3.2 Reward Model Strategy

**Options**:

1. **Text-only Reward Model**
   - Simplest approach
   - Convert multimodal outputs to text, then compute reward
   - Limitation: May not capture multimodal quality

2. **Multimodal Reward Model**
   - Native multimodal understanding
   - Requires training or fine-tuning a reward model
   - Better alignment with multimodal tasks

3. **Task-specific Rewards**
   - Tool call success rate
   - Task completion metrics
   - Custom evaluation functions

**Tasks**:
- [ ] Define reward model requirements for target use cases
- [ ] Evaluate existing multimodal reward models (if any)
- [ ] Design reward computation pipeline

### 3.3 Agent Loop Support

**Omni-Agent Loop**:

```python
@register("omni_agent")
class OmniAgentLoop(AgentLoopBase):
    """Agent loop supporting omni-modal interactions."""

    async def process_omni_info(self, messages: list[dict]) -> dict:
        """Extract all modalities from messages."""
        multi_modal_data = await self.process_vision_info(messages)
        # Add audio extraction
        audio_data = await self.process_audio_info(messages)
        if audio_data:
            multi_modal_data["audio"] = audio_data
        return multi_modal_data
```

---

## Phase 4: Multimodal Data Handling Extension

### 4.1 Current State

veRL currently supports:
- `image`: PIL.Image list
- `video`: (torch.Tensor, metadata) tuple list

### 4.2 Proposed Extension

```python
# verl/experimental/agent_loop/agent_loop.py

class AgentLoopBase:
    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Process vision and audio info from messages."""
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(...)

            if images is not None:
                multi_modal_data["image"] = images
            if videos is not None:
                multi_modal_data["video"] = videos

            # New: Audio support
            audios = await self.dataset_cls.process_audio_info(messages)
            if audios is not None:
                multi_modal_data["audio"] = audios

        return multi_modal_data
```

### 4.3 Audio Processing Pipeline

**Dataset Level** (`verl/utils/dataset/rl_dataset.py`):

```python
class RLHFDataset:
    @staticmethod
    async def process_audio_info(
        messages: list[dict],
        sample_rate: int = 16000,
    ) -> Optional[list]:
        """Extract audio data from messages.

        Args:
            messages: Conversation messages
            sample_rate: Target sample rate for audio

        Returns:
            List of audio data (format TBD)
        """
        pass
```

### 4.4 DataProto Extension

```python
# Ensure DataProto can handle audio data
class DataProto:
    batch: TensorDict
    non_tensor_batch: dict[str, Any]  # Should handle audio arrays
    meta_info: dict[str, Any]
```

---

## Phase 5: Validation Plan

### 5.1 Test Matrix

| Component | Test Case | Expected Result |
|-----------|-----------|-----------------|
| Rollout | Text-only generation | Pass |
| Rollout | Text + Image input | Pass |
| Rollout | Text + Audio input | Pass |
| Rollout | Text + Video input | Pass |
| Rollout | All modalities combined | Pass |
| Training | Single-step optimization | Loss decreases |
| Training | Multi-step PPO loop | Rewards improve |
| End-to-end | Quick demo with mock data | Pipeline completes |

### 5.2 Validation Steps

1. **Unit Tests**
   - Audio processing functions
   - Omni server generate method
   - Multimodal data collation

2. **Integration Tests**
   - Rollout with test prompts
   - Training step with dummy batch

3. **End-to-End Test**
   ```bash
   # Quick demo validation
   python examples/qwen3_omni/quickstart.py \
       --model Qwen/Qwen3-Omni \
       --backend vllm-omni \
       --algorithm ppo \
       --max-steps 10
   ```

---

## Implementation Timeline

| Phase | Tasks | Estimated Time | Dependencies |
|-------|-------|----------------|--------------|
| **Phase 1** | Rollout integration | 2-3 weeks | vLLM-Omni availability |
| **Phase 2** | Training backend | 2-3 weeks | Phase 1, backend decision |
| **Phase 3** | RL algorithm | 1-2 weeks | Phase 2, reward model design |
| **Phase 4** | Audio data handling | 1 week | Phase 1 |
| **Phase 5** | Validation | 1 week | All phases |

**Total Estimated Time**: 7-10 weeks

---

## Open Questions

1. **vLLM-Omni API**
   - What is the exact API for multimodal inputs?
   - Is there official documentation?
   - Audio input format requirements?

2. **Training Backend**
   - Performance comparison between FSDP and Megatron?
   - Memory requirements for Qwen3-Omni?

3. **Audio Preprocessing**
   - What audio features does Qwen3-Omni expect?
   - Maximum audio length supported?
   - How to handle variable-length audio?

4. **RL Algorithm**
   - Which algorithm is most suitable for omni-modal tasks?
   - How to design effective rewards for multimodal outputs?

---

## References

- [Qwen3-Omni Model Card](https://huggingface.co/Qwen/Qwen3-Omni) (TBD)
- [vLLM-Omni Documentation](https://github.com/vllm-project/vllm) (TBD)
- [veRL Multimodal Support](../experimental/agent_loop/)
- [Agent Loop Architecture](../experimental/agent_loop/agent_loop.py)

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-03-03 | - | Initial RFC draft |