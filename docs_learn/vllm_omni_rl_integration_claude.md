# vLLM-Omni 集成到 verl RL Pipeline 的现状与未来工作分析

## 一、背景与动机

verl 原本只支持 LLM（语言模型）的 RL 训练。随着多模态生成模型（如 Qwen-Image、Stable Diffusion 3.5、Wan2.2 等扩散模型）的兴起，社区希望将 verl 的 RL 能力扩展到图像/视频生成领域。核心算法是 **FlowGRPO**（Flow-based GRPO），这是一种适配扩散模型的 GRPO 变体。

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) 是 vLLM 的多模态扩展，支持扩散模型推理，被选为 rollout 引擎。

## 二、相关 PR / Issue 全景

| 编号 | 类型 | 标题 | 状态 | 作者 | 关键内容 |
|------|------|------|------|------|----------|
| #4557 | RFC Issue | Support Diffusion Generative Models and Async Reward during Rollout | Open | chenyingshu | 顶层 RFC，提出扩散模型支持 + 异步 reward 计算 |
| #4639 | RFC Issue | Support Qwen-Image Flow-GRPO Training based on vLLM-Omni | Open | chenyingshu | 具体方案设计，定义了完整的组件架构 |
| #4977 | PR | Add vllm-omni for rollout (spmd, legacy) | **Closed** | zhtmike | 早期尝试，39 文件，CLA 未签署 + 代码问题，未合入 |
| #5297 | PR | Support FlowGRPO training for QwenImage | **Open (Draft)** | zhtmike | 当前主力 PR，53 文件，已有训练曲线验证 |
| #3297 | PR | Add support for qwen-2.5-omni and audio input | Open (WIP) | TomQunChao | 音频模态支持，与 vllm-omni 相关但侧重不同 |
| #5281 | Issue | verl 不支持 qwen3-omni 模型的 GRPO 训练 | Open | LiXinYuECNU | mbridge 缺少 qwen3_omni_moe 注册，Megatron 后端不支持 |

## 三、当前架构（基于 PR #5297 + 本地 verl-omni 分支）

整体数据流：

```
Prompt → vLLM-Omni Rollout (生成图像 + latents/timesteps/log_prob)
    → Reward Model (OCR/GenEval/CLIP 等打分)
    → FlowGRPO Advantage 计算
    → FSDP Diffusers Engine 更新扩散模型权重
    → 权重同步回 vLLM-Omni → 下一轮 rollout
```

已实现的核心组件：

### 1. Rollout 层

- `vLLMOmniServerAdapter` — 与 vLLM-Omni 服务端的异步通信适配器，支持 ZMQ IPC / 共享内存权重同步
- `vLLMOmniHttpServer` — HTTP 服务封装，支持 TP 并行推理、LoRA 权重更新、sleep/wake_up 生命周期管理
- `vLLMOmniReplica` — 多节点服务编排
- `DiffusionAgentLoopWorker` + `DiffusionSingleTurnAgentLoop` — 扩散模型的 agent loop

### 2. 训练引擎

- `DiffusersFSDPEngine` — 基于 FSDP2 的扩散模型训练引擎，支持 LoRA、activation offloading、混合精度、Ulysses 序列并行

### 3. Trainer

- `RayFlowGRPOTrainer` — Ray 单控制器架构的 FlowGRPO 训练器，约 1500 行

### 4. 算法

- `compute_flow_grpo_outcome_advantage()` — FlowGRPO 优势估计
- `compute_policy_loss_flow_grpo()` — 带 clip 的策略损失，梯度只通过 log_prob 流动

### 5. Reward

- `DiffusionRewardLoopManager` / `DiffusionRewardLoopWorker` — 扩散模型 reward 计算管理
- 已支持 Qwen2.5-VL OCR reward

### 6. 数据与配置

- T2I 生成 RL 的 Dataset & Dataloader
- wandb logger 支持
- `ppo_diffusion_trainer.yaml` 完整配置

## 四、当前存在的问题与不足

### 1. PR #5297 仍为 Draft 状态

- 作者自述包含冗余代码/配置，尚未 ready for review
- 自动 review 指出测试中存在硬编码路径，CI 可移植性差
- 53 个文件改动量大，review 负担重

### 2. 与主线架构的耦合度

- `RayFlowGRPOTrainer` 是独立的 ~1500 行 trainer，与现有 `RayPPOTrainer` 大量重复
- 扩散模型 pipeline 与 LLM pipeline 完全分离（按 RFC #4639 的设计意图），但这也意味着后续维护成本高

### 3. 模型覆盖有限

- 目前只验证了 Qwen-Image，尚未覆盖 SD-3.5、Wan2.2 等其他扩散模型
- Qwen3-Omni（Issue #5281）在 Megatron 后端因 mbridge 未注册而完全不可用
- 音频模态（PR #3297）仍在 WIP

### 4. 异步 Reward 计算未实现

- RFC #4557 和 #4639 都提到了 rollout 阶段的异步 reward 计算作为加速手段，但 checklist 中标记为 TBD

### 5. 部署模式验证不充分

- Colocated / Standalone / Hybrid 三种模式的实际验证情况不明
- SPMD 模式在 PR #4977 中尝试过但随 PR 关闭

## 五、未来可做的工作

### 短期（推动 PR #5297 合入）

1. **代码精简与去重** — 消除 `RayFlowGRPOTrainer` 与 `RayPPOTrainer` 的重复逻辑，考虑通过继承或 mixin 复用
2. **测试可移植性修复** — 消除硬编码路径，使 CI 可运行
3. **配置规范化** — 清理冗余配置文件，与现有 config 体系对齐

### 中期（功能扩展）

4. **异步 Reward 计算** — 实现 rollout 阶段的异步 batch reward 计算（RFC #4557 的核心提议之一），可显著加速训练
5. **更多扩散模型支持** — 扩展到 SD-3.5、Wan2.2 等模型，验证 `DiffusersFSDPEngine` 的通用性
6. **Qwen3-Omni 支持** — 在 mbridge 中注册 `qwen3_omni_moe` 类型，打通 Megatron 后端（Issue #5281）
7. **更多 RL 算法** — 在 FlowGRPO 基础上集成 DiffusionNFT、DanceGRPO 等算法
8. **SPMD rollout 模式** — PR #4977 关闭后 SPMD 模式丢失，需要重新实现

### 长期（架构演进）

9. **统一多模态 RL Pipeline** — 将 LLM 和扩散模型的 RL pipeline 在更高层次统一，减少维护成本。当前完全分离的设计虽然避免了破坏 LLM 行为，但长期不可持续
10. **视频生成 RL** — 从图像扩展到视频生成（Wan2.2 等），需要处理更长的序列和更大的显存压力
11. **音频模态集成** — 与 PR #3297 的 Qwen-2.5-Omni 音频支持合流，实现真正的全模态 RL
12. **Reward Model 生态** — 扩展 reward 类型（GenEval、CLIP Score、人类偏好模型等），构建可插拔的 reward 框架
13. **与 vLLM-Omni 上游同步** — 跟进 vLLM-Omni 的 API 演进，保持兼容性

## 六、总结

vLLM-Omni 集成到 verl 的工作由 chenyingshu / zhtmike 团队主导，已经完成了从 RFC 设计到可运行原型的全流程。PR #5297 是当前的主力实现，包含完整的 rollout → reward → advantage → update 闭环，并已有 Qwen-Image OCR 任务的训练曲线验证。但该 PR 仍处于 Draft 阶段，代码质量和测试覆盖需要进一步打磨。最大的未完成项是异步 reward 计算和更广泛的模型/算法覆盖。社区对此方向有明确需求（多人在 issue 中表达了参与意愿），是 verl 向多模态生成 RL 扩展的关键路径。
