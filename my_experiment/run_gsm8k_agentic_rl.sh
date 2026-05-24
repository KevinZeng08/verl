#!/usr/bin/env bash
set -xeuo pipefail

# ============== 环境 ==============
cd /root/rl_research/verl
source .venv/bin/activate
export VLLM_USE_V1=1

# ============== 路径 ==============
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE="${HOME}/data/gsm8k/train.parquet"
TEST_FILE="${HOME}/data/gsm8k/test.parquet"
FUNCTION_TOOL_PATH="${HOME}/rl_research/verl/my_experiment/gsm8k_tools.py"
CKPTS_DIR="${HOME}/rl_research/verl/my_experiment/checkpoints"

# ============== 算法参数 ==============
adv_estimator=grpo
max_prompt_length=2048
max_response_length=4096

# ============== GPU 分配（8×H20）==============
# 4卡 rollout（vLLM TP=4） + 4卡 training（FSDP2）
n_gpus_rollout=4
n_gpus_training=4
gen_tp=4
fsdp_size=4

# ============== 异步训练参数 ==============
n_resp_per_prompt=16
gen_prompt_bsz=1
ppo_mini_batch_size=16
# total_train_steps = total_rollout_steps / (ppo_mini_batch_size × trigger_parameter_sync_step)
total_rollout_steps=$((64*50))
test_freq=10
staleness_threshold=0.5
trigger_parameter_sync_step=4

# ============== 启动 ==============
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.return_raw_chat=True \
    data.train_batch_size=0 \
    data.gen_batch_size=$gen_prompt_bsz \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 )) \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 4 )) \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$gen_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.function_tool_path=$FUNCTION_TOOL_PATH \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    critic.strategy=fsdp2 \
    reward.reward_manager.name=dapo \
    trainer.logger=['console','wandb'] \
    trainer.project_name='gsm8k-agentic-rl' \
    trainer.experiment_name='qwen25-7b-gsm8k-tool-agent' \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$n_gpus_training \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=$n_gpus_rollout \
    rollout.total_rollout_steps=$total_rollout_steps \
    trainer.total_epochs=2 \
    trainer.test_freq=$test_freq \
    async_training.staleness_threshold=$staleness_threshold \
    async_training.trigger_parameter_sync_step=$trigger_parameter_sync_step \
    async_training.partial_rollout=True
