#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./Qwen2.5-0.5B-Instruct}"
GRPO_SOURCE="${GRPO_SOURCE:-data/seed/seed_positions_verified_10k.jsonl}"
GRPO_INPUT="${GRPO_INPUT:-data/train/seed_grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/seed_grpo_qwen25_05b}"
PROJECT_NAME="${PROJECT_NAME:-connect4-cot-rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed-grpo-qwen25-05b}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
LOGGER="${LOGGER:-[\"console\",\"wandb\"]}"
LR="${LR:-}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:-25}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-false}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
MIN_MOVE_QUALITY="${MIN_MOVE_QUALITY:-good}"
INCLUDE_SPLITS="${INCLUDE_SPLITS:-train}"
REWARD_MODEL="${REWARD_MODEL:-connect4_verifier}"
REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-training/connect4_reward.py}"
REWARD_FUNCTION_NAME="${REWARD_FUNCTION_NAME:-compute_score}"

python training/build_grpo.py \
  --input "${GRPO_SOURCE}" \
  --output "${GRPO_INPUT}" \
  --include-splits ${INCLUDE_SPLITS} \
  --min-move-quality "${MIN_MOVE_QUALITY}"

# verl 0.8.x PPO/GRPO uses Ray workers internally. Do not wrap this entrypoint
# in torchrun; torchelastic rendezvous env vars leak into Ray workers and can
# make rank 0 wait forever on a non-listening TCPStore.
python -m verl.trainer.main_ppo_sync \
  data.train_files="${GRPO_INPUT}" \
  data.val_files="${GRPO_INPUT}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.prompt_key=prompt \
  data.max_prompt_length="${MAX_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.truncation=error \
  data.return_raw_chat=true \
  data.return_raw_input_ids=false \
  +engine.use_torch_compile="${USE_TORCH_COMPILE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  +trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
  trainer.logger="${LOGGER}" \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  reward.reward_model.enable=false \
  reward.reward_manager.source=register \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
  reward.custom_reward_function.name="${REWARD_FUNCTION_NAME}" \
  ${LR:+optim.lr="${LR}"}
