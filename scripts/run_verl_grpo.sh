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
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LOGGER="${LOGGER:-[\"console\",\"wandb\"]}"
LR="${LR:-}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:-25}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
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

# verl GRPO entrypoint. The exact module name can vary by verl release.
# Adjust the single trainer line below if your installation exposes a different module.
torchrun --standalone --nnodes="${NNODES}" --nproc_per_node="${N_GPUS_PER_NODE}" \
  -m verl.trainer.main_ppo_sync \
  data.train_files="${GRPO_INPUT}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.prompt_key=prompt \
  data.max_prompt_length="${MAX_LENGTH}" \
  data.truncation=error \
  data.return_raw_chat=true \
  data.return_raw_input_ids=false \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
  trainer.logger="${LOGGER}" \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  reward.reward_model.enable=false \
  reward.reward_manager.source=register \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
  reward.custom_reward_function.name="${REWARD_FUNCTION_NAME}" \
  ${LR:+optim.lr="${LR}"}
