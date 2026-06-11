#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./Qwen2.5-0.5B-Instruct}"
SFT_INPUT="${SFT_INPUT:-data/train/seed_sft_10k.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-data/train/seed_sft_verl.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/seed_sft_qwen25_05b}"
PROJECT_NAME="${PROJECT_NAME:-connect4-cot-rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed-sft-qwen25-05b}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LOGGER="${LOGGER:-[\"console\",\"wandb\"]}"
LR="${LR:-}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:-after_each_epoch}"

python3 training/export_verl_sft.py \
  --input "${SFT_INPUT}" \
  --output "${TRAIN_FILE}"

# Parquet export needs pandas + pyarrow in the server environment.
# verl 0.8.x SFT entrypoint.
python3 -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TRAIN_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  data.max_length="${MAX_LENGTH}" \
  data.messages_key=messages \
  model.path="${MODEL_PATH}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.logger="${LOGGER}" \
  ${LR:+optim.lr="${LR}"}
