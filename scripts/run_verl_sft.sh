#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./Qwen2.5-0.5B-Instruct}"
SFT_INPUT="${SFT_INPUT:-data/train/seed_sft_10k.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-data/train/seed_sft_verl.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/seed_sft_qwen25_05b}"
PROJECT_NAME="${PROJECT_NAME:-connect4-cot-rl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed-sft-qwen25-05b}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LOGGER="${LOGGER:-[\"console\",\"wandb\"]}"
LR="${LR:-}"

python3 training/export_verl_sft.py \
  --input "${SFT_INPUT}" \
  --output "${TRAIN_FILE}"

# Parquet export needs pandas + pyarrow in the server environment.
# The exact verl SFT entrypoint can differ by installed version.
# If your server uses a different module name, change only the next line.
python3 -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TRAIN_FILE}" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.micro_batch_size="${MICRO_BATCH_SIZE}" \
  data.max_length="${MAX_LENGTH}" \
  model.partial_pretrain="${MODEL_PATH}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.logger="${LOGGER}" \
  ${LR:+optim.lr="${LR}"}
