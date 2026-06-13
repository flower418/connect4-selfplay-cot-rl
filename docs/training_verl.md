# verl SFT / GRPO Training

## Goal

Run cold-start SFT on `Qwen2.5-0.5B-Instruct` using the verified seed dataset, then reuse the same verified positions for GRPO prompt export.

## Data

Source:

```text
data/train/seed_sft.jsonl
```

Export for verl:

```bash
python3 training/export_verl_sft.py \
  --input data/train/seed_sft.jsonl \
  --output data/train/seed_sft_verl.parquet
```

For a dependency-light sanity check:

```bash
python3 training/export_verl_sft.py \
  --input data/train/seed_sft.jsonl \
  --output data/train/seed_sft_verl.jsonl
```

Parquet export requires `pandas` and `pyarrow`.

The exported rows contain:

- `messages`: user / assistant chat turns
- `data_source`
- `ability`
- `extra_info`

## Run

Set the local base model path:

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
```

Run:

```bash
bash scripts/run_verl_sft.sh
```

If your installed verl version uses a different SFT entrypoint, edit the single module line in `scripts/run_verl_sft.sh`:

```bash
python3 -m verl.trainer.fsdp_sft_trainer
```

Keep `data/train/seed_sft.jsonl` fixed for the first baseline comparison.

## GRPO

Export GRPO prompts from verified positions:

```bash
python3 training/build_grpo.py \
  --input data/seed/seed_positions_verified_10k.jsonl \
  --output data/train/seed_grpo.jsonl \
  --include-splits train \
  --min-move-quality good
```

Run:

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
bash scripts/run_verl_grpo.sh
```

For verl 0.8.x, GRPO/PPO rollout is served by an async inference backend.
Install and select one of `vllm`, `sglang`, or `trtllm`; the script defaults to
`ROLLOUT_NAME=vllm`. On a single-GPU machine keep rollout tensor parallelism at
one:

```bash
export ROLLOUT_NAME=vllm
export ROLLOUT_TP_SIZE=1
```

The GRPO trainer starts Ray workers itself, so `scripts/run_verl_grpo.sh` invokes
`python -m verl.trainer.main_ppo_sync` directly. Do not wrap this entrypoint in
`torchrun`; torch elastic rendezvous environment variables can leak into Ray
workers and make initialization hang before the first training step.

Default GRPO input is position-level JSONL with:

- `prompt`
- `reward_model.ground_truth`
- `extra_info`

The script assumes a verl PPO/GRPO trainer module is available at
`verl.trainer.main_ppo_sync`; if your installed verl release uses a different
module name, change that single line in `scripts/run_verl_grpo.sh`.
