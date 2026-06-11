# verl SFT Training

## Goal

Run cold-start SFT on `Qwen2.5-0.5B-Instruct` using the verified seed dataset.

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
