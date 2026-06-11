# Server Setup

## Conda Environment

Create the benchmark environment:

```bash
conda env create -f environment.yml
conda activate connect4-cot-rl
```

Or create manually:

```bash
conda create -n connect4-cot-rl python=3.10 -y
conda activate connect4-cot-rl
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build if your server does not use the `environment.yml` route.

## Qwen Benchmark

Build the frozen benchmark once:

```bash
python3 evaluation/build_frozen_benchmark.py \
  --target-per-split 500 \
  --max-empty 14 \
  --max-candidates 1000000 \
  --progress-every 1000
```

Run rule baselines:

```bash
python3 evaluation/evaluate_baselines.py \
  --input data/eval/frozen_benchmark.jsonl \
  --output data/metrics/baseline_results.jsonl
```

Run local Qwen weights:

```bash
python3 evaluation/evaluate_hf_model.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --input data/eval/frozen_benchmark.jsonl \
  --output data/metrics/qwen25_05b_base.jsonl
```

Summarize:

```bash
python3 evaluation/summarize_results.py \
  --input data/metrics/qwen25_05b_base.jsonl \
  --actor-key model
```

## verl

After the benchmark is frozen and the base model has been evaluated, install verl:

```bash
pip install -r requirements-verl.txt
```

Then run SFT:

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
bash scripts/run_verl_sft.sh
```

## Regenerate 10k Seed SFT

```bash
export DEEPSEEK_MODEL=deepseek-v4-flash

python3 seed/build_large_seed_sft.py \
  --target-sft 10000 \
  --oracle-games 2500 \
  --batch-size 1000 \
  --concurrency 100 \
  --max-tokens 1800
```
