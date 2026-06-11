# Frozen Benchmark Protocol

## Purpose

This benchmark is the fixed evaluation surface for baseline and trained models. Do not regenerate or edit it after training begins.

## Oracle Standard

Benchmark labels use `connect4/strong_oracle.py`:

- exact negamax search
- alpha-beta pruning
- transposition table
- only exact-solved positions are admitted

The lighter depth-limited oracle in `connect4/oracle.py` is allowed for seed data filtering and baseline comparison, but not as benchmark truth.

## Frozen Files

```text
data/eval/frozen_benchmark.jsonl
data/eval/frozen_benchmark_manifest.jsonl
```

The manifest records seed, candidate count, split counts, oracle type, and leakage exclusions.

## Splits

The benchmark is intentionally late-game and exact-solvable. It measures decision quality against a strong oracle.

- `late_immediate_win`: one-move winning opportunities
- `late_must_block`: urgent defense against immediate opponent threats
- `late_forced_win`: exact-solved winning states where wrong moves can lose value
- `late_forced_draw`: exact-solved drawn states where maintaining draw matters
- `late_depth_gap`: positions where shallow depth-2 search disagrees with the exact oracle
- `late_regret_sensitive`: states with high value spread between legal moves

Notes:

- exact oracle is run in full-width mode for benchmark records, so every legal move has a value
- `late_forced_loss_defense` was intentionally removed because many losing states give every legal move the same exact value and do not distinguish policies
- `late_depth_gap` is included to measure positions where shallow search is misleading

## Metrics

Primary:

- `format_success_rate`
- `legal_rate`
- `oracle_best_acc`
- `mean_value_regret`

Secondary:

- split-wise performance
- comparison to `random`, `center_first`, `minimax_depth2`, `minimax_depth4`

## Commands

Build once:

```bash
python3 evaluation/build_frozen_benchmark.py \
  --target-per-split 500 \
  --max-empty 14 \
  --max-candidates 1000000 \
  --progress-every 1000
```

Default final benchmark size is 500 positions per split, 3000 positions total.

Run baselines:

```bash
python3 evaluation/evaluate_baselines.py
```

Run a Hugging Face model:

```bash
python3 evaluation/evaluate_hf_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output data/metrics/qwen25_05b_base.jsonl
```

Summarize:

```bash
python3 evaluation/summarize_results.py \
  --input data/metrics/qwen25_05b_base.jsonl \
  --actor-key model
```

## Leakage Rule

The builder excludes canonical IDs from `data/seed/seed_positions_verified.jsonl`, including mirror-equivalent positions.
