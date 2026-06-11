# Seed SFT Cold Start

## Goal

Before self-play, build a clean supervised dataset that teaches the base model:

- legal move formatting
- basic board reading
- immediate win recognition
- must-block recognition
- simple positional preference

## Formal implementation target

The formal seed pipeline is:

1. generate candidate positions
2. compute oracle truth labels
3. call `deepseek-v4-flash` for chain-of-thought generation
4. parse model output into `analysis + final column`
5. verify legality / oracle quality / tactical claims
6. export clean SFT records

The template-based seed builder remains only as a local fallback baseline.

## Recommended usage

Run:

```bash
python3 seed/generate_seed_positions.py
python3 seed/collect_seed_cot.py --limit 220 --concurrency 20 --max-tokens 3000
python3 seed/build_seed_verified.py
python3 training/build_sft.py \
  --input data/seed/seed_positions_verified.jsonl \
  --output data/train/seed_sft.jsonl
```

## What this version is and is not

This version is meant to unblock:

- schema finalization
- verifier integration
- formal DeepSeek-backed cold-start data collection
- first SFT dry runs against a real strong-model teacher

## Local config

Set the DeepSeek credential in a local ignored file such as `.env.local`.

Expected variables:

```bash
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-v4-flash
```

## 2026-06-11 collection note

The first large DeepSeek-backed seed run exposed two pipeline bugs:

- `deepseek-v4-pro` often spends many completion tokens in `reasoning_content`; `max_tokens=700` caused `finish_reason=length`, empty or incomplete `content`, and many false `illegal` samples.
- The batch collector aborted the whole run when one sample failed, losing successful samples.

Fixes:

- default collection token budget increased to `2200`, with retry attempts adding more tokens
- truncated, empty, or target-mismatched responses are treated as failed samples
- batch collection now writes successful samples and records failures separately
- parser accepts several equivalent final-move phrasings

Current generated set:

- candidates: 546
- raw DeepSeek samples: 217
- verified samples: 217
- SFT records: 217
- verified move quality: all `best`

## 10k generation

The formal large seed SFT target is 10k verified records using `deepseek-v4-flash`.

```bash
export DEEPSEEK_MODEL=deepseek-v4-flash

python3 seed/build_large_seed_sft.py --step candidates --oracle-games 2500
python3 seed/build_large_seed_sft.py --step collect \
  --target-sft 10000 \
  --batch-size 1000 \
  --concurrency 100 \
  --max-tokens 1800
```

Outputs:

```text
data/seed/seed_positions_candidates_10k.jsonl
data/seed/seed_positions_raw_10k.jsonl
data/seed/seed_positions_verified_10k.jsonl
data/train/seed_sft_10k.jsonl
data/seed/seed_sft_10k_manifest.jsonl
```
