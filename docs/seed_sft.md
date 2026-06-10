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
3. call `deepseek-v4-pro` for chain-of-thought generation
4. parse model output into `analysis + final column`
5. verify legality / oracle quality / tactical claims
6. export clean SFT records

The template-based seed builder remains only as a local fallback baseline.

## Recommended near-term usage

Run:

```bash
python3 seed/generate_seed_positions.py
python3 seed/collect_seed_cot.py
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
DEEPSEEK_MODEL=deepseek-v4-pro
```
