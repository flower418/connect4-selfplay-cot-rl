# Seed SFT Cold Start

## Goal

Before self-play, build a clean supervised dataset that teaches the base model:

- legal move formatting
- basic board reading
- immediate win recognition
- must-block recognition
- simple positional preference

## Current implementation

The repository now includes a first-pass seed pipeline:

1. `seed/build_seed_raw.py`
   - emits seed move-level raw samples from curated tactical templates
   - uses the oracle to choose the target move
   - writes `data/seed/seed_positions_raw.jsonl`

2. `seed/build_seed_verified.py`
   - runs the same verifier stack used later in self-play
   - writes `data/seed/seed_positions_verified.jsonl`

3. `training/build_sft.py`
   - exports `data/train/seed_sft.jsonl`

## Recommended near-term usage

Run:

```bash
python3 seed/build_seed_raw.py
python3 seed/build_seed_verified.py
python3 training/build_sft.py \
  --input data/seed/seed_positions_verified.jsonl \
  --output data/train/seed_sft.jsonl
```

## What this version is and is not

This version is enough to unblock:

- schema finalization
- verifier integration
- first SFT dry runs

This version is not enough yet for serious cold-start quality, because it still lacks:

- 50-100 seed games
- stronger-model-written reasoning traces
- broader tactical motifs
- deduplicated mirrored variants
- held-out seed validation split

## Next quality step

The right next improvement is:

- keep the current template-based seed builder as a deterministic baseline
- add a second seed source from oracle-guided short games
- optionally rewrite responses with a stronger model while keeping oracle action labels fixed
