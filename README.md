# connect4-selfplay-cot-rl

Four-in-a-row self-play CoT-RL project scaffold.

## Data Pipeline Architecture

The first project phase is the cold-start data pipeline for `Qwen2.5-0.5B-Instruct` SFT.

The design separates truth, language generation, parsing, verification, and export:

```text
seed candidates
  -> oracle labels
  -> DeepSeek v4 Pro CoT generation
  -> response parser
  -> rule verifier
  -> SFT JSONL export
```

Core rule: DeepSeek writes the reasoning text, but it does not define the correct move. The oracle supplies the target action and tactical labels; the verifier decides whether a generated sample is usable. Large seed generation uses `deepseek-v4-flash`.

Current cold-start artifacts:

```text
data/seed/seed_positions_candidates.jsonl   546 candidate positions
data/seed/seed_positions_raw.jsonl          217 DeepSeek-generated raw samples
data/seed/seed_positions_verified.jsonl     217 verified samples
data/train/seed_sft.jsonl                   217 SFT records
```

All verified records in the current set have `move_quality=best`.

## Pipeline Components

```text
connect4/env.py                  rules, legal moves, winners, canonical IDs
connect4/oracle.py               minimax/alpha-beta oracle labels
seed/generate_seed_positions.py  candidate position generation
seed/collect_seed_cot.py         DeepSeek v4 Pro CoT collection
seed/parse_seed_responses.py     final-move and analysis parser
seed/build_seed_verified.py      raw -> verified conversion
verification/cleaner.py          legality, move quality, faithfulness labels
training/build_sft.py            verified -> SFT JSONL export
```

The current collection path is:

```bash
python3 seed/generate_seed_positions.py --oracle-games 80
python3 seed/collect_seed_cot.py --limit 220 --concurrency 20 --max-tokens 3000
python3 seed/build_seed_verified.py \
  --input data/seed/seed_positions_raw.jsonl \
  --output data/seed/seed_positions_verified.jsonl
python3 training/build_sft.py \
  --input data/seed/seed_positions_verified.jsonl \
  --output data/train/seed_sft.jsonl
```

## Progress

### Done

- initialized repository and tracked the source project document
- implemented the Connect4 rules engine and board canonicalization
- implemented a first-pass oracle with minimax + alpha-beta
- implemented verified sample schemas and JSONL IO
- implemented first-pass verification and `faithfulness_score_v1`
- implemented SFT and GRPO export builders
- added baseline tests for rules, canonicalization, and oracle immediate-win behavior
- defined the formal cold-start direction as `oracle truth + DeepSeek CoT + verifier + SFT`
- collected 217 verified cold-start SFT samples with `deepseek-v4-pro`

### In progress

- self-play raw generation pipeline
- verl reward integration for GRPO

### Next

- run first SFT training against `data/train/seed_sft.jsonl`
- add self-play sample generator
- add verl GRPO reward integration
- freeze benchmark evaluation before comparing trained models

## Repository layout

```text
connect4/        game rules and oracle
data/            generated datasets and exports
data_pipeline/   schemas, IO, prompt helpers
docs/            project notes and pipeline docs
seed/            cold-start seed collection pipeline
tests/           unit tests
training/        dataset exporters for SFT / GRPO
verification/    rule-based cleaning and labeling
evaluation/      frozen benchmark and rule baselines
scripts/         server-side run scripts
```

## Local-only files

These stay untracked:

```text
.env.local
__pycache__/
.pytest_cache/
```

## Generated data

Current generated artifacts live under:

```text
data/
  seed/
    seed_positions_candidates.jsonl
    seed_positions_raw.jsonl
    seed_positions_verified.jsonl
  train/
    seed_sft.jsonl
```

## Source document

The initial project specification is tracked in `docs/project_spec.md`.

## Evaluation

The benchmark protocol is documented in `docs/evaluation.md`.

Current commands:

```bash
python3 evaluation/build_frozen_benchmark.py --target-per-split 500 --max-empty 14 --max-candidates 1000000
python3 evaluation/evaluate_baselines.py
python3 evaluation/evaluate_hf_model.py --model Qwen/Qwen2.5-0.5B-Instruct
```

## SFT With verl

The SFT training path is documented in `docs/training_verl.md`.

```bash
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
bash scripts/run_verl_sft.sh
```

## Regenerate 10k Seed SFT

```bash
export DEEPSEEK_MODEL=deepseek-v4-flash
python3 seed/build_large_seed_sft.py --target-sft 10000 --batch-size 1000 --concurrency 100
```

## Server Setup

Environment setup and benchmark commands are documented in `docs/server_setup.md`.
