# connect4-selfplay-cot-rl

Four-in-a-row self-play CoT-RL project scaffold.

## Current scope

This repository starts from the project specification in `docs/project_spec.md`.

The first implementation target is the data pipeline:

- raw self-play game logging
- verification and labeling
- SFT dataset building
- DPO pair building
- eval split freezing and leakage prevention

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

### In progress

- DeepSeek-backed seed SFT cold-start data pipeline
- self-play raw generation pipeline
- verl reward integration for GRPO

### Next

- build seed position corpus and verified seed labels
- export seed SFT training set for `Qwen2.5-0.5B-Instruct`
- add response parser and self-play sample generator

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
