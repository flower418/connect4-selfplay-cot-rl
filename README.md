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

## Planned repository layout

```text
connect4/
generation/
verification/
training/
evaluation/
docs/
data/
```

## Source document

The initial project specification is tracked in `docs/project_spec.md`.
