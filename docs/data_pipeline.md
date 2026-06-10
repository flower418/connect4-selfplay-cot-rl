# Data Pipeline Notes

## Work log

### Implemented

- canonical board state representation in `connect4/env.py`
- mirror-stable `canonical_id` for dedup and eval leakage filtering
- minimax + alpha-beta oracle in `connect4/oracle.py`
- raw / verified / SFT / GRPO schemas in `data_pipeline/schemas.py`
- first-pass verifier in `verification/cleaner.py`
- JSONL export builders for SFT and GRPO in `training/`
- baseline unit tests in `tests/test_env.py`

### Not implemented yet

- self-play generation runner
- response parser from free-form model text
- eval split freezer
- full `verl` reward function and rollout integration
- full faithfulness metric beyond `v1`
- DeepSeek-backed formal seed collection was selected and is being implemented with `deepseek-v4-pro`

## Training route

- base model: `Qwen2.5-0.5B-Instruct`
- training route: `Seed SFT -> iterative RS-SFT -> GRPO`
- training system target: `verl`

## Design choices

### 1. Canonical internal board format

Use a `6x7` integer matrix:

- `0`: empty
- `1`: current player side
- `-1`: opponent side

This is the source of truth for:

- rule execution
- search/oracle evaluation
- mirror canonicalization
- train/eval leakage checks

### 2. Raw sample granularity

Store self-play data as one JSONL record per move, not one record per whole game.

Reason:

- verifier works on position-response pairs
- SFT examples are position-response pairs
- GRPO prompts are also position-level
- incremental filtering is simpler

### 3. IDs

- `position_id = sha256(board_before + player_to_move)`
- `canonical_id = sha256(canonical_board(board_before) + player_to_move)`

`canonical_id` is used for:

- exact + mirror dedup
- eval leakage filtering
- split freezing

### 4. Oracle meaning

`oracle` is a programmatic Connect4 search teacher, not an LLM.

First version:

- minimax
- alpha-beta pruning
- default verifier depth `4`

It provides:

- best moves
- position value
- tactical tags
- move quality labels

### 5. Faithfulness meaning

`faithfulness` measures whether the model's chain-of-thought is aligned with:

- the move it finally plays
- the real tactical structure of the board

First version only implements `faithfulness_score_v1`:

- action consistency
- immediate-win claim correctness
- must-block claim correctness
- no illegal-action hallucination

This is enough to support early ablations without overbuilding the verifier.

### 6. verl export target

Do not bind repository data to a single trainer format.

Use verified positions as the internal source of truth, then export:

- SFT JSONL
- GRPO prompt JSONL

The GRPO export carries verifier-side ground truth under `reward_model`.

Actual reward computation can later read:

- parsed action legality
- oracle best move set
- tactical tags
- faithfulness score

## Seed SFT plan

The cold-start phase should use a curated seed corpus before any self-play.

Recommended seed flow:

1. create seed positions from tactical templates and short oracle-play games
2. attach oracle labels with verifier
3. optionally replace oracle-written reasoning with stronger-model reasoning later
4. export a clean SFT dataset for the first supervised run

The repository implementation for this phase uses:

- `data/seed/seed_positions_raw.jsonl`
- `data/seed/seed_positions_verified.jsonl`
- `data/train/seed_sft.jsonl`
