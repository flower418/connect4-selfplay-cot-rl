from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.env import PLAYER_ONE, Board, apply_move, canonical_position_id, legal_moves, new_board, render_board, winner
from connect4.oracle import evaluate_position
from connect4.strong_oracle import solve_position
from data_pipeline.io import read_jsonl, write_jsonl


SPLITS = {
    "late_immediate_win": 500,
    "late_must_block": 500,
    "late_forced_win": 500,
    "late_forced_draw": 500,
    "late_depth_gap": 500,
    "late_regret_sensitive": 500,
}


def build_frozen_benchmark(
    output_path: str,
    manifest_path: str,
    seed: int = 20260611,
    target_per_split: int = 500,
    max_empty: int = 14,
    max_candidates: int = 1_000_000,
    progress_every: int = 1000,
    train_paths: Iterable[str] = ("data/seed/seed_positions_verified.jsonl",),
    checkpoint_every: int = 25,
) -> dict:
    rng = random.Random(seed)
    train_ids = _load_train_canonical_ids(train_paths)
    seen = set(train_ids)
    records: List[Dict] = []
    split_counts = Counter()
    candidate_count = 0
    accepted_since_checkpoint = 0

    while min((split_counts[name] for name in SPLITS), default=0) < target_per_split and candidate_count < max_candidates:
        candidate_count += 1
        if progress_every > 0 and candidate_count % progress_every == 0:
            _print_progress(candidate_count, split_counts, target_per_split)
            _write_checkpoint(output_path, manifest_path, records, seed, target_per_split, max_empty, max_candidates, candidate_count, split_counts, train_ids)
        board, player = _sample_late_position(rng, max_empty=max_empty)
        canonical_id = canonical_position_id(board, player)
        if canonical_id in seen:
            continue
        try:
            strong = solve_position(board, player, max_empty=max_empty, full_width=True)
        except ValueError:
            continue
        split = _assign_split(board, player, strong)
        if split is None or split_counts[split] >= target_per_split:
            continue
        seen.add(canonical_id)
        split_counts[split] += 1
        records.append(_record(f"{split}_{split_counts[split]:04d}", split, board, player, strong, max_empty))
        accepted_since_checkpoint += 1
        if checkpoint_every > 0 and accepted_since_checkpoint >= checkpoint_every:
            _write_checkpoint(output_path, manifest_path, records, seed, target_per_split, max_empty, max_candidates, candidate_count, split_counts, train_ids)
            accepted_since_checkpoint = 0
        if progress_every > 0 and split_counts[split] % 50 == 0:
            _print_progress(candidate_count, split_counts, target_per_split)

    _write_checkpoint(output_path, manifest_path, records, seed, target_per_split, max_empty, max_candidates, candidate_count, split_counts, train_ids)
    return {
        "seed": seed,
        "target_per_split": target_per_split,
        "max_empty": max_empty,
        "max_candidates": max_candidates,
        "candidate_count": candidate_count,
        "total_records": len(records),
        "split_counts": dict(split_counts),
        "train_canonical_ids_excluded": len(train_ids),
        "oracle": "exact_negamax_alpha_beta_transposition",
        "complete": all(split_counts[name] >= target_per_split for name in SPLITS),
        "acceptance": "only exact-solved positions",
    }


def _write_checkpoint(
    output_path: str,
    manifest_path: str,
    records: List[Dict],
    seed: int,
    target_per_split: int,
    max_empty: int,
    max_candidates: int,
    candidate_count: int,
    split_counts: Counter,
    train_ids: set[str],
) -> None:
    write_jsonl(output_path, records)
    manifest = {
        "seed": seed,
        "target_per_split": target_per_split,
        "max_empty": max_empty,
        "max_candidates": max_candidates,
        "candidate_count": candidate_count,
        "total_records": len(records),
        "split_counts": dict(split_counts),
        "train_canonical_ids_excluded": len(train_ids),
        "oracle": "exact_negamax_alpha_beta_transposition",
        "complete": all(split_counts[name] >= target_per_split for name in SPLITS),
        "acceptance": "only exact-solved positions",
    }
    write_jsonl(manifest_path, [manifest])


def _load_train_canonical_ids(paths: Iterable[str]) -> set[str]:
    ids = set()
    for path in paths:
        try:
            for row in read_jsonl(path):
                if "canonical_id" in row:
                    ids.add(row["canonical_id"])
        except FileNotFoundError:
            continue
    return ids


def _print_progress(candidate_count: int, split_counts: Counter, target_per_split: int) -> None:
    parts = [f"{name}={split_counts[name]}/{target_per_split}" for name in SPLITS]
    print(f"[benchmark] candidates={candidate_count} " + " ".join(parts), flush=True)


def _sample_late_position(rng: random.Random, max_empty: int) -> tuple[Board, int]:
    for _ in range(500):
        board = new_board()
        player = PLAYER_ONE
        target_plies = 42 - rng.randint(2, max_empty)
        for ply in range(target_plies):
            if winner(board) != 0 or not legal_moves(board):
                break
            move = _sample_rollout_move(rng, board, player, ply)
            board = apply_move(board, player, move)
            player = -player
        empty = sum(cell == 0 for row in board for cell in row)
        if winner(board) == 0 and legal_moves(board) and empty <= max_empty:
            return board, player
    return new_board(), PLAYER_ONE


def _sample_rollout_move(rng: random.Random, board: Board, player: int, ply: int) -> int:
    moves = legal_moves(board)
    if ply < 4 and rng.random() < 0.8:
        center_order = [3, 2, 4, 1, 5, 0, 6]
        preferred = [move for move in center_order if move in moves]
        return preferred[rng.randrange(min(5, len(preferred)))]
    if rng.random() < 0.35:
        center_order = [3, 2, 4, 1, 5, 0, 6]
        preferred = [move for move in center_order if move in moves]
        return preferred[rng.randrange(len(preferred))]
    return rng.choice(moves)


def _assign_split(board: Board, player: int, strong) -> str | None:
    shallow = evaluate_position(board, player, depth=2)
    deep = evaluate_position(board, player, depth=4)
    if shallow.immediate_win_moves:
        return "late_immediate_win"
    if shallow.must_block_moves:
        return "late_must_block"
    exact_values = list(strong.move_values.values())
    if not exact_values:
        return None
    exact_best = max(exact_values)
    exact_worst = min(exact_values)
    exact_regret_span = exact_best - exact_worst
    if not set(shallow.best_moves).intersection(strong.best_moves):
        return "late_depth_gap"
    if strong.value == 1.0 and exact_regret_span >= 1.0:
        return "late_forced_win"
    if strong.value == 0.0 and exact_regret_span >= 1.0:
        return "late_forced_draw"
    deep_values = list(deep.move_values.values())
    if not deep_values:
        return None
    deep_regret_span = max(deep_values) - min(deep_values)
    if set(deep.best_moves).intersection(strong.best_moves) and deep_regret_span >= 20.0:
        return "late_regret_sensitive"
    return None


def _record(eval_id: str, split: str, board: Board, player: int, strong, max_empty: int) -> Dict:
    shallow2 = evaluate_position(board, player, depth=2)
    shallow4 = evaluate_position(board, player, depth=4)
    empty = sum(cell == 0 for row in board for cell in row)
    return {
        "eval_id": eval_id,
        "split": split,
        "board": board,
        "player_to_move": player,
        "player": "X" if player == PLAYER_ONE else "O",
        "legal_moves": legal_moves(board),
        "canonical_id": canonical_position_id(board, player),
        "prompt_board": render_board(board),
        "empty_cells": empty,
        "oracle_type": "exact",
        "oracle_max_empty": max_empty,
        "oracle_value": strong.value,
        "oracle_best_moves": strong.best_moves,
        "oracle_move_values": strong.move_values,
        "oracle_nodes": strong.nodes,
        "depth2_best_moves": shallow2.best_moves,
        "depth4_best_moves": shallow4.best_moves,
        "tactical_tags": _tags(shallow4),
    }


def _tags(evaln) -> List[str]:
    tags = []
    if evaln.immediate_win_moves:
        tags.append("immediate_win")
    if evaln.must_block_moves:
        tags.append("must_block")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/eval/frozen_benchmark.jsonl")
    parser.add_argument("--manifest", default="data/eval/frozen_benchmark_manifest.jsonl")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--target-per-split", type=int, default=500)
    parser.add_argument("--max-empty", type=int, default=14)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    manifest = build_frozen_benchmark(
        args.output,
        args.manifest,
        seed=args.seed,
        target_per_split=args.target_per_split,
        max_empty=args.max_empty,
        max_candidates=args.max_candidates,
        progress_every=args.progress_every,
    )
    print(manifest)


if __name__ == "__main__":
    main()
