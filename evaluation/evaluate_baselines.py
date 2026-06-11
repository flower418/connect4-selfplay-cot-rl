from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.env import Board, board_from_rows, legal_moves
from connect4.oracle import evaluate_position
from data_pipeline.io import read_jsonl, write_jsonl


Policy = Callable[[Board, int], int]


def evaluate_baselines(input_path: str, output_path: str, seed: int = 20260611) -> int:
    rng = random.Random(seed)
    policies: Dict[str, Policy] = {
        "random": lambda board, player: rng.choice(legal_moves(board)),
        "center_first": center_first_policy,
        "minimax_depth2": lambda board, player: oracle_policy(board, player, depth=2),
        "minimax_depth4": lambda board, player: oracle_policy(board, player, depth=4),
    }
    records = []
    for row in read_jsonl(input_path):
        board = board_from_rows(row["board"])
        player = int(row["player_to_move"])
        for policy_name, policy in policies.items():
            action = policy(board, player)
            records.append(_score_action(row, policy_name, action))
    write_jsonl(output_path, records)
    return len(records)


def _score_action(row: dict, policy_name: str, action: int) -> dict:
    legal = action in row["legal_moves"]
    move_values = {int(k): v for k, v in row["oracle_move_values"].items()}
    best_value = row["oracle_value"]
    chosen_value = move_values.get(action)
    value_regret = 2.0 if chosen_value is None else best_value - chosen_value
    return {
        "policy": policy_name,
        "eval_id": row["eval_id"],
        "split": row["split"],
        "action": action,
        "is_legal": legal,
        "is_oracle_best": action in row["oracle_best_moves"],
        "oracle_value": best_value,
        "chosen_value": chosen_value,
        "value_regret": value_regret,
    }


def summarize_results(results_path: str) -> dict:
    counts = defaultdict(Counter)
    regret_sum = defaultdict(float)
    for row in read_jsonl(results_path):
        key = (row["policy"], row["split"])
        counts[key]["total"] += 1
        counts[key]["legal"] += int(row["is_legal"])
        counts[key]["best"] += int(row["is_oracle_best"])
        regret_sum[key] += row["value_regret"]
    summary = {}
    for key, counter in sorted(counts.items()):
        total = counter["total"]
        summary[f"{key[0]}::{key[1]}"] = {
            "total": total,
            "legal_rate": counter["legal"] / total if total else 0.0,
            "oracle_best_acc": counter["best"] / total if total else 0.0,
            "mean_value_regret": regret_sum[key] / total if total else 0.0,
        }
    return summary


def center_first_policy(board: Board, player: int) -> int:
    for move in [3, 2, 4, 1, 5, 0, 6]:
        if move in legal_moves(board):
            return move
    raise ValueError("no legal moves")


def oracle_policy(board: Board, player: int, depth: int) -> int:
    return evaluate_position(board, player, depth=depth).best_moves[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/frozen_benchmark.jsonl")
    parser.add_argument("--output", default="data/metrics/baseline_results.jsonl")
    args = parser.parse_args()
    count = evaluate_baselines(args.input, args.output)
    print(f"wrote {count} baseline results to {args.output}")
    for key, value in summarize_results(args.output).items():
        print(key, value)


if __name__ == "__main__":
    main()
