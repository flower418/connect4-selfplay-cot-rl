from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.strong_oracle import solve_position
from data_pipeline.io import read_jsonl, write_jsonl


def audit_verified_seed(
    input_path: str,
    audit_output_path: str,
    filtered_output_path: str,
    max_empty: int = 18,
) -> dict:
    audit_records = []
    filtered_records = []
    counts = Counter()

    for row in read_jsonl(input_path):
        raw = row["raw"]
        board = tuple(tuple(r) for r in raw["board_before"])
        player = int(raw["player_to_move"])
        action = raw["parsed_action"]
        empty = sum(cell == 0 for line in board for cell in line)
        audit = {
            "canonical_id": row["canonical_id"],
            "game_id": raw["game_id"],
            "parsed_action": action,
            "empty_cells": empty,
            "status": "unknown",
        }
        try:
            strong = solve_position(board, player, max_empty=max_empty)
        except ValueError as exc:
            audit["status"] = "too_many_empty_cells"
            audit["reason"] = str(exc)
            counts["too_many_empty_cells"] += 1
            audit_records.append(audit)
            continue

        audit["status"] = "solved"
        audit["strong_best_moves"] = strong.best_moves
        audit["strong_value"] = strong.value
        audit["strong_move_values"] = strong.move_values
        audit["strong_nodes"] = strong.nodes
        audit["is_strong_best"] = action in strong.best_moves
        audit_records.append(audit)

        counts["solved"] += 1
        if action in strong.best_moves:
            counts["strong_best"] += 1
            row.setdefault("metadata", {})
            row["metadata"]["strong_oracle_audit"] = audit
            filtered_records.append(row)
        else:
            counts["rejected_not_strong_best"] += 1

    write_jsonl(audit_output_path, audit_records)
    write_jsonl(filtered_output_path, filtered_records)
    return {
        "input": sum(counts.values()) - counts["strong_best"] - counts["rejected_not_strong_best"] + counts["solved"],
        "solved": counts["solved"],
        "strong_best": counts["strong_best"],
        "rejected_not_strong_best": counts["rejected_not_strong_best"],
        "too_many_empty_cells": counts["too_many_empty_cells"],
        "filtered": len(filtered_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/seed/seed_positions_verified.jsonl")
    parser.add_argument("--audit-output", default="data/seed/seed_positions_strong_audit.jsonl")
    parser.add_argument("--filtered-output", default="data/seed/seed_positions_verified_strong.jsonl")
    parser.add_argument("--max-empty", type=int, default=18)
    args = parser.parse_args()
    summary = audit_verified_seed(args.input, args.audit_output, args.filtered_output, args.max_empty)
    print(summary)


if __name__ == "__main__":
    main()
