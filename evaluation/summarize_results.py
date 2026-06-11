from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.io import read_jsonl


def summarize(path: str, actor_key: str) -> dict:
    counts = defaultdict(Counter)
    regret_sum = defaultdict(float)
    for row in read_jsonl(path):
        actor = row.get(actor_key) or row.get("policy") or row.get("model")
        key = (actor, row["split"])
        counts[key]["total"] += 1
        counts[key]["format"] += int(row.get("format_success", True))
        counts[key]["legal"] += int(row.get("is_legal", False))
        counts[key]["best"] += int(row.get("is_oracle_best", False))
        if row.get("value_regret") is not None:
            regret_sum[key] += row["value_regret"]
    output = {}
    for key, counter in sorted(counts.items()):
        total = counter["total"]
        output[f"{key[0]}::{key[1]}"] = {
            "total": total,
            "format_success_rate": counter["format"] / total if total else 0.0,
            "legal_rate": counter["legal"] / total if total else 0.0,
            "oracle_best_acc": counter["best"] / total if total else 0.0,
            "mean_value_regret": regret_sum[key] / total if total else 0.0,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--actor-key", default="policy")
    args = parser.parse_args()
    for key, value in summarize(args.input, args.actor_key).items():
        print(key, value)


if __name__ == "__main__":
    main()
