from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.io import read_jsonl


def export_verl_sft(input_path: str, output_path: str) -> int:
    rows = []
    for item in read_jsonl(input_path):
        prompt = item["prompt"]
        response = item["response"]
        rows.append(
            {
                "data_source": item.get("data_source", "connect4_seed_sft"),
                "ability": item.get("ability", "connect4_reasoning"),
                "prompt": prompt,
                "response": response,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "extra_info": item.get("extra_info", {}),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas and pyarrow are required for parquet export") from exc
        pd.DataFrame(rows).to_parquet(output, index=False)
    else:
        with output.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def split_verl_sft(input_path: str, train_output: str, val_output: str, val_ratio: float = 0.1) -> tuple[int, int]:
    rows = []
    for item in read_jsonl(input_path):
        prompt = item["prompt"]
        response = item["response"]
        rows.append(
            {
                "data_source": item.get("data_source", "connect4_seed_sft"),
                "ability": item.get("ability", "connect4_reasoning"),
                "prompt": prompt,
                "response": response,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "extra_info": item.get("extra_info", {}),
            }
        )

    if not rows:
        raise ValueError("no rows found in input")

    val_count = max(1, int(len(rows) * val_ratio))
    train_rows = rows[:-val_count]
    val_rows = rows[-val_count:]

    train_path = Path(train_output)
    val_path = Path(val_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    if train_path.suffix == ".parquet" and val_path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas and pyarrow are required for parquet export") from exc
        pd.DataFrame(train_rows).to_parquet(train_path, index=False)
        pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    else:
        with train_path.open("w", encoding="utf-8") as fh:
            for row in train_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with val_path.open("w", encoding="utf-8") as fh:
            for row in val_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(train_rows), len(val_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train/seed_sft.jsonl")
    parser.add_argument("--output", default="data/train/seed_sft_verl.parquet")
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--val-output", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.train_output or args.val_output:
        if not args.train_output or not args.val_output:
            raise SystemExit("--train-output and --val-output must be provided together")
        train_count, val_count = split_verl_sft(args.input, args.train_output, args.val_output, args.val_ratio)
        print(f"wrote {train_count} train and {val_count} val verl SFT records")
    else:
        count = export_verl_sft(args.input, args.output)
        print(f"wrote {count} verl SFT records to {args.output}")


if __name__ == "__main__":
    main()
