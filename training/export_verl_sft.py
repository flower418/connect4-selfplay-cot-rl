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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train/seed_sft.jsonl")
    parser.add_argument("--output", default="data/train/seed_sft_verl.parquet")
    args = parser.parse_args()
    count = export_verl_sft(args.input, args.output)
    print(f"wrote {count} verl SFT records to {args.output}")


if __name__ == "__main__":
    main()
