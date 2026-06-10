from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.io import read_jsonl, write_jsonl
from data_pipeline.schemas import RawMoveSample
from verification.cleaner import verify_sample


def build_seed_verified(input_path: str, output_path: str) -> int:
    verified = []
    for row in read_jsonl(input_path):
        raw = RawMoveSample(**row)
        verified.append(verify_sample(raw).to_dict())
    write_jsonl(output_path, verified)
    return len(verified)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/seed/seed_positions_raw.jsonl")
    parser.add_argument("--output", default="data/seed/seed_positions_verified.jsonl")
    args = parser.parse_args()
    count = build_seed_verified(args.input, args.output)
    print(f"wrote {count} seed verified records to {Path(args.output)}")


if __name__ == "__main__":
    main()
