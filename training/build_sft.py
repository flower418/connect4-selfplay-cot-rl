from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.io import read_jsonl, write_jsonl
from data_pipeline.schemas import SFTRecord, VerifiedSample


def build_sft_records(input_path: str, output_path: str) -> int:
    records = []
    for row in read_jsonl(input_path):
        sample = _verified_from_dict(row)
        if sample.split != "train":
            continue
        if sample.move_quality not in {"best", "good"}:
            continue
        records.append(
            SFTRecord(
                data_source="connect4_verified",
                prompt=sample.raw.prompt,
                response=sample.raw.raw_response,
                reward_model={"style": "rule"},
                extra_info={
                    "generation": sample.raw.generation,
                    "position_id": sample.position_id,
                    "canonical_id": sample.canonical_id,
                    "faithfulness_score_v1": sample.faithfulness_score_v1,
                    "move_quality": sample.move_quality,
                },
            ).to_dict()
        )
    write_jsonl(output_path, records)
    return len(records)


def _verified_from_dict(row: dict) -> VerifiedSample:
    from data_pipeline.schemas import RawMoveSample

    raw = RawMoveSample(**row["raw"])
    payload = dict(row)
    payload["raw"] = raw
    return VerifiedSample(**payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = build_sft_records(args.input, args.output)
    print(f"wrote {count} SFT records to {Path(args.output)}")


if __name__ == "__main__":
    main()
