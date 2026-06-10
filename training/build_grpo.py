from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.io import read_jsonl, write_jsonl
from data_pipeline.schemas import GRPOPromptRecord, VerifiedSample


def build_grpo_prompts(input_path: str, output_path: str) -> int:
    prompts = []
    for row in read_jsonl(input_path):
        sample = _verified_from_dict(row)
        if sample.split != "train":
            continue
        prompts.append(
            GRPOPromptRecord(
                data_source="connect4_verified",
                prompt=[{"role": "user", "content": sample.raw.prompt}],
                reward_model={
                    "style": "connect4_verifier",
                    "ground_truth": {
                        "position_id": sample.position_id,
                        "canonical_id": sample.canonical_id,
                        "oracle_best_moves": sample.oracle_best_moves,
                        "tactical_tags": sample.tactical_tags,
                    },
                },
                extra_info={
                    "generation": sample.raw.generation,
                    "move_quality": sample.move_quality,
                    "faithfulness_score_v1": sample.faithfulness_score_v1,
                },
            ).to_dict()
        )
    write_jsonl(output_path, prompts)
    return len(prompts)


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
    count = build_grpo_prompts(args.input, args.output)
    print(f"wrote {count} GRPO prompts to {Path(args.output)}")


if __name__ == "__main__":
    main()
