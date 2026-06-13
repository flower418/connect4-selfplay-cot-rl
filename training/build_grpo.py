from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.io import read_jsonl, write_jsonl
from data_pipeline.schemas import GRPOPromptRecord, VerifiedSample


def build_grpo_prompts(
    input_path: str,
    output_path: str,
    *,
    include_splits: Sequence[str] = ("train",),
    min_move_quality: str = "good",
) -> int:
    split_set = set(include_splits)
    quality_rank = {"illegal": 0, "blunder": 1, "neutral": 2, "good": 3, "best": 4}
    min_rank = quality_rank[min_move_quality]
    prompts = []
    for row in read_jsonl(input_path):
        sample = _verified_from_dict(row)
        if sample.split not in split_set:
            continue
        if quality_rank.get(sample.move_quality, -1) < min_rank:
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
                        "split": sample.split,
                        "oracle_best_moves": sample.oracle_best_moves,
                        "oracle_value_before": sample.oracle_value_before,
                        "oracle_value_after": sample.oracle_value_after,
                        "tactical_tags": sample.tactical_tags,
                        "move_quality": sample.move_quality,
                        "cot_action_consistent": sample.cot_action_consistent,
                        "cot_mentions_immediate_win": sample.cot_mentions_immediate_win,
                        "cot_mentions_must_block": sample.cot_mentions_must_block,
                        "faithfulness_score_v1": sample.faithfulness_score_v1,
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
    parser.add_argument("--include-splits", nargs="+", default=["train"])
    parser.add_argument(
        "--min-move-quality",
        default="good",
        choices=["illegal", "blunder", "neutral", "good", "best"],
    )
    args = parser.parse_args()
    count = build_grpo_prompts(
        args.input,
        args.output,
        include_splits=args.include_splits,
        min_move_quality=args.min_move_quality,
    )
    print(f"wrote {count} GRPO prompts to {Path(args.output)}")


if __name__ == "__main__":
    main()
