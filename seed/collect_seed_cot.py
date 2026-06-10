from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.env import apply_move, legal_moves, winner
from data_pipeline.io import read_jsonl, write_jsonl
from data_pipeline.prompts import build_position_prompt
from seed.deepseek_client import DeepSeekClient, DeepSeekConfig
from seed.parse_seed_responses import parse_seed_response


SYSTEM_PROMPT = (
    "你是四子棋老师，任务是为一个已知正确落子生成高质量、简洁、真实的中文分析。"
    "你必须严格遵守事实，不要编造不存在的直接赢、强制防守或非法列。"
    "输出必须只有两段：\n"
    "分析：...\n"
    "最终落子列: N"
)


def collect_seed_cot(input_path: str, output_path: str, limit: int | None = None, sleep_seconds: float = 0.2) -> int:
    client = DeepSeekClient(DeepSeekConfig.from_env())
    rows = list(read_jsonl(input_path))
    if limit is not None:
        rows = rows[:limit]

    outputs: List[Dict] = []
    for row in rows:
        prompt = _build_teacher_prompt(row)
        response = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        message = response["choices"][0]["message"]["content"]
        parsed_cot, parsed_action = parse_seed_response(message)
        board_after = None
        is_legal = parsed_action in legal_moves(tuple(tuple(r) for r in row["board"])) if parsed_action is not None else False
        if is_legal:
            board_after = apply_move(tuple(tuple(r) for r in row["board"]), row["player_to_move"], parsed_action)
        outputs.append(
            {
                "position_name": row["position_name"],
                "source": row["source"],
                "generation": 0,
                "move_index": 0,
                "player": row["player"],
                "player_to_move": row["player_to_move"],
                "board_before": row["board"],
                "legal_moves": legal_moves(tuple(tuple(r) for r in row["board"])),
                "prompt": build_position_prompt(tuple(tuple(r) for r in row["board"]), row["player_to_move"]),
                "teacher_prompt": prompt,
                "raw_response": message,
                "parsed_cot": parsed_cot,
                "parsed_action": parsed_action,
                "is_legal": is_legal,
                "board_after": board_after,
                "winner": _winner_name(board_after) if board_after is not None else None,
                "terminal": bool(board_after is not None and winner(board_after) != 0),
                "model_path": "deepseek/deepseek-v4-pro",
                "decode_config": {
                    "provider": "deepseek",
                    "model": "deepseek-v4-pro",
                    "temperature": 0.2,
                },
                "teacher_metadata": {
                    "oracle_best_moves": row["oracle_best_moves"],
                    "oracle_value": row["oracle_value"],
                    "tactical_tags": row["tactical_tags"],
                    "api_id": response.get("id"),
                },
            }
        )
        time.sleep(sleep_seconds)
    write_jsonl(output_path, outputs)
    return len(outputs)


def _build_teacher_prompt(row: Dict) -> str:
    move = row["oracle_best_moves"][0]
    tactical = "、".join(row["tactical_tags"]) if row["tactical_tags"] else "无显式立即战术标签"
    return (
        f"{build_position_prompt(tuple(tuple(r) for r in row['board']), row['player_to_move'])}\n\n"
        "请围绕一个已知正确落子写分析，不要改动最终动作。\n"
        f"已知正确落子列: {move}\n"
        f"程序战术标签: {tactical}\n"
        "要求：\n"
        "1. 分析必须与局面事实一致。\n"
        "2. 分析必须支持最终动作。\n"
        "3. 不要提不存在的直接赢、必须防守或非法规则。\n"
        f"4. 最终必须输出：最终落子列: {move}\n"
    )


def _winner_name(board_after):
    if board_after is None:
        return None
    victor = winner(board_after)
    if victor == 1:
        return "X"
    if victor == -1:
        return "O"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/seed/seed_positions_candidates.jsonl")
    parser.add_argument("--output", default="data/seed/seed_positions_raw.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    args = parser.parse_args()
    count = collect_seed_cot(args.input, args.output, args.limit, args.sleep_seconds)
    print(f"wrote {count} DeepSeek seed raw records to {args.output}")


if __name__ == "__main__":
    main()
