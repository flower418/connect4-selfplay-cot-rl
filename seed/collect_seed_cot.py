from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

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


def collect_seed_cot(
    input_path: str,
    output_path: str,
    limit: int | None = None,
    sleep_seconds: float = 0.1,
    concurrency: int = 20,
    max_retries: int = 3,
    max_tokens: int = 2200,
    errors_output_path: str | None = None,
    offset: int = 0,
) -> int:
    client = DeepSeekClient(DeepSeekConfig.from_env())
    all_rows = list(read_jsonl(input_path))
    rows = all_rows[offset:]
    if limit is not None:
        rows = rows[:limit]

    outputs: List[Tuple[int, Dict]] = []
    errors: List[Dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_collect_one, client, index, row, sleep_seconds, max_retries, max_tokens): (index, row)
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            index, row = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "index": offset + index,
                        "position_name": row.get("position_name"),
                        "canonical_id": row.get("canonical_id"),
                        "error": str(exc),
                    }
                )
    outputs.sort(key=lambda item: item[0])
    ordered_records = [record for _, record in outputs]
    write_jsonl(output_path, ordered_records)
    if errors_output_path is not None:
        write_jsonl(errors_output_path, errors)
    return len(ordered_records)


def _collect_one(
    client: DeepSeekClient,
    index: int,
    row: Dict,
    sleep_seconds: float,
    max_retries: int,
    max_tokens: int,
) -> Tuple[int, Dict]:
    prompt = _build_teacher_prompt(row)
    last_error = None
    target_move = row["oracle_best_moves"][0]
    for attempt in range(max_retries):
        try:
            if sleep_seconds > 0:
                time.sleep(sleep_seconds * (index % 5))
            current_max_tokens = max_tokens + attempt * 1000
            response = client.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=current_max_tokens,
            )
            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
            message = response["choices"][0]["message"]["content"]
            parsed_cot, parsed_action = parse_seed_response(message)
            if finish_reason == "length":
                raise RuntimeError("DeepSeek response was truncated")
            if not message.strip():
                raise RuntimeError("DeepSeek returned empty content")
            if parsed_action != target_move:
                raise RuntimeError(f"parsed action {parsed_action} did not match target {target_move}")
            board = tuple(tuple(r) for r in row["board"])
            legal = legal_moves(board)
            board_after = None
            is_legal = parsed_action in legal if parsed_action is not None else False
            if is_legal:
                board_after = apply_move(board, row["player_to_move"], parsed_action)
            return index, {
                "position_name": row["position_name"],
                "source": row["source"],
                "generation": 0,
                "move_index": 0,
                "player": row["player"],
                "player_to_move": row["player_to_move"],
                "board_before": row["board"],
                "legal_moves": legal,
                "prompt": build_position_prompt(board, row["player_to_move"]),
                "teacher_prompt": prompt,
                "raw_response": message,
                "parsed_cot": parsed_cot,
                "parsed_action": parsed_action,
                "is_legal": is_legal,
                "board_after": board_after,
                "winner": _winner_name(board_after) if board_after is not None else None,
                "terminal": bool(board_after is not None and winner(board_after) != 0),
                "model_path": f"deepseek/{client.config.model}",
                "decode_config": {
                    "provider": "deepseek",
                    "model": client.config.model,
                    "temperature": 0.2,
                    "attempt": attempt + 1,
                    "max_tokens": current_max_tokens,
                    "finish_reason": finish_reason,
                },
                "teacher_metadata": {
                    "oracle_best_moves": row["oracle_best_moves"],
                    "oracle_value": row["oracle_value"],
                    "tactical_tags": row["tactical_tags"],
                    "api_id": response.get("id"),
                },
            }
        except Exception as exc:
            last_error = exc
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
    raise RuntimeError(f"failed to collect seed sample {row['position_name']}: {last_error}") from last_error


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
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--errors-output", default="data/seed/seed_positions_errors.jsonl")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    count = collect_seed_cot(
        args.input,
        args.output,
        args.limit,
        args.sleep_seconds,
        args.concurrency,
        args.max_retries,
        args.max_tokens,
        args.errors_output,
        args.offset,
    )
    print(f"wrote {count} DeepSeek seed raw records to {args.output}")


if __name__ == "__main__":
    main()
