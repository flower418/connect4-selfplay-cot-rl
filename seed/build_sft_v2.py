from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.env import PLAYER_ONE, Board, apply_move, board_from_rows, canonical_position_id, legal_moves, new_board, render_board, winner
from connect4.oracle import evaluate_position
from data_pipeline.io import read_jsonl, write_jsonl
from data_pipeline.prompts import build_position_prompt
from data_pipeline.schemas import SFTRecord
from seed.deepseek_client import DeepSeekClient, DeepSeekConfig
from seed.parse_seed_responses import parse_seed_response
from training.build_sft import build_sft_records
from verification.cleaner import verify_sample
from data_pipeline.schemas import RawMoveSample


SYSTEM_PROMPT = (
    "你是四子棋老师，任务是为一个已知正确落子生成高质量、简洁、真实的中文分析。"
    "第一句必须先说明当前合法列，并说明只能在这些列里选择。"
    "后续分析必须围绕双方威胁和已知正确落子，不要因为靠近中间就直接选择。"
    "不要编造不存在的直接赢、必须防守或非法规则。"
    "输出必须只有两段：\n"
    "分析：...\n"
    "最终落子列: N"
)

SPLIT_TARGETS = {
    "legal_filter": 0.50,
    "immediate_win": 0.20,
    "must_block": 0.20,
    "depth_or_forced": 0.10,
}


def build_replay_with_legal_prefix(input_path: str, output_path: str, count: int, seed: int) -> int:
    rng = random.Random(seed)
    rows = list(read_jsonl(input_path))
    rng.shuffle(rows)
    selected = rows[:count]
    output = []
    for row in selected:
        item = dict(row)
        legal = _legal_moves_from_prompt_or_extra(item)
        prefix = _legal_prefix(legal)
        response = item["response"]
        if response.startswith("分析："):
            response = "分析：" + prefix + response[len("分析：") :]
        else:
            response = f"分析：{prefix}{response}"
        item["response"] = response
        extra = dict(item.get("extra_info", {}))
        extra["v2_source"] = "replay_legal_prefix"
        extra["legal_moves"] = legal
        item["extra_info"] = extra
        output.append(item)
    write_jsonl(output_path, output)
    return len(output)


def generate_targeted_candidates(
    output_path: str,
    target_count: int,
    seed: int,
    max_empty: int = 14,
    max_attempts: int = 500_000,
    train_paths: Iterable[str] = ("data/seed/seed_positions_verified_10k.jsonl",),
    progress_every: int = 5000,
) -> int:
    rng = random.Random(seed)
    train_ids = _load_canonical_ids(train_paths)
    seen = set(train_ids)
    records: List[Dict] = []
    counts = {name: 0 for name in SPLIT_TARGETS}
    targets = _split_counts(target_count)
    attempts = 0

    while len(records) < target_count and attempts < max_attempts:
        attempts += 1
        board, player = _sample_late_position(rng, max_empty=max_empty)
        can_id = canonical_position_id(board, player)
        if can_id in seen:
            continue
        legal = legal_moves(board)
        if len(legal) >= 7:
            continue
        oracle = evaluate_position(board, player, depth=4)
        if not oracle.best_moves:
            continue
        split = _assign_target_split(board, player, oracle, legal)
        if split is None or counts[split] >= targets[split]:
            continue
        seen.add(can_id)
        counts[split] += 1
        records.append(_candidate_record(f"v2_{split}_{counts[split]:04d}", split, board, player, oracle))
        if progress_every > 0 and attempts % progress_every == 0:
            write_jsonl(output_path, records)
            print(f"[v2 candidates] attempts={attempts} total={len(records)} counts={counts}", flush=True)

    write_jsonl(output_path, records)
    manifest = {
        "target_count": target_count,
        "actual_count": len(records),
        "attempts": attempts,
        "counts": counts,
        "targets": targets,
        "max_empty": max_empty,
        "excluded_train_ids": len(train_ids),
    }
    write_jsonl(str(Path(output_path).with_suffix("")) + "_manifest.jsonl", [manifest])
    return len(records)


def collect_targeted_cot(
    input_path: str,
    output_path: str,
    limit: int | None,
    offset: int,
    concurrency: int,
    sleep_seconds: float,
    max_retries: int,
    max_tokens: int,
    errors_output_path: str,
) -> int:
    client = DeepSeekClient(DeepSeekConfig.from_env())
    rows = list(read_jsonl(input_path))[offset:]
    if limit is not None:
        rows = rows[:limit]
    outputs: List[Tuple[int, Dict]] = []
    errors: List[Dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_collect_one_targeted, client, index, row, sleep_seconds, max_retries, max_tokens): (index, row)
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
                        "target_split": row.get("target_split"),
                        "error": str(exc),
                    }
                )
    outputs.sort(key=lambda item: item[0])
    write_jsonl(output_path, [record for _, record in outputs])
    write_jsonl(errors_output_path, errors)
    return len(outputs)


def build_verified_from_raw(input_path: str, output_path: str) -> int:
    verified = []
    for row in read_jsonl(input_path):
        payload = dict(row)
        metadata = {}
        if "position_name" in payload:
            payload["game_id"] = payload.pop("position_name")
        for key in ("source", "teacher_metadata", "teacher_prompt", "target_split"):
            if key in payload:
                metadata[key] = payload.pop(key)
        raw = RawMoveSample(**payload)
        sample = verify_sample(raw)
        sample.metadata.update(metadata)
        verified.append(sample.to_dict())
    write_jsonl(output_path, verified)
    return len(verified)


def assemble_v2_dataset(replay_path: str, targeted_verified_path: str, output_path: str) -> int:
    records = list(read_jsonl(replay_path))
    replay_count = len(records)
    targeted_count = 0
    for row in read_jsonl(targeted_verified_path):
        if row["split"] != "train" or row["move_quality"] not in {"best", "good"}:
            continue
        raw = row["raw"]
        targeted_count += 1
        records.append(
            SFTRecord(
                data_source="connect4_v2_targeted",
                prompt=raw["prompt"],
                response=raw["raw_response"],
                reward_model={"style": "rule"},
                extra_info={
                    "generation": raw["generation"],
                    "position_id": row["position_id"],
                    "canonical_id": row["canonical_id"],
                    "faithfulness_score_v1": row["faithfulness_score_v1"],
                    "move_quality": row["move_quality"],
                    "v2_source": "targeted_legal_cot",
                    "target_split": row["metadata"].get("target_split"),
                    "legal_moves": raw["legal_moves"],
                },
            ).to_dict()
        )
    write_jsonl(output_path, records)
    write_jsonl(
        str(Path(output_path).with_suffix("")) + "_manifest.jsonl",
        [
            {
                "replay_path": replay_path,
                "targeted_verified_path": targeted_verified_path,
                "output_path": output_path,
                "replay_count": replay_count,
                "targeted_count": targeted_count,
                "total_count": len(records),
            }
        ],
    )
    return len(records)


def _collect_one_targeted(
    client: DeepSeekClient,
    index: int,
    row: Dict,
    sleep_seconds: float,
    max_retries: int,
    max_tokens: int,
) -> Tuple[int, Dict]:
    target_move = row["oracle_best_moves"][0]
    prompt = _build_targeted_teacher_prompt(row)
    last_error = None
    for attempt in range(max_retries):
        try:
            if sleep_seconds > 0:
                time.sleep(sleep_seconds * (index % 5))
            response = client.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens + attempt * 1000,
            )
            choice = response["choices"][0]
            message = choice["message"].get("content") or choice["message"].get("reasoning_content") or ""
            parsed_cot, parsed_action = parse_seed_response(message)
            if not message.strip():
                raise RuntimeError("DeepSeek returned empty content")
            if parsed_action != target_move:
                raise RuntimeError(f"parsed action {parsed_action} did not match target {target_move}")
            legal = row["legal_moves"]
            if not _mentions_legal_prefix(parsed_cot, legal):
                raise RuntimeError("analysis did not mention the legal-move constraint")
            board = board_from_rows(row["board"])
            is_legal = parsed_action in legal
            board_after = apply_move(board, row["player_to_move"], parsed_action) if is_legal else None
            return index, {
                "position_name": row["position_name"],
                "source": row["source"],
                "target_split": row["target_split"],
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
                "winner": _winner_name(board_after),
                "terminal": bool(board_after is not None and winner(board_after) != 0),
                "model_path": f"deepseek/{client.config.model}",
                "decode_config": {
                    "provider": "deepseek",
                    "model": client.config.model,
                    "temperature": 0.2,
                    "attempt": attempt + 1,
                    "max_tokens": max_tokens + attempt * 1000,
                    "finish_reason": choice.get("finish_reason"),
                },
                "teacher_metadata": {
                    "oracle_best_moves": row["oracle_best_moves"],
                    "oracle_value": row["oracle_value"],
                    "target_split": row["target_split"],
                    "legal_moves": legal,
                    "api_id": response.get("id"),
                },
            }
        except Exception as exc:
            last_error = exc
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
    raise RuntimeError(f"failed to collect targeted sample {row['position_name']}: {last_error}") from last_error


def _build_targeted_teacher_prompt(row: Dict) -> str:
    move = row["oracle_best_moves"][0]
    legal = "、".join(str(m) for m in row["legal_moves"])
    full = "、".join(str(c) for c in range(7) if c not in row["legal_moves"]) or "无"
    return (
        f"{build_position_prompt(board_from_rows(row['board']), row['player_to_move'])}\n\n"
        "请围绕一个已知正确落子写简洁分析，不要改动最终动作。\n"
        f"当前合法列: {legal}\n"
        f"已满列: {full}\n"
        f"已知正确落子列: {move}\n"
        f"样本类型: {row['target_split']}\n"
        "要求：\n"
        f"1. 分析第一句必须表达：当前合法的列是 {legal}，我只能在这些列里选择。\n"
        "2. 第一句之后再观察双方威胁、直接赢、防守或局面价值。\n"
        "3. 不要写成只因为靠近中间就选择某列。\n"
        "4. 分析总共2到3句，总字数不超过120个中文字符。\n"
        f"5. 最终必须输出：最终落子列: {move}\n"
    )


def _mentions_legal_prefix(cot: str, legal: List[int]) -> bool:
    if "合法" not in cot or "只能" not in cot:
        return False
    return all(str(move) in cot[:80] for move in legal)


def _split_counts(total: int) -> Dict[str, int]:
    counts = {name: int(total * ratio) for name, ratio in SPLIT_TARGETS.items()}
    while sum(counts.values()) < total:
        counts["legal_filter"] += 1
    return counts


def _sample_late_position(rng: random.Random, max_empty: int) -> tuple[Board, int]:
    for _ in range(500):
        board = new_board()
        player = PLAYER_ONE
        target_plies = 42 - rng.randint(2, max_empty)
        for ply in range(target_plies):
            if winner(board) != 0 or not legal_moves(board):
                break
            move = _sample_rollout_move(rng, board, player, ply)
            board = apply_move(board, player, move)
            player = -player
        empty = sum(cell == 0 for row in board for cell in row)
        if winner(board) == 0 and legal_moves(board) and empty <= max_empty:
            return board, player
    return new_board(), PLAYER_ONE


def _sample_rollout_move(rng: random.Random, board: Board, player: int, ply: int) -> int:
    moves = legal_moves(board)
    evaluation = evaluate_position(board, player, depth=2)
    if evaluation.immediate_win_moves and rng.random() < 0.85:
        return rng.choice(evaluation.immediate_win_moves)
    if evaluation.must_block_moves and rng.random() < 0.75:
        return rng.choice(evaluation.must_block_moves)
    if ply < 8 and rng.random() < 0.55:
        center_order = [3, 2, 4, 1, 5, 0, 6]
        preferred = [move for move in center_order if move in moves]
        return preferred[rng.randrange(len(preferred))]
    if evaluation.best_moves and rng.random() < 0.45:
        return rng.choice(evaluation.best_moves[: min(3, len(evaluation.best_moves))])
    return rng.choice(moves)


def _assign_target_split(board: Board, player: int, oracle, legal: List[int]) -> str | None:
    full_count = 7 - len(legal)
    immediate = [move for move in oracle.immediate_win_moves if move in oracle.best_moves]
    must_block = [move for move in oracle.must_block_moves if move in oracle.best_moves]
    if immediate and full_count >= 1:
        return "immediate_win"
    if must_block and full_count >= 1:
        return "must_block"
    values = list(oracle.move_values.values())
    regret_span = max(values) - min(values) if values else 0.0
    if full_count >= 2 and len(legal) <= 5:
        return "legal_filter"
    if full_count >= 1 and regret_span >= 20.0:
        return "depth_or_forced"
    return None


def _candidate_record(position_name: str, split: str, board: Board, player: int, oracle) -> Dict:
    player_name = "X" if player == PLAYER_ONE else "O"
    legal = legal_moves(board)
    return {
        "position_name": position_name,
        "source": "v2_targeted_late_game",
        "target_split": split,
        "board": board,
        "player_to_move": player,
        "player": player_name,
        "prompt_board": render_board(board),
        "legal_moves": legal,
        "full_columns": [col for col in range(7) if col not in legal],
        "oracle_best_moves": oracle.best_moves,
        "oracle_value": oracle.value,
        "oracle_move_values": oracle.move_values,
        "tactical_tags": _tags(oracle),
        "canonical_id": canonical_position_id(board, player),
    }


def _tags(oracle) -> List[str]:
    tags = []
    if oracle.immediate_win_moves:
        tags.append("immediate_win")
    if oracle.must_block_moves:
        tags.append("must_block")
    return tags


def _load_canonical_ids(paths: Iterable[str]) -> set[str]:
    ids = set()
    for path in paths:
        try:
            for row in read_jsonl(path):
                if "canonical_id" in row:
                    ids.add(row["canonical_id"])
        except FileNotFoundError:
            continue
    return ids


def _legal_moves_from_prompt_or_extra(item: Dict) -> List[int]:
    extra = item.get("extra_info", {})
    if "legal_moves" in extra:
        return list(extra["legal_moves"])
    prompt = item["prompt"]
    lines = prompt.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "0 1 2 3 4 5 6" and idx >= 6:
            board_lines = lines[idx - 6 : idx]
            board = []
            for board_line in board_lines:
                board.append([{"X": 1, "O": -1, ".": 0}[cell] for cell in board_line.split()])
            return legal_moves(board_from_rows(board))
    raise ValueError("could not recover legal moves from SFT prompt")


def _legal_prefix(legal: List[int]) -> str:
    legal_text = "、".join(str(move) for move in legal)
    return f"当前合法的列是 {legal_text}，我只能在这些列里选择。"


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
    parser.add_argument("--step", choices=["replay", "candidates", "collect", "verify", "assemble", "all"], default="all")
    parser.add_argument("--replay-input", default="data/train/seed_sft_10k.jsonl")
    parser.add_argument("--replay-output", default="data/train/seed_sft_v2_replay_2k.jsonl")
    parser.add_argument("--replay-count", type=int, default=2000)
    parser.add_argument("--candidate-output", default="data/seed/seed_positions_v2_targeted_candidates.jsonl")
    parser.add_argument("--targeted-raw-output", default="data/seed/seed_positions_v2_targeted_raw.jsonl")
    parser.add_argument("--targeted-errors-output", default="data/seed/seed_positions_v2_targeted_errors.jsonl")
    parser.add_argument("--targeted-verified-output", default="data/seed/seed_positions_v2_targeted_verified.jsonl")
    parser.add_argument("--sft-output", default="data/train/seed_sft_v2_5k.jsonl")
    parser.add_argument("--targeted-count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--max-empty", type=int, default=14)
    parser.add_argument("--max-attempts", type=int, default=500000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=3200)
    args = parser.parse_args()

    if args.step in {"all", "replay"}:
        count = build_replay_with_legal_prefix(args.replay_input, args.replay_output, args.replay_count, args.seed)
        print(f"wrote {count} replay SFT records to {args.replay_output}")
    if args.step in {"all", "candidates"}:
        count = generate_targeted_candidates(
            args.candidate_output,
            args.targeted_count,
            args.seed,
            args.max_empty,
            args.max_attempts,
        )
        print(f"wrote {count} targeted candidates to {args.candidate_output}")
    if args.step in {"all", "collect"}:
        count = collect_targeted_cot(
            args.candidate_output,
            args.targeted_raw_output,
            args.limit,
            args.offset,
            args.concurrency,
            args.sleep_seconds,
            args.max_retries,
            args.max_tokens,
            args.targeted_errors_output,
        )
        print(f"wrote {count} targeted raw records to {args.targeted_raw_output}")
    if args.step in {"all", "verify"}:
        count = build_verified_from_raw(args.targeted_raw_output, args.targeted_verified_output)
        print(f"wrote {count} targeted verified records to {args.targeted_verified_output}")
    if args.step in {"all", "assemble"}:
        count = assemble_v2_dataset(args.replay_output, args.targeted_verified_output, args.sft_output)
        print(f"wrote {count} v2 SFT records to {args.sft_output}")


if __name__ == "__main__":
    main()
