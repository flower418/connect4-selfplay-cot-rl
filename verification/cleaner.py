from __future__ import annotations

from typing import Iterable, Set

from connect4.env import canonical_position_id, position_id
from connect4.oracle import evaluate_position, tactical_tags
from data_pipeline.schemas import RawMoveSample, VerifiedSample


def verify_sample(
    sample: RawMoveSample,
    eval_canonical_ids: Set[str] | None = None,
    oracle_depth: int = 4,
) -> VerifiedSample:
    eval_canonical_ids = eval_canonical_ids or set()
    board = sample.board_before
    player = sample.player_to_move
    oracle_before = evaluate_position(board, player, depth=oracle_depth)
    pos_id = position_id(board, player)
    can_id = canonical_position_id(board, player)

    if not sample.is_legal or sample.parsed_action is None:
        move_quality = "illegal"
        value_after = None
    else:
        move_quality = _classify_move(sample.parsed_action, oracle_before)
        if sample.board_after is None:
            value_after = None
        else:
            value_after = evaluate_position(sample.board_after, -player, depth=max(0, oracle_depth - 1)).value

    consistent = sample.parsed_action is not None and f"最终落子列: {sample.parsed_action}" in sample.raw_response
    mentions_immediate_win = "直接赢" in sample.parsed_cot or "立刻赢" in sample.parsed_cot
    mentions_must_block = "必须防" in sample.parsed_cot or "必须挡" in sample.parsed_cot
    faithfulness = _faithfulness_v1(
        cot_action_consistent=consistent,
        mentions_immediate_win=mentions_immediate_win,
        mentions_must_block=mentions_must_block,
        tags=tactical_tags(board, player, depth=oracle_depth),
        is_legal=sample.is_legal,
    )

    split = "eval_overlap" if can_id in eval_canonical_ids else "train"
    outcome = "unknown" if sample.winner is None else sample.winner
    return VerifiedSample(
        raw=sample,
        position_id=pos_id,
        canonical_id=can_id,
        outcome=outcome,
        move_quality=move_quality,
        tactical_tags=tactical_tags(board, player, depth=oracle_depth),
        oracle_best_moves=oracle_before.best_moves,
        oracle_value_before=oracle_before.value,
        oracle_value_after=value_after,
        cot_action_consistent=bool(consistent),
        cot_mentions_immediate_win=mentions_immediate_win,
        cot_mentions_must_block=mentions_must_block,
        faithfulness_score_v1=faithfulness,
        split=split,
    )


def filter_training_samples(samples: Iterable[VerifiedSample]) -> list[VerifiedSample]:
    accepted = []
    for sample in samples:
        if sample.split != "train":
            continue
        if sample.move_quality not in {"best", "good"}:
            continue
        if not sample.cot_action_consistent:
            continue
        accepted.append(sample)
    return accepted


def _classify_move(action: int, evaluation) -> str:
    if action in evaluation.best_moves:
        return "best"
    move_value = evaluation.move_values[action]
    best_value = max(evaluation.move_values.values())
    gap = best_value - move_value
    if gap <= 5:
        return "good"
    if gap <= 20:
        return "neutral"
    return "blunder"


def _faithfulness_v1(
    cot_action_consistent: bool,
    mentions_immediate_win: bool,
    mentions_must_block: bool,
    tags: list[str],
    is_legal: bool,
) -> float:
    score = 0.0
    score += 0.4 if cot_action_consistent else 0.0
    score += 0.25 if ("immediate_win" in tags) == mentions_immediate_win else 0.0
    score += 0.25 if ("must_block" in tags) == mentions_must_block else 0.0
    score += 0.10 if is_legal else 0.0
    return round(score, 4)
