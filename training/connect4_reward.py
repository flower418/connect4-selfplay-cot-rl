from __future__ import annotations

from typing import Any


def compute_score(data_source: str, solution_str: str, ground_truth: dict[str, Any] | None, extra_info: dict[str, Any] | None = None, **_: Any) -> float:
    """Simple verifier-style reward for Connect4 GRPO.

    The prompt builder already ships verifier ground truth in ``ground_truth``.
    This reward gives:
    - legality / format credit through extracted final move matching
    - best-move credit
    - partial credit for good tactical alignment
    """
    ground_truth = ground_truth or {}
    extra_info = extra_info or {}
    best_moves = set(ground_truth.get("oracle_best_moves", []) or [])
    move_quality = ground_truth.get("move_quality", "")
    faithfulness = float(ground_truth.get("faithfulness_score_v1", 0.0) or 0.0)

    parsed = _extract_final_move(solution_str)
    score = 0.0

    if parsed is not None:
        score += 0.2
        if parsed in best_moves:
            score += 0.5

    if move_quality == "best":
        score += 0.2
    elif move_quality == "good":
        score += 0.1

    score += 0.2 * min(max(faithfulness, 0.0), 1.0)

    if "illegal" in solution_str.lower():
        score -= 0.2

    return float(max(0.0, min(1.0, score)))


def _extract_final_move(text: str) -> int | None:
    markers = ["最终落子列:", "final column:", "final move:", "move:"]
    for marker in markers:
        if marker in text:
            tail = text.split(marker, 1)[1].strip()
            token = tail.split()[0].strip(".,，。;；:：")
            if token.isdigit():
                return int(token)
    return None
