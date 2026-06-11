from __future__ import annotations

import re
from typing import Optional, Tuple


FINAL_MOVE_PATTERNS = [
    re.compile(r"最终\s*落子\s*列\s*[:：]\s*([0-6])"),
    re.compile(r"最终\s*选择\s*列\s*[:：]?\s*([0-6])"),
    re.compile(r"落子\s*列\s*[:：]\s*([0-6])"),
    re.compile(r"选择\s*第?\s*([0-6])\s*列"),
]
ANALYSIS_PATTERN = re.compile(r"分析\s*[:：]\s*(.*?)(?:\n\s*最终\s*落子\s*列\s*[:：]|\n\s*落子\s*列\s*[:：]|$)", re.DOTALL)


def parse_seed_response(text: str) -> Tuple[str, Optional[int]]:
    analysis_match = ANALYSIS_PATTERN.search(text)
    move_match = None
    for pattern in FINAL_MOVE_PATTERNS:
        move_match = pattern.search(text)
        if move_match:
            break
    analysis = analysis_match.group(1).strip() if analysis_match else text.strip()
    action = int(move_match.group(1)) if move_match else None
    return analysis, action
