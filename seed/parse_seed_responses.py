from __future__ import annotations

import re
from typing import Optional, Tuple


FINAL_MOVE_PATTERN = re.compile(r"最终落子列\s*[:：]\s*([0-6])")
ANALYSIS_PATTERN = re.compile(r"分析\s*[:：]\s*(.*?)(?:\n\s*最终落子列\s*[:：]|$)", re.DOTALL)


def parse_seed_response(text: str) -> Tuple[str, Optional[int]]:
    analysis_match = ANALYSIS_PATTERN.search(text)
    move_match = FINAL_MOVE_PATTERN.search(text)
    analysis = analysis_match.group(1).strip() if analysis_match else text.strip()
    action = int(move_match.group(1)) if move_match else None
    return analysis, action
