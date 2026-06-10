from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from connect4.env import Board, board_from_rows


@dataclass
class RawMoveSample:
    game_id: str
    generation: int
    move_index: int
    player: str
    player_to_move: int
    board_before: Board
    legal_moves: List[int]
    prompt: str
    raw_response: str
    parsed_cot: str
    parsed_action: Optional[int]
    is_legal: bool
    board_after: Optional[Board]
    winner: Optional[str]
    terminal: bool
    model_path: str
    decode_config: Dict[str, Any]

    def __post_init__(self) -> None:
        self.board_before = board_from_rows(self.board_before)
        if self.board_after is not None:
            self.board_after = board_from_rows(self.board_after)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerifiedSample:
    raw: RawMoveSample
    position_id: str
    canonical_id: str
    outcome: str
    move_quality: str
    tactical_tags: List[str]
    oracle_best_moves: List[int]
    oracle_value_before: float
    oracle_value_after: Optional[float]
    cot_action_consistent: bool
    cot_mentions_immediate_win: bool
    cot_mentions_must_block: bool
    faithfulness_score_v1: float
    split: str = "train"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["raw"] = self.raw.to_dict()
        return payload


@dataclass
class SFTRecord:
    data_source: str
    prompt: str
    response: str
    ability: str = "connect4_reasoning"
    reward_model: Dict[str, Any] = field(default_factory=dict)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GRPOPromptRecord:
    data_source: str
    prompt: List[Dict[str, str]]
    ability: str = "connect4_reasoning"
    reward_model: Dict[str, Any] = field(default_factory=dict)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
