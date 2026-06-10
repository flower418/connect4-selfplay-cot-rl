from __future__ import annotations

from dataclasses import dataclass
from typing import List

from connect4.env import Board, PLAYER_ONE, PLAYER_TWO, board_from_rows


@dataclass(frozen=True)
class SeedPositionTemplate:
    name: str
    board: Board
    player_to_move: int
    expected_tags: List[str]


def seed_templates() -> list[SeedPositionTemplate]:
    return [
        SeedPositionTemplate(
            name="x_immediate_horizontal_win",
            board=board_from_rows(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, -1, -1, 0],
                ]
            ),
            player_to_move=PLAYER_ONE,
            expected_tags=["immediate_win"],
        ),
        SeedPositionTemplate(
            name="o_must_block_horizontal",
            board=board_from_rows(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, -1, 0, 0],
                ]
            ),
            player_to_move=PLAYER_TWO,
            expected_tags=["must_block"],
        ),
        SeedPositionTemplate(
            name="x_center_preference_opening",
            board=board_from_rows(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            player_to_move=PLAYER_ONE,
            expected_tags=[],
        ),
        SeedPositionTemplate(
            name="x_vertical_finish",
            board=board_from_rows(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, -1, 0, 0, 0, 0, 0],
                    [1, -1, 0, 0, 0, 0, 0],
                ]
            ),
            player_to_move=PLAYER_ONE,
            expected_tags=["immediate_win"],
        ),
    ]
