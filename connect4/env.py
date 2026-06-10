from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, List, Sequence, Tuple


ROWS = 6
COLUMNS = 7
EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = -1

Board = Tuple[Tuple[int, ...], ...]


def new_board() -> Board:
    return tuple(tuple(EMPTY for _ in range(COLUMNS)) for _ in range(ROWS))


def board_from_rows(rows: Sequence[Sequence[int]]) -> Board:
    if len(rows) != ROWS:
        raise ValueError(f"expected {ROWS} rows, got {len(rows)}")
    board: List[Tuple[int, ...]] = []
    for row in rows:
        if len(row) != COLUMNS:
            raise ValueError(f"expected {COLUMNS} columns, got {len(row)}")
        board.append(tuple(int(cell) for cell in row))
    return tuple(board)


def legal_moves(board: Board) -> List[int]:
    return [col for col in range(COLUMNS) if board[0][col] == EMPTY]


def apply_move(board: Board, player: int, column: int) -> Board:
    if column < 0 or column >= COLUMNS:
        raise ValueError(f"column out of range: {column}")
    if board[0][column] != EMPTY:
        raise ValueError(f"column is full: {column}")

    mutable = [list(row) for row in board]
    for row in range(ROWS - 1, -1, -1):
        if mutable[row][column] == EMPTY:
            mutable[row][column] = player
            return tuple(tuple(c for c in r) for r in mutable)
    raise AssertionError("unreachable")


def is_full(board: Board) -> bool:
    return not legal_moves(board)


def winner(board: Board) -> int:
    for row in range(ROWS):
        for col in range(COLUMNS):
            player = board[row][col]
            if player == EMPTY:
                continue
            if _has_line(board, row, col, 0, 1, player):
                return player
            if _has_line(board, row, col, 1, 0, player):
                return player
            if _has_line(board, row, col, 1, 1, player):
                return player
            if _has_line(board, row, col, 1, -1, player):
                return player
    return EMPTY


def is_terminal(board: Board) -> bool:
    return winner(board) != EMPTY or is_full(board)


def mirror_board(board: Board) -> Board:
    return tuple(tuple(reversed(row)) for row in board)


def canonical_board(board: Board) -> Board:
    mirrored = mirror_board(board)
    return min(board, mirrored)


def position_id(board: Board, player: int) -> str:
    return _digest([board, player])


def canonical_position_id(board: Board, player: int) -> str:
    return _digest([canonical_board(board), player])


def render_board(board: Board) -> str:
    symbols = {PLAYER_ONE: "X", PLAYER_TWO: "O", EMPTY: "."}
    rows = [" ".join(symbols[cell] for cell in row) for row in board]
    footer = "0 1 2 3 4 5 6"
    return "\n".join(rows + [footer])


@dataclass(frozen=True)
class Position:
    board: Board
    player_to_move: int

    @property
    def legal_moves(self) -> List[int]:
        return legal_moves(self.board)

    @property
    def position_id(self) -> str:
        return position_id(self.board, self.player_to_move)

    @property
    def canonical_id(self) -> str:
        return canonical_position_id(self.board, self.player_to_move)


def _has_line(board: Board, row: int, col: int, dr: int, dc: int, player: int) -> bool:
    end_row = row + (3 * dr)
    end_col = col + (3 * dc)
    if not (0 <= end_row < ROWS and 0 <= end_col < COLUMNS):
        return False
    for i in range(4):
        if board[row + i * dr][col + i * dc] != player:
            return False
    return True


def _digest(parts: Iterable[object]) -> str:
    raw = repr(tuple(parts)).encode("utf-8")
    return sha256(raw).hexdigest()
