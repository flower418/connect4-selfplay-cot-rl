from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Dict, List

from connect4.env import (
    Board,
    EMPTY,
    PLAYER_ONE,
    PLAYER_TWO,
    apply_move,
    is_full,
    legal_moves,
    mirror_board,
    winner,
)


WIN_SCORE = 1_000_000


@dataclass(frozen=True)
class OracleEvaluation:
    best_moves: List[int]
    value: float
    move_values: Dict[int, float]
    immediate_win_moves: List[int]
    must_block_moves: List[int]


def evaluate_position(board: Board, player: int, depth: int = 4) -> OracleEvaluation:
    moves = legal_moves(board)
    if not moves:
        return OracleEvaluation([], 0.0, {}, [], [])

    move_values: Dict[int, float] = {}
    immediate_win_moves: List[int] = []
    for move in moves:
        child = apply_move(board, player, move)
        if winner(child) == player:
            immediate_win_moves.append(move)
            move_values[move] = WIN_SCORE
        else:
            move_values[move] = -_search(child, -player, depth - 1, -inf, inf)

    best_value = max(move_values.values())
    best_moves = [move for move, value in move_values.items() if value == best_value]
    must_block = _find_must_block_moves(board, player)
    return OracleEvaluation(
        best_moves=sorted(best_moves),
        value=float(best_value),
        move_values=move_values,
        immediate_win_moves=sorted(immediate_win_moves),
        must_block_moves=sorted(must_block),
    )


def tactical_tags(board: Board, player: int, depth: int = 4) -> List[str]:
    tags: List[str] = []
    evaluation = evaluate_position(board, player, depth=depth)
    if evaluation.immediate_win_moves:
        tags.append("immediate_win")
    if evaluation.must_block_moves:
        tags.append("must_block")
    return tags


def _search(board: Board, player: int, depth: int, alpha: float, beta: float) -> float:
    victor = winner(board)
    if victor == player:
        return WIN_SCORE + depth
    if victor == -player:
        return -WIN_SCORE - depth
    if depth == 0 or is_full(board):
        return _heuristic(board, player)

    value = -inf
    for move in _ordered_moves(board):
        child = apply_move(board, player, move)
        value = max(value, -_search(child, -player, depth - 1, -beta, -alpha))
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return value


def _ordered_moves(board: Board) -> List[int]:
    center_first = [3, 2, 4, 1, 5, 0, 6]
    moves = set(legal_moves(board))
    return [move for move in center_first if move in moves]


def _find_must_block_moves(board: Board, player: int) -> List[int]:
    opponent = -player
    winning_replies = []
    for move in legal_moves(board):
        child = apply_move(board, opponent, move)
        if winner(child) == opponent:
            winning_replies.append(move)
    return winning_replies


def _heuristic(board: Board, player: int) -> float:
    return (
        _window_score(board, player)
        - _window_score(board, -player)
        + _center_score(board, player)
        - _center_score(board, -player)
    )


def _center_score(board: Board, player: int) -> float:
    return sum(1 for row in board if row[3] == player) * 3.0


def _window_score(board: Board, player: int) -> float:
    score = 0.0
    for row in range(6):
        for col in range(7):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                window = _collect_window(board, row, col, dr, dc)
                if window is None:
                    continue
                score += _score_window(window, player)
    return score


def _collect_window(board: Board, row: int, col: int, dr: int, dc: int):
    cells = []
    for i in range(4):
        r = row + i * dr
        c = col + i * dc
        if not (0 <= r < 6 and 0 <= c < 7):
            return None
        cells.append(board[r][c])
    return cells


def _score_window(window: List[int], player: int) -> float:
    mine = window.count(player)
    opp = window.count(-player)
    empty = window.count(EMPTY)
    if mine == 4:
        return WIN_SCORE
    if opp == 4:
        return -WIN_SCORE
    if mine == 3 and empty == 1:
        return 20.0
    if mine == 2 and empty == 2:
        return 5.0
    if opp == 3 and empty == 1:
        return -18.0
    return 0.0
