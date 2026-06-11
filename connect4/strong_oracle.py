from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Dict, List, Tuple

from connect4.env import Board, apply_move, is_full, legal_moves, winner


EXACT_WIN = 1.0
EXACT_DRAW = 0.0
EXACT_LOSS = -1.0


@dataclass(frozen=True)
class StrongOracleEvaluation:
    best_moves: List[int]
    value: float
    move_values: Dict[int, float]
    solved: bool
    nodes: int


def solve_position(board: Board, player: int, max_empty: int = 18) -> StrongOracleEvaluation:
    empty = sum(cell == 0 for row in board for cell in row)
    if empty > max_empty:
        raise ValueError(f"position has {empty} empty cells; exact solve limit is {max_empty}")
    solver = _ExactSolver()
    moves = legal_moves(board)
    move_values: Dict[int, float] = {}
    immediate_wins: List[int] = []
    for move in moves:
        child = apply_move(board, player, move)
        if winner(child) == player:
            immediate_wins.append(move)
            move_values[move] = EXACT_WIN
    if immediate_wins:
        return StrongOracleEvaluation(sorted(immediate_wins), EXACT_WIN, move_values, True, solver.nodes)

    for move in moves:
        child = apply_move(board, player, move)
        move_values[move] = -solver.search(child, -player, -inf, inf)
    if not move_values:
        return StrongOracleEvaluation([], EXACT_DRAW, {}, True, solver.nodes)
    best_value = max(move_values.values())
    best_moves = sorted(move for move, value in move_values.items() if value == best_value)
    return StrongOracleEvaluation(best_moves, best_value, move_values, True, solver.nodes)


class _ExactSolver:
    def __init__(self) -> None:
        self.cache: Dict[Tuple[Board, int], float] = {}
        self.nodes = 0

    def search(self, board: Board, player: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        victor = winner(board)
        if victor == player:
            return EXACT_WIN
        if victor == -player:
            return EXACT_LOSS
        if is_full(board):
            return EXACT_DRAW

        key = (board, player)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        value = -inf
        for move in _ordered_moves(board):
            child = apply_move(board, player, move)
            if winner(child) == player:
                child_value = EXACT_WIN
            else:
                child_value = -self.search(child, -player, -beta, -alpha)
            value = max(value, child_value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
            if value == EXACT_WIN:
                break

        self.cache[key] = value
        return value


def _ordered_moves(board: Board) -> List[int]:
    center_first = [3, 2, 4, 1, 5, 0, 6]
    moves = set(legal_moves(board))
    return [move for move in center_first if move in moves]
