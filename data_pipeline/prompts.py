from __future__ import annotations

from connect4.env import Board, PLAYER_ONE, render_board


def build_position_prompt(board: Board, player_to_move: int) -> str:
    player = "X" if player_to_move == PLAYER_ONE else "O"
    return (
        "你在玩四子棋。\n"
        "请先做简要局面分析，再给出一个合法落子列。\n"
        "输出格式固定为两段：\n"
        "分析：...\n"
        "最终落子列: N\n\n"
        f"当前棋盘:\n{render_board(board)}\n"
        f"轮到 {player} 落子。"
    )
