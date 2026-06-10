from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.env import PLAYER_ONE, PLAYER_TWO, Board, apply_move, canonical_position_id, new_board, render_board, winner
from connect4.oracle import evaluate_position, tactical_tags
from data_pipeline.io import write_jsonl
from seed.templates import seed_templates


def generate_seed_positions(output_path: str, oracle_games: int = 12) -> int:
    positions: List[Dict] = []
    seen = set()

    for template in seed_templates():
        record = _build_position_record(
            position_name=template.name,
            source="template",
            board=template.board,
            player_to_move=template.player_to_move,
        )
        if record["canonical_id"] not in seen:
            seen.add(record["canonical_id"])
            positions.append(record)

    for game_index in range(oracle_games):
        for record in _oracle_game_positions(game_index):
            if record["canonical_id"] in seen:
                continue
            seen.add(record["canonical_id"])
            positions.append(record)

    write_jsonl(output_path, positions)
    return len(positions)


def _oracle_game_positions(game_index: int) -> List[Dict]:
    board = new_board()
    player = PLAYER_ONE
    records = []
    opening_cycle = [3, 2, 4, 1, 5, 0, 6]
    forced_opening = opening_cycle[game_index % len(opening_cycle)]

    for move_index in range(8):
        if winner(board) != 0:
            break
        if move_index == 0 and forced_opening in evaluate_position(board, player, depth=4).best_moves:
            chosen_move = forced_opening
        else:
            chosen_move = evaluate_position(board, player, depth=4).best_moves[0]
        if move_index >= 2:
            records.append(
                _build_position_record(
                    position_name=f"oracle_game_{game_index:03d}_move_{move_index:02d}",
                    source="oracle_game",
                    board=board,
                    player_to_move=player,
                )
            )
        board = apply_move(board, player, chosen_move)
        player = -player
    return records


def _build_position_record(position_name: str, source: str, board: Board, player_to_move: int) -> Dict:
    oracle = evaluate_position(board, player_to_move, depth=4)
    player_name = "X" if player_to_move == PLAYER_ONE else "O"
    return {
        "position_name": position_name,
        "source": source,
        "board": board,
        "player_to_move": player_to_move,
        "player": player_name,
        "prompt_board": render_board(board),
        "oracle_best_moves": oracle.best_moves,
        "oracle_value": oracle.value,
        "tactical_tags": tactical_tags(board, player_to_move, depth=4),
        "canonical_id": canonical_position_id(board, player_to_move),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/seed/seed_positions_candidates.jsonl")
    parser.add_argument("--oracle-games", type=int, default=12)
    args = parser.parse_args()
    count = generate_seed_positions(args.output, oracle_games=args.oracle_games)
    print(f"wrote {count} seed candidate positions to {args.output}")


if __name__ == "__main__":
    main()
