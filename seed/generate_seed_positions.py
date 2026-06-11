from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.env import PLAYER_ONE, Board, apply_move, canonical_position_id, legal_moves, new_board, render_board, winner
from connect4.oracle import evaluate_position, tactical_tags
from data_pipeline.io import write_jsonl
from seed.templates import seed_templates


def generate_seed_positions(
    output_path: str,
    oracle_games: int = 40,
    seed: int = 7,
    max_prefix_len: int = 8,
    max_total_plies: int = 12,
) -> int:
    rng = random.Random(seed)
    positions: List[Dict] = []
    seen = set()

    for template in seed_templates():
        record = _build_position_record(
            position_name=template.name,
            source="template",
            board=template.board,
            player_to_move=template.player_to_move,
        )
        _append_if_new(positions, seen, record)

    for game_index in range(oracle_games):
        for record in _sample_oracle_game_positions(
            rng=rng,
            game_index=game_index,
            max_prefix_len=max_prefix_len,
            max_total_plies=max_total_plies,
        ):
            _append_if_new(positions, seen, record)

    write_jsonl(output_path, positions)
    return len(positions)


def _sample_oracle_game_positions(
    rng: random.Random,
    game_index: int,
    max_prefix_len: int,
    max_total_plies: int,
) -> List[Dict]:
    board = new_board()
    player = PLAYER_ONE
    records: List[Dict] = []

    prefix_len = rng.randint(2, max_prefix_len)
    for ply in range(prefix_len):
        move = _sample_prefix_move(rng, board, player, ply, game_index)
        if move is None:
            return records
        board = apply_move(board, player, move)
        player = -player
        if winner(board) != 0:
            return records

    for ply in range(prefix_len, max_total_plies):
        if winner(board) != 0:
            break
        records.append(
            _build_position_record(
                position_name=f"oracle_game_{game_index:04d}_ply_{ply:02d}",
                source="oracle_game",
                board=board,
                player_to_move=player,
            )
        )
        move = _sample_oracle_guided_move(rng, board, player, ply)
        if move is None:
            break
        board = apply_move(board, player, move)
        player = -player
    return records


def _sample_prefix_move(
    rng: random.Random,
    board: Board,
    player: int,
    ply: int,
    game_index: int,
) -> int | None:
    moves = legal_moves(board)
    if not moves:
        return None

    # Bias early positions toward center but keep real spread.
    center_order = [3, 2, 4, 1, 5, 0, 6]
    weighted = [move for move in center_order if move in moves]
    if ply < 2:
        top = weighted[: min(5, len(weighted))]
        return top[(game_index + ply + rng.randint(0, len(top) - 1)) % len(top)]
    return moves[rng.randrange(len(moves))]


def _sample_oracle_guided_move(rng: random.Random, board: Board, player: int, ply: int) -> int | None:
    evaluation = evaluate_position(board, player, depth=4)
    if not evaluation.move_values:
        return None
    ranked = sorted(evaluation.move_values.items(), key=lambda item: (-item[1], item[0]))
    top_k = min(4, len(ranked))
    choices = ranked[:top_k]

    # Mix exploitation and diversity: early/mid positions draw from top-k with soft bias.
    weights = []
    best_value = choices[0][1]
    for idx, (_, value) in enumerate(choices):
        gap = best_value - value
        base = 1.0 / (1.0 + max(0.0, gap) / 10.0)
        weights.append(base * (0.92 ** idx))
    total = sum(weights)
    threshold = rng.random() * total
    running = 0.0
    for (move, _), weight in zip(choices, weights):
        running += weight
        if running >= threshold:
            return move
    return choices[-1][0]


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


def _append_if_new(positions: List[Dict], seen: set[str], record: Dict) -> None:
    if record["canonical_id"] in seen:
        return
    seen.add(record["canonical_id"])
    positions.append(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/seed/seed_positions_candidates.jsonl")
    parser.add_argument("--oracle-games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-prefix-len", type=int, default=8)
    parser.add_argument("--max-total-plies", type=int, default=12)
    args = parser.parse_args()
    count = generate_seed_positions(
        args.output,
        oracle_games=args.oracle_games,
        seed=args.seed,
        max_prefix_len=args.max_prefix_len,
        max_total_plies=args.max_total_plies,
    )
    print(f"wrote {count} seed candidate positions to {args.output}")


if __name__ == "__main__":
    main()
