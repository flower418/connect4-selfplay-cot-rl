import unittest

from connect4.env import (
    PLAYER_ONE,
    PLAYER_TWO,
    apply_move,
    board_from_rows,
    canonical_position_id,
    legal_moves,
    mirror_board,
    new_board,
    winner,
)
from connect4.oracle import evaluate_position
from seed.build_seed_raw import build_seed_raw
from seed.build_seed_verified import build_seed_verified
from training.build_sft import build_sft_records


class Connect4EnvTest(unittest.TestCase):
    def test_apply_move_stacks_from_bottom(self):
        board = new_board()
        board = apply_move(board, PLAYER_ONE, 3)
        board = apply_move(board, PLAYER_TWO, 3)
        self.assertEqual(board[5][3], PLAYER_ONE)
        self.assertEqual(board[4][3], PLAYER_TWO)

    def test_horizontal_win(self):
        board = board_from_rows(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
            ]
        )
        self.assertEqual(winner(board), PLAYER_ONE)

    def test_canonical_id_matches_mirror(self):
        board = board_from_rows(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, -1, 0, 0, 0, 0],
            ]
        )
        self.assertEqual(
            canonical_position_id(board, PLAYER_ONE),
            canonical_position_id(mirror_board(board), PLAYER_ONE),
        )

    def test_oracle_finds_immediate_win(self):
        board = board_from_rows(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, -1, -1, 0],
            ]
        )
        evaluation = evaluate_position(board, PLAYER_ONE, depth=4)
        self.assertIn(3, evaluation.best_moves)
        self.assertIn(3, evaluation.immediate_win_moves)

    def test_legal_moves_excludes_full_column(self):
        board = board_from_rows(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0],
            ]
        )
        self.assertNotIn(0, legal_moves(board))

    def test_seed_pipeline_builds_records(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "seed_raw.jsonl"
            verified_path = Path(tmpdir) / "seed_verified.jsonl"
            sft_path = Path(tmpdir) / "seed_sft.jsonl"
            self.assertGreater(build_seed_raw(str(raw_path)), 0)
            self.assertGreater(build_seed_verified(str(raw_path), str(verified_path)), 0)
            self.assertGreater(build_sft_records(str(verified_path), str(sft_path)), 0)
            self.assertTrue(sft_path.exists())


if __name__ == "__main__":
    unittest.main()
