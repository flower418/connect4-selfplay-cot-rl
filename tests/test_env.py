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
from seed.build_seed_verified import build_seed_verified
from seed.generate_seed_positions import generate_seed_positions
from seed.parse_seed_responses import parse_seed_response
from training.build_sft import build_sft_records
from connect4.strong_oracle import solve_position
from evaluation.evaluate_baselines import evaluate_baselines, summarize_results


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
        from data_pipeline.io import write_jsonl
        from data_pipeline.prompts import build_position_prompt
        from data_pipeline.schemas import RawMoveSample

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "seed_raw.jsonl"
            verified_path = Path(tmpdir) / "seed_verified.jsonl"
            sft_path = Path(tmpdir) / "seed_sft.jsonl"
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
            raw = RawMoveSample(
                game_id="unit_seed",
                generation=0,
                move_index=0,
                player="X",
                player_to_move=PLAYER_ONE,
                board_before=board,
                legal_moves=legal_moves(board),
                prompt=build_position_prompt(board, PLAYER_ONE),
                raw_response="分析：X 在第 3 列落子可以直接完成底行四连。\n最终落子列: 3",
                parsed_cot="X 在第 3 列落子可以直接完成底行四连。",
                parsed_action=3,
                is_legal=True,
                board_after=apply_move(board, PLAYER_ONE, 3),
                winner="X",
                terminal=True,
                model_path="unit/deepseek-v4-pro",
                decode_config={"source": "unit"},
            )
            write_jsonl(raw_path, [raw.to_dict()])
            self.assertGreater(build_seed_verified(str(raw_path), str(verified_path)), 0)
            self.assertGreater(build_sft_records(str(verified_path), str(sft_path)), 0)
            self.assertTrue(sft_path.exists())

    def test_generate_seed_positions(self):
        import tempfile
        from pathlib import Path
        from data_pipeline.io import read_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seed_candidates.jsonl"
            count = generate_seed_positions(str(path), oracle_games=2)
            self.assertGreater(count, 0)
            rows = list(read_jsonl(path))
            self.assertEqual(count, len(rows))

    def test_parse_seed_response(self):
        text = "分析：这步先占中路并保持威胁。\n最终落子列: 3"
        analysis, action = parse_seed_response(text)
        self.assertIn("占中路", analysis)
        self.assertEqual(action, 3)

    def test_strong_oracle_solves_late_position(self):
        board = new_board()
        player = PLAYER_ONE
        for move in [0, 4, 1, 5, 2]:
            board = apply_move(board, player, move)
            player = -player
        evaluation = solve_position(board, PLAYER_ONE, max_empty=37)
        self.assertTrue(evaluation.solved)
        self.assertIn(3, evaluation.best_moves)

    def test_baseline_summary_on_one_exact_record(self):
        import tempfile
        from pathlib import Path
        from data_pipeline.io import write_jsonl

        board = new_board()
        player = PLAYER_ONE
        for move in [0, 4, 1, 5, 2]:
            board = apply_move(board, player, move)
            player = -player
        strong = solve_position(board, PLAYER_ONE, max_empty=37)
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = Path(tmpdir) / "eval.jsonl"
            results_path = Path(tmpdir) / "results.jsonl"
            write_jsonl(
                eval_path,
                [
                    {
                        "eval_id": "unit",
                        "split": "late_immediate_win",
                        "board": board,
                        "player_to_move": PLAYER_ONE,
                        "legal_moves": legal_moves(board),
                        "oracle_value": strong.value,
                        "oracle_best_moves": strong.best_moves,
                        "oracle_move_values": strong.move_values,
                    }
                ],
            )
            self.assertGreater(evaluate_baselines(str(eval_path), str(results_path)), 0)
            summary = summarize_results(str(results_path))
            self.assertTrue(any(key.startswith("minimax_depth4::") for key in summary))


if __name__ == "__main__":
    unittest.main()
