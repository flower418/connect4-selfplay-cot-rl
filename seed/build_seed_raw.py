from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4.env import PLAYER_ONE, apply_move
from connect4.oracle import evaluate_position, tactical_tags
from data_pipeline.io import write_jsonl
from data_pipeline.prompts import build_position_prompt
from data_pipeline.schemas import RawMoveSample
from seed.templates import seed_templates


def build_seed_raw(output_path: str) -> int:
    records = []
    for index, template in enumerate(seed_templates()):
        oracle = evaluate_position(template.board, template.player_to_move, depth=4)
        action = oracle.best_moves[0]
        board_after = apply_move(template.board, template.player_to_move, action)
        player_name = "X" if template.player_to_move == PLAYER_ONE else "O"
        analysis = _build_seed_analysis(template.name, oracle.best_moves, tactical_tags(template.board, template.player_to_move))
        response = f"分析：{analysis}\n最终落子列: {action}"
        sample = RawMoveSample(
            game_id=f"seed_game_{index:04d}",
            generation=0,
            move_index=0,
            player=player_name,
            player_to_move=template.player_to_move,
            board_before=template.board,
            legal_moves=__import__("connect4.env", fromlist=["legal_moves"]).legal_moves(template.board),
            prompt=build_position_prompt(template.board, template.player_to_move),
            raw_response=response,
            parsed_cot=analysis,
            parsed_action=action,
            is_legal=True,
            board_after=board_after,
            winner=None,
            terminal=False,
            model_path="seed/oracle",
            decode_config={"source": "oracle_seed_builder", "depth": 4},
        )
        records.append(sample.to_dict())
    write_jsonl(output_path, records)
    return len(records)


def _build_seed_analysis(name: str, best_moves: list[int], tags: list[str]) -> str:
    if "immediate_win" in tags:
        return f"该局面存在直接获胜机会，优先完成四连。最佳列候选为 {best_moves}。模板: {name}。"
    if "must_block" in tags:
        return f"该局面需要先阻止对手下一手取胜，再考虑后续进攻。最佳列候选为 {best_moves}。模板: {name}。"
    return f"该局面没有立刻战术终结，优先选择更稳的结构性落点。最佳列候选为 {best_moves}。模板: {name}。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/seed/seed_positions_raw.jsonl")
    args = parser.parse_args()
    count = build_seed_raw(args.output)
    print(f"wrote {count} seed raw records to {Path(args.output)}")


if __name__ == "__main__":
    main()
