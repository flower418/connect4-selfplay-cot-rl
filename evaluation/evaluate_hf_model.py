from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.io import read_jsonl, write_jsonl
from data_pipeline.prompts import build_position_prompt


ACTION_PATTERNS = [
    re.compile(r"最终\s*落子\s*列\s*[:：]\s*([0-6])"),
    re.compile(r"落子\s*列\s*[:：]\s*([0-6])"),
    re.compile(r"选择\s*第?\s*([0-6])\s*列"),
]


def evaluate_hf_model(model_path: str, input_path: str, output_path: str, max_new_tokens: int = 256) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    records = []
    for row in read_jsonl(input_path):
        board = tuple(tuple(r) for r in row["board"])
        prompt = build_position_prompt(board, int(row["player_to_move"]))
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        action = parse_action(generated)
        records.append(score_model_action(row, model_path, generated, action))
    write_jsonl(output_path, records)
    return len(records)


def parse_action(text: str):
    for pattern in ACTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def score_model_action(row: dict, model_path: str, response: str, action) -> dict:
    move_values = {int(k): v for k, v in row["oracle_move_values"].items()}
    chosen_value = move_values.get(action)
    return {
        "model": model_path,
        "eval_id": row["eval_id"],
        "split": row["split"],
        "response": response,
        "parsed_action": action,
        "format_success": action is not None,
        "is_legal": action in row["legal_moves"] if action is not None else False,
        "is_oracle_best": action in row["oracle_best_moves"] if action is not None else False,
        "oracle_value": row["oracle_value"],
        "chosen_value": chosen_value,
        "value_regret": None if chosen_value is None else row["oracle_value"] - chosen_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default="data/eval/frozen_benchmark.jsonl")
    parser.add_argument("--output", default="data/metrics/model_results.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    count = evaluate_hf_model(args.model, args.input, args.output, args.max_new_tokens)
    print(f"wrote {count} model results to {args.output}")


if __name__ == "__main__":
    main()
