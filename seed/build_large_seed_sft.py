from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.io import read_jsonl, write_jsonl
from seed.build_seed_verified import build_seed_verified
from seed.collect_seed_cot import collect_seed_cot
from seed.generate_seed_positions import generate_seed_positions
from training.build_sft import build_sft_records


def build_large_seed_sft(
    target_sft: int,
    candidate_output: str,
    raw_output: str,
    verified_output: str,
    sft_output: str,
    manifest_output: str,
    oracle_games: int,
    batch_size: int,
    concurrency: int,
    max_retries: int,
    max_tokens: int,
    seed: int,
    batch_dir: str,
    resume: bool,
) -> dict:
    batch_root = Path(batch_dir)
    if not resume:
        _reset_outputs(
            [
                candidate_output,
                raw_output,
                verified_output,
                sft_output,
                manifest_output,
                str(Path(raw_output).with_suffix("")) + "_errors.jsonl",
            ],
            batch_root,
        )

    print("[stage] generating candidates", flush=True)
    candidate_count = generate_seed_positions(
        candidate_output,
        oracle_games=oracle_games,
        seed=seed,
        max_prefix_len=12,
        max_total_plies=20,
        progress_every=max(1, oracle_games // 10),
    )
    print(f"[stage] candidates ready: {candidate_count}", flush=True)
    batch_root.mkdir(parents=True, exist_ok=True)
    raw_records, error_records, offset = _load_resume_state(raw_output, batch_root) if resume else ([], [], 0)
    batch_index = offset // batch_size
    while True:
        current_sft = _line_count(sft_output)
        if current_sft >= target_sft:
            break
        if offset >= candidate_count:
            break
        batch_raw = batch_root / f"raw_batch_{batch_index:04d}.jsonl"
        batch_errors = batch_root / f"errors_batch_{batch_index:04d}.jsonl"
        print(
            f"[stage] collect batch={batch_index:04d} offset={offset} size={batch_size}",
            flush=True,
        )
        collected = collect_seed_cot(
            input_path=candidate_output,
            output_path=str(batch_raw),
            limit=batch_size,
            sleep_seconds=0.02,
            concurrency=concurrency,
            max_retries=max_retries,
            max_tokens=max_tokens,
            errors_output_path=str(batch_errors),
            offset=offset,
        )
        raw_records.extend(read_jsonl(batch_raw))
        if Path(batch_errors).exists():
            error_records.extend(read_jsonl(batch_errors))
        write_jsonl(raw_output, raw_records)
        build_seed_verified(raw_output, verified_output)
        build_sft_records(verified_output, sft_output)
        offset += batch_size
        batch_index += 1
        print(
            json.dumps(
                {
                    "batch": batch_index,
                    "offset": offset,
                    "collected": collected,
                    "raw_total": len(raw_records),
                    "sft_total": _line_count(sft_output),
                    "target_sft": target_sft,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    manifest = {
        "target_sft": target_sft,
        "candidate_count": candidate_count,
        "raw_count": _line_count(raw_output),
        "verified_count": _line_count(verified_output),
        "sft_count": _line_count(sft_output),
        "error_count": len(error_records),
        "oracle_games": oracle_games,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "max_retries": max_retries,
        "max_tokens": max_tokens,
        "seed": seed,
        "batch_dir": str(batch_root),
        "complete": _line_count(sft_output) >= target_sft,
    }
    write_jsonl(manifest_output, [manifest])
    if error_records:
        write_jsonl(str(Path(raw_output).with_suffix("")) + "_errors.jsonl", error_records)
    return manifest


def generate_large_seed_candidates(
    candidate_output: str,
    oracle_games: int,
    seed: int,
    progress_every: int,
) -> int:
    print("[stage] generating candidates", flush=True)
    candidate_count = generate_seed_positions(
        candidate_output,
        oracle_games=oracle_games,
        seed=seed,
        max_prefix_len=12,
        max_total_plies=20,
        progress_every=progress_every,
    )
    print(f"[stage] candidates ready: {candidate_count}", flush=True)
    return candidate_count


def collect_verify_export(
    candidate_output: str,
    raw_output: str,
    verified_output: str,
    sft_output: str,
    manifest_output: str,
    target_sft: int,
    batch_size: int,
    concurrency: int,
    max_retries: int,
    max_tokens: int,
    batch_dir: str,
    resume: bool,
) -> dict:
    batch_root = Path(batch_dir)
    if not resume:
        _reset_outputs(
            [
                raw_output,
                verified_output,
                sft_output,
                manifest_output,
                str(Path(raw_output).with_suffix("")) + "_errors.jsonl",
            ],
            batch_root,
        )
    batch_root.mkdir(parents=True, exist_ok=True)
    candidate_count = _line_count(candidate_output)
    raw_records, error_records, offset = _load_resume_state(raw_output, batch_root) if resume else ([], [], 0)
    batch_index = offset // batch_size
    while True:
        current_sft = _line_count(sft_output)
        if current_sft >= target_sft:
            break
        if offset >= candidate_count:
            break
        batch_raw = batch_root / f"raw_batch_{batch_index:04d}.jsonl"
        batch_errors = batch_root / f"errors_batch_{batch_index:04d}.jsonl"
        print(
            f"[stage] collect batch={batch_index:04d} offset={offset} size={batch_size}",
            flush=True,
        )
        collected = collect_seed_cot(
            input_path=candidate_output,
            output_path=str(batch_raw),
            limit=batch_size,
            sleep_seconds=0.02,
            concurrency=concurrency,
            max_retries=max_retries,
            max_tokens=max_tokens,
            errors_output_path=str(batch_errors),
            offset=offset,
        )
        raw_records.extend(read_jsonl(batch_raw))
        if Path(batch_errors).exists():
            error_records.extend(read_jsonl(batch_errors))
        write_jsonl(raw_output, raw_records)
        build_seed_verified(raw_output, verified_output)
        build_sft_records(verified_output, sft_output)
        offset += batch_size
        batch_index += 1
        print(
            json.dumps(
                {
                    "batch": batch_index,
                    "offset": offset,
                    "collected": collected,
                    "raw_total": len(raw_records),
                    "sft_total": _line_count(sft_output),
                    "target_sft": target_sft,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    manifest = {
        "target_sft": target_sft,
        "candidate_count": candidate_count,
        "raw_count": _line_count(raw_output),
        "verified_count": _line_count(verified_output),
        "sft_count": _line_count(sft_output),
        "error_count": len(error_records),
        "batch_size": batch_size,
        "concurrency": concurrency,
        "max_retries": max_retries,
        "max_tokens": max_tokens,
        "batch_dir": str(batch_root),
        "complete": _line_count(sft_output) >= target_sft,
    }
    write_jsonl(manifest_output, [manifest])
    if error_records:
        write_jsonl(str(Path(raw_output).with_suffix("")) + "_errors.jsonl", error_records)
    return manifest


def _reset_outputs(paths: list[str], batch_root: Path) -> None:
    for path in paths:
        p = Path(path)
        if p.exists():
            p.unlink()
    if batch_root.exists():
        shutil.rmtree(batch_root)


def _load_resume_state(raw_output: str, batch_root: Path) -> tuple[list[dict], list[dict], int]:
    raw_records = list(read_jsonl(raw_output)) if Path(raw_output).exists() else []
    error_records = []
    processed = 0
    if batch_root.exists():
        raw_batches = sorted(batch_root.glob("raw_batch_*.jsonl"))
        error_batches = sorted(batch_root.glob("errors_batch_*.jsonl"))
        for batch_path in raw_batches:
            processed += _line_count(str(batch_path))
        for batch_path in error_batches:
            count = _line_count(str(batch_path))
            processed += count
            error_records.extend(read_jsonl(batch_path))
    return raw_records, error_records, max(processed, len(raw_records))


def _line_count(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-sft", type=int, default=10_000)
    parser.add_argument("--candidate-output", default="data/seed/seed_positions_candidates_10k.jsonl")
    parser.add_argument("--raw-output", default="data/seed/seed_positions_raw_10k.jsonl")
    parser.add_argument("--verified-output", default="data/seed/seed_positions_verified_10k.jsonl")
    parser.add_argument("--sft-output", default="data/train/seed_sft_10k.jsonl")
    parser.add_argument("--manifest-output", default="data/seed/seed_sft_10k_manifest.jsonl")
    parser.add_argument("--oracle-games", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-dir", default="data/seed/seed_sft_10k_batches")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--step", choices=["all", "candidates", "collect"], default="all")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    if args.step in {"all", "candidates"}:
        generate_large_seed_candidates(
            candidate_output=args.candidate_output,
            oracle_games=args.oracle_games,
            seed=args.seed,
            progress_every=args.progress_every,
        )
    if args.step in {"all", "collect"}:
        manifest = collect_verify_export(
            candidate_output=args.candidate_output,
            raw_output=args.raw_output,
            verified_output=args.verified_output,
            sft_output=args.sft_output,
            manifest_output=args.manifest_output,
            target_sft=args.target_sft,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            batch_dir=args.batch_dir,
            resume=args.resume,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
