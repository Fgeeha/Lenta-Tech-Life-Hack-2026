"""Generate submission CSV by running the production pipeline on all videos.

Usage:
    PYTHONPATH=src poetry run python scripts/generate_submission.py \
        --videos Данные/Unlabeled/*.mp4 \
        --out submission.csv

Or to run on ALL videos (labeled + unlabeled):
    PYTHONPATH=src poetry run python scripts/generate_submission.py \
        --videos Данные/Unlabeled/*.mp4 Данные/*/\*.mp4 \
        --out submission.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from shelf import pipeline
from shelf.io.writer import prepare_output_dataframe

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission CSV")
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("--detector", default="hybrid")
    parser.add_argument("--ocr-engine", default="auto")
    parser.add_argument("--ocr-top-k", type=int, default=1)
    args = parser.parse_args()

    all_dfs: list[pd.DataFrame] = []
    for video_path in sorted(args.videos):
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"SKIP (not found): {video_path}")
            continue
        print(f"Processing {video_path.name} ...")
        df = pipeline.run(
            video_path,
            interval_ms=args.interval_ms,
            detector_name=args.detector,
            ocr_engine_name=None if args.ocr_engine == "auto" else args.ocr_engine,
            ocr_top_k=args.ocr_top_k,
        )
        df = prepare_output_dataframe(df)
        all_dfs.append(df)
        print(f"  → {len(df)} tags")

    if not all_dfs:
        print("No videos processed.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(combined)} rows to {args.out}")


if __name__ == "__main__":
    main()
