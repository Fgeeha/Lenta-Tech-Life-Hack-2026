"""Прогон детектора на нескольких кадрах видео.

Использование:
    poetry run python scripts/run_detector.py [video] [--conf 0.25] [--n 5] [--out /tmp/det]
"""

import argparse
import logging
from pathlib import Path

import cv2

from shelf.detect.detector import PriceTagDetector
from shelf.io.video import sample_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="Данные/43_15/43_15.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", default="/tmp/shelf_detections")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = PriceTagDetector()
    total_dets = 0

    for i, (ts, frame) in enumerate(
        sample_frames(args.video, interval_ms=1000, adaptive=False)
    ):
        if i >= args.n:
            break
        dets = detector.detect(frame)
        total_dets += len(dets)
        print(
            f"  t={ts:.1f}s → {len(dets)} детекций: {[(d.cls_name, round(d.confidence, 2)) for d in dets]}"
        )

        vis = detector.visualize(frame, dets)
        h, w = vis.shape[:2]
        scale = min(1.0, 1280 / max(w, h))
        vis_small = cv2.resize(vis, (int(w * scale), int(h * scale)))
        out_path = out_dir / f"frame_{i:02d}_{ts:.1f}s.jpg"
        cv2.imwrite(str(out_path), vis_small, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(
        f"\nИтого {total_dets} детекций на {min(i + 1, args.n)} кадрах. Результаты: {out_dir}"
    )


if __name__ == "__main__":
    main()
