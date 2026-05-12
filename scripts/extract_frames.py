"""Отладочный скрипт: извлечь и сохранить кадры из видео.

Использование:
    poetry run python scripts/extract_frames.py [video_path] [--out /tmp/frames] [--interval 500] [--n 20]
"""

import argparse
import logging
from pathlib import Path

from shelf.io.video import sample_frames, save_debug_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Извлечь кадры из видео")
    parser.add_argument("video", nargs="?", default="Данные/43_15/43_15.mp4", help="Путь к видео")
    parser.add_argument("--out", default="/tmp/shelf_frames", help="Папка для кадров")
    parser.add_argument("--interval", type=int, default=500, help="Интервал в мс")
    parser.add_argument("--n", type=int, default=20, help="Макс. кадров")
    parser.add_argument("--adaptive", action="store_true", help="Адаптивный семплинг (пропуск стоячих кадров)")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"Файл не найден: {video}")
        return

    if args.adaptive:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        import cv2

        for ts, frame in sample_frames(video, interval_ms=args.interval, adaptive=True):
            if len(saved) >= args.n:
                break
            h, w = frame.shape[:2]
            scale = min(1.0, 1280 / max(w, h))
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            dest = out_dir / f"{video.stem}_{ts:.1f}s.jpg"
            cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved.append(dest)
        print(f"Адаптивно сохранено {len(saved)} кадров в {out_dir}")
    else:
        saved = save_debug_frames(video, args.out, interval_ms=args.interval, max_frames=args.n)
        print(f"Сохранено {len(saved)} кадров в {args.out}")

    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
