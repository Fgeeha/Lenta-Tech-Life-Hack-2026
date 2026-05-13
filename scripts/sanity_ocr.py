"""Sanity-check OCR на кропах fine-tuned YOLO-детектора.

Берёт 10 случайных детекций из 43_15.mp4, сохраняет кропы в /tmp/sanity_crops/
и печатает что распозналось OCR.
"""

import logging
import random
from pathlib import Path

import cv2

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    weights = Path("models/pricetag_yolov8n.pt")
    if not weights.exists():
        print(f"Модель не найдена: {weights}")
        return

    from ultralytics import YOLO

    from shelf.io.video import sample_frames
    from shelf.ocr.engine import OCREngine
    from shelf.ocr.preprocess import preprocess_crop

    model = YOLO(str(weights))
    ocr = OCREngine()
    out_dir = Path("/tmp/sanity_crops")
    out_dir.mkdir(exist_ok=True)

    crops: list[tuple[float, cv2.Mat, tuple]] = []

    for ts, frame in sample_frames("Данные/43_15/43_15.mp4", interval_ms=500, adaptive=False):
        h, w = frame.shape[:2]
        scale = min(1.0, 1280 / max(w, h))
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        results = model(small, verbose=False, conf=0.25)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = [v / scale for v in box.xyxy[0].tolist()]
            crop = frame[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)]
            if crop.size > 0:
                crops.append((ts, crop, (int(x1), int(y1), int(x2), int(y2))))

    random.shuffle(crops)
    sample = crops[:10]
    print(f"\nВсего детекций: {len(crops)}  Анализируем: {len(sample)}\n")

    ocr_hits = 0
    price_hits = 0
    for i, (ts, crop, bbox) in enumerate(sample):
        h, w = crop.shape[:2]
        proc = preprocess_crop(crop, rotate_180=True, deskew=True, upscale=3, sharpen=True, clahe=True)
        cv2.imwrite(str(out_dir / f"crop_{i:02d}_{ts:.1f}s.jpg"), proc, [cv2.IMWRITE_JPEG_QUALITY, 92])

        lines = ocr.run(proc)
        texts = [t for _, t, c in lines if c > 0.5]
        has_price = any(any(c.isdigit() for c in t) for t in texts)
        if lines:
            ocr_hits += 1
        if has_price:
            price_hits += 1

        print(f"[{i:02d}] ts={ts:.1f}s  bbox={bbox}  crop={w}×{h}")
        print(f"     OCR: {' | '.join(texts[:5]) or '—'}")

    print(f"\n=== OCR читает {ocr_hits}/{len(sample)} ({ocr_hits/max(1,len(sample))*100:.0f}%) кропов")
    print(f"=== Цены/числа: {price_hits}/{len(sample)} кропов")
    print(f"=== Кропы сохранены: {out_dir}")


if __name__ == "__main__":
    main()
