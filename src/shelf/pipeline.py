"""Оркестратор end-to-end пайплайна video → CSV."""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from shelf.detect.detector import make_detector
from shelf.detect.tracker import Tracker
from shelf.io.video import sample_frames
from shelf.io.writer import write_csv
from shelf.ocr.engine import OCREngine
from shelf.ocr.parser import parse_ocr_result
from shelf.ocr.preprocess import preprocess_crop
from shelf.ocr.template import classify_color
from shelf.postproc.merge import merge
from shelf.qr.decoder import decode_qr
from shelf.schema import OUTPUT_COLUMNS, PriceTag

logger = logging.getLogger(__name__)

# Margin вокруг детектированного bbox для захвата всего ценника
_CROP_MARGIN = 20


def _extract_tag(
    crop_raw: "np.ndarray",
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    filename: str,
    timestamp: float,
    ocr_engine: OCREngine,
) -> PriceTag:
    """Обработать кроп ценника: OCR + QR → PriceTag.

    crop_raw — уже обрезанный регион из best_frame трека.
    """
    if crop_raw is None or crop_raw.size == 0:
        return PriceTag(
            filename=filename, frame_timestamp=timestamp, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
        )

    # --- QR ---
    # Ценники смонтированы боком (90°CCW для чтения); пробуем все ориентации
    qr_fields: dict[str, str] = {}
    for rot_code in [cv2.ROTATE_90_COUNTERCLOCKWISE, None, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]:
        rotated = cv2.rotate(crop_raw, rot_code) if rot_code is not None else crop_raw
        qr_fields = decode_qr(rotated)
        if qr_fields:
            break

    # --- Цвет ценника ---
    color = classify_color(crop_raw)

    # --- OCR ---
    proc = preprocess_crop(crop_raw, rotate_180=True, deskew=True, upscale=3, sharpen=True, clahe=True)
    ocr_lines = ocr_engine.run(proc)

    # --- Парсинг полей ---
    ocr_tag = parse_ocr_result(
        ocr_lines,
        crop=proc,
        filename=filename,
        frame_timestamp=timestamp,
        bbox=(x_min, y_min, x_max, y_max),
        color=color,
    )

    # --- Мерж QR + OCR ---
    return merge(ocr_tag, qr_fields)


def run(
    video_path: str | Path,
    interval_ms: int = 200,
    adaptive: bool = True,
    min_hits: int = 2,
    output_csv: str | Path | None = None,
    detector_name: str = "mser",
) -> pd.DataFrame:
    """Обработать видео → вернуть DataFrame по схеме OUTPUT_COLUMNS.

    Args:
        video_path: путь к .mp4 файлу
        interval_ms: интервал семплирования кадров (мс)
        adaptive: пропускать статичные кадры (оптический поток)
        min_hits: минимум кадров для трека (фильтр ложных срабатываний)
        output_csv: если задан — сохранить CSV по этому пути
    """
    video_path = Path(video_path)
    filename = video_path.name

    detector = make_detector(detector_name)
    tracker = Tracker(min_hits=min_hits)
    ocr_engine = OCREngine()

    logger.info("Запуск пайплайна: %s", filename)
    frame_count = 0

    for ts, frame in sample_frames(video_path, interval_ms=interval_ms, adaptive=adaptive):
        dets = detector.detect(frame)
        tracker.update(dets, frame, ts)
        frame_count += 1
        if frame_count % 100 == 0:
            logger.info("Кадры: %d  треки: %d", frame_count, len(tracker._states))

    best_crops = tracker.get_best_crops(min_hits=min_hits)
    logger.info("Стабильных треков (%d+ кадров): %d", min_hits, len(best_crops))

    tags: list[PriceTag] = []
    for tid, state in best_crops.items():
        if state.best_frame is None:
            continue
        d = state.best_det
        tag = _extract_tag(
            crop_raw=state.best_frame,
            x_min=d.x_min,
            y_min=d.y_min,
            x_max=d.x_max,
            y_max=d.y_max,
            filename=filename,
            timestamp=state.best_ts,
            ocr_engine=ocr_engine,
        )
        tags.append(tag)

    rows = [t.to_dict() for t in tags]
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS) if rows else pd.DataFrame(columns=OUTPUT_COLUMNS)

    if output_csv is not None:
        write_csv(tags, output_csv)
        logger.info("CSV сохранён: %s", output_csv)

    logger.info("Готово: %d уникальных ценников", len(df))
    return df
