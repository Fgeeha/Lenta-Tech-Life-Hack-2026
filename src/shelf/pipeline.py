"""Оркестратор end-to-end пайплайна."""

import logging
from pathlib import Path

import pandas as pd

from shelf.detect.detector import MSERDetector
from shelf.detect.tracker import Tracker
from shelf.io.video import sample_frames
from shelf.schema import OUTPUT_COLUMNS, PriceTag

logger = logging.getLogger(__name__)


def run(
    video_path: str | Path,
    interval_ms: int = 200,
    adaptive: bool = True,
    min_hits: int = 2,
) -> pd.DataFrame:
    """Обработать видео и вернуть DataFrame по схеме OUTPUT_COLUMNS.

    Текущий статус:
    - Этапы 0-4: детекция (MSER) + трекинг (ByteTrack) ✓
    - Этапы 5-7: QR + OCR + мерж — в разработке (заглушки)
    """
    video_path = Path(video_path)
    filename = video_path.name

    detector = MSERDetector()
    tracker = Tracker(min_hits=min_hits)

    logger.info("Запуск пайплайна: %s", filename)
    frame_count = 0

    for ts, frame in sample_frames(video_path, interval_ms=interval_ms, adaptive=adaptive):
        dets = detector.detect(frame)
        tracker.update(dets, frame, ts)
        frame_count += 1
        if frame_count % 50 == 0:
            logger.info("Обработано кадров: %d  треков: %d", frame_count, len(tracker._states))

    best_crops = tracker.get_best_crops(min_hits=min_hits)
    logger.info("Треков с min_hits>=%d: %d", min_hits, len(best_crops))

    tags: list[PriceTag] = []
    for tid, state in best_crops.items():
        d = state.best_det
        tag = PriceTag(
            filename=filename,
            frame_timestamp=state.best_ts,
            x_min=d.x_min,
            y_min=d.y_min,
            x_max=d.x_max,
            y_max=d.y_max,
        )
        tags.append(tag)

    rows = [t.to_dict() for t in tags]
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    logger.info("Готово: %d уникальных ценников найдено", len(df))
    return df
