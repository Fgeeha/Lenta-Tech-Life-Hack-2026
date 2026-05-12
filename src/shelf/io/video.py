"""Frame sampler — извлечение кадров из видео."""

import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def sample_frames(
    video_path: str | Path,
    interval_ms: int = 200,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Yield (timestamp_sec, frame) с заданным интервалом в мс."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps * interval_ms / 1000))
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                ts = frame_idx / fps
                yield ts, frame
            frame_idx += 1
    finally:
        cap.release()
