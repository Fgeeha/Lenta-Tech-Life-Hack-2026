"""Frame sampler — извлечение информативных кадров из видео."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Порог среднего оптического потока ниже которого считаем, что кадр почти дублирует предыдущий.
_FLOW_STILL_THRESHOLD = 0.35


def laplacian_sharpness(image: np.ndarray) -> float:
    """Variance of Laplacian: higher means sharper."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def glare_fraction(image: np.ndarray) -> float:
    """Approximate share of overexposed low-saturation pixels."""
    if image is None or image.size == 0:
        return 0.0
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        mask = (s < 45) & (v > 235)
    else:
        mask = image > 245
    return float(mask.mean())


def frame_quality_score(image: np.ndarray) -> float:
    """OCR-friendly frame score: sharpness penalized by glare and very dark frames."""
    if image is None or image.size == 0:
        return 0.0
    sharp = laplacian_sharpness(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    brightness = float(gray.mean()) / 255.0
    exposure_penalty = 1.0 - min(0.65, glare_fraction(image) * 2.5)
    dark_penalty = max(0.35, min(1.0, brightness / 0.35))
    return sharp * exposure_penalty * dark_penalty


def sample_frames(
    video_path: str | Path,
    interval_ms: int = 200,
    adaptive: bool = True,
    max_dim: int = 1280,
    min_sharpness: float = 0.0,
    max_frames: int | None = None,
    max_timestamp_ms: float | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Yield ``(timestamp_ms, frame)`` with a fixed/adaptive stride.

    ``timestamp_ms`` is milliseconds from the beginning of the video, which is the
    unit required by the output CSV. Older versions yielded seconds; the pipeline
    now keeps milliseconds end-to-end.  ``max_timestamp_ms`` lets callers enforce
    UI smoke-test/duration limits without decoding the rest of a long video.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(fps * interval_ms / 1000))

    logger.info(
        "Видео: %s  %.1f fps  %d кадров  %.1fs  %dx%d",
        video_path.name,
        fps,
        total,
        total / fps if fps else 0,
        w,
        h,
    )

    scale = min(1.0, max_dim / max(w, h)) if max(w, h) else 1.0
    prev_gray: np.ndarray | None = None
    frame_idx = 0
    yielded = 0

    try:
        while True:
            ts_ms = frame_idx / fps * 1000.0
            if max_timestamp_ms is not None and ts_ms > max_timestamp_ms:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                if adaptive:
                    small = (
                        cv2.resize(frame, (int(w * scale), int(h * scale)))
                        if scale < 1
                        else frame
                    )
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray,
                            gray,
                            None,
                            pyr_scale=0.5,
                            levels=2,
                            winsize=15,
                            iterations=2,
                            poly_n=5,
                            poly_sigma=1.1,
                            flags=0,
                        )
                        mag = np.sqrt(
                            flow[..., 0] ** 2 + flow[..., 1] ** 2
                        ).mean()
                        # Пропускаем только почти идентичные дубли. При остановке робота
                        # всё равно оставляем первый резкий кадр.
                        if mag < _FLOW_STILL_THRESHOLD and yielded > 0:
                            frame_idx += 1
                            prev_gray = gray
                            continue
                    prev_gray = gray

                if (
                    min_sharpness > 0
                    and laplacian_sharpness(frame) < min_sharpness
                ):
                    frame_idx += 1
                    continue

                yield ts_ms, frame
                yielded += 1
                if progress_callback and total > 0:
                    progress_callback(
                        min(0.35, 0.35 * frame_idx / total),
                        f"Извлечено кадров: {yielded}",
                    )
                if max_frames is not None and yielded >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()
        logger.info(
            "Семплирование завершено: выдано %d кадров из %d", yielded, total
        )


def save_debug_frames(
    video_path: str | Path,
    out_dir: str | Path,
    interval_ms: int = 500,
    max_frames: int = 30,
    max_dim: int = 1280,
) -> list[Path]:
    """Сохранить первые N кадров в out_dir для визуальной проверки."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)
    saved: list[Path] = []

    for ts_ms, frame in sample_frames(
        video_path, interval_ms=interval_ms, adaptive=False
    ):
        if len(saved) >= max_frames:
            break
        h, w = frame.shape[:2]
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        name = f"{video_path.stem}_{ts_ms / 1000.0:.1f}s.jpg"
        dest = out_dir / name
        cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved.append(dest)

    logger.info("Сохранено %d кадров в %s", len(saved), out_dir)
    return saved
