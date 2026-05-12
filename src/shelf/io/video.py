"""Frame sampler — извлечение кадров из видео."""

import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Порог среднего оптического потока ниже которого считаем, что робот стоит
_FLOW_STILL_THRESHOLD = 0.5


def sample_frames(
    video_path: str | Path,
    interval_ms: int = 200,
    adaptive: bool = True,
    max_dim: int = 1280,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Yield (timestamp_sec, frame) с заданным интервалом.

    При adaptive=True пропускает дубли когда робот стоит (оптический поток мал).
    max_dim: длинная сторона кадра для детекции (не меняет сохранённые координаты —
    координаты возвращаются в масштабе оригинала).
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
        "Видео: %s  %.1f fps  %d кадров  %ds  %dx%d",
        video_path.name,
        fps,
        total,
        total / fps,
        w,
        h,
    )

    scale = min(1.0, max_dim / max(w, h))
    prev_gray: np.ndarray | None = None
    frame_idx = 0
    yielded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                ts = frame_idx / fps

                if adaptive:
                    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
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
                        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
                        if mag < _FLOW_STILL_THRESHOLD:
                            frame_idx += 1
                            prev_gray = gray
                            continue
                    prev_gray = gray

                yield ts, frame
                yielded += 1

            frame_idx += 1
    finally:
        cap.release()
        logger.info("Семплирование завершено: выдано %d кадров из %d", yielded, total)


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

    for ts, frame in sample_frames(video_path, interval_ms=interval_ms, adaptive=False):
        if len(saved) >= max_frames:
            break
        h, w = frame.shape[:2]
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        name = f"{video_path.stem}_{ts:.1f}s.jpg"
        dest = out_dir / name
        cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved.append(dest)

    logger.info("Сохранено %d кадров в %s", len(saved), out_dir)
    return saved
