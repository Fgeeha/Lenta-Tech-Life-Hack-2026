"""ByteTrack-трекер через supervision.

Один track_id = один ценник (дедупликация по трекеру).
Хранит историю «лучших кадров» для каждого трека.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from shelf.detect.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """Состояние одного трека (ценника)."""

    track_id: int
    best_det: Detection  # детекция с наибольшей площадью × резкостью
    best_score: float = 0.0
    best_frame: "np.ndarray | None" = field(default=None, repr=False)
    best_ts: float = 0.0
    n_seen: int = 0  # сколько кадров трек был активен


def _sharpness(crop: np.ndarray) -> float:
    """Оценка резкости через Лапласиан."""
    import cv2

    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class Tracker:
    """ByteTrack-обёртка с сохранением лучшего кадра на трек."""

    def __init__(
        self,
        lost_track_buffer: int = 30,
        min_hits: int = 2,
        track_thresh: float = 0.1,
    ):
        self._tracker = None
        self.lost_track_buffer = lost_track_buffer
        self.min_hits = min_hits
        self.track_thresh = track_thresh
        self._states: dict[int, TrackState] = {}

    def _load(self) -> None:
        import supervision as sv

        self._tracker = sv.ByteTrack(
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=0.3,
        )
        logger.info("ByteTrack инициализирован")

    def update(self, detections: list[Detection], frame: np.ndarray, timestamp: float) -> list[tuple[int, Detection]]:
        """Обновить трекер. Вернуть [(track_id, det)] для активных треков."""
        if self._tracker is None:
            self._load()

        import supervision as sv

        if not detections:
            self._tracker.update_with_detections(sv.Detections.empty())
            return []

        boxes = np.array([[d.x_min, d.y_min, d.x_max, d.y_max] for d in detections], dtype=np.float32)
        confs = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.zeros(len(detections), dtype=int)

        sv_dets = sv.Detections(xyxy=boxes, confidence=confs, class_id=class_ids)
        tracked = self._tracker.update_with_detections(sv_dets)

        results: list[tuple[int, Detection]] = []
        if tracked.tracker_id is None:
            return results

        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
            det = Detection(x_min=x1, y_min=y1, x_max=x2, y_max=y2, confidence=conf, cls_name="tag")
            results.append((int(tid), det))

            # Обновляем лучший кадр для трека
            crop = frame[max(0, y1) : y2, max(0, x1) : x2]
            score = det.area * _sharpness(crop)

            if tid not in self._states or score > self._states[tid].best_score:
                self._states[tid] = TrackState(
                    track_id=int(tid),
                    best_det=det,
                    best_score=score,
                    best_frame=crop.copy() if crop.size > 0 else None,
                    best_ts=timestamp,
                    n_seen=self._states.get(tid, TrackState(int(tid), det)).n_seen + 1,
                )
            else:
                self._states[tid].n_seen += 1

        return results

    def get_best_crops(self, min_hits: int | None = None) -> dict[int, TrackState]:
        """Вернуть лучшие кадры для треков с min_hits+ активных кадров."""
        threshold = min_hits if min_hits is not None else self.min_hits
        return {tid: s for tid, s in self._states.items() if s.n_seen >= threshold}

    def reset(self) -> None:
        self._states.clear()
        if self._tracker is not None:
            import supervision as sv

            self._tracker = sv.ByteTrack(
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=0.3,
            )
