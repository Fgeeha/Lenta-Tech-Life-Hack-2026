"""Tracking and best-frame selection for price tags.

The tracker keeps several top-quality crops per track. OCR is later run on the
best K candidates, which is more robust than using a single frame selected only
by bbox area.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from shelf.detect.detector import Detection
from shelf.io.video import frame_quality_score, laplacian_sharpness

logger = logging.getLogger(__name__)


@dataclass
class TrackCandidate:
    """One OCR candidate crop for a track."""

    det: Detection
    crop: "np.ndarray"
    timestamp_ms: float
    score: float
    sharpness: float


@dataclass
class TrackState:
    """Состояние одного трека (ценника)."""

    track_id: int
    best_det: Detection
    best_score: float = 0.0
    best_frame: "np.ndarray | None" = field(default=None, repr=False)
    best_ts: float = 0.0
    n_seen: int = 0
    candidates: list[TrackCandidate] = field(default_factory=list, repr=False)
    # Frame with the sharpest QR zone — scored independently of OCR quality.
    # Used as a fallback decode attempt when standard candidates miss the QR.
    best_qr_frame: "np.ndarray | None" = field(default=None, repr=False)
    best_qr_score: float = 0.0
    best_qr_ts: float = 0.0


def _qr_zone_sharpness(crop: np.ndarray) -> float:
    """Laplacian variance of the expected QR-code zone in a price-tag crop.

    In the raw (unrotated) crop the tag is mounted 90°-CW, so the QR code
    occupies the bottom-right quadrant.  A high value means the QR zone is
    sharp — more likely to be decodable by WeChatQR.
    """
    h, w = crop.shape[:2]
    roi = crop[int(h * 0.40) :, int(w * 0.40) :]  # bottom-right quadrant
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _clip_box(
    det: Detection, frame_shape: tuple[int, ...], margin: int = 0
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1 = max(0, int(det.x_min) - margin)
    y1 = max(0, int(det.y_min) - margin)
    x2 = min(w, int(det.x_max) + margin)
    y2 = min(h, int(det.y_max) + margin)
    return x1, y1, x2, y2


def _iou(a: Detection, b: Detection) -> float:
    ix = max(0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    iy = max(0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """ByteTrack wrapper with OCR-oriented best-frame selection.

    If supervision/ByteTrack is unavailable, a conservative IoU tracker is used
    as a fallback so the project remains runnable in a minimal environment.
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        min_hits: int = 2,
        track_thresh: float = 0.1,
        crop_margin: int = 20,
        max_candidates: int = 3,
    ):
        self._tracker = None
        self._fallback_tracks: dict[int, Detection] = {}
        self._next_id = 1
        self.lost_track_buffer = lost_track_buffer
        self.min_hits = min_hits
        self.track_thresh = track_thresh
        self.crop_margin = crop_margin
        self.max_candidates = max_candidates
        self._states: dict[int, TrackState] = {}
        self._use_fallback = False

    def _load(self) -> None:
        try:
            import supervision as sv

            self._tracker = sv.ByteTrack(
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=0.3,
            )
            self._use_fallback = False
            logger.info("ByteTrack инициализирован")
        except Exception as exc:
            self._tracker = None
            self._use_fallback = True
            logger.warning(
                "supervision.ByteTrack недоступен (%s); используем IoU fallback",
                exc,
            )

    def update(
        self, detections: list[Detection], frame: np.ndarray, timestamp: float
    ) -> list[tuple[int, Detection]]:
        """Обновить трекер. Вернуть [(track_id, det)] для активных треков.

        ``timestamp`` is milliseconds from the beginning of the video.
        """
        if self._tracker is None and not self._use_fallback:
            self._load()

        if self._use_fallback:
            tracked = self._update_iou_fallback(detections)
        else:
            tracked = self._update_bytetrack(detections)

        for tid, det in tracked:
            self._update_state(tid, det, frame, timestamp)
        return tracked

    def _update_bytetrack(
        self, detections: list[Detection]
    ) -> list[tuple[int, Detection]]:
        import supervision as sv

        if not detections:
            self._tracker.update_with_detections(sv.Detections.empty())
            return []

        boxes = np.array(
            [[d.x_min, d.y_min, d.x_max, d.y_max] for d in detections],
            dtype=np.float32,
        )
        confs = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.zeros(len(detections), dtype=int)

        sv_dets = sv.Detections(
            xyxy=boxes, confidence=confs, class_id=class_ids
        )
        tracked = self._tracker.update_with_detections(sv_dets)

        results: list[tuple[int, Detection]] = []
        if tracked.tracker_id is None:
            return results

        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            conf = (
                float(tracked.confidence[i])
                if tracked.confidence is not None
                else 0.5
            )
            det = Detection(
                x_min=x1,
                y_min=y1,
                x_max=x2,
                y_max=y2,
                confidence=conf,
                cls_name="tag",
            )
            results.append((int(tid), det))
        return results

    def _update_iou_fallback(
        self, detections: list[Detection]
    ) -> list[tuple[int, Detection]]:
        """Simple tracker for environments without supervision."""
        assigned: set[int] = set()
        results: list[tuple[int, Detection]] = []
        new_tracks: dict[int, Detection] = {}
        for det in detections:
            best_tid = None
            best_iou = 0.0
            for tid, prev in self._fallback_tracks.items():
                if tid in assigned:
                    continue
                val = _iou(det, prev)
                if val > best_iou:
                    best_iou = val
                    best_tid = tid
            if best_tid is None or best_iou < 0.25:
                best_tid = self._next_id
                self._next_id += 1
            assigned.add(best_tid)
            new_tracks[best_tid] = det
            results.append((best_tid, det))
        self._fallback_tracks = new_tracks
        return results

    def _update_state(
        self, tid: int, det: Detection, frame: np.ndarray, timestamp_ms: float
    ) -> None:
        x1, y1, x2, y2 = _clip_box(det, frame.shape, self.crop_margin)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        sharp = laplacian_sharpness(crop)
        # Larger crop area helps OCR, but glare/blur are penalized by frame_quality_score.
        score = float(det.area) * (1.0 + frame_quality_score(crop))
        qr_score = _qr_zone_sharpness(crop)

        if tid not in self._states:
            self._states[tid] = TrackState(
                track_id=int(tid),
                best_det=det,
                best_score=score,
                best_frame=crop.copy(),
                best_ts=timestamp_ms,
                n_seen=1,
                best_qr_frame=crop.copy(),
                best_qr_score=qr_score,
                best_qr_ts=timestamp_ms,
            )
        else:
            state = self._states[tid]
            state.n_seen += 1
            if score > state.best_score:
                state.best_det = det
                state.best_score = score
                state.best_frame = crop.copy()
                state.best_ts = timestamp_ms
            if qr_score > state.best_qr_score:
                state.best_qr_score = qr_score
                state.best_qr_frame = crop.copy()
                state.best_qr_ts = timestamp_ms

        cand = TrackCandidate(
            det=det,
            crop=crop.copy(),
            timestamp_ms=timestamp_ms,
            score=score,
            sharpness=sharp,
        )
        state = self._states[tid]
        state.candidates.append(cand)
        state.candidates.sort(key=lambda c: c.score, reverse=True)
        del state.candidates[self.max_candidates :]

    def get_best_crops(
        self, min_hits: int | None = None
    ) -> dict[int, TrackState]:
        """Вернуть лучшие кадры для треков с min_hits+ активных кадров."""
        threshold = min_hits if min_hits is not None else self.min_hits
        return {
            tid: s for tid, s in self._states.items() if s.n_seen >= threshold
        }

    def reset(self) -> None:
        self._states.clear()
        self._fallback_tracks.clear()
        self._next_id = 1
        if self._tracker is not None:
            try:
                import supervision as sv

                self._tracker = sv.ByteTrack(
                    lost_track_buffer=self.lost_track_buffer,
                    minimum_matching_threshold=0.3,
                )
            except Exception:
                self._tracker = None
                self._use_fallback = True
