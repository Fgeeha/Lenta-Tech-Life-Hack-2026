"""Price-tag detectors.

Default production mode is ``hybrid``: use a trained tiled YOLO detector when
weights are available, then add MSER proposals as a recall-oriented fallback.
This is safer than falling back to generic COCO ``yolov8n.pt``, which does not
know the ``price_tag`` class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ASPECT_MIN = 0.18
_ASPECT_MAX = 2.4
_MIN_AREA_ORIG = 5_000  # px² в оригинальных координатах
_MAX_AREA_ORIG = 260_000
_MIN_W = 35
_MIN_H = 50


@dataclass
class Detection:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    cls_name: str = "tag"

    @property
    def area(self) -> int:
        return max(0, int(self.x_max - self.x_min)) * max(0, int(self.y_max - self.y_min))

    @property
    def aspect(self) -> float:
        h = self.y_max - self.y_min
        return (self.x_max - self.x_min) / h if h > 0 else 0.0

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    def clipped(self, width: int, height: int) -> "Detection":
        return Detection(
            x_min=max(0.0, min(float(width - 1), float(self.x_min))),
            y_min=max(0.0, min(float(height - 1), float(self.y_min))),
            x_max=max(0.0, min(float(width), float(self.x_max))),
            y_max=max(0.0, min(float(height), float(self.y_max))),
            confidence=float(self.confidence),
            cls_name=self.cls_name,
        )


def _iou(a: Detection, b: Detection) -> float:
    ix = max(0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    iy = max(0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _env_int(name: str, default: int) -> int:
    """Read a positive integer env override for lightweight local runs."""
    import os

    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    return value if value > 0 else default


def _nms(dets: list[Detection], iou_thr: float = 0.35) -> list[Detection]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    keep: list[Detection] = []
    for d in dets:
        if d.area <= 0:
            continue
        if any(_iou(d, k) > iou_thr for k in keep):
            continue
        keep.append(d)
    return keep


def _valid_tag_geometry(det: Detection) -> bool:
    return (
        _MIN_AREA_ORIG <= det.area <= _MAX_AREA_ORIG
        and _ASPECT_MIN <= det.aspect <= _ASPECT_MAX
        and (det.x_max - det.x_min) >= _MIN_W
        and (det.y_max - det.y_min) >= _MIN_H
    )


class MSERDetector:
    """MSER detector: no training needed, useful as fallback/high-recall proposals."""

    def __init__(
        self,
        process_width: int | None = None,
        delta: int = 5,
        min_area_scaled: int = 650,
        max_area_scaled: int = 18_000,
        max_variation: float = 0.35,
    ):
        self.process_width = (
            process_width
            if process_width is not None
            else _env_int("SHELF_MSER_PROCESS_WIDTH", 1440)
        )
        self._mser = cv2.MSER_create(
            delta=delta,
            min_area=min_area_scaled,
            max_area=max_area_scaled,
            max_variation=max_variation,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.process_width / max(w, h))
        small = (
            cv2.resize(frame, (int(w * scale), int(h * scale)))
            if scale < 1.0
            else frame
        )

        # MSER is sensitive to glare; CLAHE on grayscale improves region stability.
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _, bboxes = self._mser.detectRegions(gray)

        dets: list[Detection] = []
        for x, y, bw, bh in bboxes:
            orig_x = x / scale
            orig_y = y / scale
            orig_w = bw / scale
            orig_h = bh / scale
            det = Detection(
                orig_x,
                orig_y,
                orig_x + orig_w,
                orig_y + orig_h,
                confidence=0.35,
                cls_name="mser",
            )
            det = det.clipped(w, h)
            if not _valid_tag_geometry(det):
                continue

            # Boost rectangular regions with orange/yellow/red background typical for price tags.
            crop = frame[int(det.y_min) : int(det.y_max), int(det.x_min) : int(det.x_max)]
            color_boost = _price_tag_color_score(crop)
            text_boost = _text_edge_score(crop)
            det.confidence = min(
                0.99, 0.20 + 0.45 * color_boost + 0.35 * text_boost
            )
            if det.confidence >= 0.25:
                dets.append(det)

        result = _nms(dets, iou_thr=0.42)
        logger.debug(
            "MSERDetector: %d raw -> %d after NMS", len(dets), len(result)
        )
        return result

    def visualize(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            cv2.rectangle(
                out, (int(d.x_min), int(d.y_min)), (int(d.x_max), int(d.y_max)), (0, 255, 0), 6
            )
            cv2.putText(
                out,
                f"{d.cls_name} {d.confidence:.2f}",
                (int(d.x_min), max(40, int(d.y_min) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                4,
            )
        return out


class YOLODetector:
    """Generic YOLO wrapper. Use only with a trained price-tag model."""

    def __init__(
        self,
        model_name: str = "models/pricetag_yolov8n.pt",
        confidence: float = 0.25,
        process_width: int = 1600,
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.process_width = process_width
        self._model = None
        self._disabled = False

    def _load(self) -> None:
        from pathlib import Path

        if not Path(self.model_name).exists():
            logger.warning("YOLO weights not found: %s", self.model_name)
            self._disabled = True
            return
        from ultralytics import YOLO

        self._model = YOLO(self.model_name)
        logger.info("YOLO загружен: %s", self.model_name)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._disabled:
            return []
        if self._model is None:
            self._load()
            if self._disabled:
                return []
        h, w = frame.shape[:2]
        scale = min(1.0, self.process_width / max(w, h))
        small = (
            cv2.resize(frame, (int(w * scale), int(h * scale)))
            if scale < 1.0
            else frame
        )
        results = self._model(small, verbose=False, conf=self.confidence)[0]
        dets: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_name = results.names.get(int(box.cls[0]), "tag")
            det = Detection(
                x1 / scale,
                y1 / scale,
                x2 / scale,
                y2 / scale,
                conf,
                cls_name,
            ).clipped(w, h)
            if _valid_tag_geometry(det):
                dets.append(det)
        return _nms(dets, iou_thr=0.45)

    def visualize(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            cv2.rectangle(
                out, (int(d.x_min), int(d.y_min)), (int(d.x_max), int(d.y_max)), (255, 100, 0), 6
            )
            cv2.putText(
                out,
                f"{d.cls_name} {d.confidence:.2f}",
                (int(d.x_min), max(40, int(d.y_min) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 100, 0),
                4,
            )
        return out


class YOLOFineTunedDetector(YOLODetector):
    """Backward-compatible alias for the old fine-tuned detector name."""

    def __init__(
        self,
        weights: str | None = None,
        confidence: float = 0.05,
        process_width: int = 1600,
    ):
        super().__init__(
            model_name=weights or "models/pricetag_yolov8n.pt",
            confidence=confidence,
            process_width=process_width,
        )


class HybridDetector:
    """YOLO tiled detector + MSER fallback, merged by NMS.

    It improves recall without making the project unusable when model weights are
    not bundled in the archive.
    """

    def __init__(
        self,
        primary: object | None = None,
        fallback: object | None = None,
        nms_iou: float = 0.40,
    ):
        self.primary = primary
        self.fallback = fallback or MSERDetector()
        self.nms_iou = nms_iou
        self._primary_init_attempted = False

    def _get_primary(self):
        if self.primary is not None or self._primary_init_attempted:
            return self.primary
        self._primary_init_attempted = True
        try:
            from shelf.detect.yolo_sahi import YOLOSahiDetector

            self.primary = YOLOSahiDetector()
        except Exception as exc:
            logger.warning("YOLO tiled недоступен: %s", exc)
            self.primary = None
        return self.primary

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets: list[Detection] = []
        primary = self._get_primary()
        if primary is not None:
            try:
                dets.extend(primary.detect(frame))
            except Exception as exc:
                logger.warning(
                    "primary detector failed, using fallback only: %s", exc
                )
        try:
            dets.extend(self.fallback.detect(frame))
        except Exception as exc:
            logger.warning("fallback detector failed: %s", exc)
        return _nms(dets, iou_thr=self.nms_iou)

    def visualize(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            cv2.rectangle(
                out, (int(d.x_min), int(d.y_min)), (int(d.x_max), int(d.y_max)), (0, 220, 255), 6
            )
            cv2.putText(
                out,
                f"hybrid {d.confidence:.2f}",
                (int(d.x_min), max(40, int(d.y_min) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 220, 255),
                4,
            )
        return out


def _price_tag_color_score(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Orange/red/yellow/green labels.
    masks = [
        cv2.inRange(hsv, np.array([0, 60, 70]), np.array([28, 255, 255])),
        cv2.inRange(hsv, np.array([20, 50, 90]), np.array([48, 255, 255])),
        cv2.inRange(hsv, np.array([35, 45, 60]), np.array([88, 255, 255])),
    ]
    frac = max(float(m.mean() / 255.0) for m in masks)
    return min(1.0, frac / 0.25)


def _text_edge_score(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    frac = float(edges.mean() / 255.0)
    return min(1.0, frac / 0.08)


def make_detector(name: str = "hybrid"):
    """Фабрика детекторов.

    Supported: ``hybrid`` | ``yolo-tiled`` | ``yolo-ft`` | ``mser`` | ``yolo``.
    Environment variable ``SHELF_DETECTOR`` overrides the argument.
    """
    import os

    name = os.environ.get("SHELF_DETECTOR", name).lower()
    if name == "hybrid":
        return HybridDetector()
    if name == "yolo-tiled":
        from shelf.detect.yolo_sahi import YOLOSahiDetector

        return YOLOSahiDetector()
    if name == "yolo-ft":
        return YOLOFineTunedDetector()
    if name == "yolo":
        return YOLODetector()
    if name == "mser":
        return MSERDetector()
    raise ValueError(f"Unknown detector: {name}")


PriceTagDetector = HybridDetector
