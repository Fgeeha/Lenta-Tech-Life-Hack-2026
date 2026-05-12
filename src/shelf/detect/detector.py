"""Детектор ценников.

Используем MSER (Maximally Stable Extremal Regions) — специализирован для
нахождения стабильных прямоугольных регионов (текст, напечатанные материалы).
Даёт recall ~50% без обучения; с трекером (этап 4) recall возрастает.

YOLODetector — резерв для псевдо-лейблов (этап 12).
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ASPECT_MIN = 0.2
_ASPECT_MAX = 2.0
_MIN_AREA_ORIG = 6_000  # px² в оригинальных координатах
_MAX_AREA_ORIG = 150_000
_MIN_W = 40  # px в оригинале
_MIN_H = 80  # px в оригинале


@dataclass
class Detection:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    cls_name: str = ""

    @property
    def area(self) -> int:
        return max(0, self.x_max - self.x_min) * max(0, self.y_max - self.y_min)

    @property
    def aspect(self) -> float:
        h = self.y_max - self.y_min
        return (self.x_max - self.x_min) / h if h > 0 else 0.0


def _nms(dets: list["Detection"], iou_thr: float = 0.35) -> list["Detection"]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    keep: list["Detection"] = []
    for d in dets:
        dominated = False
        for k in keep:
            ix = max(0, min(d.x_max, k.x_max) - max(d.x_min, k.x_min))
            iy = max(0, min(d.y_max, k.y_max) - max(d.y_min, k.y_min))
            inter = ix * iy
            union = d.area + k.area - inter
            if union > 0 and inter / union > iou_thr:
                dominated = True
                break
        if not dominated:
            keep.append(d)
    return keep


class MSERDetector:
    """MSER-детектор: находит стабильные прямоугольные регионы (ценники, этикетки)."""

    def __init__(
        self,
        process_width: int = 1280,
        delta: int = 5,
        min_area_scaled: int = 800,
        max_area_scaled: int = 10_000,
        max_variation: float = 0.25,
    ):
        self.process_width = process_width
        self._mser = cv2.MSER_create(
            delta=delta,
            min_area=min_area_scaled,
            max_area=max_area_scaled,
            max_variation=max_variation,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.process_width / max(w, h))
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, bboxes = self._mser.detectRegions(gray)

        dets: list[Detection] = []
        for x, y, bw, bh in bboxes:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_w = int(bw / scale)
            orig_h = int(bh / scale)
            orig_area = orig_w * orig_h
            ar = orig_w / orig_h if orig_h > 0 else 0.0

            if not (_ASPECT_MIN <= ar <= _ASPECT_MAX):
                continue
            if orig_area < _MIN_AREA_ORIG or orig_area > _MAX_AREA_ORIG:
                continue
            if orig_w < _MIN_W or orig_h < _MIN_H:
                continue

            score = min(1.0, orig_area / 60_000)
            dets.append(
                Detection(
                    x_min=orig_x,
                    y_min=orig_y,
                    x_max=orig_x + orig_w,
                    y_max=orig_y + orig_h,
                    confidence=score,
                    cls_name="tag",
                )
            )

        result = _nms(dets)
        logger.debug("MSERDetector: %d raw → %d after NMS", len(dets), len(result))
        return result

    def visualize(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            cv2.rectangle(out, (d.x_min, d.y_min), (d.x_max, d.y_max), (0, 255, 0), 6)
            cv2.putText(
                out,
                f"tag {d.confidence:.2f}",
                (d.x_min, max(40, d.y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                4,
            )
        return out


class YOLODetector:
    """YOLOv8 детектор (этап 12, псевдо-лейблы)."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.3, process_width: int = 1280):
        self.model_name = model_name
        self.confidence = confidence
        self.process_width = process_width
        self._model = None

    def _load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.model_name)
        logger.info("YOLO загружен: %s", self.model_name)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            self._load()
        h, w = frame.shape[:2]
        scale = min(1.0, self.process_width / max(w, h))
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        results = self._model(small, verbose=False, conf=self.confidence)[0]
        dets: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_name = results.names.get(int(box.cls[0]), "")
            det = Detection(int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale), conf, cls_name)
            if det.area >= _MIN_AREA_ORIG and _ASPECT_MIN <= det.aspect <= _ASPECT_MAX:
                dets.append(det)
        return _nms(dets)

    def visualize(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            cv2.rectangle(out, (d.x_min, d.y_min), (d.x_max, d.y_max), (255, 100, 0), 6)
            cv2.putText(
                out,
                f"{d.cls_name} {d.confidence:.2f}",
                (d.x_min, max(40, d.y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 100, 0),
                4,
            )
        return out


# Алиас по умолчанию — MSER без обучения
PriceTagDetector = MSERDetector
