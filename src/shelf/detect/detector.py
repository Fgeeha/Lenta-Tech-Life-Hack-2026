"""YOLO-детектор ценников (заглушка)."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float


class PriceTagDetector:
    """YOLOv8-детектор ценников. Инициализируется лениво."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.model_name)
        logger.info("Модель загружена: %s", self.model_name)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            self._load()
        results = self._model(frame, verbose=False)[0]
        detections: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append(Detection(int(x1), int(y1), int(x2), int(y2), conf))
        return detections
