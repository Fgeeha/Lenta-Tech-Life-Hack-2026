"""Sliced YOLO inference (SAHI-style, без внешней зависимости).

Разрезает 4K-кадр на тайлы 640×640 (stride=512, overlap=128),
прогоняет tiled YOLO на каждом тайле, объединяет детекции с NMS.

Ценники 180-380px хорошо видны в тайле (28-59% ширины).
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from shelf.detect.detector import Detection, _nms

logger = logging.getLogger(__name__)

TILE_SIZE = 640
STRIDE = 512          # overlap = 128px
TILED_WEIGHTS = "models/pricetag_tiled_yolov8n.pt"


class YOLOSahiDetector:
    """Tile-based YOLO detector с слитой inferencing (без SAHI)."""

    def __init__(
        self,
        weights: str | None = None,
        confidence: float = 0.25,
        tile_size: int = TILE_SIZE,
        stride: int = STRIDE,
        nms_iou: float = 0.4,
    ):
        self.weights = weights or TILED_WEIGHTS
        self.confidence = confidence
        self.tile_size = tile_size
        self.stride = stride
        self.nms_iou = nms_iou
        self._model = None

    def _load(self) -> None:
        from ultralytics import YOLO

        w = Path(self.weights)
        if not w.exists():
            logger.warning("Веса не найдены: %s → фоллбек на yolov8n.pt", w)
            self.weights = "yolov8n.pt"
        self._model = YOLO(self.weights)
        logger.info("YOLOSahi загружен: %s", self.weights)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            self._load()

        h, w = frame.shape[:2]
        all_dets: list[Detection] = []

        # Генерируем тайлы
        ys = list(range(0, max(1, h - self.tile_size + 1), self.stride))
        xs = list(range(0, max(1, w - self.tile_size + 1), self.stride))
        # Добавляем последний тайл если не дошли до края
        if ys and ys[-1] + self.tile_size < h:
            ys.append(max(0, h - self.tile_size))
        if xs and xs[-1] + self.tile_size < w:
            xs.append(max(0, w - self.tile_size))

        for ty in ys:
            for tx in xs:
                tile = frame[ty:ty + self.tile_size, tx:tx + self.tile_size]
                if tile.shape[0] == 0 or tile.shape[1] == 0:
                    continue

                results = self._model(tile, verbose=False, conf=self.confidence)[0]
                for box in results.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    # Координаты в оригинальном кадре
                    det = Detection(
                        x_min=int(bx1 + tx),
                        y_min=int(by1 + ty),
                        x_max=int(bx2 + tx),
                        y_max=int(by2 + ty),
                        confidence=conf,
                        cls_name="tag",
                    )
                    all_dets.append(det)

        # Глобальный NMS по всем тайлам
        result = _nms(all_dets, iou_thr=self.nms_iou)
        logger.debug("YOLOSahi: %d raw → %d после NMS (%d тайлов)",
                     len(all_dets), len(result), len(ys) * len(xs))
        return result

    def visualize(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            cv2.rectangle(out, (d.x_min, d.y_min), (d.x_max, d.y_max), (0, 200, 255), 6)
            cv2.putText(out, f"tiled {d.confidence:.2f}",
                        (d.x_min, max(40, d.y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 4)
        return out
