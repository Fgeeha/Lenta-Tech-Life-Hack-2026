"""Sliced YOLO inference (SAHI-style, without the external SAHI dependency)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from shelf.detect.detector import Detection, _nms

logger = logging.getLogger(__name__)

TILE_SIZE = 640
STRIDE = 512  # overlap = 128px
TILED_WEIGHTS = "models/pricetag_tiled_yolov8n.pt"
HF_MODEL_REPO = "fgeeha/shelf-pricetag-yolov8n"
HF_MODEL_FILE = "pricetag_tiled_yolov8n.pt"


def _ensure_weights(path: str) -> str | None:
    """Return trained weights path or None.

    We intentionally do not fall back to generic COCO ``yolov8n.pt``: it produces
    confident but irrelevant objects and hurts downstream OCR.
    """
    p = Path(path)
    if p.exists():
        return str(p)
    try:
        from huggingface_hub import hf_hub_download

        logger.info("Скачиваем веса с HF Hub: %s", HF_MODEL_REPO)
        p.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILE,
            local_dir=str(p.parent),
        )
        logger.info("Веса скачаны: %s", downloaded)
        return downloaded
    except Exception as exc:
        logger.warning(
            "Trained YOLO weights unavailable (%s). Use hybrid/mser fallback or place weights at %s",
            exc,
            p,
        )
        return None


class YOLOSahiDetector:
    """Tile-based YOLO detector for small price tags in 4K frames."""

    def __init__(
        self,
        weights: str | None = None,
        confidence: float = 0.18,
        tile_size: int = TILE_SIZE,
        stride: int = STRIDE,
        nms_iou: float = 0.42,
        batch_size: int = 4,
    ):
        self.weights = weights or TILED_WEIGHTS
        self.confidence = confidence
        self.tile_size = tile_size
        self.stride = stride
        self.nms_iou = nms_iou
        self.batch_size = batch_size
        self._model = None
        self._disabled = False

    def _load(self) -> None:
        weights = _ensure_weights(self.weights)
        if weights is None:
            self._disabled = True
            return
        from ultralytics import YOLO

        self.weights = weights
        self._model = YOLO(self.weights)
        logger.info("YOLOSahi загружен: %s", self.weights)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._disabled:
            return []
        if self._model is None:
            self._load()
            if self._disabled:
                return []

        h, w = frame.shape[:2]
        tiles: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []

        ys = list(range(0, max(1, h - self.tile_size + 1), self.stride))
        xs = list(range(0, max(1, w - self.tile_size + 1), self.stride))
        if ys and ys[-1] + self.tile_size < h:
            ys.append(max(0, h - self.tile_size))
        if xs and xs[-1] + self.tile_size < w:
            xs.append(max(0, w - self.tile_size))

        for ty in ys:
            for tx in xs:
                tile = frame[ty : ty + self.tile_size, tx : tx + self.tile_size]
                if tile.shape[0] == 0 or tile.shape[1] == 0:
                    continue
                tiles.append(tile)
                offsets.append((tx, ty))

        all_dets: list[Detection] = []
        for start in range(0, len(tiles), self.batch_size):
            batch = tiles[start : start + self.batch_size]
            batch_offsets = offsets[start : start + self.batch_size]
            results = self._model(batch, verbose=False, conf=self.confidence)
            for result, (tx, ty) in zip(results, batch_offsets):
                for box in result.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    det = Detection(
                        x_min=max(0, int(bx1 + tx)),
                        y_min=max(0, int(by1 + ty)),
                        x_max=min(w, int(bx2 + tx)),
                        y_max=min(h, int(by2 + ty)),
                        confidence=conf,
                        cls_name="tag",
                    )
                    if det.area > 0:
                        all_dets.append(det)

        result = _nms(all_dets, iou_thr=self.nms_iou)
        logger.debug(
            "YOLOSahi: %d raw -> %d after NMS (%d tiles)",
            len(all_dets),
            len(result),
            len(tiles),
        )
        return result

    def visualize(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            cv2.rectangle(
                out, (d.x_min, d.y_min), (d.x_max, d.y_max), (0, 200, 255), 6
            )
            cv2.putText(
                out,
                f"tiled {d.confidence:.2f}",
                (d.x_min, max(40, d.y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 200, 255),
                4,
            )
        return out
