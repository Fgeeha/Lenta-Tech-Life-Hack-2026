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

# Local trained weights that are sometimes bundled in challenge archives.  These
# paths keep inference fully local and avoid silently downloading a model.
_LOCAL_WEIGHT_CANDIDATES = (
    TILED_WEIGHTS,
    "runs/detect/runs/detect/pricetag_tiled_v1/weights/best.pt",
    "runs/detect/runs/detect/pricetag_tiled_v1/weights/last.pt",
    "runs/detect/runs/detect/pricetag_v1/weights/best.pt",
    "runs/detect/runs/detect/pricetag_v1/weights/last.pt",
)


def _env_true(name: str, default: str = "false") -> bool:
    """Return True for truthy environment switches."""
    import os

    return os.getenv(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _candidate_weight_paths(path: str) -> list[Path]:
    """Return local weight candidates in priority order without duplicates."""
    import os

    raw: list[str] = []
    env_path = os.getenv("SHELF_YOLO_WEIGHTS", "").strip()
    if env_path:
        raw.append(env_path)
    raw.append(path)
    raw.extend(_LOCAL_WEIGHT_CANDIDATES)

    seen: set[str] = set()
    out: list[Path] = []
    for item in raw:
        p = Path(item)
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _ensure_weights(path: str) -> str | None:
    """Return a local trained weights path or None.

    The detector never falls back to generic COCO ``yolov8n.pt`` and, by default,
    never downloads anything from the network.  To opt into downloading trained
    weights for a local experiment, set ``SHELF_ALLOW_MODEL_DOWNLOAD=true``.
    """
    for candidate in _candidate_weight_paths(path):
        if candidate.exists():
            logger.info("Using local YOLO weights: %s", candidate)
            return str(candidate)

    p = Path(path)
    if not _env_true("SHELF_ALLOW_MODEL_DOWNLOAD"):
        logger.warning(
            "Trained YOLO weights unavailable locally. Place weights at %s or set "
            "SHELF_YOLO_WEIGHTS; using hybrid/mser fallback.",
            p,
        )
        return None

    try:
        from huggingface_hub import hf_hub_download

        logger.info(
            "Downloading trained YOLO weights from HF Hub: %s", HF_MODEL_REPO
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILE,
            local_dir=str(p.parent),
        )
        logger.info("Weights downloaded: %s", downloaded)
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
                        x_min=max(0.0, bx1 + tx),
                        y_min=max(0.0, by1 + ty),
                        x_max=min(float(w), bx2 + tx),
                        y_max=min(float(h), by2 + ty),
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
                out, (int(d.x_min), int(d.y_min)), (int(d.x_max), int(d.y_max)), (0, 200, 255), 6
            )
            cv2.putText(
                out,
                f"tiled {d.confidence:.2f}",
                (int(d.x_min), max(40, int(d.y_min) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 200, 255),
                4,
            )
        return out
