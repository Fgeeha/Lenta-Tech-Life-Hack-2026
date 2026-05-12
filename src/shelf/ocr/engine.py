"""PaddleOCR-обёртка (заглушка)."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(self) -> None:
        self._ocr = None

    def _load(self) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(use_angle_cls=True, lang="ru", show_log=False)
        logger.info("PaddleOCR инициализирован")

    def run(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        """Вернуть список (box, text, confidence)."""
        if self._ocr is None:
            self._load()
        results = self._ocr.ocr(image, cls=True)
        out = []
        for line in results or [[]]:
            for item in line or []:
                box, (text, conf) = item
                out.append((box, text, conf))
        return out
