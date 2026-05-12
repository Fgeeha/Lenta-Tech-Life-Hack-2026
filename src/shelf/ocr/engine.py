"""PaddleOCR-обёртка с ленивой инициализацией."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OCREngine:
    """PaddleOCR (PP-OCRv4), русский язык, ленивая загрузка."""

    def __init__(self, lang: str = "ru") -> None:
        self.lang = lang
        self._ocr = None

    def _load(self) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        logger.info("PaddleOCR инициализирован (lang=%s)", self.lang)

    def run(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        """Вернуть список (box, text, confidence) для всех найденных текстов.

        box — четыре точки [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        """
        if self._ocr is None:
            self._load()
        results = self._ocr.ocr(image, cls=True)
        out: list[tuple[list, str, float]] = []
        for page in results or []:
            for item in page or []:
                if item is None:
                    continue
                box, (text, conf) = item
                out.append((box, text, float(conf)))
        return out

    def run_texts(self, image: np.ndarray) -> list[str]:
        """Удобный метод — только тексты (без боксов и confidence)."""
        return [text for _, text, _ in self.run(image)]
