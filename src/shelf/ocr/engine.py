"""OCR-движок: PaddleOCR PP-OCRv4 EN (primary) / EasyOCR (fallback).

На Python 3.13 PaddlePaddle 2.x недоступен — используем EasyOCR.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class OCREngine:
    """PaddleOCR EN → EasyOCR fallback (HF Spaces / Python 3.13 compat).

    force_easyocr=True skips PaddleOCR and uses EasyOCR directly.
    Useful for Russian product names where PaddleOCR-RU underperforms.
    """

    def __init__(self, lang: str = "en", force_easyocr: bool = False) -> None:
        self.lang = lang
        self.force_easyocr = force_easyocr
        self._ocr = None
        self._backend = None  # "paddle" | "easyocr"

    def _load(self) -> None:
        if not self.force_easyocr:
            # Пробуем PaddleOCR (лучшее качество для чисел и EN)
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
                self._backend = "paddle"
                logger.info("OCR backend: PaddleOCR (lang=%s)", self.lang)
                return
            except Exception as exc:
                logger.warning("PaddleOCR недоступен (%s), переключаемся на EasyOCR", exc)

        # EasyOCR — лучше для RU продуктовых названий
        import easyocr
        langs = ["ru", "en"] if self.lang == "ru" else ["en"]
        self._ocr = easyocr.Reader(langs, verbose=False)
        self._backend = "easyocr"
        logger.info("OCR backend: EasyOCR (langs=%s)", langs)

    def run(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        """Вернуть список (box, text, confidence)."""
        if self._ocr is None:
            self._load()

        if self._backend == "paddle":
            return self._run_paddle(image)
        return self._run_easyocr(image)

    def _run_paddle(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        results = self._ocr.ocr(image, cls=True)
        out: list[tuple[list, str, float]] = []
        for page in results or []:
            for item in page or []:
                if item is None:
                    continue
                box, (text, conf) = item
                out.append((box, text, float(conf)))
        return out

    def _run_easyocr(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        results = self._ocr.readtext(image, detail=1)
        out: list[tuple[list, str, float]] = []
        for (box_pts, text, conf) in results:
            # EasyOCR box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            out.append((box_pts, text, float(conf)))
        return out

    def run_texts(self, image: np.ndarray) -> list[str]:
        return [text for _, text, _ in self.run(image)]
