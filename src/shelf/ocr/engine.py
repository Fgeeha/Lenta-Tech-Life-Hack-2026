"""OCR engine abstraction: PaddleOCR primary, EasyOCR fallback."""

from __future__ import annotations

import logging
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)


class OCREngine:
    """Lazy OCR backend wrapper.

    - PaddleOCR is preferred for digits/prices.
    - EasyOCR is used as fallback and for Russian product names.
    """

    _easyocr_lock = Lock()

    def __init__(self, lang: str = "en", force_easyocr: bool = False) -> None:
        self.lang = lang
        self.force_easyocr = force_easyocr
        self._ocr = None
        self._backend = None  # "paddle" | "easyocr"

    def _load(self) -> None:
        if not self.force_easyocr:
            try:
                from paddleocr import PaddleOCR

                # PaddleOCR changed parameter names across releases. Try the modern
                # call first, then the older one used by PP-OCRv4.
                try:
                    self._ocr = PaddleOCR(
                        use_textline_orientation=True, lang=self.lang
                    )
                except TypeError:
                    self._ocr = PaddleOCR(
                        use_angle_cls=True, lang=self.lang, show_log=False
                    )
                self._backend = "paddle"
                logger.info("OCR backend: PaddleOCR (lang=%s)", self.lang)
                return
            except Exception as exc:
                logger.warning(
                    "PaddleOCR недоступен (%s), переключаемся на EasyOCR", exc
                )

        try:
            import easyocr

            langs = ["ru", "en"] if self.lang == "ru" else ["en"]
            # EasyOCR model loading is not thread-safe in some environments.
            with self._easyocr_lock:
                self._ocr = easyocr.Reader(langs, verbose=False)
            self._backend = "easyocr"
            logger.info("OCR backend: EasyOCR (langs=%s)", langs)
        except Exception as exc:
            logger.warning("EasyOCR недоступен (%s); OCR отключён", exc)
            self._ocr = None
            self._backend = "none"

    def run(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        """Return a list of (box, text, confidence)."""
        if image is None or image.size == 0:
            return []
        if self._ocr is None:
            self._load()

        if self._backend == "none" or self._ocr is None:
            return []

        try:
            if self._backend == "paddle":
                return self._run_paddle(image)
            return self._run_easyocr(image)
        except Exception as exc:
            logger.warning("OCR inference failed (%s)", exc)
            return []

    def _run_paddle(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        results = self._ocr.ocr(image, cls=True)
        out: list[tuple[list, str, float]] = []
        for page in results or []:
            for item in page or []:
                if item is None:
                    continue
                box, payload = item
                if not payload:
                    continue
                text, conf = payload
                out.append((box, str(text), float(conf)))
        return out

    def _run_easyocr(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        results = self._ocr.readtext(image, detail=1)
        out: list[tuple[list, str, float]] = []
        for box_pts, text, conf in results:
            out.append((box_pts, str(text), float(conf)))
        return out

    def run_texts(self, image: np.ndarray) -> list[str]:
        return [text for _, text, _ in self.run(image)]
