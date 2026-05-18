"""OCR engine abstraction with local PaddleOCR/EasyOCR backends.

Configuration is controlled by environment variables:

- ``SHELF_OCR_ENGINE=auto|paddle_v4|paddle_v5|easyocr|none``
- ``SHELF_OCR_AUTO_ORDER=paddle_v4,paddle_v5,easyocr``
- ``SHELF_PADDLE_DET_MODEL_DIR`` / ``SHELF_PADDLE_REC_MODEL_DIR`` /
  ``SHELF_PADDLE_CLS_MODEL_DIR`` for fully local Paddle model placement.

No cloud OCR API is used.  PP-OCRv5 is optional and selected only when available
or explicitly requested; stable PP-OCRv4 remains in the default auto order until
local challenge metrics prove v5 is better on the five labeled videos.
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_VALID_ENGINES = {"auto", "paddle_v4", "paddle_v5", "easyocr", "none"}
_DEFAULT_AUTO_ORDER = ("paddle_v4", "paddle_v5", "easyocr")


def normalize_ocr_engine_name(value: str | None) -> str:
    """Normalize OCR engine selection from config/env."""
    name = (
        (value or os.getenv("SHELF_OCR_ENGINE", "auto"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    aliases = {
        "paddle": "paddle_v4",
        "paddle4": "paddle_v4",
        "ppocr4": "paddle_v4",
        "pp_ocr_v4": "paddle_v4",
        "paddle5": "paddle_v5",
        "ppocr5": "paddle_v5",
        "pp_ocr_v5": "paddle_v5",
        "off": "none",
        "disabled": "none",
    }
    name = aliases.get(name, name)
    return name if name in _VALID_ENGINES else "auto"


def _auto_order() -> list[str]:
    raw = os.getenv("SHELF_OCR_AUTO_ORDER", ",".join(_DEFAULT_AUTO_ORDER))
    order = [normalize_ocr_engine_name(x) for x in raw.split(",")]
    return [
        x for x in order if x in {"paddle_v4", "paddle_v5", "easyocr"}
    ] or list(_DEFAULT_AUTO_ORDER)


def _paddle_model_dir_kwargs() -> dict[str, str]:
    """Collect optional local Paddle model directories from env."""
    mapping = {
        "det_model_dir": os.getenv("SHELF_PADDLE_DET_MODEL_DIR"),
        "rec_model_dir": os.getenv("SHELF_PADDLE_REC_MODEL_DIR"),
        "cls_model_dir": os.getenv("SHELF_PADDLE_CLS_MODEL_DIR"),
    }
    return {k: v for k, v in mapping.items() if v}


class OCREngine:
    """Lazy OCR backend wrapper.

    PaddleOCR is preferred for digits/prices.  EasyOCR is a local fallback and is
    still useful for Russian product names.  The wrapper avoids importing heavy
    OCR packages until the first inference call.
    """

    _easyocr_lock = Lock()

    def __init__(
        self,
        lang: str = "en",
        force_easyocr: bool = False,
        engine: str | None = None,
    ) -> None:
        self.lang = lang
        self.force_easyocr = force_easyocr
        self.engine_name = (
            "easyocr" if force_easyocr else normalize_ocr_engine_name(engine)
        )
        self._ocr: Any | None = None
        self._backend: str | None = (
            None  # paddle_v4 | paddle_v5 | easyocr | none
        )

    def _load(self) -> None:
        if self.engine_name == "none":
            self._backend = "none"
            self._ocr = None
            logger.info("OCR backend disabled by SHELF_OCR_ENGINE=none")
            return

        order = (
            [self.engine_name] if self.engine_name != "auto" else _auto_order()
        )
        for backend in order:
            if backend.startswith("paddle"):
                if self._try_load_paddle(backend):
                    return
            elif backend == "easyocr":
                if self._try_load_easyocr():
                    return

        self._ocr = None
        self._backend = "none"
        logger.warning("No local OCR backend could be loaded; OCR disabled")

    def _try_load_paddle(self, backend: str) -> bool:
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            logger.info("PaddleOCR import failed for %s: %s", backend, exc)
            return False

        kwargs = _paddle_model_dir_kwargs()
        attempts = self._paddle_constructor_attempts(backend, kwargs)
        for attempt in attempts:
            try:
                self._ocr = PaddleOCR(**attempt)
                self._backend = backend
                logger.info(
                    "OCR backend: PaddleOCR %s (lang=%s)", backend, self.lang
                )
                return True
            except TypeError as exc:
                logger.debug(
                    "PaddleOCR constructor rejected %s: %s", attempt, exc
                )
            except Exception as exc:
                logger.info(
                    "PaddleOCR %s unavailable with %s: %s",
                    backend,
                    attempt,
                    exc,
                )
        return False

    def _paddle_constructor_attempts(
        self, backend: str, model_dirs: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Return constructor kwargs from modern to legacy PaddleOCR APIs."""
        version = "PP-OCRv5" if backend == "paddle_v5" else "PP-OCRv4"
        common: dict[str, Any] = {"lang": self.lang, **model_dirs}
        attempts: list[dict[str, Any]] = [
            {
                **common,
                "ocr_version": version,
                "use_textline_orientation": True,
            },
            {
                **common,
                "ocr_version": version,
                "use_angle_cls": True,
                "show_log": False,
            },
            {**common, "use_textline_orientation": True},
            {**common, "use_angle_cls": True, "show_log": False},
        ]
        if backend == "paddle_v5":
            attempts.insert(
                0,
                {
                    **common,
                    "ocr_version": "PP-OCRv5",
                    "text_detection_model_name": "PP-OCRv5_mobile_det",
                    "text_recognition_model_name": "PP-OCRv5_mobile_rec",
                    "use_textline_orientation": True,
                },
            )
        return attempts

    def _try_load_easyocr(self) -> bool:
        try:
            import easyocr

            langs = ["ru", "en"] if self.lang == "ru" else ["en"]
            with self._easyocr_lock:
                self._ocr = easyocr.Reader(langs, verbose=False)
            self._backend = "easyocr"
            logger.info("OCR backend: EasyOCR (langs=%s)", langs)
            return True
        except Exception as exc:
            logger.info("EasyOCR unavailable: %s", exc)
            return False

    def run(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        """Return a list of ``(box, text, confidence)`` tuples."""
        if image is None or image.size == 0:
            return []
        if self._backend is None:
            self._load()

        if self._backend == "none" or self._ocr is None:
            return []

        try:
            if str(self._backend).startswith("paddle"):
                return self._run_paddle(image)
            return self._run_easyocr(image)
        except Exception as exc:
            logger.warning("OCR inference failed (%s)", exc)
            return []

    def _run_paddle(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        if hasattr(self._ocr, "ocr"):
            try:
                results = self._ocr.ocr(image, cls=True)
            except TypeError:
                results = self._ocr.ocr(image)
        elif hasattr(self._ocr, "predict"):
            results = self._ocr.predict(image)
        else:
            return []
        return self._parse_paddle_results(results)

    def _parse_paddle_results(
        self, results: Any
    ) -> list[tuple[list, str, float]]:
        out: list[tuple[list, str, float]] = []
        if not results:
            return out
        for page in results if isinstance(results, list) else [results]:
            if page is None:
                continue
            if isinstance(page, dict):
                out.extend(self._parse_paddle_dict(page))
                continue
            for item in page or []:
                if item is None:
                    continue
                if isinstance(item, dict):
                    out.extend(self._parse_paddle_dict(item))
                    continue
                try:
                    box, payload = item
                    if not payload:
                        continue
                    text, conf = payload
                    out.append((box, str(text), float(conf)))
                except Exception:
                    continue
        return out

    @staticmethod
    def _parse_paddle_dict(
        page: dict[str, Any],
    ) -> list[tuple[list, str, float]]:
        texts = page.get("rec_texts") or page.get("texts") or []
        scores = page.get("rec_scores") or page.get("scores") or []
        boxes = (
            page.get("dt_polys")
            or page.get("rec_polys")
            or page.get("boxes")
            or []
        )
        out: list[tuple[list, str, float]] = []
        for idx, text in enumerate(texts):
            box = (
                boxes[idx].tolist()
                if idx < len(boxes) and hasattr(boxes[idx], "tolist")
                else (boxes[idx] if idx < len(boxes) else [])
            )
            conf = float(scores[idx]) if idx < len(scores) else 0.0
            out.append((box, str(text), conf))
        return out

    def _run_easyocr(self, image: np.ndarray) -> list[tuple[list, str, float]]:
        results = self._ocr.readtext(image, detail=1)
        out: list[tuple[list, str, float]] = []
        for box_pts, text, conf in results:
            out.append((box_pts, str(text), float(conf)))
        return out

    def run_texts(self, image: np.ndarray) -> list[str]:
        return [text for _, text, _ in self.run(image)]
