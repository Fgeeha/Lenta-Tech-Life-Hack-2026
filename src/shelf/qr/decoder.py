"""QR/barcode decoder and Lenta QR query parser."""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import parse_qsl, unquote_plus, urlparse

import cv2
import numpy as np

from shelf.qr.barcode_roi import ean13_repair
from shelf.validation import normalize_ean13

logger = logging.getLogger(__name__)

# Canonical lower-case key -> schema field. ``parse_qr_url`` is case-insensitive.
_KEY_MAP: dict[str, str] = {
    # штрихкод
    "b": "qr_code_barcode",
    "barcode": "qr_code_barcode",
    # цены
    "p1": "price1_qr",
    "price1": "price1_qr",
    "p2": "price2_qr",
    "price2": "price2_qr",
    "p3": "price3_qr",
    "price3": "price3_qr",
    "p4": "price4_qr",
    "price4": "price4_qr",
    # оптовые пороги (уровень 1)
    "wl1c": "wholesale_level_1_count",
    "wholesalelevel1count": "wholesale_level_1_count",
    "wl1p": "wholesale_level_1_price",
    "wholesalelevel1price": "wholesale_level_1_price",
    # оптовые пороги (уровень 2)
    "wl2c": "wholesale_level_2_count",
    "wholesalelevel2count": "wholesale_level_2_count",
    "wl2p": "wholesale_level_2_price",
    "wholesalelevel2price": "wholesale_level_2_price",
    # акция
    "ap": "action_price_qr",
    "actionprice": "action_price_qr",
    "ac": "action_code_qr",
    "actioncode": "action_code_qr",
}

_PRICE_FIELDS = {
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
    "action_price_qr",
}

_DIGITS_RE = re.compile(r"\d+")


def _decode_mode() -> str:
    """Return QR/barcode decoding mode: full, fast or off."""
    mode = os.getenv("SHELF_CODE_DECODE_MODE", "full").strip().lower()
    return mode if mode in {"full", "fast", "off"} else "full"


def _variant_limit(default_full: int, default_fast: int) -> int:
    """Limit expensive image variants for smoke/HF-friendly runs."""
    raw = os.getenv("SHELF_CODE_MAX_VARIANTS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning("Invalid SHELF_CODE_MAX_VARIANTS=%r; ignoring", raw)
    return default_fast if _decode_mode() == "fast" else default_full


def _barcode_repair_enabled() -> bool:
    return os.getenv(
        "SHELF_ENABLE_BARCODE_REPAIR", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_barcode(raw: str, *, strict: bool = False) -> str:
    s = re.sub(r"\D", "", str(raw))
    if not strict:
        # Backward-compatible parser mode used by historical unit tests.  Do not
        # one-digit-repair already 13-digit payloads: QR values should not be
        # silently changed into another product.
        if len(s) in {12, 14}:
            return ean13_repair(s) or s
        return s

    # Production mode: keep only valid EAN-13. Optional repair is limited to
    # trusted scanner artifacts (12-digit prefix or 14-digit extra char).
    return normalize_ean13(
        s,
        allow_repair=_barcode_repair_enabled(),
        allow_append_12=_barcode_repair_enabled(),
        allow_drop_14=True,
        allow_one_digit_repair=False,
    )


def _normalize_price(raw: str) -> str:
    original = str(raw).strip()
    has_decimal = bool(re.search(r"[,.]\d{1,2}\b", original))
    text = original.replace("\u00a0", " ").replace(",", ".")
    text = re.sub(r"[^0-9.]", "", text)
    if text.count(".") > 1:
        parts = text.split(".")
        text = "".join(parts[:-1]) + "." + parts[-1]
    if not text:
        return ""
    try:
        value = float(text)
        # QR examples in GT use dot as decimal separator. Preserve .00 when it
        # was explicitly present in the QR payload; otherwise keep integer form.
        if has_decimal:
            return f"{value:.2f}"
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    except ValueError:
        return original


def parse_qr_url(url: str, *, strict_barcode: bool = False) -> dict[str, str]:
    """Parse QR data into output schema fields.

    Supports full URLs, ``?query`` strings and bare ``b=...&p1=...`` strings.
    Key matching is case-insensitive; both short and long names are accepted.
    """
    if not url:
        return {}

    raw = unquote_plus(str(url).strip())
    if not raw:
        return {}

    if not raw.startswith(("http://", "https://")):
        raw = "https://x" + raw if raw.startswith("?") else "https://x?" + raw

    try:
        parsed = urlparse(raw)
        query = parsed.query or parsed.path.split("?", 1)[-1]
        pairs = parse_qsl(query, keep_blank_values=False)
    except Exception:
        return {}

    result: dict[str, str] = {}
    for k, v in pairs:
        field = _KEY_MAP.get(k.strip().lower())
        if not field:
            continue
        val = v.strip()
        if not val:
            continue
        if field == "qr_code_barcode":
            val = _normalize_barcode(val, strict=strict_barcode)
        elif field in _PRICE_FIELDS:
            val = _normalize_price(val)
        else:
            # Counts and action codes: trim only.
            val = val.strip()
        if val:
            result[field] = val
    return result


def _try_pyzbar(image: np.ndarray) -> list[str]:
    try:
        from pyzbar import pyzbar

        decoded = pyzbar.decode(image)
        return [
            d.data.decode("utf-8", errors="ignore")
            for d in decoded
            if d.type in ("QRCODE", "EAN13", "EAN8", "CODE128")
        ]
    except Exception as exc:
        logger.debug("pyzbar error: %s", exc)
        return []


def _try_opencv(image: np.ndarray) -> list[str]:
    out: list[str] = []
    try:
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(image)
        if data:
            out.append(data)
        # OpenCV also supports multi-QR in many builds.
        try:
            ok, decoded_info, _, _ = detector.detectAndDecodeMulti(image)
            if ok:
                out.extend([x for x in decoded_info if x])
        except Exception:
            pass
    except Exception as exc:
        logger.debug("opencv QR error: %s", exc)
    return out


def _try_qreader(image: np.ndarray) -> list[str]:
    try:
        from qreader import QReader

        reader = QReader()
        decoded = reader.detect_and_decode(image=image)
        return [x for x in decoded if x]
    except Exception as exc:
        logger.debug("qreader error: %s", exc)
        return []


def _image_variants(crop: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    rotations = (
        None,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_CLOCKWISE,
    )
    for rot in rotations:
        base = cv2.rotate(crop, rot) if rot is not None else crop
        for scale in (1.0, 1.5, 2.0, 3.0, 0.75):
            if scale != 1.0:
                h, w = base.shape[:2]
                img = cv2.resize(
                    base,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                img = base
            variants.append(img)
            gray = (
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            )
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
                gray
            )
            variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            _, otsu = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    return variants


def _roi_boxes(width: int, height: int) -> list[tuple[str, int, int, int, int]]:
    """Geometric QR/barcode zones inside an oriented price-tag crop."""
    w, h = max(1, width), max(1, height)
    return [
        ("qr_top_right", int(w * 0.48), 0, w, int(h * 0.58)),
        ("qr_right_center", int(w * 0.50), int(h * 0.18), w, int(h * 0.82)),
        ("qr_upper_band", int(w * 0.35), 0, w, int(h * 0.45)),
        ("barcode_bottom", 0, int(h * 0.55), w, h),
        ("barcode_bottom_right", int(w * 0.35), int(h * 0.50), w, h),
        ("center_right", int(w * 0.42), int(h * 0.25), w, int(h * 0.75)),
    ]


def _enhance_code_roi(roi: np.ndarray) -> list[np.ndarray]:
    """Return lightweight ROI variants for QR/linear barcode decoding."""
    if roi is None or roi.size == 0:
        return []
    variants: list[np.ndarray] = []
    for scale in (1.0, 2.0, 3.0, 4.0):
        if scale == 1.0:
            img = roi
        else:
            h, w = roi.shape[:2]
            img = cv2.resize(
                roi,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        variants.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))
        _, otsu = cv2.threshold(
            sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    return variants


def _roi_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Generate targeted code ROIs before expensive full-crop decoding."""
    if crop is None or crop.size == 0:
        return []
    out: list[np.ndarray] = []
    rotations = (
        None,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_CLOCKWISE,
    )
    for rot in rotations:
        base = cv2.rotate(crop, rot) if rot is not None else crop
        h, w = base.shape[:2]
        min_w = max(12, int(w * 0.08))
        min_h = max(12, int(h * 0.08))
        for _, x1, y1, x2, y2 in _roi_boxes(w, h):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < min_w or y2 - y1 < min_h:
                continue
            roi = base[y1:y2, x1:x2]
            out.extend(_enhance_code_roi(roi))
    return out


def _barcode_roi_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Return bottom/right barcode-oriented ROIs for pyzbar."""
    if crop is None or crop.size == 0:
        return []
    out: list[np.ndarray] = []
    rotations = (
        None,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_CLOCKWISE,
    )
    for rot in rotations:
        base = cv2.rotate(crop, rot) if rot is not None else crop
        h, w = base.shape[:2]
        for name, x1, y1, x2, y2 in _roi_boxes(w, h):
            if not name.startswith("barcode"):
                continue
            roi = base[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
            out.extend(_enhance_code_roi(roi))
    return out


def _raw_to_fields(raw: str) -> dict[str, str]:
    raw = str(raw).strip()
    if not raw:
        return {}
    digits = re.sub(r"\D", "", raw)
    if raw.isdigit() and 8 <= len(raw) <= 15:
        bc = _normalize_barcode(raw, strict=True)
        return {"qr_code_barcode": bc} if bc else {}
    # Some scanners return only a query without scheme.
    parsed = parse_qr_url(raw, strict_barcode=True)
    if parsed:
        return parsed
    # Last chance: a barcode embedded in arbitrary scanner text.
    if 12 <= len(digits) <= 14:
        bc = _normalize_barcode(digits, strict=True)
        if bc:
            return {"qr_code_barcode": bc}
    return {}


def decode_qr(crop: np.ndarray) -> dict[str, str]:
    """Read QR/barcode from a price-tag crop."""
    if crop is None or crop.size == 0 or _decode_mode() == "off":
        return {}

    # Fast pass on the original crop before generating many variants.
    for raw in _try_pyzbar(crop) + _try_opencv(crop):
        parsed = _raw_to_fields(raw)
        if parsed:
            return parsed

    roi_limit = _variant_limit(default_full=10_000, default_fast=24)
    # Geometric ROI pass: cheaper than processing dozens of full-crop variants
    # and often enough for tiny QR/barcodes in the right/bottom price-tag zones.
    for img in _roi_variants(crop)[:roi_limit]:
        for raw in _try_pyzbar(img) + _try_opencv(img):
            parsed = _raw_to_fields(raw)
            if parsed:
                return parsed

    if _decode_mode() == "fast":
        return {}

    full_limit = _variant_limit(default_full=10_000, default_fast=12)
    for img in _image_variants(crop)[:full_limit]:
        for raw in _try_pyzbar(img) + _try_opencv(img):
            parsed = _raw_to_fields(raw)
            if parsed:
                return parsed

    # QReader is heavier; call it once per crop on the two most useful variants.
    for img in _image_variants(crop)[:2]:
        for raw in _try_qreader(img):
            parsed = _raw_to_fields(raw)
            if parsed:
                return parsed

    return {}


def decode_barcode(crop: np.ndarray) -> str:
    """Read linear barcode from a crop. Returns a normalized string or ''."""
    if crop is None or crop.size == 0 or _decode_mode() == "off":
        return ""
    roi_limit = _variant_limit(default_full=10_000, default_fast=24)
    for img in _barcode_roi_variants(crop)[:roi_limit]:
        for raw in _try_pyzbar(img):
            digits = re.sub(r"\D", "", raw)
            if 8 <= len(digits) <= 15:
                normalized = _normalize_barcode(digits, strict=True)
                if normalized:
                    return normalized
    if _decode_mode() == "fast":
        return ""
    full_limit = _variant_limit(default_full=10_000, default_fast=12)
    for img in _image_variants(crop)[:full_limit]:
        for raw in _try_pyzbar(img):
            digits = re.sub(r"\D", "", raw)
            if 8 <= len(digits) <= 15:
                normalized = _normalize_barcode(digits, strict=True)
                if normalized:
                    return normalized
    return ""
