"""QR/barcode decoder and Lenta QR query parser."""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qsl, unquote_plus, urlparse

import cv2
import numpy as np

from shelf.qr.barcode_roi import ean13_repair

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


def _normalize_barcode(raw: str) -> str:
    s = re.sub(r"\D", "", str(raw))
    # QR payload is usually authoritative. Do not brute-force one-digit repair on
    # 13-digit values: it may "fix" synthetic/test barcodes into a different code.
    if len(s) == 12:
        return ean13_repair(s) or s
    if len(s) == 14:
        return ean13_repair(s) or s
    return s


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


def parse_qr_url(url: str) -> dict[str, str]:
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
            val = _normalize_barcode(val)
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


def _raw_to_fields(raw: str) -> dict[str, str]:
    raw = str(raw).strip()
    if not raw:
        return {}
    digits = re.sub(r"\D", "", raw)
    if raw.isdigit() and 8 <= len(raw) <= 15:
        return {"qr_code_barcode": _normalize_barcode(raw)}
    # Some scanners return only a query without scheme.
    parsed = parse_qr_url(raw)
    if parsed:
        return parsed
    # Last chance: a barcode embedded in arbitrary scanner text.
    if 12 <= len(digits) <= 14:
        bc = _normalize_barcode(digits)
        if bc:
            return {"qr_code_barcode": bc}
    return {}


def decode_qr(crop: np.ndarray) -> dict[str, str]:
    """Read QR/barcode from a price-tag crop."""
    if crop is None or crop.size == 0:
        return {}

    # Fast pass on the original crop before generating many variants.
    for raw in _try_pyzbar(crop) + _try_opencv(crop):
        parsed = _raw_to_fields(raw)
        if parsed:
            return parsed

    for img in _image_variants(crop):
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
    if crop is None or crop.size == 0:
        return ""
    for img in _image_variants(crop):
        for raw in _try_pyzbar(img):
            digits = re.sub(r"\D", "", raw)
            if 8 <= len(digits) <= 15:
                return _normalize_barcode(digits)
    return ""
