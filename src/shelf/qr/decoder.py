"""QR/barcode decoder and Lenta QR query parser."""

from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path
from urllib.parse import parse_qsl, unquote_plus, urlparse

import cv2
import numpy as np

from shelf.ocr.layout import roi_crops
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


def _debug_path(debug_dir: str | Path | None) -> Path | None:
    """Create and return an optional code-decoding debug directory."""
    if not debug_dir:
        return None
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _context_suffix(track_id: int | None, timestamp_ms: float | None) -> str:
    tid = "na" if track_id is None else str(track_id)
    ts = "na" if timestamp_ms is None else str(int(timestamp_ms))
    return f"track_{tid}_{ts}ms"


def _success_value(fields: dict[str, str]) -> str:
    for key in ("qr_code_barcode", "price1_qr", "price4_qr", "action_code_qr"):
        value = fields.get(key, "")
        if value:
            return value
    return "|".join(f"{k}={v}" for k, v in sorted(fields.items()))


def _log_success(
    debug_dir: str | Path | None,
    *,
    track_id: int | None,
    timestamp_ms: float | None,
    source: str,
    raw: str,
    normalized: str,
    roi_type: str,
) -> None:
    """Append one successful QR/barcode read to debug CSV when requested."""
    path = _debug_path(debug_dir)
    if path is None:
        return
    csv_path = path / "successful_code_reads.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "timestamp",
                "source",
                "raw",
                "normalized",
                "roi_type",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "track_id": "" if track_id is None else track_id,
                "timestamp": "" if timestamp_ms is None else int(timestamp_ms),
                "source": source,
                "raw": raw,
                "normalized": normalized,
                "roi_type": roi_type,
            }
        )


def _save_failed_roi_examples(
    debug_dir: str | Path | None,
    crop: np.ndarray,
    *,
    kind: str,
    track_id: int | None,
    timestamp_ms: float | None,
) -> None:
    """Save a small bounded set of failed template/geometric ROIs for inspection."""
    path = _debug_path(debug_dir)
    if path is None or crop is None or crop.size == 0:
        return
    out_dir = path / "failed_code_rois"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _context_suffix(track_id, timestamp_ms)
    samples = _named_template_rois(
        crop, {"qr", "barcode"}
    ) + _named_geometric_rois(crop, kind)
    for idx, (name, roi) in enumerate(samples[:12]):
        if roi is None or roi.size == 0:
            continue
        safe_name = re.sub(r"[^0-9A-Za-z_:-]+", "_", name)
        cv2.imwrite(
            str(out_dir / f"{kind}_{suffix}_{idx:02d}_{safe_name}.jpg"), roi
        )


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


_WECHAT_QR: "cv2.wechat_qrcode_WeChatQRCode | None" = None
_BARCODE_DETECTOR: "cv2.barcode.BarcodeDetector | None" = None


def _get_wechat_qr() -> "cv2.wechat_qrcode_WeChatQRCode | None":
    global _WECHAT_QR
    if _WECHAT_QR is None:
        try:
            _WECHAT_QR = cv2.wechat_qrcode_WeChatQRCode()
        except Exception as exc:
            logger.debug("WeChatQR init failed: %s", exc)
    return _WECHAT_QR


def _try_wechat_qr(image: np.ndarray) -> list[str]:
    """Decode QR codes using OpenCV WeChatQR (handles degraded/blurry codes better than pyzbar)."""
    reader = _get_wechat_qr()
    if reader is None:
        return []
    try:
        decoded, _ = reader.detectAndDecode(image)
        return [d for d in decoded if d]
    except Exception as exc:
        logger.debug("WeChatQR decode error: %s", exc)
        return []


def _get_barcode_detector() -> "cv2.barcode.BarcodeDetector | None":
    global _BARCODE_DETECTOR
    if _BARCODE_DETECTOR is None:
        try:
            _BARCODE_DETECTOR = cv2.barcode.BarcodeDetector()
        except Exception as exc:
            logger.debug("BarcodeDetector init failed: %s", exc)
    return _BARCODE_DETECTOR


def _try_opencv_barcode(image: np.ndarray) -> list[str]:
    """Decode linear barcodes using OpenCV BarcodeDetector (more robust than pyzbar on blurry crops)."""
    detector = _get_barcode_detector()
    if detector is None:
        return []
    try:
        retval, decoded_info, _, _ = detector.detectAndDecodeWithType(image)
        if retval and decoded_info:
            return [d for d in decoded_info if d]
        return []
    except Exception as exc:
        logger.debug("OpenCV BarcodeDetector error: %s", exc)
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


def _named_template_rois(
    crop: np.ndarray, purposes: set[str] | None = None
) -> list[tuple[str, np.ndarray]]:
    """Return template-derived semantic ROIs before generic geometric fallbacks."""
    if crop is None or crop.size == 0:
        return []
    return [
        (f"template:{name}", roi) for name, roi in roi_crops(crop, purposes)
    ]


def _named_geometric_rois(
    crop: np.ndarray, kind: str = "qr"
) -> list[tuple[str, np.ndarray]]:
    """Return named geometric ROIs used as a safe fallback."""
    if crop is None or crop.size == 0:
        return []
    h, w = crop.shape[:2]
    min_w = max(12, int(w * 0.08))
    min_h = max(12, int(h * 0.08))
    out: list[tuple[str, np.ndarray]] = []
    for name, x1, y1, x2, y2 in _roi_boxes(w, h):
        if kind == "barcode" and not name.startswith("barcode"):
            continue
        if kind == "qr" and name.startswith("barcode"):
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < min_w or y2 - y1 < min_h:
            continue
        out.append((f"geom:{name}", crop[y1:y2, x1:x2]))
    return out


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


def _named_roi_variants(
    crop: np.ndarray, kind: str = "qr"
) -> list[tuple[str, np.ndarray]]:
    """Generate template ROI variants first, then geometric ROI variants."""
    if crop is None or crop.size == 0:
        return []
    out: list[tuple[str, np.ndarray]] = []
    rotations = (
        ("upright", None),
        ("rot90ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
        ("rot180", cv2.ROTATE_180),
        ("rot90cw", cv2.ROTATE_90_CLOCKWISE),
    )
    purposes = {"barcode"} if kind == "barcode" else {"qr"}
    for rot_name, rot in rotations:
        base = cv2.rotate(crop, rot) if rot is not None else crop
        named_rois = _named_template_rois(
            base, purposes
        ) + _named_geometric_rois(base, kind)
        for roi_name, roi in named_rois:
            for variant_idx, variant in enumerate(_enhance_code_roi(roi)):
                out.append((f"{rot_name}:{roi_name}:v{variant_idx}", variant))
    return out


def _template_roi_variants(
    crop: np.ndarray, kind: str = "qr"
) -> list[np.ndarray]:
    """Expose template-derived code ROI variants for tests/debugging."""
    if crop is None or crop.size == 0:
        return []
    purposes = {"barcode"} if kind == "barcode" else {"qr"}
    out: list[np.ndarray] = []
    for _, roi in _named_template_rois(crop, purposes):
        out.extend(_enhance_code_roi(roi))
    return out


def _roi_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Generate targeted code ROIs before expensive full-crop decoding."""
    return [img for _, img in _named_roi_variants(crop, "qr")]


def _barcode_roi_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Return bottom/right barcode-oriented ROIs for pyzbar."""
    return [img for _, img in _named_roi_variants(crop, "barcode")]


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


def decode_qr(
    crop: np.ndarray,
    *,
    debug_dir: str | Path | None = None,
    track_id: int | None = None,
    timestamp_ms: float | None = None,
) -> dict[str, str]:
    """Read QR/barcode payload from a price-tag crop.

    Template-derived ROIs are tried before generic right/bottom heuristics and
    full-crop variants.  Debug artifacts are written only when ``debug_dir`` is
    provided.
    """
    if crop is None or crop.size == 0 or _decode_mode() == "off":
        return {}

    # Fast pass: cheap decoders on 4 rotations of the full crop (~0.07 s total).
    _fast_rotations = [
        ("orig", crop),
        ("rot90ccw", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("rot180", cv2.rotate(crop, cv2.ROTATE_180)),
        ("rot90cw", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)),
    ]
    for rot_name, rot_img in _fast_rotations:
        for raw in _try_pyzbar(rot_img) + _try_opencv(rot_img):
            parsed = _raw_to_fields(raw)
            if parsed:
                _log_success(
                    debug_dir,
                    track_id=track_id,
                    timestamp_ms=timestamp_ms,
                    source="qr_fast_cheap",
                    raw=raw,
                    normalized=_success_value(parsed),
                    roi_type=rot_name,
                )
                return parsed

    # WeChatQR targeted pass: extract template QR ROIs at 2× and 4× zoom, then
    # run WeChatQR on each.  WeChatQR fails on the full crop (QR is too small),
    # but succeeds on zoomed template regions (~4.8 s for 4 rotations × 2 ROIs
    # × 2 scales × 0.3 s per call).
    for rot_name, rot_img in _fast_rotations:
        for roi_name, roi in _named_template_rois(rot_img, {"qr"})[:3]:
            if roi is None or roi.size == 0:
                continue
            h_r, w_r = roi.shape[:2]
            for scale in (2.0, 4.0):
                zoomed = cv2.resize(
                    roi,
                    (max(1, int(w_r * scale)), max(1, int(h_r * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )
                for raw in _try_wechat_qr(zoomed):
                    parsed = _raw_to_fields(raw)
                    if parsed:
                        _log_success(
                            debug_dir,
                            track_id=track_id,
                            timestamp_ms=timestamp_ms,
                            source="qr_wechat_roi",
                            raw=raw,
                            normalized=_success_value(parsed),
                            roi_type=f"{rot_name}:{roi_name}:x{scale:.0f}",
                        )
                        return parsed

    roi_limit = _variant_limit(default_full=10_000, default_fast=24)
    for source, img in _named_roi_variants(crop, "qr")[:roi_limit]:
        for raw in _try_pyzbar(img) + _try_opencv(img):
            parsed = _raw_to_fields(raw)
            if parsed:
                _log_success(
                    debug_dir,
                    track_id=track_id,
                    timestamp_ms=timestamp_ms,
                    source="qr_roi",
                    raw=raw,
                    normalized=_success_value(parsed),
                    roi_type=source,
                )
                return parsed

    if _decode_mode() == "fast":
        _save_failed_roi_examples(
            debug_dir,
            crop,
            kind="qr",
            track_id=track_id,
            timestamp_ms=timestamp_ms,
        )
        return {}

    full_limit = _variant_limit(default_full=10_000, default_fast=12)
    for idx, img in enumerate(_image_variants(crop)[:full_limit]):
        for raw in _try_pyzbar(img) + _try_opencv(img):
            parsed = _raw_to_fields(raw)
            if parsed:
                _log_success(
                    debug_dir,
                    track_id=track_id,
                    timestamp_ms=timestamp_ms,
                    source="qr_full_variant",
                    raw=raw,
                    normalized=_success_value(parsed),
                    roi_type=f"full:v{idx}",
                )
                return parsed

    # QReader is heavier; call it once per crop on the two most useful variants.
    for idx, img in enumerate(_image_variants(crop)[:2]):
        for raw in _try_qreader(img):
            parsed = _raw_to_fields(raw)
            if parsed:
                _log_success(
                    debug_dir,
                    track_id=track_id,
                    timestamp_ms=timestamp_ms,
                    source="qr_qreader",
                    raw=raw,
                    normalized=_success_value(parsed),
                    roi_type=f"qreader:v{idx}",
                )
                return parsed

    _save_failed_roi_examples(
        debug_dir, crop, kind="qr", track_id=track_id, timestamp_ms=timestamp_ms
    )
    return {}


def decode_barcode(
    crop: np.ndarray,
    *,
    debug_dir: str | Path | None = None,
    track_id: int | None = None,
    timestamp_ms: float | None = None,
) -> str:
    """Read a linear barcode from a crop and return a validated EAN-13."""
    if crop is None or crop.size == 0 or _decode_mode() == "off":
        return ""
    roi_limit = _variant_limit(default_full=10_000, default_fast=24)
    for source, img in _named_roi_variants(crop, "barcode")[:roi_limit]:
        for raw in _try_pyzbar(img) + _try_opencv_barcode(img):
            digits = re.sub(r"\D", "", raw)
            if 8 <= len(digits) <= 15:
                normalized = _normalize_barcode(digits, strict=True)
                if normalized:
                    _log_success(
                        debug_dir,
                        track_id=track_id,
                        timestamp_ms=timestamp_ms,
                        source="barcode_roi",
                        raw=raw,
                        normalized=normalized,
                        roi_type=source,
                    )
                    return normalized
    if _decode_mode() == "fast":
        _save_failed_roi_examples(
            debug_dir,
            crop,
            kind="barcode",
            track_id=track_id,
            timestamp_ms=timestamp_ms,
        )
        return ""
    full_limit = _variant_limit(default_full=10_000, default_fast=12)
    for idx, img in enumerate(_image_variants(crop)[:full_limit]):
        for raw in _try_pyzbar(img) + _try_opencv_barcode(img):
            digits = re.sub(r"\D", "", raw)
            if 8 <= len(digits) <= 15:
                normalized = _normalize_barcode(digits, strict=True)
                if normalized:
                    _log_success(
                        debug_dir,
                        track_id=track_id,
                        timestamp_ms=timestamp_ms,
                        source="barcode_full_variant",
                        raw=raw,
                        normalized=normalized,
                        roi_type=f"full:v{idx}",
                    )
                    return normalized
    _save_failed_roi_examples(
        debug_dir,
        crop,
        kind="barcode",
        track_id=track_id,
        timestamp_ms=timestamp_ms,
    )
    return ""
