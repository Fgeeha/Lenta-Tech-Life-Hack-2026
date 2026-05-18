"""ROI-targeted barcode detection and EAN-13 repair.

Strategy:
1. Pass 1 — pyzbar directly on the preprocessed (already upscaled) crop.
2. Pass 2 — find barcode strip via Sobel-X + morphology, crop it, retry
             pyzbar at extra scales.
3. Pass 3 — EAN-13 repair on long digit sequences from OCR results.
"""

import re

import cv2
import numpy as np

_DIGITS_RE = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# EAN-13 utilities
# ---------------------------------------------------------------------------


def ean13_checksum_valid(s: str) -> bool:
    """Return True if s is a valid 13-digit EAN-13 string."""
    if len(s) != 13 or not s.isdigit():
        return False
    digits = [int(c) for c in s]
    total = sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == digits[12]


def ean13_repair(raw: str) -> str | None:
    """Try to recover a valid EAN-13 from a noisy digit sequence.

    Handles:
    - 13 digits, already valid
    - 12 digits → append correct check digit
    - 14 digits → try dropping first or last
    - 13 digits with one wrong digit → brute-force 1-digit fixup
    """
    s = re.sub(r"\D", "", raw)
    if not s:
        return None

    if len(s) == 13 and ean13_checksum_valid(s):
        return s

    if len(s) == 12:
        digits = [int(c) for c in s]
        total = sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits))
        check = (10 - total % 10) % 10
        candidate = s + str(check)
        if ean13_checksum_valid(candidate):
            return candidate

    if len(s) == 14:
        for cand in (s[1:], s[:-1]):
            if ean13_checksum_valid(cand):
                return cand

    if len(s) == 13:
        for i in range(13):
            for d in "0123456789":
                if d == s[i]:
                    continue
                cand = s[:i] + d + s[i + 1 :]
                if ean13_checksum_valid(cand):
                    return cand

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _pyzbar_decode(img: np.ndarray) -> list[str]:
    try:
        from pyzbar import pyzbar  # type: ignore

        decoded = pyzbar.decode(img)
        return [
            d.data.decode("utf-8", errors="ignore")
            for d in decoded
            if d.type in ("EAN13", "EAN8", "CODE128", "QRCODE")
        ]
    except Exception:
        return []


def _preprocess_variants(img: np.ndarray) -> list[np.ndarray]:
    """Return binarization variants that help pyzbar read linear barcodes."""
    gray = _to_gray(img)
    variants: list[np.ndarray] = [img]
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else otsu
    )
    _, inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    variants.append(
        cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else inv
    )
    return variants


# ---------------------------------------------------------------------------
# ROI finder
# ---------------------------------------------------------------------------


def find_barcode_strip(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Locate the barcode region in a preprocessed price-tag crop.

    The crop is expected to be already rotated 90°CCW and upscaled (~3–5×).
    Returns (x, y, w, h) of the barcode area or None.
    """
    gray = _to_gray(img)
    H, W = gray.shape

    # High vertical-edge density is the barcode signature
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    abs_x = np.abs(sobel_x).astype(np.uint8)

    # Merge barcode stripes horizontally
    kw = max(1, min(W // 8, 40))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 5))
    closed = cv2.morphologyEx(abs_x, cv2.MORPH_CLOSE, kernel)
    _, thresh = cv2.threshold(closed, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates: list[tuple[int, int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w / max(h, 1) < 2.0 or w < W * 0.18 or h < 8:
            continue
        candidates.append((x, y, w, h, w * h))

    if not candidates:
        return None

    candidates.sort(key=lambda t: -t[4])
    return candidates[0][:4]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def read_barcode_from_strip(proc_crop: np.ndarray) -> str:
    """Try to read EAN-13 from a preprocessed (upscaled, rotated) price-tag crop.

    Returns a 13-digit string or '' if nothing valid found.
    Does NOT require OCR engine — relies solely on pyzbar + EAN repair.
    Caller should wire OCR-based digit extraction separately if needed.
    """
    if proc_crop is None or proc_crop.size == 0:
        return ""

    # Pass 1: pyzbar on the full upscaled crop
    for variant in _preprocess_variants(proc_crop):
        for raw in _pyzbar_decode(variant):
            if raw.isdigit() and 8 <= len(raw) <= 14:
                repaired = ean13_repair(raw)
                if repaired:
                    return repaired

    # Pass 2: ROI crop → extra scale passes
    roi = find_barcode_strip(proc_crop)
    if roi is not None:
        x, y, w, h = roi
        H, W_img = proc_crop.shape[:2]
        pad_below = int(h * 0.6)
        y1 = max(0, y - 5)
        y2 = min(H, y + h + pad_below)
        x1 = max(0, x - 10)
        x2 = min(W_img, x + w + 10)
        roi_crop = proc_crop[y1:y2, x1:x2]

        if roi_crop.size > 0:
            for scale in (2, 3, 4):
                big = cv2.resize(
                    roi_crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                for variant in _preprocess_variants(big):
                    for raw in _pyzbar_decode(variant):
                        if raw.isdigit() and 8 <= len(raw) <= 14:
                            repaired = ean13_repair(raw)
                            if repaired:
                                return repaired

    return ""
