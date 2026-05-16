"""Validation helpers for barcodes, SKU values and normalized numeric fields.

The challenge scoring is very sensitive to false positives: a wrong barcode or
SKU is worse than an empty value.  This module centralizes the strict validators
used by OCR, QR parsing, multi-frame voting and catalog lookup.
"""

from __future__ import annotations

import re
from typing import Any

SKU_RE = re.compile(r"^2\d{11}$")


def normalize_digits(raw: Any) -> str:
    """Return only digits from ``raw`` while handling pandas float strings."""
    if raw is None:
        return ""
    try:
        if raw != raw:  # NaN
            return ""
    except Exception:
        pass
    text = str(raw).strip().replace("\u00a0", " ")
    if not text or text.lower() in {"nan", "none", "нет"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return re.sub(r"\D", "", text)


def ean13_checksum_valid(code: str) -> bool:
    """Return True only for a valid 13-digit EAN-13 code."""
    if len(code) != 13 or not code.isdigit():
        return False
    digits = [int(c) for c in code]
    total = sum(d * (3 if idx % 2 else 1) for idx, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == digits[12]


def ean13_check_digit(prefix12: str) -> str:
    """Compute the EAN-13 check digit for a 12-digit prefix."""
    if len(prefix12) != 12 or not prefix12.isdigit():
        raise ValueError("EAN-13 prefix must contain exactly 12 digits")
    digits = [int(c) for c in prefix12]
    total = sum(d * (3 if idx % 2 else 1) for idx, d in enumerate(digits))
    return str((10 - total % 10) % 10)


def repair_ean13(
    raw: Any,
    *,
    allow_append_12: bool = True,
    allow_drop_14: bool = True,
    allow_one_digit_repair: bool = False,
) -> str:
    """Conservatively repair a trusted barcode candidate into valid EAN-13.

    OCR text is not a trusted source for 12-digit values because Lenta SKU values
    have exactly the same length.  Callers should set ``allow_append_12`` only for
    QR/pyzbar/barcode-ROI sources, never for arbitrary OCR text.
    """
    digits = normalize_digits(raw)
    if ean13_checksum_valid(digits):
        return digits
    if len(digits) == 12 and allow_append_12:
        candidate = digits + ean13_check_digit(digits)
        return candidate if ean13_checksum_valid(candidate) else ""
    if len(digits) == 14 and allow_drop_14:
        for candidate in (digits[:-1], digits[1:]):
            if ean13_checksum_valid(candidate):
                return candidate
    if len(digits) == 13 and allow_one_digit_repair:
        for idx, current in enumerate(digits):
            for repl in "0123456789":
                if repl == current:
                    continue
                candidate = digits[:idx] + repl + digits[idx + 1 :]
                if ean13_checksum_valid(candidate):
                    return candidate
    return ""


def normalize_ean13(
    raw: Any,
    *,
    allow_repair: bool = False,
    allow_append_12: bool = False,
    allow_drop_14: bool = True,
    allow_one_digit_repair: bool = False,
) -> str:
    """Normalize and validate an EAN-13 candidate.

    By default this function is strict: invalid 13-digit values and 12-digit SKU
    lookalikes are rejected.  Repair is opt-in and should be enabled only for
    trusted scanner/ROI sources.
    """
    digits = normalize_digits(raw)
    if ean13_checksum_valid(digits):
        return digits
    if not allow_repair:
        return ""
    return repair_ean13(
        digits,
        allow_append_12=allow_append_12,
        allow_drop_14=allow_drop_14,
        allow_one_digit_repair=allow_one_digit_repair,
    )


def is_valid_sku(raw: Any) -> bool:
    """Return True for Lenta SKU values: exactly 12 digits starting with 2."""
    return bool(SKU_RE.fullmatch(normalize_digits(raw)))


def normalize_sku(raw: Any) -> str:
    """Return a strict SKU string or an empty value."""
    digits = normalize_digits(raw)
    return digits if SKU_RE.fullmatch(digits) else ""
