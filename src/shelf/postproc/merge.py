"""Merge OCR + QR fields and derive stable business fields."""

from __future__ import annotations

import re

from shelf.qr.barcode_roi import ean13_repair
from shelf.schema import ABSENT_VALUE, PriceTag

_QR_PRIORITY_FIELDS = {
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
}
_EMPTY = ("", ABSENT_VALUE, None)


def merge(ocr_tag: PriceTag, qr_fields: dict[str, str]) -> PriceTag:
    """Overlay QR fields over OCR result and derive missing values."""
    data = ocr_tag.__dict__.copy()
    for field, value in qr_fields.items():
        value = str(value).strip() if value is not None else ""
        if not value:
            continue
        if field in _QR_PRIORITY_FIELDS:
            data[field] = value
        elif data.get(field) in _EMPTY:
            data[field] = value

    qr_bc = data.get("qr_code_barcode", ABSENT_VALUE)
    if qr_bc not in _EMPTY:
        data["qr_code_barcode"] = _normalize_barcode(qr_bc)
        if data.get("barcode") in _EMPTY:
            data["barcode"] = data["qr_code_barcode"]

    barcode = data.get("barcode", "")
    if barcode not in _EMPTY:
        data["barcode"] = _normalize_barcode(barcode)
        if data.get("qr_code_barcode") in _EMPTY:
            data["qr_code_barcode"] = data["barcode"]

    # Lenta business mapping observed in the provided examples:
    # price1_qr ~ regular/default price, price4_qr ~ card/action price.
    price_card = data.get("price_card", "")
    price_default = data.get("price_default", "")

    if price_default in _EMPTY and data.get("price1_qr", "") not in _EMPTY:
        data["price_default"] = _fmt_price_for_ocr(data["price1_qr"])
        price_default = data["price_default"]

    if price_card in _EMPTY:
        for field in ("price4_qr", "action_price_qr", "price2_qr"):
            if data.get(field, "") not in _EMPTY:
                data["price_card"] = _fmt_price_for_ocr(data[field])
                price_card = data["price_card"]
                break

    if data.get("price4_qr", "") in _EMPTY and price_card not in _EMPTY:
        data["price4_qr"] = _fmt_price_for_qr(price_card)
    if data.get("price1_qr", "") in _EMPTY and price_default not in _EMPTY:
        data["price1_qr"] = _fmt_price_for_qr(price_default)

    # Derive discount if both prices are present.
    if data.get("discount_amount", "") in _EMPTY:
        pc = _safe_float(data.get("price_card", ""))
        pd = _safe_float(data.get("price_default", ""))
        if pc and pd and pd > pc > 0:
            pct = int((1 - pc / pd) * 100)
            if 1 <= pct <= 99:
                data["discount_amount"] = f"-{pct}%"

    return PriceTag(**data)


def _safe_float(val: str) -> float | None:
    try:
        text = (
            str(val)
            .strip()
            .replace("\u00a0", " ")
            .replace(" ", "")
            .replace(",", ".")
        )
        if not text or text == ABSENT_VALUE:
            return None
        return float(text)
    except (ValueError, TypeError):
        return None


def _fmt_price_for_ocr(val: str) -> str:
    f = _safe_float(val)
    return f"{f:.2f}".replace(".", ",") if f is not None else str(val).strip()


def _fmt_price_for_qr(val: str) -> str:
    f = _safe_float(val)
    if f is None:
        return str(val).strip()
    if f.is_integer():
        return str(int(f))
    return f"{f:.2f}"


def _normalize_barcode(raw: str) -> str:
    try:
        text = str(raw).strip()
        if text.lower().endswith(".0"):
            text = text[:-2]
        digits = re.sub(r"\D", "", text)
        repaired = ean13_repair(digits) if 12 <= len(digits) <= 14 else None
        return repaired or digits or text
    except (ValueError, OverflowError):
        return str(raw)
