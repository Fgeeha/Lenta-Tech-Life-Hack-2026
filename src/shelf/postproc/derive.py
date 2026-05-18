"""Cross-field derivation for Lenta price tags.

Applies business-logic derivations that are deterministic from other fields:
  price4_qr ↔ price_card, price1_qr ↔ price_default, barcode ↔ qr_code_barcode,
  discount_amount from price_card / price_default.

Run AFTER apply_field_defaults so defaults don't compete with real OCR values.
"""
from __future__ import annotations

import logging
import math
import re

import pandas as pd

from shelf.qr.decoder import _normalize_barcode

logger = logging.getLogger(__name__)


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return pd.isna(value)
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "нет"}


def _parse_price(value) -> float | None:
    """Parse a price string to float. Returns None if invalid."""
    if _is_empty(value):
        return None
    try:
        s = str(value).replace(",", ".").replace(" ", "").strip()
        s = re.sub(r"[^0-9.]", "", s)
        if not s or s.count(".") > 1:
            return None
        v = float(s)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def apply_field_derivation(df: pd.DataFrame) -> pd.DataFrame:
    """Cascade of cross-field derivations. Run after apply_field_defaults."""
    if df is None or df.empty:
        return df
    result = df.copy()

    # 1. price4_qr ← price_card (Lenta QR p4 == card price)
    if "price4_qr" in result.columns and "price_card" in result.columns:
        mask = result["price4_qr"].apply(_is_empty) & ~result["price_card"].apply(_is_empty)
        if mask.any():
            result.loc[mask, "price4_qr"] = result.loc[mask, "price_card"]

    # 2. price1_qr ← price_default (Lenta QR p1 == regular retail price)
    if "price1_qr" in result.columns and "price_default" in result.columns:
        mask = result["price1_qr"].apply(_is_empty) & ~result["price_default"].apply(_is_empty)
        if mask.any():
            result.loc[mask, "price1_qr"] = result.loc[mask, "price_default"]

    # 3. price_default ← price1_qr (reverse: QR fills OCR price)
    if "price_default" in result.columns and "price1_qr" in result.columns:
        mask = result["price_default"].apply(_is_empty) & ~result["price1_qr"].apply(_is_empty)
        if mask.any():
            result.loc[mask, "price_default"] = result.loc[mask, "price1_qr"]

    # 4. price_card ← price4_qr (reverse: QR fills OCR card price)
    if "price_card" in result.columns and "price4_qr" in result.columns:
        mask = result["price_card"].apply(_is_empty) & ~result["price4_qr"].apply(_is_empty)
        if mask.any():
            result.loc[mask, "price_card"] = result.loc[mask, "price4_qr"]

    # 5. barcode ↔ qr_code_barcode (bidirectional, validated EAN-13 only)
    if "qr_code_barcode" in result.columns and "barcode" in result.columns:
        # barcode → qr_code_barcode
        mask = result["qr_code_barcode"].apply(_is_empty) & ~result["barcode"].apply(_is_empty)
        for idx in result[mask].index:
            normalized = _normalize_barcode(str(result.at[idx, "barcode"]).strip(), strict=True)
            if normalized:
                result.at[idx, "qr_code_barcode"] = normalized
        # qr_code_barcode → barcode
        mask = result["barcode"].apply(_is_empty) & ~result["qr_code_barcode"].apply(_is_empty)
        for idx in result[mask].index:
            normalized = _normalize_barcode(str(result.at[idx, "qr_code_barcode"]).strip(), strict=True)
            if normalized:
                result.at[idx, "barcode"] = normalized

    # 6. discount_amount from price_card / price_default
    if "discount_amount" in result.columns:
        mask = result["discount_amount"].apply(_is_empty)
        for idx in result[mask].index:
            pc = _parse_price(result.at[idx, "price_card"]) if "price_card" in result.columns else None
            pd_val = _parse_price(result.at[idx, "price_default"]) if "price_default" in result.columns else None
            if pc is not None and pd_val is not None and pd_val > 0 and pc < pd_val:
                pct = math.floor((1.0 - pc / pd_val) * 100)
                result.at[idx, "discount_amount"] = f"-{pct}%" if 1 <= pct <= 99 else "нет"
            else:
                result.at[idx, "discount_amount"] = "нет"

    return result
