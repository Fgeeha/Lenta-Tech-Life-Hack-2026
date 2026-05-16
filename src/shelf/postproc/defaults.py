"""Field-level defaults for Lenta price tags.

Many GT fields are "нет" in >90% cases. After OCR/QR we fill remaining
empty fields with sensible defaults — never overwriting successful OCR/QR.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Fields where GT == "нет" in >90% of the 274 annotated tags.
_FIELD_DEFAULTS: dict[str, str] = {
    "price_discount": "нет",
    "price2_qr": "нет",
    "price3_qr": "нет",
    "action_price_qr": "нет",
    "action_code_qr": "нет",
    "wholesale_level_1_count": "нет",
    "wholesale_level_1_price": "нет",
    "wholesale_level_2_count": "нет",
    "wholesale_level_2_price": "нет",
    "additional_info": "нет",
}


def _is_empty(value) -> bool:
    """True for None, NaN, empty string, or string representations of missing."""
    if value is None:
        return True
    if isinstance(value, float):
        return pd.isna(value)
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def apply_field_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill empty fields with conservative defaults. Never overwrites OCR/QR data."""
    if df is None or df.empty:
        return df
    result = df.copy()
    for field, default in _FIELD_DEFAULTS.items():
        if field not in result.columns:
            continue
        mask = result[field].apply(_is_empty)
        filled = int(mask.sum())
        if filled:
            result[field] = result[field].astype(object)
            result.loc[mask, field] = default
            logger.debug("defaults: filled %d rows field=%r value=%r", filled, field, default)
    return result
