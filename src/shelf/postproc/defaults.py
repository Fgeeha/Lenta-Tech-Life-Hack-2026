"""Field-level defaults for Lenta price tags.

Many GT fields are "нет" in >90% cases. After OCR/QR we fill remaining
empty fields with sensible defaults — never overwriting successful OCR/QR.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Structurally absent fields: always "нет" per Lenta schema spec + sample.csv.
# Rule: "нет" = field is not applicable for this product category.
#        ""    = field is applicable but was not decoded (QR/OCR miss).
# price{1..4}_qr are QR positional slots → stay empty ("") when QR is not decoded.
# additional_info is template-dependent → handled by P1 abstain policy.
# wholesale_level_1_count: 100% "нет" in sample.csv; 37% have "2" in labeled GT →
#   filling with "нет" is correct for sample convention but risky for GT eval.
_FIELD_DEFAULTS: dict[str, str] = {
    # Always "нет" regardless of template:
    "price_discount": "нет",
    "action_price_qr": "нет",
    "action_code_qr": "нет",
    "wholesale_level_1_count": "нет",
    "wholesale_level_1_price": "нет",
    "wholesale_level_2_count": "нет",
    "wholesale_level_2_price": "нет",
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
            logger.debug(
                "defaults: filled %d rows field=%r value=%r",
                filled,
                field,
                default,
            )
    return result
