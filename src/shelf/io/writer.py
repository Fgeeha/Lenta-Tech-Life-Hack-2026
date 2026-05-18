"""CSV writer with strict output schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from shelf.schema import COLUMN_ALIASES, OUTPUT_COLUMNS, PriceTag

# Bbox fields output as float with 1 decimal place to match sample.csv.
_BBOX_COLS = frozenset({"x_min", "y_min", "x_max", "y_max"})

# Price-like fields where comma decimal separator must become dot in the submission.
# discount_amount ("-23%") and product_name ("Молоко, 1л") are intentionally excluded.
_PRICE_COLS = frozenset({
    "price_default",
    "price_card",
    "price_discount",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
    "action_price_qr",
})


def _format_bbox_coord(value: object) -> str:
    """Format bbox coordinate as float with 1 decimal place to match sample.csv."""
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.1f}"
    except (ValueError, TypeError):
        return str(value)


def _dot_price(value: object) -> str:
    """Normalize a price string to dot-decimal format.

    '316,99' → '316.99'.  Non-numeric strings ('' / 'нет') pass through.
    """
    s = str(value or "").strip()
    if not s or s == "нет":
        return s
    candidate = s.replace(",", ".")
    try:
        float(candidate)
        return candidate
    except ValueError:
        return s  # not a number — leave unchanged


def prepare_output_dataframe(
    tags: Iterable[PriceTag] | pd.DataFrame,
) -> pd.DataFrame:
    """Return DataFrame with exactly the required columns and no NaN values.

    Missing columns are created as empty strings. Historical aliases from old GT files
    are normalized, but the output column names are always the official ones.
    Price-like columns are normalised to dot-decimal format matching sample.csv.
    """
    if isinstance(tags, pd.DataFrame):
        df = tags.copy()
    else:
        df = pd.DataFrame([t.to_dict() for t in tags])

    df = df.rename(columns=COLUMN_ALIASES)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUTPUT_COLUMNS].copy()
    df = df.fillna("")

    # frame_timestamp stays numeric (formatted later); bbox cols formatted via _format_bbox_coord.
    for col in OUTPUT_COLUMNS:
        if col not in {"frame_timestamp"}:
            df[col] = (
                df[col].astype(str).str.strip().replace({"nan": "", "None": ""})
            )

    # Normalise price fields: comma decimal → dot (matches sample.csv format).
    for col in _PRICE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_dot_price)

    # Bbox coords: format as float with 1 decimal place to match sample.csv.
    for col in _BBOX_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_format_bbox_coord)

    # frame_timestamp: write as integer ms (no trailing .0) to match sample.csv.
    if "frame_timestamp" in df.columns:
        df["frame_timestamp"] = (
            pd.to_numeric(df["frame_timestamp"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
        )

    return df


def write_csv(
    tags: Iterable[PriceTag] | pd.DataFrame, output_path: str | Path
) -> Path:
    """Write a UTF-8-SIG CSV accepted by Excel and by the task checker."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = prepare_output_dataframe(tags)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
