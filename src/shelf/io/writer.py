"""CSV writer with strict output schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from shelf.schema import COLUMN_ALIASES, OUTPUT_COLUMNS, PriceTag


def prepare_output_dataframe(
    tags: Iterable[PriceTag] | pd.DataFrame,
) -> pd.DataFrame:
    """Return DataFrame with exactly the required columns and no NaN values.

    Missing columns are created as empty strings. Historical aliases from old GT files
    are normalized, but the output column names are always the official ones.
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

    # Keep bbox/timestamp numeric if possible; all other fields should remain strings.
    for col in OUTPUT_COLUMNS:
        if col not in {"frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
            df[col] = (
                df[col].astype(str).str.strip().replace({"nan": "", "None": ""})
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
