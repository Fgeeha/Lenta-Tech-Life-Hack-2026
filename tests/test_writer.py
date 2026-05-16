"""Regression tests for strict CSV output."""

import pandas as pd

from shelf.io.writer import prepare_output_dataframe, write_csv
from shelf.schema import OUTPUT_COLUMNS, PriceTag


def test_prepare_output_dataframe_schema_and_nan_cleanup():
    df = prepare_output_dataframe(
        pd.DataFrame({"product_name": [None], "barcode": [float("nan")]})
    )
    assert list(df.columns) == OUTPUT_COLUMNS
    assert df.loc[0, "product_name"] == ""
    assert df.loc[0, "barcode"] == ""


def test_write_csv_uses_utf8_sig_and_schema(tmp_path):
    out = tmp_path / "result.csv"
    write_csv([PriceTag(product_name="Молоко")], out)
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    df = pd.read_csv(out, encoding="utf-8-sig")
    assert list(df.columns) == OUTPUT_COLUMNS
