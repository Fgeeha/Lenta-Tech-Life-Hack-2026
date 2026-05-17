"""Tests for Phase A: field defaults and cross-field derivation."""

import math
import numpy as np
import pandas as pd
import pytest

from shelf.postproc.defaults import apply_field_defaults
from shelf.postproc.derive import apply_field_derivation


def test_defaults_fill_empty():
    df = pd.DataFrame([{
        "price_discount": "",
        "wholesale_level_1_count": None,
        "additional_info": np.nan,
        "price_card": "129.99",
    }])
    out = apply_field_defaults(df)
    assert out.loc[0, "price_discount"] == "нет"
    assert out.loc[0, "wholesale_level_1_count"] == "нет"
    assert pd.isna(out.loc[0, "additional_info"]) or out.loc[0, "additional_info"] == ""  # P1 field, not filled by P0
    assert out.loc[0, "price_card"] == "129.99"  # must not be overwritten


def test_defaults_do_not_overwrite_real_value():
    df = pd.DataFrame([{"price_discount": "12%"}])
    out = apply_field_defaults(df)
    assert out.loc[0, "price_discount"] == "12%"


def test_defaults_treats_string_nan_as_empty():
    df = pd.DataFrame([{"price_discount": "nan"}])
    out = apply_field_defaults(df)
    assert out.loc[0, "price_discount"] == "нет"


def test_derive_price4_qr_from_price_card():
    df = pd.DataFrame([{"price_card": "129.99", "price4_qr": ""}])
    out = apply_field_derivation(df)
    assert out.loc[0, "price4_qr"] == "129.99"


def test_derive_price1_qr_from_price_default():
    df = pd.DataFrame([{"price_default": "199.99", "price1_qr": ""}])
    out = apply_field_derivation(df)
    assert out.loc[0, "price1_qr"] == "199.99"


def test_derive_barcode_from_qr_code_barcode():
    # 4690491122587 is a real Lenta barcode with valid EAN-13 check digit
    df = pd.DataFrame([{"barcode": "", "qr_code_barcode": "4690491122587"}])
    out = apply_field_derivation(df)
    assert out.loc[0, "barcode"] == "4690491122587"


def test_derive_discount_amount_from_prices():
    df = pd.DataFrame([{
        "price_card": "70", "price_default": "100", "discount_amount": ""
    }])
    out = apply_field_derivation(df)
    assert out.loc[0, "discount_amount"] == "-30%"  # floor((1-70/100)*100)=30, GT format is negative


def test_derive_discount_amount_no_discount_when_equal():
    df = pd.DataFrame([{
        "price_card": "100", "price_default": "100", "discount_amount": ""
    }])
    out = apply_field_derivation(df)
    assert out.loc[0, "discount_amount"] == "нет"


def test_derive_discount_amount_handles_missing_prices():
    df = pd.DataFrame([{
        "price_card": "", "price_default": "", "discount_amount": ""
    }])
    out = apply_field_derivation(df)
    assert out.loc[0, "discount_amount"] == "нет"


def test_derive_does_not_overwrite_existing():
    df = pd.DataFrame([{"price_card": "100", "price4_qr": "50.00"}])
    out = apply_field_derivation(df)
    assert out.loc[0, "price4_qr"] == "50.00"  # must not be overwritten


def test_empty_dataframe_safe():
    df = pd.DataFrame(columns=["price_discount", "price_card"])
    out = apply_field_defaults(df)
    assert len(out) == 0
    out2 = apply_field_derivation(df)
    assert len(out2) == 0
