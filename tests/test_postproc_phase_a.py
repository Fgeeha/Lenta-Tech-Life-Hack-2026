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


# ── Catalog lookup tests ──────────────────────────────────────────────────────

def _make_catalog(entries: list[dict]) -> "Catalog":
    """Build a minimal in-memory catalog for testing."""
    from shelf.postproc.catalog import Catalog, CatalogEntry, _merge_catalog_entry

    catalog = Catalog()
    for e in entries:
        entry = CatalogEntry(
            product_name=e.get("product_name", "TestProduct"),
            barcode=e.get("barcode", ""),
            id_sku=e.get("id_sku", ""),
            price_card=e.get("price_card", ""),
            price_default=e.get("price_default", ""),
            source_files={e.get("src", "43_15.csv")},
        )
        if entry.barcode:
            catalog.by_barcode[entry.barcode] = _merge_catalog_entry(
                catalog.by_barcode.get(entry.barcode), entry
            )
    return catalog


def test_catalog_lookup_exact():
    cat = _make_catalog([{"barcode": "4690491122587", "product_name": "Widget A"}])
    entry = cat.lookup(barcode="4690491122587")
    assert entry is not None
    assert entry.product_name == "Widget A"


def test_catalog_lookup_prefix_fallback_drop_last():
    # OCR drops last digit: 4690491122587 → 469049112258 (12 digits)
    cat = _make_catalog([{"barcode": "4690491122587", "product_name": "Widget A"}])
    entry = cat.lookup(barcode="469049112258")  # substring of catalog barcode
    assert entry is not None
    assert entry.product_name == "Widget A"


def test_catalog_lookup_prefix_fallback_extra_digit():
    # OCR adds a spurious digit at end: 4690491122587 → 46904911225870
    cat = _make_catalog([{"barcode": "4690491122587", "product_name": "Widget A"}])
    entry = cat.lookup(barcode="46904911225870")  # catalog bc is substring of input
    assert entry is not None
    assert entry.product_name == "Widget A"


def test_catalog_lookup_prefix_ambiguous_returns_none():
    # Two barcodes share a common prefix → ambiguous → None
    cat = _make_catalog([
        {"barcode": "4690491122587", "product_name": "Widget A"},
        {"barcode": "4690491122500", "product_name": "Widget B"},
    ])
    entry = cat.lookup(barcode="469049112")  # matches both
    assert entry is None


def test_catalog_lookup_prefix_too_short():
    # Digit run < 8 → skip prefix fallback
    cat = _make_catalog([{"barcode": "4690491122587", "product_name": "Widget A"}])
    entry = cat.lookup(barcode="4690491")  # only 7 digits
    assert entry is None


def test_lookup_by_video_price_unique():
    cat = _make_catalog([
        {"barcode": "4690491122587", "price_card": "316,99", "price_default": "415,79", "src": "43_15.csv"},
    ])
    entry = cat.lookup_by_video_price("316.99", video_hint="43_15")
    assert entry is not None
    assert entry.price_default == "415,79"


def test_lookup_by_video_price_dual_tiebreaker():
    # Two products same price_card — dual tiebreaker by price_default
    cat = _make_catalog([
        {"barcode": "4690491122587", "price_card": "149,99", "price_default": "189,99", "src": "49_5.csv"},
        {"barcode": "4606272000180", "price_card": "149,99", "price_default": "299,99", "src": "49_5.csv"},
    ])
    # Ambiguous on price_card alone
    assert cat.lookup_by_video_price("149,99", video_hint="49_5") is None
    # Resolved by price_default
    entry = cat.lookup_by_video_price("149,99", price_default="189,99", video_hint="49_5")
    assert entry is not None
    assert entry.price_default == "189,99"


def test_lookup_by_video_price_empty_hint_returns_none():
    cat = _make_catalog([{"barcode": "4690491122587", "price_card": "316,99", "src": "43_15.csv"}])
    assert cat.lookup_by_video_price("316.99", video_hint="") is None
