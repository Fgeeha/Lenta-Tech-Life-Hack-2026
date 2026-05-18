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


def test_lookup_by_price_and_name_resolves_collision():
    cat = _make_catalog([
        {"barcode": "4690000000016", "price_card": "159,99",
         "product_name": "Молоко Простоквашино 3.2% 1л", "src": "49_5.csv"},
        {"barcode": "4690000000023", "price_card": "159,99",
         "product_name": "Кефир Домик в деревне 2.5% 1л", "src": "49_5.csv"},
        {"barcode": "4690000000030", "price_card": "159,99",
         "product_name": "Йогурт Активиа 290г", "src": "49_5.csv"},
    ])
    # Without name — 3 candidates, ambiguous
    assert cat.lookup_by_video_price_and_name(
        price_card="159.99", video_hint="49_5") is None
    # With distinctive name — resolves to one entry
    entry = cat.lookup_by_video_price_and_name(
        price_card="159.99",
        product_name_ocr="Молоко Простоквашино",
        video_hint="49_5")
    assert entry is not None
    assert "Простоквашино" in entry.product_name


def test_lookup_by_price_and_name_rejects_weak_match():
    # Both candidates are "Молоко X.X%" — OCR name "молоко 3 2" scores similarly
    cat = _make_catalog([
        {"barcode": "4690000000016", "price_card": "159,99",
         "product_name": "Молоко Простоквашино 3.2% 1л", "src": "49_5.csv"},
        {"barcode": "4690000000023", "price_card": "159,99",
         "product_name": "Молоко Домик в деревне 3.2% 1л", "src": "49_5.csv"},
    ])
    entry = cat.lookup_by_video_price_and_name(
        price_card="159.99",
        product_name_ocr="молоко 3 2",
        video_hint="49_5")
    assert entry is None


def test_lookup_by_price_and_name_requires_video():
    cat = _make_catalog([
        {"barcode": "4690000000016", "price_card": "159,99",
         "product_name": "Молоко Простоквашино 3.2% 1л", "src": "49_5.csv"},
    ])
    assert cat.lookup_by_video_price_and_name(
        price_card="159.99",
        product_name_ocr="Простоквашино",
        video_hint="") is None


# ── Fix #1: force_prices on exact barcode match ───────────────────────────────

def _make_catalog_full(entries: list[dict]) -> "Catalog":
    """Build catalog with all new fields for testing."""
    from shelf.postproc.catalog import Catalog, CatalogEntry, _merge_catalog_entry

    catalog = Catalog()
    for e in entries:
        entry = CatalogEntry(
            product_name=e.get("product_name", "TestProduct"),
            barcode=e.get("barcode", ""),
            id_sku=e.get("id_sku", ""),
            price_card=e.get("price_card", ""),
            price_default=e.get("price_default", ""),
            special_symbols=e.get("special_symbols", ""),
            code=e.get("code", ""),
            print_datetime=e.get("print_datetime", ""),
            additional_info=e.get("additional_info", ""),
            source_files={e.get("src", "43_15.csv")},
        )
        if entry.barcode:
            catalog.by_barcode[entry.barcode] = _merge_catalog_entry(
                catalog.by_barcode.get(entry.barcode), entry
            )
    return catalog


def test_force_prices_overwrites_wrong_ocr_on_exact_barcode_match():
    from shelf.postproc.catalog import apply_catalog
    from shelf.schema import PriceTag

    cat = _make_catalog_full([{
        "barcode": "4690491122587",
        "price_card": "316,99",
        "price_default": "415,79",
    }])
    tag = PriceTag(barcode="4690491122587", price_card="999,00", price_default="111,00")
    [result] = apply_catalog([tag], cat)
    assert result.price_card == "316,99"
    assert result.price_default == "415,79"


def test_no_force_prices_on_price_based_lookup():
    from shelf.postproc.catalog import apply_catalog
    from shelf.schema import PriceTag

    cat = _make_catalog_full([{
        "barcode": "4690491122587",
        "price_card": "316,99",
        "price_default": "415,79",
        "src": "43_15.csv",
    }])
    # Price-based match does NOT force prices: a wrong-price OCR error could
    # match a different product and corrupting existing OCR values is risky.
    tag = PriceTag(filename="43_15.mp4", barcode="", price_card="316,99", price_default="111,00")
    [result] = apply_catalog([tag], cat)
    assert result.price_default == "111,00"  # not overwritten — too risky


# ── New catalog fields: special_symbols, code, print_datetime, additional_info ─

def test_catalog_fills_special_symbols_from_entry():
    from shelf.postproc.catalog import apply_catalog
    from shelf.schema import PriceTag

    cat = _make_catalog_full([{
        "barcode": "4690491122587",
        "special_symbols": "К",
        "code": "13_043015",
        "print_datetime": "04.01.2026 2:00",
        "additional_info": "нет",
    }])
    tag = PriceTag(barcode="4690491122587")
    [result] = apply_catalog([tag], cat)
    assert result.special_symbols == "К"
    assert result.code == "13_043015"
    assert result.print_datetime == "04.01.2026 2:00"


def test_catalog_does_not_overwrite_existing_code():
    from shelf.postproc.catalog import apply_catalog
    from shelf.schema import PriceTag

    cat = _make_catalog_full([{
        "barcode": "4690491122587",
        "code": "13_043015",
    }])
    tag = PriceTag(barcode="4690491122587", code="ALREADY_SET")
    [result] = apply_catalog([tag], cat)
    assert result.code == "ALREADY_SET"


def test_clean_symbol_normalises_latin_k():
    from shelf.postproc.catalog import _clean_symbol

    assert _clean_symbol("K") == "К"      # Latin K → Cyrillic К
    assert _clean_symbol("К ") == "К"     # trailing space stripped
    assert _clean_symbol("Ш") == "Ш"
    assert _clean_symbol("нет") == "нет"
    assert _clean_symbol("xyz") == ""


# ── Numeric format normalization for submission ────────────────────────────────

def test_dot_price_normalizes_comma():
    from shelf.io.writer import _dot_price
    assert _dot_price("316,99") == "316.99"
    assert _dot_price("3789,49") == "3789.49"
    assert _dot_price("316.99") == "316.99"   # already dot → unchanged


def test_dot_price_leaves_absent_values():
    from shelf.io.writer import _dot_price
    assert _dot_price("нет") == "нет"
    assert _dot_price("") == ""
    assert _dot_price(None) == ""


def test_dot_price_leaves_non_numeric_strings():
    from shelf.io.writer import _dot_price
    assert _dot_price("-23%") == "-23%"        # discount_amount format
    assert _dot_price("Молоко, 1л") == "Молоко, 1л"  # product_name comma unchanged


def test_prepare_output_normalizes_price_card():
    from shelf.io.writer import prepare_output_dataframe
    from shelf.schema import PriceTag
    tag = PriceTag(price_card="316,99", price_default="415,79",
                   price4_qr="316,99", price1_qr="415,79")
    df = prepare_output_dataframe([tag])
    assert df.loc[0, "price_card"] == "316.99"
    assert df.loc[0, "price_default"] == "415.79"
    assert df.loc[0, "price4_qr"] == "316.99"
    assert df.loc[0, "price1_qr"] == "415.79"


def test_prepare_output_leaves_discount_amount_unchanged():
    from shelf.io.writer import prepare_output_dataframe
    from shelf.schema import PriceTag
    tag = PriceTag(discount_amount="-23%")
    df = prepare_output_dataframe([tag])
    assert df.loc[0, "discount_amount"] == "-23%"


# ── Bbox float precision ──────────────────────────────────────────────────────

def test_detection_preserves_float_precision():
    from shelf.detect.detector import Detection
    det = Detection(
        x_min=2011.85, y_min=1923.3, x_max=2231.6, y_max=2115.3,
        confidence=0.9, cls_name="tag"
    )
    assert det.x_min == 2011.85
    clipped = det.clipped(width=4000, height=3000)
    assert clipped.x_min == 2011.85
    assert clipped.y_min == 1923.3


def test_detection_area_is_int():
    from shelf.detect.detector import Detection
    det = Detection(x_min=10.3, y_min=20.7, x_max=110.8, y_max=120.9, confidence=0.5)
    assert isinstance(det.area, int)
    assert det.area == 100 * 100  # int(110.8-10.3) * int(120.9-20.7) = 100*100


def test_format_bbox_coord():
    from shelf.io.writer import _format_bbox_coord
    assert _format_bbox_coord(2011.9) == "2011.9"   # exact sample.csv value
    assert _format_bbox_coord(2011.0) == "2011.0"
    assert _format_bbox_coord(2011) == "2011.0"
    assert _format_bbox_coord("1923.3") == "1923.3"
    assert _format_bbox_coord(None) == ""
    assert _format_bbox_coord("") == ""
    assert _format_bbox_coord(0) == "0.0"


def test_prepare_output_formats_bbox_as_float():
    from shelf.io.writer import prepare_output_dataframe
    from shelf.schema import PriceTag
    # Use exact values from sample.csv format specification
    tag = PriceTag(x_min=2011.9, y_min=1923.3, x_max=2231.6, y_max=2115.3)
    df = prepare_output_dataframe([tag])
    assert df.loc[0, "x_min"] == "2011.9"
    assert df.loc[0, "y_min"] == "1923.3"
    assert df.loc[0, "x_max"] == "2231.6"
    assert df.loc[0, "y_max"] == "2115.3"


def test_prepare_output_bbox_integer_input():
    from shelf.io.writer import prepare_output_dataframe
    from shelf.schema import PriceTag
    tag = PriceTag(x_min=2011.0, y_min=1923.0, x_max=2231.0, y_max=2115.0)
    df = prepare_output_dataframe([tag])
    assert df.loc[0, "x_min"] == "2011.0"


# ── Distortion corrector ──────────────────────────────────────────────────────

def test_distortion_corrector_preserves_frame_shape():
    from shelf.io.distortion import DistortionCorrector
    import numpy as np
    dc = DistortionCorrector()
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    undist = dc.get_undistorted_frame(frame, frame_id=0)
    # No ROI crop in get_undistorted_frame — same dimensions
    assert undist.shape == (2160, 3840, 3)


def test_distortion_corrector_crop_preserves_coords():
    from shelf.io.distortion import DistortionCorrector
    import numpy as np
    dc = DistortionCorrector()
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    undist = dc.get_undistorted_frame(frame, frame_id=1)
    # Crop at arbitrary bbox stays same size
    crop = undist[100:300, 200:500]
    assert crop.shape == (200, 300, 3)


def test_distortion_corrector_full_frame_crops():
    from shelf.io.distortion import DistortionCorrector
    import numpy as np
    dc = DistortionCorrector()
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    undist = dc.undistort_full_frame(frame)
    # ROI crop reduces dimensions slightly
    assert undist.shape[0] <= 2160
    assert undist.shape[1] <= 3840


def test_distortion_corrector_per_frame_cache():
    from shelf.io.distortion import DistortionCorrector
    import numpy as np
    dc = DistortionCorrector()
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    out1 = dc.get_undistorted_frame(frame, frame_id=42)
    out2 = dc.get_undistorted_frame(frame, frame_id=42)
    assert out1 is out2  # same object from cache


def test_undistort_ocr_enabled_env():
    import os
    from shelf.io.distortion import undistort_ocr_enabled
    orig = os.environ.get("SHELF_UNDISTORT_OCR")
    try:
        os.environ["SHELF_UNDISTORT_OCR"] = "1"
        assert undistort_ocr_enabled() is True
        os.environ["SHELF_UNDISTORT_OCR"] = "0"
        assert undistort_ocr_enabled() is False
        os.environ.pop("SHELF_UNDISTORT_OCR", None)
        assert undistort_ocr_enabled() is False  # default off (smoke test showed regression)
    finally:
        if orig is None:
            os.environ.pop("SHELF_UNDISTORT_OCR", None)
        else:
            os.environ["SHELF_UNDISTORT_OCR"] = orig
