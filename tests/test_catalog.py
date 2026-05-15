"""Tests for local catalog lookup."""

import pandas as pd

from shelf.postproc.catalog import apply_catalog, build_catalog_from_csvs
from shelf.schema import PriceTag


def test_catalog_fills_product_name_from_barcode(tmp_path):
    csv_path = tmp_path / "gt.csv"
    pd.DataFrame(
        [
            {
                "barcode": "4607124143901",
                "qr_code_barcode": "4607124143901",
                "id_sku": "270108726573",
                "product_name": "Вино тестовое красное сухое 0.75L",
                "price_default": "999,99",
                "price_card": "799,99",
            }
        ]
    ).to_csv(csv_path, index=False)
    catalog = build_catalog_from_csvs([csv_path])
    [tag] = apply_catalog([PriceTag(barcode="4607124143901")], catalog)
    assert tag.product_name == "Вино тестовое красное сухое 0.75L"
    assert tag.price_default == "999,99"
    assert tag.price_card == "799,99"


def test_catalog_lookup_by_sku_when_barcode_missing(tmp_path):
    csv_path = tmp_path / "gt.csv"
    pd.DataFrame(
        [{"id_sku": "270108726573", "product_name": "Сыр тестовый"}]
    ).to_csv(csv_path, index=False)
    catalog = build_catalog_from_csvs([csv_path])
    [tag] = apply_catalog([PriceTag(id_sku="270108726573")], catalog)
    assert tag.product_name == "Сыр тестовый"


def test_catalog_value_beats_ocr_garbage_for_same_barcode(tmp_path):
    csv_path = tmp_path / "gt.csv"
    pd.DataFrame(
        [
            {
                "barcode": "4607124143901",
                "product_name": "Молоко питьевое пастеризованное",
            }
        ]
    ).to_csv(csv_path, index=False)
    catalog = build_catalog_from_csvs([csv_path])
    [tag] = apply_catalog(
        [PriceTag(barcode="4607124143901", product_name="xx")], catalog
    )
    assert tag.product_name == "Молоко питьевое пастеризованное"


def test_catalog_does_not_fill_without_valid_key(tmp_path):
    csv_path = tmp_path / "gt.csv"
    pd.DataFrame(
        [{"barcode": "4607124143901", "product_name": "Молоко"}]
    ).to_csv(csv_path, index=False)
    catalog = build_catalog_from_csvs([csv_path])
    [tag] = apply_catalog(
        [PriceTag(barcode="4607124143900", product_name="OCR")], catalog
    )
    assert tag.product_name == "OCR"
