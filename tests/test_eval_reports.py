"""Tests for per-tag metric diagnostics."""

import pandas as pd

from scripts.eval_on_labeled import match_and_score, write_diagnostic_reports


def test_match_and_score_writes_pass80_diagnostics(tmp_path):
    gt = pd.DataFrame(
        [
            {
                "barcode": "4607124143901",
                "qr_code_barcode": "4607124143901",
                "product_name": "Молоко",
                "price_default": "100,00",
                "price_card": "80,00",
                "price_discount": "нет",
                "discount_amount": "-20%",
                "id_sku": "270207736530",
                "price1_qr": "100.00",
                "price2_qr": "нет",
                "price4_qr": "80.00",
                "frame_timestamp": 1000,
                "x_min": 0,
                "y_min": 0,
                "x_max": 100,
                "y_max": 100,
            }
        ]
    )
    pred = gt.copy()
    summary = {
        "videos": [
            match_and_score(
                pred,
                gt,
                [
                    "product_name",
                    "price_default",
                    "price_card",
                    "price_discount",
                    "barcode",
                    "discount_amount",
                    "id_sku",
                    "qr_code_barcode",
                    "price1_qr",
                    "price2_qr",
                    "price4_qr",
                ],
                video="unit",
            )
        ],
        "overall": {},
    }
    summary["overall"] = {
        "field_accuracy": summary["videos"][0]["field_accuracy"],
        "fill_rates": summary["videos"][0]["fill_rates"],
    }
    write_diagnostic_reports(summary, tmp_path)
    assert (tmp_path / "matched_tags_debug.csv").exists()
    assert (tmp_path / "field_accuracy.csv").exists()
