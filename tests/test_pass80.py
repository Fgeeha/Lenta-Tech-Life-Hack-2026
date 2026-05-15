"""Tests for pass80 optimizer business rules."""

from shelf.postproc.pass80 import optimize_tag, optimize_tags
from shelf.schema import ABSENT_VALUE, PriceTag


def test_pass80_syncs_valid_qr_barcode_to_barcode():
    tag = PriceTag(qr_code_barcode="4607124143901")
    out, changes = optimize_tag(tag)
    assert out.barcode == "4607124143901"
    assert any(c.field == "barcode" for c in changes)


def test_pass80_does_not_turn_sku_into_barcode():
    tag = PriceTag(id_sku="270207736530")
    out, _ = optimize_tag(tag)
    assert out.id_sku == "270207736530"
    assert out.barcode == ""


def test_pass80_derives_prices_and_discount():
    tag = PriceTag(price1_qr="100.00", price4_qr="80.00")
    out, changes = optimize_tag(tag)
    assert out.price_default == "100,00"
    assert out.price_card == "80,00"
    assert out.discount_amount == "-19%"
    assert {c.field for c in changes} >= {
        "price_default",
        "price_card",
        "discount_amount",
    }


def test_pass80_swaps_inverted_prices():
    tag = PriceTag(price_card="120,00", price_default="99,00")
    out, _ = optimize_tag(tag)
    assert out.price_card == "99,00"
    assert out.price_default == "120,00"


def test_pass80_report_counts_proxy_crossing(tmp_path):
    tag = PriceTag(
        product_name="Молоко",
        price1_qr="100.00",
        price4_qr="80.00",
        price_discount=ABSENT_VALUE,
        qr_code_barcode="4607124143901",
        id_sku="270207736530",
        color="white",
    )
    out, report = optimize_tags([tag], debug_dir=tmp_path)
    assert out[0].barcode == "4607124143901"
    assert report.changes
    assert (tmp_path / "pass80_optimizer_report.csv").exists()
