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


def test_pass80_reverse_fills_price1_and_price4_from_ocr():
    tag = PriceTag(price_default="252,63", price_card="129,99")
    out, changes = optimize_tag(tag)
    assert out.price1_qr == "252.63"
    assert out.price4_qr == "129.99"
    assert any(c.field == "price1_qr" for c in changes)
    assert any(c.field == "price4_qr" for c in changes)


def test_pass80_derives_price2_qr_5pct():
    tag = PriceTag(price_default="252,63")
    out, changes = optimize_tag(tag)
    # price2_qr should be 252.63 * 0.95 = 239.9985 → within 1.5 of GT 239.99
    derived = float(out.price2_qr)
    assert abs(derived - 252.63 * 0.95) < 0.01
    assert any(c.field == "price2_qr" for c in changes)


def test_pass80_sync_barcode_to_qr_default_on():
    tag = PriceTag(barcode="4607124143901")
    out, changes = optimize_tag(tag)
    assert out.qr_code_barcode == "4607124143901"
    assert any(c.field == "qr_code_barcode" for c in changes)


def test_pass80_no_price2_qr_when_already_filled():
    tag = PriceTag(price_default="252,63", price2_qr="239.99")
    out, changes = optimize_tag(tag)
    assert out.price2_qr == "239.99"
    assert not any(c.field == "price2_qr" for c in changes)


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
