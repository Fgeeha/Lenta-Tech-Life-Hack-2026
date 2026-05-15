"""Strict validators for barcode/SKU semantics."""

from shelf.postproc.merge import merge
from shelf.qr.decoder import parse_qr_url
from shelf.schema import PriceTag
from shelf.validation import normalize_ean13, normalize_sku


def test_valid_ean13_normalizes():
    assert normalize_ean13("4607124143901") == "4607124143901"


def test_invalid_ean13_is_rejected():
    assert normalize_ean13("4607124143900") == ""


def test_twelve_digit_sku_is_not_barcode():
    assert normalize_sku("270108726573") == "270108726573"
    assert normalize_ean13("270108726573") == ""


def test_qr_barcode_not_changed_by_repair_when_strict_invalid_13():
    result = parse_qr_url(
        "https://x?b=4607124143900&p1=100", strict_barcode=True
    )
    assert "qr_code_barcode" not in result
    assert result["price1_qr"] == "100"


def test_qr_parser_backward_compatible_without_strict_flag():
    result = parse_qr_url("https://x?b=1234567890123&p1=100")
    assert result["qr_code_barcode"] == "1234567890123"


def test_merge_drops_invalid_barcode_values():
    tag = merge(PriceTag(barcode="4607124143900"), {})
    assert tag.barcode == ""
