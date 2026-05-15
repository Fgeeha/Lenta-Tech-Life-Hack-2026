"""Tests for SKU barcode→product_name catalog lookup."""

from shelf.postproc.catalog import catalog_size, lookup_product_name


def test_catalog_loads():
    assert catalog_size() > 0


def test_known_barcode_from_gt():
    # 4670025474665 is in 25_12-20 GT: Напиток SANTO STEFANO Rosso
    name = lookup_product_name("4670025474665")
    assert name, "Known GT barcode should return non-empty name"
    assert "SANTO" in name or "Напиток" in name


def test_unknown_barcode_returns_empty():
    assert lookup_product_name("0000000000000") == ""


def test_empty_input_returns_empty():
    assert lookup_product_name("") == ""
    assert lookup_product_name("abc") == ""


def test_non_digit_returns_empty():
    assert lookup_product_name("460712414390X") == ""
