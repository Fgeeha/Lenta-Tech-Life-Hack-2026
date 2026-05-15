"""Tests for cross-track deduplication."""

from shelf.postproc.dedup import deduplicate_tags
from shelf.schema import PriceTag


def test_deduplicate_by_barcode_merges_fields():
    a = PriceTag(
        filename="v.mp4",
        barcode="4607124143901",
        price_card="129,99",
        x_min=0,
        y_min=0,
        x_max=100,
        y_max=100,
    )
    b = PriceTag(
        filename="v.mp4",
        qr_code_barcode="4607124143901",
        product_name="Молоко",
        x_min=5,
        y_min=5,
        x_max=105,
        y_max=105,
    )
    result = deduplicate_tags([a, b])
    assert len(result) == 1
    assert result[0].product_name == "Молоко"
    assert result[0].price_card == "129,99"
