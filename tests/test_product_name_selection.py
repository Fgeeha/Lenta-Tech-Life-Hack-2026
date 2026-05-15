"""Tests for robust product-name candidate cleanup/selection."""

from shelf.postproc.voting import _clean_name, select_product_name_candidate


def test_product_name_preserves_useful_percent():
    assert (
        _clean_name("Молоко пастеризованное 3.2% 930мл")
        == "Молоко пастеризованное 3.2% 930мл"
    )


def test_product_name_removes_dates_sku_barcode_and_price():
    cleaned = _clean_name(
        "03.04.2026 3:08 270207736530 Сыр плавленый 129,99 руб 4607124143901"
    )
    assert "Сыр плавленый" in cleaned
    assert "270207736530" not in cleaned
    assert "4607124143901" not in cleaned
    assert "129" not in cleaned


def test_select_product_name_prefers_complete_cyrillic_candidate():
    selected = select_product_name_candidate(
        [
            "129 99",
            "Сыр",
            "Сыр плавленый сливочный 45%",
        ]
    )
    assert selected == "Сыр плавленый сливочный 45%"
