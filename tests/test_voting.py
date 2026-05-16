"""Tests for multi-frame field-level voting."""

from shelf.postproc.voting import merge_candidate_tags, vote_tags
from shelf.schema import PriceTag


def test_voting_chooses_valid_barcode_among_candidates():
    tags = [
        PriceTag(barcode="4607124143900", product_name="bad checksum"),
        PriceTag(barcode="4607124143901", product_name="good"),
        PriceTag(qr_code_barcode="4607124143901"),
    ]
    result = vote_tags(tags)
    assert result.tag.barcode == "4607124143901"
    assert result.sources["barcode"] in {"barcode", "qr"}


def test_voting_does_not_turn_sku_into_barcode():
    tag = merge_candidate_tags([PriceTag(id_sku="270108726573", barcode="")])
    assert tag.id_sku == "270108726573"
    assert tag.barcode == ""


def test_voting_selects_more_complete_product_name():
    tags = [
        PriceTag(product_name="Молоко"),
        PriceTag(
            product_name="Молоко питьевое пастеризованное Простоквашино 3.2%"
        ),
    ]
    tag = merge_candidate_tags(tags)
    assert (
        tag.product_name == "Молоко питьевое пастеризованное Простоквашино 3.2%"
    )


def test_voting_price_uses_candidate_quality_weight():
    tags = [PriceTag(price_card="99,99"), PriceTag(price_card="129,99")]
    tag = merge_candidate_tags(tags, candidate_scores=[10.0, 100.0])
    assert tag.price_card == "129,99"


def test_voting_empty_values_do_not_overwrite_good_values():
    tags = [
        PriceTag(price_card="129,99", product_name="Мёд липовый"),
        PriceTag(price_card="", product_name=""),
    ]
    tag = merge_candidate_tags(tags, candidate_scores=[1.0, 100.0])
    assert tag.price_card == "129,99"
    assert tag.product_name == "Мёд липовый"


def test_voting_uses_qr_prices_to_fill_ocr_prices_and_discount():
    tag = merge_candidate_tags(
        [PriceTag(price1_qr="189.99", price4_qr="129.99")]
    )
    assert tag.price_default == "189,99"
    assert tag.price_card == "129,99"
    assert tag.discount_amount.startswith("-")


def test_voting_swaps_inverted_card_and_default_prices():
    tag = merge_candidate_tags(
        [PriceTag(price_card="189,99", price_default="129,99")]
    )
    assert tag.price_card == "129,99"
    assert tag.price_default == "189,99"
