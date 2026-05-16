"""Tests for rule-based price-tag layout priors."""

import numpy as np

from shelf.ocr.layout import (
    Orientation,
    TemplateType,
    get_layout_prior,
    roi_crops,
)


def test_layout_returns_key_rois_inside_horizontal_crop():
    crop = np.ones((120, 260, 3), dtype=np.uint8) * 255
    prior = get_layout_prior(crop)
    assert prior.orientation == Orientation.HORIZONTAL
    for key in [
        "name_roi",
        "qr_roi",
        "barcode_roi",
        "price_card_roi",
        "price_default_roi",
        "sku_roi",
        "datetime_roi",
        "code_roi",
        "symbol_roi",
    ]:
        assert key in prior.rois
        x0, y0, x1, y1 = prior.rois[key].as_pixels(260, 120)
        assert 0 <= x0 < x1 <= 260
        assert 0 <= y0 < y1 <= 120


def test_unknown_template_is_safe_for_empty_crop():
    prior = get_layout_prior(None)
    assert prior.template_type == TemplateType.UNKNOWN
    assert prior.rois["qr_roi"].purpose == "qr"


def test_text_hints_classify_promo_wholesale_bogof():
    crop = np.ones((120, 260, 3), dtype=np.uint8) * 255
    assert (
        get_layout_prior(crop, ["акция скидка -30%"]).template_type
        == TemplateType.DISCOUNT
    )
    assert (
        get_layout_prior(crop, ["цена от 3 шт опт"]).template_type
        == TemplateType.WHOLESALE
    )
    assert (
        get_layout_prior(crop, ["BOGOF набор"]).template_type
        == TemplateType.BOGOF
    )


def test_roi_crops_are_named_and_non_empty():
    crop = np.ones((120, 260, 3), dtype=np.uint8) * 255
    rois = roi_crops(crop, purposes={"qr", "barcode"})
    assert rois
    assert all(name and img.size > 0 for name, img in rois)
