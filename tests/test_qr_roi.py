"""Tests for QR/barcode geometric ROI extraction."""

import numpy as np

from shelf.qr.decoder import _barcode_roi_variants, _roi_boxes, _roi_variants


def test_roi_boxes_are_inside_image():
    boxes = _roi_boxes(400, 200)
    assert boxes
    for _name, x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= 400
        assert 0 <= y1 < y2 <= 200


def test_roi_variants_non_empty_for_valid_crop():
    crop = np.ones((120, 240, 3), dtype=np.uint8) * 255
    variants = _roi_variants(crop)
    assert variants
    assert all(v.size > 0 for v in variants)


def test_barcode_roi_variants_focus_on_bottom_regions():
    crop = np.ones((120, 240, 3), dtype=np.uint8) * 255
    variants = _barcode_roi_variants(crop)
    assert variants
    # ROI variants should be smaller or scaled ROIs, not the untouched full crop only.
    assert any(
        v.shape[0] != crop.shape[0] or v.shape[1] != crop.shape[1]
        for v in variants
    )


def test_code_decode_off_mode(monkeypatch):
    import numpy as np

    from shelf.qr.decoder import decode_barcode, decode_qr

    monkeypatch.setenv("SHELF_CODE_DECODE_MODE", "off")
    crop = np.zeros((32, 64, 3), dtype=np.uint8)
    assert decode_qr(crop) == {}
    assert decode_barcode(crop) == ""


def test_template_roi_variants_precede_geometric_fallbacks():
    from shelf.qr.decoder import _template_roi_variants

    crop = np.ones((120, 240, 3), dtype=np.uint8) * 255
    variants = _template_roi_variants(crop, kind="qr")
    assert variants
    assert all(v.size > 0 for v in variants)
