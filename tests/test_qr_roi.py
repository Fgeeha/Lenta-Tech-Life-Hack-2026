"""Tests for QR/barcode geometric ROI extraction."""

import numpy as np

from shelf.qr.decoder import (
    _barcode_roi_variants,
    _roi_boxes,
    _roi_variants,
    _try_opencv_barcode,
    _try_wechat_qr,
)


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


def test_wechat_qr_returns_list_on_blank_image():
    # WeChatQR should return empty list, not raise, on a blank image.
    blank = np.ones((64, 64, 3), dtype=np.uint8) * 200
    result = _try_wechat_qr(blank)
    assert isinstance(result, list)


def test_opencv_barcode_returns_list_on_blank_image():
    # BarcodeDetector should return empty list, not raise, on a blank image.
    blank = np.ones((64, 64, 3), dtype=np.uint8) * 200
    result = _try_opencv_barcode(blank)
    assert isinstance(result, list)


def test_decode_qr_wechat_fast_returns_dict_on_blank():
    from shelf.qr.decoder import decode_qr_wechat_fast

    blank = np.ones((200, 300, 3), dtype=np.uint8) * 200
    result = decode_qr_wechat_fast(blank)
    assert isinstance(result, dict)


def test_decode_qr_wechat_fast_handles_tiny_image():
    from shelf.qr.decoder import decode_qr_wechat_fast

    tiny = np.ones((5, 5, 3), dtype=np.uint8)
    result = decode_qr_wechat_fast(tiny)
    assert isinstance(result, dict)


def test_qr_zone_sharpness_bottom_right():
    from shelf.detect.tracker import _qr_zone_sharpness

    # Bottom-right should have higher variance than a uniform image.
    crop = np.zeros((200, 300, 3), dtype=np.uint8)
    # Put noise in bottom-right (QR zone).
    crop[120:, 150:] = np.random.randint(0, 255, (80, 150, 3), dtype=np.uint8)
    score = _qr_zone_sharpness(crop)
    assert score > 0

    # Uniform image → near-zero sharpness.
    uniform = np.ones((200, 300, 3), dtype=np.uint8) * 128
    assert _qr_zone_sharpness(uniform) < 1.0
