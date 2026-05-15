"""Tests for EAN-13 repair and barcode ROI utilities."""

import numpy as np

from shelf.qr.barcode_roi import (
    ean13_checksum_valid,
    ean13_repair,
    find_barcode_strip,
    read_barcode_from_strip,
)

# ---------------------------------------------------------------------------
# EAN-13 checksum validation
# ---------------------------------------------------------------------------


def test_valid_ean13():
    assert ean13_checksum_valid("4607124143901")
    assert ean13_checksum_valid("4670025474665")
    assert ean13_checksum_valid("4603552017456")
    assert ean13_checksum_valid("4607030000770")


def test_invalid_ean13_wrong_checksum():
    assert not ean13_checksum_valid("4607124143900")
    assert not ean13_checksum_valid("4607124143902")


def test_invalid_ean13_short():
    assert not ean13_checksum_valid("460712414390")
    assert not ean13_checksum_valid("")


def test_invalid_ean13_non_digits():
    assert not ean13_checksum_valid("460712414390X")


# ---------------------------------------------------------------------------
# EAN-13 repair
# ---------------------------------------------------------------------------


def test_repair_already_valid():
    assert ean13_repair("4607124143901") == "4607124143901"


def test_repair_12_digits():
    # Should append correct check digit
    result = ean13_repair("460712414390")
    assert result == "4607124143901"


def test_repair_spaced_barcode():
    # 49_5 GT format: "4 607124 143901"
    result = ean13_repair("4 607124 143901")
    assert result == "4607124143901"


def test_repair_14_digits_drop_first():
    # Prepend an extra digit
    result = ean13_repair("04607124143901")
    assert result == "4607124143901"


def test_repair_14_digits_drop_last():
    result = ean13_repair("4607124143901X")
    # X stripped → 13 digits, valid
    assert result == "4607124143901"


def test_repair_one_digit_fixup():
    # Contract: repair returns SOME valid EAN-13 differing by ≤1 digit from input
    valid = "4670025474665"
    corrupted = valid[:-1] + "0"  # wrong check digit
    result = ean13_repair(corrupted)
    assert result is not None
    assert ean13_checksum_valid(result)
    diffs = sum(a != b for a, b in zip(result, corrupted))
    assert diffs <= 1


def test_repair_non_ean_returns_none():
    assert ean13_repair("12345") is None
    assert ean13_repair("abcdef") is None
    assert ean13_repair("") is None


# ---------------------------------------------------------------------------
# find_barcode_strip / read_barcode_from_strip (smoke tests on synthetic data)
# ---------------------------------------------------------------------------


def test_find_barcode_strip_empty():
    result = find_barcode_strip(np.zeros((100, 100, 3), dtype=np.uint8))
    # No barcode pattern in blank image
    assert result is None


def test_read_barcode_from_strip_empty():
    result = read_barcode_from_strip(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result == ""


def test_read_barcode_from_strip_none():
    result = read_barcode_from_strip(None)
    assert result == ""


def test_find_barcode_strip_with_vertical_lines():
    # Create a synthetic image with vertical stripes in the middle
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    # Draw barcode-like vertical stripes in a horizontal band
    for x in range(50, 350, 4):
        img[80:130, x : x + 2] = 0
    result = find_barcode_strip(img)
    assert result is not None
    x, y, w, h = result
    assert w >= 80  # should cover most of the stripe area
    assert 70 <= y <= 100  # should land around the stripe rows
