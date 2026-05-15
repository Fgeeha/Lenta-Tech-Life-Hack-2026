"""Tests for Real-ESRGAN SR module (fallback behavior when SR disabled)."""

import os

import numpy as np
import pytest


def test_sr_disabled_by_default():
    from shelf.sr.realesrgan import is_enabled

    # Default environment should have SR disabled
    assert (
        not is_enabled() or os.environ.get("SHELF_USE_SR", "").lower() == "true"
    )


def test_upscale_roi_fallback_correct_size():
    from shelf.sr.realesrgan import upscale_roi

    img = np.zeros((30, 30, 3), dtype=np.uint8)
    result = upscale_roi(img, outscale=4)
    # With SR disabled (default), should use Lanczos4 fallback → 120×120
    assert result.shape == (120, 120, 3)


def test_upscale_roi_fallback_preserves_color():
    from shelf.sr.realesrgan import upscale_roi

    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    result = upscale_roi(img, outscale=4)
    assert result.shape == (80, 80, 3)
    # Solid color should be preserved
    assert result.mean() == pytest.approx(128, abs=5)


def test_upscale_roi_none_input():
    from shelf.sr.realesrgan import upscale_roi

    result = upscale_roi(None, outscale=4)
    assert result is None


def test_upscale_roi_empty_input():
    from shelf.sr.realesrgan import upscale_roi

    img = np.zeros((0, 0, 3), dtype=np.uint8)
    result = upscale_roi(img, outscale=4)
    assert result is not None  # should not crash
