"""Tests for multi-frame alignment and blending."""

import numpy as np
import pytest

from shelf.mfsr.align_blend import align_and_blend


def _make_frame(h: int = 100, w: int = 80, color: int = 128) -> np.ndarray:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    # Add some texture for phase correlation to work with
    img[20:30, 10:60] = 200
    img[50:70, 5:75] = 80
    return img


def test_single_frame_passthrough():
    img = _make_frame()
    result = align_and_blend([(1.0, img, 0.0)])
    assert result.shape == img.shape
    np.testing.assert_array_equal(result, img)


def test_empty_frames_returns_zeros():
    result = align_and_blend([])
    assert result.shape == (1, 1, 3)


def test_two_identical_frames_blend():
    img = _make_frame()
    frames = [(2.0, img.copy(), 0.0), (1.0, img.copy(), 100.0)]
    result = align_and_blend(frames)
    assert result.shape == img.shape
    # Blended result of identical frames should be identical
    np.testing.assert_allclose(result.astype(float), img.astype(float), atol=2)


def test_small_shift_is_corrected():
    ref = _make_frame()
    import cv2
    # Shift by (2, 1) pixels
    M = np.float32([[1, 0, 2], [0, 1, 1]])
    shifted = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]))
    frames = [(2.0, ref.copy(), 0.0), (1.0, shifted, 100.0)]
    result = align_and_blend(frames, max_shift=5)
    assert result.shape == ref.shape
    # Should not crash; result should be close to ref
    diff = np.abs(result.astype(int) - ref.astype(int)).mean()
    assert diff < 20, f"Blend should be close to reference, got mean diff {diff:.1f}"


def test_large_shift_frame_skipped():
    ref = _make_frame()
    import cv2
    # Shift by 50 pixels (way above max_shift=10)
    M = np.float32([[1, 0, 50], [0, 1, 0]])
    far_shifted = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]))
    frames = [(2.0, ref.copy(), 0.0), (1.0, far_shifted, 100.0)]
    result = align_and_blend(frames, max_shift=10)
    # Only 1 frame should remain after filtering outlier
    np.testing.assert_array_equal(result, ref)


def test_five_frames_blend():
    frames = [(float(i + 1), _make_frame(color=120 + i * 5), float(i * 100)) for i in range(5)]
    result = align_and_blend(frames)
    assert result.shape == frames[0][1].shape
    assert result.dtype == np.uint8
