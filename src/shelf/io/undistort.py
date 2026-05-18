"""Lens distortion correction using official Lenta camera calibration.

Coefficients sourced from the official Lenta example (17 May 2026):
  k1=-0.276, k2=0.060, p1=0.0084, p2=-0.0016, k3=-0.0044
Camera: 3840x2160, focal_length=2.8mm, sensor_diagonal=16/2.8 mm
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

# Official Lenta camera parameters (from ChatExport 17 May 2026)
_W, _H = 3840, 2160
_FOCAL_MM = 2.8
_DIAGONAL_MM = 16.0 / 2.8  # ≈ 5.714 mm
_DIST_COEFFS = [-0.276, 0.06, 0.0084, -0.0016, -0.0044]  # k1,k2,p1,p2,k3


def _build_camera_matrix(w: int, h: int, focal_mm: float, diagonal_mm: float) -> np.ndarray:
    aspect = w / h
    h_mm = diagonal_mm / math.sqrt(aspect**2 + 1)
    w_mm = aspect * h_mm
    fx = focal_mm * w / w_mm
    fy = focal_mm * h / h_mm
    return np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float32)


class DistortionCorrector:
    """Applies lens undistortion and ROI crop (official Lenta calibration)."""

    def __init__(self) -> None:
        K = _build_camera_matrix(_W, _H, _FOCAL_MM, _DIAGONAL_MM)
        dist = np.array(_DIST_COEFFS, dtype=np.float32)
        new_K, self._roi = cv2.getOptimalNewCameraMatrix(K, dist, (_W, _H), 0, (_W, _H))
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, (_W, _H), cv2.CV_32FC1
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        undistorted = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        x, y, w, h = self._roi
        return undistorted[y : y + h, x : x + w]


_corrector: DistortionCorrector | None = None


def get_corrector() -> DistortionCorrector:
    global _corrector
    if _corrector is None:
        _corrector = DistortionCorrector()
    return _corrector


def undistort_enabled() -> bool:
    return os.environ.get("SHELF_UNDISTORT", "0").strip() not in ("0", "", "false", "False")
