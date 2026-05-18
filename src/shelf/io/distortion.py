"""Camera lens distortion correction — official Lenta calibration (17 May 2026).

Correct usage pattern:
    crop = corrector.undistort_crop_at_orig_bbox(frame, (x1, y1, x2, y2))
    # OCR on crop — text is geometrically straight
    # bbox x1/y1/x2/y2 written to submission CSV UNCHANGED (original distorted coords)

Why coords must be mapped: cv2.remap shifts each pixel by up to ~100px for strong
barrel distortion (k1=-0.276). Cropping the undistorted frame at ORIGINAL coords
samples the wrong region. cv2.undistortPoints maps original → undistorted coords.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

# Official calibration from Lenta (17 May 2026, 18:30)
_W, _H = 3840, 2160
_FOCAL_MM = 2.8
_DIAGONAL_MM = 16.0 / 2.8  # ≈ 5.714 mm
_DIST_COEFFS = [-0.276, 0.06, 0.0084, -0.0016, -0.0044]  # k1, k2, p1, p2, k3


class DistortionCorrector:
    """Undistorts crops using official Lenta camera calibration.

    Primary method: undistort_crop_at_orig_bbox(frame, bbox)
      - Remaps full frame (cached per frame_id)
      - Maps bbox corners through cv2.undistortPoints to correct location
      - Returns crop from undistorted frame at mapped coords
      - Original bbox coords are NOT modified (safe for submission CSV)

    Debug only: undistort_full_frame(frame) — applies ROI crop, changes coords.
    """

    def __init__(self) -> None:
        aspect = _W / _H
        h_mm = _DIAGONAL_MM / math.sqrt(aspect**2 + 1)
        w_mm = aspect * h_mm
        fx = _FOCAL_MM * _W / w_mm
        fy = _FOCAL_MM * _H / h_mm
        self._K = np.array([[fx, 0, _W / 2], [0, fy, _H / 2], [0, 0, 1]], dtype=np.float32)
        self._dist = np.array(_DIST_COEFFS, dtype=np.float32)

        self._new_K, self._roi = cv2.getOptimalNewCameraMatrix(
            self._K, self._dist, (_W, _H), 0, (_W, _H)
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._K, self._dist, None, self._new_K, (_W, _H), cv2.CV_32FC1
        )
        # Per-frame cache: remap costs ~200 ms; reuse across all tags in a frame
        self._cache_id: int | None = None
        self._cache_frame: np.ndarray | None = None

    def _cached_remap(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        if self._cache_id != frame_id:
            self._cache_frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
            self._cache_id = frame_id
        return self._cache_frame  # type: ignore[return-value]

    def undistort_bbox_coords(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Map bbox corners from original (distorted) to undistorted pixel coords.

        Use the returned coords to crop from the undistorted frame so the crop
        contains the same physical region as the original bbox.
        """
        x1, y1, x2, y2 = bbox
        pts = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32)
        pts_u = cv2.undistortPoints(pts, self._K, self._dist, P=self._new_K)
        return (
            float(pts_u[0, 0, 0]),
            float(pts_u[0, 0, 1]),
            float(pts_u[1, 0, 0]),
            float(pts_u[1, 0, 1]),
        )

    def undistort_crop_at_orig_bbox(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_id: int = -1,
    ) -> np.ndarray:
        """Return undistorted crop at the physical location of the original bbox.

        1. Remap full frame (cached by frame_id, -1 disables cache)
        2. Map bbox corners through cv2.undistortPoints
        3. Crop undistorted frame at mapped coords

        The original bbox is NOT modified — submission CSV coords are safe.
        """
        undist = self._cached_remap(frame, frame_id)
        ux1, uy1, ux2, uy2 = self.undistort_bbox_coords(bbox)
        h, w = undist.shape[:2]
        ix1 = max(0, min(w - 1, int(ux1)))
        iy1 = max(0, min(h - 1, int(uy1)))
        ix2 = max(ix1 + 1, min(w, int(ux2) + 1))
        iy2 = max(iy1 + 1, min(h, int(uy2) + 1))
        return undist[iy1:iy2, ix1:ix2]

    def undistort_full_frame(self, frame: np.ndarray) -> np.ndarray:
        """Remap + ROI crop. DEBUG / visualisation only — coords change."""
        undist = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        x, y, w, h = self._roi
        return undist[y : y + h, x : x + w]

    # Legacy compat
    def get_undistorted_frame(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        return self._cached_remap(frame, frame_id)


_corrector: DistortionCorrector | None = None


def get_corrector() -> DistortionCorrector:
    global _corrector
    if _corrector is None:
        _corrector = DistortionCorrector()
    return _corrector


def undistort_ocr_enabled() -> bool:
    # Smoke test v1 (May 18): wrong crop coords → 0/61. Default OFF until v2 passes.
    # Enable with SHELF_UNDISTORT_OCR=1.
    return os.environ.get("SHELF_UNDISTORT_OCR", "0").strip() not in ("0", "false", "False")


def get_undistorted_frame(frame: np.ndarray, frame_id: int) -> np.ndarray:
    return get_corrector().get_undistorted_frame(frame, frame_id)
