"""Camera lens distortion correction — official Lenta calibration (17 May 2026).

IMPORTANT: Only undistort_crop_preserve_coords() / the per-frame cache should be
used inside the pipeline.  undistort_full_frame() shifts the coordinate system
and MUST NOT be used where bbox values are later written to the submission CSV.

Correct usage pattern (per detection loop):
    undist = get_undistorted_frame(frame, frame_id)  # cached remap, no coord shift
    crop = undist[y1:y2, x1:x2]                      # crop at ORIGINAL bbox coords
    # OCR on crop — text is now straight
    # bbox x1/y1/x2/y2 written to CSV unchanged (original coords)
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
    """Undistorts frames or crops using official Lenta camera calibration.

    Two modes of use:
    - get_undistorted_frame(frame, frame_id): cached remap of full frame,
      NO ROI crop — coordinates preserved for bbox compatibility.
    - undistort_full_frame(frame): remap + ROI crop (changes coords, debug only).
    """

    def __init__(self) -> None:
        aspect = _W / _H
        h_mm = _DIAGONAL_MM / math.sqrt(aspect**2 + 1)
        w_mm = aspect * h_mm
        fx = _FOCAL_MM * _W / w_mm
        fy = _FOCAL_MM * _H / h_mm
        K = np.array([[fx, 0, _W / 2], [0, fy, _H / 2], [0, 0, 1]], dtype=np.float32)
        dist = np.array(_DIST_COEFFS, dtype=np.float32)

        # alpha=0: no black borders, slight crop at edges
        new_K, self._roi = cv2.getOptimalNewCameraMatrix(K, dist, (_W, _H), 0, (_W, _H))
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, (_W, _H), cv2.CV_32FC1
        )
        # Per-frame cache (remap is ~200 ms; reuse across all tags in a frame)
        self._cache_id: int | None = None
        self._cache_frame: np.ndarray | None = None

    def get_undistorted_frame(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """Remap full frame, NO ROI crop — original (W, H) preserved.

        Bboxes from detection remain valid in this coordinate system.
        Call once per frame; subsequent calls with the same frame_id reuse cache.
        """
        if self._cache_id != frame_id:
            self._cache_frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
            self._cache_id = frame_id
        return self._cache_frame  # type: ignore[return-value]

    def undistort_full_frame(self, frame: np.ndarray) -> np.ndarray:
        """Remap + ROI crop.  DEBUG / visualisation only — coords change."""
        undist = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        x, y, w, h = self._roi
        return undist[y : y + h, x : x + w]


_corrector: DistortionCorrector | None = None


def get_corrector() -> DistortionCorrector:
    global _corrector
    if _corrector is None:
        _corrector = DistortionCorrector()
    return _corrector


def undistort_ocr_enabled() -> bool:
    # Smoke test (May 18): undistort 49_5 → 0/61 vs baseline 1/61. Default OFF.
    # Enable with SHELF_UNDISTORT_OCR=1 if future OCR engine benefits from it.
    return os.environ.get("SHELF_UNDISTORT_OCR", "0").strip() not in ("0", "false", "False")


def get_undistorted_frame(frame: np.ndarray, frame_id: int) -> np.ndarray:
    """Module-level helper: get cached undistorted frame (no coord shift)."""
    return get_corrector().get_undistorted_frame(frame, frame_id)
