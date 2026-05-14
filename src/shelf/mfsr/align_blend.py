"""Multi-frame alignment + weighted blend for price-tag crop fusion.

Algorithm:
1. Reference frame = sharpest (highest area×sharpness score).
2. Remaining frames aligned via phase correlation (sub-pixel FFT shift).
3. Frames with excessive shift (> max_shift px) are outliers and skipped.
4. Weighted average blend with weights proportional to sharpness score.

This is a classical Drizzle-style accumulation: multiple slightly-shifted
frames give sub-pixel information without neural-network hallucinations.
Expected gain: +1-3 pp on price_card / price_default accuracy.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum allowed sub-pixel translation between frames (pixels).
# Beyond this the frames are from different positions (robot moved too far).
_MAX_SHIFT_PX = 10


def align_and_blend(
    frames: "list[tuple[float, np.ndarray, float]]",
    max_shift: int = _MAX_SHIFT_PX,
) -> "np.ndarray":
    """Align top frames to reference and return a sharpness-weighted blend.

    Args:
        frames: List of (score, crop_bgr, timestamp) sorted descending by score.
                score = det.area × Laplacian_sharpness from ByteTrack.
        max_shift: Maximum allowed translation in pixels. Frames exceeding this
                   are treated as outliers (camera moved too far) and skipped.

    Returns:
        Blended BGR uint8 image the same size as the reference frame.
    """
    if not frames:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if len(frames) == 1:
        return frames[0][1]

    ref_score, ref_img, _ = frames[0]
    if ref_img is None or ref_img.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    h, w = ref_img.shape[:2]
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Hamming window reduces boundary artefacts in phase correlation
    win_h = np.hamming(h)
    win_w = np.hamming(w)
    win2d = np.outer(win_h, win_w).astype(np.float32)

    aligned: list[tuple[float, np.ndarray]] = [(ref_score, ref_img)]

    for score, img, _ in frames[1:]:
        if img is None or img.size == 0:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Phase correlation with windowing for sub-pixel accuracy
        (dx, dy), _ = cv2.phaseCorrelate(
            ref_gray * win2d, img_gray * win2d
        )

        if abs(dx) > max_shift or abs(dy) > max_shift:
            logger.debug("Frame skip: shift=(%.1f, %.1f) > max=%d", dx, dy, max_shift)
            continue

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        aligned.append((score, warped))

    if len(aligned) == 1:
        return aligned[0][1]

    total_score = sum(s for s, _ in aligned)
    result = np.zeros((h, w, 3), dtype=np.float64)
    for score, frame_bgr in aligned:
        result += (score / total_score) * frame_bgr.astype(np.float64)

    blended = np.clip(result, 0, 255).astype(np.uint8)
    logger.debug("Blended %d/%d frames, total_score=%.0f", len(aligned), len(frames), total_score)
    return blended
