"""Предобработка кропа ценника перед OCR/QR.

Goals:
1. rotate price tags into a readable orientation;
2. reduce glare and low contrast;
3. deskew small angular errors;
4. optionally correct perspective when a rectangular paper contour is visible.
"""

from __future__ import annotations

import cv2
import numpy as np


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def deskew_angle(img: np.ndarray) -> float:
    """Estimate text skew angle in degrees using foreground min-area rectangle."""
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 30:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    if angle > 45:
        angle -= 90
    return -float(angle)


def suppress_glare(img: np.ndarray) -> np.ndarray:
    """Inpaint small overexposed glare spots.

    This helps on glossy/glass shelves without changing the whole color layout.
    """
    if img is None or img.size == 0 or img.ndim != 3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s < 45) & (v > 235)).astype(np.uint8) * 255
    # Ignore tiny salt-noise and avoid inpainting if almost the whole image is white.
    if mask.mean() < 1 or mask.mean() > 95:
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_correct(
    img: np.ndarray, min_area_frac: float = 0.20
) -> np.ndarray:
    """Try to rectify a visible rectangular price tag contour.

    If no reliable four-point contour is found, returns the original image.
    """
    if img is None or img.size == 0:
        return img
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return img
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        area = cv2.contourArea(c)
        if area < H * W * min_area_frac:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        rect = _order_points(approx.reshape(4, 2).astype("float32"))
        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_w = int(max(width_a, width_b))
        max_h = int(max(height_a, height_b))
        if max_w < 40 or max_h < 40:
            continue
        dst = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(
            img,
            M,
            (max_w, max_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return img


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on the L-channel, conservative settings."""
    if img is None or img.size == 0 or img.ndim != 3:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lch, ach, bch = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    lch = clahe_obj.apply(lch)
    return cv2.cvtColor(cv2.merge([lch, ach, bch]), cv2.COLOR_LAB2BGR)


def preprocess_crop(
    crop: np.ndarray,
    rotate_180: bool = True,  # retained for backward compatibility; means "orient for OCR"
    deskew: bool = True,
    upscale: int = 2,
    sharpen: bool = False,
    clahe: bool = False,
    glare: bool = True,
    perspective: bool = False,
    use_sr: bool | None = None,
) -> np.ndarray:
    """Prepare a price-tag crop for OCR.

    Most supplied Lenta frames store tags sideways; when ``rotate_180`` is True
    we rotate 90° counter-clockwise, matching the previous project behavior.

    When ``use_sr`` is None the SHELF_SR_ENABLED environment variable controls
    whether RealESRGAN 4x super-resolution replaces the Lanczos upscale step.
    """
    if crop is None or crop.size == 0:
        return crop

    img = crop.copy()

    if perspective:
        img = perspective_correct(img)

    if rotate_180:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if glare:
        img = suppress_glare(img)

    if deskew:
        angle = deskew_angle(img)
        if 1.0 < abs(angle) < 20:
            img = _rotate_image(img, angle)

    # SR replaces Lanczos upscale when enabled.
    _use_sr = use_sr if use_sr is not None else _sr_enabled()
    if _use_sr:
        from shelf.ocr.sr import upscale_crop as _sr_upscale

        img = _sr_upscale(img)
    elif upscale > 1:
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (w * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4
        )

    if clahe:
        img = enhance_contrast(img)

    if sharpen:
        kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
        )
        img = cv2.filter2D(img, -1, kernel)

    return img


def _sr_enabled() -> bool:
    from shelf.ocr.sr import is_enabled

    return is_enabled()


def ocr_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Small set of OCR variants ordered from safest to most aggressive."""
    if crop is None or crop.size == 0:
        return []

    if _sr_enabled():
        # Pre-process once (no upscale), then run SR once → 3 post-processing variants.
        base = preprocess_crop(
            crop, upscale=1, glare=True, deskew=True, sharpen=False, clahe=False, use_sr=False
        )
        from shelf.ocr.sr import upscale_crop as _sr_upscale

        sr_img = _sr_upscale(base)
        _k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return [
            sr_img,
            enhance_contrast(sr_img.copy()),
            cv2.filter2D(sr_img, -1, _k),
        ]

    base = preprocess_crop(
        crop, upscale=2, glare=True, deskew=True, sharpen=False, clahe=False, use_sr=False
    )
    return [
        base,
        preprocess_crop(
            crop, upscale=3, glare=True, deskew=True, sharpen=False, clahe=True, use_sr=False
        ),
        preprocess_crop(
            crop, upscale=2, glare=True, deskew=True, sharpen=True, clahe=False, use_sr=False
        ),
    ]


def qr_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Image variants useful for QR/barcode decoding."""
    if crop is None or crop.size == 0:
        return []
    variants: list[np.ndarray] = []
    for rot in (
        None,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_CLOCKWISE,
    ):
        img = cv2.rotate(crop, rot) if rot is not None else crop
        img = suppress_glare(img)
        variants.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    return variants
