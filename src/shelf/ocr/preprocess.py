"""Предобработка кропа ценника перед OCR.

Шаги:
1. Поворот на 180° (ценники смонтированы вниз головой)
2. Deskew через Canny + HoughLines (коррекция угла наклона)
3. Upscale + sharpening (для лучшего OCR)
4. CLAHE (выравнивание гистограммы) для улучшения контраста
"""

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
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def deskew_angle(img: np.ndarray) -> float:
    """Определить угол наклона текста через минимальную ограничивающую область."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect возвращает угол -90..0; нормализуем к -45..45
    if angle < -45:
        angle += 90
    return -angle  # знак для warpAffine


def preprocess_crop(
    crop: np.ndarray,
    rotate_180: bool = True,  # параметр оставлен для обратной совместимости
    deskew: bool = True,
    upscale: int = 5,
    sharpen: bool = True,
    clahe: bool = True,
) -> np.ndarray:
    """Подготовить кроп ценника для OCR.

    Ценники Ленты смонтированы боком: правильный поворот 90°CCW.
    Параметр rotate_180 сохранён для совместимости, фактически делаем 90°CCW.
    """
    if crop is None or crop.size == 0:
        return crop

    img = crop.copy()

    # 1. Поворот 90°CCW — ценники смонтированы боком, не 180°
    if rotate_180:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 2. Deskew (коррекция наклона ±15°)
    if deskew:
        angle = deskew_angle(img)
        if abs(angle) > 1.0 and abs(angle) < 30:
            img = _rotate_image(img, angle)

    # 3. Upscale
    if upscale > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)

    # 4. Sharpen
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)

    # 5. CLAHE на L-канале LAB для улучшения контраста
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lch, ach, bch = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lch = clahe_obj.apply(lch)
        img = cv2.cvtColor(cv2.merge([lch, ach, bch]), cv2.COLOR_LAB2BGR)

    return img
