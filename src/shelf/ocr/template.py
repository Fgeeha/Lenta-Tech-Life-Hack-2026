"""Классификатор шаблона ценника.

Определяет тип ценника по:
1. Цветовой гистограмме HSV (цвет фона)
2. Ключевым словам в OCR-тексте (механика акции)

Оси классификации (из §3.6 CLAUDE.md):
- Цвет: red, yellow, white, green, black
- Механика: regular (РПЦ), discount_pct (-XX%), discount_rub (-XXX₽),
            bogof (3=2), from_n (от N), up_to_n (до N)
"""

import re

import cv2
import numpy as np

# HSV-диапазоны (оранжевый у Ленты — "red" в GT)
# Saturation minimum lowered to 40 for red to handle glare-washed crops
# where the orange-red desaturates but hue stays in 0-25 range.
_COLOR_RANGES = {
    "red": [
        {"lo": np.array([0, 40, 60]), "hi": np.array([25, 255, 255])},
        {"lo": np.array([155, 40, 60]), "hi": np.array([180, 255, 255])},
    ],
    "yellow": [
        {"lo": np.array([20, 80, 80]), "hi": np.array([45, 255, 255])},
    ],
    "green": [
        {"lo": np.array([35, 60, 60]), "hi": np.array([85, 255, 255])},
    ],
}

# Ключевые слова для классификации механики
_DISCOUNT_PCT_RE = re.compile(r"-\s*\d+\s*%")
_DISCOUNT_RUB_RE = re.compile(r"-\s*\d+\s*[рр₽Р]")
_BOGOF_RE = re.compile(
    r"[23]\s*[=+]\s*[12]|при покупке|бесплатно", re.IGNORECASE
)
_FROM_N_RE = re.compile(r"от\s*\d+", re.IGNORECASE)
_CARD_RE = re.compile(r"карт|по карте|с картой", re.IGNORECASE)


def classify_color(crop: np.ndarray) -> str:
    """Определить доминирующий цвет ценника по HSV-гистограмме.

    Возвращает red/yellow/green/white/black. Для слабого цветового сигнала
    смотрим яркость: так белые регулярные ценники не становятся случайно yellow.
    """
    if crop is None or crop.size == 0:
        return "white"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    best_color = "white"
    best_frac = 0.0
    for color, ranges in _COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            mask |= cv2.inRange(hsv, r["lo"], r["hi"])
        frac = mask.sum() / (255.0 * mask.size)
        if frac > best_frac:
            best_frac = frac
            best_color = color
    if best_frac >= 0.04:
        return best_color
    v_mean = float(hsv[..., 2].mean())
    return "black" if v_mean < 70 else "white"


def classify_mechanic(texts: list[str]) -> str:
    """Определить механику ценника по OCR-текстам."""
    combined = " ".join(texts)
    if _BOGOF_RE.search(combined):
        return "bogof"
    if _FROM_N_RE.search(combined):
        return "from_n"
    if _DISCOUNT_PCT_RE.search(combined):
        return "discount_pct"
    if _DISCOUNT_RUB_RE.search(combined):
        return "discount_rub"
    if _CARD_RE.search(combined):
        return "regular_with_card"
    return "regular"
