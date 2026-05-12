"""Классификатор шаблона ценника по цвету (заглушка)."""

import numpy as np


def classify_template(crop: np.ndarray) -> str:
    """Определить цвет фона ценника по гистограмме HSV."""
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))

    if s_mean < 40:
        return "white"
    if 20 <= h_mean <= 35:
        return "yellow"
    if h_mean <= 10 or h_mean >= 170:
        return "red"
    if 35 < h_mean < 85:
        return "green"
    if 100 <= h_mean < 140:
        return "black"
    return "unknown"
