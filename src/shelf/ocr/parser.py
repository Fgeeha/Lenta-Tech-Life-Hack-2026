"""Маппинг OCR-боксов → поля ценника.

После поворота 90°CCW структура ценника (сверху вниз в кропе):
  - Белая зона: product_name, id_sku, print_datetime, barcode, QR-код
  - Оранжевая зона: price_card (крупно), price_default (мелко), discount_amount

Парсинг: сначала ищем числовые паттерны (цены, штрихкоды),
затем текстовые (название, скидка).
"""

import re
from dataclasses import dataclass

import cv2
import numpy as np

from shelf.schema import PriceTag

# --- Регулярные выражения ---
# Цена: 129, 129.99, 129,99  (от 2 до 6 цифр, опционально дробная часть)
_PRICE_RE = re.compile(r"\b(\d{2,6})(?:[.,](\d{2}))?\b")
# Дата: 03.04.2026 3:08
_DATE_RE = re.compile(r"\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2}")
# Штрихкод EAN: 8-14 цифр подряд
_BARCODE_RE = re.compile(r"\b\d{8,14}\b")
# Артикул SKU: 6-10 цифр (не EAN)
_SKU_RE = re.compile(r"\b\d{6,10}\b")
# Скидка: -48%  или  -23%
_DISCOUNT_PCT_RE = re.compile(r"[-–]\s*(\d{1,2})\s*%")
# Специальный символ: К, Л, Ш
_SPECIAL_RE = re.compile(r"\b([КкЛлШш])\b")
# Код зоны: 13_043015
_CODE_RE = re.compile(r"\b\d{2}_\d{6,}\b")


def _find_orange_rows(img: np.ndarray) -> tuple[int, int]:
    """Найти строки с оранжевым фоном (ценовая зона)."""
    if img is None or img.size == 0:
        return 0, img.shape[0] if img is not None else 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([8, 60, 80]), np.array([40, 255, 255]))
    row_sums = mask.sum(axis=1) / 255.0
    threshold = img.shape[1] * 0.15
    orange_rows = np.where(row_sums > threshold)[0]
    if len(orange_rows) == 0:
        # Нет оранжевого → берём нижние 40% как ценовую зону
        return int(img.shape[0] * 0.6), img.shape[0]
    return max(0, int(orange_rows[0]) - 10), min(img.shape[0], int(orange_rows[-1]) + 10)


def _extract_prices(texts: list[str]) -> list[float]:
    """Извлечь все числа, похожие на цены."""
    prices = []
    for text in texts:
        for m in _PRICE_RE.finditer(text):
            integer = int(m.group(1))
            frac = int(m.group(2)) if m.group(2) else 0
            val = integer + frac / 100.0
            if 1.0 <= val <= 99999.0:
                prices.append(val)
    return sorted(prices)


@dataclass
class OCRBox:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


def _normalize_box(box: list, w: int, h: int) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h


def parse_ocr_result(
    ocr_lines: list[tuple[list, str, float]],
    crop: "np.ndarray | None" = None,
    filename: str = "",
    frame_timestamp: float = 0.0,
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0),
    color: str = "red",
) -> PriceTag:
    """Извлечь поля ценника из OCR-результатов.

    crop — препроцессированный (90°CCW + upscale) кроп, используется
    для нахождения оранжевой зоны (ценовой секции).
    """
    x_min, y_min, x_max, y_max = bbox
    crop_h = max(1, y_max - y_min)
    crop_w = max(1, x_max - x_min)

    # --- Боксы ---
    boxes: list[OCRBox] = []
    for raw_box, text, conf in ocr_lines:
        if not text.strip() or conf < 0.3:
            continue
        h_img = crop.shape[0] if crop is not None else crop_h
        w_img = crop.shape[1] if crop is not None else crop_w
        x0, y0, x1, y1 = _normalize_box(raw_box, w_img, h_img)
        boxes.append(OCRBox(text=text.strip(), conf=conf, x0=x0, y0=y0, x1=x1, y1=y1))

    if not boxes:
        return PriceTag(
            filename=filename,
            frame_timestamp=frame_timestamp,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            color=color,
        )

    boxes.sort(key=lambda b: b.center_y)
    all_texts = [b.text for b in boxes]

    # --- Оранжевая зона (ценовая секция) ---
    # После 90°CCW в препроцессированном кропе оранжевая зона = нижняя часть
    price_zone_start = 0.5  # нижние 50% по умолчанию
    if crop is not None:
        oy0, oy1 = _find_orange_rows(crop)
        price_zone_start = oy0 / max(1, crop.shape[0])

    price_boxes = [b for b in boxes if b.center_y >= price_zone_start]
    info_boxes = [b for b in boxes if b.center_y < price_zone_start]

    # --- Цены ---
    price_card = ""
    price_default = ""
    all_prices = _extract_prices([b.text for b in price_boxes] + all_texts)

    if len(all_prices) >= 2:
        price_card = f"{all_prices[0]:.2f}".replace(".", ",")
        price_default = f"{all_prices[-1]:.2f}".replace(".", ",")
    elif len(all_prices) == 1:
        price_card = f"{all_prices[0]:.2f}".replace(".", ",")

    # --- Скидка ---
    discount_amount = "нет"
    for text in all_texts:
        m = _DISCOUNT_PCT_RE.search(text)
        if m:
            discount_amount = f"-{m.group(1)}%"
            break

    # --- Название продукта ---
    product_name = ""
    if info_boxes:
        # Берём боксы с наибольшим conf и размером из информационной зоны
        name_candidates = [b for b in info_boxes if len(b.text) > 3 and b.conf > 0.5]
        if name_candidates:
            product_name = " ".join(b.text for b in name_candidates[:4])

    # --- Штрихкод ---
    barcode = ""
    for text in all_texts:
        m = _BARCODE_RE.search(text)
        if m and len(m.group(0)) >= 10:
            barcode = m.group(0)
            break

    # --- Артикул ---
    id_sku = ""
    for text in all_texts:
        m = _SKU_RE.search(text)
        if m and 6 <= len(m.group(0)) <= 9:
            id_sku = m.group(0)
            break

    # --- Дата ---
    print_datetime = ""
    for text in all_texts:
        m = _DATE_RE.search(text)
        if m:
            print_datetime = m.group(0)
            break

    # --- Код зоны ---
    code = "нет"
    for text in all_texts:
        m = _CODE_RE.search(text)
        if m:
            code = m.group(0)
            break

    # --- Специальные символы ---
    special_symbols = "нет"
    for text in all_texts:
        m = _SPECIAL_RE.search(text)
        if m:
            special_symbols = m.group(1).upper()
            break

    # --- additional_info ---
    used = {barcode, id_sku, print_datetime}
    extra = [b.text for b in boxes if b.text not in used and len(b.text) > 5 and b.conf > 0.5]
    additional_info = " | ".join(extra[:2]) if extra else "нет"

    return PriceTag(
        filename=filename,
        frame_timestamp=frame_timestamp,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        product_name=product_name,
        price_default=price_default,
        price_card=price_card,
        price_discount="нет",
        barcode=barcode,
        discount_amount=discount_amount,
        id_sku=id_sku,
        print_datetime=print_datetime,
        code=code,
        additional_info=additional_info,
        color=color,
        special_symbols=special_symbols,
    )
