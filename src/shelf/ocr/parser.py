"""Маппинг OCR-боксов → поля ценника.

После поворота 90°CCW структура ценника (сверху вниз в кропе):
  - Белая зона:    product_name, id_sku, print_datetime, barcode, QR-код
  - Оранжевая зона: price_card (крупно), price_default (мелко), discount_amount

Парсинг: сначала discount (чтобы исключить % из цен), затем цены, затем текст.
"""

import re
from dataclasses import dataclass

import cv2
import numpy as np

from shelf.qr.barcode_roi import ean13_repair, read_barcode_from_strip
from shelf.schema import PriceTag

# --- Регулярные выражения ---

# Цена: 129, 129.99, 129,99.
# Минимум 3 цифры: исключаем "99" (копейки, разделённые OCR в отдельный бокс)
# и прочие двузначные шумы. Минимальная цена в Lenta GT ≥ 100 руб.
_PRICE_RE = re.compile(r"\b(\d{3,6})(?:[.,](\d{2}))?\b")

# Скидка: -48%  -23%  48%  23%  (знак минуса опционален — OCR часто не читает)
# Диапазон 1–99% (не 100+), чтобы не захватить коды и артикулы
_DISCOUNT_PCT_RE = re.compile(r"[-–]?\s*(\d{1,2})\s*%")

# Дата: 03.04.2026 3:08
_DATE_RE = re.compile(r"\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2}")

# Штрихкод EAN: 8-14 цифр
_BARCODE_RE = re.compile(r"\b\d{8,14}\b")

# Артикул SKU: 10-12 цифр (Lenta article: 12 цифр, например 270207736530)
# Специально не пересекается с EAN-13 (13 цифр)
_SKU_RE = re.compile(r"\b\d{10,12}\b")

# Специальный символ: К, Л, Ш
_SPECIAL_RE = re.compile(r"\b([КкЛлШш])\b")

# Код зоны: 13_043015
_CODE_RE = re.compile(r"\b\d{2}_\d{6,}\b")

# Паттерн «число+процент» — для фильтрации из ценового парсинга
_PCT_TOKEN_RE = re.compile(r"\d+\s*%")


def _strip_percent_tokens(text: str) -> str:
    """Удалить токены 'NN%' из строки перед поиском цен.

    BUG FIX: раньше '48%' давало price=48 вместо discount=48%.
    """
    return _PCT_TOKEN_RE.sub(" ", text)


def _extract_prices(texts: list[str]) -> list[float]:
    """Извлечь числа, похожие на цены (исключая проценты)."""
    prices = []
    for text in texts:
        clean = _strip_percent_tokens(text)
        for m in _PRICE_RE.finditer(clean):
            integer = int(m.group(1))
            frac = int(m.group(2)) if m.group(2) else 0
            val = integer + frac / 100.0
            # Разумный диапазон: 1 — 99999 руб.
            if 1.0 <= val <= 99_999.0:
                prices.append(val)
    return sorted(prices)


def _find_discount(texts: list[str]) -> str:
    """Найти скидку в виде '−XX%' или 'XX%'.

    BUG FIX: раньше требовался знак минуса, но OCR часто читает '48%' без него.
    """
    for text in texts:
        m = _DISCOUNT_PCT_RE.search(text)
        if m:
            pct = int(m.group(1))
            if 1 <= pct <= 99:
                return f"-{pct}%"
    return "нет"


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
        return int(img.shape[0] * 0.6), img.shape[0]
    return max(0, int(orange_rows[0]) - 10), min(img.shape[0], int(orange_rows[-1]) + 10)


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


def _fmt_price(val: float) -> str:
    """Форматировать цену как строку '129,00'."""
    return f"{val:.2f}".replace(".", ",")


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

    # --- Зоны ---
    price_zone_start = 0.5
    if crop is not None:
        oy0, oy1 = _find_orange_rows(crop)
        price_zone_start = oy0 / max(1, crop.shape[0])

    price_boxes = [b for b in boxes if b.center_y >= price_zone_start]
    info_boxes = [b for b in boxes if b.center_y < price_zone_start]

    # --- 1. Скидка (сначала! чтобы исключить % из ценового парсинга) ---
    discount_amount = _find_discount(all_texts)

    # --- 2. Цены ---
    # Ищем в ценовой зоне сначала, потом во всём тексте
    price_texts = [b.text for b in price_boxes]
    prices_orange = _extract_prices(price_texts)
    prices_all = _extract_prices(all_texts)

    price_card = ""
    price_default = ""

    # Оранжевая зона содержит price_card (крупно) и обычно price_default (мелко)
    # После фильтрации %: ожидаем числа типа [129, 252]
    if len(prices_orange) >= 2:
        # Меньшая = card (акционная), большая = default (без карты)
        price_card = _fmt_price(prices_orange[0])
        price_default = _fmt_price(prices_orange[-1])
    elif len(prices_orange) == 1:
        price_card = _fmt_price(prices_orange[0])
        # Ищем default в белой зоне (там он может быть тоже)
        prices_info = _extract_prices([b.text for b in info_boxes])
        if prices_info:
            candidate = max(prices_info)
            if candidate > prices_orange[0]:
                price_default = _fmt_price(candidate)
    elif len(prices_all) >= 1:
        # Фоллбек: берём из всего текста
        price_card = _fmt_price(prices_all[0])
        if len(prices_all) >= 2:
            price_default = _fmt_price(prices_all[-1])

    # --- 3. Название продукта ---
    product_name = ""
    if info_boxes:
        name_candidates = [b for b in info_boxes if len(b.text) > 3 and b.conf > 0.5]
        if name_candidates:
            product_name = " ".join(b.text for b in name_candidates[:4])

    # --- 4. Штрихкод ---
    barcode = ""
    for text in all_texts:
        m = _BARCODE_RE.search(text)
        if m and len(m.group(0)) >= 10:
            cand = m.group(0)
            barcode = ean13_repair(cand) or cand
            break

    # Fallback A: длинные цифровые последовательности из OCR + EAN-13 repair
    if not barcode:
        for text in all_texts:
            digits = re.sub(r"\D", "", text)
            if 11 <= len(digits) <= 15:
                repaired = ean13_repair(digits)
                if repaired:
                    barcode = repaired
                    break

    # Fallback B: ROI-таргетинг штрихкода через Sobel + pyzbar
    if not barcode and crop is not None:
        barcode = read_barcode_from_strip(crop)

    # --- 5. Артикул ---
    id_sku = ""
    for text in all_texts:
        m = _SKU_RE.search(text)
        if m:
            candidate = m.group(0)
            if candidate != barcode and candidate not in barcode:
                id_sku = candidate
                break

    # --- 6. Дата ---
    print_datetime = ""
    for text in all_texts:
        m = _DATE_RE.search(text)
        if m:
            print_datetime = m.group(0)
            break

    # --- 7. Код зоны ---
    code = "нет"
    for text in all_texts:
        m = _CODE_RE.search(text)
        if m:
            code = m.group(0)
            break

    # --- 8. Специальные символы ---
    special_symbols = "нет"
    for text in all_texts:
        m = _SPECIAL_RE.search(text)
        if m:
            special_symbols = m.group(1).upper()
            break

    # --- 9. additional_info ---
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
