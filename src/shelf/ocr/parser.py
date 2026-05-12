"""Маппинг OCR-боксов → поля ценника.

Алгоритм:
1. Нормализуем боксы в координаты [0,1] относительно кропа ценника
2. Классифицируем каждый бокс по позиции (верх/середина/низ, левый/правый)
3. Применяем регулярные выражения для извлечения цен, дат, артикулов
4. QR-поля вставляются через merge.py (не здесь)
"""

import re
from dataclasses import dataclass

import numpy as np

from shelf.schema import PriceTag

# Регулярные выражения
_PRICE_RE = re.compile(r"(\d{1,6})[,.](\d{2})")  # 129,99 или 129.99
_PRICE_INT_RE = re.compile(r"(\d{1,6})\s*₽?(?!\d)")  # 130 (без копеек)
_DATE_RE = re.compile(r"\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2}")  # 03.04.2026 3:08
_BARCODE_RE = re.compile(r"\b\d{8,14}\b")  # штрихкод 8-14 цифр
_SKU_RE = re.compile(r"\b\d{6,10}\b")  # артикул 6-10 цифр
_DISCOUNT_RE = re.compile(r"-\s*(\d+\s*%|\d+\s*[₽рР])")  # -48% или -150₽
_SPECIAL_RE = re.compile(r"\b[кКлЛшШ]\b")  # К/Л/Ш символы
_CODE_RE = re.compile(r"\d{2}_\d{6,}")  # код зоны типа 13_043015


def _normalize_box(box: list, w: int, h: int) -> tuple[float, float, float, float]:
    """Перевести 4-точечный bbox в (x0,y0,x1,y1) нормализованные."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h


def _extract_price(text: str) -> str:
    """Извлечь цену из строки."""
    m = _PRICE_RE.search(text)
    if m:
        return f"{m.group(1)},{m.group(2)}"
    m = _PRICE_INT_RE.search(text)
    if m:
        return m.group(1)
    return ""


@dataclass
class OCRBox:
    text: str
    conf: float
    x0: float  # нормализованные координаты
    y0: float
    x1: float
    y1: float

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def parse_ocr_result(
    ocr_lines: list[tuple[list, str, float]],
    crop: "np.ndarray | None" = None,
    filename: str = "",
    frame_timestamp: float = 0.0,
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0),
    color: str = "red",
) -> PriceTag:
    """Преобразует OCR-текст в PriceTag.

    Стратегия «сверху вниз»:
    - Верхняя треть (y < 0.33): product_name — самый высокий текст
    - Средняя область: цены — ищем паттерны price_default / price_card
    - Нижняя треть (y > 0.67): barcode, id_sku, print_datetime, code
    """
    x_min, y_min, x_max, y_max = bbox
    crop_h = max(1, y_max - y_min)
    crop_w = max(1, x_max - x_min)

    # Строим список OCRBox
    boxes: list[OCRBox] = []
    for raw_box, text, conf in ocr_lines:
        if not text.strip():
            continue
        x0, y0, x1, y1 = _normalize_box(raw_box, crop_w, crop_h)
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

    # Сортируем по Y (сверху вниз)
    boxes.sort(key=lambda b: b.center_y)

    # --- Извлечение полей ---
    product_name = ""
    price_default = ""
    price_card = ""
    discount_amount = "нет"
    barcode = ""
    id_sku = ""
    print_datetime = ""
    code = "нет"
    additional_info = "нет"
    special_symbols = "нет"

    top_boxes = [b for b in boxes if b.center_y < 0.40]
    mid_boxes = [b for b in boxes if 0.20 <= b.center_y <= 0.75]
    bot_boxes = [b for b in boxes if b.center_y > 0.60]

    # Название: самый высокий бокс с наибольшей площадью в верхней трети
    if top_boxes:
        largest = max(top_boxes, key=lambda b: b.height * b.width * b.conf)
        product_name = largest.text

    # Цены — ищем паттерны в средней зоне
    prices_found: list[str] = []
    for b in mid_boxes:
        p = _extract_price(b.text)
        if p:
            prices_found.append(p)

    if len(prices_found) >= 2:
        # Предполагаем: первая (меньшая) = без карты, вторая (крупнее) = с картой
        prices_found_sorted = sorted(prices_found, key=lambda x: float(x.replace(",", ".")))
        price_card = prices_found_sorted[0]  # наименьшая = цена по карте
        price_default = prices_found_sorted[-1]  # наибольшая = без карты
    elif len(prices_found) == 1:
        price_card = prices_found[0]

    # Скидка
    for b in boxes:
        m = _DISCOUNT_RE.search(b.text)
        if m:
            discount_amount = m.group(0).strip()
            break

    # Штрихкод (нижняя треть, длинный числовой код)
    for b in bot_boxes:
        m = _BARCODE_RE.search(b.text)
        if m and len(m.group(0)) >= 10:
            barcode = m.group(0)
            break

    # Артикул SKU (нижняя треть, 6-9 цифр, не штрихкод)
    for b in bot_boxes:
        m = _SKU_RE.search(b.text)
        if m and len(m.group(0)) < 10:
            id_sku = m.group(0)
            break

    # Дата печати
    for b in bot_boxes:
        m = _DATE_RE.search(b.text)
        if m:
            print_datetime = m.group(0)
            break

    # Код зоны
    for b in bot_boxes:
        m = _CODE_RE.search(b.text)
        if m:
            code = m.group(0)
            break

    # Специальные символы К/Л/Ш
    for b in boxes:
        m = _SPECIAL_RE.search(b.text)
        if m:
            special_symbols = m.group(0).upper()
            break

    # additional_info — тексты, которые не попали ни в одну категорию
    used_texts = {product_name, barcode, id_sku, print_datetime, code}
    extra = [b.text for b in boxes if b.text not in used_texts and len(b.text) > 5]
    if extra:
        additional_info = " | ".join(extra[:3])

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
