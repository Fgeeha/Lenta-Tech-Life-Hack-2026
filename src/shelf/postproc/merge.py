"""Склейка данных OCR + QR.

Приоритеты:
- QR-поля (barcode, цены) > OCR (QR надёжнее, не искажён перспективой)
- OCR-поле > пустая строка (если QR не дал данных — ставим OCR)
- barcode: если QR-barcode есть → копируем в barcode (кросс-валидация)
"""

from shelf.schema import PriceTag

_QR_PRIORITY_FIELDS = {
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
}


def merge(ocr_tag: PriceTag, qr_fields: dict[str, str]) -> PriceTag:
    """Наложить QR-поля поверх OCR-результата."""
    data = ocr_tag.__dict__.copy()
    for field, value in qr_fields.items():
        if value and value.strip():
            if field in _QR_PRIORITY_FIELDS:
                data[field] = value
            elif not data.get(field):
                data[field] = value

    # Если QR дал barcode → используем как основной barcode (если OCR не дал)
    qr_bc = data.get("qr_code_barcode", "нет")
    if qr_bc and qr_bc != "нет" and not data.get("barcode"):
        data["barcode"] = _normalize_barcode(qr_bc)

    # Lenta-специфичная структура: если QR не прочитан, выводим поля из OCR.
    # Статистика GT: price4_qr==price_card (96%), price1_qr==price_default (97%),
    # qr_code_barcode==barcode (98%) — устойчивые соответствия по бизнес-логике.
    _empty = ("", "нет")
    price_card = data.get("price_card", "")
    price_default = data.get("price_default", "")
    barcode = data.get("barcode", "")

    if data.get("price4_qr", "") in _empty and price_card not in _empty:
        data["price4_qr"] = price_card
    if data.get("price1_qr", "") in _empty and price_default not in _empty:
        data["price1_qr"] = price_default
    if data.get("qr_code_barcode", "") in _empty and barcode not in _empty:
        data["qr_code_barcode"] = barcode

    # Деривация discount_amount из двух цен.
    # Lenta GT всегда хранит скидку как "-NN%" (int floor, не round).
    # Формула верифицирована по всем трём GT-видео.
    if data.get("discount_amount", "") in _empty:
        pc = _safe_float(price_card)
        pd = _safe_float(price_default)
        if pc and pd and pd > pc > 0:
            pct = int((1 - pc / pd) * 100)
            if 1 <= pct <= 99:
                data["discount_amount"] = f"-{pct}%"

    return PriceTag(**data)


def _safe_float(val: str) -> float | None:
    try:
        return float(str(val).replace(",", "."))
    except (ValueError, TypeError):
        return None


def _normalize_barcode(raw: str) -> str:
    """Нормализовать штрихкод: убрать .0, lpad до 13 цифр."""
    try:
        raw = raw.strip()
        if "." in raw:
            raw = str(int(float(raw)))
        raw = raw.replace(" ", "")
        if raw.isdigit() and len(raw) < 13:
            raw = raw.zfill(13)
        return raw
    except (ValueError, OverflowError):
        return raw
