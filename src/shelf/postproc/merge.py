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

    return PriceTag(**data)


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
