"""Склейка данных OCR + QR (QR имеет приоритет)."""

from shelf.schema import PriceTag

# Поля, в которых QR доминирует над OCR
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
    "barcode",  # QR-barcode кросс-валидируется с OCR, но доверяем QR
}


def merge(ocr_tag: PriceTag, qr_fields: dict[str, str]) -> PriceTag:
    """Наложить QR-поля поверх OCR-тегов."""
    merged = ocr_tag.__dict__.copy()
    for field, value in qr_fields.items():
        if field in _QR_PRIORITY_FIELDS and value:
            merged[field] = value
    return PriceTag(**merged)
