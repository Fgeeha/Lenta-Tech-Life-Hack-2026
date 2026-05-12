"""Единый источник правды для схемы выходного CSV."""

from dataclasses import dataclass

# Порядок столбцов строго по ТЗ §2
OUTPUT_COLUMNS: list[str] = [
    # --- поля с ценника ---
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    # --- поля из QR ---
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
]

# «нет» — поле отсутствует на этом типе ценника
# ""     — поле есть, но не распозналось
_ABSENT = "нет"
_UNREAD = ""


@dataclass
class PriceTag:
    """Одна строка выходного CSV."""

    # --- из видео ---
    filename: str = _UNREAD
    frame_timestamp: float = 0.0
    x_min: int = 0
    y_min: int = 0
    x_max: int = 0
    y_max: int = 0

    # --- с ценника (текст) ---
    product_name: str = _UNREAD
    price_default: str = _UNREAD
    price_card: str = _UNREAD
    price_discount: str = _ABSENT
    barcode: str = _UNREAD
    discount_amount: str = _ABSENT
    id_sku: str = _UNREAD
    print_datetime: str = _UNREAD
    code: str = _UNREAD
    additional_info: str = _ABSENT
    color: str = _UNREAD
    special_symbols: str = _ABSENT

    # --- из QR ---
    qr_code_barcode: str = _ABSENT
    price1_qr: str = _ABSENT
    price2_qr: str = _ABSENT
    price3_qr: str = _ABSENT
    price4_qr: str = _ABSENT
    wholesale_level_1_count: str = _ABSENT
    wholesale_level_1_price: str = _ABSENT
    wholesale_level_2_count: str = _ABSENT
    wholesale_level_2_price: str = _ABSENT
    action_price_qr: str = _ABSENT
    action_code_qr: str = _ABSENT

    def to_dict(self) -> dict:
        """Вернуть строку CSV в порядке OUTPUT_COLUMNS."""
        raw = self.__dict__
        return {col: raw[col] for col in OUTPUT_COLUMNS}
