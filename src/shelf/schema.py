"""Единый источник правды для схемы выходного CSV.

Семантика значений по ТЗ:
- ``"нет"`` — параметра нет на конкретном типе ценника;
- ``""`` — параметр есть, но не распознан.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Порядок столбцов строго по ТЗ.
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

ABSENT_VALUE = "нет"
UNREAD_VALUE = ""

# Историческая опечатка встречается в приложенных разметках. В выходе всегда пишем правильное имя.
COLUMN_ALIASES: dict[str, str] = {
    "wholesale_level_1_coun": "wholesale_level_1_count",
}

_NUMERIC_COLUMNS = {"frame_timestamp", "x_min", "y_min", "x_max", "y_max"}


@dataclass
class PriceTag:
    """Одна строка выходного CSV."""

    # --- из видео ---
    filename: str = UNREAD_VALUE
    frame_timestamp: float = 0.0  # миллисекунды от начала видео
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0

    # --- с ценника (текст) ---
    product_name: str = UNREAD_VALUE
    price_default: str = UNREAD_VALUE
    price_card: str = UNREAD_VALUE
    price_discount: str = ABSENT_VALUE
    barcode: str = UNREAD_VALUE
    discount_amount: str = ABSENT_VALUE
    id_sku: str = UNREAD_VALUE
    print_datetime: str = UNREAD_VALUE
    code: str = ABSENT_VALUE
    additional_info: str = ABSENT_VALUE
    color: str = UNREAD_VALUE
    special_symbols: str = ABSENT_VALUE

    # --- из QR ---
    qr_code_barcode: str = ABSENT_VALUE
    price1_qr: str = ABSENT_VALUE
    price2_qr: str = ABSENT_VALUE
    price3_qr: str = ABSENT_VALUE
    price4_qr: str = ABSENT_VALUE
    wholesale_level_1_count: str = ABSENT_VALUE
    wholesale_level_1_price: str = ABSENT_VALUE
    wholesale_level_2_count: str = ABSENT_VALUE
    wholesale_level_2_price: str = ABSENT_VALUE
    action_price_qr: str = ABSENT_VALUE
    action_code_qr: str = ABSENT_VALUE

    def to_dict(self) -> dict[str, Any]:
        """Вернуть строку CSV в порядке ``OUTPUT_COLUMNS`` без NaN/None."""
        raw = self.__dict__
        return {
            col: _clean_value(raw.get(col, UNREAD_VALUE), col)
            for col in OUTPUT_COLUMNS
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PriceTag":
        """Создать PriceTag из произвольного словаря, учитывая алиасы колонок."""
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            normalized[COLUMN_ALIASES.get(key, key)] = value
        kwargs = {
            col: _clean_value(normalized[col], col)
            for col in OUTPUT_COLUMNS
            if col in normalized
        }
        return cls(**kwargs)


def _clean_value(value: Any, column: str | None = None) -> Any:
    """Привести значение к безопасному для CSV виду.

    pandas часто превращает пустые ячейки в NaN/None; в итоговом CSV это должно быть
    пустой строкой, а не текстом ``nan``.
    """
    if value is None:
        return UNREAD_VALUE
    try:
        # float('nan') != float('nan')
        if value != value:  # noqa: PLR0124 - быстрый NaN-check без pandas
            return UNREAD_VALUE
    except Exception:
        pass
    if column in _NUMERIC_COLUMNS:
        return value
    return str(value).strip()


def validate_columns(columns: list[str]) -> None:
    """Проверить, что порядок колонок ровно соответствует ТЗ."""
    if columns != OUTPUT_COLUMNS:
        missing = [c for c in OUTPUT_COLUMNS if c not in columns]
        extra = [c for c in columns if c not in OUTPUT_COLUMNS]
        raise ValueError(
            f"Некорректная CSV-схема. Missing={missing}, extra={extra}"
        )
