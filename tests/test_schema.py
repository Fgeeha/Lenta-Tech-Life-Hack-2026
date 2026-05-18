"""Проверка корректности OUTPUT_COLUMNS и PriceTag."""

from shelf.schema import OUTPUT_COLUMNS, PriceTag

EXPECTED_COLUMNS = [
    # из ценника
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
    # из QR
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


def test_output_columns_exact_set():
    assert set(OUTPUT_COLUMNS) == set(EXPECTED_COLUMNS), (
        f"Лишние: {set(OUTPUT_COLUMNS) - set(EXPECTED_COLUMNS)}, "
        f"Пропущены: {set(EXPECTED_COLUMNS) - set(OUTPUT_COLUMNS)}"
    )


def test_output_columns_no_duplicates():
    assert len(OUTPUT_COLUMNS) == len(set(OUTPUT_COLUMNS)), (
        "Есть дублирующиеся столбцы"
    )


def test_output_columns_order():
    assert OUTPUT_COLUMNS == EXPECTED_COLUMNS, (
        "Порядок столбцов не совпадает с ТЗ §2"
    )


def test_price_tag_to_dict_keys():
    tag = PriceTag(filename="test.mp4")
    d = tag.to_dict()
    assert list(d.keys()) == OUTPUT_COLUMNS


def test_price_tag_to_dict_no_extra_keys():
    tag = PriceTag()
    d = tag.to_dict()
    assert len(d) == len(OUTPUT_COLUMNS)
