"""Тесты парсера полей ценника — регрессии на три известных бага."""

from shelf.ocr.parser import _extract_prices, _find_discount, parse_ocr_result

# --- Bug 1: "48%" не должно давать price=48 ---


def test_extract_prices_ignores_percent():
    texts = ["48%", "129", "252"]
    prices = _extract_prices(texts)
    assert 48.0 not in prices, "48% не должно считаться ценой"
    assert 129.0 in prices
    assert 252.0 in prices


def test_extract_prices_inline_percent():
    texts = ["-48%", "129.99", "252,63"]
    prices = _extract_prices(texts)
    assert 48.0 not in prices
    assert 129.99 in prices
    assert 252.63 in prices


# --- Bug 2: "48%" без минуса должно давать discount ---


def test_find_discount_with_minus():
    assert _find_discount(["-48%"]) == "-48%"


def test_find_discount_without_minus():
    assert _find_discount(["48%"]) == "-48%"


def test_find_discount_ocr_format():
    """Реальный OCR-вывод: '48%' в середине строки."""
    assert _find_discount(["Акция", "48%", "129"]) == "-48%"


def test_find_discount_absent():
    assert _find_discount(["129", "252", "Мёд"]) == "нет"


# --- Bug 3: price_card из реальной OCR-последовательности ---


def _fake_ocr(texts: list[str]) -> list[tuple[list, str, float]]:
    """Создать фиктивный OCR-результат без реальных bbox."""
    boxes = []
    for i, t in enumerate(texts):
        # Боксы расположены вертикально сверху вниз
        y0, y1 = i * 0.1, (i + 1) * 0.1
        box = [[0, y0], [1, y0], [1, y1], [0, y1]]
        boxes.append((box, t, 0.9))
    return boxes


def test_parse_real_ocr_sequence():
    """25_12-20 ts=0: OCR читает 48%, 129, 252 — price_card должен быть 129."""
    lines = _fake_ocr(["Hanos", "Csnictom", "252", "48%", "129"])
    tag = parse_ocr_result(lines, bbox=(0, 0, 200, 200))
    # После фикса Bug 1: 48 отфильтрован, остались [129, 252]
    assert tag.price_card != "48,00", "price_card не должен быть 48 (из '48%')"
    # discount_amount должен включать "48%"
    assert "48" in tag.discount_amount, f"discount_amount={tag.discount_amount}"


def test_parse_discount_sets_minus():
    lines = _fake_ocr(["129", "252", "23%"])
    tag = parse_ocr_result(lines, bbox=(0, 0, 200, 200))
    assert tag.discount_amount == "-23%"


def test_parse_no_percent_no_discount():
    lines = _fake_ocr(["299", "399", "Мёд"])
    tag = parse_ocr_result(lines, bbox=(0, 0, 200, 200))
    assert tag.discount_amount == "нет"


def test_sku_12_digits_not_misclassified_as_barcode():
    lines = _fake_ocr(["270108726573", "129", "252"])
    tag = parse_ocr_result(lines, bbox=(0, 0, 200, 200))
    assert tag.id_sku == "270108726573"
    assert tag.barcode == ""


def test_valid_ean13_barcode_is_extracted():
    lines = _fake_ocr(["4607124143901", "129", "252"])
    tag = parse_ocr_result(lines, bbox=(0, 0, 200, 200))
    assert tag.barcode == "4607124143901"


# --- Price extraction hardening ---


def test_extract_prices_handles_dash_cents():
    assert 129.99 in _extract_prices(["129-99"])


def test_extract_prices_handles_thousands_with_comma():
    assert 1299.99 in _extract_prices(["1 299,99"])


def test_extract_prices_ignores_small_item_count():
    assert _extract_prices(["от 2 шт"]) == []


def test_parse_prices_uses_card_and_default_context():
    lines = [
        (
            [[0.1, 0.62], [0.5, 0.62], [0.5, 0.70], [0.1, 0.70]],
            "цена без карты 252,63",
            0.95,
        ),
        (
            [[0.1, 0.78], [0.7, 0.78], [0.7, 0.92], [0.1, 0.92]],
            "по карте 129-99",
            0.95,
        ),
    ]
    tag = parse_ocr_result(lines, bbox=(0, 0, 200, 200))
    assert tag.price_card == "129,99"
    assert tag.price_default == "252,63"


def test_parse_recovers_split_rubles_and_kopecks_from_boxes():
    """OCR sometimes splits a price into two boxes: '129' + '99'."""
    import numpy as np

    crop = np.zeros((120, 220, 3), dtype=np.uint8)
    lines = [
        ([[30, 82], [120, 82], [120, 110], [30, 110]], "129", 0.96),
        ([[126, 88], [158, 88], [158, 105], [126, 105]], "99", 0.94),
    ]
    tag = parse_ocr_result(lines, crop=crop, bbox=(0, 0, 220, 120))
    assert tag.price_card == "129,99"
    assert tag.price_default == ""


def test_fix_digit_concat_corrects_ocr_concatenation():
    from shelf.ocr.parser import _fix_digit_concat

    # OCR merges "2631,57" + nearby "2" → "26312"; card=1899.99 → corrected to 2631.2
    corrected = _fix_digit_concat(26312.0, 1899.99)
    assert abs(corrected - 2631.2) < 0.01  # within 1.5 of GT 2631.57
    assert abs(corrected - 2631.57) < 1.5  # passes _field_match tolerance


def test_fix_digit_concat_leaves_legitimate_price_unchanged():
    from shelf.ocr.parser import _fix_digit_concat

    # Ratio 1.6× — not a concatenation artifact, must not divide
    assert _fix_digit_concat(3789.49, 2345.99) == 3789.49
    # Ratio < 5 — no correction needed
    assert _fix_digit_concat(500.0, 400.0) == 500.0
    # card_val=0 — no division by zero risk
    assert _fix_digit_concat(1000.0, 0.0) == 1000.0


def test_product_name_cleanup_removes_service_numbers_but_keeps_percent():
    from shelf.ocr.parser import _clean_product_name

    raw = "Молоко питьевое 3.2% 4607124143901 129,99 руб 03.04.2026 3:08"
    assert _clean_product_name(raw) == "Молоко питьевое 3.2%"
