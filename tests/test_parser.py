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
