"""Тесты QR-парсера URL."""

from shelf.qr.decoder import parse_qr_url

# --- Тесты parse_qr_url ---


def test_short_keys_basic():
    url = "https://lenta.ru/tag?b=4607030000770&p1=384.29&p4=294.99"
    result = parse_qr_url(url)
    assert result["qr_code_barcode"] == "4607030000770"
    assert result["price1_qr"] == "384.29"
    assert result["price4_qr"] == "294.99"


def test_long_keys():
    url = "https://x.ru?barcode=1234567890123&price1=100.00&price4=80.00"
    result = parse_qr_url(url)
    assert result["qr_code_barcode"] == "1234567890123"
    assert result["price1_qr"] == "100.00"
    assert result["price4_qr"] == "80.00"


def test_wholesale_keys_short():
    url = "https://x?wL1C=3&wL1P=299.99&wL2C=6&wL2P=249.99"
    result = parse_qr_url(url)
    assert result["wholesale_level_1_count"] == "3"
    assert result["wholesale_level_1_price"] == "299.99"
    assert result["wholesale_level_2_count"] == "6"
    assert result["wholesale_level_2_price"] == "249.99"


def test_action_keys():
    url = "https://x?aP=199.99&aC=PROMO2026"
    result = parse_qr_url(url)
    assert result["action_price_qr"] == "199.99"
    assert result["action_code_qr"] == "PROMO2026"


def test_url_without_scheme():
    url = "?b=9780123456789&p1=500.00"
    result = parse_qr_url(url)
    assert result.get("qr_code_barcode") == "9780123456789"


def test_bare_query_string():
    url = "b=9780123456789&p1=500.00"
    result = parse_qr_url(url)
    assert result.get("qr_code_barcode") == "9780123456789"


def test_empty_string_returns_empty():
    assert parse_qr_url("") == {}


def test_url_no_matching_keys():
    url = "https://example.com?foo=bar&baz=1"
    result = parse_qr_url(url)
    assert result == {}


def test_mixed_known_unknown_keys():
    url = "https://x?b=123&unknown_key=abc&p4=50.00"
    result = parse_qr_url(url)
    assert "qr_code_barcode" in result
    assert "price4_qr" in result
    assert len(result) == 2


def test_all_price_fields():
    url = "https://x?p1=100&p2=90&p3=80&p4=70"
    result = parse_qr_url(url)
    assert result["price1_qr"] == "100"
    assert result["price2_qr"] == "90"
    assert result["price3_qr"] == "80"
    assert result["price4_qr"] == "70"


def test_realistic_lenta_qr():
    """Пример реального формата QR-кода Ленты."""
    url = "https://lenta.ru/product?b=4670025474665&p1=252.63&p2=239.99&p4=129.99"
    result = parse_qr_url(url)
    assert result["qr_code_barcode"] == "4670025474665"
    assert result["price1_qr"] == "252.63"
    assert result["price2_qr"] == "239.99"
    assert result["price4_qr"] == "129.99"
    # p3 отсутствует → не должен быть в результате
    assert "price3_qr" not in result
