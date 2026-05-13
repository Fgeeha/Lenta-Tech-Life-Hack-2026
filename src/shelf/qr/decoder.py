"""QR-декодер: pyzbar → opencv fallback → парсинг URL.

Стратегия:
1. pyzbar — быстрый, хорошо работает с прямыми QR
2. opencv QRCodeDetector — фолбэк
3. Парсинг URL: urllib.parse → маппинг коротких ключей → поля схемы
"""

import logging
from urllib.parse import parse_qs, urlparse

import numpy as np

logger = logging.getLogger(__name__)

# Маппинг QR-ключей (коротких и длинных) → имена полей OUTPUT_COLUMNS
_KEY_MAP: dict[str, str] = {
    # штрихкод
    "b": "qr_code_barcode",
    "barcode": "qr_code_barcode",
    # цены
    "p1": "price1_qr",
    "price1": "price1_qr",
    "p2": "price2_qr",
    "price2": "price2_qr",
    "p3": "price3_qr",
    "price3": "price3_qr",
    "p4": "price4_qr",
    "price4": "price4_qr",
    # оптовые пороги (уровень 1)
    "wL1C": "wholesale_level_1_count",
    "wholesaleLevel1Count": "wholesale_level_1_count",
    "wL1P": "wholesale_level_1_price",
    "wholesaleLevel1Price": "wholesale_level_1_price",
    # оптовые пороги (уровень 2)
    "wL2C": "wholesale_level_2_count",
    "wholesaleLevel2Count": "wholesale_level_2_count",
    "wL2P": "wholesale_level_2_price",
    "wholesaleLevel2Price": "wholesale_level_2_price",
    # акция
    "aP": "action_price_qr",
    "actionPrice": "action_price_qr",
    "aC": "action_code_qr",
    "actionCode": "action_code_qr",
}


def parse_qr_url(url: str) -> dict[str, str]:
    """Разобрать URL из QR-кода → словарь полей схемы.

    Поддерживает как полный URL (https://...?b=...&p1=...), так и просто
    строку с параметрами (?b=...&p1=...).
    """
    if not url:
        return {}
    # Если URL без схемы — добавим заглушку для правильного парсинга
    if not url.startswith(("http://", "https://")):
        if url.startswith("?"):
            url = "https://x" + url
        else:
            url = "https://x?" + url

    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query, keep_blank_values=False)
    except Exception:
        return {}

    result: dict[str, str] = {}
    for k, vals in qs.items():
        field = _KEY_MAP.get(k)
        if field and vals:
            result[field] = vals[0].strip()
    return result


def _try_pyzbar(image: np.ndarray) -> list[str]:
    """Попытка считать QR через pyzbar. Возвращает список декодированных строк."""
    try:
        from pyzbar import pyzbar

        decoded = pyzbar.decode(image)
        return [d.data.decode("utf-8", errors="ignore") for d in decoded if d.type in ("QRCODE", "EAN13", "EAN8")]
    except Exception as exc:
        logger.debug("pyzbar error: %s", exc)
        return []


def _try_opencv(image: np.ndarray) -> list[str]:
    """Фолбэк через OpenCV QRCodeDetector."""
    try:
        import cv2

        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(image)
        return [data] if data else []
    except Exception as exc:
        logger.debug("opencv QR error: %s", exc)
        return []


def decode_qr(crop: np.ndarray) -> dict[str, str]:
    """Считать QR из кропа ценника. Возвращает словарь полей (может быть пустым)."""
    if crop is None or crop.size == 0:
        return {}

    import cv2

    # Пробуем несколько разрешений — QR читается лучше при определённом масштабе
    results: dict[str, str] = {}
    for scale in [1.0, 2.0, 3.0, 0.5]:
        if scale != 1.0:
            h, w = crop.shape[:2]
            resized = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
        else:
            resized = crop

        # Также пробуем grayscale + OTSU для улучшения читаемости QR/штрихкода
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for img_variant in [resized, thresh_bgr]:
            for raw in _try_pyzbar(img_variant) + _try_opencv(img_variant):
                # EAN-13 / штрихкод: pyzbar возвращает голые цифры, не URL
                if raw.isdigit() and 8 <= len(raw) <= 15:
                    return {"qr_code_barcode": raw}
                parsed = parse_qr_url(raw)
                if parsed:
                    results.update(parsed)
                    return results  # нашли — выходим

    return results


def decode_barcode(crop: np.ndarray) -> str:
    """Считать штрихкод (EAN-13) из кропа. Возвращает строку или ''."""
    if crop is None or crop.size == 0:
        return ""
    try:
        from pyzbar import pyzbar

        decoded = pyzbar.decode(crop)
        for d in decoded:
            if d.type in ("EAN13", "EAN8", "CODE128"):
                return d.data.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return ""
