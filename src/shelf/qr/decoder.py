"""QR-декодер: pyzbar → qreader → opencv fallback."""

import logging
from urllib.parse import parse_qs, urlparse

import numpy as np

logger = logging.getLogger(__name__)

# Маппинг коротких ключей QR → имена полей
_QR_KEY_MAP: dict[str, str] = {
    "b": "qr_code_barcode",
    "barcode": "qr_code_barcode",
    "p1": "price1_qr",
    "price1": "price1_qr",
    "p2": "price2_qr",
    "price2": "price2_qr",
    "p3": "price3_qr",
    "price3": "price3_qr",
    "p4": "price4_qr",
    "price4": "price4_qr",
    "wL1C": "wholesale_level_1_count",
    "wholesaleLevel1Count": "wholesale_level_1_count",
    "wL1P": "wholesale_level_1_price",
    "wholesaleLevel1Price": "wholesale_level_1_price",
    "wL2C": "wholesale_level_2_count",
    "wholesaleLevel2Count": "wholesale_level_2_count",
    "wL2P": "wholesale_level_2_price",
    "wholesaleLevel2Price": "wholesale_level_2_price",
    "aP": "action_price_qr",
    "actionPrice": "action_price_qr",
    "aC": "action_code_qr",
    "actionCode": "action_code_qr",
}


def _parse_qr_url(url: str) -> dict[str, str]:
    qs = parse_qs(urlparse(url).query)
    result: dict[str, str] = {}
    for k, vals in qs.items():
        mapped = _QR_KEY_MAP.get(k)
        if mapped:
            result[mapped] = vals[0]
    return result


def decode_qr(crop: np.ndarray) -> dict[str, str]:
    """Попытаться считать QR из кропа ценника. Возвращает словарь полей."""
    raw_url = _try_pyzbar(crop) or _try_opencv(crop)
    if raw_url:
        return _parse_qr_url(raw_url)
    return {}


def _try_pyzbar(crop: np.ndarray) -> str | None:
    try:
        from pyzbar import pyzbar

        decoded = pyzbar.decode(crop)
        for obj in decoded:
            if obj.type == "QRCODE":
                return obj.data.decode("utf-8", errors="ignore")
    except Exception:
        pass
    return None


def _try_opencv(crop: np.ndarray) -> str | None:
    try:
        import cv2

        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(crop)
        return data or None
    except Exception:
        pass
    return None
