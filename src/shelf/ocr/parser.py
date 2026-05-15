"""Mapping OCR boxes to price-tag fields.

The parser is deliberately conservative: it fills a field only when there is a
strong pattern/validation signal. A wrong value is more harmful than an empty
value because empty means "present but not recognized" by the task rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np

from shelf.qr.barcode_roi import (
    ean13_checksum_valid,
    ean13_repair,
    read_barcode_from_strip,
)
from shelf.schema import ABSENT_VALUE, PriceTag

if TYPE_CHECKING:
    from shelf.ocr.engine import OCREngine
# --- Regexes -----------------------------------------------------------------

# Prices such as 129, 129.99, 1 299,99. Percent tokens are removed before matching.
_PRICE_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:[\s\u00a0]?\d{3})+|\d{1,6})(?:[.,](\d{1,2}))?(?!\d)"
)
_DISCOUNT_PCT_RE = re.compile(r"[-–−]?\s*(\d{1,2})\s*%")
_DISCOUNT_RUB_RE = re.compile(
    r"(?:[-–−]\s*)?(\d{1,5})\s*(?:р|руб\.?|₽)", re.IGNORECASE
)
_DATE_RE = re.compile(
    r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})\s+(\d{1,2})[:.](\d{2})"
)
_BARCODE_DIGITS_RE = re.compile(r"(?<!\d)(\d[\d\s\u00a0]{10,18}\d)(?!\d)")
_SKU_RE = re.compile(r"(?<!\d)(\d[\d\s\u00a0]{8,16}\d)(?!\d)")
_SPECIAL_RE = re.compile(r"^\s*([КкKkЛлLlШш])\s*$")
_CODE_RE = re.compile(r"\b\d{2}\s*_\s*\d{3,6}(?:\s*[-–]\s*\d{3,6})?\b")
_PCT_TOKEN_RE = re.compile(r"[-–−]?\s*\d{1,3}\s*%")
_CYR_RE = re.compile(r"[а-яА-ЯёЁ]")
_NON_NAME_RE = re.compile(
    r"(\d{2}[.\-/]\d{2}[.\-/]\d{2,4}|\d{8,}|\b\d+[,.]?\d*\s*(?:руб|₽|%|шт|кг|г)\b)",
    re.IGNORECASE,
)

_NUMERIC_OCR_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "О": "0",
        "о": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "З": "3",
        "з": "3",
    }
)


@dataclass
class OCRBox:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


@dataclass
class PriceCandidate:
    value: float
    text: str
    score: float
    box: OCRBox | None = None


# --- Low-level normalizers ----------------------------------------------------


def _numeric_text(text: str) -> str:
    """Fix common OCR substitutions only for numeric-looking strings."""
    if not text:
        return ""
    # Apply translation when the token already contains digits or price-like separators.
    if re.search(r"\d|[.,:%₽]", text):
        return text.translate(_NUMERIC_OCR_TRANSLATION)
    return text


def _strip_percent_tokens(text: str) -> str:
    """Remove discount tokens before price extraction."""
    return _PCT_TOKEN_RE.sub(" ", _numeric_text(text))


def _looks_like_date_or_code(text: str) -> bool:
    t = _numeric_text(text)
    return bool(_DATE_RE.search(t) or _CODE_RE.search(t))


def _price_from_match(match: re.Match[str]) -> float | None:
    raw_int = re.sub(r"[\s\u00a0]", "", match.group(1))
    if len(raw_int) > 6:
        return None
    integer = int(raw_int)
    frac_raw = match.group(2)
    if frac_raw is None:
        frac = 0
    else:
        frac = int((frac_raw + "0")[:2])
    val = integer + frac / 100.0
    if 1.0 <= val <= 99_999.99:
        return val
    return None


def _fmt_price(val: float, decimal_comma: bool = True) -> str:
    formatted = f"{val:.2f}"
    return formatted.replace(".", ",") if decimal_comma else formatted


def _normalize_price_string(raw: str, decimal_comma: bool = False) -> str:
    """Normalize a price-like value from QR/OCR."""
    prices = _extract_prices([raw])
    if not prices:
        return str(raw).strip()
    return _fmt_price(prices[0], decimal_comma=decimal_comma)


# --- Public helper functions used in tests -----------------------------------


def _extract_prices(texts: Iterable[str]) -> list[float]:
    """Extract price-like numbers, excluding percents, dates, codes and long IDs."""
    prices: list[float] = []
    for text in texts:
        if not text:
            continue
        if _looks_like_date_or_code(text):
            continue
        clean = _strip_percent_tokens(text)
        for m in _PRICE_RE.finditer(clean):
            val = _price_from_match(m)
            if val is None:
                continue
            # Do not treat obvious item counts as prices: "2 шт", "1 кг" etc.
            tail = clean[m.end() : m.end() + 5].lower()
            if re.match(r"\s*(шт|кг|г|л)\b", tail):
                continue
            prices.append(val)
    return sorted(prices)


def _find_discount(texts: Iterable[str]) -> str:
    """Find discount amount as '-NN%' or '-NNр'."""
    for text in texts:
        t = _numeric_text(text)
        m = _DISCOUNT_PCT_RE.search(t)
        if m:
            pct = int(m.group(1))
            if 1 <= pct <= 99:
                return f"-{pct}%"
    for text in texts:
        t = _numeric_text(text)
        if "-" not in t and "−" not in t and "скид" not in t.lower():
            continue
        m = _DISCOUNT_RUB_RE.search(t)
        if m:
            rub = int(m.group(1))
            if 1 <= rub <= 99_999:
                return f"-{rub}р"
    return ABSENT_VALUE


# --- Geometry / zone helpers --------------------------------------------------


def _find_orange_rows(img: np.ndarray) -> tuple[int, int]:
    """Find rows with orange/yellow/red price background."""
    if img is None or img.size == 0:
        return 0, img.shape[0] if img is not None else 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_orange = cv2.inRange(
        hsv, np.array([5, 45, 70]), np.array([45, 255, 255])
    )
    mask_red1 = cv2.inRange(
        hsv, np.array([0, 45, 70]), np.array([12, 255, 255])
    )
    mask_red2 = cv2.inRange(
        hsv, np.array([160, 45, 70]), np.array([180, 255, 255])
    )
    mask = mask_orange | mask_red1 | mask_red2
    row_sums = mask.sum(axis=1) / 255.0
    threshold = img.shape[1] * 0.12
    rows = np.where(row_sums > threshold)[0]
    if len(rows) == 0:
        return int(img.shape[0] * 0.55), img.shape[0]
    return max(0, int(rows[0]) - 10), min(img.shape[0], int(rows[-1]) + 10)


def _normalize_box(
    box: list, w: int, h: int
) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return (
        min(xs) / max(1, w),
        min(ys) / max(1, h),
        max(xs) / max(1, w),
        max(ys) / max(1, h),
    )


def _extract_price_candidates(boxes: list[OCRBox]) -> list[PriceCandidate]:
    candidates: list[PriceCandidate] = []
    for b in boxes:
        vals = _extract_prices([b.text])
        for val in vals:
            candidates.append(
                PriceCandidate(
                    value=val,
                    text=b.text,
                    score=max(1e-6, b.area) * max(0.05, b.conf),
                    box=b,
                )
            )
    return candidates


def _choose_prices(
    price_candidates: list[PriceCandidate], all_prices: list[float]
) -> tuple[str, str]:
    """Return (price_card, price_default)."""
    if price_candidates:
        # If one candidate is much larger on the image, it is usually the card/action price.
        ordered_by_score = sorted(
            price_candidates, key=lambda c: c.score, reverse=True
        )
        values = sorted({round(c.value, 2) for c in price_candidates})
        if len(values) >= 2:
            if ordered_by_score[0].score > ordered_by_score[1].score * 1.35:
                card_val = ordered_by_score[0].value
            else:
                card_val = min(values)
            default_candidates = [v for v in values if v > card_val + 0.009]
            default_val = (
                max(default_candidates) if default_candidates else max(values)
            )
            return _fmt_price(card_val), _fmt_price(default_val)
        return _fmt_price(ordered_by_score[0].value), ""

    if not all_prices:
        return "", ""
    values = sorted({round(v, 2) for v in all_prices})
    if len(values) >= 2:
        return _fmt_price(values[0]), _fmt_price(values[-1])
    return _fmt_price(values[0]), ""


# --- Field extractors ---------------------------------------------------------


def _preprocess_name_zone(crop_raw: np.ndarray, scale: int = 3) -> np.ndarray:
    img = cv2.rotate(crop_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if scale > 1:
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4
        )
    return img


def _clean_product_name(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -|•\t\n")
    parts: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"\s{2,}|\|", text):
        part = part.strip()
        if len(part) < 2 or _NON_NAME_RE.search(part):
            continue
        key = part.lower()
        if key not in seen:
            seen.add(key)
            parts.append(part)
    cleaned = " ".join(parts).strip()
    return cleaned[:300]


def _extract_name_ru(
    crop_raw: np.ndarray, ocr_ru: OCREngine, name_end_frac: float
) -> str:
    proc = _preprocess_name_zone(crop_raw)
    H = proc.shape[0]
    safe_frac = max(0.30, min(name_end_frac, 0.75))
    zone_h = int(H * safe_frac)
    zone = proc[:zone_h, :]
    if zone.size == 0 or zone.shape[0] < 10:
        return ""
    lines = ocr_ru.run(zone)
    scored: list[tuple[float, float, str]] = []
    for box_pts, text, conf in lines:
        text = text.strip()
        if conf < 0.23 or len(text) < 2:
            continue
        if not (_CYR_RE.search(text) or len(text) > 5):
            continue
        if _NON_NAME_RE.search(text):
            continue
        xs = [p[0] for p in box_pts]
        ys = [p[1] for p in box_pts]
        area = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        scored.append((min(1.0, min(ys) / max(1, H)), area * conf, text))
    if not scored:
        return ""
    # Preserve reading order: top-to-bottom, then keep reasonably high-scored fragments.
    scored.sort(key=lambda x: (x[0], -x[1]))
    selected = [t for _, _, t in scored[:8]]
    return _clean_product_name(" ".join(selected))


def _extract_barcode(
    texts: Iterable[str], proc_crop: np.ndarray | None = None
) -> str:
    if proc_crop is not None and proc_crop.size > 0:
        decoded = read_barcode_from_strip(proc_crop)
        if decoded:
            return decoded

    # Text OCR: accept only validated 13-digit EANs or 14-digit strings where dropping
    # one digit validates. Do not append a checksum to arbitrary 12-digit SKU values.
    for text in texts:
        t = _numeric_text(text)
        for m in _BARCODE_DIGITS_RE.finditer(t):
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) == 13 and ean13_checksum_valid(digits):
                return digits
            if len(digits) == 14:
                repaired = ean13_repair(digits)
                if repaired:
                    return repaired
    return ""


def _extract_sku(texts: Iterable[str], barcode: str = "") -> str:
    for text in texts:
        t = _numeric_text(text)
        for m in _SKU_RE.finditer(t):
            digits = re.sub(r"\D", "", m.group(1))
            if (
                10 <= len(digits) <= 12
                and digits != barcode
                and digits not in barcode
            ):
                return digits
    return ""


def _extract_datetime(texts: Iterable[str]) -> str:
    combined = " ".join(_numeric_text(t) for t in texts)
    m = _DATE_RE.search(combined)
    if not m:
        return ""
    day, month, year, hour, minute = m.groups()
    if len(year) == 2:
        year = "20" + year
    try:
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
        # Match GT style: one/two digit hour allowed, no leading zero required.
        return (
            f"{dt.day:02d}.{dt.month:02d}.{dt.year} {dt.hour}:{dt.minute:02d}"
        )
    except ValueError:
        return m.group(0)


def _extract_code(texts: Iterable[str]) -> str:
    for text in texts:
        m = _CODE_RE.search(_numeric_text(text))
        if m:
            return re.sub(r"\s+", "", m.group(0)).replace("–", "-")
    return ABSENT_VALUE


def _extract_special_symbol(boxes: list[OCRBox]) -> str:
    # Only trust isolated short boxes; do not take letters from product names.
    for b in boxes:
        if b.conf < 0.35 or len(b.text.strip()) > 2:
            continue
        m = _SPECIAL_RE.match(b.text)
        if m:
            s = m.group(1).upper().replace("K", "К").replace("L", "Л")
            return s
    return ABSENT_VALUE


def _box_text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _extract_additional_info(
    boxes: list[OCRBox], used_values: set[str], product_name: str
) -> str:
    extras: list[str] = []
    for b in boxes:
        text = b.text.strip()
        if len(text) < 5 or b.conf < 0.45:
            continue
        digits = re.sub(r"\D", "", text)
        if text in used_values or digits in used_values:
            continue
        if product_name and _box_text_similarity(text, product_name) > 0.65:
            continue
        if _NON_NAME_RE.search(text):
            continue
        extras.append(text)
    return " | ".join(extras[:3]) if extras else ABSENT_VALUE


# --- Main parser --------------------------------------------------------------


def parse_ocr_result(
    ocr_lines: list[tuple[list, str, float]],
    crop: "np.ndarray | None" = None,
    filename: str = "",
    frame_timestamp: float = 0.0,
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0),
    color: str = "red",
    ocr_ru: OCREngine | None = None,
    crop_raw: np.ndarray | None = None,
) -> PriceTag:
    """Extract price-tag fields from OCR results."""
    x_min, y_min, x_max, y_max = bbox
    crop_h = max(1, y_max - y_min)
    crop_w = max(1, x_max - x_min)

    boxes: list[OCRBox] = []
    for raw_box, text, conf in ocr_lines:
        text = (text or "").strip()
        if not text or conf < 0.25:
            continue
        h_img = crop.shape[0] if crop is not None else crop_h
        w_img = crop.shape[1] if crop is not None else crop_w
        x0, y0, x1, y1 = _normalize_box(raw_box, w_img, h_img)
        boxes.append(
            OCRBox(text=text, conf=float(conf), x0=x0, y0=y0, x1=x1, y1=y1)
        )

    # Even when OCR returns no boxes, pyzbar may still read the barcode from crop.
    all_texts = [b.text for b in boxes]
    barcode = _extract_barcode(all_texts, crop)

    if not boxes:
        return PriceTag(
            filename=filename,
            frame_timestamp=frame_timestamp,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            color=color,
            barcode=barcode,
        )

    boxes.sort(key=lambda b: (b.center_y, b.center_x))
    all_texts = [b.text for b in boxes]

    price_zone_start = 0.5
    if crop is not None:
        oy0, _ = _find_orange_rows(crop)
        price_zone_start = oy0 / max(1, crop.shape[0])

    price_boxes = [b for b in boxes if b.center_y >= price_zone_start]
    info_boxes = [b for b in boxes if b.center_y < price_zone_start]

    discount_amount = _find_discount(all_texts)

    price_candidates = _extract_price_candidates(price_boxes)
    all_prices = _extract_prices(all_texts)
    price_card, price_default = _choose_prices(price_candidates, all_prices)

    product_name = ""
    if ocr_ru is not None and crop_raw is not None:
        product_name = _extract_name_ru(crop_raw, ocr_ru, price_zone_start)
    elif ocr_ru is not None and crop is not None:
        product_name = _extract_name_ru(crop, ocr_ru, price_zone_start)

    if not product_name and info_boxes:
        name_candidates = [
            b.text
            for b in info_boxes
            if len(b.text) > 3
            and b.conf > 0.45
            and not _NON_NAME_RE.search(b.text)
        ]
        product_name = _clean_product_name(" ".join(name_candidates[:6]))

    if not barcode:
        barcode = _extract_barcode(all_texts, crop)

    id_sku = _extract_sku(all_texts, barcode)
    print_datetime = _extract_datetime(all_texts)
    code = _extract_code(all_texts)
    special_symbols = _extract_special_symbol(boxes)

    used = {barcode, id_sku, print_datetime, price_card, price_default}
    additional_info = _extract_additional_info(boxes, used, product_name)

    return PriceTag(
        filename=filename,
        frame_timestamp=frame_timestamp,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        product_name=product_name,
        price_default=price_default,
        price_card=price_card,
        price_discount=ABSENT_VALUE,
        barcode=barcode,
        discount_amount=discount_amount,
        id_sku=id_sku,
        print_datetime=print_datetime,
        code=code,
        additional_info=additional_info,
        color=color,
        special_symbols=special_symbols,
    )
