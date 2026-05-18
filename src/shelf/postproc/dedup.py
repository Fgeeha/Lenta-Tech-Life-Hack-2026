"""Cross-track deduplication and field-level merging."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from shelf.schema import ABSENT_VALUE, PriceTag
from shelf.validation import normalize_ean13

_EMPTY = ("", ABSENT_VALUE, None)


def dedup_by_track(tagged: list[tuple[int, PriceTag]]) -> list[PriceTag]:
    """Backward-compatible helper: choose the most complete tag per track_id."""
    best: dict[int, PriceTag] = {}
    for track_id, tag in tagged:
        prev = best.get(track_id)
        if prev is None or tag_completeness(tag) > tag_completeness(prev):
            best[track_id] = tag
    return list(best.values())


def deduplicate_tags(
    tags: list[PriceTag],
    iou_threshold: float = 0.60,
    time_window_ms: float = 1500.0,
) -> list[PriceTag]:
    """Merge duplicate rows created by track ID switches.

    Primary keys are QR/barcode. If they are missing, merge only conservative
    near-duplicates: high IoU in a short time window, or same price pair + highly
    similar product name.
    """
    result: list[PriceTag] = []
    for tag in tags:
        merged = False
        for i, existing in enumerate(result):
            if _same_physical_tag(existing, tag, iou_threshold, time_window_ms):
                result[i] = merge_tags(existing, tag)
                merged = True
                break
        if not merged:
            result.append(tag)
    result.sort(
        key=lambda t: (
            str(t.filename),
            float(t.frame_timestamp or 0),
            float(t.y_min or 0),
            float(t.x_min or 0),
        )
    )
    return result


def tag_completeness(tag: PriceTag) -> float:
    """Score row usefulness; QR/barcode/prices are more important than coordinates."""
    weights = {
        "qr_code_barcode": 3.0,
        "barcode": 3.0,
        "price_card": 2.2,
        "price_default": 1.8,
        "product_name": 1.5,
        "id_sku": 1.2,
        "print_datetime": 1.0,
        "discount_amount": 0.8,
        "code": 0.5,
        "additional_info": 0.4,
        "special_symbols": 0.4,
    }
    score = 0.0
    data = tag.__dict__
    for field, weight in weights.items():
        val = data.get(field)
        if val not in _EMPTY:
            score += weight
    area = max(0, int(tag.x_max) - int(tag.x_min)) * max(
        0, int(tag.y_max) - int(tag.y_min)
    )
    score += min(1.0, area / 80_000.0)
    return score


def merge_tags(a: PriceTag, b: PriceTag) -> PriceTag:
    """Field-wise merge preferring recognized values and more complete row metadata."""
    primary, secondary = (
        (a, b) if tag_completeness(a) >= tag_completeness(b) else (b, a)
    )
    data = primary.__dict__.copy()
    for field, value in secondary.__dict__.items():
        if data.get(field) in _EMPTY and value not in _EMPTY:
            data[field] = value
    # Coordinates/timestamp from the more complete row are kept. If the secondary
    # bbox is bigger and primary has no OCR-sensitive fields, use the bigger box.
    return PriceTag(**data)


def _same_physical_tag(
    a: PriceTag, b: PriceTag, iou_threshold: float, time_window_ms: float
) -> bool:
    ka = _barcode_key(a)
    kb = _barcode_key(b)
    if ka and kb:
        return ka == kb

    # Same timestamp/nearby bbox duplicate from tiled detector or track switch.
    if (
        a.filename == b.filename
        and abs(float(a.frame_timestamp) - float(b.frame_timestamp))
        <= time_window_ms
    ):
        if _iou(a, b) >= iou_threshold:
            return True

    # Conservative OCR key: same prices and similar product names.
    prices_a = (str(a.price_card), str(a.price_default))
    prices_b = (str(b.price_card), str(b.price_default))
    if prices_a == prices_b and all(p not in _EMPTY for p in prices_a):
        na = _norm_text(a.product_name)
        nb = _norm_text(b.product_name)
        if na and nb and SequenceMatcher(None, na, nb).ratio() >= 0.72:
            return True

    return False


def _barcode_key(tag: PriceTag) -> str:
    for raw in (tag.qr_code_barcode, tag.barcode):
        digits = normalize_ean13(raw, allow_repair=False)
        if digits:
            return digits
    return ""


def _norm_text(text: str) -> str:
    text = re.sub(
        r"[^0-9a-zа-яё]+", " ", str(text).lower(), flags=re.IGNORECASE
    ).strip()
    return re.sub(r"\s+", " ", text)


def _iou(a: PriceTag, b: PriceTag) -> float:
    ax1, ay1, ax2, ay2 = (
        float(a.x_min),
        float(a.y_min),
        float(a.x_max),
        float(a.y_max),
    )
    bx1, by1, bx2, by2 = (
        float(b.x_min),
        float(b.y_min),
        float(b.x_max),
        float(b.y_max),
    )
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0
