"""Multi-frame field-level voting for several OCR/QR results of one track.

The tracker stores top-K crops per physical price tag.  Instead of selecting one
winner crop, this module merges field evidence across all candidates.  It keeps
source/confidence metadata internally but returns the fixed 29-field ``PriceTag``
for CSV compatibility.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

from shelf.schema import ABSENT_VALUE, OUTPUT_COLUMNS, PriceTag
from shelf.validation import normalize_ean13, normalize_sku

_EMPTY = {"", ABSENT_VALUE, None}
_PRICE_FIELDS_COMMA = {"price_default", "price_card", "price_discount"}
_PRICE_FIELDS_DOT = {
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
    "action_price_qr",
}
_DATE_RE = re.compile(
    r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})\s+(\d{1,2})[:.](\d{2})"
)
_CODE_RE = re.compile(r"\b\d{2}\s*_\s*\d{3,6}(?:\s*[-–]\s*\d{3,6})?\b")
_SYMBOL_RE = re.compile(r"^[КкKkЛлLlШш]$")
_NAME_BAD_RE = re.compile(
    r"(\d{2}[.\-/]\d{2}[.\-/]\d{2,4}|\d{8,}|\b\d+[,.]?\d*\s*(?:руб|₽|%|шт|кг|г)\b)",
    re.IGNORECASE,
)
_PRICE_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:[\s\u00a0]?\d{3})+|\d{1,6})(?:[,.\-](\d{1,2}))?(?!\d)"
)
_NAME_DATE_RE = re.compile(
    r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}(?:\s+\d{1,2}[:.]\d{2})?"
)
_NAME_LONG_DIGITS_RE = re.compile(r"\d{8,}")
_NAME_PRICE_RE = re.compile(
    r"(?<!\w)\d{1,5}(?:[,.\-]\d{1,2})?\s*(?:руб\.?|₽)(?!\w)", re.IGNORECASE
)
_NAME_CODE_RE = re.compile(r"\b\d{2}\s*_\s*\d{3,6}(?:\s*[-–]\s*\d{3,6})?\b")


@dataclass(frozen=True)
class FieldDecision:
    """Internal source/confidence selected for one output field."""

    value: str
    source: str
    confidence: float


@dataclass(frozen=True)
class VotingResult:
    """Result of voting plus internal source/confidence maps."""

    tag: PriceTag
    sources: dict[str, str]
    confidences: dict[str, float]


def vote_tags(
    tags: Sequence[PriceTag], candidate_scores: Sequence[float] | None = None
) -> VotingResult:
    """Merge several OCR/QR candidates for one track into a single ``PriceTag``.

    Empty values never overwrite recognized values.  Barcode and SKU fields are
    validated strictly to avoid the common 12-digit SKU → EAN-13 false positive.
    """
    if not tags:
        empty = PriceTag()
        return VotingResult(empty, {}, {})

    weights = _normalized_weights(candidate_scores, len(tags))
    base_idx = _best_base_index(tags, weights)
    data = tags[base_idx].__dict__.copy()
    sources: dict[str, str] = {}
    confidences: dict[str, float] = {}

    # Metadata from the highest-quality crop is kept.
    for field in (
        "filename",
        "frame_timestamp",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ):
        data[field] = getattr(tags[base_idx], field)
        sources[field] = "ocr_top1"
        confidences[field] = weights[base_idx]

    field_choosers = {
        "barcode": _choose_barcode,
        "qr_code_barcode": _choose_qr_barcode,
        "id_sku": _choose_sku,
        "product_name": _choose_product_name,
        "print_datetime": _choose_datetime,
        "code": _choose_code,
        "special_symbols": _choose_symbol,
        "color": _choose_color,
        "additional_info": _choose_long_text,
    }

    for field in OUTPUT_COLUMNS:
        if field in {
            "filename",
            "frame_timestamp",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        }:
            continue
        if field in field_choosers:
            decision = field_choosers[field](tags, weights, field)
        elif field in _PRICE_FIELDS_COMMA:
            decision = _choose_price(tags, weights, field, decimal_comma=True)
        elif field in _PRICE_FIELDS_DOT:
            decision = _choose_price(tags, weights, field, decimal_comma=False)
        elif field in {
            "wholesale_level_1_count",
            "wholesale_level_2_count",
            "action_code_qr",
        }:
            decision = _choose_simple(tags, weights, field)
        else:
            decision = _choose_simple(tags, weights, field)
        if decision is not None and _present(decision.value):
            data[field] = decision.value
            sources[field] = decision.source
            confidences[field] = decision.confidence
        elif not _present(data.get(field)):
            data[field] = getattr(PriceTag(), field)

    # Cross-field consistency: valid QR barcode is also a barcode, but do not
    # invent QR data from a linear barcode when QR is truly absent.
    if not _present(data.get("barcode")) and _present(
        data.get("qr_code_barcode")
    ):
        data["barcode"] = data["qr_code_barcode"]
        sources["barcode"] = "qr"
        confidences["barcode"] = max(
            confidences.get("barcode", 0.0),
            confidences.get("qr_code_barcode", 0.0),
        )

    _apply_price_consistency(data, sources, confidences)

    return VotingResult(PriceTag(**data), sources, confidences)


def merge_candidate_tags(
    tags: Sequence[PriceTag], candidate_scores: Sequence[float] | None = None
) -> PriceTag:
    """Convenience wrapper returning only the CSV-compatible tag."""
    return vote_tags(tags, candidate_scores).tag


def _normalized_weights(scores: Sequence[float] | None, n: int) -> list[float]:
    if not scores:
        return [1.0] * n
    vals = [float(s) if s is not None else 0.0 for s in list(scores)[:n]]
    if len(vals) < n:
        vals.extend([0.0] * (n - len(vals)))
    max_val = max(vals) if vals else 0.0
    if max_val <= 0:
        return [1.0] * n
    return [0.5 + 0.5 * (v / max_val) for v in vals]


def _best_base_index(tags: Sequence[PriceTag], weights: Sequence[float]) -> int:
    scored = []
    for idx, tag in enumerate(tags):
        filled = sum(1 for value in tag.__dict__.values() if _present(value))
        area = max(0, int(tag.x_max) - int(tag.x_min)) * max(
            0, int(tag.y_max) - int(tag.y_min)
        )
        scored.append((filled + min(1.0, area / 80000.0) + weights[idx], idx))
    return max(scored)[1]


def _present(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(
        text and text.lower() not in {"nan", "none"} and text != ABSENT_VALUE
    )


def _source_for(field: str, value: str, tag: PriceTag) -> str:
    if (
        field.startswith("qr_")
        or field.endswith("_qr")
        or field
        in {
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
        }
    ):
        return "qr"
    if field == "barcode" and normalize_ean13(value):
        if normalize_ean13(tag.qr_code_barcode) == normalize_ean13(value):
            return "qr"
        return "barcode"
    return "ocr_vote"


def _weighted_choice(
    candidates: Iterable[tuple[str, float, str]],
    *,
    prefer_longest: bool = False,
) -> FieldDecision | None:
    totals: dict[str, float] = defaultdict(float)
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    display: dict[str, str] = {}
    for value, weight, source in candidates:
        if not _present(value):
            continue
        key = _norm_key(value)
        if not key:
            continue
        totals[key] += weight
        sources[key][source] += 1
        if key not in display or (
            prefer_longest and len(value) > len(display[key])
        ):
            display[key] = value
    if not totals:
        return None
    best_key = max(
        totals,
        key=lambda k: (
            totals[k],
            len(display.get(k, "")) if prefer_longest else 0,
        ),
    )
    total_weight = sum(totals.values()) or 1.0
    return FieldDecision(
        value=display[best_key],
        source=sources[best_key].most_common(1)[0][0],
        confidence=min(1.0, totals[best_key] / total_weight),
    )


def _norm_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _choose_barcode(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    candidates: list[tuple[str, float, str]] = []
    for tag, weight in zip(tags, weights):
        for source_field, source_weight, source_name in (
            ("barcode", 1.0, "barcode"),
            ("qr_code_barcode", 1.25, "qr"),
        ):
            normalized = normalize_ean13(
                getattr(tag, source_field, ""), allow_repair=False
            )
            if normalized:
                candidates.append(
                    (normalized, weight * source_weight, source_name)
                )
    return _weighted_choice(candidates)


def _choose_qr_barcode(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    candidates: list[tuple[str, float, str]] = []
    for tag, weight in zip(tags, weights):
        normalized = normalize_ean13(tag.qr_code_barcode, allow_repair=False)
        if normalized:
            candidates.append((normalized, weight * 1.25, "qr"))
    return _weighted_choice(candidates)


def _choose_sku(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    candidates = []
    valid_barcodes = {normalize_ean13(t.barcode) for t in tags} | {
        normalize_ean13(t.qr_code_barcode) for t in tags
    }
    for tag, weight in zip(tags, weights):
        sku = normalize_sku(tag.id_sku)
        if sku and sku not in valid_barcodes:
            candidates.append((sku, weight, "regex"))
    return _weighted_choice(candidates)


def _normalize_price(raw: Any) -> float | None:
    text = (
        str(raw)
        .strip()
        .replace("\u00a0", " ")
        .replace(" ", "")
        .replace(",", ".")
    )
    if not text or text.lower() in {"nan", "none", ABSENT_VALUE}:
        return None
    match = _PRICE_RE.search(text)
    if not match:
        return None
    integer = re.sub(r"\D", "", match.group(1))
    if not integer or len(integer) > 6:
        return None
    frac = match.group(2) or "00"
    try:
        value = int(integer) + int((frac + "0")[:2]) / 100.0
    except ValueError:
        return None
    if 1.0 <= value <= 99999.99:
        return value
    return None


def _format_price(value: float, *, decimal_comma: bool) -> str:
    text = f"{value:.2f}"
    if not decimal_comma and text.endswith(".00"):
        return str(int(round(value)))
    return text.replace(".", ",") if decimal_comma else text


def _format_qr_price(raw: Any) -> str:
    value = _normalize_price(raw)
    return (
        _format_price(value, decimal_comma=False)
        if value is not None
        else str(raw).strip()
    )


def _format_ocr_price(raw: Any) -> str:
    value = _normalize_price(raw)
    return (
        _format_price(value, decimal_comma=True)
        if value is not None
        else str(raw).strip()
    )


def _apply_price_consistency(
    data: dict[str, Any], sources: dict[str, str], confidences: dict[str, float]
) -> None:
    """Keep OCR and QR price fields mutually consistent after voting."""
    if not _present(data.get("price_default")) and _present(
        data.get("price1_qr")
    ):
        data["price_default"] = _format_ocr_price(data["price1_qr"])
        sources["price_default"] = "qr"
        confidences["price_default"] = max(
            confidences.get("price1_qr", 0.0), 0.75
        )

    if not _present(data.get("price_card")):
        for qr_field in ("price4_qr", "action_price_qr", "price2_qr"):
            if _present(data.get(qr_field)):
                data["price_card"] = _format_ocr_price(data[qr_field])
                sources["price_card"] = "qr"
                confidences["price_card"] = max(
                    confidences.get(qr_field, 0.0), 0.75
                )
                break

    card = _normalize_price(data.get("price_card"))
    default = _normalize_price(data.get("price_default"))
    if card is not None and default is not None and card > default + 0.009:
        # In Lenta GT, price_card/action is not higher than default price. Swap
        # conservative OCR inversions produced by line-order mistakes.
        data["price_card"], data["price_default"] = (
            _format_price(default, decimal_comma=True),
            _format_price(card, decimal_comma=True),
        )
        sources["price_card"] = sources.get("price_card", "ocr_vote") + "+swap"
        sources["price_default"] = (
            sources.get("price_default", "ocr_vote") + "+swap"
        )

    if _present(data.get("price_default")) and not _present(
        data.get("price1_qr")
    ):
        data["price1_qr"] = _format_qr_price(data["price_default"])
        sources["price1_qr"] = "derived"
        confidences["price1_qr"] = min(
            0.70, confidences.get("price_default", 0.70)
        )
    if _present(data.get("price_card")) and not _present(data.get("price4_qr")):
        data["price4_qr"] = _format_qr_price(data["price_card"])
        sources["price4_qr"] = "derived"
        confidences["price4_qr"] = min(
            0.70, confidences.get("price_card", 0.70)
        )

    if not _present(data.get("discount_amount")):
        card = _normalize_price(data.get("price_card"))
        default = _normalize_price(data.get("price_default"))
        if card is not None and default is not None and default > card > 0:
            pct = int((1 - card / default) * 100)
            if 1 <= pct <= 99:
                data["discount_amount"] = f"-{pct}%"
                sources["discount_amount"] = "derived"
                confidences["discount_amount"] = 0.65


def _choose_price(
    tags: Sequence[PriceTag],
    weights: Sequence[float],
    field: str,
    *,
    decimal_comma: bool,
) -> FieldDecision | None:
    candidates: list[tuple[str, float, str]] = []
    for tag, weight in zip(tags, weights):
        value = _normalize_price(getattr(tag, field, ""))
        if value is None:
            continue
        source = _source_for(field, getattr(tag, field), tag)
        source_weight = 1.25 if source == "qr" else 1.0
        candidates.append(
            (
                _format_price(value, decimal_comma=decimal_comma),
                weight * source_weight,
                source,
            )
        )
    return _weighted_choice(candidates)


def _clean_name(text: str) -> str:
    """Clean a product-name candidate without destroying useful percents."""
    text = str(text or "").replace("\u00a0", " ")
    text = _NAME_DATE_RE.sub(" ", text)
    text = _NAME_CODE_RE.sub(" ", text)
    text = _NAME_LONG_DIGITS_RE.sub(" ", text)
    text = _NAME_PRICE_RE.sub(" ", text)
    text = re.sub(
        r"\b(?:qr|ean|barcode|штрих\s*код|артикул|id[_\s-]*sku|цена|карта|без\s+карты|по\s+карте)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" -|•\t\n")
    if len(text) < 3 or _NAME_BAD_RE.fullmatch(text):
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"\s{2,}|\|", text):
        part = part.strip(" -|•\t\n")
        if len(part) < 3 or _NAME_BAD_RE.fullmatch(part):
            continue
        letters = len(re.findall(r"[a-zа-яё]", part.lower()))
        digits = len(re.findall(r"\d", part))
        if letters < 2 or digits > max(8, letters * 2):
            continue
        key = re.sub(r"\s+", " ", part.lower())
        if key not in seen:
            seen.add(key)
            parts.append(part)
    return " ".join(parts)[:300]


def _name_score(name: str, weight: float) -> float:
    text = _clean_name(name)
    if not text:
        return 0.0
    lower = text.lower()
    letters = len(re.findall(r"[a-zа-яё]", lower))
    cyr = len(re.findall(r"[а-яё]", lower))
    digits = len(re.findall(r"\d", text))
    tokens = len(re.findall(r"[a-zа-яё0-9]+", lower))
    numeric_noise = digits / max(1, letters + digits)
    cyr_ratio = cyr / max(1, letters)
    penalty = max(0.25, 1.0 - numeric_noise)
    return (letters + cyr_ratio * 8.0 + 3.0 * tokens) * weight * penalty


def select_product_name_candidate(
    candidates: Sequence[str], weights: Sequence[float] | None = None
) -> str:
    """Select the cleanest product-name candidate from OCR/catalog sources."""
    if not candidates:
        return ""
    w = list(weights or [1.0] * len(candidates))
    if len(w) < len(candidates):
        w.extend([1.0] * (len(candidates) - len(w)))
    best: tuple[float, str] | None = None
    for candidate, weight in zip(candidates, w):
        name = _clean_name(candidate)
        if not name:
            continue
        score = _name_score(name, float(weight))
        if (
            best is None
            or score > best[0]
            or (abs(score - best[0]) < 1e-6 and len(name) > len(best[1]))
        ):
            best = (score, name)
    return best[1] if best else ""


def _choose_product_name(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    candidates = [str(t.product_name or "") for t in tags]
    selected = select_product_name_candidate(candidates, weights)
    if not selected:
        return None
    best_score = _name_score(selected, 1.0)
    total_score = max(
        1.0,
        sum(
            _name_score(name, weight)
            for name, weight in zip(candidates, weights)
        ),
    )
    return FieldDecision(
        selected, "ocr_vote", min(1.0, best_score / total_score)
    )


def _choose_datetime(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    candidates: list[tuple[str, float, str]] = []
    for tag, weight in zip(tags, weights):
        value = _normalize_datetime(tag.print_datetime)
        if value:
            candidates.append((value, weight, "regex"))
    return _weighted_choice(candidates)


def _normalize_datetime(raw: Any) -> str:
    match = _DATE_RE.search(str(raw or ""))
    if not match:
        return ""
    day, month, year, hour, minute = match.groups()
    if len(year) == 2:
        year = "20" + year
    try:
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
    except ValueError:
        return ""
    return f"{dt.day:02d}.{dt.month:02d}.{dt.year} {dt.hour}:{dt.minute:02d}"


def _choose_code(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    candidates: list[tuple[str, float, str]] = []
    for tag, weight in zip(tags, weights):
        match = _CODE_RE.search(str(tag.code or ""))
        if match:
            value = re.sub(r"\s+", "", match.group(0)).replace("–", "-")
            candidates.append((value, weight, "regex"))
    return _weighted_choice(candidates)


def _choose_symbol(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    candidates: list[tuple[str, float, str]] = []
    for tag, weight in zip(tags, weights):
        value = str(tag.special_symbols or "").strip()
        if _SYMBOL_RE.fullmatch(value):
            value = value.upper().replace("K", "К").replace("L", "Л")
            candidates.append((value, weight, "regex"))
    return _weighted_choice(candidates)


def _choose_color(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    return _choose_simple(tags, weights, field)


def _choose_long_text(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    candidates = []
    for tag, weight in zip(tags, weights):
        value = str(getattr(tag, field, "") or "").strip()
        if not _present(value) or _NAME_BAD_RE.fullmatch(value):
            continue
        candidates.append(
            (value, weight * max(1.0, len(value) / 20.0), "ocr_vote")
        )
    return _weighted_choice(candidates, prefer_longest=True)


def _choose_simple(
    tags: Sequence[PriceTag], weights: Sequence[float], field: str
) -> FieldDecision | None:
    candidates = []
    for tag, weight in zip(tags, weights):
        value = str(getattr(tag, field, "") or "").strip()
        if _present(value):
            candidates.append((value, weight, _source_for(field, value, tag)))
    return _weighted_choice(candidates)


def values_similar(a: str, b: str) -> float:
    """Expose lightweight text similarity for tests/debugging."""
    return SequenceMatcher(None, _norm_key(a), _norm_key(b)).ratio()
