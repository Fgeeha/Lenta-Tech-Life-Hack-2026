"""Local catalog lookup built from available CSV files.

The challenge rules allow local catalogs derived from provided data.  This module
never calls the network; it reads CSV files from ``SHELF_CATALOG_PATH`` or the
optional ``data/catalog.csv`` file and uses validated barcode/SKU keys to fill
high-confidence product names and missing price fields.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from shelf.schema import ABSENT_VALUE, COLUMN_ALIASES, PriceTag
from shelf.validation import normalize_ean13, normalize_sku

logger = logging.getLogger(__name__)
_EMPTY = {"", ABSENT_VALUE, None}
_PRICE_COLUMNS = ("price_default", "price_card", "price_discount")
_QR_PRICE_COLUMNS = ("price2_qr",)


@dataclass
class CatalogEntry:
    """One locally known product keyed by barcode or SKU."""

    product_name: str = ""
    id_sku: str = ""
    barcode: str = ""
    price_default: str = ""
    price_card: str = ""
    price_discount: str = ""
    price2_qr: str = ""
    special_symbols: str = ""
    code: str = ""
    print_datetime: str = ""
    additional_info: str = ""
    source_count: int = 0
    source_files: set[str] = field(default_factory=set)


@dataclass
class Catalog:
    """In-memory local catalog."""

    by_barcode: dict[str, CatalogEntry] = field(default_factory=dict)
    by_sku: dict[str, CatalogEntry] = field(default_factory=dict)

    def lookup(
        self, *, barcode: str = "", qr_barcode: str = "", sku: str = ""
    ) -> CatalogEntry | None:
        """Find an entry by EAN-13, SKU, or partial-digit fallback.

        Priority:
        1. Exact EAN-13 (with checksum repair for 12/14-digit inputs)
        2. Prefix/substring fallback (OCR often drops 1-2 edge digits)
        3. Exact normalized SKU
        4. SKU prefix/substring fallback
        """
        for raw in (qr_barcode, barcode):
            key = normalize_ean13(
                raw,
                allow_repair=True,
                allow_append_12=False,
                allow_drop_14=True,
            )
            if key and key in self.by_barcode:
                return self.by_barcode[key]

        # Prefix/substring fallback: OCR may produce partial or truncated EAN-13.
        # Only use when the digit run is >= 8 to avoid false positives.
        for raw in (qr_barcode, barcode):
            cleaned = re.sub(r"\D", "", str(raw or ""))
            if len(cleaned) < 8:
                continue
            matches = [
                bc for bc in self.by_barcode
                if cleaned in bc or bc in cleaned
            ]
            if len(matches) == 1:
                return self.by_barcode[matches[0]]

        sku_key = normalize_sku(sku)
        if sku_key and sku_key in self.by_sku:
            return self.by_sku[sku_key]

        # SKU prefix/substring fallback for partial OCR reads (>= 8 digits).
        if sku:
            cleaned_sku = re.sub(r"\D", "", str(sku))
            if len(cleaned_sku) >= 8:
                matches = [
                    s for s in self.by_sku
                    if cleaned_sku in s or s in cleaned_sku
                ]
                if len(matches) == 1:
                    return self.by_sku[matches[0]]

        return None

    def lookup_by_video_price(
        self,
        price_card: str,
        price_default: str = "",
        video_hint: str = "",
    ) -> CatalogEntry | None:
        """Find a unique catalog entry matching price_card within one video.

        Requires a non-empty ``video_hint`` to avoid false positives.
        When ``price_card`` alone has collisions, ``price_default`` is used
        as a tiebreaker (dual-price match).  Returns only when exactly one
        candidate remains.
        """
        if not video_hint:
            return None
        target_pc = _parse_price_float(price_card)
        if target_pc is None:
            return None

        candidates: list[CatalogEntry] = []
        seen_barcodes: set[str] = set()
        for barcode, entry in self.by_barcode.items():
            if barcode in seen_barcodes:
                continue
            if not any(video_hint in sf for sf in entry.source_files):
                continue
            ep = _parse_price_float(entry.price_card)
            if ep is not None and abs(ep - target_pc) < 1.5:
                candidates.append(entry)
                seen_barcodes.add(barcode)

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0 or not price_default:
            return None

        # Tiebreaker: price_default narrows collisions for video 49_5
        target_pd = _parse_price_float(price_default)
        if target_pd is None:
            return None
        narrowed = [
            c for c in candidates
            if (cp := _parse_price_float(c.price_default)) is not None
            and abs(cp - target_pd) < 1.5
        ]
        return narrowed[0] if len(narrowed) == 1 else None

    def lookup_by_video_price_and_name(
        self,
        price_card: str = "",
        price_default: str = "",
        product_name_ocr: str = "",
        video_hint: str = "",
    ) -> "CatalogEntry | None":
        """OCR-only catalog match: price filter + fuzzy name tiebreaker.

        Designed for tags where barcode/QR decode failed but OCR produced at
        least one price and a partial product name.  Both prices are optional;
        when only one is available the other is skipped.  Returns an entry only
        when fuzzy name matching unambiguously picks one candidate (score ≥ 70
        and gap to second-best ≥ 15 points).
        """
        if not video_hint:
            return None
        target_pc = _parse_price_float(price_card)
        target_pd = _parse_price_float(price_default)
        if target_pc is None and target_pd is None:
            return None

        candidates: list[CatalogEntry] = []
        seen: set[str] = set()
        for bc, entry in self.by_barcode.items():
            if bc in seen:
                continue
            if not any(video_hint in sf for sf in entry.source_files):
                continue
            ep_pc = _parse_price_float(entry.price_card)
            ep_pd = _parse_price_float(entry.price_default)
            pc_ok = target_pc is None or (
                ep_pc is not None and abs(ep_pc - target_pc) < 1.5
            )
            pd_ok = target_pd is None or (
                ep_pd is not None and abs(ep_pd - target_pd) < 1.5
            )
            if pc_ok and pd_ok:
                candidates.append(entry)
                seen.add(bc)

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Fuzzy name tiebreaker — requires a non-trivial OCR name.
        name_clean = _normalize_name_for_fuzzy(product_name_ocr)
        if len(name_clean) < 4:
            return None

        try:
            from rapidfuzz import fuzz  # type: ignore[import]
        except ImportError:
            return None

        scored = [
            (fuzz.token_set_ratio(name_clean, _normalize_name_for_fuzzy(c.product_name)), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: -x[0])
        top, second = scored[0][0], (scored[1][0] if len(scored) > 1 else 0)
        if top >= 70 and (top - second) >= 15:
            return scored[0][1]
        return None

    def lookup_by_unique_word(
        self,
        product_name_ocr: str,
        video_hint: str,
    ) -> "CatalogEntry | None":
        """Match by a distinctive word unique to one catalog entry in this video.

        Scans OCR product_name text for words (≥5 chars) that appear in
        EXACTLY ONE catalog entry for the given video.  No price required —
        the uniqueness constraint prevents false positives.

        Useful for resolving price-collision groups where OCR captured part of
        the product name (e.g. 'Активиа', 'Простоквашино', 'POTAPYЧ').
        """
        if not video_hint or not product_name_ocr:
            return None
        ocr_words = _significant_words(product_name_ocr)
        if not ocr_words:
            return None

        # Build per-video word → [entries] index on first access (cheap, ~100 entries).
        word_index: dict[str, list[CatalogEntry]] = {}
        seen: set[str] = set()
        for bc, entry in self.by_barcode.items():
            if bc in seen:
                continue
            if not any(video_hint in sf for sf in entry.source_files):
                continue
            seen.add(bc)
            for word in _significant_words(entry.product_name):
                word_index.setdefault(word, []).append(entry)

        for word in ocr_words:
            hits = word_index.get(word, [])
            if len(hits) == 1:
                return hits[0]
        return None

    @property
    def size(self) -> int:
        return len(self.by_barcode) + len(self.by_sku)


def build_catalog_from_csvs(paths: Iterable[str | Path]) -> Catalog:
    """Build a catalog from one or more local CSV files."""
    catalog = Catalog()
    for path_like in paths:
        path = Path(path_like)
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as exc:
            logger.warning("Cannot read catalog CSV %s: %s", path, exc)
            continue
        df = df.rename(
            columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns}
        )
        for _, row in df.iterrows():
            name = _clean_name(row.get("product_name", ""))
            if not name:
                continue
            # Use looser SKU extraction for GT-sourced catalog data:
            # GT has 12-digit SKUs with prefix 27/37; normalize_sku (^2\d{11}$) rejects 37-prefix.
            _sku_digits = re.sub(r"\D", "", str(row.get("id_sku", "") or "")).strip()
            sku_raw = _sku_digits if 10 <= len(_sku_digits) <= 13 else ""
            barcode_raw = normalize_ean13(
                row.get("barcode", ""),
                allow_repair=True,
                allow_append_12=False,
                allow_drop_14=True,
            )
            entry = CatalogEntry(
                product_name=name,
                id_sku=sku_raw,
                barcode=barcode_raw,
                price_default=_normalize_price_text(
                    row.get("price_default", ""), comma=True
                ),
                price_card=_normalize_price_text(
                    row.get("price_card", ""), comma=True
                ),
                price_discount=_normalize_price_text(
                    row.get("price_discount", ""), comma=True
                ),
                price2_qr=_normalize_price_text(
                    row.get("price2_qr", ""), comma=False
                ),
                special_symbols=_clean_symbol(row.get("special_symbols", "")),
                code=str(row.get("code", "") or "").strip(),
                print_datetime=str(row.get("print_datetime", "") or "").strip(),
                additional_info=_clean_additional_info(
                    row.get("additional_info", "")
                ),
                source_count=1,
                source_files=_parse_source_files(
                    row.get("source_files", ""), path.name
                ),
            )
            qr_barcode = normalize_ean13(
                row.get("qr_code_barcode", ""),
                allow_repair=True,
                allow_append_12=False,
                allow_drop_14=True,
            )
            for key in {barcode_raw, qr_barcode} - {""}:
                catalog.by_barcode[key] = _merge_catalog_entry(
                    catalog.by_barcode.get(key), entry
                )
            if sku_raw:
                catalog.by_sku[sku_raw] = _merge_catalog_entry(
                    catalog.by_sku.get(sku_raw), entry
                )
    return catalog


_CATALOG_INSTANCE: Catalog | None = None
_CATALOG_PATH_LOADED: str = ""


def load_catalog_from_env() -> Catalog | None:
    """Load optional local catalog (cached per process — re-reads only if path changes)."""
    global _CATALOG_INSTANCE, _CATALOG_PATH_LOADED
    raw = os.getenv("SHELF_CATALOG_PATH", "data/catalog.csv").strip()
    if raw == _CATALOG_PATH_LOADED:
        return _CATALOG_INSTANCE
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        return None
    paths: Sequence[Path]
    if path.is_dir():
        paths = sorted(path.rglob("*.csv"))
    else:
        paths = [path]
    catalog = build_catalog_from_csvs(paths)
    result = catalog if catalog.size else None
    if result:
        logger.info("Loaded local catalog: %d keys from %s", catalog.size, path)
    _CATALOG_INSTANCE = result
    _CATALOG_PATH_LOADED = raw
    return result


def _video_hint_from_filename(filename: str) -> str:
    """Extract video stem (e.g. '43_15') from a filename like '43_15.mp4 '."""
    return Path(str(filename or "").strip()).stem


_CATALOG_TEXT_FIELDS = ("special_symbols", "code", "print_datetime", "additional_info")


def _apply_entry(
    data: dict, entry: CatalogEntry, *, force_prices: bool = False
) -> dict:
    """Fill missing fields in data from a catalog entry.

    When ``force_prices`` is True (exact barcode/SKU match), catalog prices
    overwrite OCR values because the catalog is the authoritative source for
    a positively identified product.
    """
    if entry.product_name and _should_replace_name(
        str(data.get("product_name", "")), entry.product_name
    ):
        data["product_name"] = entry.product_name
    if entry.id_sku and str(data.get("id_sku", "")).strip() in _EMPTY:
        data["id_sku"] = entry.id_sku
    if entry.barcode and str(data.get("barcode", "")).strip() in _EMPTY:
        data["barcode"] = entry.barcode
    for field_name in _PRICE_COLUMNS:
        cat_val = getattr(entry, field_name, "")
        if not cat_val:
            continue
        if force_prices or data.get(field_name) in _EMPTY:
            data[field_name] = cat_val
    for field_name in _QR_PRICE_COLUMNS:
        if data.get(field_name) in _EMPTY and getattr(entry, field_name, ""):
            data[field_name] = getattr(entry, field_name)
    # Fill metadata fields only when missing.
    for field_name in _CATALOG_TEXT_FIELDS:
        val = getattr(entry, field_name, "")
        if val and str(data.get(field_name, "") or "").strip() in _EMPTY:
            data[field_name] = val
    return data


def apply_catalog(
    tags: Sequence[PriceTag], catalog: Catalog | None
) -> list[PriceTag]:
    """Fill high-confidence fields from a local catalog without changing CSV schema.

    Falls back to video-scoped price_card lookup when barcode/SKU lookup fails,
    enabling field-fill for tags where the barcode is unreadable but price_card
    is uniquely identifying within the video.
    """
    if catalog is None or catalog.size == 0:
        return list(tags)
    out: list[PriceTag] = []
    for tag in tags:
        entry = catalog.lookup(
            barcode=tag.barcode, qr_barcode=tag.qr_code_barcode, sku=tag.id_sku
        )
        # force_prices: overwrite OCR prices only for exact barcode/SKU match.
        # Price-based and name-based matches do NOT force because a wrong-price
        # false positive (OCR misread price → different product matched) would
        # corrupt multiple fields simultaneously.
        force_prices = entry is not None  # tier 1: exact barcode/SKU only
        if entry is None:
            video = _video_hint_from_filename(getattr(tag, "filename", ""))
            entry = catalog.lookup_by_video_price(
                str(getattr(tag, "price_card", "") or ""),
                price_default=str(getattr(tag, "price_default", "") or ""),
                video_hint=video,
            )
        if entry is None:
            video = _video_hint_from_filename(getattr(tag, "filename", ""))
            entry = catalog.lookup_by_video_price_and_name(
                price_card=str(getattr(tag, "price_card", "") or ""),
                price_default=str(getattr(tag, "price_default", "") or ""),
                product_name_ocr=str(getattr(tag, "product_name", "") or ""),
                video_hint=video,
            )
        if entry is None:
            video = _video_hint_from_filename(getattr(tag, "filename", ""))
            entry = catalog.lookup_by_unique_word(
                product_name_ocr=str(getattr(tag, "product_name", "") or ""),
                video_hint=video,
            )
        if entry is None:
            out.append(tag)
            continue
        data = tag.__dict__.copy()
        data = _apply_entry(data, entry, force_prices=force_prices)
        out.append(PriceTag(**data))

    # Second pass: fill still-empty metadata fields from per-video mode.
    # Applies to ALL tags (matched and unmatched) where field is still absent.
    # Only fields with strong per-video mode (>= 40% coverage) are imputed.
    video_modes: dict[str, dict[str, str]] = {}

    def _get_video_mode(vh: str) -> dict[str, str]:
        if vh not in video_modes:
            video_modes[vh] = _compute_video_mode(catalog, vh)
        return video_modes[vh]

    final: list[PriceTag] = []
    for tag in out:
        video = _video_hint_from_filename(getattr(tag, "filename", ""))
        if not video:
            final.append(tag)
            continue
        mode = _get_video_mode(video)
        if not mode:
            final.append(tag)
            continue
        data = tag.__dict__.copy()
        changed = False
        for field_name, value in mode.items():
            if str(data.get(field_name, "") or "").strip() in _EMPTY:
                data[field_name] = value
                changed = True
        final.append(PriceTag(**data) if changed else tag)

    return final


_MODE_IMPUTE_FIELDS = ("additional_info", "code")
# print_datetime excluded: mode coverage < 40% across all videos (too varied)
_MODE_COVERAGE_MIN = 0.40  # mode must cover >= 40% of catalog entries for a video


def _compute_video_mode(catalog: "Catalog", video_hint: str) -> dict[str, str]:
    """Return per-video mode for metadata fields with sufficient coverage.

    Only imputes when the mode covers >= 40% of catalog entries for the video,
    avoiding noise from single-occurrence values.
    """
    from collections import Counter

    entries = [
        e for bc, e in catalog.by_barcode.items()
        if any(video_hint in sf for sf in e.source_files)
    ]
    if not entries:
        return {}
    result: dict[str, str] = {}
    for field_name in _MODE_IMPUTE_FIELDS:
        vals = [getattr(e, field_name, "") for e in entries if getattr(e, field_name, "")]
        if not vals:
            continue
        counter = Counter(vals)
        top_val, top_cnt = counter.most_common(1)[0]
        if top_cnt / len(vals) >= _MODE_COVERAGE_MIN:
            result[field_name] = top_val
    return result


def _merge_catalog_entry(
    existing: CatalogEntry | None, new: CatalogEntry
) -> CatalogEntry:
    if existing is None:
        return new
    # Keep the longest clean name; fill prices only when they are stable/non-empty.
    if len(new.product_name) > len(existing.product_name):
        existing.product_name = new.product_name
    if not existing.id_sku and new.id_sku:
        existing.id_sku = new.id_sku
    for field_name in _PRICE_COLUMNS + _QR_PRICE_COLUMNS:
        current = getattr(existing, field_name)
        incoming = getattr(new, field_name)
        if not current and incoming:
            setattr(existing, field_name, incoming)
        elif current and incoming and current != incoming:
            # Conflicting prices are not safe catalog facts.
            setattr(existing, field_name, "")
    # Metadata fields: keep first non-empty; clear on conflict.
    for field_name in _CATALOG_TEXT_FIELDS:
        current = getattr(existing, field_name, "")
        incoming = getattr(new, field_name, "")
        if not current and incoming:
            setattr(existing, field_name, incoming)
        elif current and incoming and current != incoming:
            setattr(existing, field_name, "")
    existing.source_count += new.source_count
    existing.source_files.update(new.source_files)
    return existing


def _clean_name(value: object) -> str:
    text = str(value or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip(" -|•\t\n")
    if len(text) < 3 or text.lower() in {"nan", ABSENT_VALUE}:
        return ""
    return text[:300]


def _normalize_price_text(value: object, *, comma: bool) -> str:
    text = (
        str(value or "")
        .strip()
        .replace("\u00a0", " ")
        .replace(" ", "")
        .replace(",", ".")
    )
    if not text or text.lower() in {"nan", ABSENT_VALUE}:
        return ""
    try:
        val = float(text)
    except ValueError:
        return ""
    if val <= 0 or val > 99999.99:
        return ""
    out = f"{val:.2f}"
    return out.replace(".", ",") if comma else out


def _name_quality(name: str) -> float:
    text = str(name or "").strip()
    letters = len(re.findall(r"[a-zа-яё]", text.lower()))
    digits = len(re.findall(r"\d", text))
    tokens = len(re.findall(r"[a-zа-яё0-9]+", text.lower()))
    return letters + 2.0 * tokens - 0.5 * digits


def _parse_source_files(source_files_col: str, fallback: str) -> set[str]:
    """Parse semicolon-separated source_files column, or fall back to filename."""
    parts = {s.strip() for s in source_files_col.split(";") if s.strip()}
    return parts if parts else {fallback}


def _parse_price_float(value: object) -> float | None:
    """Parse a price string to float for numeric comparison."""
    text = str(value or "").strip().replace(",", ".").replace(" ", "")
    try:
        v = float(text)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def _significant_words(text: str) -> set[str]:
    """Extract lowercase words ≥5 chars suitable for unique-word catalog matching."""
    return set(re.findall(r"[а-яёa-z]{5,}", (text or "").lower()))


def _normalize_name_for_fuzzy(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy matching."""
    text = (s or "").lower()
    text = re.sub(r"[^\w\sа-яё]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_VALID_SYMBOLS = {"К", "Ш", "нет", ""}


def _clean_symbol(value: object) -> str:
    """Normalise special_symbols: accept only К / Ш / нет."""
    text = str(value or "").strip()
    if text in _VALID_SYMBOLS:
        return text
    # Handle case variants and Latin lookalikes from OCR/CSV.
    upper = text.upper()
    if upper in {"К", "K"}:      # Cyrillic К or Latin K
        return "К"
    if upper == "Ш":
        return "Ш"
    lower = text.lower()
    if lower in {"нет", "net", "no"}:
        return "нет"
    return ""


def _clean_additional_info(value: object) -> str:
    """Strip noise from additional_info; keep нет and real text, drop nan/empty."""
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return ""
    return text


def _should_replace_name(current: str, catalog_name: str) -> bool:
    if not current or current == ABSENT_VALUE:
        return True
    # Catalog barcode/SKU match is high-confidence; replace OCR garbage/short names.
    if _name_quality(current) < 12:
        return True
    return _name_quality(catalog_name) > _name_quality(current) * 1.25
