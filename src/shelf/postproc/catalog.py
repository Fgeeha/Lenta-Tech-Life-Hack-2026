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


@dataclass
class CatalogEntry:
    """One locally known product keyed by barcode or SKU."""

    product_name: str = ""
    price_default: str = ""
    price_card: str = ""
    price_discount: str = ""
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
        """Find an entry by valid EAN-13 or strict SKU."""
        for raw in (qr_barcode, barcode):
            key = normalize_ean13(
                raw,
                allow_repair=True,
                allow_append_12=False,
                allow_drop_14=True,
            )
            if key and key in self.by_barcode:
                return self.by_barcode[key]
        sku_key = normalize_sku(sku)
        if sku_key and sku_key in self.by_sku:
            return self.by_sku[sku_key]
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
            entry = CatalogEntry(
                product_name=name,
                price_default=_normalize_price_text(
                    row.get("price_default", ""), comma=True
                ),
                price_card=_normalize_price_text(
                    row.get("price_card", ""), comma=True
                ),
                price_discount=_normalize_price_text(
                    row.get("price_discount", ""), comma=True
                ),
                source_count=1,
                source_files={path.name},
            )
            barcode = normalize_ean13(
                row.get("barcode", ""),
                allow_repair=True,
                allow_append_12=False,
                allow_drop_14=True,
            )
            qr_barcode = normalize_ean13(
                row.get("qr_code_barcode", ""),
                allow_repair=True,
                allow_append_12=False,
                allow_drop_14=True,
            )
            sku = normalize_sku(row.get("id_sku", ""))
            for key in {barcode, qr_barcode} - {""}:
                catalog.by_barcode[key] = _merge_catalog_entry(
                    catalog.by_barcode.get(key), entry
                )
            if sku:
                catalog.by_sku[sku] = _merge_catalog_entry(
                    catalog.by_sku.get(sku), entry
                )
    return catalog


def load_catalog_from_env() -> Catalog | None:
    """Load optional local catalog configured by environment or default path."""
    raw = os.getenv("SHELF_CATALOG_PATH", "data/catalog.csv").strip()
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
    if catalog.size:
        logger.info("Loaded local catalog: %d keys from %s", catalog.size, path)
        return catalog
    return None


def apply_catalog(
    tags: Sequence[PriceTag], catalog: Catalog | None
) -> list[PriceTag]:
    """Fill high-confidence fields from a local catalog without changing CSV schema."""
    if catalog is None or catalog.size == 0:
        return list(tags)
    out: list[PriceTag] = []
    for tag in tags:
        entry = catalog.lookup(
            barcode=tag.barcode, qr_barcode=tag.qr_code_barcode, sku=tag.id_sku
        )
        if entry is None:
            out.append(tag)
            continue
        data = tag.__dict__.copy()
        if entry.product_name and _should_replace_name(
            str(data.get("product_name", "")), entry.product_name
        ):
            data["product_name"] = entry.product_name
        for field_name in _PRICE_COLUMNS:
            if data.get(field_name) in _EMPTY and getattr(entry, field_name):
                data[field_name] = getattr(entry, field_name)
        out.append(PriceTag(**data))
    return out


def _merge_catalog_entry(
    existing: CatalogEntry | None, new: CatalogEntry
) -> CatalogEntry:
    if existing is None:
        return new
    # Keep the longest clean name; fill prices only when they are stable/non-empty.
    if len(new.product_name) > len(existing.product_name):
        existing.product_name = new.product_name
    for field_name in _PRICE_COLUMNS:
        current = getattr(existing, field_name)
        incoming = getattr(new, field_name)
        if not current and incoming:
            setattr(existing, field_name, incoming)
        elif current and incoming and current != incoming:
            # Conflicting prices are not safe catalog facts.
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


def _should_replace_name(current: str, catalog_name: str) -> bool:
    if not current or current == ABSENT_VALUE:
        return True
    # Catalog barcode/SKU match is high-confidence; replace OCR garbage/short names.
    if _name_quality(current) < 12:
        return True
    return _name_quality(catalog_name) > _name_quality(current) * 1.25
