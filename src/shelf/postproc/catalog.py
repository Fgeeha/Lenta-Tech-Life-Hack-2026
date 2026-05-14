"""Barcode → product_name lookup built from GT CSV data.

Catalog is built once from Данные/**/*.csv and stored as data/sku_catalog.json.
At runtime, provides O(1) lookup: catalog.lookup_product_name(barcode) -> str.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CATALOG: dict[str, str] | None = None
_CATALOG_PATH = Path(__file__).parents[3] / "data" / "sku_catalog.json"


def _load() -> dict[str, str]:
    global _CATALOG
    if _CATALOG is None:
        try:
            with open(_CATALOG_PATH, encoding="utf-8") as f:
                _CATALOG = json.load(f)
            logger.info("SKU catalog loaded: %d entries", len(_CATALOG))
        except FileNotFoundError:
            logger.warning("SKU catalog not found at %s", _CATALOG_PATH)
            _CATALOG = {}
    return _CATALOG


def lookup_product_name(barcode: str) -> str:
    """Return product_name for barcode, or '' if not in catalog."""
    if not barcode or not barcode.strip().isdigit():
        return ""
    return _load().get(barcode.strip(), "")


def catalog_size() -> int:
    return len(_load())
