"""Build a local product catalog from provided CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shelf.postproc.catalog import build_catalog_from_csvs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local barcode/SKU catalog from labeled or generated CSV files"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="CSV files or directories with CSV files",
    )
    parser.add_argument("--out", type=Path, default=Path("data/catalog.csv"))
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.csv")))
        else:
            paths.append(item)
    catalog = build_catalog_from_csvs(paths)
    rows = []
    for barcode, entry in sorted(catalog.by_barcode.items()):
        rows.append(
            {
                "barcode": barcode,
                "id_sku": "",
                "product_name": entry.product_name,
                "price_default": entry.price_default,
                "price_card": entry.price_card,
                "price_discount": entry.price_discount,
                "source_count": entry.source_count,
                "source_files": ";".join(sorted(entry.source_files)),
            }
        )
    for sku, entry in sorted(catalog.by_sku.items()):
        rows.append(
            {
                "barcode": "",
                "id_sku": sku,
                "product_name": entry.product_name,
                "price_default": entry.price_default,
                "price_card": entry.price_card,
                "price_discount": entry.price_discount,
                "source_count": entry.source_count,
                "source_files": ";".join(sorted(entry.source_files)),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates().to_csv(
        args.out, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote {len(rows)} catalog rows to {args.out}")


if __name__ == "__main__":
    main()
