"""Catalog coverage diagnostic.

Replays catalog lookup against GT field values to answer:
  "Which lookup branch fires for each tag — and how many remain unresolved?"

This is a STATIC analysis (no video reading required).  It uses GT barcodes /
prices as if OCR were perfect, giving the theoretical upper bound for each
lookup branch.

Usage:
  python scripts/diag_catalog_coverage.py
  python scripts/diag_catalog_coverage.py --out reports/catalog_coverage.json

Output JSON:
  {
    "overall": { "total": 274, branch_counts... },
    "per_video": { "<name>": { branch_counts } },
    "unresolved_samples": [ { "video": ..., "barcode": ..., "price_card": ... }, ... ]
  }

Lookup branch names (in priority order):
  exact_barcode       — normalize_ean13(gt_barcode) hit catalog.by_barcode
  prefix_barcode      — digit-substring fallback on gt_barcode (≥8 digits, 1 match)
  exact_qr_barcode    — normalize_ean13(gt_qr_barcode) hit catalog.by_barcode
  prefix_qr_barcode   — digit-substring fallback on gt_qr_barcode
  exact_sku           — normalize_sku(gt_sku) hit catalog.by_sku
  prefix_sku          — digit-substring fallback on gt_sku (≥8 digits, 1 match)
  video_price_single  — lookup_by_video_price(price_card) unique within video
  video_price_dual    — lookup_by_video_price(price_card+price_default) unique
  no_match            — catalog has no entry for this tag
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shelf.postproc.catalog import (
    Catalog,
    CatalogEntry,
    _parse_price_float,
    build_catalog_from_csvs,
)
from shelf.validation import normalize_ean13, normalize_sku

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATA_ROOT = Path("Данные")
LABELED_NAMES = ["25_12-20", "25_2-10", "26_12-20", "43_15", "49_5"]
CATALOG_PATH = Path("data/catalog.csv")

BRANCHES = [
    "exact_barcode",
    "prefix_barcode",
    "exact_qr_barcode",
    "prefix_qr_barcode",
    "exact_sku",
    "prefix_sku",
    "video_price_single",
    "video_price_dual",
    "no_match",
]


def _norm_bc(val: object) -> str:
    s = str(val or "").strip()
    s = re.sub(r"\s+", "", s)
    try:
        if "." in s and s.replace(".", "").isdigit():
            s = str(int(float(s)))
        if s.isdigit() and len(s) < 13:
            s = s.zfill(13)
    except (ValueError, OverflowError):
        pass
    return s


def _classify_branch(
    catalog: Catalog,
    barcode_raw: str,
    qr_barcode_raw: str,
    sku_raw: str,
    price_card: str,
    price_default: str,
    video: str,
) -> str:
    """Return which branch first resolves this tag."""
    # 1. Exact EAN-13 via barcode
    key = normalize_ean13(barcode_raw, allow_repair=True, allow_append_12=False, allow_drop_14=True)
    if key and key in catalog.by_barcode:
        return "exact_barcode"

    # 2. Prefix/substring fallback on barcode
    cleaned = re.sub(r"\D", "", str(barcode_raw or ""))
    if len(cleaned) >= 8:
        matches = [bc for bc in catalog.by_barcode if cleaned in bc or bc in cleaned]
        if len(matches) == 1:
            return "prefix_barcode"

    # 3. Exact EAN-13 via QR barcode
    key = normalize_ean13(qr_barcode_raw, allow_repair=True, allow_append_12=False, allow_drop_14=True)
    if key and key in catalog.by_barcode:
        return "exact_qr_barcode"

    # 4. Prefix/substring fallback on QR barcode
    cleaned = re.sub(r"\D", "", str(qr_barcode_raw or ""))
    if len(cleaned) >= 8:
        matches = [bc for bc in catalog.by_barcode if cleaned in bc or bc in cleaned]
        if len(matches) == 1:
            return "prefix_qr_barcode"

    # 5. Exact SKU
    sku_key = normalize_sku(sku_raw)
    if sku_key and sku_key in catalog.by_sku:
        return "exact_sku"

    # 5b. Loose SKU (catalog uses raw digits 10-13, not normalize_sku)
    sku_digits = re.sub(r"\D", "", str(sku_raw or ""))
    if 10 <= len(sku_digits) <= 13 and sku_digits in catalog.by_sku:
        return "exact_sku"

    # 6. SKU prefix fallback
    if len(sku_digits) >= 8:
        matches = [s for s in catalog.by_sku if sku_digits in s or s in sku_digits]
        if len(matches) == 1:
            return "prefix_sku"

    # 7. Video-price lookup
    if video:
        target_pc = _parse_price_float(price_card)
        if target_pc is not None:
            candidates: list[CatalogEntry] = []
            seen: set[str] = set()
            for bc, entry in catalog.by_barcode.items():
                if bc in seen:
                    continue
                if not any(video in sf for sf in entry.source_files):
                    continue
                ep = _parse_price_float(entry.price_card)
                if ep is not None and abs(ep - target_pc) < 1.5:
                    candidates.append(entry)
                    seen.add(bc)

            if len(candidates) == 1:
                return "video_price_single"

            if len(candidates) > 1:
                target_pd = _parse_price_float(price_default)
                if target_pd is not None:
                    narrowed = [
                        c for c in candidates
                        if (cp := _parse_price_float(c.price_default)) is not None
                        and abs(cp - target_pd) < 1.5
                    ]
                    if len(narrowed) == 1:
                        return "video_price_dual"

    return "no_match"


def _load_gt(name: str) -> pd.DataFrame | None:
    csv = DATA_ROOT / name / f"{name}.csv"
    if not csv.exists():
        logger.warning("GT CSV not found: %s", csv)
        return None
    df = pd.read_csv(csv, dtype=str, keep_default_na=False, decimal=",")
    if "wholesale_level_1_coun" in df.columns:
        df = df.rename(columns={"wholesale_level_1_coun": "wholesale_level_1_count"})
    for col in ["barcode", "qr_code_barcode"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm_bc)
    return df


def run_diag(catalog: Catalog) -> dict:
    overall: dict[str, int] = defaultdict(int)
    per_video: dict[str, dict[str, int]] = {}
    unresolved: list[dict] = []
    branch_examples: dict[str, list[dict]] = defaultdict(list)

    for name in LABELED_NAMES:
        df = _load_gt(name)
        if df is None:
            continue
        vcounts: dict[str, int] = defaultdict(int)
        for _, row in df.iterrows():
            bc = str(row.get("barcode", "") or "")
            qr = str(row.get("qr_code_barcode", "") or "")
            sku = str(row.get("id_sku", "") or "")
            pc = str(row.get("price_card", "") or "")
            pd_val = str(row.get("price_default", "") or "")

            branch = _classify_branch(catalog, bc, qr, sku, pc, pd_val, name)
            overall[branch] += 1
            vcounts[branch] += 1

            sample = {"video": name, "barcode": bc, "qr_barcode": qr[:20],
                      "sku": sku, "price_card": pc, "product_name": str(row.get("product_name", ""))[:40]}
            if branch == "no_match":
                unresolved.append(sample)
            elif len(branch_examples[branch]) < 3:
                branch_examples[branch].append(sample)

        per_video[name] = dict(vcounts)

    total = sum(overall.values())
    result = {
        "overall": {
            "total": total,
            **{b: overall.get(b, 0) for b in BRANCHES},
            "resolved": total - overall.get("no_match", 0),
            "resolved_pct": round(100 * (total - overall.get("no_match", 0)) / max(1, total), 1),
        },
        "per_video": {n: {b: per_video.get(n, {}).get(b, 0) for b in BRANCHES} for n in LABELED_NAMES},
        "branch_examples": dict(branch_examples),
        "unresolved_samples": unresolved[:30],
    }
    return result


def _print_report(r: dict) -> None:
    o = r["overall"]
    total = o["total"]
    print(f"\n{'='*65}")
    print(f"CATALOG COVERAGE DIAGNOSTIC  ({total} GT tags, upper bound)")
    print(f"{'='*65}")
    print(f"  Resolved (any branch): {o['resolved']:3d}/{total}  ({o['resolved_pct']:.1f}%)")
    print()
    print(f"  {'Branch':<24} {'Count':>6}  {'%':>6}  Bar")
    print(f"  {'-'*54}")
    for b in BRANCHES:
        n = o.get(b, 0)
        pct = 100 * n / max(1, total)
        bar = "█" * int(pct / 2)
        print(f"  {b:<24} {n:>6}  {pct:>5.1f}%  {bar}")

    print(f"\n  Per-video breakdown:")
    print(f"  {'Video':<14} " + "  ".join(f"{b[:8]:>8}" for b in BRANCHES))
    for name in LABELED_NAMES:
        pv = r["per_video"].get(name, {})
        row_vals = "  ".join(f"{pv.get(b, 0):>8}" for b in BRANCHES)
        print(f"  {name:<14} {row_vals}")

    nm = r["overall"].get("no_match", 0)
    if nm:
        print(f"\n  Sample unresolved tags ({nm} total):")
        for s in r["unresolved_samples"][:10]:
            print(f"    [{s['video']}] bc={s['barcode'][:13] or '—':13s}  pc={s['price_card']:8s}  {s['product_name'][:35]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Catalog coverage diagnostic")
    ap.add_argument("--catalog", default=str(CATALOG_PATH), help="Path to catalog CSV or dir")
    ap.add_argument("--out", default="reports/catalog_coverage.json", help="Output JSON path")
    args = ap.parse_args()

    cat_path = Path(args.catalog)
    if not cat_path.exists():
        print(f"ERROR: catalog not found at {cat_path}", file=sys.stderr)
        sys.exit(1)
    paths = sorted(cat_path.rglob("*.csv")) if cat_path.is_dir() else [cat_path]
    catalog = build_catalog_from_csvs(paths)
    print(f"Loaded catalog: {len(catalog.by_barcode)} barcodes, {len(catalog.by_sku)} SKUs")

    result = run_diag(catalog)
    _print_report(result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nFull report → {out}")


if __name__ == "__main__":
    main()
