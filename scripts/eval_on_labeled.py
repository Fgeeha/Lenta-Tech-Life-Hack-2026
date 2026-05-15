"""Evaluate the full video pipeline on the locally available labeled videos.

The script is intentionally self-contained and robust to missing data: if the
private challenge videos are not present in ``--data-root`` it exits cleanly and
prints the exact paths that are missing.  Metrics are computed with the same
matching idea used during the project: predictions are matched to GT rows by
barcode/QR, IoU and timestamp proximity, then field-level correctness is scored.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from shelf import pipeline
from shelf.io.writer import prepare_output_dataframe
from shelf.qr.barcode_roi import ean13_checksum_valid
from shelf.schema import ABSENT_VALUE, COLUMN_ALIASES, OUTPUT_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("Данные")
LABELED_NAMES = ["25_12-20", "25_2-10", "26_12-20", "43_15", "49_5"]

# Historical compact metric, kept for comparability with previous project notes.
COMPACT_EVAL_FIELDS = [
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price4_qr",
]

ALL_VALUE_FIELDS = [
    c
    for c in OUTPUT_COLUMNS
    if c
    not in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}
]


@dataclass(frozen=True)
class LabeledVideo:
    """Paths for one labeled challenge video."""

    name: str
    video_path: Path
    csv_path: Path


def find_labeled_videos(
    data_root: Path, names: Iterable[str] | None = None
) -> list[LabeledVideo]:
    """Return expected video/CSV pairs under ``data_root``."""
    selected = list(names or LABELED_NAMES)
    items: list[LabeledVideo] = []
    for name in selected:
        folder = data_root / name
        items.append(
            LabeledVideo(
                name=name,
                video_path=folder / f"{name}.mp4",
                csv_path=folder / f"{name}.csv",
            )
        )
    return items


def _to_text(val: Any) -> str:
    if val is None:
        return ""
    try:
        if val != val:  # NaN without importing numpy
            return ""
    except Exception:
        pass
    return str(val).strip()


def _norm_digits(val: Any) -> str:
    """Normalize a barcode-like value read by pandas as int/float/string."""
    s = _to_text(val)
    if not s or s.lower() == "nan" or s == ABSENT_VALUE:
        return ""
    s = s.replace("\u00a0", " ").strip()
    try:
        if re.fullmatch(r"\d+\.0", s):
            s = str(int(float(s)))
    except (ValueError, OverflowError):
        pass
    digits = re.sub(r"\D", "", s)
    return digits


def _norm_barcode(val: Any) -> str:
    digits = _norm_digits(val)
    if (
        len(digits) == 14
        and digits.endswith("0")
        and ean13_checksum_valid(digits[:-1])
    ):
        return digits[:-1]
    if len(digits) < 13 and digits:
        return digits.zfill(13)
    return digits


def _norm_sku(val: Any) -> str:
    digits = _norm_digits(val)
    return digits


def _to_float(val: Any, default: float = 0.0) -> float:
    s = _to_text(val).replace("\u00a0", " ").replace(" ", "").replace(",", ".")
    if not s or s == ABSENT_VALUE:
        return default
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def normalize_gt(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize quirks in the provided GT CSV files."""
    df = df.rename(
        columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns}
    ).copy()
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUTPUT_COLUMNS]
    if "filename" in df.columns:
        df["filename"] = df["filename"].apply(_to_text)
    for col in ["barcode", "qr_code_barcode"]:
        df[col] = df[col].apply(_norm_barcode)
    df["id_sku"] = df["id_sku"].apply(_norm_sku)
    for col in ["frame_timestamp", "x_min", "y_min", "x_max", "y_max"]:
        df[col] = df[col].apply(_to_float)
    return df


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (
        max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        + max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        - inter
    )
    return inter / union if union > 0 else 0.0


def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[0-9a-zа-яё]+", a.lower(), flags=re.IGNORECASE))
    tb = set(re.findall(r"[0-9a-zа-яё]+", b.lower(), flags=re.IGNORECASE))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _field_match(pred_val: Any, gt_val: Any, field: str = "") -> bool:
    """Return whether a predicted field matches GT with task-specific tolerance."""
    p = _to_text(pred_val).lower()
    g = _to_text(gt_val).lower()
    if g in ("", "nan", ABSENT_VALUE):
        return True
    if p in ("", "nan", ABSENT_VALUE):
        return False

    if field in {"barcode", "qr_code_barcode", "id_sku"}:
        return _norm_digits(p) == _norm_digits(g)

    p_num = re.sub(r"[,.]", ".", p.replace(" ", ""))
    g_num = re.sub(r"[,.]", ".", g.replace(" ", ""))
    try:
        return abs(float(p_num) - float(g_num)) < 1.5
    except ValueError:
        pass

    if p == g:
        return True
    if field in {"product_name", "additional_info"} and len(g) > 10:
        return _token_overlap(p, g) >= 0.40
    return False


def _value_present(val: Any) -> bool:
    s = _to_text(val)
    return bool(s and s.lower() != "nan" and s != ABSENT_VALUE)


def fill_rates(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of rows with a recognized value for every output column."""
    if len(df) == 0:
        return {c: 0.0 for c in OUTPUT_COLUMNS}
    return {
        c: (
            float(df[c].apply(_value_present).sum()) / len(df)
            if c in df.columns
            else 0.0
        )
        for c in OUTPUT_COLUMNS
    }


def match_and_score(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, eval_fields: list[str]
) -> dict[str, Any]:
    """Match predictions to GT and compute metric@80%, avg field score and recall."""
    pred_df = prepare_output_dataframe(pred_df)
    used_pred: set[int] = set()
    field_hits: dict[str, int] = {f: 0 for f in eval_fields}
    match_scores: list[float] = []

    for _, gt_row in gt_df.iterrows():
        gt_bc = _norm_barcode(gt_row.get("barcode", ""))
        gt_qr = _norm_barcode(gt_row.get("qr_code_barcode", ""))
        gt_bbox = (
            _to_float(gt_row.get("x_min", 0)),
            _to_float(gt_row.get("y_min", 0)),
            _to_float(gt_row.get("x_max", 0)),
            _to_float(gt_row.get("y_max", 0)),
        )
        gt_ts_ms = _to_float(gt_row.get("frame_timestamp", 0))

        best_pred: int | None = None
        best_score = -1.0
        for pred_i, pred_row in pred_df.iterrows():
            if pred_i in used_pred:
                continue
            score = 0.0
            pred_bc = _norm_barcode(pred_row.get("barcode", ""))
            pred_qr = _norm_barcode(pred_row.get("qr_code_barcode", ""))
            if gt_bc and pred_bc and gt_bc == pred_bc:
                score += 2.0
            if gt_qr and pred_qr and gt_qr == pred_qr:
                score += 2.0
            pred_bbox = (
                _to_float(pred_row.get("x_min", 0)),
                _to_float(pred_row.get("y_min", 0)),
                _to_float(pred_row.get("x_max", 0)),
                _to_float(pred_row.get("y_max", 0)),
            )
            score += _iou(gt_bbox, pred_bbox) * 3.0
            pred_ts_ms = _to_float(pred_row.get("frame_timestamp", 0))
            if abs(pred_ts_ms - gt_ts_ms) <= 3000.0:
                score += 0.5
            if score > best_score:
                best_score = score
                best_pred = int(pred_i)

        if best_pred is None or best_score <= 0.3:
            match_scores.append(0.0)
            continue

        used_pred.add(best_pred)
        correct = 0
        for field in eval_fields:
            if field not in pred_df.columns or field not in gt_df.columns:
                continue
            ok = _field_match(
                pred_df.loc[best_pred, field],
                gt_row.get(field, ""),
                field=field,
            )
            if ok:
                correct += 1
                field_hits[field] += 1
        match_scores.append(correct / max(1, len(eval_fields)))

    n_gt = len(gt_df)
    n_matched = len(used_pred)
    n_pass = sum(1 for s in match_scores if s >= 0.80)
    return {
        "n_gt": n_gt,
        "n_pred": len(pred_df),
        "n_matched": n_matched,
        "detection_recall": n_matched / max(1, n_gt),
        "n_pass_80": n_pass,
        "metric_80": n_pass / max(1, n_gt),
        "avg_field": sum(match_scores) / max(1, n_gt),
        "field_accuracy": {
            f: field_hits[f] / max(1, n_gt) for f in eval_fields
        },
        "barcode_count": (
            int(pred_df["barcode"].apply(_value_present).sum())
            if "barcode" in pred_df
            else 0
        ),
        "qr_barcode_count": (
            int(pred_df["qr_code_barcode"].apply(_value_present).sum())
            if "qr_code_barcode" in pred_df
            else 0
        ),
        "fill_rates": fill_rates(pred_df),
    }


def evaluate_video(
    item: LabeledVideo, args: argparse.Namespace, eval_fields: list[str]
) -> dict[str, Any] | None:
    """Run pipeline on one video and score it against GT."""
    if not item.video_path.exists() or not item.csv_path.exists():
        logger.warning(
            "skip %s: missing video=%s csv=%s",
            item.name,
            item.video_path.exists(),
            item.csv_path.exists(),
        )
        return None

    gt_df = normalize_gt(pd.read_csv(item.csv_path, decimal=","))
    out_csv = (
        args.output_dir / f"{item.name}.pred.csv" if args.output_dir else None
    )
    pred_df = pipeline.run(
        item.video_path,
        interval_ms=args.interval_ms,
        adaptive=args.adaptive,
        min_hits=args.min_hits,
        detector_name=args.detector,
        ocr_top_k=args.ocr_top_k,
        max_ocr_variants=args.max_ocr_variants,
        ocr_engine_name=args.ocr_engine,
        max_duration_sec=args.max_duration_sec,
        output_csv=out_csv,
    )
    scores = match_and_score(pred_df, gt_df, eval_fields=eval_fields)
    scores["video"] = item.name
    return scores


def summarize(
    results: list[dict[str, Any]], eval_fields: list[str]
) -> dict[str, Any]:
    """Aggregate per-video metrics."""
    total_gt = sum(int(r["n_gt"]) for r in results)
    total_pass = sum(int(r["n_pass_80"]) for r in results)
    total_pred = sum(int(r["n_pred"]) for r in results)
    total_matched = sum(int(r["n_matched"]) for r in results)
    avg_field = sum(
        float(r["avg_field"]) * int(r["n_gt"]) for r in results
    ) / max(1, total_gt)
    field_acc = {
        f: sum(
            float(r["field_accuracy"].get(f, 0.0)) * int(r["n_gt"])
            for r in results
        )
        / max(1, total_gt)
        for f in eval_fields
    }
    fill = {
        f: sum(
            float(r["fill_rates"].get(f, 0.0)) * max(1, int(r["n_pred"]))
            for r in results
        )
        / max(1, total_pred)
        for f in OUTPUT_COLUMNS
    }
    return {
        "videos": results,
        "overall": {
            "n_gt": total_gt,
            "n_pred": total_pred,
            "n_matched": total_matched,
            "detection_recall": total_matched / max(1, total_gt),
            "n_pass_80": total_pass,
            "metric_80": total_pass / max(1, total_gt),
            "avg_field": avg_field,
            "barcode_count": sum(int(r["barcode_count"]) for r in results),
            "qr_barcode_count": sum(
                int(r["qr_barcode_count"]) for r in results
            ),
            "field_accuracy": field_acc,
            "fill_rates": fill,
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact human-readable report."""
    print("\n=== Per-video metrics ===")
    for r in summary["videos"]:
        print(
            f"  {r['video']:12s} metric@80={r['metric_80']:.3f} "
            f"avg_field={r['avg_field']:.3f} recall={r['detection_recall']:.3f} "
            f"rows={r['n_pred']} gt={r['n_gt']} bc={r['barcode_count']} qr_bc={r['qr_barcode_count']}"
        )
    o = summary["overall"]
    print("\n=== Overall ===")
    print(
        f"  metric@80={o['metric_80']:.3f} avg_field={o['avg_field']:.3f} "
        f"recall={o['detection_recall']:.3f} rows={o['n_pred']} gt={o['n_gt']} "
        f"bc={o['barcode_count']} qr_bc={o['qr_barcode_count']}"
    )
    print("\n=== Weakest fields (accuracy) ===")
    for field, acc in sorted(o["field_accuracy"].items(), key=lambda kv: kv[1])[
        :12
    ]:
        print(f"  {field:<30} {acc:.3f}")
    print("\n=== Field fill rates over predictions ===")
    for field, rate in sorted(o["fill_rates"].items(), key=lambda kv: kv[1])[
        :12
    ]:
        print(f"  {field:<30} {rate:.3f}")


def append_metrics_md(
    summary: dict[str, Any], args: argparse.Namespace
) -> None:
    """Append a reproducible metrics row to docs/METRICS.md."""
    import subprocess

    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "no-git"
    )
    today = date.today().isoformat()
    o = summary["overall"]
    command = (
        f"PYTHONPATH=src python scripts/eval_on_labeled.py --data-root {args.data_root} "
        f"--interval-ms {args.interval_ms} --detector {args.detector} --ocr-engine {args.ocr_engine} --ocr-top-k {args.ocr_top_k}"
    )
    per_video = "; ".join(
        f"{r['video']} avg={r['avg_field']:.3f} m80={r['metric_80']:.3f}"
        for r in summary["videos"]
    )
    with Path("docs/METRICS.md").open("a", encoding="utf-8") as f:
        f.write(
            f"| {today} | {commit} | labeled-5 full pipeline | {o['metric_80']:.3f} | "
            f"avg_field={o['avg_field']:.3f}; recall={o['detection_recall']:.3f}; "
            f"rows={o['n_pred']}; gt={o['n_gt']}; bc={o['barcode_count']}; qr_bc={o['qr_barcode_count']}; "
            f"cmd=`{command}`; {per_video} |\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ShelfWatch on locally available labeled videos"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Subset of video names, e.g. 25_2-10 26_12-20",
    )
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--detector", default="hybrid")
    parser.add_argument(
        "--ocr-engine",
        default="auto",
        choices=["auto", "paddle_v4", "paddle_v5", "easyocr", "none"],
    )
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--ocr-top-k", type=int, default=3)
    parser.add_argument("--max-ocr-variants", type=int, default=2)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument("--adaptive", action="store_true", default=False)
    parser.add_argument(
        "--eval-fields", choices=["compact", "all"], default="compact"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--append-metrics", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_fields = (
        COMPACT_EVAL_FIELDS
        if args.eval_fields == "compact"
        else ALL_VALUE_FIELDS
    )
    items = find_labeled_videos(args.data_root, args.videos)
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in items:
        if not item.video_path.exists() or not item.csv_path.exists():
            missing.append(
                f"{item.name}: video={item.video_path} csv={item.csv_path}"
            )
            logger.warning("missing %s", missing[-1])
            continue
        logger.info("Processing %s", item.name)
        result = evaluate_video(item, args, eval_fields)
        if result:
            results.append(result)

    if not results:
        print("No labeled videos were evaluated. Missing expected files:")
        for line in missing:
            print(f"  - {line}")
        if args.json_out:
            args.json_out.write_text(
                json.dumps(
                    {
                        "rows": 0,
                        "videos": [],
                        "missing": missing,
                        "overall": {
                            "n_gt": 0,
                            "n_pred": 0,
                            "n_matched": 0,
                            "detection_recall": None,
                            "metric_80": None,
                            "avg_field": None,
                            "barcode_count": 0,
                            "qr_barcode_count": 0,
                            "fill_rates": fill_rates(pd.DataFrame()),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return

    summary = summarize(results, eval_fields)
    print_summary(summary)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.append_metrics:
        append_metrics_md(summary, args)


if __name__ == "__main__":
    main()
