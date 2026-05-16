"""GT-guided ceiling evaluation for OCR/QR post-processing.

DIAGNOSTIC ONLY — GT bboxes are used only for ceiling estimation, never during
production inference.  The script is robust to missing private videos: it prints
all missing paths and exits without appending misleading zero rows to METRICS.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from shelf.ocr.engine import OCREngine
from shelf.ocr.parser import parse_ocr_result
from shelf.ocr.preprocess import ocr_variants
from shelf.ocr.template import classify_color
from shelf.postproc.catalog import apply_catalog, load_catalog_from_env
from shelf.postproc.merge import merge
from shelf.postproc.pass80 import optimize_tag
from shelf.postproc.voting import merge_candidate_tags
from shelf.qr.decoder import decode_barcode, decode_qr, decode_qr_wechat_fast
from shelf.schema import ABSENT_VALUE, OUTPUT_COLUMNS, PriceTag

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path("Данные")
LABELED_NAMES = ["25_12-20", "25_2-10", "26_12-20", "43_15", "49_5"]

# Set SHELF_QR_MULTIFRAME_RADIUS_MS>0 to also try nearby frames for QR decode.
# Each offset adds one decode_qr_wechat_fast() call (~0.05-0.2s); 0 = disabled.
_QR_MF_RADIUS_MS = int(
    __import__("os").environ.get("SHELF_QR_MULTIFRAME_RADIUS_MS", "0")
)
_QR_MF_STEP_MS = int(
    __import__("os").environ.get("SHELF_QR_MULTIFRAME_STEP_MS", "50")
)
_QR_MULTIFRAME_OFFSETS_MS: list[int] = (
    [
        off
        for off in range(-_QR_MF_RADIUS_MS, _QR_MF_RADIUS_MS + 1, _QR_MF_STEP_MS)
        if off != 0
    ]
    if _QR_MF_RADIUS_MS > 0
    else []
)

QR_EVAL_FIELDS = {
    "qr_code_barcode",
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

EVAL_FIELDS_FULL = [
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

EVAL_FIELDS_NO_QR = [
    field for field in EVAL_FIELDS_FULL if field not in QR_EVAL_FIELDS
]

# Для обратной совместимости старого вывода field_accuracy
EVAL_FIELDS = EVAL_FIELDS_FULL


def labeled_paths(data_root: Path) -> list[tuple[str, Path, Path]]:
    """Return expected labeled video/CSV paths under ``data_root``."""
    return [
        (
            name,
            data_root / name / f"{name}.mp4",
            data_root / name / f"{name}.csv",
        )
        for name in LABELED_NAMES
    ]


def _normalize_gt(df: pd.DataFrame) -> pd.DataFrame:
    if "wholesale_level_1_coun" in df.columns:
        df = df.rename(
            columns={"wholesale_level_1_coun": "wholesale_level_1_count"}
        )
    if "filename" in df.columns:
        df["filename"] = df["filename"].astype(str).str.strip()
    for col in ["barcode", "qr_code_barcode"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm_bc)
    if "id_sku" in df.columns:
        df["id_sku"] = df["id_sku"].apply(
            lambda v: re.sub(r"\s+", "", str(v).strip()) if pd.notna(v) else ""
        )
    return df


def _norm_bc(val: Any) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    s = re.sub(r"\s+", "", s)
    try:
        if "." in s:
            s = str(int(float(s)))
        if s.isdigit() and len(s) < 13:
            s = s.zfill(13)
    except (ValueError, OverflowError):
        pass
    return s


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word tokens (case-insensitive)."""
    ta = set(re.findall(r"\w+", str(a).lower()))
    tb = set(re.findall(r"\w+", str(b).lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _field_match(pred_val: Any, gt_val: Any, field: str = "") -> bool:
    p = str(pred_val).strip().lower()
    g = str(gt_val).strip().lower()
    if g in (ABSENT_VALUE, "nan", ""):
        return True
    if p in ("", ABSENT_VALUE, "nan"):
        return False
    if field in {"barcode", "qr_code_barcode", "id_sku"}:
        return re.sub(r"\D", "", p) == re.sub(r"\D", "", g)
    p_num = re.sub(r"[,.]", ".", p.replace(" ", ""))
    g_num = re.sub(r"[,.]", ".", g.replace(" ", ""))
    try:
        return abs(float(p_num) - float(g_num)) < 1.5
    except ValueError:
        pass
    if p == g:
        return True
    if field in ("product_name", "additional_info") and len(g) > 10:
        return _token_overlap(p, g) >= 0.40
    return False


def _present(val: Any) -> bool:
    s = str(val or "").strip()
    return bool(s and s.lower() not in {"nan", "none"} and s != ABSENT_VALUE)


def _gt_has_qr(gt_row: pd.Series) -> bool:
    """Return True if GT row contains any real QR-related value."""
    for field in QR_EVAL_FIELDS:
        if field not in gt_row:
            continue
        value = str(gt_row.get(field, "")).strip()
        if (
            value
            and value.lower() not in {"nan", "none"}
            and value != ABSENT_VALUE
        ):
            return True
    return False


def _score_fields(
    pred: dict[str, Any],
    gt_row: pd.Series,
    fields: list[str],
) -> tuple[float, int]:
    """Return row score and number of correct fields for selected fields."""
    correct = 0
    for field in fields:
        if _field_match(
            pred.get(field, ""), gt_row.get(field, ""), field=field
        ):
            correct += 1
    return correct / max(1, len(fields)), correct


def _extract_one(
    frame: cv2.Mat, row: pd.Series, ocr: OCREngine, ocr_ru: OCREngine | None
) -> dict[str, Any]:
    """Run production OCR+QR parser on one GT bbox and return field values."""
    x1, y1, x2, y2 = (
        int(row.x_min),
        int(row.y_min),
        int(row.x_max),
        int(row.y_max),
    )
    h, w = frame.shape[:2]
    mx = max(5, int((x2 - x1) * 0.10))
    my = max(5, int((y2 - y1) * 0.10))
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    crop_raw = frame[y1:y2, x1:x2]
    if crop_raw.size == 0:
        return {}

    qr_fields = decode_qr(crop_raw)
    linear_barcode = decode_barcode(crop_raw)
    if linear_barcode and not qr_fields.get("barcode"):
        qr_fields["barcode"] = linear_barcode

    color = classify_color(crop_raw)
    candidate_tags = []
    for proc in (ocr_variants(crop_raw) or [crop_raw])[:3]:
        ocr_lines = ocr.run(proc)
        ocr_tag = parse_ocr_result(
            ocr_lines,
            crop=proc,
            crop_raw=crop_raw,
            filename=str(row.get("filename", "")),
            frame_timestamp=float(row.get("frame_timestamp", 0)),
            bbox=(x1, y1, x2, y2),
            color=color,
            ocr_ru=ocr_ru,
        )
        candidate_tags.append(merge(ocr_tag, qr_fields))
    merged = (
        merge_candidate_tags(candidate_tags)
        if candidate_tags
        else merge(
            parse_ocr_result(
                [],
                filename=str(row.get("filename", "")),
                frame_timestamp=float(row.get("frame_timestamp", 0)),
                bbox=(x1, y1, x2, y2),
                color=color,
            ),
            qr_fields,
        )
    )
    merged, _ = optimize_tag(merged, catalog=load_catalog_from_env())
    return merged.__dict__


def eval_video(
    name: str,
    video_path: Path,
    csv_path: Path,
    ocr: OCREngine,
    ocr_ru: OCREngine | None,
) -> dict[str, Any]:
    """Evaluate one video using GT bboxes."""
    gt_df = _normalize_gt(pd.read_csv(csv_path, decimal=","))
    cap = cv2.VideoCapture(str(video_path))

    scores_full: list[float] = []
    scores_no_qr: list[float] = []
    scores_no_qr_has_qr_gt: list[float] = []
    scores_no_qr_no_qr_gt: list[float] = []

    field_hits: dict[str, int] = {f: 0 for f in EVAL_FIELDS_FULL}
    field_hits_no_qr: dict[str, int] = {f: 0 for f in EVAL_FIELDS_NO_QR}
    fill_hits: dict[str, int] = {f: 0 for f in OUTPUT_COLUMNS}

    barcode_count = 0
    qr_barcode_count = 0
    gt_has_qr_count = 0
    gt_no_qr_count = 0

    for _, gt_row in gt_df.iterrows():
        ts_ms = float(gt_row.get("frame_timestamp", 0))
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ret, frame = cap.read()

        if not ret:
            scores_full.append(0.0)
            scores_no_qr.append(0.0)

            if _gt_has_qr(gt_row):
                gt_has_qr_count += 1
                scores_no_qr_has_qr_gt.append(0.0)
            else:
                gt_no_qr_count += 1
                scores_no_qr_no_qr_gt.append(0.0)
            continue

        pred = _extract_one(frame, gt_row, ocr, ocr_ru)

        # Multi-frame QR fallback: if QR not decoded from GT-timestamp frame,
        # try nearby offsets with a lightweight WeChatQR scan (~0.05-0.2s/frame).
        # Enabled via SHELF_QR_MULTIFRAME_RADIUS_MS env var (default 0 = off).
        if _QR_MULTIFRAME_OFFSETS_MS and not _present(pred.get("qr_code_barcode")):
            x1r = int(gt_row.x_min); y1r = int(gt_row.y_min)
            x2r = int(gt_row.x_max); y2r = int(gt_row.y_max)
            fh_r, fw_r = frame.shape[:2]
            mx_r = max(5, int((x2r - x1r) * 0.10))
            my_r = max(5, int((y2r - y1r) * 0.10))
            x1r = max(0, x1r - mx_r); y1r = max(0, y1r - my_r)
            x2r = min(fw_r, x2r + mx_r); y2r = min(fh_r, y2r + my_r)
            for off_ms in _QR_MULTIFRAME_OFFSETS_MS:
                alt_ts = ts_ms + off_ms
                if alt_ts < 0:
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, alt_ts)
                ok, alt_frame = cap.read()
                if not ok:
                    continue
                alt_fh, alt_fw = alt_frame.shape[:2]
                alt_crop = alt_frame[
                    y1r : min(alt_fh, y2r), x1r : min(alt_fw, x2r)
                ]
                if alt_crop.size == 0:
                    continue
                extra_qr = decode_qr_wechat_fast(alt_crop)
                if extra_qr:
                    for k, v in extra_qr.items():
                        if k not in pred or not _present(pred.get(k)):
                            pred[k] = v
                    # Re-run pass80 + catalog to propagate QR fields:
                    # price1_qr→price_default, price4_qr→price_card,
                    # barcode→id_sku+product_name via catalog, discount derivation.
                    _catalog = load_catalog_from_env()
                    _tag = PriceTag(**{k: pred.get(k, "") for k in PriceTag.__dataclass_fields__})
                    [_tag] = apply_catalog([_tag], _catalog)
                    _tag, _ = optimize_tag(_tag, catalog=_catalog)
                    pred = _tag.__dict__
                    logger.info("QR multiframe decode at offset %+dms", off_ms)
                    break
            # Restore cap position after multi-frame scan
            cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)

        barcode_count += int(_present(pred.get("barcode")))
        qr_barcode_count += int(_present(pred.get("qr_code_barcode")))

        for col in OUTPUT_COLUMNS:
            fill_hits[col] += int(_present(pred.get(col)))

        score_full, correct_full = _score_fields(
            pred,
            gt_row,
            EVAL_FIELDS_FULL,
        )
        scores_full.append(score_full)

        for field in EVAL_FIELDS_FULL:
            if _field_match(
                pred.get(field, ""), gt_row.get(field, ""), field=field
            ):
                field_hits[field] += 1

        score_no_qr, correct_no_qr = _score_fields(
            pred,
            gt_row,
            EVAL_FIELDS_NO_QR,
        )
        scores_no_qr.append(score_no_qr)

        for field in EVAL_FIELDS_NO_QR:
            if _field_match(
                pred.get(field, ""), gt_row.get(field, ""), field=field
            ):
                field_hits_no_qr[field] += 1

        if _gt_has_qr(gt_row):
            gt_has_qr_count += 1
            scores_no_qr_has_qr_gt.append(score_no_qr)
        else:
            gt_no_qr_count += 1
            scores_no_qr_no_qr_gt.append(score_no_qr)

    cap.release()

    n_total = len(gt_df)

    n_pass_full = sum(1 for s in scores_full if s >= 0.80)
    n_pass_no_qr = sum(1 for s in scores_no_qr if s >= 0.80)
    n_pass_no_qr_has_qr_gt = sum(1 for s in scores_no_qr_has_qr_gt if s >= 0.80)
    n_pass_no_qr_no_qr_gt = sum(1 for s in scores_no_qr_no_qr_gt if s >= 0.80)

    return {
        "video": name,
        "n_gt": n_total,
        # Старый full-вариант
        "n_pass": n_pass_full,
        "metric": n_pass_full / max(1, n_total),
        "avg_field": sum(scores_full) / max(1, len(scores_full)),
        # Новый no-QR-вариант
        "n_pass_no_qr": n_pass_no_qr,
        "metric_no_qr": n_pass_no_qr / max(1, n_total),
        "avg_field_no_qr": sum(scores_no_qr) / max(1, len(scores_no_qr)),
        # no-QR по группам GT
        "gt_has_qr_count": gt_has_qr_count,
        "gt_no_qr_count": gt_no_qr_count,
        "n_pass_no_qr_has_qr_gt": n_pass_no_qr_has_qr_gt,
        "metric_no_qr_has_qr_gt": n_pass_no_qr_has_qr_gt
        / max(1, gt_has_qr_count),
        "avg_field_no_qr_has_qr_gt": (
            sum(scores_no_qr_has_qr_gt) / max(1, len(scores_no_qr_has_qr_gt))
        ),
        "n_pass_no_qr_no_qr_gt": n_pass_no_qr_no_qr_gt,
        "metric_no_qr_no_qr_gt": n_pass_no_qr_no_qr_gt / max(1, gt_no_qr_count),
        "avg_field_no_qr_no_qr_gt": (
            sum(scores_no_qr_no_qr_gt) / max(1, len(scores_no_qr_no_qr_gt))
        ),
        "barcode_count": barcode_count,
        "qr_barcode_count": qr_barcode_count,
        "field_accuracy": {
            f: field_hits[f] / max(1, n_total) for f in EVAL_FIELDS_FULL
        },
        "field_accuracy_no_qr": {
            f: field_hits_no_qr[f] / max(1, n_total) for f in EVAL_FIELDS_NO_QR
        },
        "fill_rates": {
            f: fill_hits[f] / max(1, n_total) for f in OUTPUT_COLUMNS
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ceiling metrics."""
    total_gt = sum(r["n_gt"] for r in results)
    total_pass = sum(r["n_pass"] for r in results)
    total_pass_no_qr = sum(r["n_pass_no_qr"] for r in results)

    total_gt_has_qr = sum(r["gt_has_qr_count"] for r in results)
    total_gt_no_qr = sum(r["gt_no_qr_count"] for r in results)

    total_pass_no_qr_has_qr_gt = sum(
        r["n_pass_no_qr_has_qr_gt"] for r in results
    )
    total_pass_no_qr_no_qr_gt = sum(r["n_pass_no_qr_no_qr_gt"] for r in results)

    return {
        "videos": results,
        "overall": {
            "n_gt": total_gt,
            # Full metric
            "metric_80": total_pass / max(1, total_gt),
            "avg_field": sum(r["avg_field"] * r["n_gt"] for r in results)
            / max(1, total_gt),
            "n_pass": total_pass,
            # No-QR metric
            "metric_80_no_qr": total_pass_no_qr / max(1, total_gt),
            "avg_field_no_qr": sum(
                r["avg_field_no_qr"] * r["n_gt"] for r in results
            )
            / max(1, total_gt),
            "n_pass_no_qr": total_pass_no_qr,
            # No-QR metric for GT rows with QR
            "gt_has_qr_count": total_gt_has_qr,
            "metric_80_no_qr_has_qr_gt": total_pass_no_qr_has_qr_gt
            / max(1, total_gt_has_qr),
            "avg_field_no_qr_has_qr_gt": sum(
                r["avg_field_no_qr_has_qr_gt"] * r["gt_has_qr_count"]
                for r in results
            )
            / max(1, total_gt_has_qr),
            "n_pass_no_qr_has_qr_gt": total_pass_no_qr_has_qr_gt,
            # No-QR metric for GT rows without QR
            "gt_no_qr_count": total_gt_no_qr,
            "metric_80_no_qr_no_qr_gt": total_pass_no_qr_no_qr_gt
            / max(1, total_gt_no_qr),
            "avg_field_no_qr_no_qr_gt": sum(
                r["avg_field_no_qr_no_qr_gt"] * r["gt_no_qr_count"]
                for r in results
            )
            / max(1, total_gt_no_qr),
            "n_pass_no_qr_no_qr_gt": total_pass_no_qr_no_qr_gt,
            "barcode_count": sum(r["barcode_count"] for r in results),
            "qr_barcode_count": sum(r["qr_barcode_count"] for r in results),
            "field_accuracy": {
                f: sum(r["field_accuracy"][f] * r["n_gt"] for r in results)
                / max(1, total_gt)
                for f in EVAL_FIELDS_FULL
            },
            "field_accuracy_no_qr": {
                f: sum(
                    r["field_accuracy_no_qr"][f] * r["n_gt"] for r in results
                )
                / max(1, total_gt)
                for f in EVAL_FIELDS_NO_QR
            },
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable ceiling report."""
    print("\n=== Ceiling (GT bboxes, production OCR) ===")

    for r in summary["videos"]:
        print(
            f"  {r['video']:15s}: "
            f"full={r['metric']:.3f} avg={r['avg_field']:.3f} "
            f"pass={r['n_pass']}/{r['n_gt']} | "
            f"no_qr={r['metric_no_qr']:.3f} avg_no_qr={r['avg_field_no_qr']:.3f} "
            f"pass_no_qr={r['n_pass_no_qr']}/{r['n_gt']} | "
            f"bc={r['barcode_count']} qr_bc={r['qr_barcode_count']}"
        )

    o = summary["overall"]

    print("\n=== OVERALL ===")
    print(
        f"  FULL      metric@80%={o['metric_80']:.3f}  "
        f"avg_field={o['avg_field']:.3f}  "
        f"pass={o['n_pass']}/{o['n_gt']}  "
        f"bc={o['barcode_count']} qr_bc={o['qr_barcode_count']}"
    )

    print(
        f"  NO-QR     metric@80%={o['metric_80_no_qr']:.3f}  "
        f"avg_field={o['avg_field_no_qr']:.3f}  "
        f"pass={o['n_pass_no_qr']}/{o['n_gt']}"
    )

    print("\n=== NO-QR metric by GT QR availability ===")
    print(
        f"  GT has QR : metric@80%={o['metric_80_no_qr_has_qr_gt']:.3f}  "
        f"avg_field={o['avg_field_no_qr_has_qr_gt']:.3f}  "
        f"pass={o['n_pass_no_qr_has_qr_gt']}/{o['gt_has_qr_count']}"
    )
    print(
        f"  GT no QR  : metric@80%={o['metric_80_no_qr_no_qr_gt']:.3f}  "
        f"avg_field={o['avg_field_no_qr_no_qr_gt']:.3f}  "
        f"pass={o['n_pass_no_qr_no_qr_gt']}/{o['gt_no_qr_count']}"
    )

    print("\n=== Точность по полям FULL (weighted avg) ===")
    for field, acc in sorted(o["field_accuracy"].items(), key=lambda kv: kv[1]):
        bar = "█" * int(acc * 20)
        print(f"  {field:<28} {acc:>7.3f}  {bar}")

    print("\n=== Точность по полям NO-QR (weighted avg) ===")
    for field, acc in sorted(
        o["field_accuracy_no_qr"].items(), key=lambda kv: kv[1]
    ):
        bar = "█" * int(acc * 20)
        print(f"  {field:<28} {acc:>7.3f}  {bar}")


def append_metrics_md(summary: dict[str, Any]) -> None:
    """Append a metrics row only when at least one video was evaluated."""
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
    per_video = "  ".join(
        f"{r['video']}={r['metric']:.3f}" for r in summary["videos"]
    )
    with open("docs/METRICS.md", "a", encoding="utf-8") as f:
        f.write(
            f"| {today} | {commit} | ceiling (GT bboxes) | {o['metric_80']:.3f} | "
            f"avg_field={o['avg_field']:.3f}; "
            f"no_qr={o['metric_80_no_qr']:.3f}; "
            f"avg_no_qr={o['avg_field_no_qr']:.3f}; "
            f"has_qr_no_qr={o['metric_80_no_qr_has_qr_gt']:.3f}; "
            f"no_qr_gt_no_qr={o['metric_80_no_qr_no_qr_gt']:.3f}; "
            f"bc={o['barcode_count']}; qr_bc={o['qr_barcode_count']}; "
            f"{per_video} |\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR/QR ceiling with GT bboxes"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--ocr-engine",
        default="auto",
        choices=["auto", "paddle_v4", "paddle_v5", "easyocr", "none"],
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--append-metrics", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ocr = OCREngine(engine=args.ocr_engine)
    ocr_ru = OCREngine(lang="ru", force_easyocr=True)
    results: list[dict[str, Any]] = []
    missing: list[str] = []

    for name, video_path, csv_path in labeled_paths(args.data_root):
        if not video_path.exists() or not csv_path.exists():
            missing.append(f"{name}: video={video_path} csv={csv_path}")
            logger.warning(
                "Пропуск %s — video_exists=%s csv_exists=%s",
                name,
                video_path.exists(),
                csv_path.exists(),
            )
            continue
        logger.warning("Обрабатываем %s...", name)
        results.append(eval_video(name, video_path, csv_path, ocr, ocr_ru))

    if not results:
        print("No ceiling videos were evaluated. Missing expected files:")
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
                            "metric_80": None,
                            "avg_field": None,
                            "barcode_count": 0,
                            "qr_barcode_count": 0,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return

    summary = summarize(results)
    print_summary(summary)
    if args.json_out:
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.append_metrics:
        append_metrics_md(summary)
        print("\n  Метрика записана в docs/METRICS.md")


if __name__ == "__main__":
    main()
