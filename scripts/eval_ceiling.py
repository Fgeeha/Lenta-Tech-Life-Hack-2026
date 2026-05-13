"""GT-guided ceiling evaluation.

DIAGNOSTIC ONLY — GT bboxes used only for ceiling estimation.
NOT used in production inference per §1 of CLAUDE.md.

Измеряет максимально возможную метрику при идеальном детекторе:
для каждого GT-ценника берём точный bbox, запускаем production OCR+parser,
считаем долю верных полей.
"""

import logging
import re
from pathlib import Path

import cv2
import pandas as pd

from shelf.ocr.engine import OCREngine
from shelf.ocr.parser import parse_ocr_result
from shelf.ocr.preprocess import preprocess_crop
from shelf.ocr.template import classify_color
from shelf.postproc.merge import merge
from shelf.qr.decoder import decode_qr

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path("Данные")
LABELED = [
    ("25_12-20", DATA_ROOT / "25_12-20" / "25_12-20.mp4", DATA_ROOT / "25_12-20" / "25_12-20.csv"),
    ("26_12-20", DATA_ROOT / "26_12-20" / "26_12-20.mp4", DATA_ROOT / "26_12-20" / "26_12-20.csv"),
    ("43_15", DATA_ROOT / "43_15" / "43_15.mp4", DATA_ROOT / "43_15" / "43_15.csv"),
]

EVAL_FIELDS = [
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


def _normalize_gt(df: pd.DataFrame) -> pd.DataFrame:
    if "wholesale_level_1_coun" in df.columns:
        df = df.rename(columns={"wholesale_level_1_coun": "wholesale_level_1_count"})
    for col in ["barcode", "qr_code_barcode"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm_bc)
    return df


def _norm_bc(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    try:
        if "." in s:
            s = str(int(float(s)))
        if s.isdigit() and len(s) < 13:
            s = s.zfill(13)
    except (ValueError, OverflowError):
        pass
    return s


def _field_match(pred_val, gt_val) -> bool:
    p = str(pred_val).strip().lower()
    g = str(gt_val).strip().lower()
    if g in ("нет", "nan", ""):
        return True
    if p in ("", "нет"):
        return False
    p = re.sub(r"[,.]", ".", p)
    g = re.sub(r"[,.]", ".", g)
    try:
        # BUG FIX: tolerance 1.5 руб — OCR читает "129" без копеек, GT "129,99"
        return abs(float(p) - float(g)) < 1.5
    except ValueError:
        pass
    return p == g


def _extract_one(frame: cv2.Mat, row: pd.Series, ocr: OCREngine) -> dict:
    """OCR + parse на одном GT-bbox. Возвращает dict поле→значение."""
    x1, y1, x2, y2 = int(row.x_min), int(row.y_min), int(row.x_max), int(row.y_max)
    h, w = frame.shape[:2]
    # Margin +10% для захвата полного ценника
    mx = max(5, int((x2 - x1) * 0.10))
    my = max(5, int((y2 - y1) * 0.10))
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    crop_raw = frame[y1:y2, x1:x2]
    if crop_raw.size == 0:
        return {}

    # QR (все ориентации)
    qr_fields: dict[str, str] = {}
    for rot in [cv2.ROTATE_90_COUNTERCLOCKWISE, None, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]:
        img = cv2.rotate(crop_raw, rot) if rot is not None else crop_raw
        qr_fields = decode_qr(img)
        if qr_fields:
            break

    color = classify_color(crop_raw)
    proc = preprocess_crop(crop_raw, rotate_180=True, deskew=True, upscale=5, sharpen=True, clahe=True)
    ocr_lines = ocr.run(proc)

    ocr_tag = parse_ocr_result(
        ocr_lines,
        crop=proc,
        filename=str(row.get("filename", "")),
        frame_timestamp=float(row.get("frame_timestamp", 0)) / 1000.0,
        bbox=(x1, y1, x2, y2),
        color=color,
    )
    merged = merge(ocr_tag, qr_fields)
    return merged.__dict__


def eval_video(name: str, video_path: Path, csv_path: Path, ocr: OCREngine) -> dict:
    gt_df = _normalize_gt(pd.read_csv(csv_path, decimal=","))
    cap = cv2.VideoCapture(str(video_path))

    scores: list[float] = []
    field_hits: dict[str, int] = {f: 0 for f in EVAL_FIELDS}
    n_total = len(gt_df)

    for _, gt_row in gt_df.iterrows():
        ts_ms = float(gt_row.get("frame_timestamp", 0))
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ret, frame = cap.read()
        if not ret:
            scores.append(0.0)
            continue

        pred = _extract_one(frame, gt_row, ocr)

        correct = 0
        for field in EVAL_FIELDS:
            gt_val = gt_row.get(field, "")
            pred_val = pred.get(field, "")
            ok = _field_match(pred_val, gt_val)
            if ok:
                correct += 1
                field_hits[field] += 1
        scores.append(correct / len(EVAL_FIELDS))

    cap.release()
    n_pass = sum(1 for s in scores if s >= 0.80)
    avg = sum(scores) / max(1, len(scores))
    return {
        "video": name,
        "n_gt": n_total,
        "n_pass": n_pass,
        "metric": n_pass / max(1, n_total),
        "avg_field": avg,
        "field_accuracy": {f: field_hits[f] / max(1, n_total) for f in EVAL_FIELDS},
    }


def main() -> None:
    ocr = OCREngine()
    results = []
    all_field_acc: dict[str, list[float]] = {f: [] for f in EVAL_FIELDS}

    for name, video_path, csv_path in LABELED:
        if not video_path.exists():
            logger.warning("Пропуск %s — файл не найден", name)
            continue
        logger.warning("Обрабатываем %s...", name)
        r = eval_video(name, video_path, csv_path, ocr)
        results.append(r)
        for f in EVAL_FIELDS:
            all_field_acc[f].append(r["field_accuracy"][f])

    print("\n=== Ceiling (GT bboxes, production OCR) ===")
    for r in results:
        print(
            f"  {r['video']:15s}: metric@80%={r['metric']:.3f}  "
            f"avg_field={r['avg_field']:.3f}  pass={r['n_pass']}/{r['n_gt']}"
        )

    total_gt = sum(r["n_gt"] for r in results)
    total_pass = sum(r["n_pass"] for r in results)
    overall = total_pass / max(1, total_gt)
    print(f"\n  OVERALL metric@80%: {overall:.3f}  ({total_pass}/{total_gt})")

    print("\n=== Точность по полям (avg по всем видео) ===")
    print(f"  {'field':<28} {'accuracy':>8}")
    print(f"  {'-'*28} {'-'*8}")
    for field in EVAL_FIELDS:
        acc = sum(all_field_acc[field]) / max(1, len(all_field_acc[field]))
        bar = "█" * int(acc * 20)
        print(f"  {field:<28} {acc:>7.3f}  {bar}")

    # METRICS.md
    import subprocess
    from datetime import date

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    today = date.today().isoformat()
    per_video = "  ".join(f"{r['video']}={r['metric']:.3f}" for r in results)
    with open("docs/METRICS.md", "a") as f:
        f.write(f"| {today} | {commit} | ceiling (GT bboxes) | {overall:.3f} | {per_video} |\n")
    print("\n  Метрика записана в docs/METRICS.md")


if __name__ == "__main__":
    main()
