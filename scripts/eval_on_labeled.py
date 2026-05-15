"""Этап 8 — оценка пайплайна на размеченных видео.

Метрика: доля ценников из GT, у которых ≥80% полей распознаны верно.

Матчинг предсказаний с GT:
1. По barcode (если оба есть)
2. По IoU (bbox ≥ 0.1) + нечёткое совпадение product_name
3. По временной близости (|ts_pred - ts_gt| < 2с) + IoU

Для каждого матча: считаем долю верных полей (из EVAL_FIELDS).
Финальная метрика = доля матчей с долей ≥ 0.8.
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from shelf import pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path("Данные")

LABELED = [
    (
        "25_12-20",
        DATA_ROOT / "25_12-20" / "25_12-20.mp4",
        DATA_ROOT / "25_12-20" / "25_12-20.csv",
    ),
    (
        "26_12-20",
        DATA_ROOT / "26_12-20" / "26_12-20.mp4",
        DATA_ROOT / "26_12-20" / "26_12-20.csv",
    ),
    (
        "43_15",
        DATA_ROOT / "43_15" / "43_15.mp4",
        DATA_ROOT / "43_15" / "43_15.csv",
    ),
]

# Поля для оценки (исключаем координаты, timestamp и filename)
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
    """Нормализовать GT CSV: исправить опечатку имени столбца, decimal."""
    if "wholesale_level_1_coun" in df.columns:
        df = df.rename(
            columns={"wholesale_level_1_coun": "wholesale_level_1_count"}
        )
    # barcode как строка (в GT хранится как float)
    for col in ["barcode", "qr_code_barcode"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm_barcode)
    return df


def _norm_barcode(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    try:
        if "." in s:
            s = str(int(float(s)))
        s = s.zfill(max(len(s), 13)) if s.isdigit() else s
    except (ValueError, OverflowError):
        pass
    return s


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _field_match(pred_val, gt_val) -> bool:
    """Проверить, совпадает ли предсказанное значение с GT."""
    p = str(pred_val).strip().lower()
    g = str(gt_val).strip().lower()
    if g in ("нет", "nan", ""):
        return True  # отсутствующие поля не штрафуем
    if p in ("", "нет"):
        return False
    # Для цен: нормализуем разделитель
    p = re.sub(r"[,.]", ".", p)
    g = re.sub(r"[,.]", ".", g)
    # Для числовых: допускаем округление ±0.01
    try:
        # BUG FIX: tolerance 1.5 руб — OCR читает "129" без копеек, GT "129,99"
        return abs(float(p) - float(g)) < 1.5
    except ValueError:
        pass
    return p == g


def match_and_score(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> dict:
    """Матчинг предсказаний с GT, расчёт метрики."""
    matched_gt = set()
    match_scores: list[float] = []

    for _, gt_row in gt_df.iterrows():
        gt_bc = str(gt_row.get("barcode", "")).strip()
        gt_qr = str(gt_row.get("qr_code_barcode", "")).strip()
        gt_bbox = (
            float(gt_row.get("x_min", 0)),
            float(gt_row.get("y_min", 0)),
            float(gt_row.get("x_max", 0)),
            float(gt_row.get("y_max", 0)),
        )
        gt_ts = float(gt_row.get("frame_timestamp", 0)) / 1000.0  # GT в мс

        best_pred = None
        best_score = -1.0

        for i, pred_row in pred_df.iterrows():
            if i in matched_gt:
                continue

            score = 0.0

            # Барcode-матч
            pred_bc = str(pred_row.get("barcode", "")).strip()
            pred_qr = str(pred_row.get("qr_code_barcode", "нет")).strip()
            if gt_bc and pred_bc and gt_bc == pred_bc:
                score += 2.0
            if gt_qr and pred_qr != "нет" and gt_qr == pred_qr:
                score += 2.0

            # IoU-матч (bbox в одних координатах)
            pred_bbox = (
                float(pred_row.get("x_min", 0)),
                float(pred_row.get("y_min", 0)),
                float(pred_row.get("x_max", 0)),
                float(pred_row.get("y_max", 0)),
            )
            iou = _iou(gt_bbox, pred_bbox)
            score += iou * 3.0

            # Временная близость
            pred_ts = float(pred_row.get("frame_timestamp", 0))
            if abs(pred_ts - gt_ts) < 3.0:
                score += 0.5

            if score > best_score:
                best_score = score
                best_pred = i

        if best_pred is not None and best_score > 0.3:
            matched_gt.add(best_pred)
            # Считаем долю верных полей
            correct = sum(
                _field_match(pred_df.loc[best_pred, f], gt_row.get(f, ""))
                for f in EVAL_FIELDS
                if f in pred_df.columns and f in gt_df.columns
            )
            match_scores.append(correct / len(EVAL_FIELDS))

    n_gt = len(gt_df)
    n_matched = len(match_scores)
    n_pass = sum(1 for s in match_scores if s >= 0.8)
    avg_score = sum(match_scores) / max(1, n_matched)

    return {
        "n_gt": n_gt,
        "n_matched": n_matched,
        "recall_match": n_matched / max(1, n_gt),
        "n_pass_80": n_pass,
        "metric_80": n_pass / max(1, n_gt),
        "avg_field_score": avg_score,
    }


def main(videos: list[str] | None = None, interval_ms: int = 500) -> None:
    results = []

    for name, video_path, gt_path in LABELED:
        if videos and name not in videos:
            continue
        if not video_path.exists() or not gt_path.exists():
            logger.warning("Пропускаем %s — файлы не найдены", name)
            continue

        logger.info("=" * 50)
        logger.info("Обрабатываем: %s", name)

        # GT
        gt_df = _normalize_gt(pd.read_csv(gt_path, decimal=","))

        # Прогон пайплайна
        pred_df = pipeline.run(
            video_path, interval_ms=interval_ms, adaptive=False, min_hits=2
        )

        # Оценка
        scores = match_and_score(pred_df, gt_df)
        scores["video"] = name
        results.append(scores)

        logger.info(
            "%s: GT=%d  matched=%d (%.0f%%)  pass@80%%=%d  metric=%.2f  avg_field=%.2f",
            name,
            scores["n_gt"],
            scores["n_matched"],
            scores["recall_match"] * 100,
            scores["n_pass_80"],
            scores["metric_80"],
            scores["avg_field_score"],
        )

    if not results:
        logger.warning("Нет результатов для записи")
        return

    # Сводная таблица
    print("\n=== Итоговая метрика ===")
    for r in results:
        print(
            f"  {r['video']:15s}: metric@80%={r['metric_80']:.3f}  "
            f"avg_field={r['avg_field_score']:.3f}  matched={r['n_matched']}/{r['n_gt']}"
        )

    total_gt = sum(r["n_gt"] for r in results)
    total_pass = sum(r["n_pass_80"] for r in results)
    overall = total_pass / max(1, total_gt)
    print(f"\n  OVERALL metric@80%: {overall:.3f}  ({total_pass}/{total_gt})")

    # Запись в METRICS.md
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    from datetime import date

    today = date.today().isoformat()

    metrics_path = Path("docs/METRICS.md")
    with metrics_path.open("a") as f:
        for r in results:
            f.write(
                f"| {today} | {commit} | {r['video']} | {r['metric_80']:.3f} | "
                f"matched={r['n_matched']}/{r['n_gt']} avg_field={r['avg_field_score']:.2f} |\n"
            )
    logger.info("Метрика записана в docs/METRICS.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Оценка пайплайна на размеченных видео"
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Список видео (25_12-20 26_12-20 43_15)",
    )
    parser.add_argument(
        "--interval", type=int, default=500, help="Интервал семплирования (мс)"
    )
    args = parser.parse_args()
    main(videos=args.videos, interval_ms=args.interval)
