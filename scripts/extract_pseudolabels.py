"""Этап 12.1 — Извлечение псевдо-лейблов для дообучения YOLOv8.

Два источника:
1. GT CSV (25_12-20, 26_12-20) — точные bbox от организаторов хакатона
2. MSER-трекер + расширение bbox ×1.3 — для unlabeled видео

Val: все детекции из 43_15 (GT CSV) — честная оценка.
Train: 25_12-20 + 26_12-20 (GT) + unlabeled (MSER, опционально).

Выходная структура:
    data/pseudo/
        images/{train,val}/  ← JPEG-кадры
        labels/{train,val}/  ← YOLO txt (class cx cy w h нормированные)
        high/                ← кадры с QR-верификацией
        mid/                 ← без QR (только форм-фактор)

Использование:
    poetry run python scripts/extract_pseudolabels.py [--dry-run] [--unlabeled]
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path("Данные")
OUT_ROOT = Path("data/pseudo")

# GT CSV для train (25_12-20, 26_12-20) и val (43_15)
GT_TRAIN = [
    (
        DATA_ROOT / "25_12-20" / "25_12-20.mp4",
        DATA_ROOT / "25_12-20" / "25_12-20.csv",
    ),
    (
        DATA_ROOT / "26_12-20" / "26_12-20.mp4",
        DATA_ROOT / "26_12-20" / "26_12-20.csv",
    ),
]
GT_VAL = [
    (DATA_ROOT / "43_15" / "43_15.mp4", DATA_ROOT / "43_15" / "43_15.csv"),
]
UNLABELED = sorted((DATA_ROOT / "Unlabeled").glob("*.mp4"))

# Расширение bbox на 30% для GT (ценники бывают слегка обрезаны)
BBOX_EXPAND = 1.25
# Расширение для MSER-псевдо-лейблов (более шумный источник)
MSER_EXPAND = 1.40

# Фильтры качества
MIN_AREA_FRAC = 0.003  # ≥ 0.3% площади кадра
MAX_AREA_FRAC = 0.20  # ≤ 20%
ASPECT_MIN = 0.3
ASPECT_MAX = 3.5


def _norm_barcode(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    try:
        if "." in s:
            s = str(int(float(s)))
    except (ValueError, OverflowError):
        pass
    return s


def _load_gt(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, decimal=",")
    if "wholesale_level_1_coun" in df.columns:
        df = df.rename(
            columns={"wholesale_level_1_coun": "wholesale_level_1_count"}
        )
    return df


def _bbox_to_yolo(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    frame_w: int,
    frame_h: int,
    expand: float = 1.0,
) -> tuple | None:
    """Перевести bbox в нормированный YOLO-формат с расширением."""
    bw = (x_max - x_min) * expand
    bh = (y_max - y_min) * expand
    cx = (x_min + x_max) / 2 / frame_w
    cy = (y_min + y_max) / 2 / frame_h
    nw = bw / frame_w
    nh = bh / frame_h
    # Clip в [0, 1]
    cx = max(0.01, min(0.99, cx))
    cy = max(0.01, min(0.99, cy))
    nw = min(nw, min(2 * cx, 2 * (1 - cx)))
    nh = min(nh, min(2 * cy, 2 * (1 - cy)))
    # Фильтры
    area_frac = nw * nh
    ar = nw / nh if nh > 0 else 0
    if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
        return None
    if not (ASPECT_MIN <= ar <= ASPECT_MAX):
        return None
    return cx, cy, nw, nh


def _try_qr(
    frame: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int
) -> bool:
    """Попробовать декодировать QR в bbox (верификация что это реальный ценник)."""
    from shelf.qr.decoder import decode_qr

    crop = frame[max(0, y_min) : y_max, max(0, x_min) : x_max]
    if crop.size == 0:
        return False
    for rot in [None, cv2.ROTATE_180]:
        img = cv2.rotate(crop, rot) if rot is not None else crop
        if decode_qr(img):
            return True
    return False


def _save_sample(
    frame: np.ndarray,
    stem: str,
    bboxes_yolo: list[tuple],
    split: str,
    high_quality: bool = False,
    dry_run: bool = False,
) -> int:
    """Сохранить кадр + label-файл. Возвращает 1 если успешно, 0 если пропущено."""
    if not bboxes_yolo:
        return 0

    out_img = OUT_ROOT / "images" / split / f"{stem}.jpg"
    out_lbl = OUT_ROOT / "labels" / split / f"{stem}.txt"

    if not dry_run:
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(out_lbl, "w") as f:
            for cx, cy, nw, nh in bboxes_yolo:
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        if high_quality:
            hq_dir = OUT_ROOT / "high" / split
            hq_dir.mkdir(parents=True, exist_ok=True)
            (hq_dir / f"{stem}.jpg").symlink_to(out_img.resolve())

    return 1


def extract_gt(
    video_path: Path,
    csv_path: Path,
    split: str,
    expand: float,
    max_per_ts: int = 1,
    dry_run: bool = False,
) -> dict:
    """Извлечь лейблы из GT CSV (точные аннотации хакатона)."""
    df = _load_gt(csv_path)
    video_stem = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamps = sorted(df.frame_timestamp.unique())
    saved_frames = 0
    saved_boxes = 0
    hq_count = 0

    for ts_ms in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_ms))
        ret, frame = cap.read()
        if not ret:
            logger.warning(
                "Не удалось прочитать кадр ts=%d в %s", ts_ms, video_stem
            )
            continue

        rows = df[df.frame_timestamp == ts_ms]
        bboxes = []
        any_qr = False

        for _, row in rows.iterrows():
            yolo = _bbox_to_yolo(
                row.x_min,
                row.y_min,
                row.x_max,
                row.y_max,
                frame_w,
                frame_h,
                expand=expand,
            )
            if yolo is None:
                continue
            bboxes.append(yolo)

            # QR-верификация для первых нескольких боксов
            if not any_qr:
                any_qr = _try_qr(
                    frame,
                    int(row.x_min),
                    int(row.y_min),
                    int(row.x_max),
                    int(row.y_max),
                )

        if not bboxes:
            continue

        stem = f"{video_stem}_{ts_ms}"
        n = _save_sample(
            frame, stem, bboxes, split, high_quality=any_qr, dry_run=dry_run
        )
        saved_frames += n
        saved_boxes += len(bboxes) * n
        if any_qr:
            hq_count += n

    cap.release()
    return {"frames": saved_frames, "boxes": saved_boxes, "hq": hq_count}


def extract_mser_pseudo(
    video_path: Path,
    split: str,
    expand: float,
    stride_ms: int,
    dry_run: bool = False,
) -> dict:
    """Извлечь псевдо-лейблы через MSER + ByteTrack для unlabeled видео."""
    from shelf.detect.detector import MSERDetector
    from shelf.detect.tracker import Tracker
    from shelf.io.video import sample_frames

    video_stem = video_path.stem
    detector = MSERDetector()
    tracker = Tracker(min_hits=3)

    for ts, frame in sample_frames(
        video_path, interval_ms=stride_ms, adaptive=True
    ):
        dets = detector.detect(frame)
        tracker.update(dets, frame, ts)

    best_crops = tracker.get_best_crops(min_hits=3)
    frame_h_orig, frame_w_orig = None, None

    cap = cv2.VideoCapture(str(video_path))
    frame_w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    saved_frames = 0
    saved_boxes = 0
    hq_count = 0
    seen_stems: set[str] = set()

    for tid, state in best_crops.items():
        if state.best_frame is None:
            continue
        d = state.best_det
        yolo = _bbox_to_yolo(
            d.x_min,
            d.y_min,
            d.x_max,
            d.y_max,
            frame_w_orig,
            frame_h_orig,
            expand=expand,
        )
        if yolo is None:
            continue

        # Загружаем полный кадр по timestamp
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, state.best_ts * 1000)
        ret, full_frame = cap.read()
        cap.release()
        if not ret:
            continue

        ts_key = f"{int(state.best_ts * 10)}"
        stem = f"{video_stem}_mser_{ts_key}_t{tid}"
        if stem in seen_stems:
            continue
        seen_stems.add(stem)

        any_qr = _try_qr(full_frame, d.x_min, d.y_min, d.x_max, d.y_max)
        n = _save_sample(
            full_frame,
            stem,
            [yolo],
            split,
            high_quality=any_qr,
            dry_run=dry_run,
        )
        saved_frames += n
        saved_boxes += n
        if any_qr:
            hq_count += n

    return {"frames": saved_frames, "boxes": saved_boxes, "hq": hq_count}


def write_dataset_yaml() -> None:
    """Записать dataset.yaml для Ultralytics."""
    yaml_path = OUT_ROOT / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: price_tag\n"
    )
    logger.info("Записан %s", yaml_path)


def log_metrics(stats: dict) -> None:
    """Дописать итоги в docs/METRICS.md."""
    import subprocess
    from datetime import date

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    today = date.today().isoformat()
    line = (
        f"| {today} | {commit} | pseudo-labels | — | "
        f"train={stats['train_boxes']}boxes/{stats['train_frames']}frames "
        f"val={stats['val_boxes']}boxes/{stats['val_frames']}frames "
        f"hq={stats['hq']} |\n"
    )
    with open("docs/METRICS.md", "a") as f:
        f.write(line)
    logger.info("Метрика записана")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Извлечь псевдо-лейблы для YOLO"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только подсчёт, без записи файлов",
    )
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="Добавить unlabeled видео через MSER",
    )
    parser.add_argument(
        "--stride", type=int, default=500, help="Интервал MSER (мс)"
    )
    parser.add_argument(
        "--expand", type=float, default=BBOX_EXPAND, help="Расширение bbox"
    )
    args = parser.parse_args()

    total = {
        "train_frames": 0,
        "train_boxes": 0,
        "val_frames": 0,
        "val_boxes": 0,
        "hq": 0,
    }

    # --- Train: GT из 25_12-20 и 26_12-20 ---
    for video_path, csv_path in GT_TRAIN:
        logger.info("GT train: %s", video_path.name)
        r = extract_gt(
            video_path,
            csv_path,
            "train",
            expand=args.expand,
            dry_run=args.dry_run,
        )
        logger.info(
            "  frames=%d  boxes=%d  hq=%d", r["frames"], r["boxes"], r["hq"]
        )
        total["train_frames"] += r["frames"]
        total["train_boxes"] += r["boxes"]
        total["hq"] += r["hq"]

    # --- Val: GT из 43_15 ---
    for video_path, csv_path in GT_VAL:
        logger.info("GT val: %s", video_path.name)
        r = extract_gt(
            video_path,
            csv_path,
            "val",
            expand=args.expand,
            dry_run=args.dry_run,
        )
        logger.info(
            "  frames=%d  boxes=%d  hq=%d", r["frames"], r["boxes"], r["hq"]
        )
        total["val_frames"] += r["frames"]
        total["val_boxes"] += r["boxes"]
        total["hq"] += r["hq"]

    # --- Опционально: MSER на unlabeled ---
    if args.unlabeled:
        for video_path in UNLABELED:
            logger.info("MSER pseudo: %s", video_path.name)
            r = extract_mser_pseudo(
                video_path,
                "train",
                expand=MSER_EXPAND,
                stride_ms=args.stride,
                dry_run=args.dry_run,
            )
            logger.info(
                "  frames=%d  boxes=%d  hq=%d", r["frames"], r["boxes"], r["hq"]
            )
            total["train_frames"] += r["frames"]
            total["train_boxes"] += r["boxes"]
            total["hq"] += r["hq"]

    if not args.dry_run:
        write_dataset_yaml()
        log_metrics(total)

    logger.info(
        "=== Итого: train=%d/%d  val=%d/%d  hq=%d ===",
        total["train_boxes"],
        total["train_frames"],
        total["val_boxes"],
        total["val_frames"],
        total["hq"],
    )


if __name__ == "__main__":
    main()
