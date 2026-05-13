"""Нарезка GT-кадров на тайлы 640×640 для tile-based YOLO.

Проблема full-frame YOLO: ценники ≈47px при resize 4K→640 → плохо детектируются.
Решение: режем 4K на тайлы 640×640 (stride 512, overlap 128).
Ценники в тайле: 180-380px → 28-59% от тайла → отлично обнаруживаются.

Из 24 train-кадров: ~120-240 тайлов с непустыми лейблами.
Val-тайлы: из 43_15 GT (2 кадра → ~10-20 тайлов).

Использование:
    poetry run python scripts/extract_tiles.py [--dry-run]
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
OUT_ROOT = Path("data/tiles")
PSEUDO_ROOT = Path("data/pseudo")

TILE_SIZE = 640
STRIDE = 512        # overlap = 640 - 512 = 128px (20%)
MIN_BBOX_IOU = 0.5  # минимальная доля пересечения bbox с тайлом


def _load_gt(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, decimal=",")
    if "wholesale_level_1_coun" in df.columns:
        df = df.rename(columns={"wholesale_level_1_coun": "wholesale_level_1_count"})
    return df


def _bbox_intersection(
    box: tuple[float, float, float, float],
    tile_x: int, tile_y: int,
) -> tuple[float, float, float, float] | None:
    """Пересечь bbox с тайлом. Вернуть (cx,cy,nw,nh) YOLO или None."""
    bx1, by1, bx2, by2 = box
    tx1, ty1 = tile_x, tile_y
    tx2, ty2 = tile_x + TILE_SIZE, tile_y + TILE_SIZE

    # Пересечение в абсолютных координатах
    ix1 = max(bx1, tx1)
    iy1 = max(by1, ty1)
    ix2 = min(bx2, tx2)
    iy2 = min(by2, ty2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    box_area = (bx2 - bx1) * (by2 - by1)
    if box_area <= 0:
        return None

    # Отбрасываем если в тайл попало < MIN_BBOX_IOU от bbox
    if inter_area / box_area < MIN_BBOX_IOU:
        return None

    # YOLO-координаты относительно тайла
    cx = (ix1 + ix2) / 2 - tile_x
    cy = (iy1 + iy2) / 2 - tile_y
    nw = ix2 - ix1
    nh = iy2 - iy1

    # Нормировать в [0,1]
    return (
        max(0.0, min(1.0, cx / TILE_SIZE)),
        max(0.0, min(1.0, cy / TILE_SIZE)),
        min(1.0, nw / TILE_SIZE),
        min(1.0, nh / TILE_SIZE),
    )


def process_frame(
    frame: np.ndarray,
    gt_rows: pd.DataFrame,
    stem: str,
    split: str,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Нарезать один кадр на тайлы, сохранить с GT-лейблами."""
    h, w = frame.shape[:2]
    img_dir = OUT_ROOT / "images" / split
    lbl_dir = OUT_ROOT / "labels" / split
    if not dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    n_tiles = 0
    n_boxes = 0

    # GT-боксы из кадра
    gt_bboxes = [
        (float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max))
        for _, row in gt_rows.iterrows()
    ]

    for ty in range(0, h - TILE_SIZE + 1, STRIDE):
        for tx in range(0, w - TILE_SIZE + 1, STRIDE):
            yolo_boxes = []
            for box in gt_bboxes:
                yolo = _bbox_intersection(box, tx, ty)
                if yolo is not None:
                    yolo_boxes.append(yolo)

            if not yolo_boxes:
                continue  # тайл без ценников — пропускаем

            tile_stem = f"{stem}_t{tx}_{ty}"
            if not dry_run:
                tile = frame[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
                cv2.imwrite(str(img_dir / f"{tile_stem}.jpg"), tile,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(lbl_dir / f"{tile_stem}.txt", "w") as f:
                    for cx, cy, nw, nh in yolo_boxes:
                        f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            n_tiles += 1
            n_boxes += len(yolo_boxes)

    return n_tiles, n_boxes


def extract_from_csv(
    video_path: Path, csv_path: Path, split: str, dry_run: bool = False
) -> dict:
    """Извлечь тайлы для всех GT-временных меток видео."""
    gt_df = _load_gt(csv_path)
    video_stem = video_path.stem
    cap = cv2.VideoCapture(str(video_path))

    total_tiles, total_boxes = 0, 0

    for ts_ms in sorted(gt_df.frame_timestamp.unique()):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_ms))
        ret, frame = cap.read()
        if not ret:
            continue

        rows = gt_df[gt_df.frame_timestamp == ts_ms]
        stem = f"{video_stem}_{int(ts_ms)}"
        t, b = process_frame(frame, rows, stem, split, dry_run)
        total_tiles += t
        total_boxes += b

    cap.release()
    return {"tiles": total_tiles, "boxes": total_boxes}


def write_dataset_yaml() -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats: dict[str, dict] = {}

    # --- Train: 25_12-20 и 26_12-20 ---
    for name in ["25_12-20", "26_12-20"]:
        video = DATA_ROOT / name / f"{name}.mp4"
        csv = DATA_ROOT / name / f"{name}.csv"
        if not video.exists():
            continue
        logger.info("Train tiles: %s", name)
        r = extract_from_csv(video, csv, "train", dry_run=args.dry_run)
        stats[name] = r
        logger.info("  tiles=%d  boxes=%d", r["tiles"], r["boxes"])

    # --- Val: 43_15 ---
    val_video = DATA_ROOT / "43_15" / "43_15.mp4"
    val_csv = DATA_ROOT / "43_15" / "43_15.csv"
    if val_video.exists():
        logger.info("Val tiles: 43_15")
        r = extract_from_csv(val_video, val_csv, "val", dry_run=args.dry_run)
        stats["43_15_val"] = r
        logger.info("  tiles=%d  boxes=%d", r["tiles"], r["boxes"])

    if not args.dry_run:
        write_dataset_yaml()

    total_train = sum(stats[k]["tiles"] for k in stats if k != "43_15_val")
    total_val = stats.get("43_15_val", {}).get("tiles", 0)
    logger.info("=== Итого: train_tiles=%d  val_tiles=%d ===", total_train, total_val)


if __name__ == "__main__":
    main()
