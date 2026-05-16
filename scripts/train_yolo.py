"""Этап 12.4 — Дообучение YOLOv8n на псевдо-лейблах ценников.

Использование:
    poetry run python scripts/train_yolo.py [--epochs 30] [--batch 8]
"""

import argparse
import logging
import subprocess
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_YAML = Path("data/pseudo/dataset.yaml")
MODELS_DIR = Path("models")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--name", default="pricetag_v1")
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        logger.error(
            "dataset.yaml не найден. Запусти сначала: scripts/extract_pseudolabels.py"
        )
        return

    MODELS_DIR.mkdir(exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(args.model)
    logger.info(
        "Начало обучения: %s  epochs=%d  batch=%d  imgsz=%d",
        args.model,
        args.epochs,
        args.batch,
        args.imgsz,
    )

    # results = model.train(
    model.train(
        data=str(DATASET_YAML.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=15,
        project="runs/detect",
        name=args.name,
        cache=False,
        augment=True,
        mosaic=0.5,
        degrees=10,  # ценники под углом
        perspective=0.001,  # перспектива от камеры робота
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,  # перевёрнутые ценники (180° mounting)
        fliplr=0.5,
        scale=0.3,
        translate=0.1,
        verbose=False,
        plots=False,
        save=True,
        exist_ok=True,
    )

    # Копируем лучшую модель
    best_pt = Path(f"runs/detect/{args.name}/weights/best.pt")
    if best_pt.exists():
        import shutil

        dest = MODELS_DIR / "pricetag_yolov8n.pt"
        shutil.copy2(best_pt, dest)
        logger.info("Лучшая модель: %s", dest)

    # Метрика
    try:
        metrics = model.val(data=str(DATASET_YAML.resolve()), verbose=False)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        logger.info("Val mAP50=%.3f  mAP50-95=%.3f", map50, map50_95)

        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        today = date.today().isoformat()
        with open("docs/METRICS.md", "a") as f:
            f.write(
                f"| {today} | {commit} | YOLO-train | — | "
                f"mAP50={map50:.3f} mAP50-95={map50_95:.3f} "
                f"epochs={args.epochs} batch={args.batch} |\n"
            )
        logger.info("Метрика записана в docs/METRICS.md")
    except Exception as exc:
        logger.warning("Не удалось записать метрику: %s", exc)


if __name__ == "__main__":
    main()
