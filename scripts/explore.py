"""Этап 1 — разведочный анализ данных.

Выводит:
- параметры всех видео (FPS, длительность, разрешение, число кадров)
- статистику GT-CSV (строк, заполненность полей)
"""

import logging
from pathlib import Path

import cv2
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path("Данные")

LABELED = [
    DATA_ROOT / "25_12-20" / "25_12-20.mp4",
    DATA_ROOT / "26_12-20" / "26_12-20.mp4",
    DATA_ROOT / "43_15" / "43_15.mp4",
]
UNLABELED = sorted((DATA_ROOT / "Unlabeled").glob("*.mp4"))

GT_CSVS = [
    DATA_ROOT / "25_12-20" / "25_12-20.csv",
    DATA_ROOT / "26_12-20" / "26_12-20.csv",
    DATA_ROOT / "43_15" / "43_15.csv",
]


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps else 0
    cap.release()
    return {
        "path": path.name,
        "fps": round(fps, 2),
        "frames": total,
        "duration_s": round(duration, 1),
        "resolution": f"{w}x{h}",
    }


def csv_stats(path: Path) -> dict:
    df = pd.read_csv(path)
    rows = len(df)
    # Считаем долю «нет» и NaN для каждого поля
    field_stats = {}
    for col in df.columns:
        n_absent = (df[col].astype(str).str.strip() == "нет").sum()
        n_nan = df[col].isna().sum()
        n_empty = (df[col].astype(str).str.strip() == "").sum()
        filled = rows - n_nan - n_absent - n_empty
        field_stats[col] = {
            "filled": filled,
            "absent_нет": int(n_absent),
            "nan": int(n_nan),
            "empty": int(n_empty),
        }
    return {"path": path.name, "rows": rows, "columns": list(df.columns), "fields": field_stats}


def main() -> None:
    print("\n" + "=" * 70)
    print("ВИДЕО — параметры")
    print("=" * 70)

    all_videos = LABELED + UNLABELED
    rows = []
    for p in all_videos:
        info = video_info(p)
        tag = "labeled" if p in LABELED else "unlabeled"
        info["tag"] = tag
        rows.append(info)
        print(
            f"  [{tag:10s}] {info['path']:25s}  {info['fps']:6.2f} fps  "
            f"{info['duration_s']:7.1f}s  {info['frames']:6d} кадров  {info['resolution']}"
        )

    print("\n" + "=" * 70)
    print("GT-CSV — структура и заполненность")
    print("=" * 70)

    for csv_path in GT_CSVS:
        stats = csv_stats(csv_path)
        print(f"\n--- {stats['path']} ({stats['rows']} строк) ---")
        # Колонки отличающиеся от нашей схемы
        from shelf.schema import OUTPUT_COLUMNS

        our_cols = set(OUTPUT_COLUMNS)
        gt_cols = set(stats["columns"])
        extra = gt_cols - our_cols
        missing = our_cols - gt_cols
        if extra:
            print(f"  ЛИШНИЕ столбцы в GT:    {sorted(extra)}")
        if missing:
            print(f"  ОТСУТСТВУЮТ в GT:       {sorted(missing)}")

        print(f"  {'Поле':<35} {'заполн':>7} {'нет':>7} {'NaN':>7} {'пусто':>7}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for col, s in stats["fields"].items():
            flag = " !" if s["nan"] > 0 else ""
            print(f"  {col:<35} {s['filled']:>7} {s['absent_нет']:>7} {s['nan']:>7} {s['empty']:>7}{flag}")

    print("\n" + "=" * 70)
    print("КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ")
    print("=" * 70)
    print(
        """
  1. В 26_12-20.csv и 43_15.csv: опечатка 'wholesale_level_1_coun'
     вместо 'wholesale_level_1_count'. Нужен alias при матчинге.

  2. Координаты боксов — вещественные числа (не int).

  3. barcode/qr_code_barcode в GT хранится как float (4.67e+12).
     Нужна нормализация: int → str с ведущим нулём (EAN-13, 13 цифр).

  4. frame_timestamp — секунды от начала видео (float).
     Для дедупликации: несколько кадров одного ценника → берём min timestamp.

  5. filename в GT: вида '25_12-20/2.mp4' (с папкой+номером кадра)
     или просто '26_12-20.mp4'. Не ссылается на оригинальный .mp4 напрямую.

  6. Поле 'price3_qr' чаще всего 'нет' — промежуточная цена редко.
     'action_price_qr' / 'action_code_qr' — почти всегда 'нет'.

  7. Доминирующий цвет: 'red' (акция МНЦ). Белых ценников мало.

  8. Unlabeled: три видео — 25_12-20, 26_12-20, 26_2-10.
     Первые два — повторные съёмки тех же стеллажей (другое время суток?).
"""
    )


if __name__ == "__main__":
    main()
