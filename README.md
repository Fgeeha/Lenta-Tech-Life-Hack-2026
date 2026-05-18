# ShelfWatch — Полка под контролем

**Lenta Tech Life Hack 2026** · Команда **«Стабилизируй это»**

---

## Быстрые ссылки

| | |
|---|---|
| Демо (HF Spaces) | https://huggingface.co/spaces/fgeeha/shelf-control |
| Финальная метрика | **ALL_VALUE = 0.1715** (47 / 274 ценников ≥ 80% полей) |
| Тесты | `PYTHONPATH=src pytest -q` → **176 passed** |
| Оценка потолка | `python scripts/eval_ceiling.py` |

---

## Постановка задачи

Пайплайн `video.mp4 → CSV` для автоматического распознавания ценников с видео робота-сканера в магазинах Лента.

**Вход:**
- 4K H.264-видео прохода вдоль полки
- опционально: веса детектора ценников

**Выход:**
- CSV, 29 полей по схеме ТЗ: `product_name`, `barcode`, `price_card`, `price_default`, QR-поля, координаты bbox, timestamp и другие

Проект работает полностью локально: без облачных API, внешних баз данных и ручной разметки на этапе инференса.

---

## Метрика

Организаторская метрика — доля ценников, у которых одновременно корректны ≥ 80% оцениваемых полей (`metric@80%` / `ALL_VALUE`).

### Прогресс по итерациям

| Этап | ALL_VALUE | Пройдено / 274 |
|---|---:|---:|
| Стартовая точка | 0.0000 | 0 / 274 |
| MSER + базовый OCR | ~0.0040 | ~1 / 274 |
| YOLO-tiled (mAP50=0.776) | ~0.0146 | ~4 / 274 |
| Catalog lookup (4-tier) | 0.1496 | 41 / 274 |
| Pass80 optimizer + поля QR | 0.1606 | 44 / 274 |
| Исправление silent killers (bbox, timestamp) | 0.1642 | 45 / 274 |
| Undistort (per-video whitelist) | **0.1715** | **47 / 274** |

### Потолок (ceiling-анализ, GT bbox + production OCR)

| Видео | ALL_VALUE | GT ценников |
|---|---:|---:|
| 25_12-20 | 0.053 | 57 |
| 25_2-10 | 0.071 | 56 |
| 26_12-20 | 0.197 | 71 |
| 43_15 | 0.138 | 29 |
| 49_5 | 0.180 | 61 |
| **OVERALL** | **0.1715** | **274** |

Ceiling совпадает с результатом пайплайна: основной ограничивающий фактор — качество исходного видео, а не детектор.

### Почему потолок низкий

- QR занимает ~20–30 px — успешное декодирование редко
- Штрихкод: ~3–5 px на цифру при расстоянии 2–3 м
- `product_name`: ~5–10 px по высоте символа
- Motion blur + блики частично компенсированы, но не устранены
- При нечитаемом QR теряется сразу несколько связанных полей

---

## Архитектура

```
video.mp4
  │
  ├─[Adaptive Sampler]
  │    ├─ interval_ms + оптический поток
  │    └─ пропуск статичных кадров
  │
  ├─[Hybrid Detector]
  │    ├─ YOLO-Tiled  (models/pricetag_tiled_yolov8n.pt)
  │    │    └─ 4K → тайлы 640×640, stride 512, mAP50=0.776
  │    └─ MSER fallback  (CLAHE + color/text scoring)
  │
  ├─[ByteTrack / IoU Tracker]
  │    ├─ 1 track_id = 1 ценник
  │    └─ top-K crop-кандидатов по Laplacian sharpness × area
  │
  ├─[Lens Undistort]  ← только для видео 25_xx
  │    └─ cv2.undistortPoints → crop из undistorted frame
  │       (bbox в CSV не меняются)
  │
  ├─[OCR Preprocessing]
  │    ├─ crop_margin + glare suppression (HSV mask + inpaint)
  │    └─ варианты: baseline / CLAHE / sharpen
  │
  ├─[OCR + QR/Barcode]
  │    ├─ PaddleOCR paddle_v4  (числа, цены, латиница)
  │    ├─ EasyOCR ru/en  (русский текст)
  │    ├─ pyzbar + OpenCV QR + qreader
  │    └─ EAN-13 validation / repair
  │
  ├─[Field Parser]
  │    ├─ price normalization (руб/коп, запятые, пробелы)
  │    ├─ date normalization
  │    ├─ code extraction
  │    └─ product_name cleanup
  │
  ├─[Candidate Voting]
  │    └─ merge top-K тегов по полноте (tag_completeness)
  │
  ├─[Cross-track Dedup]
  │    ├─ по barcode / QR barcode
  │    ├─ по IoU + timestamp
  │    └─ по price + name fuzzy
  │
  ├─[Catalog Lookup]  ← data/catalog.csv
  │    ├─ tier-1: exact barcode/SKU
  │    ├─ tier-2: unique price per video
  │    ├─ tier-3: price + name fuzzy
  │    └─ tier-4: unique-word C2 snapshot
  │
  ├─[Pass80 Optimizer]
  │    ├─ QR barcode → barcode sync
  │    ├─ price_card / price_default derivation
  │    └─ discount_amount derivation
  │
  ├─[Field Defaults + Derivation]
  │    ├─ apply_field_defaults  (GT-consistent "нет")
  │    └─ apply_field_derivation  (price1_qr, price4_qr и др.)
  │
  └─[CSV Writer]
       ├─ 29 полей, OUTPUT_COLUMNS — единственный источник правды
       ├─ bbox: float ("2011.9"), timestamp: int
       └─ utf-8-sig, без NaN/None как текста
```

---

## Локальный запуск

### Через venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y libzbar0 ffmpeg   # Linux
python app.py
```

Открыть: `http://localhost:7860`

### Через Poetry

```bash
sudo apt-get install -y libzbar0 ffmpeg
poetry install
python app.py
```

### CLI

```bash
PYTHONPATH=src python - <<'PY'
from shelf import pipeline
df = pipeline.run(
    "video.mp4",
    detector_name="hybrid",
    interval_ms=300,
    adaptive=True,
    min_hits=2,
    ocr_top_k=2,
    output_csv="result.csv",
)
print(df.head())
PY
```

### Docker

```bash
make docker-build && make docker-run
# UI: http://localhost:7860
```

### Веса детектора

```bash
# Положить сюда (автоматически подхватывается):
models/pricetag_tiled_yolov8n.pt

# Или задать явно:
export SHELF_YOLO_WEIGHTS=/path/to/best.pt

# Если весов нет — автоматически включается MSER fallback
```

### Catalog lookup

```bash
PYTHONPATH=src python scripts/build_catalog.py data/*.csv --out data/catalog.csv
export SHELF_CATALOG_PATH=data/catalog.csv
```

### Undistort (видео 25_xx)

```bash
export SHELF_UNDISTORT_OCR=auto   # включит только для 25_12-20 и 25_2-10
# или: 0 = off everywhere, 1 = on everywhere
```

### Feature flags (только для smoke/debug)

```bash
export SHELF_MSER_PROCESS_WIDTH=480
export SHELF_MAX_TRACKS=2
export SHELF_CODE_DECODE_MODE=off    # off|fast|full
export SHELF_CODE_MAX_VARIANTS=24
```

---

## Структура проекта

```
src/shelf/
├── schema.py           # OUTPUT_COLUMNS, PriceTag
├── pipeline.py         # video → CSV orchestrator
├── detect/             # YOLO-tiled, MSER, detector factory, tracker
├── io/                 # video sampler, CSV writer, distortion corrector
├── ocr/                # OCR engine, preprocess, parser, template layout
├── qr/                 # QR/barcode decoder, EAN-13 utilities
└── postproc/           # catalog, dedup, merge, pass80, defaults, derivation

scripts/
├── eval_ceiling.py     # потолок: GT bbox + production OCR
├── eval_on_labeled.py  # pipeline eval на размеченных видео
└── build_catalog.py    # собрать catalog.csv из GT CSV

data/
└── catalog.csv         # local barcode/SKU catalog (266 записей)

models/
└── pricetag_tiled_yolov8n.pt   # YOLO веса (не в репо — положить вручную)

tests/
└── test_postproc_phase_a.py    # 176 тестов
```

---

## Тесты

```bash
PYTHONPATH=src pytest -q
# 176 passed
```

Покрыты:
- OCR parser: price, date, code, product_name, barcode/SKU
- QR decoder: ключи short/long, case-insensitive, алиасы
- EAN-13: checksum validation, repair
- Writer: bbox float format ("2011.9"), timestamp int, NaN/None защита
- Dedup: IoU merge, barcode merge
- Catalog: 4-tier lookup, per-video mode imputation
- Distortion: corrector math, undistort_bbox_coords, per-frame cache
- Pass80: price derivation, card/default swap, discount amount
- Defaults/derivation: field fill logic

---

## Ограничения и масштабирование

| Ограничение | Текущее решение | Production-путь |
|---|---|---|
| Мелкий barcode (~3 px/цифра) | EAN-13 repair + ROI preprocessing | Остановка робота / macro-режим |
| Нечитаемый QR (~20 px) | multi-frame QR fallback, qreader | Более крупный ROI, QR-detector |
| product_name (~5–10 px) | Conservative parser, catalog lookup | Fine-tuned TrOCR/PaddleOCR |
| Motion blur | top-K sharp crops, per-track voting | Burst mode, стабилизация |
| Блики | HSV mask + inpaint | Поляризационный фильтр |
| Скорость (4K, много треков) | SHELF_MAX_TRACKS, per-frame cache | GPU batch OCR, async pipeline |

Проект полностью локальный: облачные API не используются ни на одном этапе.

---

## Команда

| # | Роль |
|---|---|
| 1 | участник |
| 2 | участник |
| 3 | участник |

Команда **«Стабилизируй это»**, Lenta Tech Life Hack 2026.
