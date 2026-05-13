# Полка под контролем — Lenta Tech Life Hack 2026

**Демо (без авторизации) → https://huggingface.co/spaces/lenta-hack/shelf-control**

📊 **metric@80% = 0.013** · детекция **157/157** · 3 размеченных видео

[Метрики](docs/METRICS.md) | [Деплой](docs/DEPLOYMENT.md) | [Архитектура](#архитектура-пайплайна)

---

Пайплайн `video.mp4 → CSV` для автоматического распознавания ценников с видео робота-сканера в магазинах Лента.

## Быстрый старт

```bash
# Требования: Python 3.10+, Poetry, libzbar0
sudo apt-get install libzbar0

# Установить зависимости
poetry install

# Запустить Gradio UI
python app.py
# → http://localhost:7860
```

## CLI / API

```python
from shelf import pipeline

df = pipeline.run("video.mp4", interval_ms=500, detector_name="yolo-tiled")
df.to_csv("result.csv", index=False)
```

## Архитектура пайплайна

```
video.mp4
  │
  ├─[Frame Sampler]── адаптивная выборка (оптический поток), шаг 200–500мс
  │
  ├─[YOLO-Tiled]───── 4K → тайлы 640×640 (stride 512, overlap 128)
  │                   mAP50 = 0.776 · recall = 100%
  │
  ├─[ByteTrack]────── трекинг: 1 track_id = 1 ценник
  │                   лучший кадр = max(area × Laplacian sharpness)
  │
  ├─[Preprocess]───── 90°CCW + deskew + upscale×5 + CLAHE + sharpen
  │
  ├─[PaddleOCR EN]─── price_card, price_default, discount_amount
  │
  ├─[QR/Barcode]───── pyzbar (4 ориентации) → barcode, price*_qr
  │
  ├─[Field Merge]───── деривация: price4_qr←price_card, price1_qr←price_default,
  │                   discount_amount←int((1-pc/pd)*100)%
  │
  └─[CSV Writer]───── 29 полей по схеме ТЗ
```

## Текущая метрика

| Видео | metric@80% | avg_field | matched/GT |
|---|---|---|---|
| 25_12-20 | 0.000 | 0.199 | 57/57 |
| 26_12-20 | **0.028** | 0.178 | 71/71 |
| 43_15 | 0.000 | 0.201 | 29/29 |
| **OVERALL** | **0.013** | — | **157/157** |

Потолок (GT bboxes, production OCR): avg_field=0.220, metric@80%=0.006.
Детекция перестала быть узким местом; ограничение — OCR качество (product_name 0%, id_sku 3%).

История улучшений:

| Шаг | metric@80% (pipeline) | avg_field (ceiling) |
|---|---|---|
| baseline MSER | 0.000 | 0.091 |
| YOLO full-frame fine-tune | 0.000 | 0.091 |
| YOLO-tiled mAP50=0.776 | 0.000 | 0.154 |
| parser bug fixes (×3) | 0.000 | 0.155 |
| derived QR fields | 0.000 | 0.190 |
| price regex ≥3 digits | **0.013** | 0.220 |
| discount derivation | 0.013 | 0.221 |

## Выходной CSV (29 полей)

**С ценника:** `filename, product_name, price_default, price_card, price_discount, barcode, discount_amount, id_sku, print_datetime, code, additional_info, color, special_symbols, frame_timestamp, x_min, y_min, x_max, y_max`

**Из QR:** `qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr, wholesale_level_1_count, wholesale_level_1_price, wholesale_level_2_count, wholesale_level_2_price, action_price_qr, action_code_qr`

## Оценка

```bash
# Ceiling (GT bboxes + production OCR)
poetry run python scripts/eval_ceiling.py

# Pipeline eval (все 3 видео)
SHELF_DETECTOR=yolo-tiled poetry run python scripts/eval_on_labeled.py --interval 500
```

## Стек

- **Python 3.10**, Poetry
- **ultralytics YOLOv8n** — тайловый детектор (mAP50=0.776)
- **supervision ByteTrack** — трекинг и выбор лучшего кадра
- **PaddleOCR PP-OCRv4** (EN) — OCR цен и скидок
- **pyzbar + OpenCV** — декодирование QR и EAN-13
- **Gradio 5** — UI
- Все модели локальные, облачных API нет

## Структура проекта

```
src/shelf/
├── schema.py          # OUTPUT_COLUMNS, PriceTag (источник правды)
├── pipeline.py        # end-to-end оркестратор
├── io/                # video sampler, CSV writer
├── detect/            # YOLOSahiDetector, MSERDetector, ByteTrack
├── ocr/               # PaddleOCR engine, preprocess, parser
├── qr/                # QR decoder, URL parser
├── postproc/          # merge (QR+OCR), деривация полей
└── ui/                # Gradio app

scripts/
├── extract_tiles.py   # нарезка 4K GT-кадров в тайлы 640×640
├── eval_ceiling.py    # потолок: GT bboxes + production OCR
└── eval_on_labeled.py # pipeline eval на размеченных видео

models/
└── pricetag_tiled_yolov8n.pt  # fine-tuned, mAP50=0.776
```

## Docker

```bash
make docker-build && make docker-run
# → http://localhost:7860
```

## Тесты

```bash
poetry run pytest  # 28 тестов
```

## Ограничения

- QR-коды не декодируются (≈50px в 4K, motion blur)
- product_name: 0% (кириллица не читается EN-моделью)
- id_sku: 3% (12-значный артикул в мелком шрифте)
- Метрика @80% требует 9/11 полей; максимум без QR — 8/11 (72.7%)
