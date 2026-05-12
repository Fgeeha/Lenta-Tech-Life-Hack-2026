# Полка под контролем — Lenta Tech Life Hack 2026

Пайплайн `video → CSV` для автоматического распознавания ценников с видео робота-сканера в магазине Лента.

## Быстрый старт

```bash
# Установить зависимости (Python 3.10+, Poetry)
make install

# Запустить Gradio UI
make run
# → http://localhost:7860
```

## Установка

### Требования

- Python 3.10 (или 3.11)
- Poetry 1.8+
- `libzbar0` для pyzbar: `sudo apt-get install libzbar0`

```bash
# Базовая установка (UI + детектор)
poetry install --with dev

# Полная установка (+ PaddleOCR)
poetry install --with ml,ocr,dev
```

## Использование

### Gradio UI

```bash
python app.py
```

Откройте http://localhost:7860, загрузите `.mp4` файл и нажмите «Запустить». Результат — CSV по схеме из ТЗ.

### CLI / API

```python
from shelf import pipeline

df = pipeline.run("Данные/43_15/43_15.mp4", interval_ms=200)
df.to_csv("result.csv", index=False)
```

### Оценка на размеченных видео

```bash
make eval
# или
poetry run python scripts/eval_on_labeled.py --videos 43_15 --interval 500
```

## Архитектура пайплайна

```
video.mp4
  │
  ├─[frame sampler]── адаптивная выборка кадров (оптический поток)
  │
  ├─[MSER detector]── поиск стабильных прямоугольных регионов
  │
  ├─[ByteTrack]────── трекинг (один track_id = один ценник)
  │                   лучший кадр = max(площадь × резкость)
  │
  ├─[OCR]──────────── PaddleOCR PP-OCRv4, rot180 + deskew + upscale
  │                   → product_name, price_card, discount_amount
  │
  ├─[QR decoder]───── pyzbar + opencv, 4 ориентации
  │                   → barcode, price1_qr..price4_qr
  │
  └─[CSV writer]───── 29 полей по схеме ТЗ §2
```

## Схема выходного CSV

29 полей строго в порядке ТЗ:

**С ценника:** `filename, product_name, price_default, price_card, price_discount, barcode, discount_amount, id_sku, print_datetime, code, additional_info, color, special_symbols, frame_timestamp, x_min, y_min, x_max, y_max`

**Из QR:** `qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr, wholesale_level_1_count, wholesale_level_1_price, wholesale_level_2_count, wholesale_level_2_price, action_price_qr, action_code_qr`

- `"нет"` — поле отсутствует на этом типе ценника
- `""` — поле есть, но не распозналось

## Текущая метрика

| Видео     | metric@80% | avg_field | matched/GT |
|-----------|-----------|-----------|-----------|
| 43_15     | 0.000     | 0.09      | 23/29     |
| (baseline — этапы 0-8) | | | |

**Узкое место**: OCR на маленьких MSER-кропах (~100px) не читает текст. Улучшение — этап 12 (псевдо-лейблинг + YOLO fine-tune).

## Ограничения

- QR-коды не декодируются — слишком маленькие (≈50px) + перспективные искажения
- OCR работает частично — ценники под углом, motion blur
- Детектор MSER даёт ~50% recall на отдельных кадрах (ByteTrack повышает через накопление)
- Все модели локальные, без облачных API

## Стек

- **Python 3.10**, Poetry
- **OpenCV** — видео, морфология, оптический поток
- **supervision** — ByteTrack
- **PaddleOCR** PP-OCRv4 — OCR русского текста
- **pyzbar** — декодирование QR/штрихкодов
- **Gradio 5** — UI
- **ultralytics** YOLOv8 — детектор (резерв для псевдо-лейблов)

## Запуск тестов

```bash
make test
# → 19 тестов: schema, QR-парсер, video sampler
```

## Docker

```bash
make docker-build
make docker-run
# → http://localhost:7860
```

## Структура проекта

```
src/shelf/
├── schema.py          # OUTPUT_COLUMNS, PriceTag (источник правды)
├── pipeline.py        # end-to-end оркестратор
├── io/                # video sampler, CSV writer
├── detect/            # MSER detector, ByteTrack
├── ocr/               # PaddleOCR, preprocessor, parser, template
├── qr/                # QR decoder, URL parser
├── postproc/          # merge (QR+OCR), dedup
└── ui/                # Gradio app
```
