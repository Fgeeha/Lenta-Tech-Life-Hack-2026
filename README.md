# ShelfWatch — Полка под контролем

**Lenta Tech Life Hack 2026**

🚀 **Демо без авторизации:** https://huggingface.co/spaces/fgeeha/shelf-control  
📊 **metric@80% = 0.004** · detection recall **100%** · 5 размеченных видео · 274 ценника  
🎯 **Ceiling-анализ:** `avg_field = 0.219` на GT-bboxes

---

## Что это

ShelfWatch — это пайплайн `video.mp4 → CSV` для автоматического распознавания ценников с видео робота-сканера в магазинах Лента.

Входные данные:

- 4K H.264-видео с прохода вдоль полки;
- опционально — обученные веса детектора ценников.

Выход:

- CSV-файл с 29 полями по схеме задания:
  `product_name`, `barcode`, `price_card`, `price_default`, QR-поля, координаты bbox, timestamp и другие.

Проект полностью запускается локально: без облачных API, внешних баз данных и ручной разметки на этапе инференса.

---

## Быстрый старт

### Вариант 1 — через venv и requirements.txt

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Linux system dependencies для pyzbar/video:
sudo apt-get install -y libzbar0 ffmpeg

python app.py
```

Откройте:

```text
http://localhost:7860
```

Загрузите видео и скачайте CSV.

---

### Вариант 2 — через Poetry

```bash
sudo apt-get install -y libzbar0 ffmpeg

poetry install

python app.py
```

---

### CLI-запуск

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

---

### Docker

```bash
make docker-build
make docker-run
```

После запуска UI будет доступен на:

```text
http://localhost:7860
```

---

## Веса детектора

Для лучшего качества положите обученные веса сюда:

```text
models/pricetag_tiled_yolov8n.pt
```

Также `yolo-tiled`/`hybrid` автоматически ищет локальные веса, если они уже лежат в challenge-архиве:

```text
runs/detect/runs/detect/pricetag_tiled_v1/weights/best.pt
runs/detect/runs/detect/pricetag_v1/weights/best.pt
```

Можно явно указать путь:

```bash
export SHELF_YOLO_WEIGHTS=/path/to/pricetag_tiled.pt
```

По умолчанию проект **не скачивает** модели из сети и **не возвращается** к generic COCO `yolov8n.pt`, потому что COCO не содержит класса ценника. Для локального эксперимента с автоскачиванием обученных весов нужно явно включить:

```bash
export SHELF_ALLOW_MODEL_DOWNLOAD=true
```

Если веса отсутствуют, режим `hybrid` автоматически использует MSER fallback. Это позволяет проекту запускаться локально и в Docker даже без обученной модели, но качество fallback-детекции ниже, чем у fine-tuned YOLO.

### Локальный catalog lookup

Если доступны локальные CSV с barcode/SKU/product_name, можно построить каталог без внешних API:

```bash
PYTHONPATH=src python scripts/build_catalog.py data/*.csv --out data/catalog.csv
```

При валидном `barcode` или `id_sku` catalog lookup может заполнить `product_name` и пустые price-поля; OCR остаётся fallback. Переопределение пути:

```bash
export SHELF_CATALOG_PATH=/path/to/catalog.csv
```

### Быстрые smoke/HF flags

Production defaults не меняются, но для быстрых проверок на CPU есть feature flags:

```bash
export SHELF_MSER_PROCESS_WIDTH=480      # ускорить MSER fallback в smoke-run
export SHELF_MAX_TRACKS=2                # обработать только top-scored tracks
export SHELF_CODE_DECODE_MODE=off        # off|fast|full для QR/barcode smoke
export SHELF_CODE_MAX_VARIANTS=24        # лимит QR/barcode вариантов
```

---

## Архитектура пайплайна

```text
video.mp4
  │
  ├─[Adaptive Sampler]
  │    ├─ шаг по времени: interval_ms
  │    ├─ оптический поток
  │    └─ пропуск статичных кадров
  │
  ├─[Hybrid Detector]
  │    ├─ YOLO-Tiled, если есть models/pricetag_tiled_yolov8n.pt
  │    │    ├─ 4K → тайлы 640×640
  │    │    ├─ stride 512
  │    │    ├─ overlap 128
  │    │    └─ YOLOv8n fine-tuned, mAP50 = 0.776
  │    │
  │    └─ MSER fallback, если YOLO-весов нет
  │         ├─ CLAHE grayscale
  │         ├─ color/text scoring
  │         └─ aspect/area filtering
  │
  ├─[Tracking]
  │    ├─ ByteTrack, если доступен supervision
  │    ├─ IoU tracker fallback
  │    ├─ 1 track_id = 1 ценник
  │    └─ top-K crop-кандидатов на каждый track
  │
  ├─[Crop Quality Scoring]
  │    ├─ Laplacian sharpness
  │    ├─ glare fraction
  │    ├─ brightness penalty
  │    └─ score = area × frame_quality_score
  │
  ├─[Preprocess]
  │    ├─ crop_margin, чтобы захватить весь ценник, QR и штрихкод
  │    ├─ подавление бликов через HSV mask + inpaint
  │    ├─ безопасная perspective correction
  │    ├─ baseline crop
  │    ├─ CLAHE-вариант
  │    └─ sharpen-вариант
  │
  ├─[OCR]
  │    ├─ PaddleOCR для чисел, цен и латиницы
  │    ├─ EasyOCR ru/en для русского текста
  │    ├─ OCR top-K кадров
  │    └─ conservative parser без агрессивного угадывания
  │
  ├─[QR / Barcode]
  │    ├─ pyzbar
  │    ├─ OpenCV QR single/multi
  │    ├─ qreader fallback
  │    ├─ повороты
  │    ├─ масштабы
  │    ├─ CLAHE
  │    ├─ Otsu threshold
  │    └─ ROI barcode decoding
  │
  ├─[Field Parsing]
  │    ├─ price normalization
  │    ├─ date normalization
  │    ├─ code extraction
  │    ├─ SKU/barcode separation
  │    ├─ QR key aliases
  │    └─ product_name cleanup
  │
  ├─[Cross-track Deduplication]
  │    ├─ по barcode / QR barcode
  │    ├─ по IoU + близкому timestamp
  │    └─ по одинаковым ценам + похожему названию
  │
  └─[CSV Writer]
       ├─ 29 полей по схеме ТЗ
       ├─ OUTPUT_COLUMNS как единственный источник правды
       ├─ utf-8-sig
       └─ без NaN/None как текста
```

---

## Что реализовано

### Видео и предобработка

- `sample_frames()` возвращает `timestamp_ms`, а не секунды.
- Добавлена оценка качества кадра/crop:
  - Laplacian sharpness;
  - glare fraction;
  - brightness penalty.
- Добавлено подавление бликов через HSV mask + inpaint.
- Добавлена безопасная перспективная коррекция.
- Добавлены OCR-варианты crop:
  - baseline;
  - CLAHE;
  - sharpen.

---

### Детекция

- Основной режим по умолчанию: `hybrid`.
- `hybrid = yolo-tiled + MSER fallback`.
- Убран fallback на generic COCO `yolov8n.pt`, потому что COCO не знает класса ценника и даёт нерелевантные боксы.
- MSER fallback усилен:
  - CLAHE на grayscale;
  - color/text scoring;
  - более гибкие ограничения по aspect/area.

---

### Трекинг и дедупликация

- Tracker хранит top-K лучших crop-кандидатов на track.
- Лучший crop выбирается по качеству:
  `area × frame_quality_score`.
- `crop_margin` реально используется, чтобы OCR видел весь ценник, QR и штрихкод.
- Добавлен fallback IoU tracker при отсутствии `supervision`.
- Добавлена cross-track дедупликация:
  - по barcode / QR barcode;
  - по IoU + близкому timestamp;
  - по одинаковым ценам + похожему названию.

---

### OCR и парсинг полей

- Парсер стал консервативнее: не угадывает поля без сильного паттерна.
- Исправлена критичная ошибка: 12-значный `id_sku` больше не превращается в `barcode`.
- Улучшена нормализация цен:
  - `129`;
  - `129.99`;
  - `1 299,99`.
- Улучшена нормализация дат:
  - `03.04.2026 3:08`;
  - `03-04-26 03.08`.
- Улучшено извлечение `code`, включая варианты вида:

```text
01_025019 - 026015
```

- `product_name` очищается от:
  - цен;
  - дат;
  - длинных цифровых ID;
  - дублей.
- Для цен используется не только значение, но и размер OCR-бокса:
  крупная цена чаще соответствует `price_card` / акционной цене.

---

### QR-код и barcode

- QR-парсер стал case-insensitive.
- Поддержаны короткие и длинные ключи:
  - `b` / `barcode`;
  - `p1` / `price1`;
  - `wL1C` / `wholesaleLevel1Count`;
  - `aP` / `actionPrice`.
- Добавлены дополнительные декодеры и варианты:
  - pyzbar;
  - OpenCV QR single/multi;
  - qreader fallback;
  - rotations;
  - resize;
  - CLAHE;
  - Otsu.
- QR barcode не подвергается агрессивному one-digit repair, если он уже 13-значный.
- 12-значный SKU не считается barcode.

---

### CSV

- `OUTPUT_COLUMNS` в `src/shelf/schema.py` — единственный источник правды.
- Добавлен `prepare_output_dataframe()`.
- CSV всегда сохраняется:
  - в правильном порядке колонок;
  - в кодировке `utf-8-sig`.
- Учитывается алиас старой разметки:

```text
wholesale_level_1_coun -> wholesale_level_1_count
```

- `NaN` / `None` не попадают в CSV как текст.

---

### UI

UI переведён на режим `hybrid` по умолчанию.

Добавлены параметры:

- detector;
- interval_ms;
- min_hits;
- OCR top-K кадров;
- лимит длительности видео;
- adaptive sampling.

Добавлен прогресс обработки.

Превью расширено:

- цены;
- barcode;
- QR barcode;
- bbox;
- timestamp.

---

## CSV schema

Выходные колонки определены в `src/shelf/schema.py` как `OUTPUT_COLUMNS`.

### Поля ценника

```text
filename
product_name
price_default
price_card
price_discount
barcode
discount_amount
id_sku
print_datetime
code
additional_info
color
special_symbols
frame_timestamp
x_min
y_min
x_max
y_max
```

### QR-поля

```text
qr_code_barcode
price1_qr
price2_qr
price3_qr
price4_qr
wholesale_level_1_count
wholesale_level_1_price
wholesale_level_2_count
wholesale_level_2_price
action_price_qr
action_code_qr
```

### Семантика значений

- `нет` — параметр отсутствует на данном типе ценника;
- пустая строка `""` — параметр существует, но не был распознан;
- `NaN` и `None` не записываются в CSV как текстовые значения.

---

## Метрики

### Ceiling-анализ

Ceiling-анализ показывает максимально достижимое качество при идеальной детекции, когда используются GT bbox и production OCR.

Запуск:

```bash
poetry run python scripts/eval_ceiling.py
```

| Видео | metric@80% | avg_field | GT ценников |
|---|---:|---:|---:|
| 25_12-20 | 0.000 | 0.236 | 57 |
| 25_2-10 | 0.000 | 0.187 | 56 |
| 26_12-20 | **0.014** | **0.243** | 71 |
| 43_15 | 0.000 | 0.219 | 29 |
| 49_5 | 0.000 | 0.209 | 61 |
| **OVERALL** | **0.004** | **0.219** | **274** |

---

### Точность по полям

| Поле | Accuracy | Источник |
|---|---:|---|
| price_discount | 0.972 | `нет` в 100% GT — baseline |
| price4_qr | 0.458 | деривация из `price_card` |
| price_card | 0.420 | OCR оранжевой зоны |
| price2_qr | 0.112 | часть GT = `нет`, остальное из QR |
| discount_amount | 0.200 | деривация из `price_card` / `price_default` |
| price1_qr | 0.082 | деривация из `price_default` |
| price_default | 0.069 | OCR меньшего шрифта |
| id_sku | 0.048 | 12-значный артикул |
| barcode | 0.019 | QR-path / OCR-path |
| qr_code_barcode | 0.019 | QR-path |
| product_name | 0.008 | fuzzy token-overlap |

---

### Почему metric@80% низкий

Порог `metric@80%` требует корректно заполнить большую часть ключевых полей ценника.

Среднее число корректных полей остаётся низким из-за физических ограничений исходного видео:

- QR-коды слишком мелкие;
- barcode-цифры занимают несколько пикселей;
- product_name написан мелким шрифтом;
- часть кадров имеет motion blur и блики;
- при неуспешном QR-декодировании сразу теряется несколько связанных полей.

Ключевой путь к резкому росту `metric@80%` — стабильное QR/barcode decoding и улучшение OCR по ценовым полям.

---

## История улучшений

| Изменение | avg_field ceiling |
|---|---:|
| baseline MSER + OCR | 0.091 |
| YOLO-tiled mAP50=0.776 | 0.154 |
| parser bug fixes | 0.155 |
| QR field derivation | 0.190 |
| price regex ≥3 digits | 0.220 |
| discount derivation | 0.221 |
| +2 видео: 49_5, 25_2-10 | — |
| EAN-13 repair + ROI barcode | 0.219 |
| upscale=2 без CLAHE | **0.219 / price_card +4pp** |

---

## Обработка сложных случаев

### Физические ограничения

| Проблема | Измеренный факт | Вывод |
|---|---|---|
| Barcode цифры | При расстоянии 2–3 м от полки в 4K-кадре каждая цифра штрихкода занимает примерно 3–5 px. | Физическое ограничение текущего видео. |
| QR-коды | QR на ценнике занимает примерно 20–30 px. Успешное декодирование было редким. | Нужны более крупные ROI, остановка робота или multi-frame/SR. |
| product_name | Мелкий шрифт даёт примерно 5–10 px по высоте символа. | OCR названия товара остаётся самым сложным полем. |
| Motion blur | Движущийся робот + выдержка дают размытие. | Частично компенсируется выбором top-K резких кадров. |
| Блики | Глянцевые ценники дают засветки на оранжевой зоне. | Частично компенсируется HSV mask + inpaint. |

---

### Как пайплайн обрабатывает сложные случаи

#### Размытые ценники

- Используется трекинг по нескольким кадрам.
- Для каждого track хранятся top-K crop-кандидатов.
- Качество crop оценивается через sharpness, glare и brightness.
- Слишком слабые crop-кандидаты не должны перетирать хорошие значения.

#### Нечитаемый barcode

- 12-значный SKU не превращается в barcode.
- 13-значный barcode должен быть валидирован.
- Если barcode не прочитан, поле остаётся пустым.
- Случайные OCR-цифры не записываются как barcode.

#### Нечитаемый product_name

- Название очищается от цен, дат, ID и дублей.
- Если сильного сигнала нет, поле остаётся пустым.
- Приоритет — не угадывать мусорное название.

#### Несоответствие форматов в GT

Обрабатываются известные особенности разметки:

- `wholesale_level_1_coun` автоматически приводится к `wholesale_level_1_count`;
- пробелы в barcode нормализуются;
- лишние пробелы в `filename` удаляются при загрузке.

---

## Тесты

Запуск:

```bash
PYTHONPATH=src pytest -q
```

Текущий результат:

```text
48 passed
```

Покрыты регрессионные сценарии:

- timestamp в миллисекундах;
- QR parser с короткими, длинными и case-insensitive ключами;
- price / discount parsing;
- отделение SKU от barcode;
- EAN-13 repair utilities;
- cross-track deduplication;
- strict output schema;
- корректный порядок CSV-колонок;
- отсутствие `NaN` / `None` как текста в CSV.

---

## Воспроизводимость метрик

```bash
# Все тесты
PYTHONPATH=src pytest -q

# Ceiling-анализ на GT bbox
poetry run python scripts/eval_ceiling.py

# Pipeline evaluation на размеченных видео
poetry run python scripts/eval_on_labeled.py
```

Для финального сравнения метрик убедитесь, что обученные веса лежат здесь:

```text
models/pricetag_tiled_yolov8n.pt
```

---

## Ограничения по ТЗ

- Все модели разворачиваются локально.
- Облачные API не используются.
- Ручная разметка на этапе инференса не используется.
- Проект запускается через Python/Poetry/Docker.
- HF Spaces deployment поддерживается.
- Тяжёлые улучшения должны быть опциональными и не ломать CPU basic deployment.

---

## Структура проекта

```text
src/shelf/
├── schema.py          # OUTPUT_COLUMNS, PriceTag, CSV schema
├── pipeline.py        # end-to-end video → CSV
├── detect/            # YOLO-tiled, MSER fallback, detector factory
├── ocr/               # OCR engine, preprocess, parser
├── qr/                # QR decoder, barcode ROI, EAN-13 utilities
├── postproc/          # merge, derivation, deduplication
└── ui/                # Gradio UI

scripts/
├── eval_ceiling.py     # ceiling: GT bboxes + production OCR
├── eval_on_labeled.py  # pipeline eval на размеченных видео
└── extract_tiles.py    # подготовка тайлов для обучения YOLO

docs/
├── CEILING_ANALYSIS.md
├── METRICS.md
└── DEPLOYMENT.md

models/
└── pricetag_tiled_yolov8n.pt
```

---

## Стек

| Компонент | Технология |
|---|---|
| Детекция | YOLOv8n tiled inference + MSER fallback |
| Трекинг | ByteTrack / IoU tracker fallback |
| OCR | PaddleOCR + EasyOCR ru/en |
| QR | pyzbar + OpenCV QR + qreader |
| Barcode | pyzbar + ROI preprocessing + EAN-13 validation/repair |
| UI | Gradio |
| Деплой | HuggingFace Spaces / Docker |
| Тесты | pytest |

---

## Дальнейшее развитие

Рекомендуемые следующие шаги:

1. Добавить field-level voting по top-K OCR crop-кандидатам.
2. Усилить QR/barcode ROI decoding внутри crop ценника.
3. Добавить строгую EAN-13 check digit validation для всех barcode-кандидатов.
4. Улучшить price extraction с учётом bbox size, позиции и соседних слов.
5. Построить локальный catalog lookup из доступных GT CSV:
   - `barcode -> product_name`;
   - `id_sku -> product_name`.
6. Добавить опциональный multi-frame fusion / super-resolution для QR и barcode ROI.
7. Fine-tune OCR на синтетических ценниках Ленты.
8. Обновлять `METRICS.md` и `DECISIONS.md` после каждого этапа.

---

## Production-уровень

Для устойчивой работы в магазине потребуются:

| Ограничение | Практическое решение |
|---|---|
| Мелкий barcode | остановка робота перед полкой, macro-режим камеры |
| Нечитаемый QR | более крупный ROI, QR-detector, multi-frame fusion |
| product_name | synthetic fine-tuning OCR / TrOCR |
| motion blur | burst-mode, стабилизация, более короткая выдержка |
| glare | поляризационный фильтр, улучшенная подсветка, glare suppression |

---

## Краткий статус

Проект уже содержит рабочий локальный пайплайн:

```text
video → hybrid detection → tracking → top-K crops → preprocessing → OCR/QR/barcode → deduplication → 29-field CSV
```

Текущая версия делает упор на воспроизводимость, conservative parsing и корректный CSV-вывод без агрессивного угадывания полей.