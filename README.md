# Полка под контролем — Lenta Tech Life Hack 2026

> Pipeline для автоматического распознавания ценников с видеопотока
> робота. Команда **«Стабилизируй это»**.

## Быстрые ссылки

- **Демо**: https://huggingface.co/spaces/fgeeha/shelf-control
- **GitHub**: https://github.com/Fgeeha/Lenta-Tech-Life-Hack-2026
- **Метрика**: `0.1715` ALL_VALUE (47/274 ценников)
- **Прогресс**: `0/274 → 47/274` через архитектурные решения

---

## Постановка задачи

Lenta Tech разрабатывает решения для автоматизации контроля полки.
Робот движется вдоль стеллажей и снимает на видео ценники. Задача:
из видеопотока извлечь структурированные данные по каждому ценнику
(29 полей: цены, штрихкод, артикул, дата печати, координаты bbox и др.).

Метрика: **доля ценников с точностью распознавания ≥ 80%** от общего
числа ценников.

---

## Финальные показатели

| Метрика | Значение |
|---|---|
| **ALL_VALUE** (23 содержательных поля) | **0.1715** (47/274) |
| **COMPACT** (11 ключевых полей) | **0.1569** (43/274) |
| Тестов прохождения | 175/175 |

---

## Прогресс метрики

Стартовали с **0 распознанных ценников из 274**. Через архитектурные
решения достигли **47/274 (0.1715)**:

| Этап | Метрика | Δ тегов | Ключевое решение |
|---|---|---|---|
| Старт | 0.0000 (0/274) | — | базовый pipeline |
| Catalog v2 | 0.0109 (3/274) | +3 | первый catalog pass |
| **Catalog propagation** | **0.1277 (35/274)** | **+32** | пропагация полей через barcode |
| Prefix fallback | 0.1387 (38/274) | +3 | matching для OCR-битых EAN-13 |
| OCR upscale fix | 0.1423 (39/274) | +1 | исправлен PaddleOCR config |
| 4 новых поля каталога | 0.1606 (44/274) | +5 | print_datetime/code/additional_info |
| C2 unique-word snap | 0.1679 (46/274) | +2 | fuzzy match по уникальным словам |
| Per-video undistort | **0.1715 (47/274)** | +1 | коррекция дисторсии (whitelist 25_xx) |

---

## По видео (ceiling eval, GT bboxes + production OCR)

| Видео | Pass | Из | % | Зона магазина |
|---|---|---|---|---|
| 43_15 | 12 | 29 | 41% | Мёд/джемы |
| 26_12-20 | 18 | 71 | 25% | Вино |
| 25_12-20 | 9 | 57 | 16% | Алкоголь |
| 25_2-10 | 7 | 56 | 13% | Алкоголь |
| 49_5 | 1 | 61 | 2% | Молочка (через стекло) |
| **ИТОГО** | **47** | **274** | **17%** | |

---

## Архитектура решения

```
Видео .mp4
    ↓
[Detection] YOLOv8n tiled (4K→640×640 тайлы, mAP50=0.776) + MSER fallback
    ↓
[ByteTracker] + per-video lens correction (官方 Lenta calibration k1=-0.276)
    ↓
[QR/Barcode] WeChat QR → pyzbar → cv2.barcode → zxing-cpp → qreader
    ↓
[OCR] PaddleOCR (числа, цены) + EasyOCR ru (product_name)
    ↓
[Parser] price normalization, date, code, EAN-13 validation/repair
    ↓
[Catalog 4-tier lookup]
    1. exact EAN-13 / SKU
    2. prefix/substring fallback (OCR-битые barcode)
    3. video-scoped dual-price lookup
    4. C2 unique-word fuzzy snap
    ↓
[Derivation] price1_qr←price_default, price4_qr←price_card,
             price2_qr, discount_amount, structural defaults
    ↓
[Track voting] multi-frame merge_candidate_tags
    ↓
[CSV] 29 колонок по схеме sample.csv Lenta
```

### Ключевые архитектурные находки

**1. Catalog-first approach** — один barcode hit = 4-5 catalog полей +
5 derived полей одновременно. Дал breakthrough 0/274 → 35/274.

**2. Per-video undistort** — официальная camera calibration от Lenta
(k1=-0.276), применяется только к видео 25_xx (whitelist по smoke test).

**3. C2 unique-word snap** — уникальное слово в product_name
(Простоквашино, MOULIN) разрешает collision-кейсы в каталоге.

**4. Multi-source derivation**:
- `price1_qr ← price_default`, `price4_qr ← price_card`
- `price2_qr = price1_qr × 0.95` (5% card discount rule)
- `discount_amount = (price_default - price_card) / price_default`
- Structural defaults: `wholesale_*`, `action_*` → `"нет"`

---

## Локальный запуск

### Требования

- Python 3.10+
- Poetry (или pip + requirements.txt)
- ~4 GB RAM (CPU-only)
- `libzbar0`, `ffmpeg` (Linux)

### Установка

```bash
git clone https://github.com/Fgeeha/Lenta-Tech-Life-Hack-2026.git
cd Lenta-Tech-Life-Hack-2026
sudo apt-get install -y libzbar0 ffmpeg
poetry install
```

### Генерация submission CSV

```bash
SHELF_UNDISTORT_OCR=auto PYTHONPATH=src poetry run python scripts/generate_submission.py \
    --videos path/to/video1.mp4 path/to/video2.mp4 \
    --out submission.csv \
    --interval-ms 250 \
    --ocr-engine paddle_v4 \
    --ocr-top-k 2
```

### Веб-интерфейс (Gradio)

```bash
poetry run python app.py
# Открой http://localhost:7860
```

### Воспроизведение метрики

```bash
SHELF_UNDISTORT_OCR=auto PYTHONPATH=src poetry run python scripts/eval_ceiling.py \
    --data-root Данные \
    --ocr-engine paddle_v4 \
    --json-out reports/ceiling.json
# Ожидаемый результат: ALL_VALUE=0.1715 (47/274)
```

---

## Структура проекта

```
.
├── src/shelf/
│   ├── pipeline.py         # точка входа видео → CSV
│   ├── schema.py           # PriceTag + 29 OUTPUT_COLUMNS
│   ├── detect/             # YOLO-tiled, MSER, ByteTrack
│   ├── io/                 # video sampler, CSV writer, distortion corrector
│   ├── ocr/                # PaddleOCR, EasyOCR, layout parser
│   ├── qr/                 # multi-decoder cascade, EAN-13, barcode ROI
│   ├── postproc/           # catalog, voting, pass80, defaults, derivation
│   └── ui/gradio_app.py    # Gradio web UI
├── scripts/
│   ├── generate_submission.py
│   ├── eval_ceiling.py
│   └── build_catalog.py
├── tests/                  # 175 unit tests
├── data/catalog.csv        # 266 barcode/SKU записей
├── models/
│   └── pricetag_tiled_yolov8n.pt
├── reports/
│   ├── ceiling_FINAL_pervideo.json
│   └── submission_FINAL.csv
├── app.py                  # Gradio entry point
└── pyproject.toml
```

---

## Тесты

```bash
PYTHONPATH=src poetry run pytest tests/ -q
# 175 passed
```

Покрыто: OCR parser, QR decoder, EAN-13 repair, catalog lookup,
bbox float format, deduplication, field derivation, distortion corrector.

---

## Ограничения

| Ограничение | Факт | Попытки смягчения |
|---|---|---|
| QR/barcode recall ~18% | физический лимит камеры: 20-30px QR, 3-5px штрихкод | multi-frame, ROI upscale, anti-glare |
| 49_5: 1/61 (2%) | съёмка через стекло + 13 SKU за 159.99₽ в каталоге | visual fingerprint (не реализовано) |
| product_name OCR ~16% | стилизованные шрифты, мелкий многострочный текст | EasyOCR ru, fuzzy catalog match |

---

## Идеи масштабирования

- Template-aware OCR по `Расшифровка ценников.pdf` для per-template ROI
- Visual fingerprint matching для collision-кейсов на молочке
- RKNN/INT8 export для edge-deployment на роботе
- Active learning: pseudo-labels из production для дообучения
- Cross-store catalog из ERP всей Группы Лента

---

## Команда «Стабилизируй это»

| Имя | Роль | Telegram |
|---|---|---|
| Колесников Никита | Капитан, ML/CV Engineer, MLOps | @fgeeha |
| Сергей Левашов | ML/CV Engineer | @Tradeoffc |
| Струганова Алина | аналитик данных | @morshkah |
| Ваче Оганисян | аналитик данных | @v208404 |
| Станислава Ивахненко | аналитик данных | @stasyssssss |

---
