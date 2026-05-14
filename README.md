# ShelfWatch — Полка под контролем

**Lenta Tech Life Hack 2026**

🚀 **Демо (без авторизации):** https://huggingface.co/spaces/fgeeha/shelf-control  
📊 **metric@80% = 0.004** · detection recall **100%** (157/157) · 5 размеченных видео · 274 ценника  
🎯 **Ceiling-анализ:** avg_field = 0.219 на GT-bboxes — единственная команда, измерившая физический потолок задачи

---

## Что это

Пайплайн `video.mp4 → CSV` для автоматического распознавания ценников с видео робота-сканера в магазинах Лента.  
Входные данные — 4K H.264-видео. Выход — CSV с 29 полями по схеме ТЗ (product_name, barcode, price_card, price_default, QR-поля и другие).

---

## Быстрый старт

```bash
# Зависимости: Python 3.10+, Poetry, libzbar0
sudo apt-get install libzbar0
poetry install

# Gradio UI
python app.py           # → http://localhost:7860

# CLI
python -c "
from shelf import pipeline
df = pipeline.run('video.mp4', detector_name='yolo-tiled', interval_ms=500)
df.to_csv('result.csv', index=False)
"
```

```bash
# Docker
make docker-build && make docker-run   # → http://localhost:7860
```

---

## Архитектура пайплайна

```
video.mp4
  │
  ├─[Adaptive Sampler]── оптический поток, шаг 200мс; пропускает статичные кадры
  │
  ├─[YOLO-Tiled]──────── 4K → тайлы 640×640 (stride 512, overlap 128)
  │                      YOLOv8n, fine-tuned, mAP50 = 0.776, recall = 100%
  │
  ├─[ByteTrack]────────── 1 track_id = 1 ценник
  │                      лучший кадр = argmax(bbox_area × Laplacian_sharpness)
  │
  ├─[Preprocess]─────────  90°CCW + deskew + upscale×2 (без CLAHE/sharpen)
  │                      *upscale×2 предотвращает слияние числовых токенов
  │
  ├─[OCR EN (PaddleOCR)]── price_card, price_default, discount_amount
  │   + [OCR RU (EasyOCR)]  product_name (верхняя зона кропа)
  │   + [EAN repair]        EAN-13 checksum repair на OCR-цифрах
  │   + [ROI barcode]       Sobel-X ROI + pyzbar для штрихкода
  │
  ├─[QR Decoder]─────────── pyzbar + qreader (4 ориентации, 4 масштаба)
  │                         barcode, price1-4_qr, wholesale, action fields
  │
  ├─[Field Derivation]───── price4_qr ← price_card       (96% match GT)
  │                         price1_qr ← price_default     (97% match GT)
  │                         price_default ← price1_qr     (обратная, QR-path)
  │                         qr_code_barcode ↔ barcode      (двусторонняя)
  │                         discount_amount ← int((1−pc/pd)×100)%
  │
  └─[CSV Writer]─────────── 29 полей по схеме ТЗ §2
```

---

## Метрики

### Ceiling-анализ (GT bboxes + production OCR)

> «Потолок» — максимально достижимая метрика при идеальном детекторе.
> Запустить: `poetry run python scripts/eval_ceiling.py`

| Видео | metric@80% | avg_field | GT ценников |
|-------|-----------|-----------|-------------|
| 25_12-20 | 0.000 | 0.236 | 57 |
| 25_2-10  | 0.000 | 0.187 | 56 |
| 26_12-20 | **0.014** | **0.243** | 71 |
| 43_15    | 0.000 | 0.219 | 29 |
| 49_5     | 0.000 | 0.209 | 61 |
| **OVERALL** | **0.004** | **0.219** | **274** |

### Точность по полям (ceiling, avg по 5 видео)

| Поле | Accuracy | Источник |
|------|----------|---------|
| price_discount | 0.972 | «нет» в 100% GT — baseline |
| price4_qr | 0.458 | деривация ← price_card |
| price_card | 0.420 | OCR оранжевой зоны |
| price2_qr | 0.112 | 11.7% GT = «нет»; остальное из QR |
| discount_amount | 0.200 | деривация из price_card / price_default |
| price1_qr | 0.082 | деривация ← price_default |
| price_default | 0.069 | OCR (меньший шрифт в оранжевой зоне) |
| id_sku | 0.048 | 12-значный артикул, редко читается |
| barcode | 0.019 | QR-path (0.7% QR-decoded) |
| qr_code_barcode | 0.019 | QR-path |
| product_name | 0.008 | fuzzy token-overlap ≥ 0.40 |

### Почему metric@80% = 0.004

Порог 80% требует ≥ 9 из 11 полей. Среднее число верных полей — 2.4.  
Единственный путь к порогу — QR-декодирование (при успехе сразу +6 полей).  
QR-успех: **2 ценника из 274 (0.7%)** — ограничение разрешения видео, см. ниже.

### История улучшений

| Изменение | avg_field (ceiling) |
|-----------|---------------------|
| baseline MSER + OCR | 0.091 |
| YOLO-tiled mAP50=0.776 | 0.154 |
| parser bug fixes | 0.155 |
| QR field derivation | 0.190 |
| price regex ≥3 digits | 0.220 |
| discount derivation | 0.221 |
| +2 видео (49_5, 25_2-10) | — |
| EAN-13 repair + ROI barcode | 0.219 |
| upscale=2 без CLAHE | **0.219 / price_card +4pp** |

---

## Обработка сложных случаев

*Этот раздел добавлен по требованию организаторов от 14.05.*

### Физические ограничения (неустранимые при текущих данных)

| Проблема | Измеренный факт | Вывод |
|----------|----------------|-------|
| **Barcode цифры** | При расстоянии 2-3 м от полки в 4K-кадре каждая цифра штрихкода занимает ~3-5 px. pyzbar, qreader, WeChatQR дают **0/274 прямых** декодирований EAN-13 через полосы. | Физическое ограничение. OCR-B шрифт под полосами = 5-8 px на букву. |
| **QR-коды** | QR на ценнике Lenta занимает ~20-30 px в 4K-кадре. Успешное декодирование: **2/274 (0.7%)**. qreader на полном 4K-кадре: **0 дополнительных**. | Требуется либо ближе к полке, либо другая камера. |
| **product_name** | Мелкий шрифт ~3-5 мм на реальном ценнике → 5-10 px в кадре. EasyOCR ru+en: token overlap ≥ 0.40 в **0.8%** случаев. | Нечитаем на текущем разрешении. |
| **Motion blur** | Движущийся робот + выдержка → размытие кадра. ByteTrack + Laplacian sharpness отсекает худшие кадры, но blur остаётся. | Частично компенсируется выбором лучшего кадра. |

### Как пайплайн обрабатывает эти случаи

**Размытые / перекрытые ценники:**
- ByteTrack аккумулирует треки по нескольким кадрам (`min_hits=2`)
- Лучший кадр = `argmax(bbox_area × Laplacian_variance)` — выбирается максимально резкий
- Слишком маленькие кропы (< 20px) полностью пропускаются

**Нечитаемый barcode:**
- При пустом `barcode` строка сопоставляется по приоритету №2: `frame_timestamp + bbox` с допусками
- Detection recall = 100% (все ценники попадают в матч)
- Поле оставляется пустым (`""`), не заполняется случайными числами

**Нечитаемый product_name:**
- Пустое поле (`""`) — не ошибка по спецификации
- Приоритет точности: лучше не угадывать, чем дать неверное название

**Несоответствие форматов в GT:**
- `43_15.csv`: опечатка колонки `wholesale_level_1_coun` → автоматически переименовывается
- `49_5.csv`: пробелы в barcode `"4 607124 143901"` → стрипятся в `_norm_bc`
- `43_15.csv`: trailing space в filename → `.str.strip()` при загрузке

**QR-decoded кейс (best-case):**
- При успешном QR: сразу 6+ полей (barcode, qr_code_barcode, price1-4_qr)
- Каскад деривации: price_default ← price1_qr → discount_amount вычисляется
- Итого: до 9/11 полей корректны → проходит порог 80%

### Что нужно для production-уровня

| Ограничение | Решение |
|-------------|---------|
| Мелкий барcode | Остановка робота перед полкой + macro-режим камеры |
| Нечитаемый QR | Fine-tune QR-детектора (WeChatQR YOLO) под low-res; или NFC-чипы на ценниках |
| product_name | Fine-tune TrOCR на синтетических ценниках Lenta (30-60% ожидаемый рост) |
| motion blur | 8K-камера или burst-mode (несколько кадров подряд) |

---

## Тесты и воспроизводимость

```bash
poetry run pytest          # 43 теста, <1 сек
poetry run python scripts/eval_ceiling.py    # ceiling на 5 видео (~10 мин)
poetry run python scripts/eval_on_labeled.py # pipeline на размеченных видео
```

**Ограничения по ТЗ:**
- ✅ Все модели разворачиваются локально (Poetry / Docker), без облачных API
- ✅ Веса: YOLOv8n 6 MB + PaddleOCR ~50 MB + EasyOCR ~150 MB
- ✅ Ручная разметка не использовалась (обучение YOLO — на GT-bboxes из CSV)
- ✅ Нет зависимостей от внешних баз данных (barcode lookup и т.п.)

---

## Структура проекта

```
src/shelf/
├── schema.py          # OUTPUT_COLUMNS (29 полей), PriceTag
├── pipeline.py        # end-to-end: video → List[PriceTag]
├── detect/            # YOLOSahiDetector, MSERDetector, ByteTrack
├── ocr/               # OCREngine (Paddle/Easy), preprocess, parser
├── qr/                # QR decoder, barcode_roi, EAN-13 repair
├── postproc/          # merge (QR+OCR+деривация полей)
└── ui/                # Gradio app

scripts/
├── eval_ceiling.py    # ceiling: GT bboxes + production OCR (5 видео)
├── eval_on_labeled.py # pipeline eval на размеченных видео
└── extract_tiles.py   # нарезка 4K-кадров для обучения YOLO

docs/
├── CEILING_ANALYSIS.md   # детальный ceiling-анализ
├── METRICS.md            # история метрики
└── DEPLOYMENT.md         # HF Spaces deployment guide

models/
└── pricetag_tiled_yolov8n.pt  # YOLOv8n fine-tuned, mAP50=0.776
```

---

## Стек

| Компонент | Технология |
|-----------|-----------|
| Детектор | YOLOv8n (ultralytics), tile-based SAHI-style inference |
| Трекер | ByteTrack (supervision) |
| OCR числа/EN | PaddleOCR PP-OCRv4 EN |
| OCR текст/RU | EasyOCR [ru, en] |
| Штрихкод | pyzbar + qreader + OpenCV WeChatQR |
| UI | Gradio 5 |
| Деплой | HuggingFace Spaces (Docker) |
| Тесты | pytest (43 теста) |
