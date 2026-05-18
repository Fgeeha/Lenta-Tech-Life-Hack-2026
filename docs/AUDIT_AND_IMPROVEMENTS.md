# Аудит и улучшения пайплайна распознавания ценников

## Краткий аудит исходного проекта

Проект уже содержал правильную базовую идею: детекция ценников, трекинг, OCR, QR-парсинг, CSV-схема и Gradio UI. Главные проблемы были не в отсутствии модулей, а в деталях, которые сильно бьют по итоговому CSV:

1. `frame_timestamp` фактически считался в секундах, хотя по ТЗ нужен timestamp в миллисекундах. Из-за этого также не работал лимит `max_duration_sec`, потому что секунды сравнивались с миллисекундами.
2. `_CROP_MARGIN` был объявлен, но не использовался. OCR получал слишком тесный crop, часто без QR/штрихкода/части названия.
3. Default detector был `mser`, хотя README и UI ориентировали на `yolo-tiled`. При отсутствии весов YOLO fallback уходил в обычный COCO `yolov8n.pt`, что лучше заменить на MSER fallback, а не на нерелевантную COCO-модель.
4. Один трек давал один лучший crop только по `area * Laplacian`. Это лучше, чем случайный кадр, но не учитывает glare, экспозицию и то, что OCR может быть лучше на другом соседнем кадре.
5. 12-значный `id_sku` мог быть ошибочно превращён в 13-значный barcode через EAN repair.
6. QR-парсер был чувствителен к регистру ключей и мог агрессивно “исправлять” 13-значный barcode из QR.
7. CSV writer не централизовал нормализацию схемы, алиас `wholesale_level_1_coun` из старой разметки и `NaN`.
8. Дедупликация была только “один track_id -> одна строка”; при ID switch один физический ценник мог попасть в CSV дважды.
9. UI не давал управлять важными quality/speed параметрами: detector, interval, top-K OCR кадров, лимит длительности.

## Что реализовано в этой версии

### A. Видео и предобработка

- `sample_frames()` теперь возвращает `timestamp_ms`, а не секунды.
- Добавлена оценка качества кадра/crop: Laplacian sharpness, glare fraction, brightness penalty.
- Добавлено подавление бликов через HSV mask + inpaint.
- Добавлена перспективная коррекция как безопасная опция.
- Добавлены OCR-варианты crop: безопасный baseline, CLAHE-вариант, sharpen-вариант.

### B. Детекция

- Новый default detector: `hybrid`.
- `hybrid = yolo-tiled` при наличии обученных весов + MSER fallback.
- Убран fallback на generic COCO `yolov8n.pt` для `yolo-tiled`, потому что COCO не знает класса ценника и даёт нерелевантные боксы.
- MSER стал устойчивее: CLAHE на grayscale, color/text scoring, более гибкие ограничения по aspect/area.

### C. Трекинг и дедупликация

- Tracker теперь хранит top-K лучших crop-кандидатов на трек, а не один crop.
- Лучший crop выбирается по качеству: `area * frame_quality_score`, где учитываются sharpness/glare/brightness.
- Реально используется `crop_margin`, чтобы OCR видел весь ценник, QR и штрихкод.
- Добавлен fallback IoU tracker на случай отсутствия `supervision`.
- Добавлена cross-track дедупликация:
  - по barcode/QR barcode;
  - по IoU + близкому timestamp;
  - по одинаковым ценам + похожему названию.

### D. OCR и парсинг полей

- Парсер стал консервативнее: не угадывает поля без сильного паттерна.
- Исправлена критичная ошибка: 12-значный `id_sku` больше не превращается в barcode.
- Улучшена нормализация цен: `129`, `129.99`, `1 299,99`.
- Улучшена нормализация дат: `03.04.2026 3:08`, `03-04-26 03.08`.
- Улучшено извлечение `code`: поддерживаются варианты вида `01_025019 - 026015`.
- `product_name` очищается от цен, дат, длинных цифровых ID и дублей.
- Для цен используется не только значение, но и размер OCR-бокса: крупная цена обычно `price_card`/акционная цена.

### E. QR-код

- QR-парсер стал case-insensitive.
- Поддержаны короткие и длинные ключи: `b/barcode`, `p1/price1`, `wL1C/wholesaleLevel1Count`, `aP/actionPrice` и т.д.
- Добавлены дополнительные декодеры/варианты: pyzbar, OpenCV single/multi QR, qreader fallback.
- Пробуются повороты, масштабы, CLAHE и Otsu-варианты.
- QR barcode теперь не подвергается агрессивному one-digit repair, если он уже 13-значный.

### F. CSV

- `OUTPUT_COLUMNS` остаётся единственным источником правды.
- Добавлен `prepare_output_dataframe()`.
- Writer всегда сохраняет CSV в правильном порядке колонок и в `utf-8-sig`.
- Учитывается алиас старой разметки `wholesale_level_1_coun -> wholesale_level_1_count`.
- `NaN`/`None` не попадают в CSV как текст.

### G. UI

- UI переведён на режим `hybrid` по умолчанию.
- Добавлены параметры:
  - detector;
  - interval_ms;
  - min_hits;
  - OCR top-K кадров;
  - лимит длительности видео;
  - adaptive sampling.
- Добавлен прогресс обработки.
- Превью расширено: цены, barcode, QR barcode, bbox, timestamp.

### H. Тесты

Добавлены/обновлены регрессионные тесты:

- timestamp в миллисекундах;
- 12-значный SKU не становится barcode;
- валидный EAN-13 извлекается как barcode;
- QR ключи не зависят от регистра;
- cross-track дедупликация по barcode слиянием полей.

Текущий результат: `48 passed`.

## Приоритеты дальнейшего улучшения

### Быстро, 1-2 дня

1. Положить реальные обученные веса в `models/pricetag_tiled_yolov8n.pt`.
2. Разметить/проверить 200-500 кадров через CVAT/Label Studio.
3. Прогнать `scripts/eval_on_labeled.py` до/после и сравнить:
   - detection recall;
   - duplicate rate;
   - OCR field accuracy;
   - CSV row-level score.
4. Подобрать `interval_ms`, `min_hits`, `ocr_top_k` на валидационных видео.

### Среднесрочно, 1-2 недели

1. Дообучить YOLO/RT-DETR на реальных bbox ценников с аугментациями: blur, glare, perspective, compression, low-light.
2. Добавить отдельный детектор QR/штрихкода внутри crop.
3. Сделать layout parser по шаблонам ценников: регулярный, акция, распродажа, BOGOF, МНЦ.
4. Обучить CRNN/TrOCR/PaddleOCR fine-tune на синтетике ценников Ленты.

### Максимально качественно

1. Делать multi-frame super-resolution для QR/штрихкода по треку.
2. Использовать homography/optical flow для стабилизации одного ценника между кадрами.
3. Вести track-level evidence store: каждое поле выбирается голосованием по всем кадрам.
4. Сделать model ensemble: YOLO-tiled + RT-DETR + QR detector + OCR field detector.
5. Добавить active learning: UI показывает crop-и с низкой confidence, оператор быстро исправляет, затем модель дообучается.

## Проверочный чек-лист

- [ ] `pytest -q` проходит без ошибок.
- [ ] В CSV ровно 29 колонок и порядок совпадает с `OUTPUT_COLUMNS`.
- [ ] `frame_timestamp` измеряется в миллисекундах.
- [ ] Один физический ценник не дублируется на соседних кадрах.
- [ ] При пустом QR поля остаются `нет`, а не случайными значениями.
- [ ] `id_sku` и `barcode` не путаются.
- [ ] Для каждого найденного ценника bbox лежит внутри кадра.
- [ ] UI позволяет скачать CSV после обработки.

## Stage 4 next-step implementation

Implemented without changing detector weights or the 29-field CSV schema:

1. `src/shelf/ocr/layout.py` — rule-based template/orientation classification and semantic ROIs for name, QR, barcode, price, SKU, datetime, code and symbols.
2. `src/shelf/qr/decoder.py` — template ROI decoding precedes geometric ROI and full-crop decoding; optional debug files capture successful and failed code reads.
3. `src/shelf/postproc/pass80.py` — conservative consistency optimizer for barcode/QR barcode, QR->OCR prices, price swaps, discount derivation and catalog-backed name fill.
4. `scripts/eval_on_labeled.py` — per-tag diagnostics and field-level reports for metric@80 debugging.
5. `src/shelf/postproc/voting.py` — product-name candidate cleanup/selection removes service noise while preserving useful percents.
6. Pipeline debug mode now writes product-name, price and pass80 optimizer candidate CSVs under `debug_dir`.

The key remaining blocker is full metric verification on all five private videos.  The public/local smoke test confirms launchability, not metric growth.
