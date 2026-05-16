# ADR — Architecture Decision Records

Одна строка на решение: дата, что выбрал, почему, альтернатива.

| Дата       | Решение                                           | Почему                                              | Альтернатива                |
|------------|---------------------------------------------------|-----------------------------------------------------|-----------------------------|
| 2026-05-12 | Poetry как менеджер зависимостей                 | Lockfile, group-зависимости, нативно для HF Spaces  | pip + requirements.txt      |
| 2026-05-12 | Python 3.10 (из /opt/python-3.10)               | 3.11 недоступен локально; 3.10 совместим с PaddleOCR | Python 3.12/3.13           |
| 2026-05-12 | Gradio для UI                                    | Нативно на HF Spaces, минимум кода                  | Streamlit, FastAPI+HTML     |
| 2026-05-12 | HF Spaces для деплоя                             | Бесплатный CPU, git push деплой, Gradio из коробки  | Render free tier + Docker   |
| 2026-05-12 | PaddleOCR (paddleocr group) — отдельная группа  | Тяжёлая зависимость, не нужна на этапах 0–4         | EasyOCR, Tesseract          |
| 2026-05-12 | pyzbar + qreader (ml group) — отдельная группа  | qreader тянет torch; ставим опционально             | opencv QRCodeDetector       |
| 2026-05-12 | Видео: 3840×2160, ~20 fps, 15–89 сек           | Высокое разрешение — ценники хорошо читаются; семплинг 200 мс (4 кадра/сек) достаточен | 500 мс |
| 2026-05-12 | GT: 157 уникальных ценников (57+71+29)          | Все три видео — красные ценники (АПЦ), 1 жёлтый; белых нет в GT — модель должна уметь и их | — |
| 2026-05-12 | GT опечатка: wholesale_level_1_coun             | В 26_12-20.csv и 43_15.csv усечённое имя столбца. Eval-скрипт нормализует через rename | — |
| 2026-05-12 | barcode в GT — float (4.67e+12)                 | pandas читает EAN-13 как float64; нормализация: round→int→str, lpad до 13 цифр | — |
| 2026-05-12 | print_datetime заполнено в ~65% строк           | Поле сложное для OCR (мелкий шрифт); в приоритете — barcode, price, product_name | — |
| 2026-05-12 | price3_qr / action_price_qr — почти всегда «нет» | Редкий тип ценника; распознаём, но не оптимизируем специально | — |
| 2026-05-12 | special_symbols: К (короб) или Ш (штука)        | Два символа в правом нижнем углу; парсим как отдельное поле | — |
| 2026-05-12 | MSER как детектор вместо цвета/YOLO/Canny        | Color не работает (ценники 0-50% orange, фон тоже оранжевый); Canny даёт один blob; YOLO без файнтюна не видит ценники; MSER recall ~50% на одном кадре | GroundingDINO (тяжело) |
| 2026-05-12 | ByteTrack (supervision) для дедупликации        | Трекер стабилизирует MSER-детекцию через кадры; min_hits=2 отсекает случайные регионы | DeepSORT, SORT        |
| 2026-05-12 | Лучший кадр трека = max(площадь × резкость Лапласиана) | Резкость через Var(Laplacian) коррелирует с читаемостью OCR | centre frame of track |
| 2026-05-12 | Ценники смонтированы перевёрнуто (180°) на стеллаже | Камера робота смотрит вбок; белая и оранжевая части ценника имеют разную ориентацию | —              |
| 2026-05-12 | OCR на полном кадре вместо мелких кропов              | Кропы ценника 100-200px; OCR работает хуже на малых размерах; full-frame даёт больше строк | только кропы |
| 2026-05-12 | QR-декодирование не работает на данных видео           | QR ~50px в 4K кадре, перспективные искажения, pyzbar/qreader/opencv — все возвращают None. Fallback: OCR числовых паттернов | дообучить QRdet |
| 2026-05-12 | Цена price_card читается OCR (129 conf=0.79)          | Крупные цифры читаются несмотря на угол и blur; price_default и другие поля — слабее | —              |
| 2026-05-13 | OCR lang='en' вместо 'ru'                             | EN модель читает числа и латиницу точнее (129✓, 252✓, 48%✓). RU мусор на числах     | TrOCR          |
| 2026-05-13 | Поворот ценника 90°CCW (не 180°)                      | Ценники смонтированы боком на стеллаже; 90°CCW = правильная ориентация для чтения    | —              |
| 2026-05-13 | Train/val split: 25_12-20+26_12-20→train, 43_15→val   | 43_15 не в train — честная метрика; GT CSV как обучающие лейблы (не ручная разметка) | random split   |
| 2026-05-13 | YOLOv8n fine-tune: mAP50=0.368, conf=0.02 для infer  | Модель неуверена (tiny objects, 24 images), ByteTrack не фильтрует на полных видео   | SAHI sliding window |
| 2026-05-13 | Matching 100% (57+71+29), avg_field=0.103-0.128       | Матчинг по IoU+timestamp отлично работает; bottleneck = OCR не читает правильные цены | — |
| 2026-05-13 | Bottleneck: MSER субрегионы ≠ bbox ценников          | OCR читает числа из случайных прямоугольников. Fix: tight YOLO bbox на full ценnik   | GT-guided OCR  |
| 2026-05-15 | Усилен evaluation script для 5 размеченных видео | Перед изменениями OCR нужен воспроизводимый отчёт по avg_field, metric@80, barcode, QR и fill-rate; в архиве проекта нет приватных видео, поэтому скрипт теперь корректно сообщает об отсутствующих путях | Оставить старый скрипт на 3 видео с ошибкой timestamp-ms |
| 2026-05-15 | Голосование по полям на top-K crop-кандидатах | Трек уже хранит несколько crop-кандидатов; объединение результатов на уровне полей повышает устойчивость OCR/QR без изменения detection recall | Оставить один лучший crop по полноте распознавания |
| 2026-05-15 | Строгая валидация barcode и SKU | Предотвращает превращение 12-значного SKU в barcode и отбрасывает ложные невалидные EAN-13 | Агрессивно применять checksum repair ко всем значениям |
| 2026-05-15 | Извлечение цен с учётом контекстных слов и геометрии OCR-боксов | Цены сильно влияют на avg_field; контекст вроде «по карте» / «без карты» снижает путаницу между price_card и price_default | Просто сортировать все числа и угадывать min/max |
| 2026-05-15 | Декодирование QR/barcode по ROI перед обработкой всего crop | QR и barcode обычно находятся справа или снизу; ROI-декодирование повышает шанс распознавания при меньших вычислительных затратах | Запускать тяжёлые преобразования только на всём crop |
| 2026-05-15 | Поддержка PP-OCRv5 mobile через конфиг | Это допустимая локальная OCR-альтернатива, но полные метрики сейчас недоступны, поэтому стабильный auto-режим оставляет PP-OCRv4 перед PP-OCRv5 | Сделать PP-OCRv5 безусловным вариантом по умолчанию |
| 2026-05-15 | Локальный catalog lookup из CSV-файлов | Разрешённые локальные данные могут заполнять product_name и цены, если распознан валидный barcode или SKU | Использовать внешние product API, что запрещено условиями |

| 2026-05-15 | Не скачивать YOLO-веса по сети по умолчанию, но искать локальные веса из `models/` и `runs/detect/...` | Требование локального запуска; в архиве есть обученные веса, а generic COCO fallback запрещён | Автоскачивание с HF Hub или generic `yolov8n.pt` |
| 2026-05-15 | Добавить `max_timestamp_ms` в `sample_frames()` | UI/smoke/test-duration limit должен останавливать декодирование видео, а не только обработку после чтения лишних кадров | Читать всё видео и прерывать только в pipeline |
| 2026-05-15 | Ввести безопасные runtime flags `SHELF_MSER_PROCESS_WIDTH`, `SHELF_MAX_TRACKS`, `SHELF_CODE_DECODE_MODE`, `SHELF_CODE_MAX_VARIANTS` | Они позволяют делать быстрые smoke/HF runs без изменения production defaults и без ухудшения detection recall в обычном режиме | Уменьшить качество default-детекции/QR-декодирования |
| 2026-05-15 | Сливать OCR-цены, разбитые на рубли и копейки (`129` + `99`) | Реальный OCR часто отделяет копейки; это повышает `price_card`/`price_default` без агрессивного угадывания | Оставлять `99` как отдельную ложную цену |
| 2026-05-15 | Добавить cross-field consistency для OCR/QR цен | QR `price1/price4` и OCR `price_default/price_card` описывают одни и те же бизнес-цены; безопасно заполнять пустые поля и менять местами явно инвертированные цены | Считать все price-поля независимыми |
| 2026-05-15 | Ужесточить cleanup `product_name` без удаления полезных процентов | Product name был слабым полем; нужно вырезать даты/barcode/SKU/цены, но не ломать названия вроде `Молоко 3.2%` | Отбрасывать все строки с `%` или цифрами |
| 2026-05-15 | Исправить eval JSON при отсутствии приватных данных | Скрипты должны быть честно воспроизводимы в публичной среде и не должны писать нулевые метрики как будто это результат алгоритма | Падать с ошибкой FileNotFound или append zero-row |
| 2026-05-15 | Add template layout priors from Lenta tag materials | The task templates keep QR, barcode, name, SKU/date and price zones in stable relative locations; broad rule-based ROIs can improve code/OCR attempts without touching detector recall | Keep only generic geometric right/bottom ROI heuristics |
| 2026-05-15 | Try template ROI decoding before geometric/full-crop QR/barcode decoding | Barcode and QR are the first matching key and also fill many content fields; ROI-first decoding is cheaper and more targeted than full-crop transforms | Run all heavy variants over the entire crop first |
| 2026-05-15 | Add per-tag pass80 diagnostics to eval_on_labeled.py | metric@80 depends on moving near-threshold tags over 80%, so aggregate avg_field is insufficient for prioritization | Only print overall metric and weakest fields |
| 2026-05-15 | Add conservative pass80 optimizer | Business-consistent field propagation can improve completeness without using GT: QR barcode -> barcode, QR price -> OCR price, price consistency and discount derivation | Fill every empty field heuristically, risking false positives |
| 2026-05-15 | Keep barcode -> QR synchronization behind SHELF_PASS80_SYNC_BARCODE_TO_QR | Some templates may have no QR, so inventing QR fields from a linear barcode can hurt correctness | Always copy barcode into qr_code_barcode |
| 2026-05-15 | Rebuild local catalog from all available local CSVs | Local CSV-derived catalog is allowed and can improve product_name by barcode/SKU without external APIs | Query external product databases, which is prohibited |
| 2026-05-15 | Preserve product-name percents while stripping dates, IDs, codes and prices | Product names may include values like 3.2%, while barcode/SKU/date tokens were polluting names | Remove all digits/percent tokens and lose useful product names |
