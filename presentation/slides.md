---
marp: true
theme: default
paginate: true
style: |
  section { font-family: 'Segoe UI', sans-serif; }
  h1 { color: #c00; }
  h2 { color: #333; border-bottom: 2px solid #c00; }
  code { background: #f5f5f5; }
  .metric { font-size: 2em; font-weight: bold; color: #c00; }
---

# Полка под контролем
## Lenta Tech Life Hack 2026

**Автоматическое распознавание ценников с видео робота-сканера**

---
**Демо:** https://huggingface.co/spaces/Fgeeha/shelf-control
**Репо:** https://github.com/Fgeeha/Lenta-Tech-Life-Hack-2026

---

## Команда

| | |
|---|---|
| **Роль** | ML/CV Engineer + Backend/Deploy |
| **Стек** | Python, OpenCV, YOLOv8, PaddleOCR, Gradio |
| **Подход** | Соло-команда, полный цикл от данных до деплоя |

---

## Задача и данные

**Входные данные:**
- 5 размеченных видео (4K, 20fps, 15–89 сек)
- 274 GT ценника с 29 полями каждый (25_12-20, 25_2-10, 26_12-20, 43_15, 49_5)
- Ценники на рельсах стеллажей, смонтированы под 90°

**Сложности:**
- Ценники ≈50px при resize 4K→640 (стандартный YOLO провал)
- Угол обзора + motion blur + перспектива
- Нет разметки для обучения — только GT-CSV

**Метрика:** доля ценников, у которых ≥80% из 11 полей распознаны верно

---

## Архитектура

```
video.mp4
  │
  ├─[Adaptive Sampler]── оптический поток, шаг 200–500мс
  │
  ├─[YOLO-Tiled SAHI]─── 4K → тайлы 640×640 (stride 512)
  │                       обученный на 102 GT-кадрах, mAP50=0.776
  │
  ├─[ByteTrack]──────── 1 track_id = 1 ценник
  │                     top-K кандидатов по (area × резкость)
  │                     + best_qr_frame по резкости QR-зоны [Stage B]
  │
  ├─[QR / EAN-13]─────── WeChatQR (OpenCV 4.13) → полный QR URL
  │                       pyzbar + OTSU + 4 ориентации
  │                       fallback: best_qr_frame [Stage B]
  │
  ├─[OCR Pipeline]────── rot 90°CCW + deskew + upscale×5 + CLAHE
  │                       PaddleOCR EN → цены, скидки
  │
  ├─[Catalog]──────────── barcode → product_name + id_sku (456 строк)
  │
  ├─[Pass80 Optimizer]─── price1_qr↔price_default, price4_qr↔price_card
  │                        price2_qr = price1_qr × 0.95
  │                        barcode ↔ qr_code_barcode (sync)
  │
  └─[CSV]──────────────── 29 полей по схеме ТЗ
```

---

## Ключевые решения

### 1. Tile-based YOLO (главный прорыв)

| | Full-frame | Tiled |
|---|---|---|
| Размер ценника в модели | ~47px | **180–380px** |
| mAP50 | 0.368 | **0.776** |
| Recall (pipeline) | 55% | **100%** |

Без разметки → псевдо-лейблы из GT-CSV → 102 train / 22 val тайла

### 2. ByteTrack

Аккумулирует детекции через время → 100% recall на уровне видео

---

## Ключевые решения (2)

### 3. OCR bug fixes (каскадный эффект)

| Баг | Симптом | Исправление |
|---|---|---|
| `48%` → price=48 | price_card всегда неверный | strip % перед ценами |
| `_PRICE_RE` ≥2 цифр | `"99"` (копейки) → price_card=99 | **≥3 цифры** (+14pp price_card) |
| Скидка без минуса | discount_amount=нет | regex `[-–]?` опциональный |

### 4. Деривация полей из бизнес-логики Lenta

GT-анализ: `price4_qr==price_card` (96%), `discount_amount = -int((1-pc/pd)*100)%` (100%)

→ поля без QR выводим из OCR-цен

---

## Метрики (ceiling eval — GT bboxes, 5 видео, 274 ценника)

```
Поле               Точность    Примечание
─────────────────────────────────────────────
price_discount      97.1%      GT всегда «нет» → автоматическое совпадение
price_card          42.7%      OCR читает крупную карточную цену
price4_qr           46.7%      заполняется из price_card
discount_amount     21.9%      выводится из price_card / price_default
product_name         0.7%      OCR не справляется; каталог даёт точные данные
id_sku               5.1%      каталог: 379/456 записей теперь с id_sku ✓
qr_code_barcode      2.9%      3/274 расшифровано WeChatQR
price1_qr            7.3%      точно из QR URL; иначе ≈ price_default

metric@80% = 0.011 (3/274)  |  avg_field = 0.227
```

**Для 3 расшифрованных ценников**: 10–11/11 полей верны = **100% pass rate**.
**EAN-13 штрих-код**: 0 декодирований даже при 8× зуме — motion blur необратим.

---

## Почему metric@80% = 1.1%?

**Строгость метрики**: ≥9 из 11 полей должны быть верны **одновременно** для одного ценника.

| Без QR — максимум 5/11 полей: | С QR — 10–11/11 полей: |
|---|---|
| price_discount ✓ (авто) | + qr_code_barcode ✓ |
| price_card ✓ (OCR, 43%) | + barcode ✓ (sync) |
| price4_qr ✓ (из price_card) | + price1_qr ✓ |
| price_default ✗ (6% OCR) | + price2_qr ✓ (дериват) |
| barcode ✗ (0% — blur) | + price4_qr ✓ |
| id_sku ✗ (нет штрихкода) | + id_sku ✓ (каталог) |
| product_name ✗ (нет штрихкода) | + product_name ✓ |

**98.2%** GT-ценников имеют QR в разметке. Расшифровка QR = проход.
Без QR: P(pass) ≈ 43% × 6% × 0% (barcode) = **≈ 0%**

---

## Ключевое открытие: узкое место в QR-коде

Каждый ценник Lenta содержит QR-код с URL вида:
```
?b=4690491122587&p1=2631.57&p2=2499.99&p4=1899.99
```

Расшифровка QR → **6 полей сразу**: qr_code_barcode, barcode,
price1_qr, price2_qr, price4_qr, затем catalog → product_name + id_sku

**GT-анализ: 98.2% ценников имеют QR в разметке → расшифровка QR = проход metric@80% для любого тега.**

Но только 3/274 QR-кода читаемы — все три в видео 26\_12-20, кадр ts=15833 мс
(один конкретный ракурс робота). Остальные 271/274 QR-кодов **нечитаемы ни в одном
кадре видео** — это подтверждает мультикадровый тест (±300 мс, шаг 50 мс, 274 тега):

```
metric@80% без мультикадрового скана:   0.011 (3/274)
metric@80% с мультикадровым сканом ±300: 0.011 (3/274)  ← без изменений
```

**Фундаментальное ограничение**: motion blur камеры робота + малый QR (~40×40 px)
→ информация необратимо потеряна в исходном видео.

---

## Stage A+B: что сделано

| Улучшение | Статус |
|---|---|
| **WeChatQR** (OpenCV 4.13) → полный QR URL с ценами | ✅ |
| OpenCV **BarcodeDetector** для линейных кодов | ✅ |
| **id_sku** добавлен в каталог (379/456 записей) | ✅ |
| Каталог расширен: 266 → **456** строк | ✅ |
| WeChatQR запускается **до** pyzbar (приоритет URL над EAN) | ✅ |
| **RealESRGAN** x4plus: patch torchvision, модель подключена | ✅ |
| **QR-zone sharpness tracking**: ByteTrack хранит кадр с max резкостью QR-зоны | ✅ |
| **Мультикадровый fallback**: если top-K кандидатов не дали QR → пробуем QR-zone кадр | ✅ |
| `decode_qr_wechat_fast()`: WeChatQR только по QR-зоне (0.05-0.2 с vs 4.8 с полный) | ✅ |

**Для 3 расшифрованных ценников**: score 10–11/11 vs 9/11 ранее.
**Stage B**: производственный пайплайн теперь выбирает лучший кадр отдельно для QR и для OCR.

---

## Демо UI

**https://huggingface.co/spaces/fgeeha/shelf-control**

- Загрузи `.mp4` → нажми «Запустить» → получи CSV
- ByteTrack отбирает 3 лучших кадра на ценник
- WeChatQR сканирует каждый кандидат
- Градиентный прогресс-бар + превью в Gradio

---

## Ограничения и следующие шаги

**Фундаментальная проблема**: QR-коды на видео слишком маленькие/размытые.
- Типичный QR: ~50–80px в кропе → нечитаем без SR или лучших кадров
- EAN-13 штрих-код: 0 успешных декодирований даже при 8× зуме — blur необратим
- Ceiling: ≥9/11 требует расшифровки QR — без него максимум ≈5/11

**Реализовано (Stage B):**
- ByteTrack хранит кадр с наилучшей резкостью QR-зоны независимо от OCR-кандидатов
- `decode_qr_wechat_fast()`: 0.05-0.2 с/вызов (vs 4.8 с для полного каскада)
- Stage B не меняет метрику для ТЕКУЩИХ данных — QR нечитаем физически

**Следующие шаги для реального улучшения:**
1. **Лучшая камера на роботе**: статичные кадры без motion blur (>1 мс экспозиция)
2. **QR-специфичный SR**: EDSR на синтетике QR для восстановления ~40px кодов
3. **ONNX + INT8**: для развёртывания на роботе (rknn-ready)

---

## Ссылки

| |                                                     |
|---|-----------------------------------------------------|
| **Демо** | https://huggingface.co/spaces/fgeeha/shelf-control  |
| **Репо** | https://github.com/fgeeha/Lenta-Tech-Life-Hack-2026 |
| **Метрики** | docs/METRICS.md                                     |
| **Деплой** | docs/DEPLOYMENT.md                                  |

```
git clone https://github.com/fgeeha/Lenta-Tech-Life-Hack-2026
cd Lenta-Tech-Life-Hack-2026
poetry install && python app.py
```
