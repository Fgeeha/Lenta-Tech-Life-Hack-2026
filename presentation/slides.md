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
- 3 размеченных видео (4K, 20fps, 14–42 сек)
- 157 GT ценников с 29 полями каждый
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
  │                     лучший кадр = max(area × резкость)
  │
  ├─[OCR Pipeline]────── rot 90°CCW + deskew + upscale×5 + CLAHE
  │                       PaddleOCR EN → цены, скидки
  │
  ├─[QR / EAN-13]─────── pyzbar + 4 ориентации + OTSU threshold
  │
  ├─[Field Derivation]─── price4_qr←price_card, discount_amount←prices
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

## Метрики

```
                    Pipeline      Ceiling (GT bboxes)
metric@80%:         0.013         0.006
avg_field:          0.193         0.220
Detection recall:   157/157       —
price_card:         ~52%          51.8%
price4_qr (deriv):  ~53%          52.7%
discount_amount:    ~15%          14.6%
```

**История:**

```
baseline MSER         →  metric=0.000  avg_field=0.091
+ YOLO-tiled          →  metric=0.000  avg_field=0.154
+ parser bug fixes    →  metric=0.000  avg_field=0.190
+ price regex fix     →  metric=0.013  avg_field=0.220
+ discount derivation →  metric=0.013  avg_field=0.221
```

---

## Демо UI

**https://huggingface.co/spaces/fgeeha/shelf-control**

- Загрузи `.mp4` → нажми «Запустить» → получи CSV
- Первые 90 сек видео (CPU basic, free tier)
- Детектор: YOLO-Tiled по умолчанию

![Интерфейс](../docs/ui_screenshot.png)

---

## Ограничения и масштабирование

**Что не работает сейчас:**
- `product_name` 0% — EN-модель не знает кириллицу
- `id_sku` 3% — 12-значный артикул в мелком шрифте
- QR-коды ≈50px — слишком маленькие для pyzbar
- Максимум без QR: **8/11 = 72.7%** (ниже порога 80%)

**Следующие шаги:**
- Ручная разметка 200 ценников + fine-tune PaddleOCR RU → product_name ≥70%
- TrOCR / EasyOCR на name-zone → id_sku
- ONNX + INT8 для rknn (упоминалось в ТЗ как плюс)
- Docker Compose + pipeline-параллелизм для реального робота

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
