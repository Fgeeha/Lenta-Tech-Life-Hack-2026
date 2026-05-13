# Деплой на Hugging Face Spaces

## Ссылка

https://huggingface.co/spaces/lenta-hack/shelf-control

## Воспроизведение деплоя

### 1. Создать Space

```
Name:    shelf-control
SDK:     Gradio
Hardware: CPU basic (free)
Visibility: Public
```

### 2. Добавить remote и запушить

```bash
git remote add hf https://huggingface.co/spaces/lenta-hack/shelf-control
git push hf Master:main
```

### 3. Переменные среды (Settings → Variables)

```
SHELF_DETECTOR=yolo-tiled
```

### 4. Файлы в репо для Spaces

| Файл | Назначение |
|---|---|
| `app.py` | точка входа (импортирует `build_app`, вызывает `app.launch()`) |
| `requirements.txt` | зависимости (сгенерирован `poetry export`) |
| `models/pricetag_tiled_yolov8n.pt` | веса детектора (~6 МБ, закоммичены) |

### Обновить requirements.txt

```bash
poetry export -f requirements.txt --without-hashes --without dev -o requirements.txt
git add requirements.txt && git commit -m "chore: update requirements.txt"
git push hf Master:main
```

## Ограничения HF Spaces CPU

- Первый вызов: ~30–60 сек на загрузку моделей (PaddleOCR, YOLO)
- Обработка: ~2–5 мин на 30-секундное 4K видео
- Лимит в UI: первые 90 секунд видео (настраивается `_MAX_DURATION_SEC`)
- Интервал по умолчанию: 500 мс (баланс скорость/качество на CPU)

## Локальный запуск

```bash
sudo apt-get install libzbar0  # декодирование QR/EAN
poetry install
python app.py  # → http://localhost:7860
```

## Docker

```bash
docker build -t shelf-control .
docker run -p 7860:7860 shelf-control
```
