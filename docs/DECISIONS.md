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
