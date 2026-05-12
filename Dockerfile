FROM python:3.10-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Poetry
RUN pip install poetry==2.2.1

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --with ml,ocr --no-interaction

COPY src/ ./src/
COPY app.py ./

EXPOSE 7860

CMD ["python", "app.py"]
