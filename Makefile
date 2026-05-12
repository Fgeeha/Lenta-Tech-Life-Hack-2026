.PHONY: install run eval test format docker-build docker-run

install:  ## poetry install (all groups)
	poetry install --with ml,ocr,dev

run:  ## запустить Gradio UI
	poetry run python app.py

eval:  ## прогон метрики по размеченным видео
	poetry run python scripts/eval_on_labeled.py

test:  ## запустить тесты
	poetry run pytest -q

format:  ## ruff fix + black
	poetry run ruff check --fix .
	poetry run black .

docker-build:  ## собрать Docker-образ
	docker build -t shelf .

docker-run:  ## запустить контейнер
	docker run -p 7860:7860 shelf
