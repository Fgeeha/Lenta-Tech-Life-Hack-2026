.PHONY: install run eval test format poetry-install poetry-run poetry-eval poetry-test poetry-format docker-build docker-run

install:  ## install runtime/dev dependencies with pip
	pip install -r requirements.txt

run:  ## start Gradio UI
	PYTHONPATH=src python app.py

eval:  ## run metric evaluation on labeled videos
	PYTHONPATH=src python scripts/eval_on_labeled.py

test:  ## run tests
	PYTHONPATH=src pytest -q

format:  ## ruff fix + black
	ruff check --fix .
	black .

docker-build:  ## build Docker image
	docker build -t shelf .

docker-run:  ## run Docker container
	docker run -p 7860:7860 shelf

poetry-install:  ## poetry install (all groups)
	poetry install --with ml,ocr,dev

poetry-run:  ## запустить Gradio UI
	poetry run python app.py

poetry-eval:  ## прогон метрики по размеченным видео
	poetry run python scripts/eval_on_labeled.py

poetry-test:  ## запустить тесты
	poetry run pytest -q

poetry-format:  ## ruff fix + black
	poetry run ruff check --fix .
	poetry run black .
