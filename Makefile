.PHONY: help install poetry-install \
        run poetry-run \
        test poetry-test \
        format poetry-format \
        eval poetry-eval \
        ceiling poetry-ceiling \
        catalog poetry-catalog \
        docker-build docker-run

# ── default ───────────────────────────────────────────────────────────────────
help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*##"}{printf "  %-28s %s\n", $$1, $$2}'

# ── install ───────────────────────────────────────────────────────────────────
install:  ## pip install runtime/dev deps
	pip install -r requirements.txt

poetry-install:  ## poetry install (ml + ocr + dev groups)
	poetry install --with ml,ocr,dev

# ── run UI ───────────────────────────────────────────────────────────────────
run:  ## start Gradio UI (plain python)
	PYTHONPATH=src python app.py

poetry-run:  ## start Gradio UI (poetry)
	PYTHONPATH=src poetry run python app.py

# ── tests ─────────────────────────────────────────────────────────────────────
test:  ## run test suite (plain python)
	PYTHONPATH=src pytest -q

poetry-test:  ## run test suite (poetry)
	PYTHONPATH=src poetry run pytest -q

# ── formatting ────────────────────────────────────────────────────────────────
format:  ## ruff + black (plain python)
	ruff check --fix .
	black .

poetry-format:  ## ruff + black (poetry)
	poetry run ruff check --fix .
	poetry run black .

# ── ceiling eval (GT bboxes — diagnostic only) ────────────────────────────────
ceiling:  ## ceiling eval with GT bboxes (plain python)
	PYTHONPATH=src python scripts/eval_ceiling.py \
	  --data-root Данные \
	  --ocr-engine auto \
	  --json-out reports/ceiling_latest.json \
	  --append-metrics

poetry-ceiling:  ## ceiling eval with GT bboxes (poetry)
	PYTHONPATH=src poetry run python scripts/eval_ceiling.py \
	  --data-root Данные \
	  --ocr-engine auto \
	  --json-out reports/ceiling_latest.json \
	  --append-metrics

# ── full pipeline eval ────────────────────────────────────────────────────────
eval:  ## full pipeline eval on labeled videos (plain python)
	PYTHONPATH=src python scripts/eval_on_labeled.py \
	  --data-root Данные \
	  --interval-ms 250 \
	  --detector hybrid \
	  --ocr-engine auto \
	  --ocr-top-k 3 \
	  --reports-dir reports/eval_latest \
	  --json-out reports/eval_latest.json \
	  --append-metrics

poetry-eval:  ## full pipeline eval on labeled videos (poetry)
	PYTHONPATH=src poetry run python scripts/eval_on_labeled.py \
	  --data-root Данные \
	  --interval-ms 250 \
	  --detector hybrid \
	  --ocr-engine auto \
	  --ocr-top-k 3 \
	  --reports-dir reports/eval_latest \
	  --json-out reports/eval_latest.json \
	  --append-metrics

# ── catalog ───────────────────────────────────────────────────────────────────
catalog:  ## rebuild local SKU catalog from labeled GT CSVs (plain python)
	PYTHONPATH=src python scripts/build_catalog.py Данные --out data/catalog.csv

poetry-catalog:  ## rebuild local SKU catalog from labeled GT CSVs (poetry)
	PYTHONPATH=src poetry run python scripts/build_catalog.py Данные --out data/catalog.csv

# ── docker ────────────────────────────────────────────────────────────────────
docker-build:  ## build Docker image
	docker build -t shelf .

docker-run:  ## run Docker container on port 7860
	docker run -p 7860:7860 shelf
