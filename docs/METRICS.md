# История метрики

Метрика: доля ценников, у которых ≥80% полей распознаны верно.

| Дата | Commit | Видео | Метрика | Примечание |
|------|--------|-------|---------|------------|
| —    | —      | —     | —       | таблица пустая до этапа 8 |
| 2026-05-12 | 3eda52f | 43_15 | 0.000 | matched=23/29 avg_field=0.09 |
| 2026-05-13 | 85155e1 | pseudo-labels | — | train=121boxes/24frames val=24boxes/2frames hq=0 |
| 2026-05-13 | 85155e1 | YOLO-train | — | mAP50=0.368 mAP50-95=0.100 epochs=50 batch=8 |
| 2026-05-13 | 85155e1 | 25_12-20 | 0.000 | matched=0/57 avg_field=0.00 |
| 2026-05-13 | 85155e1 | 25_12-20 | 0.000 | matched=57/57 avg_field=0.13 |
| 2026-05-13 | 85155e1 | 26_12-20 | 0.000 | matched=71/71 avg_field=0.10 |
| 2026-05-13 | 85155e1 | 43_15 | 0.000 | matched=29/29 avg_field=0.10 |
| 2026-05-13 | 574d2f4 | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 574d2f4 | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 3c207bf | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 3c207bf | 43_15 | 0.000 | matched=2/29 avg_field=0.09 |
| 2026-05-13 | 3c207bf | 43_15 | 0.000 | matched=29/29 avg_field=0.15 |
| 2026-05-13 | 3c207bf | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 3c207bf | ceiling (GT bboxes) | 0.000 | 25_12-20=0.000  26_12-20=0.000  43_15=0.000 |
| 2026-05-13 | 3c207bf | 25_12-20 | 0.000 | matched=55/57 avg_field=0.14 |
| 2026-05-13 | 3c207bf | 26_12-20 | 0.000 | matched=71/71 avg_field=0.14 |
| 2026-05-13 | 3c207bf | 43_15 | 0.000 | matched=29/29 avg_field=0.17 |
| 2026-05-13 | 459ec0d | ceiling (GT bboxes) | 0.006 | 25_12-20=0.000  26_12-20=0.014  43_15=0.000 |
| 2026-05-13 | 459ec0d | 25_12-20 | 0.000 | matched=55/57 avg_field=0.16 |
| 2026-05-13 | 459ec0d | 26_12-20 | 0.028 | matched=71/71 avg_field=0.16 |
| 2026-05-13 | 459ec0d | 43_15 | 0.000 | matched=29/29 avg_field=0.24 |
| 2026-05-13 | 432c971 | 25_12-20 | 0.000 | matched=57/57 avg_field=0.20 |
| 2026-05-13 | 432c971 | 26_12-20 | 0.028 | matched=71/71 avg_field=0.18 |
| 2026-05-13 | 432c971 | 43_15 | 0.000 | matched=29/29 avg_field=0.20 |
| 2026-05-13 | 2bf2154 | ceiling (GT bboxes) | 0.006 | 25_12-20=0.000  26_12-20=0.014  43_15=0.000 |
| 2026-05-14 | 3029815 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | 3029815 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | 3029815 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | d5ada29 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | d5ada29 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | 7221017 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | f687799 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-14 | f687799 | ceiling (GT bboxes) | 0.004 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.014  43_15=0.000  49_5=0.000 |
| 2026-05-15 | d40a4cd | ceiling (GT bboxes) | 0.011 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |
| 2026-05-15 | bd06c60 | ceiling (GT bboxes) | 0.011 | 25_12-20=0.000  25_2-10=0.000  26_12-20=0.042  43_15=0.000  49_5=0.000 |


## 2026-05-15 — Stage 3 reproducibility / OCR-QR postprocessing pass

### Baseline from repository docs before this pass

- Historical full 5-video ceiling baseline from `docs/CEILING_ANALYSIS.md`: `metric@80%=0.004`, `avg_field=0.219`, `GT=274`.
- Latest saved ceiling rows in this file before Stage 3: `metric@80%=0.011` for 5 videos (`25_12-20=0.000`, `25_2-10=0.000`, `26_12-20=0.042`, `43_15=0.000`, `49_5=0.000`). That compact row did not store weighted `avg_field`.
- Weakest fields from the archived ceiling analysis: `product_name=0.008`, `barcode=0.019`, `qr_code_barcode=0.019`, `id_sku=0.048`, `price_default=0.069`, `price1_qr=0.082`. Stronger fields: `price_card=0.420`, `price4_qr=0.458`, `price_discount=0.972`.
- Baseline test suite in the supplied archive: `85 passed`.

### Commands run in this pass

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/eval_on_labeled.py   --data-root /mnt/data/nonexistent   --interval-ms 250 --detector hybrid --ocr-engine auto --ocr-top-k 3   --json-out /mnt/data/work/eval_missing_after.json
PYTHONPATH=src python scripts/eval_ceiling.py   --data-root /mnt/data/nonexistent --ocr-engine none   --json-out /mnt/data/work/ceiling_missing_after.json
SHELF_MSER_PROCESS_WIDTH=480 SHELF_MAX_TRACKS=2 SHELF_CODE_DECODE_MODE=off   PYTHONPATH=src python - <<'PY'
from shelf import pipeline
pipeline.run('/mnt/data/26_2-10.mp4', detector_name='mser', interval_ms=1000,
             adaptive=False, min_hits=1, max_duration_sec=0.01,
             ocr_engine_name='none', ocr_top_k=1, max_ocr_variants=1,
             output_csv='/mnt/data/work/smoke_off.csv')
PY
```

### Availability of GT videos

The archive used in this environment does not contain the private 5-video folders under `Данные/25_12-20`, `Данные/25_2-10`, `Данные/26_12-20`, `Данные/43_15`, `Данные/49_5`. Therefore full `avg_field`, `metric@80%`, per-video and detection-recall numbers cannot be honestly recomputed here. Both eval scripts now exit successfully and write JSON that lists all missing video/CSV paths instead of appending misleading zero-metric rows.

### Smoke-test on attached video `26_2-10.mp4`

- Video opened and the pipeline completed on a deliberately tiny first-frame smoke run.
- Output CSV schema: `29/29` expected columns.
- Rows in smoke CSV: `2` candidate rows.
- Barcode count: `0`; QR barcode count: `0` in this smoke mode because `SHELF_CODE_DECODE_MODE=off` was used to avoid expensive QR/barcode decoding during the single-frame runtime check.
- Fill-rate in smoke mode: metadata/bbox/timestamp filled for `2/2` rows; OCR-dependent fields intentionally empty without OCR; absent optional fields remain `нет` rather than `NaN` text in the CSV.

### After-code verification

- Test suite after this pass: `99 passed`.
- Full metrics after this pass: not recomputed because the private GT videos are absent.
- Eval reproducibility improved: `scripts/eval_on_labeled.py` and `scripts/eval_ceiling.py` now produce structured missing-data JSON with `barcode_count=0`, `qr_barcode_count=0`, and 29-field fill-rate skeletons where applicable.

### Expected impact when GT videos are available

The changes target the previously weakest groups without modifying detection defaults:

- price fields: split ruble/kopeck OCR boxes (`129` + `99`) are merged; card/default inversion is corrected; QR price fields can fill OCR prices and derive discount;
- product_name: cleanup removes dates, barcodes, SKU-like long IDs and explicit price tokens while preserving meaningful product percentages such as `3.2%`;
- barcode/SKU: EAN-13 remains strict and 12-digit SKU values remain SKU-only;
- QR/barcode runtime: full mode remains default; `SHELF_CODE_DECODE_MODE=fast|off`, `SHELF_CODE_MAX_VARIANTS`, `SHELF_MAX_TRACKS`, and `SHELF_MSER_PROCESS_WIDTH` are debug/HF-friendly feature flags, not metric hacks.
