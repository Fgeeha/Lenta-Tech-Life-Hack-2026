
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

## 2026-05-15 — Stage 4 template ROI / pass80 diagnostics pass

### Baseline before this pass

The repository already stored the following comparable baseline:

- historical 5-video ceiling: `metric@80%=0.004`, `avg_field=0.219`, `GT=274`;
- latest saved ceiling row: `metric@80%=0.011`;
- weakest fields: `product_name=0.008`, `barcode=0.019`, `qr_code_barcode=0.019`, `id_sku=0.048`, `price_default=0.069`, `price1_qr=0.082`;
- stronger fields: `price_card≈0.420`, `price4_qr≈0.458`, `price_discount≈0.972`;
- baseline test suite before this task: `99 passed`.

### Commands run in this pass

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/eval_on_labeled.py \
  --data-root /mnt/data/nonexistent \
  --ocr-engine none \
  --json-out reports/eval_missing_stage4.json \
  --reports-dir reports/missing_diag
PYTHONPATH=src python scripts/eval_ceiling.py \
  --data-root /mnt/data/nonexistent \
  --ocr-engine none \
  --json-out reports/ceiling_missing_stage4.json
SHELF_MSER_PROCESS_WIDTH=480 SHELF_MAX_TRACKS=2 SHELF_CODE_DECODE_MODE=off \
PYTHONPATH=src python - <<'PY'
from shelf import pipeline
pipeline.run(
    '/mnt/data/26_2-10.mp4', detector_name='mser', interval_ms=1000,
    adaptive=False, min_hits=1, max_duration_sec=0.01,
    ocr_engine_name='none', ocr_top_k=1, max_ocr_variants=1,
    output_csv='reports/smoke_stage4.csv', debug_dir='reports/smoke_debug',
)
PY
PYTHONPATH=src python scripts/build_catalog.py '#U0414#U0430#U043d#U043d#U044b#U0435' --out data/catalog.csv
```

### Result in this environment

- `pytest`: `113 passed`.
- Smoke-test on `/mnt/data/26_2-10.mp4`: pipeline finished, `2` rows, `29` columns.
- `data/catalog.csv`: rebuilt from available local CSV files, `266` rows.
- Full 5-video metric was **not** recomputed: two private labeled folders (`25_12-20`, `49_5`) are absent from this archive, and a partial ceiling run over available videos with local runtime constraints exceeded the 300s tool limit.  No fake after-metric is claimed.

### New reports

`eval_on_labeled.py --reports-dir` writes:

- `matched_tags_debug.csv` — one row per GT tag with `matched_by`, correct field count, missing/wrong fields and predicted/GT values;
- `field_accuracy.csv` — field accuracy plus fill-rate;
- `pass80_candidates.csv` — matched tags that are already relatively close to the 80% threshold;
- `failed_near_threshold.csv` — non-pass tags close enough that 1-2 better fields may flip the metric.

### Expected metric impact when full GT is available

The largest expected gains are in QR/barcode and price propagation:

1. template ROI decoding should increase QR/barcode attempts in the exact regions where templates place codes;
2. QR price fields now fill missing `price_default`/`price_card` more consistently;
3. pass80 optimizer should flip near-threshold tags without changing the CSV schema;
4. catalog lookup can raise `product_name` when barcode/SKU is valid.

These are expected gains, not claimed measured gains, until the complete five-video GT set is mounted locally.
