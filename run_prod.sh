#!/usr/bin/env bash
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd ~/Project/Lenta-Tech-Life-Hack-2026
exec >> reports/prod_run.log 2>&1

TIMEOUT=/usr/bin/timeout
POETRY=$(which poetry)

echo "==============================="
echo "Prod runner started: $(date)"
echo "TIMEOUT=$TIMEOUT POETRY=$POETRY"
echo "==============================="

# 1. Smoke 43_15
echo "[1/3] Smoke 43_15..."
$TIMEOUT 900 \
  env SHELF_WECHAT_QR_TIMEOUT=2 PYTHONPATH=src \
  $POETRY run python scripts/eval_on_labeled.py \
    --data-root Данные --videos 43_15 \
    --detector hybrid --ocr-engine auto --ocr-top-k 1 --interval-ms 500 \
    --json-out reports/smoke_43_15_phase_a.json
SMOKE_EXIT=$?
echo "[1/3] smoke exit=$SMOKE_EXIT"

if [ -f reports/smoke_43_15_phase_a.json ]; then
    python3 -c "
import json
with open('reports/smoke_43_15_phase_a.json') as f: d=json.load(f)
o=d.get('overall',{})
print(f'SMOKE 43_15 avg_field={o.get(\"avg_field\",0):.4f} metric@80={o.get(\"metric_80\",0):.4f} n={o.get(\"n_pass_80\",0)}/{o.get(\"n_gt\",0)}')
"
fi

# 2. Full production eval — только если smoke OK
if [ $SMOKE_EXIT -eq 0 ] && [ -f reports/smoke_43_15_phase_a.json ]; then
    echo "[2/3] Full production eval on all 5 videos..."
    START=$(date +%s)
    $TIMEOUT 14400 \
      env SHELF_WECHAT_QR_TIMEOUT=2 PYTHONPATH=src \
      $POETRY run python scripts/eval_on_labeled.py \
        --data-root Данные \
        --detector hybrid --ocr-engine auto --ocr-top-k 1 --interval-ms 500 \
        --json-out reports/eval_phase_a_full.json \
        --reports-dir reports/eval_phase_a_full/
    PROD_EXIT=$?
    END=$(date +%s)
    echo "[2/3] prod exit=$PROD_EXIT elapsed=$((END-START))s"

    if [ -f reports/eval_phase_a_full.json ]; then
        python3 -c "
import json
with open('reports/eval_phase_a_full.json') as f: d=json.load(f)
o=d.get('overall',{})
print(f'PROD avg_field={o.get(\"avg_field\",0):.4f} metric@80={o.get(\"metric_80\",0):.4f} n={o.get(\"n_pass_80\",0)}/{o.get(\"n_gt\",0)}')
for r in d.get('results',[]):
    print(f'  {r.get(\"video\")}: m={r.get(\"metric_80\",0):.3f} af={r.get(\"avg_field\",0):.3f} n={r.get(\"n_pass_80\",0)}/{r.get(\"n_gt\",0)}')
"
    fi
else
    echo "[2/3] SKIPPED smoke failed"
fi

# 3. Commit & push
echo "[3/3] Commit & push..."
TODAY=$(date +%F)
{
    echo ""
    echo "| $TODAY | smoke_43_15_phase_a | smoke with Phase A | see reports/smoke_43_15_phase_a.json |"
    echo "| $TODAY | full_prod_phase_a | full prod with Phase A | see reports/eval_phase_a_full.json |"
} >> docs/METRICS.md

git -c user.name="contributor" -c user.email="noreply@example.com" add \
  reports/smoke_43_15_phase_a.json \
  reports/eval_phase_a_full.json \
  reports/eval_phase_a_full/ \
  docs/METRICS.md 2>/dev/null

git -c user.name="contributor" -c user.email="noreply@example.com" \
  commit -m "metrics: Phase A production eval (smoke + full 5-video)" || echo "Nothing to commit"

git push origin nkolesnikov-test-big-work || echo "Push failed"

echo "==============================="
echo "Prod runner finished: $(date)"
echo "==============================="