#!/usr/bin/env bash
set -u
export PATH=$HOME/.local/bin:/opt/homebrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
POETRY=$(command -v poetry || true)
if [ -z "$POETRY" ]; then
    echo "FATAL: poetry not found in PATH=$PATH" >&2
    exit 1
fi
cd ~/Project/Lenta-Tech-Life-Hack-2026
exec >> reports/day_run.log 2>&1

TIMEOUT=/usr/bin/timeout
POETRY=$(which poetry)

echo "==============================="
echo "Day runner started: $(date)"
echo "TIMEOUT=$TIMEOUT POETRY=$POETRY"
echo "==============================="

# 1. Ceiling Phase A (re-run with Phase A integrated)
echo "[1/4] Re-running ceiling eval with Phase A..."
$TIMEOUT 2400 \
  env SHELF_WECHAT_QR_TIMEOUT=2 PYTHONPATH=src \
  $POETRY run python scripts/eval_ceiling.py \
    --data-root Данные --ocr-engine auto \
    --json-out reports/ceiling_phase_a_v2.json
CEIL_EXIT=$?
echo "[1/4] ceiling exit=$CEIL_EXIT"

if [ -f reports/ceiling_phase_a_v2.json ]; then
    python3 -c "
import json
with open('reports/ceiling_phase_a_v2.json') as f: d=json.load(f)
o=d.get('overall',{})
print(f'CEILING v2 avg_field={o.get(\"avg_field\",0):.4f} metric@80={o.get(\"metric_80\",0):.4f} n={o.get(\"n_pass_80\",0)}/{o.get(\"n_gt\",0)}')
"
fi

# 2. Smoke 43_15 to verify timeout fix on production path
echo "[2/4] Smoke 43_15..."
$TIMEOUT 900 \
  env SHELF_WECHAT_QR_TIMEOUT=2 PYTHONPATH=src \
  $POETRY run python scripts/eval_on_labeled.py \
    --data-root Данные --videos 43_15 \
    --detector hybrid --ocr-engine auto --ocr-top-k 1 --interval-ms 500 \
    --json-out reports/smoke_43_15_phase_a.json
SMOKE_EXIT=$?
echo "[2/4] smoke exit=$SMOKE_EXIT"

if [ -f reports/smoke_43_15_phase_a.json ]; then
    python3 -c "
import json
with open('reports/smoke_43_15_phase_a.json') as f: d=json.load(f)
o=d.get('overall',{})
print(f'SMOKE 43_15 avg_field={o.get(\"avg_field\",0):.4f} metric@80={o.get(\"metric_80\",0):.4f} n={o.get(\"n_pass_80\",0)}/{o.get(\"n_gt\",0)}')
"
fi

# 3. Full production eval — only if smoke succeeded
if [ $SMOKE_EXIT -eq 0 ] && [ -f reports/smoke_43_15_phase_a.json ]; then
    echo "[3/4] Full production eval on all 5 videos..."
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
    echo "[3/4] prod exit=$PROD_EXIT elapsed=$((END-START))s"

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
    echo "[3/4] SKIPPED — smoke failed (exit=$SMOKE_EXIT)"
fi

# 4. Final commit & push
echo "[4/4] Final commit & push..."
TODAY=$(date +%F)
{
    echo ""
    echo "| $TODAY | ceiling_v2 | Phase A ceiling re-eval | see reports/ceiling_phase_a_v2.json |"
    echo "| $TODAY | smoke_43_15 | Phase A smoke | see reports/smoke_43_15_phase_a.json |"
    echo "| $TODAY | full_prod | Phase A full prod | see reports/eval_phase_a_full.json |"
} >> docs/METRICS.md

git -c user.name="contributor" -c user.email="noreply@example.com" add \
  reports/ceiling_phase_a_v2.json \
  reports/smoke_43_15_phase_a.json \
  reports/eval_phase_a_full.json \
  reports/eval_phase_a_full/ \
  docs/METRICS.md 2>/dev/null

git -c user.name="contributor" -c user.email="noreply@example.com" \
  commit -m "metrics: Phase A re-eval (ceiling + smoke + full prod)" || echo "Nothing to commit"

git push origin nkolesnikov-test-big-work || echo "Push failed, check manually"

echo "==============================="
echo "Day runner finished: $(date)"
echo "==============================="