#!/usr/bin/env bash
# Statistical workflow on an already-produced canonical prediction table.
#
# Usage:
#   bash analysis/run_statistical_workflow.sh --predictions CSV [--out DIR]
#        [--reference-model NAME] [--fig3-regime REGIME] [--fig3-target SPEC]
#        [--device-runs CSV [--static CSV]]
#
#   --predictions  task,model,regime,split,person_id,sample_id,y_true,y_score
#                  (val rows: threshold selection only; test rows: held out)
#   --device-runs  model,run,latency_s,throughput_wps -- externally collected device
#                  log; summarized only, nothing is measured here (optional)
#   --static       model,params_m,memory_mb (optional)
#
# Seed 2024 and B = 1,000 participant-level bootstrap replicates throughout.
set -euo pipefail

usage() { sed -n 2,14p "$0"; exit 1; }
abspath() { (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")"); }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PRED="" RUNS="" STATIC="" OUT="$REPO/results"
REF="HiMAE (1.2M)" FIG3_REGIME="Fine-tuned" FIG3_TARGET="0.95"
while [ $# -gt 0 ]; do
  case "$1" in
    --predictions) PRED="$(abspath "$2")"; shift 2 ;;
    --device-runs) RUNS="$(abspath "$2")"; shift 2 ;;
    --static) STATIC="$(abspath "$2")"; shift 2 ;;
    --out) mkdir -p "$2"; OUT="$(cd "$2" && pwd)"; shift 2 ;;
    --reference-model) REF="$2"; shift 2 ;;
    --fig3-regime) FIG3_REGIME="$2"; shift 2 ;;
    --fig3-target) FIG3_TARGET="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$PRED" ] || usage
[ -z "$STATIC" ] || [ -n "$RUNS" ] || { echo "--static requires --device-runs"; exit 1; }

SEED=2024
NBOOT=1000
PY="${PYTHON:-python}"
mkdir -p "$OUT"
cd "$REPO/analysis"

# 1. Table 1 AUROC + CIs, Results 2.1 dAUROC (bootstrap CI, paired DeLong P).
"$PY" bootstrap_auc.py --predictions "$PRED" --reference-model "$REF" \
  --n-boot "$NBOOT" --seed "$SEED" --output-dir "$OUT"

# 2. Fig. 3 / Extended Data Tables 3-8: validation-only thresholds, frozen on test.
"$PY" select_validation_thresholds.py --predictions "$PRED" \
  --targets 0.95 0.90 0.80 --output "$OUT/validation_thresholds.csv"
"$PY" apply_operating_points.py --predictions "$PRED" \
  --thresholds "$OUT/validation_thresholds.csv" --output "$OUT/operating_points.csv"
"$PY" bootstrap_operating_points.py --predictions "$PRED" \
  --thresholds "$OUT/validation_thresholds.csv" --reference-model "$REF" \
  --n-boot "$NBOOT" --seed "$SEED" --output-dir "$OUT"
"$PY" prepare_fig3_results.py --input "$OUT/operating_points.csv" \
  --regime "$FIG3_REGIME" --target-specificity "$FIG3_TARGET" --output "$OUT/fig3_clinical.csv"

# 3. Table 1 on-device rows / Fig. 2: summaries of the supplied device log (optional).
if [ -n "$RUNS" ]; then
  if [ -n "$STATIC" ]; then
    "$PY" summarize_device_benchmarks.py --runs "$RUNS" --static "$STATIC" --reference-model "$REF" --output-dir "$OUT"
  else
    "$PY" summarize_device_benchmarks.py --runs "$RUNS" --reference-model "$REF" --output-dir "$OUT"
  fi
fi

echo "Done. Outputs in $OUT"
