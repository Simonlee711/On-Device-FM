#!/usr/bin/env bash
# End-to-end statistical workflow.
# Usage: bash analysis/run_statistical_workflow.sh PREDICTIONS_CSV DEVICE_RUNS_CSV [STATIC_CSV] [OUT_DIR]
#   PREDICTIONS_CSV : task,model,regime,split,person_id,y_true,y_score[,sample_id]
#   DEVICE_RUNS_CSV : model,run,latency_s,throughput_wps,energy_j[,battery_start_pct,battery_end_pct,duration_s,n_windows]
#   STATIC_CSV      : model,params_m,memory_mb,flops_gflops (optional; pass "" to skip)
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n 2,6p "$0"; exit 1
fi

abspath() { (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")"); }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PRED="$(abspath "$1")"
RUNS="$(abspath "$2")"
STATIC="${3:-}"
[ -n "$STATIC" ] && STATIC="$(abspath "$STATIC")"
OUT="${4:-$REPO/results}"
SEED=2024
NBOOT=1000

mkdir -p "$OUT"
cd "$REPO/analysis"

# 1. Table 1 (AUROC rows) + Fig. 1: participant-level paired AUROC bootstrap.
python bootstrap_auc.py --predictions "$PRED" --n-boot "$NBOOT" --seed "$SEED" --output-dir "$OUT"

# 2. Fig. 3 / Extended Data Tables 3-6: validation-only thresholds, fixed on test.
python select_validation_thresholds.py --predictions "$PRED" \
  --targets 0.95 0.90 0.80 --output "$OUT/validation_thresholds.csv"
python apply_operating_points.py --predictions "$PRED" \
  --thresholds "$OUT/validation_thresholds.csv" --output "$OUT/operating_points.csv"
python bootstrap_operating_points.py --predictions "$PRED" \
  --thresholds "$OUT/validation_thresholds.csv" \
  --n-boot "$NBOOT" --seed "$SEED" --output-dir "$OUT"
python prepare_fig3_results.py --input "$OUT/operating_points.csv" \
  --regime Fine-tuned --target-specificity 0.95 --output "$OUT/fig3_clinical.csv"

# 3. Table 1 (on-device rows) + Fig. 2: latency/throughput/energy summaries, Welch tests.
if [ -n "$STATIC" ]; then
  python summarize_device_energy.py --runs "$RUNS" --static "$STATIC" --output-dir "$OUT"
else
  python summarize_device_energy.py --runs "$RUNS" --output-dir "$OUT"
fi

echo "Done. Outputs in $OUT"
