#!/usr/bin/env bash
# Rebuild examples/synthetic_predictions.csv from the released synthetic data and checkpoint.
#
# Two frozen linear probes on the same participant-disjoint split (--split-seed 77):
#   "HiMAE (1.2M)"                               stage-5 bottleneck (the demo probe)
#   "HiMAE stage-1 probe (synthetic comparator)" stage-1 embedding, used only so that
#                                                 the paired analyses have a comparator
# Neither is a manuscript baseline; the file exercises the statistical workflow:
#   bash analysis/run_statistical_workflow.sh --predictions examples/synthetic_predictions.csv \
#        --out results/synthetic --fig3-regime Frozen
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$REPO"

"$PY" pvc/downstream_eval/binary_linear_prob.py --h5 pvc_10s_synth.h5 --checkpoint himae_synth.ckpt \
  --task PVC --model-name "HiMAE (1.2M)" --regime Frozen --embedding-stage last \
  --predictions-output "$TMP/stage5.csv"
"$PY" pvc/downstream_eval/binary_linear_prob.py --h5 pvc_10s_synth.h5 --checkpoint himae_synth.ckpt \
  --task PVC --model-name "HiMAE stage-1 probe (synthetic comparator)" --regime Frozen --embedding-stage 1 \
  --predictions-output "$TMP/stage1.csv"
"$PY" analysis/merge_predictions.py --inputs "$TMP/stage5.csv" "$TMP/stage1.csv" \
  --output examples/synthetic_predictions.csv
