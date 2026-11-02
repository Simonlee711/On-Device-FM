# Legacy synthetic demo outputs

These two files were produced by an **earlier version** of the demo notebook on the
**synthetic** PVC data (`pvc_10s_synth.h5`). They are kept only for provenance. They
are not clinical data, they do not correspond to any number in the manuscript, and
they are not inputs to any supported script.

| File | Contents |
|---|---|
| `pvc_predictions_legacy_synthetic.csv` | `patient_id,label,p_pvc` for 11,172 synthetic windows (515 positive; ROC-AUC ≈ 0.766) |
| `pvc_linear_probe_legacy_synthetic.pt` | linear-probe weights (`fc.weight` 1×256, `fc.bias`) from the same run |

Why they are superseded:

- the windows were fed to the backbone at the stored 25 Hz instead of being resampled to the 100 Hz pretraining rate;
- windows were split at random, so the same synthetic participant can appear in training and test;
- the CSV has no `task`, `model`, `regime`, `split` or `sample_id` column, so it cannot be paired with other models and has no validation rows for threshold selection.

The current notebook and `pvc/downstream_eval/binary_linear_prob.py` use the 100 Hz pretraining input and a participant-disjoint split, and write the canonical schema directly; `examples/synthetic_predictions.csv` is a canonical synthetic table produced that way.

If you need the legacy CSV in the canonical schema (for example to compare against an old run), convert it explicitly. It has no split or window identifier, so both must be declared:

```bash
python analysis/convert_predictions.py \
  --input legacy/pvc_predictions_legacy_synthetic.csv \
  --task PVC --model "HiMAE (legacy demo)" --regime Frozen \
  --split test --sample-id-from-row-order \
  --output results/legacy_predictions_canonical.csv
```
