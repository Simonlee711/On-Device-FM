# Reproducibility

This file maps every manuscript figure and main result to the script that produces it, the files it needs, the input schema, the command, the random seed and the output.

## What can and cannot be regenerated here

The clinical datasets, the trained clinical checkpoints, the baseline model implementations (ResNet, PaPaGei, EfficientNet, MAE1D), the Wear OS benchmarking application and the original device telemetry are proprietary and are **not** in this repository. Consequently, no manuscript number can be regenerated from this repository alone. Scripts therefore fall into three kinds, and every row below says which kind it is:

| Kind | Meaning |
|---|---|
| **Regenerates** | Computes the result from files shipped in this repository (synthetic data, synthetic checkpoint, model definition). |
| **Processes supplied results** | Deterministic or bootstrap statistics over a prediction table that must be supplied. With the clinical prediction table it reproduces the manuscript numbers; with `examples/synthetic_predictions.csv` it runs end to end on synthetic data. It cannot create the clinical predictions. |
| **Summarizes supplied device logs** | Descriptive statistics over latency/throughput values that were measured on the watch by external software. It does not measure latency, throughput or anything else. |

Shipped artifacts:

| File | What it is |
|---|---|
| `pvc_10s_synth.h5`, `pvc_10s_synth_metadata.csv` | synthetic PVC windows (25 Hz, 10 s; 55,861 windows, 18 synthetic participants) |
| `himae_synth.ckpt` | HiMAE pretraining autoencoder checkpoint (main-text configuration) |
| `examples/synthetic_predictions.csv` | canonical predictions on the synthetic data (two frozen probes), rebuilt by `examples/make_synthetic_predictions.sh` |
| `legacy/` | superseded outputs of an earlier demo notebook; not inputs to any supported script (see `legacy/README.md`) |

## Map of manuscript items

| Manuscript item | Producing script(s) | Kind | Required files | Seed | Output |
|---|---|---|---|---|---|
| Table 1 AUROC and 95% CI (PVC, hypertension; frozen, fine-tuned) | `analysis/bootstrap_auc.py` | Processes supplied results | clinical prediction table, [canonical schema](#canonical-prediction-schema) (not provided) | 2024, B = 1,000 | `auroc_summary.csv`: `auroc`, `participant_bootstrap_ci_low/_high` |
| Results 2.1 ΔAUROC, 95% CI, P | `analysis/bootstrap_auc.py` | Processes supplied results | same | 2024, B = 1,000 | `auroc_paired_differences.csv`: `delta_auroc`, `participant_bootstrap_ci_low/_high`, `paired_delong_p_value` |
| Table 1 parameters, memory (HiMAE) | `analysis/model_complexity.py` | Regenerates | model definition only | none | `model_complexity.csv`, row `pretraining_autoencoder` (see [provenance](#model-and-measurement-provenance)) |
| Table 1 parameters / memory (baselines) | none | — | externally supplied constants | — | — |
| Table 1 CPU latency and throughput, Results 2.2 | `analysis/summarize_device_benchmarks.py` | Summarizes supplied device logs | device log `model,run,latency_s,throughput_wps` (not provided) | none | `device_summary.csv`, `device_welch_tests.csv` |
| Fig. 1 panels 1–2 (cohort, workflow) | none (schematic) | — | — | — | — |
| Fig. 1 panel 3 (ROC curves) | `analysis/prepare_fig1_results.py` → `plot/fig1.py` | Processes supplied results | clinical prediction table (frozen probes + ResNet; PVC, hypertension, laboratory tasks) | none | `figures/<task>_roc.{png,pdf}`, `figures/roc_summary.csv` |
| Fig. 2 panels 1–2 (acquisition, benchmark schematic) | none (schematic) | — | — | — | — |
| Fig. 2 panel 3 (ratios to HiMAE) | `analysis/summarize_device_benchmarks.py` → `plot/fig2_3.py --efficiency-csv --baseline-model "HiMAE (1.2M)"` | Summarizes supplied device logs | device log + static constants (not provided) | none | `fig2_efficiency.csv`, `figures/*_relative.png` |
| Fig. 3 bar heights and CIs | `analysis/select_validation_thresholds.py` → `analysis/bootstrap_operating_points.py` | Processes supplied results | clinical prediction table with `val` and `test` rows | 2024, B = 1,000 | `operating_points_bootstrap.csv`: `tp`, `tp_ci_low`, `tp_ci_high` |
| Fig. 3 drawing | `analysis/apply_operating_points.py` → `analysis/prepare_fig3_results.py` → `plot/fig2_3.py --clinical-csv` | Processes supplied results | as above | none | `figures/clinical_*.png` (point estimates for one regime/target per call; this script does not draw the CI error bars or the three-target layout of the published figure) |
| Extended Data Tables 3–5 | `analysis/bootstrap_operating_points.py` | Processes supplied results | as Fig. 3 | 2024, B = 1,000 | `operating_points_bootstrap.csv` (one row per task/regime/target/model, each metric with `_ci_low/_ci_high`) |
| Extended Data Tables 6–8, Results 2.3 ΔTP | `analysis/bootstrap_operating_points.py` | Processes supplied results | as Fig. 3 | 2024, B = 1,000 | `operating_points_paired_tp.csv`: `delta_tp`, `ci_low`, `ci_high`, `p_value` |
| Fig. 4 (frozen and few-shot AUROC) | `pvc/downstream_eval/fewshot_finetune.py` → `plot/fig4.py` | HiMAE curves: Regenerates *given the clinical data and checkpoint* (not provided). Baseline curves: Processes supplied results | clinical HDF5 windows + clinical HiMAE checkpoint (not provided); baseline rows appended to the same CSV | `--split-seed 77`, `--run-seed 1000` + run index | `finetuning.csv`, `figures/fewshot_<dataset>.{png,pdf}` |
| Extended Data Tables 1–2 (cohort characteristics) | none | — | clinical metadata (not provided) | — | — |
| HiMAE pretraining (Methods) | `pretrain/himae.py` | Regenerates given unlabeled PPG shards (not provided) | HDF5 shards + index CSV | `set_seed(42)` | Lightning checkpoint |
| Architecture matches Methods | `tests/test_architecture.py` | Regenerates | `himae_synth.ckpt` | none | pytest pass/fail |

Repository-only analyses that are not manuscript display items:

| Item | Script(s) | Kind | Required files | Seed | Output |
|---|---|---|---|---|---|
| Synthetic PVC linear probe + canonical export | `HiMAE_PVC_Detection.ipynb` or `pvc/downstream_eval/binary_linear_prob.py` | Regenerates | `pvc_10s_synth.h5`, `himae_synth.ckpt` | `--seed 7`, `--split-seed 77` | `results/pvc_predictions.csv` (canonical), `results/pvc_linear_probe.pt` |
| Canonical synthetic example | `examples/make_synthetic_predictions.sh` | Regenerates | as above | as above | `examples/synthetic_predictions.csv` |
| Statistical workflow on synthetic data | `analysis/run_statistical_workflow.sh` | Processes supplied results | `examples/synthetic_predictions.csv` | 2024, B = 1,000 | all `analysis/` outputs above |
| Multi-resolution (per-stage) probes | `pvc/downstream_eval/multiresolution_probe.py` → `plot/fig_multiresolution.py` | Regenerates | `pvc_10s_synth.h5`, `himae_synth.ckpt` | `--split-seed 77`, bootstrap `--seed 2024` | `multires_probe_summary.csv`, `multires_probe_paired.csv`, `multires_predictions.csv`, `figures/multiresolution_probe.{png,pdf}` |
| Confusion-matrix grid | `plot/e-fig1_3.py` | Processes supplied results | threshold-sweep counts `threshold,task,model,regime,sensitivity,specificity,tp,fp,tn,fn` | none | `figures/extended_figure1_3.{png,pdf}` |

`plot/e-fig1_3.py` selects rows at one common **score** threshold (default 0.05). It does not read `operating_points.csv` and does not use validation-selected thresholds, so it is not the method behind Extended Data Tables 3–5 or Fig. 3.

## Commands

All commands run from the repository root with the environment described in `README.md`.

### Statistical workflow (Table 1 AUROC, Results 2.1, Fig. 3, Extended Data Tables 3–8)

```bash
bash analysis/run_statistical_workflow.sh --predictions PREDICTIONS.csv --out results \
  [--reference-model "HiMAE (1.2M)"] [--fig3-regime Fine-tuned] [--fig3-target 0.95] \
  [--device-runs DEVICE_LOG.csv [--static STATIC.csv]]
```

This runs, with seed 2024 and B = 1,000:

```bash
python analysis/bootstrap_auc.py --predictions P.csv --reference-model "HiMAE (1.2M)" --n-boot 1000 --seed 2024 --output-dir results
python analysis/select_validation_thresholds.py --predictions P.csv --targets 0.95 0.90 0.80 --output results/validation_thresholds.csv
python analysis/apply_operating_points.py --predictions P.csv --thresholds results/validation_thresholds.csv --output results/operating_points.csv
python analysis/bootstrap_operating_points.py --predictions P.csv --thresholds results/validation_thresholds.csv \
  --reference-model "HiMAE (1.2M)" --n-boot 1000 --seed 2024 --output-dir results
python analysis/prepare_fig3_results.py --input results/operating_points.csv --regime Fine-tuned --target-specificity 0.95 --output results/fig3_clinical.csv
```

To run it end to end on the shipped synthetic example (no clinical data needed):

```bash
bash analysis/run_statistical_workflow.sh --predictions examples/synthetic_predictions.csv \
  --out results/synthetic --fig3-regime Frozen
```

Expected synthetic output: HiMAE (1.2M) test AUROC 0.909 (11,209 windows, 197 positive, 4 participants). With only four test participants the participant-bootstrap CIs are very wide; the synthetic run demonstrates the pipeline, not the manuscript's statistical power.

### Producing a canonical prediction table

- HiMAE frozen and fine-tuned predictions: `pvc/downstream_eval/fewshot_finetune.py ... --include-frozen --shots ... all --predictions-output P.csv` (regimes `Frozen` and `Fine-tuned`, run `--predictions-run`, default 0).
- Frozen linear probe only: `pvc/downstream_eval/binary_linear_prob.py --predictions-output P.csv`.
- Per-model files are combined with `python analysis/merge_predictions.py --inputs A.csv B.csv ... --output P.csv`, which re-validates the merged table.
- Legacy exports (`patient_id,label,p_pvc`) are converted with `analysis/convert_predictions.py`; missing split or window identifiers must be declared explicitly (`--split`, `--sample-id-from-row-order`).

### Fig. 1 panel 3

```bash
python analysis/prepare_fig1_results.py --input P.csv --regime Frozen --output results/fig1_predictions.csv
python plot/fig1.py --input results/fig1_predictions.csv --output-dir figures
```

### Table 1 on-device rows and Fig. 2

```bash
python analysis/model_complexity.py --output results/model_complexity.csv --static-output results/himae_static.csv
python analysis/summarize_device_benchmarks.py --runs DEVICE_LOG.csv --static STATIC.csv \
  --reference-model "HiMAE (1.2M)" --output-dir results
python plot/fig2_3.py --efficiency-csv results/fig2_efficiency.csv --output-dir figures --baseline-model "HiMAE (1.2M)"
```

`STATIC.csv` must contain one row per model; for the baselines these constants are externally supplied.

### Fig. 4

```bash
python pvc/downstream_eval/fewshot_finetune.py --h5 DATA.h5 --checkpoint CKPT --dataset PVC \
  --model-name "HiMAE (1.2M)" --shots 256 512 1024 2048 all --runs 5 --include-frozen --output results/finetuning.csv
python plot/fig4.py --input results/finetuning.csv --output-dir figures
```

On the synthetic files (`--h5 pvc_10s_synth.h5 --checkpoint himae_synth.ckpt`) this regenerates synthetic curves only.

## Canonical prediction schema

Defined and validated in `analysis/prediction_schema.py`; every statistical script reads exactly this schema.

```text
task,model,regime,split,person_id,sample_id,y_true,y_score
```

| Column | Meaning |
|---|---|
| `task` | task name, e.g. `PVC` |
| `model` | display name, e.g. `HiMAE (1.2M)`; must match `--reference-model` |
| `regime` | e.g. `Frozen`, `Fine-tuned` |
| `split` | `val` (threshold selection only) or `test` (held out) |
| `person_id` | participant; bootstrap replicates resample participants |
| `sample_id` | window identifier, unique within participant |
| `y_true` | 0/1 |
| `y_score` | finite positive-class score; a window is positive when `y_score >= threshold` |

Validation rejects missing values, non-binary labels, non-finite scores and duplicated windows. Paired analyses (`analysis/_bootstrap.py::wide_by_model`) additionally reject, within each task/regime: label disagreement between models for the same `(person_id, sample_id)` (checked on the raw rows, before deduplication); duplicated pairing identifiers within a model; and window sets that differ between models.

## Statistical methods and column provenance

The bootstrap resamples participants with replacement (all windows of a sampled participant kept together), B = 1,000, seed 2024; CIs are 2.5th/97.5th percentiles. Every replicate applies one resample to all models of a task/regime, so paired differences use identical participants with identical multiplicities.

**AUROC (`analysis/bootstrap_auc.py`).** Methods states that ΔAUROC P values come from the paired DeLong test and the 95% CIs from the paired participant-level bootstrap. The reported P values include values such as P < 0.001 and P = 0.013, which a 1,000-replicate bootstrap cannot produce (its smallest two-sided P is 2/1,001 ≈ 0.002), so they are DeLong P values. The script therefore labels every column by method:

| Column | Method | Reported in the manuscript |
|---|---|---|
| `auroc` | AUROC on the full test set | Table 1 point estimate |
| `participant_bootstrap_ci_low/_high` (summary) | percentile CI of AUROC | Table 1 CI |
| `delta_auroc` | AUROC_reference − AUROC_comparator | Results 2.1 |
| `participant_bootstrap_ci_low/_high` (paired) | percentile CI of paired bootstrap differences | Results 2.1 CI |
| `paired_delong_p_value` | ordinary two-sided paired DeLong test (DeLong 1988; Sun & Xu 2014), windows treated as independent | Results 2.1 P |
| `paired_delong_se/_z/_wald_ci_*`, `delong_*` (summary) | Wald quantities from the same DeLong covariance | no |
| `clustered_delong_*` | participant-clustered DeLong (Obuchowski 1997) | no (sensitivity analysis) |
| `participant_bootstrap_p_value` | 2·min{Pr(d ≤ 0), Pr(d ≥ 0)}, (k+1)/(B+1) correction | no (sensitivity analysis) |

`tests/test_statistics.py` checks that `paired_delong_p_value` equals `delong_paired_test(y, a, b)` without clusters, that the DeLong implementation matches the direct O(M·N) formula (with ties), that the clustered variant reduces to it with one window per participant, and that the CI columns equal the percentile interval of the paired bootstrap differences.

**Operating points.** `select_validation_thresholds.py` reads only `split == val` rows and picks, per task/model/regime/target, the threshold minimizing |validation specificity − target| (ties: higher validation sensitivity, then larger threshold). `apply_operating_points.py` applies it once to the test rows. `bootstrap_operating_points.py` applies the same fixed thresholds before resampling and recomputes TP/FP/TN/FN, sensitivity, specificity, PPV, NPV, balanced accuracy, Youden index and ΔTP = TP_reference − TP_comparator in each replicate; the ΔTP P value is 2·min{Pr(d ≤ 0), Pr(d ≥ 0)} with the (k+1)/(B+1) correction. `tests/test_statistics.py` contains synthetic fixtures with known thresholds, known confusion matrices and a known paired ΔTP, and checks that test rows cannot change thresholds and that every replicate equals the multiplicity-weighted per-participant counts at the frozen threshold.

**Device logs.** Mean, median, 95th percentile, SD and a 95% t-interval of the mean across passes; two-sided Welch t-tests against the reference model; no multiplicity adjustment.

## Model and measurement provenance

Three HiMAE artifacts are involved and they are not interchangeable:

| Artifact | Where | Parameters |
|---|---|---|
| Pretraining autoencoder (encoder + decoder) | `pvc/utils/model_arch/himae.py` (`PAPER_CONFIG`), trained by `pretrain/himae.py`, weights in `himae_synth.ckpt` | 1,245,953 |
| Downstream classifier (encoder + temporal mean-pool + linear head) | `fewshot_finetune.py::EncoderClassifier`, `binary_linear_prob.py` | 700,929 (encoder 700,672 + head 257) |
| Deployed inference artifact (TorchScript, run by the Wear OS app) | proprietary deployment toolchain; not released | not determinable here |

`analysis/model_complexity.py` reports the first two from the model definition (`scope = pretraining_autoencoder | downstream_classifier`):

| Scope | params | memory (fp32) |
|---|---|---|
| `pretraining_autoencoder` | 1.246 M | 4.98 MB = 4.75 MiB |
| `downstream_classifier` | 0.701 M | 2.80 MB = 2.67 MiB |

Relation to Table 1 (HiMAE column): the parameter count (1.2M) and memory (4.8) correspond to the pretraining autoencoder, with memory in MiB (`memory_mib`). Baseline parameters and memory are externally supplied constants.

Table 1 CPU latency and throughput were measured on a Samsung Galaxy Watch 8 by the Wear OS benchmarking application on the deployed inference artifact. Neither the application, the deployed artifacts nor the telemetry are released. `analysis/summarize_device_benchmarks.py` only summarizes a supplied log of those measurements; it labels its outputs `source = external device log (supplied input; not measured by this script)`. Energy consumption and battery discharge were not measured in this study (Discussion), and no script in this repository analyzes them.

## Architecture configuration

`pvc/utils/model_arch/himae.py` defines `PAPER_CONFIG`, the configuration reported in the Methods, and is the only HiMAE definition: `pretrain/himae.py` subclasses it, and the notebook and downstream scripts instantiate it with these defaults.

| Main text (Methods) | `PAPER_CONFIG` | `himae_synth.ckpt` |
|---|---|---|
| 10-s windows at 100 Hz (1,000 samples) | `sampling_freq=100`, `seg_len=10` | `f_s=100`, `T=10` |
| Channel hierarchy [16, 32, 64, 128, 256] | `channels=(16, 32, 64, 128, 256)` | same |
| Kernel size 5, stride 2 | `kernel_size=5`, `stride=2` | same (weights shape-checked) |
| Patch size 5, masking ratio 0.8 | `patch_len=5`, `mask_ratio=0.8` | `patch=5`, `mask_ratio=0.8` |
| 1.2M parameters | 1,245,953 | loads with `strict=True` |

Downstream data stored at other rates (the synthetic PVC file is 25 Hz) are resampled to 100 Hz and min-max normalized per window by `pvc/utils/signal_utils.py` before every forward pass.

Pretraining defaults follow the Methods: mask ratio 0.8, loss on masked samples only, contiguous block masking with probability `block_mask_prob` (default 0.5; `num_mask_blocks=4`) and otherwise uniformly random patch masking; AdamW with linear warmup (`warmup_ratio=0.1`) and cosine decay; early stopping on validation reconstruction loss.

## Multi-resolution probes (repository analysis)

`HiMAE.encode()` returns the feature map of each of the five encoder stages; stage *k* has temporal step 2^k samples and receptive field 13, 37, 85, 181, 373 samples (130 ms – 3.73 s at 100 Hz). Each stage is mean-pooled over time (`--pooling mean|max|meanmax`) and probed with standardized L2 logistic regression (C chosen on validation AUROC); the concatenation of all stages is probed as well; the best stage is selected on validation only. `multires_probe_paired.csv` reports Δ = AUROC_stage − AUROC_stage5 with the same method-named columns as `bootstrap_auc.py`.

## Plot input schemas

| Script | Required columns | Notes |
|---|---|---|
| `plot/fig1.py` | `task,model,y_true,y_score` | one ROC panel per task |
| `plot/fig2_3.py --efficiency-csv` | `model` + any of `memory,cpu_latency,cpu_throughput` | one row per model; `--baseline-model` adds ratio plots |
| `plot/fig2_3.py --clinical-csv` | `task,model` + any of `tp,sensitivity,balanced_accuracy,youden_index` | one row per task/model |
| `plot/fig4.py` | `dataset,model,shots,auc` (optional `run`) | mean with interquartile range across runs |
| `plot/e-fig1_3.py` | `threshold,task,model,regime,sensitivity,specificity,tp,fp,tn,fn` | reads every CSV in `--data-dir` |
| `plot/fig_multiresolution.py` | `task,level,stage,receptive_field_ms,test_auroc,ci_low,ci_high` | output of `multiresolution_probe.py` |

## Determinism

Plotting and threshold/operating-point scripts are deterministic. Bootstrap scripts take `--seed` (default 2024); with identical inputs, seed, package versions and arguments they produce identical outputs. On CPU, `binary_linear_prob.py` and the notebook produce byte-identical prediction files for the same seeds. Rendering may differ slightly across Matplotlib versions and fonts.
