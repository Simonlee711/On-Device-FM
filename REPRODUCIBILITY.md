# Reproducibility

This file maps each main-text table and figure to its generating script, required input schema, random seed, and expected output artifact.

## Quick start: full statistical workflow

```bash
bash analysis/run_statistical_workflow.sh results/predictions.csv results/device_runs.csv results/himae_static.csv results
```

Input schemas are given below. All stochastic steps use seed `2024` and `B = 1,000` participant-level bootstrap replicates.

## Map of tables and figures

| Manuscript item | Script(s) | Input | Seed | Output artifact |
|---|---|---|---|---|
| Table 1, AUROC rows (+ paired ΔAUROC, Results §2.1) | `analysis/bootstrap_auc.py` | predictions schema (below) | 2024 | `results/auroc_summary.csv`, `results/auroc_paired_differences.csv` |
| Table 1, on-device rows; Fig. 2 | `analysis/summarize_device_energy.py` → `plot/fig2_3.py --efficiency-csv` | device-runs schema (below) + static constants | none | `results/device_summary.csv`, `results/device_welch_tests.csv`, `results/device_discharge.csv`, `results/fig2_efficiency.csv`, `figures/*_absolute.png`, `figures/*_relative.png` |
| Fig. 1 (ROC panels) | `analysis/prepare_fig1_results.py` → `plot/fig1.py` | `pvc_predictions.csv` | none | `figures/<task>_roc.{png,pdf}`, `figures/roc_summary.csv` |
| Fig. 3 (true positives at validation-selected specificity) | `analysis/select_validation_thresholds.py` → `analysis/apply_operating_points.py` → `analysis/prepare_fig3_results.py` → `plot/fig2_3.py --clinical-csv` | predictions schema | none | `results/validation_thresholds.csv`, `results/operating_points.csv`, `results/fig3_clinical.csv`, `figures/clinical_*.png` |
| Fig. 3 CIs, paired ΔTP (Results §2.3); Extended Data Tables 3–6 | `analysis/bootstrap_operating_points.py` | predictions schema + `validation_thresholds.csv` | 2024 | `results/operating_points_bootstrap.csv`, `results/operating_points_paired_tp.csv` |
| Fig. 4 (few-shot curves) | `pvc/downstream_eval/fewshot_finetune.py` → `plot/fig4.py` | HDF5 windows + HiMAE checkpoint | `--split-seed 77`, `--run-seed 1000` (+run index) | `results/finetuning.csv`, `figures/fewshot_<dataset>.{png,pdf}` |
| Extended Data Figs. 1–3 (confusion matrices) | `plot/e-fig1_3.py` | `results/operating_points.csv`-style counts | none | `figures/extended_figure1_3.{png,pdf}`, `figures/extended_figure1_3_summary.csv` |
| Synthetic PVC linear probe + prediction export | `HiMAE_PVC_Detection.ipynb`, `pvc/downstream_eval/binary_linear_prob.py` | `pvc_10s_synth.h5`, `himae_synth.ckpt` | see script | `pvc_predictions.csv`, `pvc_linear_probe.pt` |
| HiMAE pretraining | `pretrain/himae.py` | unlabeled 10-s PPG windows | `set_seed(42)` | Lightning checkpoint |
| Table 1, parameters / memory / FLOPs (HiMAE) | `analysis/model_complexity.py` | model definition only (needs PyTorch) | none | `results/model_complexity.csv`, optional `--static-output` for `summarize_device_energy.py --static` |

## Prediction schema (statistical workflow)

```text
task,model,regime,split,person_id,y_true,y_score[,sample_id]
```

- `regime`: e.g. `Frozen`, `Fine-tuned`.
- `split`: `val` (used **only** for threshold selection) or `test` (held out).
- `person_id`: participant identifier; bootstrap resamples participants with all their windows kept together.
- `sample_id`: window identifier within participant. Paired analyses require every model in a task/regime to be scored on the same test windows.

Build a combined table from per-model exports with `analysis/merge_predictions.py`.

## HiMAE pretraining

**Script:** `pretrain/himae.py`

Defaults follow the Methods: 100 Hz × 10 s windows (1,000 samples), channel hierarchy `[16, 32, 64, 128, 256]`, kernel size 5, stride 2, patch size 5, GELU + BatchNorm, mask ratio 0.8, loss on masked samples only. Each window is masked with contiguous block masking with probability `block_mask_prob` (default 0.5; `num_mask_blocks=4` runs of patches), otherwise with uniformly random patch masking. Optimizer is AdamW with linear warmup (`warmup_ratio=0.1`) followed by cosine decay; early stopping on validation reconstruction loss.

## Table 1 — parameters, memory, FLOPs

**Script:** `analysis/model_complexity.py` (requires PyTorch)

```bash
python analysis/model_complexity.py --output results/model_complexity.csv \
  --static-output results/himae_static.csv
```

Parameters and model-weight memory (fp32; reported in MB = 1e6 bytes and MiB = 2^20 bytes) are counted from the model definition. FLOPs are multiply-accumulates of convolution and linear layers for one 1,000-sample window, one MAC = one FLOP (the fvcore convention; `--flops-per-mac 2` doubles). Two scopes are reported: `autoencoder` (full encoder–decoder, 1.25M parameters) and `deployed` (encoder + linear head used downstream). Baseline model complexities are supplied as static constants because their implementations are not part of this release.

## Table 1 (AUROC) — paired participant-level bootstrap

**Script:** `analysis/bootstrap_auc.py`

```bash
python analysis/bootstrap_auc.py --predictions results/predictions.csv \
  --reference-model "HiMAE (1.2M)" --n-boot 1000 --seed 2024 --output-dir results
```

**Method:** Point estimates on the original test set. Participants resampled with replacement (1,000 replicates); 95% CI from the 2.5th/97.5th percentiles. Within each replicate ΔAUROC = AUROC_reference − AUROC_comparator. Two-sided P = 2·min{Pr(Δ ≤ 0), Pr(Δ ≥ 0)} using (k+1)/(B+1). Replicates with a single class are skipped (`n_valid_boot`).

**Outputs:** `auroc_summary.csv` (`task,regime,model,auroc,ci_low,ci_high,n_windows,n_positive,n_participants,n_boot,n_valid_boot,seed`); `auroc_paired_differences.csv` (`task,regime,reference,comparator,delta_auroc,ci_low,ci_high,p_value,...`).

## Fig. 3 / Extended Data Tables 3–6 — validation-selected operating points

**Scripts:** `analysis/select_validation_thresholds.py`, `analysis/apply_operating_points.py`, `analysis/bootstrap_operating_points.py`

```bash
python analysis/select_validation_thresholds.py --predictions results/predictions.csv \
  --targets 0.95 0.90 0.80 --output results/validation_thresholds.csv
python analysis/apply_operating_points.py --predictions results/predictions.csv \
  --thresholds results/validation_thresholds.csv --output results/operating_points.csv
python analysis/bootstrap_operating_points.py --predictions results/predictions.csv \
  --thresholds results/validation_thresholds.csv --n-boot 1000 --seed 2024 --output-dir results
```

**Method:** For each task/regime/model, the threshold minimizing |validation specificity − target| is chosen on the validation split only (ties → higher validation sensitivity, then larger threshold). It is frozen and applied once to the test split; achieved test specificity may differ from the target. Bootstrap replicates reuse the same frozen threshold. Within each replicate, TP/FP/TN/FN, sensitivity, specificity, PPV, NPV, balanced accuracy, Youden index and ΔTP = TP_reference − TP_comparator are recomputed.

**Outputs:** `validation_thresholds.csv`; `operating_points.csv` (point estimates; input to `prepare_fig3_results.py` and `plot/e-fig1_3.py`); `operating_points_bootstrap.csv` (each metric with `<metric>_ci_low/_ci_high`); `operating_points_paired_tp.csv` (`delta_tp,ci_low,ci_high,p_value`).

## Table 1 (on-device) / Fig. 2 — latency, throughput, energy

**Script:** `analysis/summarize_device_energy.py`

**Required input schema (one row per benchmark pass):**

```text
model,run,latency_s,throughput_wps,energy_j[,battery_start_pct,battery_end_pct,duration_s,n_windows]
```

Optional static constants (`--static`): `model,params_m,memory_mb,flops_gflops`.

```bash
python analysis/summarize_device_energy.py --runs results/device_runs.csv \
  --static results/himae_static.csv --reference-model "HiMAE (1.2M)" --output-dir results
python plot/fig2_3.py --efficiency-csv results/fig2_efficiency.csv --output-dir figures --baseline-model "HiMAE (1.2M)"
```

**Method:** Mean, median, 95th percentile and a 95% t-interval of the mean across repeated passes. Between-model mean differences vs. the reference use two-sided Welch t-tests. Battery discharge is summarized as %/hour and windows per 1% battery when discharge columns are present.

**Random seed:** None (deterministic summaries of measured runs).

**Outputs:** `device_summary.csv`, `device_welch_tests.csv`, `device_discharge.csv`, `fig2_efficiency.csv` (`model,flops,memory,cpu_latency,cpu_throughput,energy,params`).

---

# Plotting scripts

## Environment

The plotting scripts are written for Python 3 and use standard scientific Python packages including `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` where applicable.

Install the project dependencies before running the figure scripts. All scripts are designed to read results from CSV files rather than embedding experimental result values in the plotting code.

## Figure 1 — ROC curves

**Script:** `plot/fig1.py`

**Purpose:** Computes and plots ROC curves and AUROC values for each task/model pair from model prediction scores.

**Required input schema:**

```text
task,model,y_true,y_score
```

- `task`: task or dataset name.
- `model`: model display name.
- `y_true`: binary ground-truth label (`0` or `1`).
- `y_score`: continuous model score or probability for the positive class.

Optional columns may include `sample_id` and `person_id`.

**Example command:**

```bash
python plot/fig1.py --input predictions.csv --output-dir figures
```

**Random seed:** None. ROC coordinates and AUROC are deterministically computed from the supplied labels and prediction scores.

**Expected outputs:**

```text
figures/<task>_roc.png
figures/<task>_roc.pdf
figures/roc_summary.csv
```

The summary CSV contains `task`, `model`, `auroc`, and `n_samples`.

---

## Figures 2–3 — Efficiency and clinical comparisons

**Script:** `plot/fig2_3.py`

**Purpose:** Generates model-efficiency plots and clinical-comparison plots from tabular result files.

### Efficiency input

**Required columns:**

```text
model
```

At least one of the following metric columns must also be present:

```text
flops,memory,cpu_latency,cpu_throughput
```

Each model should appear once in the efficiency CSV.

**Example command:**

```bash
python plot/fig2_3.py --efficiency-csv efficiency_results.csv --output-dir figures --baseline-model "HiMAE (1.2M)"
```

When `--baseline-model` is provided, relative values are computed by dividing each model's absolute measurement by the selected baseline model's measurement.

### Clinical-comparison input

**Required columns:**

```text
task,model
```

At least one of the following metric columns must also be present:

```text
tp,sensitivity,balanced_accuracy,youden_index
```

Each task/model pair should appear once in the clinical CSV.

**Example command:**

```bash
python plot/fig2_3.py --clinical-csv clinical_results.csv --output-dir figures
```

Both inputs can be provided in the same invocation.

**Random seed:** None. The script performs deterministic transformations and plotting of the supplied measurements.

**Expected outputs:**

For available efficiency metrics:

```text
figures/flops_absolute.png
figures/memory_absolute.png
figures/cpu_latency_absolute.png
figures/cpu_throughput_absolute.png
```

If `--baseline-model` is specified, corresponding relative plots are also produced:

```text
figures/flops_relative.png
figures/memory_relative.png
figures/cpu_latency_relative.png
figures/cpu_throughput_relative.png
```

For available clinical metrics:

```text
figures/clinical_tp.png
figures/clinical_sensitivity.png
figures/clinical_balanced_accuracy.png
figures/clinical_youden_index.png
```

Only plots corresponding to columns present in the input CSVs are generated.

---

## Figure 4 — Few-shot learning curves

**Script:** `plot/fig4.py`

**Purpose:** Plots few-shot AUROC curves for each dataset from a long-form results table.

**Required input schema:**

```text
dataset,model,shots,auc
```

- `dataset`: dataset or task name.
- `model`: model display name.
- `shots`: few-shot setting; may be numeric or a string such as `all`.
- `auc`: AUROC/AUC value in `[0, 1]`.

An optional `run` column may be included for repeated runs:

```text
dataset,model,shots,run,auc
```

When repeated runs are present, the script reports the arithmetic mean for each dataset/model/shot setting and draws the interquartile range (25th–75th percentile) across runs as error bars, matching the Methods. Error bars can be disabled with `--no-error-bars`.

**Example command:**

```bash
python plot/fig4.py --input fewshot_results.csv --output-dir figures
```

**Random seed:** None. No stochastic operations are used by the plotting pipeline.

**Expected outputs:**

```text
figures/fewshot_<dataset>.png
figures/fewshot_<dataset>.pdf
```

Dataset names are normalized for use in filenames.

---

## Extended Figures 1–3 — Confusion matrices

**Script:** `plot/e-fig1_3.py`

**Purpose:** Loads confusion-matrix operating-point results, selects a common model-score threshold, and renders a grid of row-normalized confusion matrices.

**Required input schema:**

```text
threshold,task,model,regime,sensitivity,specificity,tp,fp,tn,fn
```

The script reads all CSV files in the directory supplied with `--data-dir`.

**Example command:**

```bash
python plot/e-fig1_3.py --data-dir data --threshold 0.05 --output-dir figures
```

The requested threshold and threshold-matching tolerance can be configured with `--threshold` and `--threshold-tolerance`. Missing configured panels can be allowed with `--allow-missing`.

**Random seed:** None. Confusion-matrix values and derived metrics are computed deterministically from the supplied counts.

**Expected outputs:**

```text
figures/extended_figure1_3.png
figures/extended_figure1_3.pdf
figures/extended_figure1_3_summary.csv
```

The summary CSV records the selected threshold, count-derived FPR, sensitivity, specificity, balanced accuracy, Youden index, confusion-matrix counts, class totals, and source filename for each plotted panel.

---

## Determinism

The plotting scripts do not use stochastic data generation. The bootstrap scripts take an explicit `--seed` (default `2024`); with identical inputs, seed, package versions and arguments they produce identical outputs. Minor rendering differences can occur across Matplotlib versions, operating systems, or font configurations.
