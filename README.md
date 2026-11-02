# HiMAE: Hierarchical Masked Autoencoders

### On-device foundation models enable real-time population-scale detection of clinical events on wearable devices

**Authors:** [Simon A. Lee](https://simon-a-lee.github.io), Hao Zhou, Minji Han, Rachel Choi, Cyrus Tanade, Md Sazzad Hissain Khan, Juhyeon Lee, Li Zhu, Md Mahbubur Rahman, Viswam Nathan, Mehrab Bin Morshed, Migyeong Gwak, Keum San Chun, Jeffrey N. Chiang, Sharanya Arcot Desai

---

# Brief Description

Self-supervised masked autoencoding for physiological waveforms with HiMAE, applied to premature ventricular contraction (PVC) detection. This repository contains a PyTorch implementation of the hierarchical 1-D convolutional masked autoencoder ("HiMAE") in the configuration reported in the manuscript, its pretraining logic, a participant-disjoint linear-probe and fine-tuning pipeline on 10-second PPG windows, and the statistical scripts used for the manuscript's AUROC and operating-point analyses.

# What is and is not provided

**Provided:** the model definition, pretraining code, downstream evaluation code, statistical and plotting scripts, and a **synthetic** PVC example (synthetic windows, a HiMAE checkpoint and canonical synthetic predictions) on which every script can be executed.

**Not provided:** the clinical and wearable datasets, the trained clinical model checkpoints, the baseline implementations (ResNet, PaPaGei, EfficientNet, MAE1D), the Wear OS benchmarking application, the deployed on-device model artifacts and the original Wear OS device telemetry. These are proprietary and subject to privacy, regulatory and commercial restrictions (see the manuscript's Data and Code Availability statements).

## Synthetic versus clinical outputs

Everything you can run here produces **synthetic** results. They show that the code executes correctly; they are not, and are not expected to be close to, the clinical numbers in the manuscript.

| Output | Produced from | Manuscript counterpart |
|---|---|---|
| `results/pvc_predictions.csv` (notebook or `binary_linear_prob.py`) | synthetic data + synthetic checkpoint | none (demo) |
| `examples/synthetic_predictions.csv` | same, two frozen probes | none (demo input to the statistical workflow) |
| `analysis/` outputs on `examples/synthetic_predictions.csv` | synthetic predictions | same scripts and columns as Table 1, Results 2.1, Fig. 3, Extended Data Tables 3–8, but synthetic values |
| `analysis/` outputs on the clinical prediction table | clinical predictions (not provided) | Table 1, Results 2.1, Fig. 3, Extended Data Tables 3–8 |
| `analysis/summarize_device_benchmarks.py` outputs | an externally collected device log (not provided) | Table 1 on-device rows, Fig. 2; the script summarizes the log and measures nothing |
| `legacy/` | an earlier demo notebook | none; superseded (see `legacy/README.md`) |

`REPRODUCIBILITY.md` maps each manuscript figure and result to its script, inputs, schema, command, seed and output, and states for each whether it can be regenerated from this repository.

---

# What's in the repo

```text
HiMAE_PVC_Detection.ipynb        <- end-to-end synthetic PVC demo: paper config, bottleneck probe, multi-resolution probes
himae_synth.ckpt                 <- HiMAE pretraining-autoencoder checkpoint (100 Hz, 10 s, patch 5, channels [16..256])
pvc_10s_synth.h5                 <- synthetic PVC windows (stored at 25 Hz; resampled to 100 Hz on load)
pvc_10s_synth_metadata.csv       <- metadata for the synthetic windows
requirements.txt                 <- dependencies (with tested versions)

examples/
    synthetic_predictions.csv        <- canonical synthetic predictions (input to the statistical workflow)
    make_synthetic_predictions.sh    <- rebuilds it from the synthetic data and checkpoint

legacy/                              <- superseded synthetic outputs of an earlier notebook (not used by any script)

pretrain/
    himae.py                         <- Lightning masked-autoencoder pretraining (needs your own unlabeled PPG shards)

pvc/
    utils/
        model_arch/himae.py          <- HiMAE + PAPER_CONFIG (single source of truth for the architecture)
        signal_utils.py              <- load, resample to 100 Hz, normalize, participant-disjoint split
        helper_logger.py, helper_models.py
    downstream_eval/
        binary_linear_prob.py        <- frozen linear probe; script form of the notebook; canonical export
        fewshot_finetune.py          <- frozen + few-shot / full fine-tuning (Fig. 4); optional canonical export
        multiresolution_probe.py     <- per-stage linear probes with bootstrap CIs
        helpers.py                   <- dataset summary utility

analysis/
    prediction_schema.py             <- canonical prediction schema + validation (used by every script below)
    convert_predictions.py           <- legacy export -> canonical schema
    merge_predictions.py             <- combine per-model canonical files
    bootstrap_auc.py, delong.py      <- AUROC, participant-bootstrap CIs, paired DeLong P (Table 1, Results 2.1)
    select_validation_thresholds.py  <- validation-only operating-point selection
    apply_operating_points.py        <- frozen thresholds applied once to the test split
    bootstrap_operating_points.py    <- operating-point CIs and paired ΔTP (Fig. 3, Extended Data Tables 3-8)
    prepare_fig{1,2,3}_results.py    <- adapters to the plotting inputs
    model_complexity.py              <- analytic parameters / memory of the HiMAE model definitions
    summarize_device_benchmarks.py   <- summaries + Welch tests of a supplied latency/throughput log
    run_statistical_workflow.sh      <- runs the prediction-based analyses (and, optionally, the device-log summary)

plot/                                <- Fig. 1 (ROC), Fig. 2/3 bar charts, Fig. 4, confusion-matrix grid, multi-resolution plot
tests/                               <- architecture, schema/alignment, statistics and device/complexity tests
REPRODUCIBILITY.md                   <- manuscript item -> script, inputs, schema, command, seed, output
```

---

# System Requirements

## Supported environment

- **Python 3.10 or newer.** The full test suite and the demo were last run with Python 3.14.6 on macOS (Apple silicon, CPU only), with the package versions listed in `requirements.txt`.
- **PyTorch 2.x.** CPU is sufficient for the demo, the probes and all analysis scripts. A CUDA build is useful only for pretraining or large-scale fine-tuning.
- Linux and macOS are expected to work; Windows has not been tested.

## Hardware

No non-standard hardware is required. The synthetic demo runs on a laptop CPU with a few GB of RAM. The manuscript's pretraining used four NVIDIA Tesla T4 GPUs (about 12 hours); this is not needed for anything in this repository except reproducing pretraining at scale.

---

# Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For a specific CUDA build, install PyTorch first with the command from <https://pytorch.org>, then run `pip install -r requirements.txt`. Installation takes about 5–15 minutes, mostly for the PyTorch wheel.

`pretrain/himae.py` logs to Weights & Biases; to run it offline:

```bash
export WANDB_DISABLED=true
```

## How to invoke the code

This is a collection of scripts, not an installable package. Run every command **from the repository root** and call scripts by path (`python analysis/bootstrap_auc.py ...`, `python pvc/downstream_eval/fewshot_finetune.py ...`), not with `python -m`. Each script puts the directories it needs on `sys.path` itself. Start Jupyter from the repository root as well, because the notebook uses relative paths.

Check the installation:

```bash
python -m pytest tests -q
```

---

# Demo: synthetic PVC linear probe

The demo freezes the pretrained encoder, fits one logistic layer on the mean-pooled stage-5 (bottleneck) embedding, and repeats the probe on every encoder stage. Windows are resampled to the 100 Hz pretraining rate, participants are split into disjoint train / validation / test sets (`SPLIT_SEED = 77`), and test AUROC is reported on held-out participants.

Notebook:

1. Start Jupyter from the repository root and open `HiMAE_PVC_Detection.ipynb`.
2. The defaults point to `pvc_10s_synth.h5` and `himae_synth.ckpt`; for other data set `H5_PATH`, `SIGNAL_KEY` (`ppg` or `ecg`) and, if needed, `INPUT_FS`. `CFG` is built from `PAPER_CONFIG` and should not need editing.
3. Run all cells. The probe and canonical predictions are written to `results/`.

The same steps as scripts:

```bash
# bottleneck linear probe; writes canonical predictions (identical to the notebook's)
python pvc/downstream_eval/binary_linear_prob.py --h5 pvc_10s_synth.h5 --checkpoint himae_synth.ckpt \
  --predictions-output results/pvc_predictions.csv --probe-output results/pvc_linear_probe.pt

# per-stage (multi-resolution) linear probes with participant-level bootstrap CIs
python pvc/downstream_eval/multiresolution_probe.py --h5 pvc_10s_synth.h5 --checkpoint himae_synth.ckpt \
  --task PVC --output-dir results/multiresolution
python plot/fig_multiresolution.py --input results/multiresolution/multires_probe_summary.csv --output-dir figures

# frozen + few-shot fine-tuning (slow on CPU); --embedding-stage last | 1..5 | all
python pvc/downstream_eval/fewshot_finetune.py --h5 pvc_10s_synth.h5 --checkpoint himae_synth.ckpt \
  --shots 256 512 1024 2048 all --runs 5 --include-frozen --output results/finetuning.csv
python plot/fig4.py --input results/finetuning.csv --output-dir figures

# statistical workflow on the shipped canonical synthetic predictions
bash analysis/run_statistical_workflow.sh --predictions examples/synthetic_predictions.csv \
  --out results/synthetic --fig3-regime Frozen
```

## Expected demo output

On the synthetic example (55,861 windows, 18 participants; test split 11,209 windows from 4 participants):

```text
Bottleneck (stage 5) linear probe   test ROC-AUC: 0.909   (validation 0.903)

Per-stage probes (test ROC-AUC; receptive field):
  stage 1  0.749  (130 ms)     stage 4  0.830  (1.81 s)
  stage 2  0.848  (370 ms)     stage 5  0.907  (3.73 s)
  stage 3  0.832  (850 ms)     all stages concatenated  0.903
```

`results/pvc_predictions.csv` has the canonical schema used by every statistical script:

```text
task,model,regime,split,person_id,sample_id,y_true,y_score
PVC,HiMAE (1.2M),Frozen,val,2,0,0,0.675...
```

The bottleneck probe takes about two minutes on a laptop CPU; small numerical differences can occur across hardware and library versions. These are synthetic numbers and say nothing about clinical performance.

---

# Data format

## Pretraining data

Pretraining expects a CSV that indexes samples stored in HDF5 shards. Each row references a shard path and a sample key containing a `normalized_waveform` dataset:

```text
local_path,global_idx
/path/to/shard_A.h5,000123
/path/to/shard_B.h5,000987
```

`h5py.File(local_path)[global_idx]["normalized_waveform"][:]` must return a one-dimensional float array of length `f_s × T` (1,000 for 100 Hz × 10 s). Set `CONSOLIDATED_METADATA_PATH` in `pretrain/himae.py` to this CSV.

## Downstream data

Downstream scripts read one HDF5 file with:

- `/ppg` (or `/ecg`) with shape `[N, L]`;
- `/labels` with shape `[N]`, binary;
- `/patient_ids` with shape `[N]` (needed for participant-disjoint splitting and the participant bootstrap).

Windows are resampled to 100 Hz and min-max normalized to [-1, 1] before every forward pass (`pvc/utils/signal_utils.py`). The stored rate is inferred from the window length or set with `--input-fs` / `INPUT_FS`.

## Prediction tables

All statistical scripts read the canonical schema above (defined in `analysis/prediction_schema.py`; column definitions in `REPRODUCIBILITY.md`). Exporters in this repository write it directly. Older exports with `patient_id,label,p_pvc` are converted with `analysis/convert_predictions.py`, which requires missing split and window identifiers to be declared explicitly.

---

# Running HiMAE on your own data

1. Segment recordings into 10-second windows and store them in the downstream HDF5 format above, with participant IDs.
2. Point `--h5` (or `H5_PATH` in the notebook) at the file and `--checkpoint` at a HiMAE checkpoint.
3. Run `binary_linear_prob.py` (frozen probe) or `fewshot_finetune.py --predictions-output ...` (frozen and fine-tuned) to obtain canonical predictions.
4. Combine models with `analysis/merge_predictions.py` and run `analysis/run_statistical_workflow.sh`.

The default bottleneck dimensionality is 256. If you change the encoder channels, the linear heads adapt automatically in the scripts above. For much longer sequences, additional encoder depth may be needed to keep a useful bottleneck temporal resolution after repeated stride-2 downsampling.

---

# Troubleshooting

- **CUDA is not available:** install a CPU PyTorch build; nothing in the demo requires a GPU.
- **W&B asks for authentication:** `export WANDB_DISABLED=true`.
- **`ModuleNotFoundError` for `prediction_schema`, `pvc` or `_bootstrap`:** call the script by its path from the repository root, not with `python -m`.
- **"looks like a legacy export":** convert the file with `analysis/convert_predictions.py`.
- **"Models disagree on y_true", "not scored on the same windows" or "Duplicated ... rows":** paired analyses need every model of a task/regime to be scored on the same `(person_id, sample_id)` windows with the same labels; fix the export rather than the analysis.
- **HDF5 keys do not match:** pass `--signal-key`, `--label-key`, `--patient-key`.

---

# Acknowledgements

We thank Minji Han and Rachel Choi for their expertise in UX/UI design and for crafting the specialized visualizations not supported by standard Python libraries; their design contributions were essential to this work.

We also thank Praveen Raja, Matthew Wiggins, and Mike Freedman for their invaluable feedback and insightful discussions throughout the project.
