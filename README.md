# <center> 🔎 HiMAE: Hierarchical Masked Auto Encoders 🔍 </center>

### On-device foundation models enable real-time population-scale detection of clinical events on wearable devices

**Authors:** [Simon A. Lee](https://simon-a-lee.github.io), Hao Zhou, Minji Han, Rachel Choi, Cyrus Tanade, Md Sazzad Hissain Khan, Juhyeon Lee, Li Zhu, Md Mahbubur Rahman, Viswam Nathan, Mehrab Bin Morshed, Migyeong Gwak, Keum San Chun, Jeffrey N. Chiang, Sharanya Arcot Desai

---

# Brief Description

Self-supervised masked autoencoding for physiological waveforms with HiMAE for PVC detection. This repository contains a PyTorch/Lightning implementation of a hierarchical 1-D convolutional masked autoencoder ("HiMAE"), a minimal pretraining script, and a reproducible linear-probe pipeline on 10-second PPG segments.

The repository is intended as a research implementation of the core model and evaluation workflow described in the manuscript. The production codebase and participant-level datasets used in the study contain proprietary components and data that cannot be publicly released. A synthetic PVC example is therefore provided so that the architecture, checkpoint loading, feature extraction, and downstream linear-probe workflow can be inspected and executed without access to proprietary data.

---

# What's in the repo

The root directory includes a pretraining checkpoint, a reference linear probe, a small metadata CSV for the synthetic PVC task, example predictions, and a demonstration notebook.

```text
HiMAE_PVC_Detection.ipynb        <- end-to-end wiring for PVC linear probe
himae_synth.ckpt                 <- Lightning checkpoint for HiMAE backbone
pvc_linear_probe.pt              <- state_dict for reference linear probe
pvc_10s_synth_metadata.csv       <- example metadata (fs=25 Hz, 10 s windows)
pvc_predictions.csv              <- example inference outputs (p_pvc per segment)

pretrain/
    himae.py                     <- minimal Lightning trainer for masked AE

analysis/
    bootstrap_auc.py                 <- paired participant-level AUROC bootstrap (Table 1)
    select_validation_thresholds.py  <- validation-only operating-point selection
    apply_operating_points.py        <- frozen thresholds applied to the test split
    bootstrap_operating_points.py    <- operating-point CIs and paired TP differences (Fig. 3)
    summarize_device_energy.py       <- latency/throughput/energy/discharge + Welch tests (Table 1, Fig. 2)
    model_complexity.py              <- analytic params / memory / FLOPs for HiMAE (Table 1)
    prepare_fig{1,2,3}_results.py    <- adapters to the plotting inputs
    merge_predictions.py             <- concatenate per-model prediction exports
    run_statistical_workflow.sh      <- end-to-end statistical workflow

plot/                                <- Fig. 1-4 and Extended Data Fig. 1-3 plotting

pvc/
    utils/                       <- logger and model registry
    helper_logger.py
    helper_models.py
    model_arch/
        himae.py                 <- 1-D CNN HiMAE backbone (encoder/decoder)
    downstream_eval/
        binary_linear_prob.py    <- script for linear probe training/eval
        fewshot_finetune.py      <- fewshot and fintune training/eval
        helpers.py               <- analysis utilities

LICENSE
README.md
REPRODUCIBILITY.md               <- table/figure -> script, schema, seed, output
```

---

# System Requirements

## Software

The research implementation is designed for Python 3.10 or newer and PyTorch 2.x.

Recommended dependencies are:

- Python >= 3.10
- PyTorch 2.x
- torchvision 0.x
- torchaudio 2.x
- Lightning / PyTorch Lightning
- torchmetrics
- NumPy
- pandas
- scikit-learn
- h5py
- matplotlib
- PyYAML
- tabulate
- boto3
- s3fs
- Weights & Biases (`wandb`; optional for local/demo execution)

A CUDA-enabled PyTorch installation can be used when an NVIDIA GPU is available. CPU-only execution is sufficient for the included demonstration and linear-probe inference.

## Operating systems

The code uses standard Python/PyTorch tooling and is intended to run on modern Linux and macOS systems. Windows should also work with an appropriate PyTorch installation, although it has not been specifically validated for the research release.

For submission purposes, users should install a PyTorch build compatible with their operating system, Python version, and CUDA version (if applicable).

## Tested software versions

The repository was developed for the following software family versions:

- Python 3.10+
- PyTorch 2.x
- PyTorch Lightning / Lightning 2.x
- scikit-learn 1.x
- NumPy 1.x or 2.x
- pandas 2.x

Because dependency patch versions may differ across systems, the installation command below specifies compatible major versions rather than a single platform-specific lockfile. If exact environment replication is required, users are encouraged to create and pin a local `requirements.txt` or Conda environment from a successful installation.

## Hardware requirements

No non-standard hardware is required for the included synthetic-data demonstration.

- **CPU:** sufficient for loading the model, extracting embeddings, running the provided linear probe, and reproducing the example inference workflow.
- **GPU:** optional for the demo; recommended for pretraining or large-scale fine-tuning.
- **Memory:** a typical modern desktop or laptop with several GB of available RAM is sufficient for the included synthetic example.
- **Research-scale pretraining:** the manuscript reports pretraining distributed across four NVIDIA Tesla T4 GPUs; this hardware is not required for running the included demo.

---

# Installation

Create a Python environment and install the required dependencies. Use a CUDA-enabled PyTorch wheel if you have a compatible NVIDIA GPU; otherwise install the standard CPU build from PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

pip install "torch==2.*" "torchvision==0.*" "torchaudio==2.*" \
    --index-url https://download.pytorch.org/whl/cu121

pip install lightning pytorch-lightning torchmetrics \
    h5py s3fs boto3 pandas numpy tabulate matplotlib \
    scikit-learn pyyaml wandb
```

For a CPU-only installation, install PyTorch using the appropriate command from the official PyTorch installation instructions instead of the CUDA-specific command above.

W&B logging is enabled by default in pretraining. To disable online logging:

```bash
export WANDB_DISABLED=true
```

## Typical installation time

On a normal desktop or laptop with a broadband connection, installation typically takes approximately **5-15 minutes**, depending primarily on the PyTorch wheel size, network speed, and whether CUDA libraries must also be downloaded.

---

# Data Format

## Pretraining data

Pretraining expects a CSV that indexes samples stored in HDF5 shards. Each row references a shard path and a sample key containing a `normalized_waveform` dataset:

```text
local_path,global_idx
/path/to/shard_A.h5,000123
/path/to/shard_B.h5,000987
...
```

Each call of the form

```python
h5py.File(local_path)[global_idx]["normalized_waveform"][:]
```

should yield a one-dimensional floating-point array of length

\[
L = f_s \times T
\]

where \(f_s\) is the sampling frequency and \(T\) is the segment duration.

## Downstream PVC data

Downstream PVC evaluation expects an HDF5 file with contiguous datasets for signals and labels, for example:

- `/ppg` with shape `[N, L]`, or `/ecg` with shape `[N, L]`
- `/labels` with shape `[N]` containing binary labels
- optionally `/patient_ids` with shape `[N]`

The included `pvc_10s_synth_metadata.csv` describes synthetic segments sampled at 25 Hz with 10-second windows (`L = 250`) and a binary `pvc` label.

The demonstration notebook shows how to use either an HDF5 file in this format or synthetic tensors generated within the notebook.

---

# Demo: PVC Linear Probe

The PVC demo freezes the pretrained encoder and fits or loads a single logistic layer on top of mean-pooled bottleneck features.

The simplest route is the Jupyter notebook:

```text
HiMAE_PVC_Detection.ipynb
```

## Instructions to run the demo

1. Create and activate the Python environment described above.
2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `HiMAE_PVC_Detection.ipynb`.
4. Set the following notebook variables as needed:
   - `H5_PATH`
   - `META_PATH`
   - `SIGNAL_KEY` (`ppg` or `ecg`)
   - configuration values in the `CFG` block
5. For the included synthetic example, use the provided 25 Hz sampling frequency and 10-second segment duration.
6. Point the backbone to:

   ```text
   himae_synth.ckpt
   ```

7. Point the reference probe to:

   ```text
   pvc_linear_probe.pt
   ```

8. Run all cells in order.

The notebook can either load the provided reference probe or fit a new linear probe on extracted HiMAE embeddings.

If a pure-script workflow is preferred, the corresponding evaluation logic is implemented in:

```text
pvc/downstream_eval/binary_linear_prob.py
```

The script contains S3 helper utilities for the original research environment. For local execution, use `_read_one_h5_from_local` with a local HDF5 path and construct the `cfg` dictionary using the same settings shown in the notebook.

---

# Expected Demo Output

A successful run should:

- load the HiMAE backbone checkpoint;
- load or train the PVC linear probe;
- extract frozen HiMAE embeddings from the example segments;
- produce one PVC probability per segment;
- calculate aggregate binary-classification performance; and
- optionally write a CSV containing identifiers, labels, and predicted PVC probabilities.

When using the provided example files, the output CSV follows the structure:

```text
patient_id,label,p_pvc
...
```

The provided `pvc_predictions.csv` contains **11,172 synthetic segments**, including **515 positive PVC segments** (4.61% prevalence).

The expected aggregate sanity-check performance is approximately:

```text
ROC-AUC: 0.766
```

Small numerical differences may occur across environments or when a new linear probe is trained rather than loading the provided reference probe.

## Expected demo runtime

Runtime depends on hardware and whether the reference linear probe is loaded or trained from scratch.

- Loading the supplied backbone and probe and performing inference on the included synthetic example should typically complete within **a few minutes on a normal desktop CPU**.
- GPU execution is generally faster but is not required.
- Training a new probe may take longer depending on CPU/GPU performance and the number of epochs selected.

These timings refer only to the included demonstration and not to full-scale HiMAE pretraining.

---

# Reference Results

The included `pvc_predictions.csv` contains 11,172 synthetic segments with a PVC prevalence of 4.61% (515 positives). Using the provided backbone and a simple linear probe, the aggregate metric on that split is:

- ROC-AUC approximately **0.766**

This synthetic example is intended as a functional sanity check of the model-loading and evaluation pipeline. It is not intended to reproduce the quantitative clinical performance reported in the manuscript.

---

# Running HiMAE on Your Own Data

To use the research implementation with another PPG dataset:

1. Segment each recording into fixed-length windows.
2. Ensure each window has a consistent sampling frequency and duration.
3. Store the waveforms in an HDF5 dataset with shape `[N, L]`.
4. Store binary labels in a separate dataset with shape `[N]`.
5. Optionally provide participant identifiers so that subject-disjoint train/test splitting can be enforced.
6. Update the notebook or script configuration:
   - `sampling_freq`
   - `seg_len`
   - signal key
   - input file path
   - checkpoint path
7. Load the HiMAE checkpoint and extract encoder features.
8. Freeze the encoder and train a downstream linear classifier, or adapt the downstream evaluation code for full fine-tuning.

For example:

```python
H5_PATH = "/path/to/your/data.h5"
SIGNAL_KEY = "ppg"

CFG = {
    "sampling_freq": 100,
    "seg_len": 10,
}
```

For manuscript-style PPG inputs, a 100 Hz signal segmented into 10-second windows corresponds to:

```text
L = 1000 samples
```

If encoder channel dimensions are modified, update the linear-probe input dimensionality to match the resulting HiMAE bottleneck representation.

---

# Reproducing and Extending the Method

The repository is intentionally modular. To adapt HiMAE to a new task:

- point the data loader to the appropriate HDF5 shards;
- set the sampling frequency and segment duration;
- retain the masked reconstruction objective for self-supervised pretraining;
- extract frozen encoder representations for linear probing, or fine-tune the encoder jointly with a downstream head.

The default bottleneck dimensionality is 256. If encoder channel dimensions are changed, update downstream classifier dimensions accordingly.

For substantially longer sequences, additional encoder depth or other architectural changes may be appropriate to preserve a useful bottleneck temporal resolution after repeated stride-2 downsampling.

---

# Reproduction of Manuscript Results

The full participant-level physiological and clinical datasets used in the manuscript are proprietary and are subject to privacy, regulatory, data-use, and commercial restrictions. They are therefore not distributed with this repository.

Accordingly, this public research release is intended to reproduce the **core software methodology and execution pathway**, rather than to reproduce every numerical result in the manuscript from the original participant-level data.

The repository provides:

- the HiMAE model architecture;
- masked-autoencoder pretraining logic;
- checkpoint loading;
- frozen-representation feature extraction;
- an illustrative downstream PVC linear-probe workflow;
- synthetic example data; and
- expected example outputs.

Numerical source data underlying manuscript display items are provided separately with the manuscript where permitted.

Researchers with access to appropriately formatted PPG datasets can use the instructions above to run the same model and downstream evaluation workflow on their own data.

---

# Notes on Research-Scale Training

The public synthetic demonstration is designed to run on conventional hardware. Full research-scale self-supervised pretraining is substantially more computationally intensive.

For the experiments described in the manuscript:

- pretraining used large-scale 10-second PPG windows;
- training was performed with PyTorch Lightning;
- pretraining converged in approximately **12 hours** when distributed across **four NVIDIA Tesla T4 GPUs**.

These specifications are reported for transparency and are **not** requirements for running the included demonstration.

---

# Troubleshooting

## CUDA is not available

Install a CPU-compatible PyTorch build and run the demonstration on CPU. GPU acceleration is optional for the included example.

## W&B requests authentication

Disable online W&B logging before running:

```bash
export WANDB_DISABLED=true
```

## HDF5 keys do not match

Ensure that the signal and label keys passed to the notebook or evaluation script match the keys in your HDF5 file.

## Shape mismatch in the linear probe

The probe input dimensionality must match the HiMAE encoder output. If the channel configuration of the encoder has been changed, modify the classifier input size accordingly.

## Different sampling frequency or segment duration

Update `sampling_freq`, `seg_len`, and any dependent model or preprocessing settings so that the input sequence length is handled consistently.

---

# Acknowledgements

We thank Minji Han and Rachel Choi for their expertise in UX/UI design and for crafting the specialized visualizations not supported by standard Python libraries; their design contributions were essential to this work.

We also thank Praveen Raja, Matthew Wiggins, and Mike Freedman for their invaluable feedback and insightful discussions throughout the project.
