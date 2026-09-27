# Federated HiT-NeXt for Multilabel ECG Classification

**PyTorch · Federated Learning · ECG · Hierarchical Transformers · Distributed Training**

This repository studies **federated learning (FL) for multilabel 12-lead ECG classification under cross-dataset heterogeneity**. Three public ECG datasets — **CODE-15**, **PTB-XL**, and **Chapman-Shaoxing** — are treated as simulated institutional clients with different sample sizes, class prevalences, demographic distributions, and multilabel burden.

The experiments compare local supervised training, balanced FL, unbalanced FL, and cross-domain transfer. The main result is that FL is most useful when local supervision is scarce: low-resource clients benefit substantially from collaboration, while the average cost to high-resource clients remains comparatively small.

> **Scope:** this is a research simulation of cross-silo FL using public datasets. The implementation does not provide formal privacy guarantees such as secure aggregation or differential privacy.

## HiT-NeXt backbone

The model used in this repository is the **official HiT-NeXt implementation** from:

> Pedro Dutenhefner, Turi Rezende, José Geraldo Fernandes, et al.  
> **A hybrid hierarchical transformer model for ECG classification and age prediction.**  
> *Computers in Biology and Medicine*, 203:111462, 2026.  
> [Paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010482526000235?via%3Dihub)

HiT-NeXt is a hierarchical ECG architecture combining convolutional processing with local shifted-window self-attention. The hierarchy progressively reduces temporal resolution while increasing representation dimensionality, allowing the network to combine fine-grained waveform morphology with longer-range rhythm information.

```text
12-lead ECG
    │
    ▼
┌─────────────────────────────────┐
│ Stage 1                         │
│ Conv patch merging              │
│ Local / shifted-window blocks   │
│ dim=96 · 4 blocks · 4 heads     │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Stage 2                         │
│ Conv patch merging              │
│ Local / shifted-window blocks   │
│ dim=192 · 4 blocks · 8 heads    │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Stage 3                         │
│ Conv patch merging              │
│ Local / shifted-window blocks   │
│ dim=384 · 12 blocks · 16 heads  │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Stage 4                         │
│ Conv patch merging              │
│ Local / shifted-window blocks   │
│ dim=768 · 4 blocks · 32 heads   │
└────────────────┬────────────────┘
                 ▼
        Classification head
                 │
                 ▼
          6 ECG abnormalities
```

The six harmonized targets are **1dAVb, RBBB, LBBB, sinus bradycardia, atrial fibrillation, and sinus tachycardia**.

## What is implemented

- **Local supervised training** for each ECG dataset.
- **Balanced federated learning** with equal local-data budgets.
- **Unbalanced federated learning** with low- and high-resource clients.
- **FedAvg-style aggregation**, weighted by client sample count by default, with uniform aggregation also available.
- **Single-GPU and multi-GPU DDP** training.
- HDF5-based ECG loading with lazy I/O, deterministic subsampling, and distributed samplers.
- AMP, gradient accumulation, AdamW, LR scheduling, checkpointing, resume, and early stopping.
- Per-class threshold selection on validation data for imbalanced multilabel evaluation.
- Cross-domain evaluation across all three datasets.

## Repository structure

```text
hit_next_fl/
├── datasets/                 # HDF5/CSV ECG loading
├── federated/                # clients, server, aggregation, FL configuration
├── models/                   # official HiT-NeXt implementation
├── optimizers/               # optimizer and scheduler utilities
├── scripts/                  # training/statistics/result-analysis logic
├── tasks/                    # model/task abstraction and evaluation
├── trainers/                 # supervised AMP/DDP training loop
├── utils/                    # configuration, paths, seeds, CLI helpers
├── ecg_supervised_train_exe.py
├── ecg_federated_train_exe.py
├── ecg_dataset_statistics_exe.py
├── ecg_results_analysis_exe.py
└── requirements.txt
```

For a technical review, the most relevant files are:

- `models/ecg_hit_next.py` — HiT-NeXt architecture.
- `datasets/ecg_dataset.py` — HDF5/CSV data pipeline.
- `tasks/ecg_supervised_task.py` — task setup, optimization, and evaluation.
- `federated/client.py` — local client training.
- `federated/server.py` — federated rounds, aggregation, validation, and checkpoints.
- `federated/run.py` — FL experiment orchestration.

## Data

Each dataset is stored as preprocessed ECG signals in HDF5 plus a row-aligned CSV containing the six target labels. The training pipeline center-crops ECGs to **2,560 samples × 12 leads**.

Dataset download and preprocessing are not performed by the training scripts. Paths are configured through `.env`; see `example.env`.

## Installation

```bash
git clone https://github.com/TuriAndrade/hit_next_fl.git
cd hit_next_fl

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp example.env .env
```

Configure the dataset paths and output directory in `.env` before running experiments.

> Training currently requires CUDA.

## Training

### Local supervised training

```bash
python ecg_supervised_train_exe.py \
  --model-name ecg_hit_next \
  --dataset-name code15 \
  --n-train-samples 1000 \
  --seed 0 \
  --save-dir supervised \
  --save-name code15_1k
```

Dataset names can be changed to `ptbxl` or `chapman`.

### Balanced federated learning

```bash
python ecg_federated_train_exe.py \
  --clients code15,ptbxl,chapman \
  --n-train-samples 1000,1000,1000 \
  --rounds 10 \
  --local-epochs 1 \
  --aggregation weighted \
  --keep-best \
  --early-stopping \
  --seed 0 \
  --save-dir federated \
  --save-name balanced_1k
```

### Unbalanced federated learning

Example with CODE-15 as the low-resource client:

```bash
python ecg_federated_train_exe.py \
  --clients code15,ptbxl,chapman \
  --n-train-samples 1000,10000,10000 \
  --rounds 10 \
  --local-epochs 1 \
  --aggregation weighted \
  --keep-best \
  --early-stopping \
  --seed 0 \
  --save-dir federated \
  --save-name unbalanced_code15_1k
```

With multiple visible GPUs, the training entry points automatically use PyTorch DDP/NCCL.

## Evaluation

The study reports macro-averaged **accuracy, precision, recall, and F1**, with **macro F1** as the primary metric because of the strong class imbalance.

For each class, the evaluation code selects a decision threshold using the target validation set before computing test metrics. Cross-domain results should therefore be interpreted as **target-calibrated transfer**, not calibration-free deployment to a completely unseen institution.

## Main results

### Balanced FL with 1k examples per client

| Target | Individual | Balanced FL | Difference |
|---|---:|---:|---:|
| CODE-15 | 0.103 | **0.315** | **+0.212** |
| PTB-XL | 0.276 | **0.412** | **+0.135** |
| Chapman-Shaoxing | 0.376 | **0.417** | **+0.041** |

Balanced FL improves macro F1 for all clients when local supervision is limited.

### Unbalanced FL

When one client has 1k examples and the other two have 10k:

| Low-resource client | Individual 1k | Unbalanced FL | Difference |
|---|---:|---:|---:|
| CODE-15 | 0.103 | **0.415** | **+0.312** |
| PTB-XL | 0.276 | **0.636** | **+0.359** |
| Chapman-Shaoxing | 0.376 | **0.585** | **+0.209** |

Across the three scenarios, the low-resource client gains **+0.294 macro F1 on average** relative to individual training. High-resource clients experience an average **-0.027 macro-F1 change**.

With larger local datasets, strong in-domain models can outperform the global model on some clients. FL nevertheless remains consistently stronger than relying on models trained only on foreign datasets in the experiments.

The accompanying project report contains the full experimental setup, dataset statistics, cross-domain results, and discussion.

## Limitations

- Public datasets simulate institutions; this is not a live multi-hospital deployment.
- No secure aggregation or differential privacy is implemented.
- Only FedAvg-style aggregation is evaluated.
- Cross-domain evaluation uses target-domain validation data for threshold selection.
- Experiments focus on six labels shared across the three datasets.

## Author

**Turi Rezende**  
Department of Computer Science, Universidade Federal de Minas Gerais (UFMG)
