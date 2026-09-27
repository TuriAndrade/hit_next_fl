# Federated HiT-NeXt for Multilabel ECG Classification

**PyTorch · Federated Learning · ECG · Hierarchical Transformers · Distributed Training**

This repository studies **federated learning (FL) for multilabel 12-lead ECG classification under cross-dataset heterogeneity**. Three public ECG datasets — **CODE-15**, **PTB-XL**, and **Chapman-Shaoxing** — are treated as simulated institutional clients, each with different sample sizes, class prevalences, demographic distributions, and multilabel burden.

The model is **HiT-NeXt**, a hybrid hierarchical ECG architecture that combines convolutional patch merging with local, shifted-window transformer attention. The main question is not simply whether a federated model can match centralized-style training, but **when collaboration is useful across heterogeneous institutions and who benefits from it**.

The experiments compare:

- **Local supervised training** on each client independently.
- **Balanced FL**, where every client receives the same local-data budget.
- **Unbalanced FL**, where one low-resource client has 1k training examples while the other two have 10k each.
- **Cross-domain transfer**, where locally trained models are evaluated on other ECG datasets.

The main empirical finding is that FL is most useful when local supervision is scarce: balanced FL improves all three clients in the 1k regime, while unbalanced FL gives substantial gains to a 1k low-resource client with a comparatively small average cost to high-resource clients.

> **Scope note:** this is a research simulation of cross-silo FL using public datasets. Raw ECGs remain local to each simulated client, but the implementation does **not** provide formal privacy guarantees such as secure aggregation or differential privacy.

### If you are reviewing this repo for an interview

A useful five-minute code walkthrough is:

1. **`models/ecg_hit_next.py`** — the hierarchical ConvNet/Transformer backbone and window-attention implementation.
2. **`datasets/ecg_dataset.py`** — the HDF5/CSV contract, lazy I/O, split handling, deterministic subsampling, and DDP-aware loaders.
3. **`tasks/ecg_supervised_task.py`** — task abstraction, optimizer/scheduler setup, gradient-accumulation calculation, and per-class thresholded evaluation.
4. **`federated/client.py`** — how a client receives the global state, trains locally, and returns an update plus its effective sample count.
5. **`federated/server.py`** — deterministic client selection, FedAvg aggregation, global validation, best-round tracking, checkpointing, resume, and final evaluation.
6. **`scripts/ecg_supervised_train.py` / `federated/run.py`** — CLI orchestration and automatic single-/multi-GPU execution.

The code is structured so that ECG-specific model/task logic is separate from the federated orchestration, which is the main architectural choice worth inspecting.

---

## What this repository demonstrates

From an engineering perspective, the project includes more than a single training script:

- A reusable **task registry** that separates model construction, task configuration, and training entry points.
- A generic **HDF5 + CSV ECG data pipeline** with deterministic subsampling and train/validation/test split support.
- **Single-GPU and multi-GPU DDP** training for both supervised and federated experiments.
- A modular **client/server federated-learning implementation** with weighted or uniform FedAvg-style aggregation.
- Deterministic client sampling, experiment seeds, configuration serialization, checkpointing, resume support, early stopping, and final per-client evaluation.
- Automatic mixed precision, gradient accumulation, LR/weight-decay scheduling, and validation-loss model selection.
- Per-class validation threshold selection for imbalanced multilabel evaluation.
- Scripts for dataset statistics and experiment/result analysis.

---

## Research questions

The accompanying study is organized around four questions:

1. **Does FL help when each institution has little labeled data?**
2. **Does that advantage persist as each institution gets more local data?**
3. **Can high-resource institutions improve a low-resource institution through FL?**
4. **What is the cost of collaboration for the high-resource institutions?**

These questions are evaluated on six harmonized ECG abnormalities:

| Label | Abnormality |
|---|---|
| `1dAVb` | First-degree atrioventricular block |
| `RBBB` | Right bundle branch block |
| `LBBB` | Left bundle branch block |
| `SB` | Sinus bradycardia |
| `AF` | Atrial fibrillation |
| `ST` | Sinus tachycardia |

---

## Datasets as federated clients

The federation deliberately uses **different source datasets instead of IID partitions of one dataset**. This creates a more realistic cross-silo stress test: clients differ in label prevalence, positive-label density, demographics, acquisition conditions, and dataset size.

| Client | Records | Mean positive labels / record | Records with ≥1 target |
|---|---:|---:|---:|
| CODE-15 | 345,779 | 0.120 | 10.92% |
| PTB-XL | 21,799 | 0.277 | 24.47% |
| Chapman-Shaoxing | 44,539 | 0.614 | 57.98% |

A particularly strong source of heterogeneity is the label prior. For example, sinus bradycardia appears in **1.62%** of CODE-15 and **2.92%** of PTB-XL records, but in **37.10%** of Chapman-Shaoxing records. Sinus tachycardia similarly rises from **2.19% / 3.79%** to **15.79%**. These shifts make this substantially different from federating random partitions of a common dataset.

---

## Model: HiT-NeXt

HiT-NeXt is a hierarchical ECG model designed to combine **local waveform morphology** with progressively larger temporal context.

```text
12-lead ECG: (T, 12)
        │
        ▼
┌───────────────────────────────┐
│ Stage 1                       │
│ Conv patch merging            │
│ Local / shifted-window attn.  │
│ dim=96, 4 blocks, 4 heads     │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Stage 2                       │
│ dim=192, 4 blocks, 8 heads    │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Stage 3                       │
│ dim=384, 12 blocks, 16 heads  │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Stage 4                       │
│ dim=768, 4 blocks, 32 heads   │
└───────────────┬───────────────┘
                ▼
        MLP classification head
                │
                ▼
          6 output logits
```

### Default model configuration

| Component | Default |
|---|---|
| Input | `2560 × 12` ECG samples/leads |
| Stages | 4 |
| Hidden dimensions | `[96, 192, 384, 768]` |
| Blocks per stage | `[4, 4, 12, 4]` |
| Attention heads | `[4, 8, 16, 32]` |
| Downsampling factor | `4` |
| Attention window | `10` |
| MLP expansion factor | `4` |
| Patch merge convolution | kernel `10`, stride `4`, padding `4` |
| Attention | local shifted-window attention |
| Positional information | combined relative/contextual positional encoding |
| Similarity | cosine similarity enabled |
| Output MLP | `768 → 512 → 6` |

The hierarchy is useful for ECGs because relevant evidence exists at different temporal scales: wave morphology and QRS shape are local, while rate and rhythm abnormalities require broader context.

The architecture is based on the published HiT-NeXt work:

> P. Dutenhefner, T. Rezende, J. G. Fernandes, et al. **“A hybrid hierarchical transformer model for ECG classification and age prediction.”** *Computers in Biology and Medicine*, 203:111462, 2026. https://doi.org/10.1016/j.compbiomed.2026.111462

---

## Federated-learning design

At federated round `t`:

1. The server distributes the current global model to the selected clients.
2. Each selected client independently trains the model on its local ECG data.
3. Clients return updated model state dictionaries and their local example counts.
4. The server aggregates client models using FedAvg-style parameter averaging.
5. The global model is validated on every configured client.
6. The server records history, updates the best checkpoint, and optionally early-stops.
7. The selected final global model is evaluated independently on every client.

### Aggregation

The default is example-count weighted aggregation:

\[
\theta_{t+1} = \sum_k \frac{n_k}{\sum_j n_j}\theta_{t+1}^{(k)}.
\]

The implementation also supports `--aggregation uniform`.

In the **balanced** experiments, clients use equal local sample counts, so weighted and uniform aggregation are effectively equivalent. In the **unbalanced** experiments, example-count weighting gives the 10k clients more influence than the 1k client.

### Client participation

All clients participate by default. The server also supports partial participation through either:

- `--client-fraction`
- `--clients-per-round`

When only a subset is selected, sampling is deterministic for a fixed experiment seed and round index.

### Model selection

A useful implementation detail for interpreting results: **federated checkpoint selection and early stopping are based on aggregated validation loss**, while the main scientific metric in the report is **macro F1**.

---

## Repository structure

```text
hit_next_fl/
├── datasets/
│   └── ecg_dataset.py            # HDF5 + CSV ECG dataset and DataLoader factory
├── federated/
│   ├── aggregation.py            # FedAvg-style state-dict aggregation
│   ├── client.py                 # Local client train/validate/evaluate lifecycle
│   ├── config.py                 # FL/client configuration and client sampling
│   ├── distributed.py            # DDP helpers and synchronization
│   ├── run.py                    # Federated experiment CLI implementation
│   └── server.py                 # Rounds, aggregation, validation, checkpoints
├── models/
│   └── ecg_hit_next.py           # HiT-NeXt implementation
├── optimizers/
│   └── adamw.py                  # AdamW + warmup/cosine scheduling utilities
├── scripts/
│   ├── ecg_dataset_statistics.py # Dataset descriptive statistics
│   ├── ecg_results_analysis.py   # Experiment aggregation/analysis
│   └── ecg_supervised_train.py   # Local supervised training implementation
├── tasks/
│   ├── ecg_hit_next_classification.py
│   └── ecg_supervised_task.py    # Task, loaders, optimization, evaluation
├── trainers/
│   └── supervised_trainer.py     # AMP/DDP training loop and checkpointing
├── utils/                        # Dataset paths, config, seeds, CLI helpers
├── ecg_dataset_statistics_exe.py
├── ecg_federated_train_exe.py
├── ecg_results_analysis_exe.py
├── ecg_supervised_train_exe.py
├── example.env
└── requirements.txt
```

The `TaskDefinition` registry is intentionally used to decouple the CLI/training infrastructure from the concrete model and task. The current registered pair is `ecg_hit_next` + `clf`, but the organization makes additional model/task combinations straightforward to add.

---

## Data contract

The repository expects **preprocessed HDF5 signals plus a row-aligned CSV** for each dataset. Dataset download/preprocessing is not performed by the training scripts.

### HDF5

Expected layout:

```text
dataset.h5
├── signal              float32, shape (N, T, 12)
├── exam_id             int64, shape (N,)
├── train/
│   └── hdf5_index      int64 indices into signal
├── val/
│   └── hdf5_index
└── test/
    └── hdf5_index
```

The split groups are optional if the dataset is instantiated without a split, but the training workflow uses `train`, `val`, and `test` groups.

### CSV

CSV row `i` must correspond to HDF5 `signal[i]`. For classification, the CSV must contain:

```text
1dAVb,RBBB,LBBB,SB,AF,ST
```

Target values are converted to `float32`.

### Loading behavior

The dataset implementation:

- validates the CSV/HDF5 row count;
- validates required target columns;
- lazily opens HDF5 files with `swmr=True`;
- center-crops each ECG to **2,560 temporal samples**;
- supports deterministic random subsampling with a seed;
- uses `DistributedSampler` automatically under DDP.

---

## Installation

```bash
git clone https://github.com/TuriAndrade/hit_next_fl.git
cd hit_next_fl

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Main dependencies include PyTorch, NumPy, pandas, h5py, SciPy, matplotlib, `python-dotenv`, timm, and einops.

> Training is currently **CUDA-only**. Both supervised and federated entry points raise an error when CUDA is unavailable.

---

## Configure dataset paths

Copy the example environment file:

```bash
cp example.env .env
```

Then configure your prepared datasets and output directory:

```dotenv
CODE15_H5_PATH=/path/to/code15.h5
CODE15_CSV_PATH=/path/to/code15.csv

PTBXL_H5_PATH=/path/to/ptbxl.h5
PTBXL_CSV_PATH=/path/to/ptbxl.csv

CHAPMAN_H5_PATH=/path/to/chapman.h5
CHAPMAN_CSV_PATH=/path/to/chapman.csv

SAVE_DIR=/path/to/experiments
```

The top-level `*_exe.py` entry points load `.env` before dispatching to the implementation modules.

---

## Run local supervised training

Example: train HiT-NeXt on 1,000 CODE-15 training ECGs.

```bash
python ecg_supervised_train_exe.py \
  --model-name ecg_hit_next \
  --dataset-name code15 \
  --n-train-samples 1000 \
  --seed 0 \
  --save-dir supervised \
  --save-name code15_1k
```

The same entry point can be used with `ptbxl` or `chapman`.

Model and task defaults can be overridden with JSON dictionaries:

```bash
python ecg_supervised_train_exe.py \
  --model-name ecg_hit_next \
  --dataset-name ptbxl \
  --model-extra-args '{"window_size": 10}' \
  --task-extra-args '{"epochs": 50, "ref_lr": 0.0001}' \
  --save-dir supervised \
  --save-name ptbxl_custom
```

### Default classification training setup

The registered ECG classification task uses:

- `BCEWithLogitsLoss`
- effective batch size `128`
- per-GPU batch size `64`
- automatic gradient accumulation when needed
- up to `100` epochs
- early stopping with patience `5`
- AMP enabled
- AdamW
- LR schedule: `1e-5 → 1e-4 → 1e-5`, with 5 warmup epochs followed by cosine decay
- weight decay `1e-3`
- best-model selection by validation loss

---

## Run balanced federated learning

Example: 1,000 local training examples per client.

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

For the 5k and 10k regimes, change the per-client sample list to:

```text
5000,5000,5000
10000,10000,10000
```

`--n-train-samples`, `--n-val-samples`, and `--n-test-samples` accept one value per configured client, either as a comma-separated string or JSON-style list.

---

## Run unbalanced federated learning

Example: CODE-15 is the low-resource client with 1k examples; PTB-XL and Chapman-Shaoxing use 10k each.

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

To rotate the low-resource institution, keep client order fixed and change the sample-count vector, for example:

```text
CODE-15 low-resource:   1000,10000,10000
PTB-XL low-resource:   10000,1000,10000
Chapman low-resource:  10000,10000,1000
```

---

## Multi-GPU behavior

Both training entry points inspect the number of visible CUDA devices.

- **1 visible GPU:** training runs directly on that device.
- **>1 visible GPU:** the script automatically spawns one process per GPU and uses PyTorch DDP/NCCL.

To force a particular GPU subset:

```bash
CUDA_VISIBLE_DEVICES=0,1 python ecg_federated_train_exe.py ...
```

In supervised training, effective batch size is maintained through gradient accumulation when the requested global batch size exceeds the aggregate per-GPU batch capacity. The default task uses `batch_size=128` and `gpu_batch_size=64`; therefore, more than two GPUs require adjusting one of those values, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python ecg_supervised_train_exe.py \
  --model-name ecg_hit_next \
  --dataset-name code15 \
  --task-extra-args '{"batch_size": 256, "gpu_batch_size": 64}'
```

The task validates that the effective batch size is divisible by `gpu_batch_size × world_size`, so invalid distributed configurations fail early instead of silently changing optimization semantics.

---

## Evaluation

The task reports accuracy, precision, recall, and F1 for each abnormality and uses macro-averaged metrics in the study.

Because the targets are strongly imbalanced, **macro F1 is the primary metric** in the report.

For each class, evaluation searches validation thresholds over:

```text
0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5
```

and selects the threshold maximizing validation F1 by default before evaluating the test set.

### Important cross-domain interpretation

The report's cross-domain experiments use **thresholds selected on the target domain's validation set**. They therefore measure **target-calibrated transfer**, not zero-label/calibration-free deployment to a completely unseen institution.

---

## Experiment outputs

Supervised runs save experiment metadata and evaluation summaries under `SAVE_DIR`:

```text
<run>/
├── config.json
├── eval_summary.json
└── ... trainer checkpoints / history
```

Federated runs additionally produce global FL artifacts such as:

```text
<run>/
├── config.json
├── global_best.pt       # when best-model selection is enabled
# or global_last.pt
├── federated_summary.json
├── final_eval/
│   └── eval_summary.json
└── clients/             # optional per-client checkpoints
```

A federated checkpoint stores the selected model state together with round/history/configuration information, and runs can be resumed with `--resume-from`.

---

## Main results

Results below are from the accompanying June 2026 project report.

### 1. Balanced FL is most valuable in the 1k regime

Macro F1 with 1,000 training examples per client:

| Target client | Individual | Balanced FL | Δ vs. individual |
|---|---:|---:|---:|
| CODE-15 | 0.103 | **0.315** | **+0.212** |
| PTB-XL | 0.276 | **0.412** | **+0.135** |
| Chapman-Shaoxing | 0.376 | **0.417** | **+0.041** |

All three clients benefit when local supervision is scarce.

### 2. The local FL advantage decreases with more data

At 5k and 10k local examples, individual in-domain training becomes competitive and can outperform the global model on CODE-15 and PTB-XL. However, the federated model remains better than the **mean foreign model** for every target in those regimes.

This suggests that, in this experiment, FL shifts from being a local-data booster to being primarily a **collaboration and cross-domain robustness mechanism** as local datasets grow.

### 3. High-resource clients strongly help a 1k client

In the unbalanced setting, one client has 1k examples while the other two have 10k:

| Low-resource client | Individual 1k | Balanced FL 1k | Unbalanced FL | Δ vs. individual |
|---|---:|---:|---:|---:|
| CODE-15 | 0.103 | 0.315 | **0.415** | **+0.312** |
| PTB-XL | 0.276 | 0.412 | **0.636** | **+0.359** |
| Chapman-Shaoxing | 0.376 | 0.417 | **0.585** | **+0.209** |

Across the three scenarios, the low-resource client gains on average:

- **+0.294 macro F1** vs. individual 1k training.
- **+0.164 macro F1** vs. balanced 1k FL.

### 4. The average high-resource cost is much smaller

Average high-resource macro-F1 changes in the unbalanced experiments:

| 10k client | Individual 10k | Unbalanced FL | Δ |
|---|---:|---:|---:|
| CODE-15 | 0.663 | 0.559 | -0.105 |
| PTB-XL | 0.778 | 0.735 | -0.043 |
| Chapman-Shaoxing | 0.565 | 0.633 | +0.068 |

The average effect across high-resource clients is **-0.027 macro F1**, compared with a **+0.294** average gain for the low-resource client.

The trade-off is therefore favorable on average in this experimental setting, but it is **not uniform across domains**: CODE-15 pays a noticeably larger cost than Chapman-Shaoxing, which actually improves.

### Aggregate view

| Comparison | Macro-F1 difference |
|---|---:|
| Individual cross-domain − individual in-domain | -0.145 |
| Balanced FL − individual in-domain | +0.011 |
| Balanced FL − foreign-model mean | +0.156 |
| Unbalanced low-resource − individual 1k | +0.294 |
| Unbalanced low-resource − balanced 1k | +0.164 |
| Unbalanced high-resource − individual 10k | -0.027 |

---

## Interpretation

The experiments do **not** support the claim that one global federated model universally replaces a strong local model.

Instead, they support a more specific conclusion:

> **Federated learning is most useful here as a mechanism for supporting data-scarce institutions and improving robustness across heterogeneous ECG domains.**

With enough local data, domain-specific models can exploit local prevalence and acquisition patterns more effectively. With little local data, collaboration exposes the model to substantially more positive examples and ECG variability. In the unbalanced setting, that extra supervision transfers strongly to the low-resource client.

The behavior also highlights an important FL problem: **statistical heterogeneity changes who benefits from collaboration**. The direction and magnitude of the trade-off depend on which domain is low-resource and which domains dominate the federation.

---

## Engineering design notes

A few implementation choices that are easy to miss when only looking at the final metrics:

### Lazy HDF5 access

The dataset keeps signals in HDF5 and opens the file lazily per process/worker instead of loading the complete ECG corpus into memory. This matters for large waveform datasets and avoids sharing a live HDF5 handle across worker processes.

### Deterministic experiments

Subsets are generated from seeded permutations, client selection is seeded by round, and run configurations are serialized. This makes comparisons between local-data regimes much easier to audit.

### DDP is infrastructure-level, not model-specific

The same task and trainer abstractions are used in single-GPU and multi-GPU execution. The data loader switches to `DistributedSampler`, while the trainer handles DDP synchronization, mixed precision, and accumulation.

### Federated responsibilities are separated

- `FederatedClient` owns local task creation, training, validation, and evaluation.
- `FederatedServer` owns rounds, client selection, aggregation, global validation, checkpoint selection, and final evaluation.
- `FederatedConfig` owns experiment validation and deterministic client selection.

This keeps FL orchestration separate from ECG/model-specific logic.

### The framework supports controlled ablations

Both model configuration and task/training configuration can be overridden from the CLI as JSON. Sample budgets are independently specified per client, which is what makes the balanced/unbalanced experiments reproducible without separate code paths.

---

## Limitations

The project intentionally has a narrower scope than a production medical FL system:

1. **Simulated institutions.** Clients are public datasets, not hospitals connected through a live federated deployment.
2. **No formal privacy layer.** Raw data are not exchanged, but there is no secure aggregation, differential privacy, update encryption, or adversarial leakage evaluation.
3. **Target-calibrated transfer.** Cross-domain test metrics use thresholds selected on the target validation set.
4. **One FL family.** The experiments focus on FedAvg-style global aggregation rather than FedProx, SCAFFOLD, personalized FL, client-specific normalization, or mixture-of-experts approaches.
5. **Three sample-size regimes.** The study evaluates 1k, 5k, and 10k local examples rather than the full scaling curve.
6. **Six harmonized labels.** This enables direct cross-dataset comparison but discards dataset-specific diagnostic richness.
7. **Prepared data required.** The repository expects already harmonized HDF5/CSV inputs; dataset acquisition and complete preprocessing pipelines are outside the current training workflow.

These constraints are important when interpreting the project as **evidence about statistical heterogeneity and collaboration**, rather than as a production privacy-preserving medical system.

---

## Suggested extensions

Natural next steps include:

- FedProx or SCAFFOLD for stronger non-IID optimization baselines.
- Personalized heads or client-specific normalization.
- Secure aggregation and differential privacy experiments.
- Calibration-free cross-domain evaluation.
- Larger and mismatched label spaces.
- More clients and partial-client participation experiments.
- Explicit communication-cost and convergence analysis.
- Repeated-seed confidence intervals for the FL trade-off estimates.

---

## Reproducibility checklist

Before running an experiment:

- [ ] Prepare each dataset as row-aligned HDF5 + CSV.
- [ ] Ensure HDF5 contains the expected train/val/test indices.
- [ ] Set all six dataset paths in `.env`.
- [ ] Set `SAVE_DIR`.
- [ ] Make the intended GPUs visible through `CUDA_VISIBLE_DEVICES`.
- [ ] Record the experiment seed.
- [ ] Record per-client train/val/test sample counts.
- [ ] Record aggregation mode, number of rounds, and local epochs.
- [ ] Keep the saved `config.json` and evaluation summaries with reported results.

---

## References

- Dutenhefner, P., Rezende, T., Fernandes, J. G., et al. *A hybrid hierarchical transformer model for ECG classification and age prediction*. **Computers in Biology and Medicine**, 203:111462, 2026. https://doi.org/10.1016/j.compbiomed.2026.111462
- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS, 2017. https://proceedings.mlr.press/v54/mcmahan17a.html
- Ribeiro, A. H., et al. *Automatic diagnosis of the 12-lead ECG using a deep neural network*. **Nature Communications**, 2020.
- Wagner, P., et al. *PTB-XL, a large publicly available electrocardiography dataset*. **Scientific Data**, 2020.
- Zheng, J., et al. *A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients*. **Scientific Data**, 2020.

---

## Project status

This repository is research code associated with a study of **federated HiT-NeXt under cross-dataset ECG heterogeneity**. The current implementation is designed for controlled experiments and reproducible analysis rather than deployment in a clinical production environment.
