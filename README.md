# CoBiMNet

Official PyTorch implementation of **CoBiMNet: Context-Guided Bidirectional Memory for Period-Aware Multivariate Forecasting**.

CoBiMNet combines a Cycle-enhanced Pyramidal Hybrid Encoder (CyPHE) for intra-variable periodic and multiscale temporal modeling with a Context-Guided Bidirectional Memory (CoBiM) module for shared-projection bidirectional inter-variable interaction.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Datasets

The seven public benchmarks (`ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`, `weather.csv`, `electricity.csv`, `traffic.csv`) follow the standard splits used in [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Place the CSV files under `./dataset/`:

```text
dataset/
  ETTh1.csv
  ETTh2.csv
  ETTm1.csv
  ETTm2.csv
  weather.csv
  electricity.csv
  traffic.csv
```

Chronological splits are 6:2:2 for ETT (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`) and 7:1:2 for `Weather`, `Electricity`, and `Traffic`. StandardScaler statistics are fitted strictly on the training segment.

## Usage

### Single Run

Train and evaluate on ETTh1 (`L = 720`, `H = 96`):

```bash
python main.py --dataset ETTh1 --data_dir ./dataset --pred_len 96
```

Each dataset uses the hyperparameters reported in the paper by default. For limited GPU memory on high-dimensional datasets (`Electricity`, `Traffic`), pass `--micro_batch_size` (e.g., `--micro_batch_size 4` or `1`) to enable gradient accumulation while preserving the effective batch size.

### Reproduce Main Results

Run all four horizons (`96, 192, 336, 720`) on a single benchmark:

```bash
bash scripts/run_etth1.sh
bash scripts/run_etth2.sh
bash scripts/run_ettm1.sh
bash scripts/run_ettm2.sh
bash scripts/run_weather.sh
bash scripts/run_electricity.sh
bash scripts/run_traffic.sh
```

Or run all 28 benchmark settings sequentially:

```bash
bash scripts/run_all.sh
```

You can override `DATA_DIR` and `DEVICE` via environment variables:

```bash
DATA_DIR=/path/to/dataset DEVICE=cuda:0 bash scripts/run_etth1.sh
```

### Ablation Flags

The five architectural ablations in the paper can be run directly via CLI flags:

- `--no_cobim`: removes the CoBiM block (`w/o CoBiM`)
- `--unidirectional`: uses unidirectional causal mLSTM scanning (`UniMemory`)
- `--no_context`: removes the global context branch (`w/o Context`)
- `--no_cycle`: removes pre-pyramid cycle enhancement (`w/o Cycle`)
- `--a24_gru`: replaces fine-scale self-attention with GRU (`A24-GRU`)

## Hyperparameters

| Dataset | Variables | Cycle `P` | Batch Size | Backbone LR | Cycle LR | LR Schedule | Epochs | Patience | Select From |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| ETTh1 | 7 | 24 | 256 | 1e-4 | 5e-4 | exp (0.9) | 300 | 50 | 5 |
| ETTh2 | 7 | 24 | 256 | 1e-4 | 5e-4 | exp (0.9) | 300 | 50 | 5 |
| ETTm1 | 7 | 96 | 256 | 1e-4 | 5e-4 | step_half | 300 | 50 | 5 |
| ETTm2 | 7 | 96 | 256 | 1e-4 | 5e-4 | exp (0.9) | 300 | 50 | 5 |
| Weather | 21 | 144 | 64 | 1e-4 | 5e-4 | exp (0.9) | 300 | 50 | 1 |
| Electricity | 321 | 24 | 16 | 5e-4 | 1e-4 | exp (0.9) | 100 | 50 | 5 |
| Traffic | 862 | 168 | 16 | 5e-4 | 1e-4 | exp (0.9) | 200 | 30 | 5 |

All runs use look-back length `L = 720`, latent width `D = 512`, pyramid windows `[24, 48, 72, 144]`, `P_len = 24`, initial gate `tanh(g_0) = 0.30`, Adam with MAE loss, cycle weight decay `1e-4`, and fixed seed `2025`.

## Repository Structure

```text
.
  main.py                  # Training and evaluation entrypoint
  data_provider/
    dataset.py             # Chronological splits, scaling, and phase indexing
  exp/
    runner.py              # Training loop, LR scheduling, and evaluation
  layers/
    revin.py               # Reversible Instance Normalization
    cyphe.py               # Cycle-Enhanced Pyramidal Hybrid Encoder
    cobim.py               # Context-Guided Bidirectional Memory
    mlstm/                 # Multi-head matrix LSTM layers and backends
  models/
    cobimnet.py            # CoBiMNet assembly
  scripts/                 # Benchmark run scripts for all 7 datasets
```

## Acknowledgement

This codebase builds upon open-source implementations from [mLANet](https://doi.org/10.1016/j.knosys.2025.113971), [PRformer](https://github.com/usualheart/prformer), [xLSTM](https://github.com/NX-AI/xlstm), [GTR](https://openreview.net/forum?id=QUJBPSfyui), and [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
