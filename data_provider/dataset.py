from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


DATASET_DEFAULTS = {
    "ETTh1": {
        "filename": "ETTh1.csv",
        "split": "ett_h",
        "cycle_len": 24,
        "batch_size": 256,
        "lr": 1e-4,
        "cycle_lr": 5e-4,
        "lr_schedule": "exp",
        "epochs": 300,
        "patience": 50,
        "select_from_epoch": 5,
    },
    "ETTh2": {
        "filename": "ETTh2.csv",
        "split": "ett_h",
        "cycle_len": 24,
        "batch_size": 256,
        "lr": 1e-4,
        "cycle_lr": 5e-4,
        "lr_schedule": "exp",
        "epochs": 300,
        "patience": 50,
        "select_from_epoch": 5,
    },
    "ETTm1": {
        "filename": "ETTm1.csv",
        "split": "ett_m",
        "cycle_len": 96,
        "batch_size": 256,
        "lr": 1e-4,
        "cycle_lr": 5e-4,
        "lr_schedule": "step_half",
        "epochs": 300,
        "patience": 50,
        "select_from_epoch": 5,
    },
    "ETTm2": {
        "filename": "ETTm2.csv",
        "split": "ett_m",
        "cycle_len": 96,
        "batch_size": 256,
        "lr": 1e-4,
        "cycle_lr": 5e-4,
        "lr_schedule": "exp",
        "epochs": 300,
        "patience": 50,
        "select_from_epoch": 5,
    },
    "Weather": {
        "filename": "weather.csv",
        "split": "ratio_712",
        "cycle_len": 144,
        "batch_size": 64,
        "lr": 1e-4,
        "cycle_lr": 5e-4,
        "lr_schedule": "exp",
        "epochs": 300,
        "patience": 50,
        "select_from_epoch": 1,
    },
    "Electricity": {
        "filename": "electricity.csv",
        "split": "ratio_712",
        "cycle_len": 24,
        "batch_size": 16,
        "lr": 5e-4,
        "cycle_lr": 1e-4,
        "lr_schedule": "exp",
        "epochs": 100,
        "patience": 50,
        "select_from_epoch": 5,
    },
    "Traffic": {
        "filename": "traffic.csv",
        "split": "ratio_712",
        "cycle_len": 168,
        "batch_size": 16,
        "lr": 5e-4,
        "cycle_lr": 1e-4,
        "lr_schedule": "exp",
        "epochs": 200,
        "patience": 30,
        "select_from_epoch": 5,
    },
}


def compute_split_borders(total_rows: int, split_mode: str) -> Tuple[int, int, int]:
    if split_mode == "ett_h":
        train_end, val_end, test_end = 8640, 11520, 14400
    elif split_mode == "ett_m":
        train_end, val_end, test_end = 34560, 46080, 57600
    else:
        num_train = int(total_rows * 0.7)
        num_test = int(total_rows * 0.2)
        train_end, val_end, test_end = num_train, total_rows - num_test, total_rows

    if total_rows < test_end:
        raise ValueError(f"Dataset has {total_rows} rows, fewer than required {test_end}.")
    return train_end, val_end, test_end


class ForecastDataset(Dataset):
    """Sliding-window multivariate dataset returning (history, target, phase_index)."""

    def __init__(
        self,
        values: np.ndarray,
        start: int,
        stop: int,
        seq_len: int,
        pred_len: int,
        cycle_len: int,
    ):
        self.values = values
        self.start = start
        self.stop = stop
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cycle_len = cycle_len

        if len(self) < 1:
            raise ValueError(
                f"Split [{start}, {stop}) is too short for seq_len={seq_len}, pred_len={pred_len}."
            )

    def __len__(self) -> int:
        return self.stop - self.start - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int):
        begin = self.start + index
        origin = begin + self.seq_len
        x = self.values[begin:origin]
        y = self.values[origin : origin + self.pred_len]
        phase = np.int64(origin % self.cycle_len)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(phase, dtype=torch.long)


def load_dataset_splits(
    csv_path: Path,
    split_mode: str,
    seq_len: int,
    pred_len: int,
    cycle_len: int,
) -> Tuple[ForecastDataset, ForecastDataset, ForecastDataset, int]:
    """Load CSV, standardize strictly on the training segment, and construct train/val/test datasets."""
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("date", "OT")]
    if "OT" in df.columns:
        feature_cols.append("OT")

    raw = df[feature_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite values found in {csv_path}.")

    train_end, val_end, test_end = compute_split_borders(len(raw), split_mode)
    scaler = StandardScaler()
    scaler.fit(raw[:train_end])
    values = scaler.transform(raw[:test_end]).astype(np.float32)

    train_ds = ForecastDataset(values, 0, train_end, seq_len, pred_len, cycle_len)
    val_ds = ForecastDataset(values, train_end - seq_len, val_end, seq_len, pred_len, cycle_len)
    test_ds = ForecastDataset(values, val_end - seq_len, test_end, seq_len, pred_len, cycle_len)
    return train_ds, val_ds, test_ds, values.shape[1]
