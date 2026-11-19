import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from data_provider.dataset import DATASET_DEFAULTS, load_dataset_splits
from models.cobimnet import CoBiMNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _MicroBatchSampler:
    def __init__(self, dataset, batch_size: int, micro_batch_size: int):
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.logical = BatchSampler(RandomSampler(dataset), batch_size, drop_last=True)

    def __iter__(self):
        for batch in self.logical:
            for start in range(0, len(batch), self.micro_batch_size):
                yield batch[start : start + self.micro_batch_size]

    def __len__(self) -> int:
        return len(self.logical) * math.ceil(self.batch_size / self.micro_batch_size)


def _make_train_loader(dataset, batch_size: int, micro_batch_size: int, num_workers: int, pin_memory: bool):
    if 0 < micro_batch_size < batch_size:
        sampler = _MicroBatchSampler(dataset, batch_size, micro_batch_size)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    sq_sum = 0.0
    abs_sum = 0.0
    total_elems = 0
    total_windows = 0

    for x, y, phase in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        phase = phase.to(device, non_blocking=True)

        pred = model(x, cycle_index=phase)
        diff = pred - y
        count = y.numel()
        sq_sum += float(diff.square().sum().item())
        abs_sum += float(diff.abs().sum().item())
        total_elems += count
        total_windows += x.size(0)

    return {
        "mse": sq_sum / total_elems,
        "mae": abs_sum / total_elems,
        "windows": total_windows,
    }


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_scheduler(optimizer, schedule: str, gamma: float):
    if schedule == "exp":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: gamma ** max(0, step - 3))
    if schedule == "step_half":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** max(0, step - 1))
    raise ValueError(f"Unsupported lr_schedule: {schedule}")


def run_experiment(args) -> dict:
    set_seed(args.seed)
    device = _resolve_device(args.device)
    pin_memory = device.type == "cuda"

    split_mode = DATASET_DEFAULTS[args.dataset]["split"] if args.dataset in DATASET_DEFAULTS else "ratio_712"
    csv_path = Path(args.data_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    train_ds, val_ds, test_ds, enc_in = load_dataset_splits(
        csv_path=csv_path,
        split_mode=split_mode,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        cycle_len=args.cycle_len,
    )

    micro_bs = args.micro_batch_size if args.micro_batch_size > 0 else args.batch_size
    micro_bs = min(micro_bs, args.batch_size)
    eval_bs = args.eval_batch_size if args.eval_batch_size > 0 else micro_bs

    train_loader = _make_train_loader(train_ds, args.batch_size, micro_bs, args.num_workers, pin_memory)
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = CoBiMNet(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=enc_in,
        d_model=args.d_model,
        cycle_len=args.cycle_len,
        period_len=args.period_len,
        gate_init=args.gate_init,
        windows=args.windows,
        dropout=args.dropout,
        use_cycle=not args.no_cycle,
        use_cobim=not args.no_cobim,
        bidirectional=not args.unidirectional,
        use_context=not args.no_context,
        a24_readout="gru" if args.a24_gru else "attention",
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"Dataset={args.dataset} | L={args.seq_len} H={args.pred_len} N={enc_in} "
        f"| Params={num_params:,} | Device={device}",
        flush=True,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else save_dir / "best.pth"

    best_epoch = 0
    best_val_mae = float("inf")
    start_time = time.time()

    if not args.eval_only:
        param_groups = model.parameter_groups(
            backbone_lr=args.lr,
            cycle_lr=args.cycle_lr,
            cycle_weight_decay=args.cycle_weight_decay,
        )
        optimizer = torch.optim.Adam(param_groups)
        scheduler = _build_scheduler(optimizer, args.lr_schedule, args.lr_gamma)

        stale = 0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)

            accumulated = 0
            seen = 0
            train_loss_sum = 0.0

            for x, y, phase in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                phase = phase.to(device, non_blocking=True)

                pred = model(x, cycle_index=phase)
                loss = F.l1_loss(pred, y)
                (loss * (x.size(0) / args.batch_size)).backward()

                accumulated += x.size(0)
                seen += x.size(0)
                train_loss_sum += float(loss.detach().item()) * x.size(0)

                if accumulated >= args.batch_size:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0

            val_metrics = evaluate(model, val_loader, device)
            val_mae = val_metrics["mae"]
            lrs = [g["lr"] for g in optimizer.param_groups]
            print(
                f"Epoch {epoch:03d}/{args.epochs} | train_mae={train_loss_sum / seen:.5f} "
                f"| val_mae={val_mae:.5f} | val_mse={val_metrics['mse']:.5f} "
                f"| lr={lrs[0]:.2e} | time={time.time() - t0:.1f}s",
                flush=True,
            )

            if epoch >= args.select_from_epoch:
                if val_mae <= best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch
                    stale = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    stale += 1

            scheduler.step()
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).", flush=True)
                break

        if best_epoch == 0:
            best_epoch = args.epochs
            torch.save(model.state_dict(), ckpt_path)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)

    val_final = evaluate(model, val_loader, device)
    test_final = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time

    results = {
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "enc_in": enc_in,
        "seed": args.seed,
        "parameters": num_params,
        "best_epoch": best_epoch,
        "val_mse": val_final["mse"],
        "val_mae": val_final["mae"],
        "test_mse": test_final["mse"],
        "test_mae": test_final["mae"],
        "test_windows": test_final["windows"],
        "elapsed_seconds": round(elapsed, 2),
    }

    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"[Test] {args.dataset} H={args.pred_len} | "
        f"MSE={test_final['mse']:.4f} | MAE={test_final['mae']:.4f} "
        f"| Val MAE={val_final['mae']:.4f} (epoch {best_epoch})",
        flush=True,
    )
    return results
