import argparse
from pathlib import Path

from data_provider.dataset import DATASET_DEFAULTS
from exp.runner import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="CoBiMNet: Multivariate Long-Term Time Series Forecasting")

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="ETTh1",
        choices=list(DATASET_DEFAULTS.keys()) + ["custom"],
        help="Benchmark dataset name or 'custom'.",
    )
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Directory containing CSV files.")
    parser.add_argument("--data_path", type=str, default=None, help="Explicit CSV file path (overrides --data_dir).")
    parser.add_argument("--seq_len", type=int, default=720, help="Look-back window length L.")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction horizon H.")

    # Architecture
    parser.add_argument("--d_model", type=int, default=512, help="Latent variable token dimension D.")
    parser.add_argument("--windows", type=int, nargs="+", default=[24, 48, 72, 144], help="Pyramid window sizes.")
    parser.add_argument("--cycle_len", type=int, default=None, help="Cycle table length P.")
    parser.add_argument("--period_len", type=int, default=24, help="Local 2D fusion kernel span P_len.")
    parser.add_argument("--gate_init", type=float, default=0.3, help="Initial cycle residual gate tanh(g_0).")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability inside CoBiM.")

    # Ablation switches
    parser.add_argument("--no_cycle", action="store_true", help="Disable pre-pyramid cycle enhancement (w/o Cycle).")
    parser.add_argument("--no_cobim", action="store_true", help="Remove CoBiM block (w/o CoBiM).")
    parser.add_argument("--unidirectional", action="store_true", help="Use unidirectional causal mLSTM (UniMemory).")
    parser.add_argument("--no_context", action="store_true", help="Disable global context modulation (w/o Context).")
    parser.add_argument("--a24_gru", action="store_true", help="Replace fine-scale self-attention with GRU (A24-GRU).")

    # Optimization
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=None, help="Effective training batch size.")
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=0,
        help="Optional micro-batch size for gradient accumulation (0 disables accumulation).",
    )
    parser.add_argument("--eval_batch_size", type=int, default=0, help="Validation/test batch size (0 uses micro_batch_size).")
    parser.add_argument("--lr", type=float, default=None, help="Backbone learning rate.")
    parser.add_argument("--cycle_lr", type=float, default=None, help="Cycle branch learning rate.")
    parser.add_argument("--cycle_weight_decay", type=float, default=1e-4, help="Weight decay for cycle branch.")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default=None,
        choices=["exp", "step_half"],
        help="Learning rate schedule ('exp' or 'step_half').",
    )
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="Exponential decay base for 'exp' schedule.")
    parser.add_argument("--epochs", type=int, default=None, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Early-stopping patience.")
    parser.add_argument("--select_from_epoch", type=int, default=None, help="First epoch eligible for checkpointing.")

    # Runtime
    parser.add_argument("--device", type=str, default="auto", help="Device string: 'auto', 'cuda:0', or 'cpu'.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for checkpoints and metrics.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate --checkpoint directly.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for evaluation.")

    args = parser.parse_args()

    defaults = DATASET_DEFAULTS.get(args.dataset, DATASET_DEFAULTS["ETTh1"])
    if args.data_path is None:
        if args.dataset == "custom":
            parser.error("--data_path is required when --dataset custom.")
        args.data_path = str(Path(args.data_dir) / defaults["filename"])
    if args.cycle_len is None:
        args.cycle_len = defaults["cycle_len"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.cycle_lr is None:
        args.cycle_lr = defaults["cycle_lr"]
    if args.lr_schedule is None:
        args.lr_schedule = defaults["lr_schedule"]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.patience is None:
        args.patience = defaults["patience"]
    if args.select_from_epoch is None:
        args.select_from_epoch = defaults["select_from_epoch"]
    if args.save_dir is None:
        args.save_dir = str(Path("results") / f"{args.dataset}_L{args.seq_len}_H{args.pred_len}_seed{args.seed}")

    return args


if __name__ == "__main__":
    run_experiment(parse_args())
