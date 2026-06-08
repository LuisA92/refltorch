"""Helpers for resolving paths inside a training run directory.

A run directory holds a `run_metadata.yaml` that points at the W&B log
directory; predictions and per-epoch artifacts live under
`<wandb_log>/predictions`. These helpers centralize that convention so
analysis scripts do not each re-derive it.
"""

import re
from pathlib import Path

from .yaml import load_config


def open_run(run_dir) -> tuple[dict, Path]:
    """Resolve a run directory to its config and W&B log directory.

    Args:
        run_dir: Path to a run directory containing `run_metadata.yaml`.

    Returns:
        Tuple of (config, wandb_log) where config is the parsed
        `run_metadata.yaml` and wandb_log is the directory that holds the
        W&B `files/` and `predictions/` subtrees.
    """
    run_dir = Path(run_dir)
    run_metadata = next(iter(run_dir.glob("run_metadata.yaml")))
    config = load_config(run_metadata)
    wandb_log = Path(config["wandb"]["log_dir"]).parent
    return config, wandb_log


def glob_predictions(wandb_log, pattern: str, *, sort: bool = True) -> list[Path]:
    """Glob files under a run's predictions directory.

    Args:
        wandb_log: The W&B log directory (as returned by `open_run`).
        pattern: Glob pattern matched recursively under `predictions/`
            (e.g. `peaks.csv`, `merged.html`, `preds_epoch_*.parquet`).
        sort: Whether to sort the returned paths.

    Returns:
        List of matching paths.
    """
    pred_dir = Path(wandb_log) / "predictions"
    files = pred_dir.glob(f"**/{pattern}")
    return sorted(files) if sort else list(files)


def epoch_from_path(path) -> int:
    """Extract the epoch index from a path containing an `epoch_<n>` segment.

    Args:
        path: Path or string containing an `epoch_<n>` segment.

    Returns:
        The integer epoch.

    Raises:
        ValueError: If no `epoch_<n>` segment is present.
    """
    match = re.search(r"epoch_(\d+)", str(path))
    if match is None:
        raise ValueError(f"No 'epoch_<n>' segment in path: {path}")
    return int(match.group(1))
