"""Shared workflow assembly for the dials_output analysis scripts.

Holds constants and helpers specific to this script family's directory and
config conventions: resolution/intensity bin edges, the run-prediction
filename schema, the dials_phenix_cfg dataclass, and a peaks.csv reader.

Generic, broadly reusable helpers live in `refltorch` instead:
- reference-config helpers: `refltorch.tools` (get_reference_*).
- DIALS/PHENIX parsers: `refltorch.tools` (parse_dials_merging_stats,
  parse_phenix_r_values).
- run-path resolution: `refltorch.io` (open_run, glob_predictions,
  epoch_from_path).
- bin-axis + figure helpers: `refltorch.plots._shared`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from refltorch.io import load_config, open_run
from refltorch.tools import get_reference_metadata

logger = logging.getLogger(__name__)

INTENSITY_EDGES = [0, 10, 25, 50, 100, 300, 600, 1000, 1500, 2500, 5000, 10000]
DIALS_EDGES_9B7C = [
    0,
    1.1,
    1.11,
    1.14,
    1.16,
    1.18,
    1.21,
    1.23,
    1.27,
    1.30,
    1.34,
    1.49,
    1.56,
    1.64,
    1.74,
    2.06,
    2.36,
    2.97,
]


@dataclass
class DialsPhenixConfig:
    """Schema for `dials_phenix_cfg.yaml` (written by create_config.py).

    Used both to write the config and to read it back (e.g. in
    process_single_refl.py), so it is the single source of truth for the
    field set. Optional fields carry defaults so older configs that omit
    them still load.
    """

    refl_files: list
    expt_file: str
    dials_env: str
    phenix_env: str
    phenix_eff: str
    pdb: str
    paired_ref_eff: str
    paired_model_eff: str
    columns: str = "intensity"
    merged_mtz: list | None = None
    user_selected_labels: str = ""
    french_wilson_scale: bool = True


def add_run_epoch_cols(
    lf: pl.LazyFrame,
    *,
    path_col: str = "filenames",
    epoch_pattern: str = r"/epoch_(\d+)/",
) -> pl.LazyFrame:
    """Add `run_id` and `epoch` columns extracted from a path column.

    Args:
        lf: LazyFrame scanned with `include_file_paths=path_col`.
        path_col: Column holding the source file path of each row.
        epoch_pattern: Regex (one capture group) for the epoch number;
            override for layouts like `/train_epoch_(\\d+).parquet`.

    Returns:
        The LazyFrame with `run_id` (str) and `epoch` (Int32) columns.
    """
    return lf.with_columns(
        [
            pl.col(path_col).str.extract(r"/run-[^/]+-([^/]+)/", 1).alias("run_id"),
            pl.col(path_col)
            .str.extract(epoch_pattern, 1)
            .cast(pl.Int32)
            .alias("epoch"),
        ]
    )


def _resolve_data_dir(run_cfg) -> Path | None:
    """Resolve a run's dataset directory from its training config.

    Reads `data_loader.args.data_dir` (the integrator convention; the
    `shoebox_file_names.data_dir` override wins if present), falling back to
    `global_vars.data_dir` for other config vintages.

    Args:
        run_cfg: Parsed `run_metadata.yaml` (has a `config` path key).

    Returns:
        The dataset directory, or None if it cannot be read.
    """
    try:
        cfg = load_config(run_cfg["config"])
    except (KeyError, FileNotFoundError, TypeError, OSError):
        return None
    try:
        dl = cfg["data_loader"]["args"]
        sbox = dl.get("shoebox_file_names") or {}
        return Path(sbox.get("data_dir") or dl["data_dir"])
    except (KeyError, TypeError):
        pass
    try:
        return Path(cfg["global_vars"]["data_dir"])
    except (KeyError, TypeError):
        return None


def _resolve_metadata_path(run_cfg) -> Path | None:
    """Resolve a run's `metadata.pt` path from its training config.

    The filename comes from `data_loader.args.shoebox_file_names.reference`
    (the per-reflection reference table, e.g. `metadata.pt`) under the
    dataset directory.

    Args:
        run_cfg: Parsed `run_metadata.yaml` (has a `config` path key).

    Returns:
        The full `metadata.pt` path, or None if it cannot be resolved.
    """
    data_dir = _resolve_data_dir(run_cfg)
    if data_dir is None:
        return None
    ref_name = "metadata.pt"
    try:
        cfg = load_config(run_cfg["config"])
        sbox = cfg["data_loader"]["args"].get("shoebox_file_names") or {}
        ref_name = sbox.get("reference") or "metadata.pt"
    except (KeyError, FileNotFoundError, TypeError, OSError):
        pass
    return data_dir / ref_name


def assemble_run_data(models) -> dict:
    """Assemble per-run paths and metadata for the comparison scripts.

    Leaner sibling of `compare_models._get_run_data`: it skips the W&B
    history call and only gathers what the coset-analysis plots need.

    Args:
        models: List of model specs, each a dict with `run_dir` and optional
            `label`, `name`, `epoch`, and `metadata` keys.

    Returns:
        Mapping from run_id to a dict with `run_cfg`, `run_id`, `label`,
        `model_name`, `model_metadata`, `wandb_log_dir`, `preds`,
        `phenix_logs`, `data_dir`, `metadata_auto` (auto-resolved
        `metadata.pt` path), `epoch` (requested, may be None), and
        `metadata` (explicit override path, may be None).
    """
    run_data = {}
    for spec in models:
        run_cfg, wandb_log = open_run(spec["run_dir"])
        run_id = run_cfg["wandb"]["run_id"]

        model_meta = get_reference_metadata(run_cfg)
        model_name = spec.get("name") or model_meta["qi_name"]
        label = spec.get("label") or f"{model_name}_{run_id}"

        pred_dir = wandb_log / "predictions"
        run_data[run_id] = {
            "run_cfg": run_cfg,
            "run_id": run_id,
            "label": label,
            "model_name": model_name,
            "model_metadata": model_meta,
            "wandb_log_dir": wandb_log.as_posix(),
            "preds": list(pred_dir.glob("**/preds_epoch_*.parquet")),
            "phenix_logs": list(pred_dir.glob("**/phenix_out/refine*.log")),
            "data_dir": _resolve_data_dir(run_cfg),
            "metadata_auto": _resolve_metadata_path(run_cfg),
            "epoch": spec.get("epoch"),
            "metadata": spec.get("metadata"),
        }
    return run_data


def _find_metadata_pt(run, override=None) -> Path | None:
    """Find a run's `metadata.pt`, trying override then data-dir candidates.

    Args:
        run: A `run_data` entry (uses its `metadata` and `data_dir` keys).
        override: Optional explicit `metadata.pt` path, taking precedence.

    Returns:
        The first existing candidate path, or None if none are found.
    """
    candidates = []
    for cand in (override, run.get("metadata"), run.get("metadata_auto")):
        if cand is not None:
            candidates.append(Path(cand))
    data_dir = run.get("data_dir")
    if data_dir is not None:
        data_dir = Path(data_dir)
        candidates.append(data_dir / "metadata.pt")
        candidates.append(data_dir.parent / "metadata.pt")
        candidates.extend(sorted(data_dir.glob("**/metadata.pt")))
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_detector_positions(run, *, override=None):
    """Load detector x/y lookup arrays for a run, indexed by `refl_ids`.

    Detector positions are not stored in the prediction parquets; they live
    in `metadata.pt` and are indexed by the integer `refl_ids` column.

    Args:
        run: A `run_data` entry (uses its `metadata`/`data_dir` keys).
        override: Optional explicit `metadata.pt` path, taking precedence.

    Returns:
        Tuple of (x_lut, y_lut) numpy arrays indexable by integer
        `refl_ids`, or None if no `metadata.pt` is found.
    """
    try:
        import torch
    except ImportError:
        logger.warning("torch not available; cannot load detector positions")
        return None

    path = _find_metadata_pt(run, override)
    if path is None:
        return None
    meta = torch.load(path, map_location="cpu", weights_only=False)
    x = np.asarray(meta["xyzcal.px.0"])
    y = np.asarray(meta["xyzcal.px.1"])
    return x, y


def resolve_dials_cols(columns) -> tuple[str, str]:
    """Return the DIALS reference (intensity, background) column names.

    Handles both parquet vintages: the dotted `intensity.prf.value` /
    `background.mean` names and the bare `intensity` / `background` names.

    Args:
        columns: Iterable of column names present in the predictions frame.

    Returns:
        Tuple of (dials_intensity_col, dials_background_col).
    """
    cols = set(columns)
    i_col = "intensity.prf.value" if "intensity.prf.value" in cols else "intensity"
    bg_col = "background.mean" if "background.mean" in cols else "background"
    return i_col, bg_col


def join_models_on_refl(
    pred_lf,
    run_a,
    epoch_a,
    run_b,
    epoch_b,
    *,
    value_cols,
    id_col="refl_ids",
) -> pl.DataFrame:
    """Inner-join two runs' single-epoch predictions on the reflection id.

    Args:
        pred_lf: Combined predictions LazyFrame with `run_id`, `epoch`, and
            `id_col` columns.
        run_a: First run id.
        epoch_a: Epoch selected for the first run.
        run_b: Second run id.
        epoch_b: Epoch selected for the second run.
        value_cols: Columns carried through, renamed with `_a`/`_b` suffixes.
        id_col: Shared per-reflection id column to join on.

    Returns:
        Joined DataFrame with `id_col` plus `<col>_a` and `<col>_b` columns.
    """

    def _side(run, epoch, suffix):
        return (
            pred_lf.filter((pl.col("run_id") == run) & (pl.col("epoch") == epoch))
            .select([id_col, *value_cols])
            .rename({c: f"{c}{suffix}" for c in value_cols})
        )

    left = _side(run_a, epoch_a, "_a")
    right = _side(run_b, epoch_b, "_b")
    return left.join(right, on=id_col, how="inner").collect()


def read_peaks_csv(path) -> pl.DataFrame:
    """Read an anomalous peaks.csv into a polars DataFrame.

    Args:
        path: Path to a peaks.csv with `seqid`, `residue`, `peakz` columns.

    Returns:
        DataFrame sorted by `seqid`.
    """
    return pl.read_csv(
        path,
        columns=["seqid", "residue", "peakz"],
        schema_overrides={
            "seqid": pl.Int64,
            "residue": pl.String,
            "peakz": pl.Float64,
        },
    ).sort("seqid")
