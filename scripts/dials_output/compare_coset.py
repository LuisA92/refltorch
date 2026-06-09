# Head-to-head, model-vs-model comparison plots for coset analysis.
#
# Complements compare_models.py (which compares each model against DIALS).
# This script answers "what does coset inclusion change?" by comparing models
# directly to each other and over the detector face.
#
# Figures produced (saved under --save-dir):
#   detector/detector_{stat}_{mean,median,min,max}.png: cross-model hexbin
#       grids (one panel per model) on a shared color scale + colorbar, for
#       model bg, model intensity, and the model-minus-DIALS residuals.
#   detector/detector_diff_{bg,intensity}.png: pairwise model-A-minus-model-B
#       difference maps over the detector (shared diverging colorbar).
#   pairs/{labelA}__vs__{labelB}.{bg,intensity}.png: model-vs-model log scatter.
#   pairs/corr_{bg,intensity}_vs_resolution.png: per-pair correlation vs d.
#   residual_{bg,intensity}_vs_resolution.png: per-model mean residual vs d.
#   investigate/bin_{d}_bg_*.png: focused diagnostics for the worst
#       bg-correlation resolution bin (per-model + minus-control detector
#       maps, model-vs-control log scatters, background histogram).
#   refinement_values_overlay.png: all models on one R-value axes.
#   correlations.csv: Pearson/Spearman table.
#
# Reuses the same YAML config as compare_models.py (models/save_dir), plus two
# optional keys:
#   models[].epoch:    pin a checkpoint for that model (default: last).
#   models[].metadata: path to metadata.pt for detector x/y (default: auto from
#                      the run's data_dir). A top-level `metadata:` applies to all.
#
#   uv run python scripts/dials_output/compare_coset.py --config models.yaml

import argparse
import itertools
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from out_utils import (
    DIALS_EDGES_9B7C,
    add_run_epoch_cols,
    assemble_run_data,
    join_models_on_refl,
    load_detector_positions,
    resolve_dials_cols,
)

from refltorch.cli.utils import setup_logging
from refltorch.io import epoch_from_path, load_config
from refltorch.plots import save_figure as _savefig
from refltorch.plots import setup_mpl_config
from refltorch.plots._shared import get_bins as _get_bins
from refltorch.plots._shared import get_palette as _get_palette
from refltorch.plots._shared import set_mpl_fonts
from refltorch.plots.detector_plots import plot_hexbin_detector_grid
from refltorch.plots.metric_plots import plot_binned_metric
from refltorch.plots.refinement_plots import plot_r_values
from refltorch.plots.scatter_plots import plot_scatter_identity
from refltorch.tools import parse_phenix_r_values

logger = logging.getLogger(__name__)

setup_mpl_config()
sns.set_theme(
    context="paper",
    style="ticks",
    font_scale=1.0,
)
set_mpl_fonts(10)

# Detector hexbin stats: (filename tag, array key, colorbar label, diverging).
DETECTOR_STATS = [
    ("model_bg", "qbg_mean", "model background", False),
    ("model_intensity", "qi_mean", "model intensity", False),
    ("resid_bg", "resid_bg", "model - DIALS background", True),
    ("resid_intensity", "resid_intensity", "model - DIALS intensity", True),
]
# Hexbin reductions applied within each detector cell.
REDUCTIONS = {
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-to-head model comparison plots for coset analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config listing models to compare (see file header)",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        help="List of run-directory paths (alternative to --config)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Path to save directory",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Override path to a metadata.pt for detector x/y (applies to all)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )
    return parser.parse_args()


def _resolve_models(args) -> tuple[list[dict], str | None]:
    """Resolve the models to compare from --config or --run-dirs.

    Mirrors `compare_models._resolve_models` but also carries the optional
    per-model `epoch` and `metadata` keys (and a top-level `metadata`).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of (models, save_dir).
    """
    if args.config is not None:
        if args.run_dirs is not None:
            raise ValueError("Pass either --config or --run-dirs, not both")
        cfg = load_config(args.config)
        if "models" not in cfg or not cfg["models"]:
            raise ValueError("config must contain a non-empty `models` list")
        top_meta = cfg.get("metadata", args.metadata)
        models = []
        for entry in cfg["models"]:
            if "run_dir" not in entry:
                raise ValueError("each model entry must have a `run_dir`")
            models.append(
                {
                    "run_dir": entry["run_dir"],
                    "label": entry.get("label"),
                    "name": entry.get("name"),
                    "epoch": entry.get("epoch"),
                    "metadata": entry.get("metadata", top_meta),
                }
            )
        return models, cfg.get("save_dir", args.save_dir)

    if not args.run_dirs:
        raise ValueError("Provide either --config or --run-dirs")
    models = [
        {
            "run_dir": rd,
            "label": None,
            "name": None,
            "epoch": None,
            "metadata": args.metadata,
        }
        for rd in args.run_dirs
    ]
    return models, args.save_dir


def _resolve_epochs(run_data, pred_lf) -> dict:
    """Resolve each run's concrete epoch (requested, else the last available).

    Args:
        run_data: The run_data mapping from `assemble_run_data`.
        pred_lf: Combined predictions LazyFrame with `run_id`/`epoch`.

    Returns:
        Mapping from run_id to the concrete epoch integer to plot.
    """
    avail = (
        pred_lf.select(["run_id", "epoch"])
        .unique()
        .collect()
        .group_by("run_id")
        .agg(pl.col("epoch").max().alias("last"), pl.col("epoch").alias("all"))
    )
    last = {r: lst for r, lst in zip(avail["run_id"], avail["last"])}
    all_epochs = {r: set(es) for r, es in zip(avail["run_id"], avail["all"])}

    resolved = {}
    for run in run_data:
        want = run_data[run]["epoch"]
        if want is None:
            resolved[run] = last[run]
        elif want in all_epochs.get(run, set()):
            resolved[run] = want
        else:
            logger.warning(
                "run %s has no epoch %s; using last (%s)", run, want, last[run]
            )
            resolved[run] = last[run]
    return resolved


def _symmetric_limit(values, hi=99) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    v = float(np.percentile(np.abs(arr), hi))
    return -v, v


def _corr(df, a, b, method) -> float | None:
    """Return the Pearson or Spearman correlation of two columns, or None."""
    sub = df.select(a, b).drop_nulls()
    if sub.height < 2:
        return None
    return sub.select(pl.corr(a, b, method=method)).item()


def _detector_arrays(run_data, run, pred_lf, epoch, dials_i, dials_bg, *, d_range=None):
    """Per-reflection detector x/y and stats for one run (optional d-range).

    Args:
        run_data: The run_data mapping.
        run: Run id to extract.
        pred_lf: Combined predictions LazyFrame.
        epoch: Epoch to select for this run.
        dials_i: DIALS intensity column name.
        dials_bg: DIALS background column name.
        d_range: Optional `(lo, hi)` resolution window (keeps `lo <= d < hi`).

    Returns:
        Dict with `refl_ids`, `x`, `y`, `qbg_mean`, `qi_mean`, `resid_bg`,
        `resid_intensity` numpy arrays, or None if positions are unavailable.
    """
    pos = load_detector_positions(run_data[run], override=run_data[run].get("metadata"))
    if pos is None:
        return None
    x_lut, y_lut = pos

    flt = (pl.col("run_id") == run) & (pl.col("epoch") == epoch)
    if d_range is not None:
        lo, hi = d_range
        flt = flt & (pl.col("d") >= lo) & (pl.col("d") < hi)
    df = (
        pred_lf.filter(flt)
        .select(["refl_ids", "qbg_mean", "qi_mean", dials_i, dials_bg])
        .collect()
    )
    rid = df["refl_ids"].to_numpy().astype(np.int64)
    qbg = df["qbg_mean"].to_numpy()
    qi = df["qi_mean"].to_numpy()
    ref_i = df[dials_i].to_numpy()
    ref_bg = df[dials_bg].to_numpy()

    n = len(x_lut)
    keep = (rid >= 0) & (rid < n)
    if not keep.all():
        logger.warning(
            "run %s: %d/%d refl_ids out of metadata range; dropping",
            run,
            int((~keep).sum()),
            keep.size,
        )
    rid = rid[keep]
    return {
        "refl_ids": rid,
        "x": x_lut[rid],
        "y": y_lut[rid],
        "qbg_mean": qbg[keep],
        "qi_mean": qi[keep],
        "resid_bg": (qbg - ref_bg)[keep],
        "resid_intensity": (qi - ref_i)[keep],
    }


def _plot_detector(run_data, pred_lf, sel_epoch, dials_i, dials_bg, save_dir):
    """Cross-model detector hexbin grids with a unified colorbar.

    One grid per (stat, reduction) puts every model side by side on a shared
    color scale (mean/median/min/max), plus pairwise model-difference grids
    that localize where models disagree on the detector.
    """
    arrays = {}
    for run in run_data:
        arr = _detector_arrays(
            run_data, run, pred_lf, sel_epoch[run], dials_i, dials_bg
        )
        if arr is None:
            logger.warning("no metadata.pt for run %s; skipping detector plots", run)
            continue
        arrays[run] = arr
    if not arrays:
        return

    out_dir = save_dir / "detector"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, key, clabel, diverging in DETECTOR_STATS:
        for red_name, red_fn in REDUCTIONS.items():
            panels = [
                (run_data[r]["label"], arrays[r]["x"], arrays[r]["y"], arrays[r][key])
                for r in arrays
            ]
            cmap, vmin, vmax = None, None, None
            if diverging:
                cmap = "RdBu_r"
                vmin, vmax = _symmetric_limit(np.concatenate([p[3] for p in panels]))
            fig, _ = plot_hexbin_detector_grid(
                panels,
                reduce=red_fn,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                clabel=f"{clabel} ({red_name})",
                suptitle=f"{clabel} over detector ({red_name})",
            )
            _savefig(fig, f"{out_dir}/detector_{name}_{red_name}.png")

    _plot_detector_differences(run_data, arrays, out_dir)


def _plot_detector_differences(run_data, arrays, out_dir):
    """Pairwise (model A - model B) detector difference grids, shared colorbar."""
    runs = list(arrays)
    for metric_name, key in [("bg", "qbg_mean"), ("intensity", "qi_mean")]:
        panels = []
        for run_a, run_b in itertools.combinations(runs, 2):
            a, b = arrays[run_a], arrays[run_b]
            merged = pl.DataFrame(
                {"refl_ids": a["refl_ids"], "va": a[key], "x": a["x"], "y": a["y"]}
            ).join(
                pl.DataFrame({"refl_ids": b["refl_ids"], "vb": b[key]}),
                on="refl_ids",
                how="inner",
            )
            if merged.height == 0:
                continue
            title = f"{run_data[run_a]['label']} - {run_data[run_b]['label']}"
            panels.append(
                (
                    title,
                    merged["x"].to_numpy(),
                    merged["y"].to_numpy(),
                    (merged["va"] - merged["vb"]).to_numpy(),
                )
            )
        if not panels:
            continue
        vmin, vmax = _symmetric_limit(np.concatenate([p[3] for p in panels]))
        fig, _ = plot_hexbin_detector_grid(
            panels,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            clabel=f"{metric_name} difference (A - B)",
            suptitle=f"pairwise {metric_name} difference over detector",
        )
        _savefig(fig, f"{out_dir}/detector_diff_{metric_name}.png")


def _plot_pairwise(run_data, pred_lf, sel_epoch, has_d, save_dir):
    """All-pairwise model-vs-model bg/intensity scatters and correlation table.

    Returns:
        Tuple of (rows, corr_by_bin) where rows is the list of correlation
        table dicts and corr_by_bin maps metric -> {pair_id -> per-bin frame}.
    """
    out_dir = save_dir / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    value_cols = ["qbg_mean", "qi_mean"] + (["d"] if has_d else [])
    runs = list(run_data)
    corr_by_bin = {"qbg_mean": {}, "qi_mean": {}}

    for run_a, run_b in itertools.combinations(runs, 2):
        label_a = run_data[run_a]["label"]
        label_b = run_data[run_b]["label"]
        pair_id = f"{label_a}__vs__{label_b}"

        joined = join_models_on_refl(
            pred_lf,
            run_a,
            sel_epoch[run_a],
            run_b,
            sel_epoch[run_b],
            value_cols=value_cols,
        )
        if joined.height == 0:
            logger.warning("no shared reflections for %s; skipping", pair_id)
            continue

        for metric, key, log_scale, ax_label in [
            ("bg", "qbg_mean", True, "background"),
            ("intensity", "qi_mean", True, "intensity"),
        ]:
            col_a, col_b = f"{key}_a", f"{key}_b"
            pear = _corr(joined, col_a, col_b, "pearson")
            spear = _corr(joined, col_a, col_b, "spearman")
            rows.append(
                {
                    "kind": "model_vs_model",
                    "metric": metric,
                    "a": label_a,
                    "b": label_b,
                    "pearson": pear,
                    "spearman": spear,
                    "n": joined.height,
                }
            )

            sdf = joined
            if log_scale:
                sdf = joined.filter((pl.col(col_a) > 0) & (pl.col(col_b) > 0))
            xa = sdf[col_a].to_numpy()
            yb = sdf[col_b].to_numpy()
            if xa.size == 0:
                continue
            hi = float(max(xa.max(), yb.max()))
            lo = float(min(xa.min(), yb.min())) if log_scale else 0.0
            title = f"{metric}: r={_fmt(pear)}, rho={_fmt(spear)} (n={sdf.height})"
            fig, _ = plot_scatter_identity(
                xa,
                yb,
                identity=(max(lo, 1e-6) if log_scale else lo, hi),
                xlabel=f"{label_a} {ax_label}",
                ylabel=f"{label_b} {ax_label}",
                title=title,
                xscale="log" if log_scale else "linear",
                yscale="log" if log_scale else "linear",
            )
            _savefig(fig, f"{out_dir}/{pair_id}.{metric}.png")

            if has_d:
                corr_by_bin[key][pair_id] = _corr_per_bin(joined, col_a, col_b)

    if has_d:
        _plot_corr_vs_resolution(corr_by_bin, out_dir)
    return rows, corr_by_bin


def _corr_per_bin(joined, col_a, col_b) -> pl.DataFrame:
    """Pearson correlation of two columns within each resolution bin."""
    bin_labels, _ = _get_bins(edges=DIALS_EDGES_9B7C)
    return (
        joined.with_columns(
            pl.col("d_a").cut(DIALS_EDGES_9B7C, labels=bin_labels).alias("bin_labels")
        )
        .group_by("bin_labels")
        .agg(corr=pl.corr(col_a, col_b))
    )


def _plot_corr_vs_resolution(corr_by_bin, out_dir):
    """One correlation-vs-resolution line per model pair, per metric."""
    _, base_df = _get_bins(edges=DIALS_EDGES_9B7C)
    base_df = base_df.collect()
    for key, metric in [("qbg_mean", "bg"), ("qi_mean", "intensity")]:
        df_map = corr_by_bin[key]
        if not df_map:
            continue
        pair_ids = list(df_map)
        fig, _ = plot_binned_metric(
            df_map,
            base_df,
            series_ids=pair_ids,
            labels={p: p for p in pair_ids},
            y_key="corr",
            x_label="resolution bin",
            y_label=f"{metric} correlation (A vs B)",
            title=f"model-vs-model {metric} correlation",
        )
        _savefig(fig, f"{out_dir}/corr_{metric}_vs_resolution.png")


def _plot_residual_vs_resolution(
    run_data, pred_lf, sel_epoch, dials_i, dials_bg, save_dir
):
    """Per-model mean (model - DIALS) residual across resolution bins."""
    bin_labels, base_df = _get_bins(edges=DIALS_EDGES_9B7C)
    base_df = base_df.collect()
    runs = list(run_data)
    labels = {r: run_data[r]["label"] for r in runs}

    for metric, model_col, ref_col in [
        ("bg", "qbg_mean", dials_bg),
        ("intensity", "qi_mean", dials_i),
    ]:
        df_map = {}
        for run in runs:
            df = (
                pred_lf.filter(
                    (pl.col("run_id") == run) & (pl.col("epoch") == sel_epoch[run])
                )
                .with_columns(
                    pl.col("d")
                    .cut(DIALS_EDGES_9B7C, labels=bin_labels)
                    .alias("bin_labels"),
                    (pl.col(model_col) - pl.col(ref_col)).alias("residual"),
                )
                .group_by("bin_labels")
                .agg(mean_resid=pl.col("residual").mean())
                .collect()
            )
            df_map[run] = df
        fig, _ = plot_binned_metric(
            df_map,
            base_df,
            series_ids=runs,
            labels=labels,
            y_key="mean_resid",
            x_label="resolution bin",
            y_label=f"mean {metric} residual (model - DIALS)",
            title=f"{metric} residual vs resolution",
        )
        _savefig(fig, f"{save_dir}/residual_{metric}_vs_resolution.png")


def _dials_corr_rows(run_data, pred_lf, sel_epoch, dials_i, dials_bg) -> list[dict]:
    """Per-model Pearson/Spearman of model bg/intensity against DIALS."""
    rows = []
    for run in run_data:
        df = (
            pred_lf.filter(
                (pl.col("run_id") == run) & (pl.col("epoch") == sel_epoch[run])
            )
            .select(["qbg_mean", "qi_mean", dials_i, dials_bg])
            .collect()
        )
        for metric, model_col, ref_col in [
            ("bg", "qbg_mean", dials_bg),
            ("intensity", "qi_mean", dials_i),
        ]:
            rows.append(
                {
                    "kind": "model_vs_dials",
                    "metric": metric,
                    "a": run_data[run]["label"],
                    "b": "DIALS",
                    "pearson": _corr(df, model_col, ref_col, "pearson"),
                    "spearman": _corr(df, model_col, ref_col, "spearman"),
                    "n": df.height,
                }
            )
    return rows


def _find_control(run_data) -> str:
    """Return the run id of the control model (label `control`), else the first."""
    for run in run_data:
        if str(run_data[run]["label"]).strip().lower() == "control":
            return run
    return next(iter(run_data))


def _parse_bin_range(label) -> tuple[float, float] | None:
    """Parse a closed bin label like `1.3 - 1.34` into `(lo, hi)`, else None."""
    if label is None or " - " not in str(label):
        return None
    lo, hi = str(label).split(" - ")
    try:
        return float(lo), float(hi)
    except ValueError:
        return None


def _worst_bin(corr_by_bin, key, control_label):
    """Find the closed resolution bin with the lowest correlation.

    Prefers pairs that involve the control; falls back to all pairs.

    Returns:
        Tuple of (bin_label, (lo, hi)) for the worst bin, or None.
    """
    df_map = corr_by_bin.get(key, {})
    if not df_map:
        return None
    relevant = {p: d for p, d in df_map.items() if control_label in p} or df_map

    per_bin = {}
    for frame in relevant.values():
        for row in frame.iter_rows(named=True):
            label, corr = row["bin_labels"], row["corr"]
            if label is None or corr is None:
                continue
            per_bin.setdefault(label, []).append(corr)

    candidates = [
        (label, rng, min(corrs))
        for label, corrs in per_bin.items()
        if (rng := _parse_bin_range(label)) is not None
    ]
    if not candidates:
        return None
    label, rng, _ = min(candidates, key=lambda item: item[2])
    return label, rng


def _investigate_worst_bin(
    run_data, pred_lf, sel_epoch, dials_i, dials_bg, corr_by_bin, save_dir
):
    """Drill into the worst bg-correlation resolution bin vs the control.

    Produces (within that bin) per-model background detector maps, model minus
    control difference maps, model-vs-control log scatters, and a background
    histogram, to localize where and why the models diverge from the control.
    """
    control = _find_control(run_data)
    control_label = run_data[control]["label"]
    worst = _worst_bin(corr_by_bin, "qbg_mean", control_label)
    if worst is None:
        logger.warning("could not identify a worst bg-correlation bin; skipping")
        return
    label, (lo, hi) = worst
    logger.info(
        "investigating worst bg-correlation bin '%s' (d in [%s, %s)) vs %s",
        label,
        lo,
        hi,
        control_label,
    )

    out_dir = save_dir / "investigate"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"d_{lo}_{hi}".replace(".", "p")

    arrays = {}
    for run in run_data:
        arr = _detector_arrays(
            run_data, run, pred_lf, sel_epoch[run], dials_i, dials_bg, d_range=(lo, hi)
        )
        if arr is not None and arr["x"].size:
            arrays[run] = arr

    if arrays:
        panels = [
            (
                run_data[r]["label"],
                arrays[r]["x"],
                arrays[r]["y"],
                arrays[r]["qbg_mean"],
            )
            for r in arrays
        ]
        fig, _ = plot_hexbin_detector_grid(
            panels,
            clabel="model background (mean)",
            suptitle=f"background over detector, bin {label}",
        )
        _savefig(fig, f"{out_dir}/bin_{tag}_bg_detector.png")
        _plot_bin_diff_vs_control(
            run_data, arrays, control, control_label, label, tag, out_dir
        )

    _plot_bin_scatter_vs_control(
        run_data,
        pred_lf,
        sel_epoch,
        control,
        control_label,
        (lo, hi),
        label,
        tag,
        out_dir,
    )
    _plot_bin_histogram(run_data, pred_lf, sel_epoch, (lo, hi), label, tag, out_dir)


def _plot_bin_diff_vs_control(
    run_data, arrays, control, control_label, label, tag, out_dir
):
    """Per-model (model - control) background detector maps within the bin."""
    if control not in arrays:
        return
    cref = pl.DataFrame(
        {"refl_ids": arrays[control]["refl_ids"], "vc": arrays[control]["qbg_mean"]}
    )
    panels = []
    for run, arr in arrays.items():
        if run == control:
            continue
        merged = pl.DataFrame(
            {
                "refl_ids": arr["refl_ids"],
                "va": arr["qbg_mean"],
                "x": arr["x"],
                "y": arr["y"],
            }
        ).join(cref, on="refl_ids", how="inner")
        if merged.height == 0:
            continue
        panels.append(
            (
                f"{run_data[run]['label']} - {control_label}",
                merged["x"].to_numpy(),
                merged["y"].to_numpy(),
                (merged["va"] - merged["vc"]).to_numpy(),
            )
        )
    if not panels:
        return
    vmin, vmax = _symmetric_limit(np.concatenate([p[3] for p in panels]))
    fig, _ = plot_hexbin_detector_grid(
        panels,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        clabel=f"bg - {control_label} bg",
        suptitle=f"background vs control over detector, bin {label}",
    )
    _savefig(fig, f"{out_dir}/bin_{tag}_bg_diff_vs_control.png")


def _plot_bin_scatter_vs_control(
    run_data, pred_lf, sel_epoch, control, control_label, d_range, label, tag, out_dir
):
    """Model-vs-control background log scatter within the bin, per model."""
    lo, hi = d_range
    for run in run_data:
        if run == control:
            continue
        joined = join_models_on_refl(
            pred_lf,
            run,
            sel_epoch[run],
            control,
            sel_epoch[control],
            value_cols=["qbg_mean", "d"],
        ).filter(
            (pl.col("d_a") >= lo)
            & (pl.col("d_a") < hi)
            & (pl.col("qbg_mean_a") > 0)
            & (pl.col("qbg_mean_b") > 0)
        )
        if joined.height == 0:
            continue
        xa = joined["qbg_mean_a"].to_numpy()
        yb = joined["qbg_mean_b"].to_numpy()
        pear = _corr(joined, "qbg_mean_a", "qbg_mean_b", "pearson")
        spear = _corr(joined, "qbg_mean_a", "qbg_mean_b", "spearman")
        bound = (float(min(xa.min(), yb.min())), float(max(xa.max(), yb.max())))
        title = f"bin {label} bg: r={_fmt(pear)}, rho={_fmt(spear)} (n={joined.height})"
        fig, _ = plot_scatter_identity(
            xa,
            yb,
            identity=(max(bound[0], 1e-6), bound[1]),
            xlabel=f"{run_data[run]['label']} background",
            ylabel=f"{control_label} background",
            title=title,
            xscale="log",
            yscale="log",
        )
        _savefig(fig, f"{out_dir}/bin_{tag}_bg_{run_data[run]['label']}_vs_control.png")


def _plot_bin_histogram(run_data, pred_lf, sel_epoch, d_range, label, tag, out_dir):
    """Overlaid per-model log-background histograms within the bin."""
    lo, hi = d_range
    palette = _get_palette(list(run_data))
    fig, ax = plt.subplots(figsize=(7, 5))
    drew = False
    for run in run_data:
        vals = (
            pred_lf.filter(
                (pl.col("run_id") == run)
                & (pl.col("epoch") == sel_epoch[run])
                & (pl.col("d") >= lo)
                & (pl.col("d") < hi)
                & (pl.col("qbg_mean") > 0)
            )
            .select("qbg_mean")
            .collect()["qbg_mean"]
            .to_numpy()
        )
        if vals.size == 0:
            continue
        ax.hist(
            np.log10(vals),
            bins=60,
            histtype="step",
            label=run_data[run]["label"],
            color=palette[run],
        )
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("log10 model background")
    ax.set_ylabel("count")
    ax.set_title(f"background distribution, bin {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fig, f"{out_dir}/bin_{tag}_bg_hist.png")


def _plot_rvalue_overlay(run_data, save_dir):
    """Overlay every model's final R-work/R-free on a single axes."""
    dfs = []
    for run in run_data:
        logs = run_data[run]["phenix_logs"]
        if not logs:
            continue
        for f in logs:
            rvals = parse_phenix_r_values(f)
            dfs.append(
                pl.DataFrame(
                    {
                        "run_id": run,
                        "epoch": epoch_from_path(f),
                        "Label": run_data[run]["label"],
                        **rvals,
                    }
                )
            )
    if not dfs:
        logger.warning("no phenix logs found; skipping r-value overlay")
        return

    long_df = (
        pl.concat(dfs, how="vertical_relaxed")
        .unpivot(
            ["r_free_start", "r_work_start", "r_free_final", "r_work_final"],
            index=["run_id", "epoch", "Label"],
        )
        .with_columns(
            pl.when(pl.col("variable").str.contains("work"))
            .then(pl.lit("r_work"))
            .otherwise(pl.lit("r_free"))
            .alias("Metric"),
            pl.when(pl.col("variable").str.contains("final"))
            .then(pl.lit("final"))
            .otherwise(pl.lit("start"))
            .alias("stage"),
        )
        .filter(pl.col("stage") == "final")
    )
    palette = _get_palette(long_df["Label"].unique().to_list())
    fig, _ = plot_r_values(
        long_df, palette=palette, title="R-values (final) across models"
    )
    _savefig(fig, f"{save_dir}/refinement_values_overlay.png")


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main():
    args = parse_args()
    setup_logging(args.verbose)

    models, save_dir_arg = _resolve_models(args)
    run_data = assemble_run_data(models)

    first = next(iter(run_data.values()))
    save_dir = Path(save_dir_arg or Path(first["wandb_log_dir"]) / "plots" / "coset")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("saving figures under %s", save_dir)

    all_preds = [f for run in run_data.values() for f in run["preds"]]
    if not all_preds:
        raise ValueError("no prediction parquets found for any run")
    pred_lf = pl.scan_parquet(all_preds, include_file_paths="filenames")
    pred_lf = add_run_epoch_cols(pred_lf)

    schema = pred_lf.collect_schema().names()
    if "refl_ids" not in schema:
        raise ValueError("predictions lack a `refl_ids` column; cannot align models")
    dials_i, dials_bg = resolve_dials_cols(schema)
    has_dials = dials_i in schema and dials_bg in schema
    has_d = "d" in schema
    sel_epoch = _resolve_epochs(run_data, pred_lf)
    logger.info("selected epochs: %s", sel_epoch)

    corr_rows = []
    if has_dials:
        _plot_detector(run_data, pred_lf, sel_epoch, dials_i, dials_bg, save_dir)
    else:
        logger.warning("no DIALS reference columns; skipping detector/residual plots")

    pairwise_rows, corr_by_bin = _plot_pairwise(
        run_data, pred_lf, sel_epoch, has_d, save_dir
    )
    corr_rows += pairwise_rows

    if has_dials:
        corr_rows += _dials_corr_rows(run_data, pred_lf, sel_epoch, dials_i, dials_bg)
        if has_d:
            _plot_residual_vs_resolution(
                run_data, pred_lf, sel_epoch, dials_i, dials_bg, save_dir
            )
            _investigate_worst_bin(
                run_data,
                pred_lf,
                sel_epoch,
                dials_i,
                dials_bg,
                corr_by_bin,
                save_dir,
            )

    if corr_rows:
        pl.DataFrame(corr_rows).write_csv(f"{save_dir}/correlations.csv")
        logger.info("wrote %s/correlations.csv", save_dir)

    _plot_rvalue_overlay(run_data, save_dir)
    logger.info("done")


if __name__ == "__main__":
    main()
