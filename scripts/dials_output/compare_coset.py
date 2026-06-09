# Head-to-head, model-vs-model comparison plots for coset analysis.
#
# Complements compare_models.py (which compares each model against DIALS).
# This script answers "what does coset inclusion change?" by comparing models
# directly to each other and over the detector face.
#
# Figures produced (saved under --save-dir):
#   <run>/detector_{model_bg,model_intensity}.png    — stat over detector
#   <run>/detector_resid_{bg,intensity}.png          — (model - DIALS) over detector
#   pairs/{labelA}__vs__{labelB}.{bg,intensity}.png  — model-vs-model scatter
#   pairs/corr_{bg,intensity}_vs_resolution.png      — per-pair correlation vs d
#   residual_{bg,intensity}_vs_resolution.png        — per-model mean residual vs d
#   refinement_values_overlay.png                    — all models on one R-value axes
#   correlations.csv                                 — Pearson/Spearman table
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
from refltorch.plots.detector_plots import plot_hexbin_detector
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


def _robust_limits(values, lo=1, hi=99) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


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


def _plot_detector(run_data, pred_lf, sel_epoch, dials_i, dials_bg, save_dir):
    """Hexbin model bg/intensity and (model - DIALS) residuals over detector."""
    for run in run_data:
        pos = load_detector_positions(
            run_data[run], override=run_data[run].get("metadata")
        )
        if pos is None:
            logger.warning("no metadata.pt for run %s; skipping detector plots", run)
            continue
        x_lut, y_lut = pos
        label = run_data[run]["label"]
        out_dir = save_dir / f"{run}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = (
            pred_lf.filter(
                (pl.col("run_id") == run) & (pl.col("epoch") == sel_epoch[run])
            )
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
        rid, qbg, qi, ref_i, ref_bg = (
            rid[keep],
            qbg[keep],
            qi[keep],
            ref_i[keep],
            ref_bg[keep],
        )
        x, y = x_lut[rid], y_lut[rid]

        panels = [
            ("model_bg", qbg, "model background (mean)", None, _robust_limits(qbg)),
            (
                "model_intensity",
                qi,
                "model intensity (mean)",
                None,
                _robust_limits(qi),
            ),
            (
                "resid_bg",
                qbg - ref_bg,
                "model - DIALS background",
                "RdBu_r",
                _symmetric_limit(qbg - ref_bg),
            ),
            (
                "resid_intensity",
                qi - ref_i,
                "model - DIALS intensity",
                "RdBu_r",
                _symmetric_limit(qi - ref_i),
            ),
        ]
        for name, c, clabel, cmap, (vmin, vmax) in panels:
            fig, _ = plot_hexbin_detector(
                x,
                y,
                c,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                clabel=clabel,
                title=f"{label} (epoch {sel_epoch[run]})",
            )
            _savefig(fig, f"{out_dir}/detector_{name}.png")


def _plot_pairwise(run_data, pred_lf, sel_epoch, has_d, save_dir):
    """All-pairwise model-vs-model bg/intensity scatters and correlation table.

    Returns the list of correlation-table rows (each a dict).
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
            ("bg", "qbg_mean", False, "background"),
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
    return rows


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

    corr_rows += _plot_pairwise(run_data, pred_lf, sel_epoch, has_d, save_dir)

    if has_dials:
        corr_rows += _dials_corr_rows(run_data, pred_lf, sel_epoch, dials_i, dials_bg)
        if has_d:
            _plot_residual_vs_resolution(
                run_data, pred_lf, sel_epoch, dials_i, dials_bg, save_dir
            )

    if corr_rows:
        pl.DataFrame(corr_rows).write_csv(f"{save_dir}/correlations.csv")
        logger.info("wrote %s/correlations.csv", save_dir)

    _plot_rvalue_overlay(run_data, save_dir)
    logger.info("done")


if __name__ == "__main__":
    main()
