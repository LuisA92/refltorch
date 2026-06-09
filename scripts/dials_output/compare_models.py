# Figures produced (saved to --save-dir):
#   train_val_{elbo,nll,kl}.png          — training/validation loss curves
#   train_val_{elbo,nll,kl}_gap.png      — train/val loss gap
#   merging_stats.{cchalf,rpim,isigi,ccanom}.run_*.model_*.png — per-run merging stats
#   anomalous_{total,mean,max,min,median}_signal.png — anomalous signal across epochs
#   anomalous_iod_{204,205,206}_model_peaks.png      — per-residue peak heights
#   refinement_values_final_run_*.png                 — R-work/R-free over epochs
#
# Requires: W&B run directories, DIALS merged.html, PHENIX refine*.log
#
# Models to compare can be given either with --run-dirs (quick path; labels
# default to "<qi_name>_<run_id>") or with a --config YAML:
#
#   save_dir: /path/to/out          # optional
#   seqids: [204, 205, 206]         # optional
#   models:
#     - run_dir: /abs/run-A
#       label: "Gamma (global)"     # optional legend label
#       name: gammaA                # optional model-name override
#     - run_dir: /abs/run-B
#       label: "Gamma (per-bin)"

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import pandas as pd
import polars as pl
import seaborn as sns
import wandb
from out_utils import DIALS_EDGES_9B7C, INTENSITY_EDGES, add_run_epoch_cols

from refltorch.cli.utils import setup_logging
from refltorch.io import epoch_from_path, load_config, open_run
from refltorch.plots import save_figure as _savefig
from refltorch.plots import setup_mpl_config
from refltorch.plots._shared import (
    get_bins as _get_bins,
)
from refltorch.plots._shared import (
    get_palette as _get_palette,
)
from refltorch.plots._shared import set_mpl_fonts
from refltorch.plots.correlation_plots import plot_correlation_vs_bin
from refltorch.plots.loss_plots import plot_loss_gap, plot_train_val_metric
from refltorch.plots.merging_stats import plot_stat_vs_resolution
from refltorch.plots.metric_plots import (
    plot_binned_metric,
    plot_binned_metric_by_epoch,
    plot_metric_over_epoch,
)
from refltorch.plots.refinement_plots import plot_r_values
from refltorch.plots.scatter_plots import plot_scatter_identity
from refltorch.tools import (
    get_reference_metadata as _get_reference_metadata,
)
from refltorch.tools import parse_dials_merging_stats, parse_phenix_r_values

logger = logging.getLogger(__name__)

# Setup mpl config for consistent plotting
setup_mpl_config()

sns.set_theme(
    context="paper",  # correct for LaTeX
    style="ticks",
    font_scale=1.0,  # IMPORTANT: don't rescale fonts
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to compare outputs from different models"
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
        help="List of paths to the run-directories (alternative to --config)",
    )
    parser.add_argument(
        "--seqids",
        nargs="+",
        default=[204, 205, 206],
        help="List of anomalous atom sequence ids",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Path to save directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )

    return parser.parse_args()


def _resolve_models(args) -> tuple[list[dict], str | None, list]:
    """Resolve the models to compare from --config or --run-dirs.

    Each returned model spec is a dict with `run_dir` and optional `label`
    (legend text) and `name` (model-name override). Exactly one of --config
    or --run-dirs must be given.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of (models, save_dir, seqids).
    """
    if args.config is not None:
        if args.run_dirs is not None:
            raise ValueError("Pass either --config or --run-dirs, not both")
        cfg = load_config(args.config)
        if "models" not in cfg or not cfg["models"]:
            raise ValueError("config must contain a non-empty `models` list")
        models = []
        for entry in cfg["models"]:
            if "run_dir" not in entry:
                raise ValueError("each model entry must have a `run_dir`")
            models.append(
                {
                    "run_dir": entry["run_dir"],
                    "label": entry.get("label"),
                    "name": entry.get("name"),
                }
            )
        save_dir = cfg.get("save_dir")
        seqids = cfg.get("seqids", [204, 205, 206])
        return models, save_dir, seqids

    if not args.run_dirs:
        raise ValueError("Provide either --config or --run-dirs")
    models = [{"run_dir": rd, "label": None, "name": None} for rd in args.run_dirs]
    return models, args.save_dir, args.seqids


def _get_reference_paths(run_dirs) -> dict[str, Path]:
    # since all models use the same reference, we can
    # just grab the first config file
    path = Path(run_dirs[0])
    run_cfg = list(path.glob("run_metadata.yaml"))[0]
    run_cfg = load_config(run_cfg)
    cfg = load_config(run_cfg["config"])
    output_cfg = cfg["output"]
    return {k: Path(v) for k, v in output_cfg.items()}


def _plot_metric(
    run_ids: Iterable,
    df_map: dict[str, pl.DataFrame],
    base_df: pl.DataFrame,
    run_data: dict,
    x_label: str,
    y_label: str,
    title: str,
    x_key: str = "bin_id",
    y_key: str = "mean_qi_var",
    y_scale: bool | None = None,
):
    # plotting one binned metric line per model
    run_ids = list(run_ids)
    labels = {r: run_data[r]["label"] for r in run_ids}
    return plot_binned_metric(
        df_map,
        base_df,
        series_ids=run_ids,
        labels=labels,
        x_key=x_key,
        y_key=y_key,
        x_label=x_label,
        y_label=y_label,
        title=title,
        y_log=y_scale is not None,
    )


def _plot_per_epoch_metric(
    df,
    base_df: pl.DataFrame,
    x_label: str,
    y_label: str,
    title: str,
    fname: str,
    epochs,
    y_scale: str | None = None,
    x_key: str = "bin_id",
    y_key: str = "mean_qi_var",
):
    # one binned metric line per epoch, saved to fname
    fig, _ = plot_binned_metric_by_epoch(
        df,
        base_df,
        epochs=epochs,
        x_key=x_key,
        y_key=y_key,
        x_label=x_label,
        y_label=y_label,
        title=title,
        y_log=y_scale is not None,
    )
    _savefig(fig, fname)


def _get_df_map(
    lf,
    run_ids,
    edges,
    bin_labels,
    bin_key="qi_mean",
    group_key=["bin_labels"],
) -> dict:
    df_map = {}
    for r in run_ids:
        lf_r = lf.filter(pl.col("run_id") == r)

        lf_ = (
            lf_r.with_columns(
                pl.col(bin_key)
                .cut(
                    edges,
                    labels=bin_labels,
                )
                .alias("bin_labels")
            )
            .group_by(group_key)
            .agg(
                fano=(pl.col("qi_var") / pl.col("qi_mean")).mean(),
                mean_qi_var=pl.col("qi_var").mean(),
                var_qi_var=pl.col("qi_var").var(),
                min_qi_var=pl.col("qi_var").min(),
                max_qi_var=pl.col("qi_var").max(),
                mean_qi_mean=pl.col("qi_mean").mean(),
                var_qi_mean=pl.col("qi_mean").var(),
                min_qi_mean=pl.col("qi_mean").min(),
                max_qi_mean=pl.col("qi_mean").max(),
                n=pl.len(),
            )
        )
        df_map[r] = lf_.collect()
    return df_map


def _plot_anomalous_metric(
    peak_lf: pl.LazyFrame,
    run_ids: Iterable,
    epoch_df: pl.DataFrame,
    run_data: dict,
    reference_data: pl.LazyFrame | None,
    metric: str,
):
    # set up reference data if provided
    ref_value = None
    if reference_data is not None:
        reference_data = reference_data.filter(pl.col("seqid").is_in([204, 205, 206]))
        ref_lf = reference_data.select(
            total_signal=pl.col("peakz").sum(),
            mean_signal=pl.col("peakz").mean(),
            max_signal=pl.col("peakz").max(),
            min_signal=pl.col("peakz").min(),
            median_signal=pl.col("peakz").median(),
        ).collect()
        ref_value = ref_lf[metric].item()

    palette = _get_palette(run_ids)
    series = []
    for r in run_ids:
        label = run_data[r]["label"]
        df = peak_lf.filter(pl.col("run_id") == r).collect()
        df = epoch_df.join(df, on="epoch", how="left").sort("epoch")
        series.append((label, df["epoch"], df[metric], palette[r]))

    return plot_metric_over_epoch(
        series,
        ref_value=ref_value,
        ref_label="DIALS",
        x_label="epoch",
        y_label=f"{metric}",
        title=f"{metric} over epochs",
    )


def _get_val_loss(wb_data):
    df = pl.DataFrame(wb_data)
    df = df.select(["trainer/global_step", "epoch", "val elbo", "val nll", "val kl"])
    return df.drop_nulls()


def _get_train_loss(wb_data):
    df = pl.DataFrame(wb_data)
    df = df.select(
        ["trainer/global_step", "epoch", "train elbo", "train nll", "train kl"]
    )
    return df.drop_nulls()


def _get_anomalous_peak_paths(reference_paths, run_data):
    # MODEL anomalous peak heights
    all_peaks = [csv for run in run_data.values() for csv in run.get("peak.csv", [])]
    return all_peaks


def _get_reference_merging_stats(reference_paths):
    # Getting reference DIALS merged.html
    ref_merge_stats_df = None
    if reference_paths.get("dials_merge_html") is not None:
        # Getting reference merging statistics
        ref_tbl1, ref_tbl2 = pd.read_html(reference_paths["dials_merge_html"])
        ref_merge_stats_df = parse_dials_merging_stats(ref_tbl2).collect()

    return ref_merge_stats_df


def _get_peak_lf(
    run_data,
    reference_paths,
) -> pl.LazyFrame:
    peak_files = _get_anomalous_peak_paths(
        reference_paths=reference_paths,
        run_data=run_data,
    )
    lf = pl.scan_csv(
        peak_files,
        include_file_paths="filenames",
        schema_overrides={
            "seqid": pl.Int64,
            "run_id": pl.Int64,
            "epoch": pl.Int64,
            "peakz": pl.Float32,
        },
    )

    # Extracting run_id and epoch from filename and appending as columns
    lf = add_run_epoch_cols(lf)
    return lf


def _get_merging_stat_df(
    run_data: dict,
) -> pl.DataFrame:
    # NOTE:
    # Plot merging statistics from DIALS html files
    merging_stat_dfs = []
    for run in run_data.values():
        htmls = run["merged.html"]
        model_name = run["model_name"]
        run_id = run["run_id"]
        label = run["label"]
        for h in htmls:
            epoch = epoch_from_path(h)
            tbl1, tbl2 = pd.read_html(h)
            merging_stat_dfs.append(
                parse_dials_merging_stats(
                    tbl2,
                    epoch=epoch,
                    run_id=run_id,
                    model_name=model_name,
                ).with_columns(label=pl.lit(label))
            )

    merging_stat_df = pl.concat(merging_stat_dfs).collect()
    return merging_stat_df


def _plot_merging_stats(
    run_ids,
    merging_stat_df,
    ref_merge_stats_df,
    save_dir: Path,
):
    save_dir.mkdir(exist_ok=True)

    # (y_key, ylabel, yticks, filename tag)
    specs = [
        ("cchalf", "CChalf", [0.0, 0.25, 0.5, 0.75, 1.0], "cchalf"),
        ("rpim", "Rpim", [0.0, 0.25, 0.5, 0.75, 1.0], "rpim"),
        ("meani_sigi", "I/Sig(I)", [0.0, 25, 50, 75, 100, 125], "isigi"),
        ("ccanom", "CCanom", [-0.5, -0.25, 0.0, 0.25], "ccanom"),
    ]

    for run in run_ids:
        df_ = merging_stat_df.filter(pl.col("run_id") == run)
        model_name = df_["model_name"].unique().item()
        label = df_["label"].unique().item()

        for y_key, ylabel, yticks, tag in specs:
            fig, _ = plot_stat_vs_resolution(
                df_,
                y_key=y_key,
                ref=ref_merge_stats_df[y_key],
                ylabel=ylabel,
                yticks=yticks,
                title=label,
            )
            _savefig(
                fig,
                f"{save_dir}/merging_stats.{tag}.run_{run}.model_{model_name}.png",
            )


def _get_corr_df(pred_lf):
    dials_edges = DIALS_EDGES_9B7C
    bin_labels, base_df = _get_bins(edges=dials_edges)

    corr_df = (
        pred_lf.with_columns(
            pl.col("d").cut(dials_edges, labels=bin_labels).alias("d_bins"),
        )
        .group_by(["run_id", "epoch", "d_bins"])
        .agg(
            corr_I=pl.corr("qi_mean", "intensity.prf.value"),
            corr_var_I=pl.corr("qi_var", "intensity.prf.variance"),
            corr_bg=pl.corr("qbg_mean", "background.mean"),
        )
        .sort("epoch")
        .collect()
    )
    return corr_df


def _plot_correlations(
    corr_df,
    save_dir: Path,
):
    corr_keys = ["corr_I", "corr_bg", "corr_var_I"]
    for (run,), df_run in corr_df.group_by("run_id"):
        out_dir = save_dir / f"{run}"
        out_dir.mkdir(exist_ok=True)

        for y_key in corr_keys:
            fig, _ = plot_correlation_vs_bin(df_run, y_key=y_key)
            _savefig(fig, f"{out_dir}/run_{run}_dials_{y_key}.png")


def _get_rval_df(
    run_ids,
    run_data,
) -> pl.DataFrame:
    dfs = []
    for run in run_ids:
        for f in run_data[run]["phenix_logs"]:
            rvals = parse_phenix_r_values(f)
            epoch = epoch_from_path(f)
            df = pl.DataFrame(
                {
                    "run_id": run,
                    "epoch": epoch,
                    "fname": f.as_posix(),
                    "model_name": run_data[run]["model_name"],
                    "label": run_data[run]["label"],
                    **rvals,
                },
            )
            dfs.append(df)
    return pl.concat(dfs)


def _get_rval_long_df(
    df,
) -> pl.DataFrame:
    # Long format for plotting
    long_df = df.unpivot(
        [
            "r_free_start",
            "r_work_start",
            "r_free_final",
            "r_work_final",
        ],
        index=[
            "run_id",
            "epoch",
            "model_name",
            "label",
        ],
    )

    # adding identifiers for styling
    long_df = long_df.with_columns(
        [
            pl.when(pl.col("variable").str.contains("work"))
            .then(pl.lit("r_work"))
            .otherwise(pl.lit("r_free"))
            .alias("Metric"),
            pl.when(pl.col("variable").str.contains("final"))
            .then(pl.lit("final"))
            .otherwise(pl.lit("start"))
            .alias("stage"),
        ]
    )
    long_df = long_df.with_columns(
        Label=pl.col("label"),
    )
    return long_df


def _get_pred_lf(
    run_data,
) -> pl.LazyFrame:
    all_preds = [f for run in run_data.values() for f in run.get("preds", [])]
    pred_lf = pl.scan_parquet(
        all_preds,
        include_file_paths="filenames",
    )
    pred_lf = add_run_epoch_cols(pred_lf)
    return pred_lf


def _plot_run_merging_stats(run_ids, pred_lf, save_dir: Path):
    pad = 2.0

    for run in run_ids:
        out_dir = save_dir / f"{run}"
        out_dir.mkdir(exist_ok=True)

        df = (
            pred_lf.filter(pl.col("run_id") == run)
            .select(
                [
                    "epoch",
                    "qi_mean",
                    "qi_var",
                    "qbg_mean",
                    "intensity.prf.value",
                    "intensity.prf.variance",
                    "background.mean",
                ]
            )
            .collect()
        )

        for (epoch,), df_epoch in df.group_by("epoch"):
            # Filter: positive DIALS variance only
            df_good = df_epoch.filter(pl.col("intensity.prf.variance") > 0)
            n_total = len(df_epoch)

            # All reflections (symlog)
            x_max = (
                df_epoch.select(pl.max("qi_mean", "intensity.prf.value"))
                .max_horizontal()
                .item()
            ) * pad
            y_min = df_epoch.select(pl.min("intensity.prf.value")).item() * pad
            fig, _ = plot_scatter_identity(
                df_epoch["qi_mean"],
                df_epoch["intensity.prf.value"],
                identity=(0, x_max),
                title=f"Run {run} | Epoch {epoch}\nAll reflections (n={n_total:,})",
                xlabel="Model Intensity",
                ylabel="DIALS intensity.prf.value",
                xlim=(0.0, x_max),
                ylim=(y_min, x_max),
                xscale="symlog",
                yscale="symlog",
            )
            _savefig(fig, f"{out_dir}/run_{run}_vs_dials_I_{epoch}.png")

            # Filtered: positive variance only (log-log)
            df_pos = df_good.filter(
                (pl.col("qi_mean") > 0) & (pl.col("intensity.prf.value") > 0)
            )
            if len(df_pos) > 0:
                x_max_f = (
                    df_pos.select(pl.max("qi_mean", "intensity.prf.value"))
                    .max_horizontal()
                    .item()
                ) * pad
                fig, _ = plot_scatter_identity(
                    df_pos["qi_mean"],
                    df_pos["intensity.prf.value"],
                    identity=(1e-1, x_max_f),
                    title=(
                        f"Run {run} | Epoch {epoch}\n"
                        f"Filtered: var>0 & I>0 (n={len(df_pos):,} / {n_total:,})"
                    ),
                    xlabel="Model Intensity",
                    ylabel="DIALS intensity.prf.value",
                    xscale="log",
                    yscale="log",
                )
                _savefig(fig, f"{out_dir}/run_{run}_vs_dials_I_filtered_{epoch}.png")

            # Background
            bg_model_key = "qbg_mean"
            bg_dials_key = "background.mean"
            x_max = (
                df_epoch.select(pl.max(bg_model_key, bg_dials_key))
                .max_horizontal()
                .item()
            )
            y_min = df_epoch.select(pl.min(bg_dials_key)).item()
            fig, _ = plot_scatter_identity(
                df_epoch[bg_model_key],
                df_epoch[bg_dials_key],
                identity=(0, x_max),
                title=f"Run {run} | Epoch {epoch}",
                xlabel="Model bg",
                ylabel=bg_dials_key,
                xlim=(0.0, x_max),
                ylim=(y_min, x_max),
            )
            _savefig(fig, f"{out_dir}/run_{run}_vs_dials_bg_{epoch}.png")

            # Variance
            df_var = df_good.filter(pl.col("qi_var") > 0)
            if len(df_var) > 0:
                x_max_v = (
                    df_var.select(pl.max("qi_var", "intensity.prf.variance"))
                    .max_horizontal()
                    .item()
                ) * pad
                fig, _ = plot_scatter_identity(
                    df_var["qi_var"],
                    df_var["intensity.prf.variance"],
                    identity=(1e-1, x_max_v),
                    title=(
                        f"Run {run} | Epoch {epoch}\n"
                        f"var>0 (n={len(df_var):,} / {n_total:,})"
                    ),
                    xlabel="Model var(I)",
                    ylabel="intensity.prf.variance",
                    xscale="log",
                    yscale="log",
                )
                _savefig(fig, f"{out_dir}/run_{run}_vs_dials_var_epoch_{epoch}.png")


# FIX: Modify to plot the start/final rvalues as two separate plots
def _plot_r_values(
    long_df: pl.DataFrame,
    save_dir: str | Path,
    linewidth: int = 1,
    ref_path: Path | None = None,
):
    # shared palette over all labels so colors are stable across runs
    palette = _get_palette(long_df["Label"].unique().to_list())
    ref_vals = parse_phenix_r_values(ref_path) if ref_path is not None else None

    for (run,), run_df in long_df.group_by(pl.col("run_id")):
        # plot only final r-values
        run_df = run_df.filter(pl.col("stage") == "final")
        label = run_df["Label"].unique().to_list()[0]
        title = f"Final refinement values vs epoch\nmodel: {label}"

        save_dir.mkdir(exist_ok=True)
        fig, _ = plot_r_values(
            run_df,
            palette=palette,
            ref_vals=ref_vals,
            linewidth=linewidth,
            title=title,
        )
        _savefig(fig, f"{save_dir}/refinement_values_final_run_{run}.png")


def _get_loss_dfs(run_data) -> tuple[pl.DataFrame, pl.DataFrame]:
    val_loss_dfs = []
    train_loss_dfs = []

    for v in run_data.values():
        run_id = v["run_id"]
        model_name = v["model_name"]
        label = v["label"]
        val_loss_dfs.append(
            _get_val_loss(v["loss_df"]).with_columns(
                run_id=pl.lit(run_id),
                name=pl.lit(model_name),
                label=pl.lit(label),
                Stage=pl.lit("val"),
            )
        )
        train_loss_dfs.append(
            _get_train_loss(v["loss_df"]).with_columns(
                run_id=pl.lit(run_id),
                name=pl.lit(model_name),
                label=pl.lit(label),
                Stage=pl.lit("train"),
            )
        )

    val_loss_df = pl.concat(val_loss_dfs)
    train_loss_df = pl.concat(train_loss_dfs)

    return train_loss_df, val_loss_df


def _plot_loss_gap(
    loss_gap_df,
    metric,
    save_dir,
):
    fig, _ = plot_loss_gap(loss_gap_df, metric=metric)
    _savefig(fig, f"{save_dir}/train_val_{metric}.png")


def _get_loss_gap_df(train_loss_df, val_loss_df) -> pl.DataFrame:
    # combining train and val
    loss_gap_df = val_loss_df.join(
        train_loss_df,
        on=["epoch", "run_id"],
        how="left",
    )
    loss_gap_df = loss_gap_df.with_columns(
        kl_gap=pl.col("val kl") - pl.col("train kl"),
        nll_gap=pl.col("val nll") - pl.col("train nll"),
        elbo_gap=pl.col("val elbo") - pl.col("train elbo"),
    )
    return loss_gap_df


def _get_long_loss_df(
    val_loss_df,
    train_loss_df,
) -> pl.DataFrame:
    val_long_ = val_loss_df.unpivot(
        ["val nll", "val kl", "val elbo"],
        index=["run_id", "epoch", "Stage", "label"],
    )
    train_long_ = train_loss_df.unpivot(
        ["train nll", "train kl", "train elbo"],
        index=["run_id", "epoch", "Stage", "label"],
    )
    long_loss_df = val_long_.vstack(train_long_)
    return long_loss_df


def _plot_train_val_loss(
    long_loss_df: pl.DataFrame,
    save_dir: str | Path,
):
    palette = _get_palette(long_loss_df["label"].unique().to_list())

    # Metrics to plot together
    metrics = {
        "nll": ("train nll", "val nll"),
        "kl": ("train kl", "val kl"),
        "elbo": ("train elbo", "val elbo"),
    }

    for k, variables in metrics.items():
        ymax = 2000 if k in ["nll", "elbo"] else None
        fig, _ = plot_train_val_metric(
            long_loss_df,
            variables=variables,
            palette=palette,
            title=f"Train/Val {k.upper()} vs epoch ",
            ylabel=f"{k.upper()}",
            ymax=ymax,
        )
        _savefig(fig, f"{save_dir}/train_val_{k}.png")


def _get_run_data(models):
    # Dictionary for model metadata
    run_data = {}

    # Getting data paths for each model
    for spec in models:
        run_cfg, wandb_log = open_run(spec["run_dir"])

        # wandb metadata
        project = run_cfg["wandb"]["project"]
        run_id = run_cfg["wandb"]["run_id"]

        model_meta = _get_reference_metadata(run_cfg)
        model_name = spec.get("name") or model_meta["qi_name"]
        label = spec.get("label") or f"{model_name}_{run_id}"

        # get prediction-output paths
        pred_dir = wandb_log / "predictions"

        # Getting metrics from W&B
        loss_df = wandb.Api().run(project + "/" + run_id).history()

        # Store model metadata
        run_data[run_id] = {
            "run_cfg": run_cfg,
            "run_id": run_id,
            "model_metadata": model_meta,
            "model_name": model_name,
            "label": label,
            "wandb_log_dir": wandb_log.as_posix(),
            "loss_df": loss_df,
            "merged.html": list(pred_dir.glob("**/merged.html")),
            "preds": list(pred_dir.glob("**/preds_epoch_*.parquet")),
            "peak.csv": list(pred_dir.glob("**/*.csv")),
            "phenix_logs": list(pred_dir.glob("**/phenix_out/refine*.log")),
            "train_metrics": list(
                (wandb_log / "files/train_metrics").glob("*.parquet")
            ),
        }
    return run_data


def _get_save_dir(save_dir, run_data):
    # number of runs to analyze
    n_models = len(run_data)

    # use user-specified save_dir if passed
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    elif n_models > 1:
        raise ValueError(
            "No save_dir specified. If analyzing more than 1 runs, you must specify a `save_dir`"
        )
    else:
        rd = next(iter(run_data.values()))
        if rd["wandb_log_dir"] is None:
            raise ValueError("W&B directory not found")
        save_dir = Path(rd["wandb_log_dir"]) / "plots"
        save_dir.mkdir(exist_ok=True)

    return save_dir


def main():
    set_mpl_fonts(10.95)
    args = parse_args()
    setup_logging(args.verbose)

    # Models to analyze (from --config or --run-dirs)
    models, save_dir_arg, seqids = _resolve_models(args)
    run_dirs = [m["run_dir"] for m in models]
    logger.info(f"Comparing {len(models)} model(s): {run_dirs}")

    # Getting data for each model
    run_data = _get_run_data(models)

    # Directory to save outputs
    save_dir = _get_save_dir(save_dir_arg, run_data)

    # ids of all runs in analysis
    run_ids = list(run_data.keys())

    # Getting paths to reference data
    reference_paths = _get_reference_paths(
        run_dirs=run_dirs,
    )

    #################
    # MERGING STATS #
    #################

    ###################
    # ANOMALOUS PEAKS #
    ###################

    # DIALS reference peak heights
    # Getting reference anomalous peaks.csv
    # Enforcing that all analyzed models use the same reference data

    # Get LazyFrame peak data
    lf = _get_peak_lf(
        run_data=run_data,
        reference_paths=reference_paths,
    )

    ref_peak_lf = None
    if reference_paths.get("anomalous_peaks") is not None:
        ref_peak_lf = pl.scan_csv(reference_paths["anomalous_peaks"])

    ##################
    # DIALS VS MODEL #
    ##################

    ref_merge_stats_df = _get_reference_merging_stats(reference_paths=reference_paths)

    ##############################
    # Anomalous Peak Processing
    # list of epochs
    epochs = (
        lf.select(pl.col("epoch").unique().sort())
        .collect()
        .get_column("epoch")
        .to_list()
    )


    # epoch_df
    epoch_df = pl.DataFrame({"epoch": epochs})

    # filter for anomalous residues
    anom_residues = ["IOD", "MET", "CYS"]

    # NOTE:
    # peak summary stats
    peak_lf = lf.filter(pl.col("seqid").is_in([204, 205, 206]))
    peak_lf = peak_lf.group_by(pl.col(["epoch", "run_id"])).agg(
        total_signal=pl.col("peakz").sum(),
        mean_signal=pl.col("peakz").mean(),
        max_signal=pl.col("peakz").max(),
        min_signal=pl.col("peakz").min(),
        median_signal=pl.col("peakz").median(),
    )
    # Plotting aggregated anomalous metrics
    metrics = [
        "total_signal",
        "mean_signal",
        "max_signal",
        "min_signal",
        "median_signal",
    ]
    for m in metrics:
        fig, _ = _plot_anomalous_metric(
            peak_lf=peak_lf,
            run_ids=run_ids,
            epoch_df=epoch_df,
            run_data=run_data,
            reference_data=ref_peak_lf,
            metric=m,
        )
        _savefig(fig, f"{save_dir}/anomalous_{m}.png")

    # NOTE:
    # plotting anomalous peak heights
    ref_peak_df = ref_peak_lf.collect()
    palette = _get_palette(run_ids)
    for s in seqids:
        # reference peak
        ref_peak = ref_peak_df.filter(pl.col("seqid") == s)["peakz"].item()

        df_seq = lf.filter(pl.col("seqid") == s).collect().sort(["epoch", "seqid"])

        series = []
        for rid in run_ids:
            df = df_seq.filter(pl.col("run_id") == rid)
            df = epoch_df.join(df, on="epoch", how="left").sort(["epoch", "seqid"])

            label = run_data[rid]["label"]
            series.append((label, df["epoch"], df["peakz"], palette[rid]))

        fig, _ = plot_metric_over_epoch(
            series,
            ref_value=ref_peak,
            ref_label="DIALS",
            x_label="epoch",
            y_label="peakz",
            title=f"Iodine {s} across models",
        )
        _savefig(fig, f"{save_dir}/anomalous_iod_{s}_model_peaks.png")

    # Plotting Fano binned by resolution
    run_data.values()

    all_train_metrics = [
        f for run in run_data.values() for f in run.get("train_metrics")
    ]
    lf_train_metrics = pl.scan_parquet(
        all_train_metrics, include_file_paths="filenames"
    )
    lf_train_metrics = add_run_epoch_cols(
        lf_train_metrics, epoch_pattern=r"/train_epoch_(\d+).parquet"
    )

    bin_labels, base_df = _get_bins(edges=INTENSITY_EDGES)
    base_df = base_df.collect()

    # TODO: Make this this into a function
    # Iterate over each run
    for r in run_ids:
        lf_r = lf_train_metrics.filter(pl.col("run_id") == r)
        # iterate over each epoch
        lf_ = (
            lf_r.with_columns(
                pl.col("qi_mean")
                .cut(INTENSITY_EDGES, labels=bin_labels)
                .alias("bin_labels")
            )
            .group_by(["epoch", "bin_labels"])
            .agg(
                fano=pl.col("qi_var").mean() / pl.col("qi_mean").mean(),
                n=pl.len(),
            )
            .sort(["epoch", "bin_labels"])
        ).collect()

        fig, _ = plot_binned_metric_by_epoch(
            lf_,
            base_df,
            epochs=epochs,
            x_key="bin_id",
            y_key="fano",
            x_label="intensity bin",
            y_label="fano",
            x_tick_labels=base_df["bin_labels"].to_list(),
        )
        _savefig(fig, f"{save_dir}/run_{r}_fano.png")
    # END TODO

    # %%
    # plotting the mean qi per model
    # todo: make this into a function
    dials_edges = DIALS_EDGES_9B7C
    bin_labels, base_df = _get_bins(edges=dials_edges)
    base_df = base_df.collect()
    bin_label = "resolution"

    # BINNING BY RESOLUTION
    # getting pl.DataFrame map
    df_map = _get_df_map(
        lf_train_metrics,
        run_ids,
        dials_edges,
        bin_labels,
        bin_key="d",
    )

    # Plotting mean qi.var

    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title=f"Mean qI.variance over {bin_label} bin",
        x_label="resolution bin",
        y_label="mean qi.variance",
        x_key="bin_id",
        y_key="mean_qi_var",
    )

    _savefig(fig, f"{save_dir}/all_runs.mean_qi_var.binnned_by_res.png")

    # Plotting mean qi.mean
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title=f"Mean qI.mean over {bin_label} bin",
        x_label=f"{bin_label} bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="mean_qi_mean",
    )

    _savefig(fig, f"{save_dir}/all_runs.mean_qi_var.binned_by_{bin_label}.png")

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="fano",
    )

    _savefig(fig, f"{save_dir}/all_runs.mean_fano.binned_by_resolution.png")

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="var_qi_mean",
    )

    _savefig(fig, f"{save_dir}/var_qi_mean_models_res_bin.png")

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="var_qi_var",
        y_scale=True,
    )

    _savefig(fig, f"{save_dir}/var_qi_var_models_res_bin.png")

    # per epoch
    # binned by ersolution
    df_map = _get_df_map(
        lf_train_metrics,
        run_ids,
        dials_edges,
        bin_labels,
        bin_key="d",
        group_key=["epoch", "bin_labels"],
    )

    # Per epoch
    # binned by resolution
    for r in run_ids:
        out_dir = save_dir / f"{r}"
        out_dir.mkdir(exist_ok=True)

        df_ = df_map[r]
        model_name = run_data[r]["model_metadata"]["qi_name"]
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="resolution bin",
            y_label="mean(qi.var)",
            x_key="bin_id",
            y_key="mean_qi_var",
            title=f"Mean qi.var over epoch\nmodel {model_name}",
            fname=f"{out_dir}/run_{r}_mean_qi_var_resbin_per_epoch.png",
            epochs=epochs,
            y_scale="log",
        )
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="resolution bin",
            y_label="mean(fano)",
            x_key="bin_id",
            y_key="fano",
            title=f"Mean fano over epoch for model {model_name}",
            fname=f"{out_dir}/run_{r}_mean_fano_resbin_per_epoch.png",
            epochs=epochs,
            y_scale="log",
        )
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="resolution_bin",
            y_label="Mean var(qi.mean)",
            x_key="bin_id",
            y_key="var_qi_mean",
            title=f"Mean var(qi.mean) over epoch for model {model_name}",
            fname=f"{out_dir}/run_{r}_var_qi_var_resbin_per_epoch.png",
            epochs=epochs,
            y_scale="log",
        )

    # BINNING BY INTENSITY
    # getting pl.DataFrame map
    intensity_edges = INTENSITY_EDGES
    bin_labels, base_df = _get_bins(edges=intensity_edges)
    base_df = base_df.collect()

    df_map = _get_df_map(
        lf_train_metrics,
        run_ids,
        intensity_edges,
        bin_labels,
    )

    # Plotting mean qi.var
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title="Mean qI.variance over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.variance",
        x_key="bin_id",
        y_key="mean_qi_var",
    )

    _savefig(fig, f"{save_dir}/mean_qi_var_models_intensity_bins.png")

    # Plotting mean qi.mean
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title="Mean qI.mean over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="mean_qi_mean",
    )

    _savefig(fig, f"{save_dir}/mean_qi_mean_models_intensity_bins.png")

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="fano",
    )

    _savefig(fig, f"{save_dir}/mean_fano_models_intensity_bins.png")

    # Plotting variance(qi.mean)

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="var_qi_mean",
    )

    _savefig(fig, f"{save_dir}/var_qi_mean_models_intensity_bins.png")

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        run_data=run_data,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="var_qi_var",
        y_scale=True,
    )

    _savefig(fig, f"{save_dir}/var_qi_var_models_intensity_bins.png")

    # TODO:
    # Per epoch metrics
    df_map = _get_df_map(
        lf_train_metrics,
        run_ids,
        intensity_edges,
        bin_labels,
        bin_key="qi_mean",
        group_key=["epoch", "bin_labels"],
    )

    for r in run_ids:
        out_dir = save_dir / f"{r}"
        out_dir.mkdir(exist_ok=True)

        df_ = df_map[r]
        model_name = run_data[r]["model_metadata"]["qi_name"]
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="bin",
            y_label="",
            x_key="bin_id",
            y_key="mean_qi_var",
            title=f"Mean qi.var over epoch for model {model_name}",
            fname=f"{out_dir}/run_{r}_mean_qi_var_per_epoch.png",
            epochs=epochs,
        )
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="bin",
            y_label="",
            x_key="bin_id",
            y_key="fano",
            title=f"Mean fano over epoch for model {model_name}",
            fname=f"{out_dir}/run_{r}_mean_fano_per_epoch.png",
            epochs=epochs,
        )
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="bin",
            y_label="",
            x_key="bin_id",
            y_key="var_qi_mean",
            title=f"Mean var(qi.var) over epoch for model {model_name}",
            fname=f"{out_dir}/run_{r}_var_qi_var_per_epoch.png",
            epochs=epochs,
        )

    train_loss_df, val_loss_df = _get_loss_dfs(run_data=run_data)

    loss_gap_df = _get_loss_gap_df(
        train_loss_df=train_loss_df,
        val_loss_df=val_loss_df,
    )

    # Plotting loss gap metrics
    _plot_loss_gap(
        loss_gap_df=loss_gap_df,
        metric="kl_gap",
        save_dir=save_dir,
    )
    _plot_loss_gap(
        loss_gap_df=loss_gap_df,
        metric="elbo_gap",
        save_dir=save_dir,
    )
    _plot_loss_gap(
        loss_gap_df=loss_gap_df,
        metric="nll_gap",
        save_dir=save_dir,
    )

    # Get long loss dataframe
    long_loss_df = _get_long_loss_df(
        val_loss_df=val_loss_df,
        train_loss_df=train_loss_df,
    )

    # Plot train/val loss
    _plot_train_val_loss(
        long_loss_df=long_loss_df,
        save_dir=save_dir,
    )

    # DataFrame of merging stats
    merging_stat_df = _get_merging_stat_df(run_data=run_data)

    # Merging stats out_dir
    # Plot merging stats for each run
    _plot_merging_stats(
        run_ids=run_ids,
        merging_stat_df=merging_stat_df,
        ref_merge_stats_df=ref_merge_stats_df,
        save_dir=save_dir,
    )

    # Getting all predictions
    pred_lf = _get_pred_lf(run_data=run_data)

    # Plotting merging statistics for each run
    _plot_run_merging_stats(
        run_ids=run_ids,
        pred_lf=pred_lf,
        save_dir=save_dir,
    )

    # Getting correlation df
    corr_df = _get_corr_df(pred_lf)

    # Plot correlations
    _plot_correlations(corr_df, save_dir)

    #######################
    # REFINEMENT R-VALUES #
    #######################

    # Get rval-df
    r_val_df = _get_rval_df(run_ids, run_data)

    # Get long df
    long_df = _get_rval_long_df(r_val_df)

    # Plot-rvalues
    _plot_r_values(
        long_df=long_df,
        linewidth=2,
        save_dir=save_dir,
        ref_path=reference_paths["phenix_refine_log"],
    )


if __name__ == "__main__":
    main()
