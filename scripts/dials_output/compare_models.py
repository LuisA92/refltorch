import argparse
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import wandb

from refltorch.cli.utils import setup_logging
from refltorch.io import load_config
from refltorch.plots import setup_mpl_config

logger = logging.getLogger(__name__)

# Setup mpl config for consistent plotting
setup_mpl_config()


def set_mpl_fonts(base_pt):
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": base_pt,
            "axes.labelsize": base_pt,
            "xtick.labelsize": base_pt - 1,
            "ytick.labelsize": base_pt - 1,
            "legend.fontsize": base_pt - 1,
            "axes.titlesize": base_pt + 1,
        }
    )


sns.set_theme(
    context="paper",  # correct for LaTeX
    style="ticks",
    font_scale=1.0,  # IMPORTANT: don’t rescale fonts
)

COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

CATEGORICAL_HEX_COLORS = {}
for k, v in enumerate(COLORS):
    key = k + 1
    CATEGORICAL_HEX_COLORS[key] = COLORS[:key]


def set_figsize(
    fraction=0.6,
    ratio=0.6,
    textwidth_pt=452.9679,
    paper="a4",
):
    # article, 1in margins
    if textwidth_pt is None:
        if paper.lower() == "a4":
            textwidth_pt = 452.97  # ~6.26 in
        elif paper.lower() == "letter":
            textwidth_pt = 468.0  #  ~6.48 in
        else:
            raise ValueError(f"Unknown paper type: {paper}")

    inches_per_pt = 1.0 / 72.27

    fig_width = textwidth_pt * inches_per_pt * fraction
    fig_height = fig_width * ratio

    return fig_width, fig_height


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to compare outputs from different models"
    )

    parser.add_argument(
        "--run-dirs",
        nargs="+",
        help="List of paths to the run-directories",
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


def plot_fano_over_epoch(
    fano_df,
    bin_label_key: str,
    edges: list,
):
    # Global bin order
    _, base_df = _get_bins(edges)
    base_df = base_df.collect()
    x_bins, x_labels = base_df["bin_id"].to_list(), base_df["bin_labels"].to_list()
    x_bins = np.array(x_bins)

    epochs = sorted(fano_df["epoch"].unique())
    n_epochs = len(epochs)

    cmap = sns.cubehelix_palette(start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True)
    colors = cmap(np.linspace(0, 1, n_epochs))

    fig, ax = plt.subplots(figsize=set_figsize())

    for e, c in zip(epochs, colors):
        data = fano_df.filter(pl.col("epoch") == e)

        ax.plot(
            x_bins,
            data["fano"],
            color=c,
            alpha=0.8,
        )

    # Set ticks ONCE using global bins
    ax.set_xticks(x_bins)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_yscale("log")
    ax.set_ylabel("Fano factor")
    ax.set_xlabel("Resolution bin (Intensity value)")
    ax.set_title("Mean Fano over resolution and epoch")
    ax.grid()

    return fig


def _get_bins(
    edges: list,
) -> tuple[list, pl.LazyFrame]:
    bin_labels = [f"{a} - {b}" for a, b in zip(edges[:-1], edges[1:])]
    bin_labels.insert(0, f"<{edges[0]}")
    bin_labels.append(f">{edges[-1]}")

    reversed_labels = list(reversed(bin_labels))

    base_df = pl.DataFrame(
        {
            "bin_labels": reversed_labels,
            "bin_id": list(range(len(reversed_labels))),
        },
        schema={
            "bin_labels": pl.Categorical,
            "bin_id": pl.Int32,
        },
    ).lazy()

    return bin_labels, base_df


# FIX:
# Come up with a method to generate intensity and resolution edges
# In the case of dials, we can perhaps just get them from the reference data
# Let's try to do so

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


def _get_reference_metadata(run_config: dict) -> dict:
    cfg = load_config(run_config["config"])
    out = {
        "qbg_name": cfg["surrogates"]["qbg"]["name"],
        "qi_name": cfg["surrogates"]["qi"]["name"],
        "max_epochs": cfg["trainer"]["max_epochs"],
        "integrator_name": cfg["integrator"]["name"],
        "pbg": cfg["loss"]["args"]["pbg_cfg"],
        "pi": cfg["loss"]["args"]["pi_cfg"],
        "pprf": cfg["loss"]["args"]["pprf_cfg"],
    }
    return out


def _get_reference_data(ref_data_path: Path) -> tuple[Path, ...]:
    ref_peaks = list(ref_data_path.glob("reference_data/*peaks.csv"))[0]
    ref_merged_html = list(ref_data_path.glob("reference_data/*merged*.html"))[0]
    return ref_peaks, ref_merged_html


def _get_reference_paths(run_dirs) -> dict[str, Path]:
    # since all models use the same reference, we can
    # just grab the first config file
    path = Path(run_dirs[0])
    run_cfg = list(path.glob("run_metadata.yaml"))[0]
    run_cfg = load_config(run_cfg)
    cfg = load_config(run_cfg["config"])
    output_cfg = cfg["output"]
    return {k: Path(v) for k, v in output_cfg.items()}


def _get_reference_data_path(run_config: dict) -> Path:
    cfg = load_config(run_config["config"])
    return Path(cfg["global_vars"]["data_dir"]).parent


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
    # plotting the mean qi_var per model

    fig, ax = plt.subplots(figsize=set_figsize(ratio=0.6, textwidth_pt=452.9679))
    labels = base_df["bin_labels"].to_list()
    ticks = base_df["bin_id"].to_list()

    palette = _get_palette(run_ids)

    for r in run_ids:
        df_ = df_map[r]

        df = base_df.join(df_, on="bin_labels", how="left").sort("bin_id")

        # plot
        model_name = run_data[r]["model_metadata"]["qi_name"]
        ax.plot(
            df[x_key],
            df[y_key],
            label=f"{model_name}_{r}",
            color=palette[r],
        )

    ax.set_xlabel(x_label)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(y_label)

    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % 2 != 0:
            label.set_visible(False)

    ax.set_title(title)
    if y_scale is not None:
        ax.set_yscale("log")
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    return fig, ax


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
    # plotting the mean qi_var per model

    n_epochs = len(epochs)
    cmap = sns.cubehelix_palette(start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True)
    colors = cmap(np.linspace(0, 1, n_epochs))

    fig, ax = plt.subplots(figsize=set_figsize())

    for c, e in zip(colors, epochs):
        df_ = df.filter(pl.col("epoch") == e)
        df_ = base_df.join(df_, on="bin_labels", how="left").sort("bin_id")

        ax.plot(
            df_[x_key],
            df_[y_key],
            c=c,
        )
    ax.grid()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if y_scale is not None:
        ax.set_yscale(y_scale)
    fig.savefig(
        fname,
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)


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
    if reference_data is not None:
        ref_lf = reference_data.select(
            total_signal=pl.col("peakz").sum(),
            mean_signal=pl.col("peakz").mean(),
            max_signal=pl.col("peakz").max(),
            min_signal=pl.col("peakz").min(),
            median_signal=pl.col("peakz").median(),
        ).collect()

    fig, ax = plt.subplots(figsize=set_figsize())
    palette = _get_palette(run_ids)
    for r in run_ids:
        label = run_data[r]["model_metadata"]["qi_name"]
        lf = peak_lf.filter(pl.col("run_id") == r)
        df = lf.collect()
        df = epoch_df.join(df, on="epoch", how="left").sort("epoch")
        ax.plot(df["epoch"], df[metric], label=label, color=palette[r])
    if reference_data is not None:
        ax.axhline(ref_lf[metric].to_list(), c="red", label="DIALS")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"{metric} over epochs")
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    return fig, ax


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


def get_dials_merging_stats(
    tbl: pd.DataFrame,
    epoch: int | None = None,
    run_id: str | None = None,
    model_name: str | None = None,
    keys: tuple[str, ...] = (
        "resolution",
        "n_refls",
        "n_unique",
        "multiplicity",
        "completeness",
        "mean_i",
        "meani_sigi",
        "rmerge",
        "rmeas",
        "rpim",
        "ranom",
        "cchalf",
        "ccanom",
    ),
) -> pl.LazyFrame:
    if len(keys) != len(tbl.columns):
        raise ValueError("keys must match number of columns")

    data = {}

    for key, col in zip(keys, tbl.columns):
        values = tbl[col].tolist()

        if key == "cchalf":
            values = [float(str(v).strip("*")) for v in values]

        if key == "ccanom":
            values = [float(str(v).strip("*")) for v in values]

        data[key] = values

    # df = pl.LazyFrame(data).with_columns(epoch=epoch, run_id=run_id)
    lf = pl.LazyFrame(data)

    if epoch is not None:
        lf = lf.with_columns(epoch=pl.lit(epoch))
    if run_id is not None:
        lf = lf.with_columns(run_id=pl.lit(run_id))
    if model_name is not None:
        lf = lf.with_columns(model_name=pl.lit(model_name))

    return lf


def _get_r_vals(phenix_log: Path):
    pattern1 = re.compile(r"Start R-work")
    pattern2 = re.compile(r"Final R-work")
    matches = {}

    with phenix_log.open("r") as file:
        lines = file.readlines()
        matched_lines_start = [line.strip() for line in lines if pattern1.search(line)]
        matched_lines_final = [line.strip() for line in lines if pattern2.search(line)]

        matches["r_work_start"] = float(
            re.findall(r"\d\.\d+", matched_lines_start[0])[0]
        )
        matches["r_free_start"] = float(
            re.findall(r"\d\.\d+", matched_lines_start[0])[1]
        )

        matches["r_work_final"] = float(
            re.findall(r"\d\.\d+", matched_lines_final[0])[0]
        )
        matches["r_free_final"] = float(
            re.findall(r"\d\.\d+", matched_lines_final[0])[1]
        )
    return matches


def _get_anomalous_peak_paths(reference_paths, run_data):
    ref_peak_lf = None
    check_phenix_logs = reference_paths.get(
        "anomalous_peaks", ValueError("key not in reference_paths: anomalous_peaks")
    )
    if check_phenix_logs is not None:
        ref_peak_lf = pl.scan_csv(reference_paths["anomalous_peaks"])

    # MODEL anomalous peak heights
    all_peaks = [csv for run in run_data.values() for csv in run.get("peak.csv", [])]
    return all_peaks


def _get_reference_merging_stats(reference_paths):
    # Getting reference DIALS merged.html
    ref_merge_stats_df = None
    check_merged_html = reference_paths.get(
        "dials_merge_html", ValueError("key not in reference_paths: dials_merge_html")
    )
    if check_merged_html is not None:
        # Getting reference merging statistics
        ref_tbl1, ref_tbl2 = pd.read_html(reference_paths["dials_merge_html"])
        ref_merge_stats_df = get_dials_merging_stats(ref_tbl2).collect()

    return ref_merge_stats_df


def _plot_anom_peak_stats(
    lf,
):
    epochs = (
        lf.select(pl.col("epoch").unique().sort())
        .collect()
        .get_column("epoch")
        .to_list()
    )

    # TODO: Add seqids to args list of peak sequence ids to analyze
    seqids = [204, 205, 206]

    # epoch_df
    epoch_df = pl.DataFrame({"epoch": epochs})

    # filter for anomalous residues
    anom_residues = ["IOD", "MET", "CYS"]

    # NOTE:
    # peak summary stats
    peak_lf = lf.filter(pl.col("residue").is_in(anom_residues))
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
        fig, ax = _plot_anomalous_metric(
            peak_lf=peak_lf,
            run_ids=run_ids,
            epoch_df=epoch_df,
            run_data=run_data,
            reference_data=ref_peak_lf,
            metric=m,
        )

        fig.savefig(
            f"{save_dir}/anomalous_{m}.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

    pass


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

    # Extracting epoch from filename and appending as column
    lf = lf.with_columns(
        [
            pl.col("filenames").str.extract(r"/run-[^/]+-([^/]+)/", 1).alias("run_id"),
            pl.col("filenames")
            .str.extract(r"/epoch_(\d+)/", 1)
            .cast(pl.Int32)  # optional
            .alias("epoch"),
        ]
    )
    return lf


def _get_merging_stat_df(
    run_data: dict,
) -> pl.DataFrame:
    # NOTE:
    # Plot merging statistics from DIALS html files
    pattern = re.compile(r"epoch_(\d+)")
    merging_stat_dfs = []
    for run in run_data.values():
        htmls = run["merged.html"]
        model_name = run["model_name"]
        run_id = run["run_id"]
        for h in htmls:
            search = re.search(pattern, h.as_posix())
            if search:
                hit = search.group()
                epoch = int(hit.split("_")[-1])
            else:
                raise ValueError(
                    "Incorrect filename formatting; Check your directory structure"
                )
            tbl1, tbl2 = pd.read_html(h)
            merging_stat_dfs.append(
                get_dials_merging_stats(
                    tbl=tbl2,
                    epoch=epoch,
                    run_id=run_id,
                    model_name=model_name,
                )
            )

    merging_stat_df = pl.concat(merging_stat_dfs).collect()
    return merging_stat_df


def _plot_merging_stats(
    run_ids,
    merging_stat_df,
    ref_merge_stats_df,
    save_dir: Path,
):
    im_frac = 0.6
    out_dir = save_dir
    out_dir.mkdir(exist_ok=True)

    for run in run_ids:
        df_ = merging_stat_df.filter(pl.col("run_id") == run)

        model_name = df_["model_name"].unique().item()
        run_id = df_["run_id"].unique().item()

        # colorbar
        cmap = sns.cubehelix_palette(
            start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True
        )
        norm = mpl.colors.Normalize(
            vmin=df_["epoch"].min(),
            vmax=df_["epoch"].max(),
        )
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # cchalf plot
        # setting up figure
        fig, ax = plt.subplots(figsize=set_figsize(fraction=im_frac))

        sns.lineplot(
            data=df_,
            x="resolution",
            y="cchalf",
            hue="epoch",
            palette=cmap,
            hue_norm=norm,
            legend=False,
            ax=ax,
        )
        ax.plot(ref_merge_stats_df["cchalf"], label="DIALS", color="red")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel("Resolution bin")
        ax.set_ylabel("CChalf")
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        labels = ax.get_xticklabels()
        for i, label in enumerate(labels):
            if i % 2 != 0:
                label.set_visible(False)

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Epoch", rotation=90)
        ax.legend()
        ax.grid(alpha=0.5)
        ax.set_title(f"{model_name}")

        fig.savefig(
            f"{out_dir}/merging_stats.cchalf.run_{run}.model_{model_name}.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)

        # rpim plot
        fig, ax = plt.subplots(figsize=set_figsize(fraction=im_frac))

        sns.lineplot(
            data=df_,
            x="resolution",
            y="rpim",
            hue="epoch",
            palette=cmap,
            hue_norm=norm,
            legend=False,
            ax=ax,
        )
        ax.plot(ref_merge_stats_df["rpim"], label="DIALS", color="red")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel("Resolution bin")
        ax.set_ylabel("Rpim")
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(alpha=0.5)
        ax.set_title(f"{model_name}")

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Epoch", rotation=90)
        ax.legend()

        labels = ax.get_xticklabels()
        for i, label in enumerate(labels):
            if i % 2 != 0:
                label.set_visible(False)

        fig.savefig(
            f"{out_dir}/merging_stats.rpim.run_{run}.model_{model_name}.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

        # isigi plot
        fig, ax = plt.subplots(figsize=set_figsize(fraction=im_frac))

        sns.lineplot(
            data=df_,
            x="resolution",
            y="meani_sigi",
            hue="epoch",
            palette=cmap,
            hue_norm=norm,
            legend=False,
            ax=ax,
        )
        ax.plot(ref_merge_stats_df["meani_sigi"], label="DIALS", color="red")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel("Resolution bin")
        ax.set_ylabel("I/Sig(I)")
        ax.set_title(f"{model_name}")
        ax.grid(alpha=0.5)
        ax.set_yticks([0.0, 25, 50, 75, 100, 125])

        labels = ax.get_xticklabels()
        for i, label in enumerate(labels):
            if i % 2 != 0:
                label.set_visible(False)

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Epoch", rotation=90)
        ax.legend()

        fig.savefig(
            f"{out_dir}/merging_stats.isigi.run_{run}.model_{model_name}.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)

        # ccanom plot
        fig, ax = plt.subplots(figsize=set_figsize(fraction=im_frac))

        sns.lineplot(
            data=df_,
            x="resolution",
            y="ccanom",
            hue="epoch",
            palette=cmap,
            hue_norm=norm,
            legend=False,
            ax=ax,
        )
        ax.plot(ref_merge_stats_df["ccanom"], label="DIALS", color="red")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel("Resolution bin")
        ax.set_ylabel("CCanom")
        ax.set_title(f"{model_name}")
        ax.set_yticks([-0.5, -0.25, 0.0, 0.25])

        labels = ax.get_xticklabels()
        for i, label in enumerate(labels):
            if i % 2 != 0:
                label.set_visible(False)

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Epoch", rotation=90)
        ax.legend()
        ax.grid(alpha=0.5)
        fig.savefig(
            f"{out_dir}/merging_stats.ccanom.run_{run}.model_{model_name}.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)


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
    for (run,), df_run in corr_df.group_by("run_id"):
        out_dir = save_dir / f"{run}"
        out_dir.mkdir(exist_ok=True)

        # Plotting intensity correlation
        cmap = sns.cubehelix_palette(
            start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True
        )
        norm = mpl.colors.Normalize(
            vmin=df_run["epoch"].min(),
            vmax=df_run["epoch"].max(),
        )
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig, ax = plt.subplots(figsize=set_figsize())
        sns.lineplot(
            data=df_run,
            x="d_bins",
            y="corr_I",
            hue="epoch",
            hue_norm=norm,
            palette=cmap,
            legend=False,
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=50)
        ax.grid()
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Epoch", rotation=90)

        labels = ax.get_xticklabels()
        for i, label in enumerate(labels):
            if i % 2 != 0:
                label.set_visible(False)

        fig.savefig(
            f"{out_dir}/run_{run}_dials_corr_I.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

        # Plotting background correlation

        fig, ax = plt.subplots(figsize=set_figsize())
        sns.lineplot(
            data=df_run,
            x="d_bins",
            y="corr_bg",
            hue="epoch",
            hue_norm=norm,
            palette=cmap,
            legend=False,
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=50)
        ax.grid()
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Epoch", rotation=90)

        labels = ax.get_xticklabels()
        for i, label in enumerate(labels):
            if i % 2 != 0:
                label.set_visible(False)

        fig.savefig(
            f"{out_dir}/run_{run}_dials_corr_bg.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

        # Plotting var(I) correlation
        fig, ax = plt.subplots(figsize=set_figsize())
        sns.lineplot(
            data=df_run,
            x="d_bins",
            y="corr_var_I",
            hue="epoch",
            hue_norm=norm,
            palette=cmap,
            legend=False,
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=50)
        ax.grid()
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Epoch", rotation=90)

        labels = ax.get_xticklabels()
        for i, label in enumerate(labels):
            if i % 2 != 0:
                label.set_visible(False)

        fig.savefig(
            f"{out_dir}/run_{run}_dials_corr_var_I.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)


def _get_rval_df(
    run_ids,
    run_data,
) -> pl.DataFrame:
    dfs = []
    pattern = re.compile(r"epoch_(\d+)")
    for run in run_ids:
        for f in run_data[run]["phenix_logs"]:
            rvals = _get_r_vals(f)
            s = re.search(pattern, f.as_posix())
            if s:
                epoch = int(s.groups()[0])
            df = pl.DataFrame(
                {
                    "run_id": run,
                    "epoch": epoch,
                    "fname": f.as_posix(),
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
        ],
    )

    # adding identifiers for styling
    long_df = long_df.with_columns(
        [
            pl.when(pl.col("variable").str.contains("work"))
            .then(pl.lit("r_work"))
            .otherwise(pl.lit("r_free"))
            .alias("metric"),
            pl.when(pl.col("variable").str.contains("final"))
            .then(pl.lit("final"))
            .otherwise(pl.lit("start"))
            .alias("stage"),
        ]
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
    pred_lf = pred_lf.with_columns(
        [
            pl.col("filenames").str.extract(r"/run-[^/]+-([^/]+)/", 1).alias("run_id"),
            pl.col("filenames")
            .str.extract(r"/epoch_(\d+)/", 1)
            .cast(pl.Int32)  # optional
            .alias("epoch"),
        ]
    )
    return pred_lf


def _plot_run_merging_stats(run_ids, pred_lf, save_dir: Path):
    # Out directory

    # Plot hyper parameters
    n_samples = 10_000
    pad = 2.0
    alpha = 0.4
    alpha2 = 0.5

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
            .group_by("epoch", maintain_order=True)
            .head(n_samples)
            .collect()
        )

        for (epoch,), df_epoch in df.group_by("epoch"):
            # Getting max values

            # Plotting dials I vs model I
            x_max = (
                df_epoch.select(pl.max("qi_mean", "intensity.prf.value"))
                .max_horizontal()
                .item()
            ) * pad
            y_min = df_epoch.select(pl.min("intensity.prf.value")).item() * pad

            fig, ax = plt.subplots(figsize=set_figsize())

            sns.scatterplot(
                data=df_epoch,
                x="qi_mean",
                y="intensity.prf.value",
                alpha=alpha,
                ax=ax,
                c="black",
            )
            ax.set_title(f"Run {run}\nEpoch {epoch}")
            ax.set_xlabel("Model Intensity")
            ax.plot(
                [0, x_max],
                [0, x_max],
                c="red",
                alpha=alpha2,
            )
            ax.set_xlim(xmin=0.0, xmax=x_max)
            ax.set_ylim(ymin=y_min, ymax=x_max)
            ax.set_ylabel("intensity.prf.value")
            ax.set_yscale("symlog")
            ax.set_xscale("symlog")
            ax.grid()

            fig.savefig(
                f"{out_dir}/run_{run}_vs_dials_I_{epoch}.png",
                transparent=True,
                dpi=300,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.02,
            )

            plt.close(fig)

            # Plotting dials bg vs model bg
            bg_model_key = "qbg_mean"
            bg_dials_key = "background.mean"

            x_max = (
                df_epoch.select(pl.max(bg_model_key, bg_dials_key))
                .max_horizontal()
                .item()
            )

            y_min = df_epoch.select(pl.min(bg_dials_key)).item()

            fig, ax = plt.subplots(figsize=set_figsize())
            sns.scatterplot(
                data=df_epoch,
                x=bg_model_key,
                y=bg_dials_key,
                alpha=alpha,
                ax=ax,
                c="black",
            )
            ax.set_title(f"Run {run}\nEpoch {epoch}")
            ax.set_xlabel("Model bg")
            ax.plot(
                [0, x_max],
                [0, x_max],
                c="red",
                alpha=alpha2,
            )
            ax.set_xlim(xmin=0.0, xmax=x_max)
            ax.set_ylim(ymin=y_min, ymax=x_max)
            ax.set_ylabel(bg_dials_key)
            ax.grid()

            fig.savefig(
                f"{out_dir}/run_{run}_vs_dials_bg_{epoch}.png",
                transparent=True,
                dpi=300,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.02,
            )

            plt.close(fig)

            # Plotting dials var(I) vs model var(I)

            i_var_model_key = "qi_var"
            i_var_dials_key = "intensity.prf.variance"

            x_max = (
                df_epoch.select(pl.max(i_var_model_key, i_var_dials_key))
                .max_horizontal()
                .item()
            ) * pad
            y_min = df_epoch.select(pl.min(i_var_dials_key)).item() * pad

            fig, ax = plt.subplots(figsize=set_figsize())
            sns.scatterplot(
                data=df_epoch,
                x="qi_var",
                y="intensity.prf.variance",
                alpha=alpha,
                ax=ax,
                c="black",
            )
            ax.plot(
                [0, x_max],
                [0, x_max],
                c="red",
                alpha=alpha2,
            )

            ax.set_title(f"Run {run}\nEpoch {epoch}")
            ax.set_xlabel("Model var(I)")
            ax.set_ylabel("intensity.prf.variance")

            # ax.set_xlim(xmin=0.0, xmax=x_max)
            # ax.set_ylim(ymin=y_min, ymax=x_max)

            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.grid()

            fig.savefig(
                f"{out_dir}/run_{run}_vs_dials_var_epoch_{epoch}.png",
                transparent=True,
                dpi=300,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.02,
            )

            plt.close(fig)


# FIX: Modify to plot the start/final rvalues as two separate plots
def _plot_r_values(
    long_df: pl.DataFrame,
    save_dir: str | Path,
    linewidth: int = 2,
    ref_path: Path | None = None,
):
    # plot color

    c_pallete = sns.color_palette("Dark2")
    palette = {
        "r_work": "black",
        "r_free": "blue",
    }

    # plot line style
    dashes = {
        "start": (2, 2),
        "final": "",
    }
    if ref_path is not None:
        ref_vals = _get_r_vals(ref_path)

    for (run,), run_df in long_df.group_by(pl.col("run_id")):
        # out dir
        out_dir = save_dir
        out_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=set_figsize())

        sns.lineplot(
            data=run_df,
            x="epoch",
            y="value",
            hue="metric",
            style="stage",
            palette=palette,
            dashes=dashes,
            linewidth=linewidth,
            ax=ax,
        )

        if ref_path is not None:
            assert isinstance(ref_vals, dict)
            ax.axhline(
                ref_vals["r_work_final"],
                color="red",
                label="DIALS rwork final",
            )
            ax.axhline(
                ref_vals["r_free_final"],
                color="red",
                linestyle="--",
                label="DIALS rfree final",
            )

        ax.set_xlabel("epoch")
        ax.set_ylabel("r value")
        ax.set_title(f"R-values vs epoch\nrun_id: {run}")
        ax.grid()
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        fig.savefig(
            f"{out_dir}/run_{run}_test_unpivot.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)


def _get_loss_dfs(run_data) -> tuple[pl.DataFrame, pl.DataFrame]:
    val_loss_dfs = []
    train_loss_dfs = []

    for v in run_data.values():
        val_loss_dfs.append(
            _get_val_loss(v["loss_df"]).with_columns(
                run_id=pl.lit(v["run_id"]),
                name=pl.lit(v["model_name"]),
                Stage=pl.lit("val"),
            )
        )
        train_loss_dfs.append(
            _get_train_loss(v["loss_df"]).with_columns(
                run_id=pl.lit(v["run_id"]),
                name=pl.lit(v["model_name"]),
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
    # TODO: Modify so that `model_name` is in the legend
    fig, ax = plt.subplots(figsize=set_figsize())
    sns.lineplot(
        data=loss_gap_df,
        x="epoch",
        y=metric,
        hue="run_id",
        palette="Dark2",
    )
    ax.grid()
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_title(f"Train/Val {metric} gap")

    plt.savefig(
        f"{save_dir}/train_val_{metric}.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )


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
        index=["run_id", "epoch", "Stage"],
    )
    train_long_ = train_loss_df.unpivot(
        ["train nll", "train kl", "train elbo"],
        index=["run_id", "epoch", "Stage"],
    )
    long_loss_df = val_long_.vstack(train_long_)
    return long_loss_df


def _get_palette(run_ids) -> dict:
    if len(run_ids) in CATEGORICAL_HEX_COLORS:
        hex_colors = CATEGORICAL_HEX_COLORS[len(run_ids)]
        return dict(zip(run_ids, hex_colors))
    else:
        c_pallete = sns.color_palette("Dark2")
        return {run_id: c for run_id, c in zip(run_ids, c_pallete)}


def _plot_train_val_loss(
    long_loss_df: pl.DataFrame,
    save_dir: str | Path,
):
    run_ids = long_loss_df["run_id"].unique()
    palette = _get_palette(run_ids=run_ids)

    # plot line style
    dashes = {
        "val": ":",
        "train": "",
    }

    # Metrics to plot together
    metrics = {
        "nll": ("train nll", "val nll"),
        "kl": ("train kl", "val kl"),
        "elbo": ("train elbo", "val elbo"),
    }

    for k, v in metrics.items():
        plot_df = long_loss_df.filter(pl.col("variable").is_in(v))
        ymin = plot_df["value"].min()

        fig, ax = plt.subplots(figsize=set_figsize())
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="value",
            hue="run_id",  # color by run_id
            style="Stage",  # Stage is train or val
            palette=palette,
            dashes=dashes,
            linewidth=2,
            ax=ax,
        )

        ax.set_title(f"Train/Val {k.upper()} vs epoch ")
        ax.grid()
        ax.set_ylabel(f"{k.upper()}")

        if k in ["nll", "elbo"]:
            ax.set_ylim(ymax=2000)
        ax.set_ylim(ymin=ymin)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        fig.savefig(
            f"{save_dir}/train_val_{k}.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)


def _get_run_data(run_dirs):
    # Dictionary for model metadata
    run_data = {}
    wandb_log = None

    # Getting data paths for each run
    for rd in run_dirs:
        path = Path(rd)
        run_cfg = list(path.glob("run_metadata.yaml"))[0]
        run_cfg = load_config(run_cfg)

        # wandb metadata
        project = run_cfg["wandb"]["project"]
        run_id = run_cfg["wandb"]["run_id"]

        model_meta = _get_reference_metadata(run_cfg)
        wandb_log = Path(run_cfg["wandb"]["log_dir"]).parent

        # get peak.csv paths
        pred_dir = Path(wandb_log / "predictions/")

        # Getting metrics from W&B
        loss_df = wandb.Api().run(project + "/" + run_id).history()

        # Store model metadata
        run_data[run_id] = {
            "run_cfg": run_cfg,
            "run_id": run_id,
            "model_metadata": model_meta,
            "model_name": model_meta["qi_name"],
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


def _get_save_dir(args, run_data):
    # number of runs to analyze
    n_models = len(run_data)

    # use user-specified save_dir if passed
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
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

    # Runs to analyze
    run_dirs = args.run_dirs
    logger.info(f"Run directories: {run_dirs}")

    # Getting data for each run
    run_data = _get_run_data(
        run_dirs=run_dirs,
    )

    # Directory to save outputs
    save_dir = _get_save_dir(
        args=args,
        run_data=run_data,
    )

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
    check_phenix_logs = reference_paths.get(
        "anomalous_peaks", ValueError("key not in reference_paths: anomalous_peaks")
    )
    if check_phenix_logs is not None:
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

    # TODO: Add seqids to args list of peak sequence ids to analyze
    seqids = [204, 205, 206]

    # epoch_df
    epoch_df = pl.DataFrame({"epoch": epochs})

    # filter for anomalous residues
    anom_residues = ["IOD", "MET", "CYS"]

    # NOTE:
    # peak summary stats
    peak_lf = lf.filter(pl.col("residue").is_in(anom_residues))
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
        fig, ax = _plot_anomalous_metric(
            peak_lf=peak_lf,
            run_ids=run_ids,
            epoch_df=epoch_df,
            run_data=run_data,
            reference_data=ref_peak_lf,
            metric=m,
        )

        fig.savefig(
            f"{save_dir}/anomalous_{m}.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

    # NOTE:
    # plotting anomalous peak heights
    ref_peak_df = ref_peak_lf.collect()
    palette = _get_palette(run_ids)
    for s in seqids:
        # reference peak
        ref_peak = ref_peak_df.filter(pl.col("seqid") == s)["peakz"].item()

        fig, ax = plt.subplots(figsize=set_figsize())
        df_seq = lf.filter(pl.col("seqid") == s).collect().sort(["epoch", "seqid"])

        for rid in run_ids:
            df = df_seq.filter(pl.col("run_id") == rid)
            df = epoch_df.join(df, on="epoch", how="left").sort(["epoch", "seqid"])

            # use surrogate prior name as label
            label = f"{run_data[rid]['model_metadata']['qi_name']}_{rid}"

            sns.lineplot(
                x=df["epoch"],
                y=df["peakz"],
                label=label,
                color=palette[rid],
                ax=ax,
            )

        ax.axhline(ref_peak, c="red", label="DIALS")
        ax.set_xlabel("epoch")
        ax.set_ylabel("peakz")
        ax.set_title(f"Iodine {s} across models")
        ax.legend()
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.grid()

        fig.savefig(
            f"{save_dir}/anomalous_iod_{s}_model_peaks.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)

    # Plotting Fano binned by resolution
    run_data.values()

    all_train_metrics = [
        f for run in run_data.values() for f in run.get("train_metrics")
    ]
    lf_train_metrics = pl.scan_parquet(
        all_train_metrics, include_file_paths="filenames"
    )
    lf_train_metrics = lf_train_metrics.with_columns(
        [
            pl.col("filenames").str.extract(r"/run-[^/]+-([^/]+)/", 1).alias("run_id"),
            pl.col("filenames")
            .str.extract(r"/train_epoch_(\d+).parquet", 1)
            .cast(pl.Int32)  # optional
            .alias("epoch"),
        ]
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

        cmap = sns.cubehelix_palette(
            start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True
        )
        cmap_list = cmap(np.linspace(0.0, 1, len(epochs), retstep=True)[0])

        fig, ax = plt.subplots(figsize=set_figsize())
        for c, ((e,), lf_epoch) in zip(
            cmap_list,
            lf_.group_by("epoch", maintain_order=True),
        ):
            joined = base_df.join(lf_epoch, on="bin_labels", how="left")
            ax.plot(joined["bin_id"], joined["fano"], c=c)
        ax.set_xlabel("intensity bin")
        ax.set_xticklabels(joined["bin_labels"], rotation=45, ha="right")
        ax.set_ylabel("fano")
        ax.grid()

        fig.savefig(
            f"test_out/run_{r}_fano.png",
            transparent=True,
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        plt.close(fig)
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

    fig.savefig(
        f"{save_dir}/all_runs.mean_qi_var.binnned_by_res.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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

    fig.savefig(
        f"{save_dir}/all_runs.mean_qi_var.binned_by_{bin_label}.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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

    fig.savefig(
        f"{save_dir}/all_runs.mean_fano.binned_by_resolution.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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

    fig.savefig(
        f"{save_dir}/var_qi_mean_models_res_bin.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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

    fig.savefig(
        f"{save_dir}/var_qi_var_models_res_bin.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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

    fig.savefig(
        f"{save_dir}/mean_qi_var_models_intensity_bins.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

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

    fig.savefig(
        f"{save_dir}/mean_qi_mean_models_intensity_bins.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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

    fig.savefig(
        f"{save_dir}/mean_fano_models_intensity_bins.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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

    fig.savefig(
        f"{save_dir}/var_qi_mean_models_intensity_bins.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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

    fig.savefig(
        f"{save_dir}/var_qi_mean_models_intensity_bins.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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

    fig.savefig(
        f"{save_dir}/var_qi_var_models_intensity_bins.png",
        transparent=True,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

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
    # NOTE: Turn off for faster run time
    # _plot_run_merging_stats(
    #     run_ids=run_ids,
    #     pred_lf=pred_lf,
    #     save_dir=save_dir,
    # )

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

    import numpy as np
