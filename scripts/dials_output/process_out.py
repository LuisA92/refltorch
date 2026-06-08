import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from matplotlib.figure import Figure
from out_utils import (
    DIALS_EDGES_9B7C,
    INTENSITY_EDGES,
    read_peaks_csv,
)

from refltorch.io import epoch_from_path, glob_predictions, open_run
from refltorch.plots._shared import get_bins as _get_bins
from refltorch.plots._shared import save_figure
from refltorch.plots.cmaps import get_cube_helix_map
from refltorch.tools import (
    get_reference_data as _get_reference_data,
)
from refltorch.tools import (
    get_reference_data_path as _get_reference_data_path,
)
from refltorch.tools import (
    get_reference_metadata as _get_reference_metadata,
)
from refltorch.tools import (
    parse_dials_merging_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to process and upload integration output results to W&B"
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to run-directory",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Path to save directory",
    )

    return parser.parse_args()


def plot_anomalous_peaks(
    peaks: list[Path],
    ref_peak_path: Path | None = None,
):
    peak_columns = ["seqid", "residue", "peakz"]

    ref_df = None
    if ref_peak_path is not None:
        ref_df = pl.read_csv(
            ref_peak_path,
            columns=peak_columns,
        )

    peak_dfs = {}
    seqids = {}
    for n, p in enumerate(peaks):
        peak_dfs[n] = read_peaks_csv(p)

        for s, r in peak_dfs[n][["seqid", "residue"]].iter_rows():
            seqids[s] = r

    df = pl.DataFrame(
        {"seqid": list(seqids.keys()), "residue": list(seqids.values())}
    ).sort("seqid")

    if ref_df is not None:
        df = df.join(ref_df, on=["seqid", "residue"], how="left")

    for k, v in peak_dfs.items():
        df = df.join(v, on=["seqid", "residue"], how="left", suffix=f"_{k}")

    header = (df["seqid"].cast(pl.String) + "_" + df["residue"]).to_list()
    values = [x.to_list() for x in df[:, 2:].transpose().iter_columns()]

    # Format values
    for sub_list in values:
        for i, el in enumerate(sub_list):
            if el is None:
                sub_list[i] = "-"
            elif isinstance(el, float):
                sub_list[i] = f"{el:.2f}"

    # Insert epoch column
    header.insert(0, "epoch")
    epoch_list = list(range(1, 20, 2))
    if ref_df is not None:
        epoch_list.insert(0, "dials")
    values.insert(0, [str(x) for x in epoch_list])

    n_rows = len(values[0])
    n_cols = len(header)

    # color scheme
    row_colors = ["#f2f2f2" if i % 2 == 0 else "#ffffff" for i in range(n_rows)]
    cell_fill_colors = [row_colors] * n_cols

    # Wider first column, uniform others
    column_widths = [80] + [110] * (n_cols - 1)

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=column_widths,
                header=dict(
                    values=header,
                    fill_color="#d9d9d9",
                    align="center",
                    font=dict(size=13, color="black"),
                    height=35,
                ),
                cells=dict(
                    values=values,
                    fill_color=cell_fill_colors,
                    align="center",
                    font=dict(size=12),
                    height=28,
                ),
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
    )

    return fig


# TODO: Add colorbars to denote epoch
def plot_merging_stat_subplots(
    dfs: dict[str, pl.DataFrame],
    ref_df: pl.DataFrame | None = None,
) -> Figure:
    # Plot merging stat subplots
    n_logged_epochs = len([int(k) for k in list(dfs.keys())])
    cmap_list = get_cube_helix_map(n_logged_epochs)

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    axs = ax.ravel()
    for (k, v), c in zip(dfs.items(), cmap_list):
        axs[0].plot(v["cchalf"], color=c)
        axs[0].set_ylabel("CChalf")
        axs[0].set_xticks(range(v.height), v["resolution"], rotation=55)
        axs[1].plot(v["rpim"], color=c)
        axs[1].set_ylabel("Rpim")
        axs[1].set_xticks(range(v.height), v["resolution"], rotation=55)
        axs[2].plot(v["ccanom"], color=c)
        axs[2].set_ylabel("CCanom")
        axs[2].set_xticks(range(v.height), v["resolution"], rotation=55)
        axs[3].plot(v["meani_sigi"], color=c)
        axs[3].set_ylabel("I/SigI")
        axs[3].set_xticks(range(v.height), v["resolution"], rotation=55)
    for a in axs:
        a.set_xlabel("resolution")
        a.grid()

    if ref_df is not None:
        # plot reference data
        axs[0].plot(ref_df["cchalf"], color="red")
        axs[1].plot(ref_df["rpim"], color="red")
        axs[2].plot(ref_df["ccanom"], color="red")
        axs[3].plot(ref_df["meani_sigi"], color="red")

    return fig


def get_global_bins(fano_df, bin_label_key):
    bins = fano_df.select(["bin_id", bin_label_key]).unique().sort("bin_id")
    return bins["bin_id"].to_list(), bins[bin_label_key].to_list()


def get_fano_intensity_lazy(scan: pl.LazyFrame, edges: list):
    bin_labels, base_df = _get_bins(edges)
    fano_df = (
        scan.with_columns(
            pl.col("qi_mean").cut(edges, labels=bin_labels).alias("bin_labels")
        )
        .group_by("bin_labels")
        .agg(
            n=pl.len(),
            fano=pl.col("qi_var").mean() / pl.col("qi_mean").mean(),
        )
    )
    return base_df.join(fano_df, on="bin_labels", how="left").sort("bin_id")


def get_fano_lazy(scan: pl.LazyFrame, edges: list) -> pl.LazyFrame:
    bin_labels, base_df = _get_bins(edges=edges)

    label_to_id = {label: i for i, label in enumerate(bin_labels)}

    fano_df = (
        scan.with_columns(pl.col("d").cut(edges, labels=bin_labels).alias("bin_labels"))
        .group_by("bin_labels")
        .agg(
            fano=pl.col("qi_var").mean() / pl.col("qi_mean").mean(),
            n=pl.len(),
        )
        .sort("bin_labels")
    )

    return base_df.join(fano_df, on="bin_labels", how="left").sort("bin_id")


def fano_for_epochs(files: list[Path]):
    dfs_resolution = []
    dfs_intensity = []

    for f in files:
        epoch = int(f.stem.split("_")[-1])

        lf = pl.scan_parquet(f).with_columns(pl.lit(epoch).alias("epoch"))

        fano = get_fano_lazy(
            lf,
            edges=DIALS_EDGES_9B7C,
        ).with_columns(pl.lit(epoch).alias("epoch"))

        fano_intensity = get_fano_intensity_lazy(
            lf, edges=INTENSITY_EDGES
        ).with_columns(pl.lit(epoch).alias("epoch"))

        dfs_resolution.append(fano)
        dfs_intensity.append(fano_intensity)

    return pl.concat(dfs_resolution).collect(), pl.concat(dfs_intensity).collect()


def _plot_iodine_(
    data: dict,
    seqid: int,
    ref_peak_df: pl.DataFrame,
    model_name: str,
    linewidth: int = 2,
):
    epochs = list(data.keys())
    vals = [v for _, v in data.items()]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(
        epochs,
        vals,
        c="black",
        linewidth=linewidth,
    )
    ax.axhline(
        y=ref_peak_df.filter(pl.col("seqid") == seqid)["peakz"].item(),
        c="red",
        label="DIALS",
        linewidth=linewidth,
    )
    ax.grid(which="both")
    ax.set_ylim(ymin=0.0, ymax=35.0)
    ax.set_ylabel("peakz")
    ax.set_xlabel("epoch")
    ax.set_title(f"Iodine {seqid}\n{model_name}")
    ax.legend()
    return fig


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

    colors = get_cube_helix_map(n_epochs)

    fig, ax = plt.subplots(figsize=(8, 4))

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


def main():
    args = parse_args()

    # loading config file
    run_dir = Path(args.run_dir)

    config, wandb_log = open_run(run_dir)

    # Reference data
    # reference_data path
    ref_data_path = _get_reference_data_path(config)
    ref_peak, ref_merged_html = _get_reference_data(ref_data_path)
    ref_peak_df = read_peaks_csv(ref_peak)
    ref_tbl1, ref_tbl2 = pd.read_html(ref_merged_html)
    ref_df = parse_dials_merging_stats(ref_tbl2).collect()

    # model metadata
    model_meta = _get_reference_metadata(config)

    # directory to save images
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = wandb_log / "plots"

    save_dir.mkdir(exist_ok=True)

    # Getting data files
    peaks = glob_predictions(wandb_log, "peaks.csv", sort=False)
    htmls = glob_predictions(wandb_log, "merged.html", sort=False)

    html_dfs = {}
    for h in htmls:
        epoch = epoch_from_path(h.as_posix())
        tbl1, tbl2 = pd.read_html(h)
        html_dfs[epoch] = parse_dials_merging_stats(tbl2).collect()

    fig = plot_merging_stat_subplots(html_dfs, ref_df)
    fig.suptitle(f"{model_meta['qi_name']} merging stats")
    plt.tight_layout()
    save_figure(fig, f"{save_dir}/{model_meta['qbg_name']}_dials_subplots.png", dpi=400)

    ## Iodine Plots
    # Getting plot of iodines
    iod204, iod205, iod206 = {}, {}, {}

    peak_dfs = {}
    seqids = {}

    for n, p in enumerate(peaks):
        peak_dfs[n] = read_peaks_csv(p)

        for s, r in peak_dfs[n][["seqid", "residue"]].iter_rows():
            seqids[s] = r

    for k, v in peak_dfs.items():
        if 204 in v["seqid"]:
            mask = v["seqid"] == 204
            iod204[k] = v["peakz"].filter(mask).item()
        else:
            iod204[k] = 0.0
        if 205 in v["seqid"]:
            mask = v["seqid"] == 205
            iod205[k] = v["peakz"].filter(mask).item()
        else:
            iod205[k] = 0.0
        if 206 in v["seqid"]:
            mask = v["seqid"] == 206
            iod206[k] = v["peakz"].filter(mask).item()
        else:
            iod206[k] = 0.0

    model_name = model_meta["qi_name"]

    # Plotting iodines
    fig = _plot_iodine_(
        iod204, seqid=204, ref_peak_df=ref_peak_df, model_name=model_name
    )
    plt.tight_layout()
    save_figure(fig, f"{save_dir}/{model_name}_iod204.png")

    fig = _plot_iodine_(
        iod205, seqid=205, ref_peak_df=ref_peak_df, model_name=model_name
    )
    plt.tight_layout()
    save_figure(fig, f"{save_dir}/{model_name}_iod205.png")

    fig = _plot_iodine_(
        iod206, seqid=206, ref_peak_df=ref_peak_df, model_name=model_name
    )
    plt.tight_layout()
    save_figure(fig, f"{save_dir}/{model_name}_iod206.png")

    #  PLOTTING FANO
    train_metric_files = list((wandb_log / "files/train_metrics").glob("*.parquet"))
    fano_df_res, fano_df_intensity = fano_for_epochs(train_metric_files)

    fig = plot_fano_over_epoch(
        fano_df_res, bin_label_key="bin_id", edges=DIALS_EDGES_9B7C
    )
    plt.tight_layout()
    save_figure(
        fig,
        f"{save_dir}/{model_meta['qi_name']}_avg_fano_over_epoch.bin.resolution.png",
    )

    # Plotting fano over intensity
    fig = plot_fano_over_epoch(
        fano_df_intensity, bin_label_key="intensity_bin", edges=INTENSITY_EDGES
    )
    plt.tight_layout()
    save_figure(
        fig,
        f"{save_dir}/{model_meta['qi_name']}_avg_fano_over_epoch.bin.intensity.png",
    )


# TODO: Plot iodines from different gammamodels
# TODO: Write a function to calculate Intensity bins
# TODO: Write a function to calculate Resolution bins


if __name__ == "__main__":
    main()
    pl.String
