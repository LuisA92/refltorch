import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from refltorch.io import load_config


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
        type=str,
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
        peak_dfs[n] = pl.read_csv(
            p,
            columns=["seqid", "residue", "peakz"],
            schema_overrides={
                "seqid": pl.Int64,
                "residue": pl.String,
                "peakz": pl.Float64,
            },
        ).sort("seqid")

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
    cmap = sns.cubehelix_palette(start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True)
    cmap_list = cmap(np.linspace(0.0, 1, n_logged_epochs, retstep=True)[0])

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


def get_dials_merging_stats(
    tbl: pd.DataFrame,
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
) -> pl.DataFrame:
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

    df = pl.DataFrame(data)

    return df


def _get_peak_csvs(
    wandb_log: Path,
) -> list:
    log_dir = wandb_log / "predictions/"
    peaks = log_dir.glob("**/peaks.csv")
    return list(peaks)


def _get_dials_htmls(
    wandb_log: Path,
) -> list:
    log_dir = wandb_log / "predictions/"
    htmls = log_dir.glob("**/merged.html")
    return list(htmls)


def _get_reference_data_path(run_config: dict) -> Path:
    cfg = load_config(run_config["config"])
    return Path(cfg["global_vars"]["data_dir"]).parent


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


def _peaks_as_polars(peak_csv: Path):
    df = pl.read_csv(
        peak_csv,
        columns=[
            "seqid",
            "residue",
            "peakz",
        ],
        schema_overrides={
            "seqid": pl.Int64,
            "residue": pl.String,
            "peakz": pl.Float64,
        },
    ).sort("seqid")
    return df


def get_global_bins(fano_df, bin_label_key):
    bins = fano_df.select(["bin_id", bin_label_key]).unique().sort("bin_id")
    return bins["bin_id"].to_list(), bins[bin_label_key].to_list()


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


INTENSITY_EDGES = [0, 10, 25, 50, 100, 300, 600, 1000, 1500, 2500, 5000, 10000]


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

    cmap = sns.cubehelix_palette(start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True)
    colors = cmap(np.linspace(0, 1, n_epochs))

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

    run_metadata = list(run_dir.glob("run_metadata.yaml"))[0]
    config = load_config(run_metadata)

    # Reference data
    # reference_data path
    ref_data_path = _get_reference_data_path(config)
    ref_peak, ref_merged_html = _get_reference_data(ref_data_path)
    ref_peak_df = _peaks_as_polars(ref_peak)
    ref_tbl1, ref_tbl2 = pd.read_html(ref_merged_html)
    ref_df = get_dials_merging_stats(ref_tbl2)

    # model metadata
    model_meta = _get_reference_metadata(config)

    # w&b log-dir
    wandb_log = Path(config["wandb"]["log_dir"]).parent

    # directory to save images
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = wandb_log / "plots"

    save_dir.mkdir(exist_ok=True)

    # Getting data files
    peaks = _get_peak_csvs(wandb_log)
    htmls = _get_dials_htmls(wandb_log)

    pattern = re.compile(r"epoch_(\d+)")
    html_dfs = {}
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
        html_dfs[epoch] = get_dials_merging_stats(tbl2)

    fig = plot_merging_stat_subplots(html_dfs, ref_df)
    fig.suptitle(f"{model_meta['qi_name']} merging stats")
    plt.tight_layout()
    fig.savefig(f"{save_dir}/{model_meta['qbg_name']}_dials_subplots.png", dpi=400)

    ## Iodine Plots
    # Getting plot of iodines
    iod204, iod205, iod206 = {}, {}, {}

    peak_dfs = {}
    seqids = {}

    for n, p in enumerate(peaks):
        peak_dfs[n] = pl.read_csv(
            p,
            columns=["seqid", "residue", "peakz"],
            schema_overrides={
                "seqid": pl.Int64,
                "residue": pl.String,
                "peakz": pl.Float64,
            },
        ).sort("seqid")

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
    fig.savefig(f"{save_dir}/{model_name}_iod204.png")
    plt.close(fig)

    fig = _plot_iodine_(
        iod205, seqid=205, ref_peak_df=ref_peak_df, model_name=model_name
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/{model_name}_iod205.png")
    plt.close(fig)

    fig = _plot_iodine_(
        iod206, seqid=206, ref_peak_df=ref_peak_df, model_name=model_name
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/{model_name}_iod206.png")
    plt.close(fig)

    #  PLOTTING FANO
    train_metric_files = list((wandb_log / "files/train_metrics").glob("*.parquet"))
    fano_df_res, fano_df_intensity = fano_for_epochs(train_metric_files)

    fig = plot_fano_over_epoch(
        fano_df_res, bin_label_key="bin_id", edges=DIALS_EDGES_9B7C
    )
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/{model_meta['qi_name']}_avg_fano_over_epoch.bin.resolution.png"
    )

    # Plotting fano over intensity
    fig = plot_fano_over_epoch(
        fano_df_intensity, bin_label_key="intensity_bin", edges=INTENSITY_EDGES
    )
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/{model_meta['qi_name']}_avg_fano_over_epoch.bin.intensity.png"
    )


# TODO: Plot iodines from different gammamodels
# TODO: Write a function to calculate Intensity bins
# TODO: Write a function to calculate Resolution bins


if __name__ == "__main__":
    main()
    pl.String
