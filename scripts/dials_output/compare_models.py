import argparse
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

from refltorch.io import load_config
from refltorch.plots import setup_mpl_config

# Setup mpl config for consistent plotting
setup_mpl_config()


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


def _get_reference_data_path(run_config: dict) -> Path:
    cfg = load_config(run_config["config"])
    return Path(cfg["global_vars"]["data_dir"]).parent


def _plot_metric(
    run_ids: Iterable,
    df_map: dict[str, pl.DataFrame],
    base_df: pl.DataFrame,
    model_metadata: dict,
    x_label: str,
    y_label: str,
    title: str,
    x_key: str = "bin_id",
    y_key: str = "mean_qi_var",
    y_scale: bool | None = None,
):
    # plotting the mean qi_var per model
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = base_df["bin_labels"].to_list()
    ticks = base_df["bin_id"].to_list()

    for r in run_ids:
        df_ = df_map[r]
        # iterate over each epoch

        df = base_df.join(df_, on="bin_labels", how="left").sort("bin_id")

        # plot
        model_name = model_metadata[r]["model_metadata"]["qi_name"]
        ax.plot(
            df[x_key],
            df[y_key],
            label=model_name,
        )

    ax.set_xlabel(x_label)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if y_scale is not None:
        ax.set_yscale("log")
    ax.legend()
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

    fig, ax = plt.subplots(figsize=(10, 8))

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
    fig.savefig(fname)


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
    model_metadata: dict,
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

    fig, ax = plt.subplots()
    for r in run_ids:
        label = model_metadata[r]["model_metadata"]["qi_name"]
        lf = peak_lf.filter(pl.col("run_id") == r)
        df = lf.collect()
        df = epoch_df.join(df, on="epoch", how="left").sort("epoch")
        ax.plot(df["epoch"], df[metric], label=label)
    if reference_data is not None:
        ax.axhline(ref_lf[metric].to_list(), c="red", label="DIALS")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over epochs")
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    return fig, ax


def _get_val_loss(wb_data):
    df = pl.DataFrame(wb_data)
    df = df.select(["trainer/global_step", "epoch", "val/loss", "val nll", "val kl"])
    return df.drop_nulls()


def _get_train_loss(wb_data):
    df = pl.DataFrame(wb_data)
    df = df.select(
        ["trainer/global_step", "epoch", "train/loss", "train nll", "train kl"]
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


def main():
    args = parse_args()
    run_dirs = args.run_dirs
    print(run_dirs)
    n_models = len(run_dirs)

    model_metadata = {}

    wandb_log = None
    for rd in run_dirs:
        path = Path(rd)
        run_metadata = list(path.glob("run_metadata.yaml"))[0]
        run_metadata = load_config(run_metadata)

        # wandb metadata
        project = run_metadata["wandb"]["project"]
        run_id = run_metadata["wandb"]["run_id"]

        model_meta = _get_reference_metadata(run_metadata)
        wandb_log = Path(run_metadata["wandb"]["log_dir"]).parent

        # get peak.csv paths
        pred_dir = Path(wandb_log / "predictions/")

        # Getting metrics from W&B
        loss_df = wandb.Api().run(project + "/" + run_id).history()

        # Store model metadata
        model_metadata[run_id] = {
            "run_metadata": run_metadata,
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

    # FIX: this is brittle, make save-directory more robust
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        if wandb_log is None:
            raise ValueError("W&B directory not found")
        save_dir = wandb_log / "plots"
        save_dir.mkdir(exist_ok=True)
    # ENDFIX

    ## Plotting anomalous iodine peak heights as function of epoch

    # Scanning peak.csv files
    all_peaks = [
        csv for run in model_metadata.values() for csv in run.get("peak.csv", [])
    ]
    lf = pl.scan_csv(
        all_peaks,
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

    # reference data if available
    ref_data_path = _get_reference_data_path(run_metadata)
    ref_peak, ref_merged_html = _get_reference_data(ref_data_path)
    ref_peak_lf = pl.scan_csv(ref_peak)
    ref_tbl1, ref_tbl2 = pd.read_html(ref_merged_html)
    ref_merging_stats = get_dials_merging_stats(ref_tbl2).collect()

    # list of epochs
    epochs = (
        lf.select(pl.col("epoch").unique().sort())
        .collect()
        .get_column("epoch")
        .to_list()
    )
    # TODO: Add seqids to args
    seqids = [204, 205, 206]
    run_ids = list(model_metadata.keys())

    # epoch_df = lf.select("epoch").unique().collect().sort("epoch")
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
            model_metadata=model_metadata,
            reference_data=ref_peak_lf,
            metric=m,
        )
        plt.tight_layout()
        fig.savefig(f"{save_dir}/{m}.png")

    # NOTE:
    # plotting anomalous peak heights
    ref_peak_df = ref_peak_lf.collect()
    for s in seqids:
        # reference peak
        ref_peak = ref_peak_df.filter(pl.col("seqid") == s)["peakz"].item()

        fig, ax = plt.subplots(figsize=(8, 5))
        df_seq = lf.filter(pl.col("seqid") == s).collect().sort(["epoch", "seqid"])

        for rid in run_ids:
            df = df_seq.filter(pl.col("run_id") == rid)
            df = epoch_df.join(df, on="epoch", how="left").sort(["epoch", "seqid"])

            # use surrogate prior name as label
            label = model_metadata[rid]["model_metadata"]["qi_name"]

            sns.lineplot(
                x=df["epoch"],
                y=df["peakz"],
                label=label,
                ax=ax,
            )

        ax.axhline(ref_peak, c="red", label="DIALS")
        ax.set_xlabel("epoch")
        ax.set_ylabel("peakz")
        ax.set_title(f"Iodine {s} across models")
        ax.legend()
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.grid()
        plt.tight_layout()
        fig.savefig(f"{save_dir}/iod{s}_model_peaks.png")

    # Plotting Fano binned by resolution
    model_metadata.values()

    all_train_metrics = [
        f for run in model_metadata.values() for f in run.get("train_metrics")
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

        fig, ax = plt.subplots(figsize=(7, 5))
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
        plt.tight_layout()
        fig.savefig(f"test_out/run_{r}_fano.png")
    # END TODO

    # %%
    # plotting the mean qi per model
    # todo: make this into a function
    dials_edges = DIALS_EDGES_9B7C
    bin_labels, base_df = _get_bins(edges=dials_edges)
    base_df = base_df.collect()

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
        model_metadata=model_metadata,
        title="Mean qI.variance over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.variance",
        x_key="bin_id",
        y_key="mean_qi_var",
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/mean_qi_var_model_bin_by_res.png")
    plt.close(fig)

    # Plotting mean qi.mean
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        model_metadata=model_metadata,
        title="Mean qI.mean over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="mean_qi_mean",
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/mean_qi_mean_models_resolution.png")
    plt.close(fig)

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        model_metadata=model_metadata,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="fano",
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/mean_fano_models_resolution.png")
    plt.close(fig)

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        model_metadata=model_metadata,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="var_qi_mean",
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/var_qi_mean_models_res_bin.png")

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        model_metadata=model_metadata,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="var_qi_var",
        y_scale=True,
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/var_qi_var_models_res_bin.png")

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
        df_ = df_map[r]
        model_name = model_metadata[r]["model_metadata"]["qi_name"]
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="bin",
            y_label="",
            x_key="bin_id",
            y_key="mean_qi_var",
            title=f"Mean qi.var over epoch for model {model_name}",
            fname=f"{save_dir}/run_{r}_mean_qi_var_resbin_per_epoch.png",
            epochs=epochs,
            y_scale="log",
        )
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="bin",
            y_label="",
            x_key="bin_id",
            y_key="fano",
            title=f"Mean fano over epoch for model {model_name}",
            fname=f"{save_dir}/run_{r}_mean_fano_resbin_per_epoch.png",
            epochs=epochs,
            y_scale="log",
        )
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="bin",
            y_label="",
            x_key="bin_id",
            y_key="var_qi_mean",
            title=f"Mean var(qi.var) over epoch for model {model_name}",
            fname=f"{save_dir}/run_{r}_var_qi_var_resbin_per_epoch.png",
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
        model_metadata=model_metadata,
        title="Mean qI.variance over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.variance",
        x_key="bin_id",
        y_key="mean_qi_var",
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/mean_qi_var_models_intensity_bins.png")

    # Plotting mean qi.mean
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        model_metadata=model_metadata,
        title="Mean qI.mean over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="mean_qi_mean",
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/mean_qi_mean_models_intensity_bins.png")

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        model_metadata=model_metadata,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="fano",
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/mean_fano_models_intensity_bins.png")

    # Plotting variance(qi.mean)

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        model_metadata=model_metadata,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="var_qi_mean",
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/var_qi_mean_models_intensity_bins.png")

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        model_metadata=model_metadata,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="var_qi_mean",
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/var_qi_mean_models_intensity_bins.png")

    # Plotting mean fano
    fig, ax = _plot_metric(
        run_ids=run_ids,
        df_map=df_map,
        base_df=base_df,
        model_metadata=model_metadata,
        title="Mean fano over resolution bin",
        x_label="intensity bin",
        y_label="mean qi.mean",
        x_key="bin_id",
        y_key="var_qi_var",
        y_scale=True,
    )
    plt.tight_layout()
    fig.savefig(f"{save_dir}/var_qi_var_models_intensity_bins.png")

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
        df_ = df_map[r]
        model_name = model_metadata[r]["model_metadata"]["qi_name"]
        _plot_per_epoch_metric(
            df_,
            base_df=base_df,
            x_label="bin",
            y_label="",
            x_key="bin_id",
            y_key="mean_qi_var",
            title=f"Mean qi.var over epoch for model {model_name}",
            fname=f"{save_dir}/run_{r}_mean_qi_var_per_epoch.png",
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
            fname=f"{save_dir}/run_{r}_mean_fano_per_epoch.png",
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
            fname=f"{save_dir}/run_{r}_var_qi_var_per_epoch.png",
            epochs=epochs,
        )

    colors = mpl.colormaps["Dark2"].colors

    # NOTE:
    # train and validation ELBO
    fig, ax = plt.subplots()
    for i, r in enumerate(run_ids):
        val_loss_df = _get_val_loss(model_metadata[r]["loss_df"])
        train_loss_df = _get_train_loss(model_metadata[r]["loss_df"])
        model_name = model_metadata[r]["model_metadata"]["qi_name"]

        sns.lineplot(
            data=val_loss_df,
            x="epoch",
            y="val/loss",
            label=f"val: {model_name}",
            linestyle="--",
            c=colors[i % len(colors)],
            ax=ax,
        )
        sns.lineplot(
            x=train_loss_df["epoch"],
            y=train_loss_df["train/loss"],
            label=f"train: {model_name}",
            c=colors[i % len(colors)],
            ax=ax,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative log likelihood")
    ax.set_ylim(ymax=3000)
    ax.set_title("ELBO vs Epoch")
    ax.grid()
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(f"{save_dir}/elbo_loss.png")

    # NOTE:
    # train and validation NLL
    fig, ax = plt.subplots()
    for i, r in enumerate(run_ids):
        val_loss_df = _get_val_loss(model_metadata[r]["loss_df"])
        train_loss_df = _get_train_loss(model_metadata[r]["loss_df"])
        model_name = model_metadata[r]["model_metadata"]["qi_name"]

        sns.lineplot(
            x=val_loss_df["epoch"],
            y=val_loss_df["val nll"],
            label=f"val: {model_name}",
            linestyle="--",
            c=colors[i % len(colors)],
        )
        sns.lineplot(
            x=train_loss_df["epoch"],
            y=train_loss_df["train nll"],
            label=f"train: {model_name}",
            c=colors[i % len(colors)],
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative log likelihood")
    ax.set_ylim(ymax=3000)
    ax.set_title("NLL vs Epoch")
    ax.grid()
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(f"{save_dir}/nll_loss.png")

    # NOTE:
    # train and validation KL
    fig, ax = plt.subplots()
    for i, r in enumerate(run_ids):
        val_loss_df = _get_val_loss(model_metadata[r]["loss_df"])
        train_loss_df = _get_train_loss(model_metadata[r]["loss_df"])
        model_name = model_metadata[r]["model_metadata"]["qi_name"]

        sns.lineplot(
            x=val_loss_df["epoch"],
            y=val_loss_df["val kl"],
            label=f"val: {model_name}",
            linestyle="--",
            c=colors[i % len(colors)],
            ax=ax,
        )
        sns.lineplot(
            x=train_loss_df["epoch"],
            y=train_loss_df["train kl"],
            label=f"train: {model_name}",
            c=colors[i % len(colors)],
            ax=ax,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("KL divergence")
    ax.grid()
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(f"{save_dir}/kl_loss.png")

    # NOTE:
    # val/train ELBO gap
    fig, ax = plt.subplots()
    for i, r in enumerate(run_ids):
        val_loss_df = _get_val_loss(model_metadata[r]["loss_df"])
        train_loss_df = _get_train_loss(model_metadata[r]["loss_df"])

        # combining train and val
        combined_loss_df = val_loss_df.join(train_loss_df, on="epoch", how="left")
        combined_loss_df = combined_loss_df.with_columns(
            elbo_diff=pl.col("val/loss") - pl.col("train/loss")
        )

        # label for plot
        model_name = model_metadata[r]["model_metadata"]["qi_name"]
        run_id = model_metadata[r]["run_id"]

        #
        sns.lineplot(
            x=combined_loss_df["epoch"],
            y=combined_loss_df["elbo_diff"],
            label=f"val: {model_name}",
            c=colors[i % len(colors)],
            ax=ax,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_loss - train_loss")
    ax.set_title("validation/training loss gap")
    ax.grid()
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(f"{save_dir}/elbo_gap.png")

    # NOTE:
    # val/train NLL gap
    fig, ax = plt.subplots()
    for r in run_ids:
        val_loss_df = _get_val_loss(model_metadata[r]["loss_df"])
        train_loss_df = _get_train_loss(model_metadata[r]["loss_df"])

        # combining train and val
        combined_loss_df = val_loss_df.join(train_loss_df, on="epoch", how="left")
        combined_loss_df = combined_loss_df.with_columns(
            nll_diff=pl.col("val nll") - pl.col("train nll")
        )

        # label for plot
        model_name = model_metadata[r]["model_metadata"]["qi_name"]

        #
        sns.lineplot(
            x=combined_loss_df["epoch"],
            y=combined_loss_df["nll_diff"],
            label=f"val: {model_name}",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_nll - train_nll")
    ax.set_title("validation/training NLL gap")
    ax.grid()
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(f"{save_dir}/nll_gap.png")

    # NOTE:
    # val/train KL gap
    fig, ax = plt.subplots()
    for r in run_ids:
        val_loss_df = _get_val_loss(model_metadata[r]["loss_df"])
        train_loss_df = _get_train_loss(model_metadata[r]["loss_df"])

        # combining train and val
        combined_loss_df = val_loss_df.join(train_loss_df, on="epoch", how="left")
        combined_loss_df = combined_loss_df.with_columns(
            kl_diff=pl.col("val kl") - pl.col("train kl")
        )

        # label for plot
        model_name = model_metadata[r]["model_metadata"]["qi_name"]

        #
        sns.lineplot(
            x=combined_loss_df["epoch"],
            y=combined_loss_df["kl_diff"],
            label=f"val: {model_name}",
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("val_kl - train_kl")
    ax.set_title("validation/training KL gap")
    ax.grid()
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(f"{save_dir}/kl_gap.png")

    # NOTE:
    # Plot merging statistics from DIALS html files
    pattern = re.compile(r"epoch_(\d+)")
    html_dfs = []
    for run in model_metadata.values():
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
            html_dfs.append(
                get_dials_merging_stats(
                    tbl=tbl2,
                    epoch=epoch,
                    run_id=run_id,
                    model_name=model_name,
                )
            )

        html_df = pl.concat(html_dfs)

    # NOTE:
    # Plot merging statistics for each run
    # Save a .png for each run
    html_df = html_df.collect()
    for run in run_ids:
        df_ = html_df.filter(pl.col("run_id") == run)

        # %%
        # model metadata
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

        # setting up figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.ravel()

        # cchalf plot
        sns.lineplot(
            data=df_,
            x="resolution",
            y="cchalf",
            hue="epoch",
            palette=cmap,
            hue_norm=norm,
            legend=False,
            ax=axs[0],
        )
        axs[0].plot(ref_merging_stats["cchalf"], label="DIALS", color="red")
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
        axs[0].set_xlabel("Resolution bin")
        axs[0].set_ylabel("CChalf")

        # rpim plot
        sns.lineplot(
            data=df_,
            x="resolution",
            y="rpim",
            hue="epoch",
            palette=cmap,
            hue_norm=norm,
            legend=False,
            ax=axs[1],
        )
        axs[1].plot(ref_merging_stats["rpim"], label="DIALS", color="red")
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45)
        axs[1].set_xlabel("Resolution bin")
        axs[1].set_ylabel("Rpim")
        axs[1].grid(alpha=0.5)

        # isigi plot
        sns.lineplot(
            data=df_,
            x="resolution",
            y="meani_sigi",
            hue="epoch",
            palette=cmap,
            hue_norm=norm,
            legend=False,
            ax=axs[2],
        )
        axs[2].plot(ref_merging_stats["meani_sigi"], label="DIALS", color="red")
        axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45)
        axs[2].set_xlabel("Resolution bin")
        axs[2].set_ylabel("I/Sig(I)")

        # ccanom plot
        sns.lineplot(
            data=df_,
            x="resolution",
            y="ccanom",
            hue="epoch",
            palette=cmap,
            hue_norm=norm,
            legend=False,
            ax=axs[3],
        )
        axs[3].plot(ref_merging_stats["ccanom"], label="DIALS", color="red")
        axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=45)
        axs[3].set_xlabel("Resolution bin")
        axs[3].set_ylabel("CCanom")

        for ax in axs:
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Epoch", fontsize=8, rotation=90)
            ax.legend()
            ax.grid(0.5)

        plt.suptitle(f"Merging statistics\n{model_name}\nwb id: {run_id}")
        plt.tight_layout()

        fig.savefig(f"{save_dir}/{run}_merging_stats.png")

    # Getting all predictions
    all_preds = [f for run in model_metadata.values() for f in run.get("preds", [])]
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

    # NOTE:
    # Plotting merging statistics for each run

    # Plot hyper parameters
    n_samples = 10_000
    pad = 2.0
    alpha = 0.4
    alpha2 = 0.5

    for run in run_ids:
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

            fig, ax = plt.subplots(figsize=(5, 5))

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
            plt.tight_layout()
            fig.savefig(f"{save_dir}/dials_vs_{run}_I_{epoch}.png")
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

            fig, ax = plt.subplots(figsize=(5, 5))
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
            plt.tight_layout()
            fig.savefig(f"{save_dir}/dials_vs_{run}_bg_{epoch}.png")
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

            fig, ax = plt.subplots(figsize=(5, 5))
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
            plt.tight_layout()
            fig.savefig(f"{save_dir}/dials_vs_{run}_var_{epoch}.png")

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
            corr_bg=pl.corr("qi_mean", "background.mean"),
        )
        .sort("epoch")
        .collect()
    )

    for (run,), df_run in corr_df.group_by("run_id"):
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

        fig, ax = plt.subplots()
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
        cbar.set_label("Epoch", fontsize=8, rotation=90)

        plt.tight_layout()
        fig.savefig(f"{save_dir}/{run}_corr_I.png")

        # Plotting background correlation

        fig, ax = plt.subplots()
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
        cbar.set_label("Epoch", fontsize=8, rotation=90)

        plt.tight_layout()
        fig.savefig(f"{save_dir}/{run}_corr_bg.png")

        # Plotting var(I) correlation

        fig, ax = plt.subplots()
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
        cbar.set_label("Epoch", fontsize=8, rotation=90)

        plt.tight_layout()
        fig.savefig(f"{save_dir}/{run}_corr_var_I.png")

        # TODO:
        # Plot Phenix refinement values
        dfs = []
        pattern = re.compile(r"epoch_(\d+)")
        for run in run_ids:
            for f in model_metadata[run]["phenix_logs"]:
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
        r_val_df = pl.concat(dfs)

        # Long format for plotting
        long_df = r_val_df.unpivot(
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

        # plot color
        palette = {
            "r_work": "black",
            "r_free": "blue",
        }

        # plot line style
        dashes = {
            "start": (2, 2),
            "final": "",
        }

        for (run,), run_df in long_df.group_by(pl.col("run_id")):
            fig, ax = plt.subplots(figsize=(8, 5))

            sns.lineplot(
                data=run_df,
                x="epoch",
                y="value",
                hue="metric",
                style="stage",
                palette=palette,
                dashes=dashes,
                linewidth=2,
                ax=ax,
            )

            ax.set_xlabel("epoch")
            ax.set_ylabel("r value")
            ax.set_title(f"R-values vs epoch\nrun_id: {run}")
            ax.grid()
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            fig.savefig(f"{save_dir}/{run}_test_unpivot.png")  # %%


if __name__ == "__main__":
    main()
