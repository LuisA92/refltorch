import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

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
        type=list,
        help="List of paths to the run-directories",
    )
    parser.add_argument(
        "--seqids",
        type=list,
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


def main():
    args = parse_args()
    run_dirs = args.run_dirs
    n_models = len(run_dirs)
    save_dir = args.save_dir

    peak_csvs = []
    train_metric_files = []
    pt_files = []
    model_metadata = {}

    # Setting up save_dir
    # save_dir = Path("test_out")
    # save_dir.mkdir(exist_ok=True)

    wandb_log = None
    for rd in run_dirs:
        path = Path(rd)
        run_metadata = list(path.glob("run_metadata.yaml"))[0]
        run_metadata = load_config(run_metadata)

        # wandb id
        run_id = run_metadata["wandb"]["run_id"]

        model_meta = _get_reference_metadata(run_metadata)
        wandb_log = Path(run_metadata["wandb"]["log_dir"]).parent

        # get peak.csv paths
        pred_dir = Path(wandb_log / "predictions/")
        peak_csvs.extend(list(pred_dir.glob("**/*.csv")))

        # get train_metrics.parquet paths
        train_metric_files.extend(
            list((wandb_log / "files/train_metrics").glob("*.parquet"))
        )

        pt_files.extend(list((wandb_log / "predictions").glob("**/preds.pt")))

        # Store model metadata
        model_metadata[run_id] = {
            "run_metadata": run_metadata,
            "run_id": run_id,
            "model_metadata": model_meta,
            "wandb_log_dir": wandb_log.as_posix(),
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

    lf = pl.scan_csv(peak_csvs, include_file_paths="filenames")
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
    ref_df = pl.scan_csv(ref_peak)

    # list of epochs
    epochs = lf.collect()["epoch"].to_list()

    # TODO: Add seqids to args
    seqids = [204, 205, 206]

    run_ids = lf.select("run_id").unique().collect().to_series()
    epoch_df = lf.select("epoch").unique().collect().sort("epoch")

    # plotting anomalous peak heights
    for s in seqids:
        fig, ax = plt.subplots(figsize=(8, 5))
        for rid in run_ids:
            df = (
                lf.filter((pl.col("seqid").is_in(seqids)) & (pl.col("run_id") == rid))
                .collect()
                .sort(["epoch", "seqid"])
            )
            df = epoch_df.join(df, on="epoch", how="left").sort(["epoch", "seqid"])

            # use surrogate prior name as label
            label = model_metadata[rid]["model_metadata"]["qi_name"]

            epochs = df["epoch"].unique()
            y = df.filter(pl.col("seqid") == s)["peakz"].unique()
            ax.plot(epochs, y, label=label)

        ax.set_xlabel("epoch")
        ax.set_ylabel("peakz")
        ax.set_title(f"Iodine {s} across models")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        fig.savefig(f"{save_dir}/iod{s}_model_peaks.png")

    # Plotting Fano binned by resolution

    lf_train_metrics = pl.scan_parquet(
        train_metric_files, include_file_paths="filenames"
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
        for c, e in zip(cmap_list, epochs):
            lf_epoch = lf_.filter(pl.col("epoch") == e)
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


if __name__ == "__main__":
    main()
