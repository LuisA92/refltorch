import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

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


def main():
    args = parse_args()
    run_dirs = args.run_dirs
    n_models = len(run_dirs)

    peak_csvs = []
    train_metric_files = []
    model_metadata = {}

    # Setting up save_dir
    # save_dir = Path("test_out")
    # save_dir.mkdir(exist_ok=True)

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

        # Store model metadata
        model_metadata[run_id] = {
            "run_metadata": run_metadata,
            "run_id": run_id,
            "model_metadata": model_meta,
            "wandb_log_dir": wandb_log.as_posix(),
        }

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

    #

    ref_data_path = _get_reference_data_path(run_metadata)
    ref_peak, ref_merged_html = _get_reference_data(ref_data_path)
    ref_df = pl.scan_csv(ref_peak)
    epochs = ref_df.select("epoch")

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
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        for e in epochs:
            lf_ = lf_.filter(pl.col("epoch") == e)
            joined = base_df.join(lf_, on="bin_labels", how="left")

            ax.plot()

        joined = base_df.join(lf_, on="bin_labels", how="left").sort("bin_id")
        plot_fano_over_epoch(lf_)
        print(joined.collect())


def _plot_iodine(
    df,
    ax,
    label,
    epochs,
    seqid: int,
    ref_df: None = None,
):
    y = df.filter(pl.col("seqid") == seqid)["peakz"].unique()
    ax.plot(epochs, y, label=label)
    if ref_df is not None:
        ax.plot(
            epochs,
        )
    return ax


if __name__ == "__main__":
    main()

"""
['./gammaC_run_55410873/']
['./gammaC_run_55410873/','./gammaB_run_55340166/','./gammaA_run_55340165/','./folded_normal_A_run_55416467/']
"""

lf = (
    pl.scan_parquet(train_metrics, include_file_paths="filenames")
    .with_columns(
        run_id=pl.col("filenames").str.extract(r"/run-[^/]+-([^/]+)/", 1),
        epoch=pl.col("filenames")
        .str.extract(r"/train_epoch_(\d+).parquet", 1)
        .cast(pl.Int32),
    )
    .with_columns(
        bin_labels=pl.when(pl.col("qi_mean") < 0)
        .then(pl.lit("<0"))
        .otherwise(pl.col("qi_mean").cut(INTENSITY_EDGES, labels=bin_labels))
    )
)


lf_agg = lf.group_by(["run_id", "epoch", "bin_labels"]).agg(
    fano=pl.col("qi_var").mean() / pl.col("qi_mean").mean(),
    n=pl.len(),
)

lf_joined = base_df.join(lf_agg, on="bin_labels", how="left").sort("bin_id")

df_all = lf_joined.collect()

