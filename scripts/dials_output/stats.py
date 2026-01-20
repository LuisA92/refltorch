# TODO: Fix this file
import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
import torch
import wandb
from dials.array_family import flex

from refltorch.io import load_config


class ReflectionTable:
    def __init__(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        self._df = pl.DataFrame(*args, **kwargs)
        self.res_bins = None

    def __getattr__(self, name: str) -> Any:
        "Redicrect any calls to the underlying Polars DataFrame"
        return getattr(self._df, name)

    def describe(
        self,
        percentiles: Sequence[float] | float | None = (0.25, 0.50, 0.75),
    ):
        """
        Summary statistics for a ReflectionTable.
        Defaults to reporting summary statistics for intentisty and background columns.

        Args:
            percentiles:

        Returns:

        """
        if not self.columns:
            msg = "cannot describe a DataFrame that has no columns"
            raise TypeError(msg)

        return (
            self.select(
                [
                    "intensity.prf.value",
                    "intensity.prf.variance",
                    "intensity.sum.value",
                    "intensity.sum.variance",
                    "background.sum.value",
                    "background.mean",
                ]
            )
            .lazy()
            .describe(percentiles=percentiles)
        )


def read_refl_file(
    source: str | Path,
    columns: None = None,
):
    """
    Function to read a .refl file as a polars.DataFrame

    Args:
        source:
        columns:

    Returns:

    """
    # if isinstance(source,dials_array_family_flex_ext.reflection_table):
    if isinstance(source, (str, Path)):
        try:
            tbl = flex.reflection_table.from_file(source)
            h, k, l = tbl["miller_index"].as_vec3_double().parts()

            df = ReflectionTable(
                {
                    "h": h,
                    "k": k,
                    "l": l,
                    "intensity.prf.value": tbl["intensity.prf.value"],
                    "intensity.prf.variance": tbl["intensity.prf.variance"],
                    "intensity.sum.value": tbl["intensity.sum.value"],
                    "intensity.sum.variance": tbl["intensity.sum.variance"],
                    "background.sum.value": tbl["background.sum.value"],
                    "background.mean": tbl["background.mean"],
                    "d": tbl["d"],
                    "refl_ids": tbl["refl_ids"],
                }
            )

            # assining reflection bins
            nbins = 20
            max = df._df["d"].max()
            max_90 = (
                df._df["d"].describe(percentiles=(0.25, 0.5, 0.90))[-2]["value"].item()
            )
            right = np.linspace(max_90, df._df["d"].min(), nbins)
            left = np.hstack([max, np.linspace(max_90, df._df["d"].min(), nbins)])[:-1]
            df.res_bins = np.vstack([left, right]).T.tolist()

            return df

        except OSError:
            print("source could not be read by flex.reflection_table.from_file")
        except ValueError:
            print("The input source is not a valid .refl file")
    else:
        print("Input must be a str or Path to a .refl file")


class CustomNorm(mpl.colors.Normalize):
    def __init__(self, vmin=0, vmax=1, cmap_min=0.0, cmap_max=1.0):
        super().__init__(vmin, vmax)
        self.cmap_min = cmap_min
        self.cmap_max = cmap_max

    def __call__(self, value, clip=None):
        # First normalize to [0, 1]
        normalized = super().__call__(value, clip)
        # Then map to [cmap_min, cmap_max]
        return self.cmap_min + normalized * (self.cmap_max - self.cmap_min)


class Data:
    def __init__(self, path):
        self.model_report = dict()
        self.path = path
        self.html_files = list(path.glob("**/merged.html"))
        self.peak_files = list(path.glob("**/peaks.csv"))
        self.reference_html = list(reference_path.glob("merged*.html"))[0]
        self.reference_peaks = list(reference_path.glob("reference_peaks.csv"))[0]
        #        self.avg_train_metrics = pl.read_csv(list(path.glob("**/avg*train*.csv"))[0])
        #        self.avg_val_metrics = pl.read_csv(list(path.glob("**/avg*val*.csv"))[0])
        self.wandb_id = path.name.split("-")[-1]
        self.config_path = list(path.glob("**/config_copy.yaml"))[0]
        self.config = load_config(list(path.glob("**/config_copy.yaml"))[0])
        self.path_to_val_labels = list(self.path.glob("**/val_labels.csv"))[0]
        self.path_to_train_labels = list(self.path.glob("**/train_labels.csv"))[0]
        val_labels = (
            pl.read_csv(list(self.path.glob("**/val_labels.csv"))[0])
            .with_columns((pl.lit("val")).alias("label"))
            .rename({"val_ids": "dataid"})
        )
        train_labels = (
            pl.read_csv(list(self.path.glob("**/train_labels.csv"))[0])
            .with_columns((pl.lit("train")).alias("label"))
            .rename({"train_ids": "dataid"})
        )
        self.labels = val_labels.vstack(train_labels).sort("dataid")
        # self.val_epochs = self.avg_val_metrics["epoch"].to_numpy()
        # self.train_epochs = self.avg_train_metrics["epoch"].to_numpy()
        self.plot_log = dict()


#    def plot_metric(self, ax, metric, log=False):
#        # matching epochs by odd number
#        epochs_filtered = self.avg_train_metrics.filter(pl.col("epoch") % 2 > 0)[
#            "epoch"
#        ].to_list()
#        train_filtered = self.avg_train_metrics.filter(pl.col("epoch") % 2 > 0)
#        val_filtered = self.avg_val_metrics.filter(pl.col("epoch") % 2 > 0)
#
#        ax.plot(
#            epochs_filtered,
#            train_filtered[metric],
#            label=f"train_{metric}",
#            color="black",
#        )
#        ax.plot(
#            epochs_filtered, val_filtered[metric], label=f"val_{metric}", color="blue"
#        )
#        ax.set_title(f"train and val {metric}")
#        ax.set_xticks(
#            np.linspace(1, epochs_filtered[-1], len(epochs_filtered)), epochs_filtered
#        )
#        ax.set_xlabel("epoch")
#        ax.set_ylabel(f"{metric}")
#        if metric in {"avg_nll", "avg_loss"}:
#            ax.set_ylim(1200, 1400)
#        ax.legend()
#        ax.grid()
#
#        if log:
#            train_key = f"train_{metric}"
#            val_key = f"val_{metric}"
#            self.plot_log[train_key] = train_filtered[metric].to_list()
#            self.plot_log[val_key] = val_filtered[metric].to_list()

#    def plot_metric_diff(self, ax, metric, log=False):
#        # matching epochs by odd number
#        epochs_filtered = self.avg_train_metrics.filter(pl.col("epoch") % 2 > 0)[
#            "epoch"
#        ].to_list()
#        train_filtered = self.avg_train_metrics.filter(pl.col("epoch") % 2 > 0)
#        val_filtered = self.avg_val_metrics.filter(pl.col("epoch") % 2 > 0)
#        avg_loss_diff = (train_filtered[metric] - val_filtered[metric]).abs()
#
#        ax.plot(epochs_filtered, avg_loss_diff, color="k")
#        ax.set_title(f"train/val {metric} gap")
#        ax.set_xticks(
#            np.linspace(1, epochs_filtered[-1], len(epochs_filtered)), epochs_filtered
#        )
#        ax.set_xlabel("epoch")
#        ax.set_ylabel(f"Abs(train_{metric.strip('avg_')} - val_{metric.strip('avg_')})")
#        ax.grid()
#        if log:
#            key = f"{metric}_val_train_gap"
#            self.plot_log[key] = avg_loss_diff


if __name__ == "__main__":
    # load data

    import os

    os.environ["WANDB_DIR"] = "/n/netscratch/hekstra_lab/Lab/laldama/lightning_logs/"
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--path",
        type=str,
    )
    argparser.add_argument(
        "--reference_path",
        type=str,
        default="/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/reference_data/",
    )
    args = argparser.parse_args()

    # -
    # NOTE: 2025-07-09
    reference_path = Path(args.reference_path)
    path = Path(args.path)

    # path = Path(
    # "/Users/luis/Downloads/lightning_logs/lightning_logs/wandb/run-20250708_135815-ifnq043z"
    # )
    # reference_path = Path("/Users/luis/integratorv3/data/pass1/")

    id = path.name.split("-")[-1]
    data = Data(path)

    run = wandb.init(project="test_new_analysis_v3", id=id)

    # -
    # NOTE: Functions to plot metric and their difference

    # plot train/loss avg_loss and
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))

    # data.plot_metric(axes[0, 0], "avg_loss", log=True)
    # data.plot_metric_diff(axes[0, 1], "avg_loss", log=True)
    # data.plot_metric(axes[1, 0], "avg_nll", log=True)
    # data.plot_metric_diff(axes[1, 1], "avg_nll", log=True)
    # data.plot_metric(axes[2, 0], "avg_kl", log=True)
    # data.plot_metric_diff(axes[2, 1], "avg_kl", log=True)

    plt.tight_layout()
    plt.savefig(f"{path.as_posix()}/avg_metrics.png", dpi=600)
    wandb.log({"Avg metrics": wandb.Image(plt.gcf())})
    plt.clf()

    # -

    metrics = ["CC½", "CCano", "Rpim", "Mean I/σ(I)"]
    merging_stats_dfs = []

    for h in data.html_files:
        df = pd.read_html(h)

        epoch = int(h.parents[2].name.split("_")[-1])
        cc_half = [
            float(entry.strip("*")) if isinstance(entry, str) else entry
            for entry in df[1]["CC½"].to_list()
        ]
        cc_ano = [
            float(entry.strip("*")) if isinstance(entry, str) else entry
            for entry in df[1]["CCano"]
        ]
        rpim = df[1]["Rpim"].to_list()
        meanIsigI = df[1]["Mean I/σ(I)"].to_list()
        resolution = df[1]["Resolution (Å)"].to_list()
        merging_stats_dfs.append(
            pl.DataFrame(
                {
                    "epoch": epoch,
                    "Resolution (Å)": [resolution],
                    "CC½": [cc_half],
                    "CCano": [cc_ano],
                    "Rpim": [rpim],
                    "Mean I/σ(I)": [meanIsigI],
                }
            )
        )

    merging_stat_df = pl.concat(merging_stats_dfs).sort("epoch")

    # -
    metrics = ["CC½", "CCano", "Rpim", "Mean I/σ(I)"]
    y_range = {
        "CC½": {"ymin": 0.0, "ymax": 1.10},
        "CCano": {"ymin": -0.5, "ymax": 0.6},
        "Rpim": {"ymin": -0.10, "ymax": 1.0},
        "Mean I/σ(I)": {"ymin": -10, "ymax": 140},
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    cmap = sns.cubehelix_palette(start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True)
    cmap_list = cmap(np.linspace(0.0, 1, len(data.val_epochs[1:]), retstep=2)[0])
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        ref_df = pd.read_html(data.reference_html)
        if ref_df[1][metric].dtype == "O":
            ref_vals = [
                float(entry.strip("*")) for entry in ref_df[1][metric].to_list()
            ]
        else:
            ref_vals = ref_df[1][metric].to_list()

        for color, e in zip(cmap_list, merging_stat_df["epoch"]):
            df = merging_stat_df.filter(pl.col("epoch") == e)
            vals = merging_stat_df.filter(pl.col("epoch") == e)[metric].item()
            x_ticks = np.linspace(
                1,
                df["Resolution (Å)"].list.len().item(),
                df["Resolution (Å)"].list.len().item(),
            )
            ax.plot(x_ticks, vals, color=color)
            ax.set_xticks(x_ticks, df["Resolution (Å)"].item(), rotation=55)
            ax.set_xlabel("Resolution (Å)")
            ax.set_ylabel(f"{metric}", fontsize=12)
            ax.set_ylim(y_range[metric]["ymin"], y_range[metric]["ymax"])
            ax.grid(alpha=0.5)
        ax.plot(x_ticks, ref_vals, color="red", label="DIALS")
        ax.legend()
        norm = CustomNorm(
            vmin=0, vmax=data.val_epochs[-1] - 1, cmap_min=0.0, cmap_max=1.0
        )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Epoch")
    fig.suptitle(
        f"Merging statistics\nwandb_id: {data.wandb_id}\nβ₁: {data.config['components']['loss']['args']['pprf_weight']}, β₁: {data.config['components']['loss']['args']['pi_weight']}, β₃: {data.config['components']['loss']['args']['pbg_weight']}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(f"{path.as_posix()}/merging_stats.png", dpi=600)

    wandb.log({"Merging Stats": wandb.Image(plt.gcf())})
    plt.clf()

    # NOTE: code to plot anomalous peak heights
    ref_peaks = pl.read_csv(data.reference_peaks)

    epochs = []
    for c in data.peak_files:
        suffix = int(c.parents[3].name.split("_")[-1])
        epochs.append(suffix)
        ref_peaks = ref_peaks.join(
            pl.read_csv(c)["seqid", "residue", "peakz"],
            on=["seqid", "residue"],
            how="full",
            suffix="_" + str(suffix),
            coalesce=True,
        )

    epochs.sort()
    selectors = ["peakz_" + str(e) for e in epochs]
    ref_peaks_sorted = ref_peaks.sort("seqid")

    columns = [
        l + "<br>" + r
        for l, r in zip(
            ref_peaks_sorted["residue"].to_list(),
            ref_peaks_sorted["seqid"].cast(str).to_list(),
        )
    ]
    # peakz_df = ref_peaks_sorted.select(selectors).transpose(column_names=columns)
    peakz_df = ref_peaks_sorted.select(selectors)

    # start new
    merging_stat_df = merging_stat_df.with_columns(peak_names=columns)
    peakz = []
    for e, c in zip(epochs, peakz_df.columns):
        print(e)
        peakz.append(peakz_df[c].to_list())
    merging_stat_df = merging_stat_df.hstack(pl.DataFrame({"peakz": peakz}))
    # end new
    # max height and corresponding epoch
    max_peak_index = int(np.argmax(peakz_df.max().to_numpy()))
    max_peak_epoch = peakz_df[:, max_peak_index].name.replace("peakz", "epoch")
    max_peak = peakz_df[:, max_peak_index].max()

    columns.insert(0, "epoch")
    values = np.round(peakz_df.to_numpy(), 2).astype(str)
    ref_values = np.round(
        ref_peaks_sorted["peakz"].to_numpy().reshape(ref_peaks_sorted.height, 1), 2
    ).astype(str)
    values = np.hstack([ref_values, values])
    epochs.insert(0, "reference")
    values = np.vstack([np.array(epochs, dtype=str), values])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=columns,
                    line_color="darkslategray",
                    fill_color="#CDCDCD",
                    align="center",
                    font=dict(color="black", size=16),
                ),
                cells=dict(
                    values=values,
                    line_color="darkslategray",
                    fill_color=[["white", "#F3F3F3"] * len(values)],
                    align="center",
                    height=30,
                    font=dict(
                        color="black",
                        size=14,
                    ),
                ),
            )
        ]
    )
    fig.update_layout(
        title="Anomalous peak heights",
        margin=dict(l=20, r=20, t=60, b=20),
        width=1250,  # Full width
        height=None,  # Auto height
    )

    wandb.log({"Anomalous peak heights": wandb.Html(fig.to_html())})

    # -
    # NOTE: Code to get start/final r-free and r-work from phenix.logs
    # reference_path = Path("/Users/luis/Downloads/lightning_logs/reference_data")

    # string to match
    pattern1 = re.compile(r"Start R-work")
    pattern2 = re.compile(r"Final R-work")
    log_files = list(path.glob("**/phenix_out/refine*.log"))
    log_files.insert(0, list(reference_path.glob("**/refine*.log"))[0])

    epoch_start = []
    rwork_start = []
    rfree_start = []
    epoch_final = []
    rwork_final = []
    rfree_final = []

    for log_file in log_files:
        with log_file.open("r") as f:
            lines = f.readlines()
        if re.search("epoch", log_file.as_posix()):
            epoch = re.findall(r"epoch_\d*", log_file.as_posix())[0].split("_")[-1]

        else:
            epoch = "reference"

        matched_lines_start = [line.strip() for line in lines if pattern1.search(line)]
        matched_lines_final = [line.strip() for line in lines if pattern2.search(line)]

        if matched_lines_start:
            epoch_start.append(epoch)
            rwork_start.append(re.findall(r"\d+\.\d+", matched_lines_start[0])[0])
            rfree_start.append(re.findall(r"\d+\.\d+", matched_lines_start[0])[1])
        if matched_lines_final:
            epoch_final.append(epoch)
            rwork_final.append(re.findall(r"\d+\.\d+", matched_lines_final[0])[0])
            rfree_final.append(re.findall(r"\d+\.\d+", matched_lines_final[0])[1])

    rvals_df = pl.DataFrame(
        {
            "epoch": epoch_start,
            "Rwork_start": rwork_start,
            "Rfree_start": rfree_start,
            "Rwork_final": rwork_final,
            "Rfree_final": rfree_final,
        }
    )
    rvals_df = rvals_df.with_columns(
        (pl.col("Rfree_final").cast(float) - pl.col("Rwork_final").cast(float))
        .round(4)
        .alias("Rfree-Rwork")
    )
    rvals_df = rvals_df[0].vstack(
        rvals_df[1:]
        .with_columns(pl.col("epoch").cast(int))
        .sort("epoch")
        .with_columns(pl.col("epoch").cast(str))
    )

    merging_stat_df = merging_stat_df.hstack(
        rvals_df[1:][
            "Rwork_start", "Rfree_start", "Rwork_final", "Rfree_final", "Rfree-Rwork"
        ]
    )

    fill_colors = [["#f9f9f9", "#e6e6e6"][(i % 2)] for i in range(len(epochs))]

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[10, 10, 10, 10],
                header=dict(
                    values=list(rvals_df.columns),
                    fill_color="lightgrey",
                    align="center",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=rvals_df.to_numpy().T,
                    fill_color=[fill_colors],
                    align="center",
                    font=dict(size=12),
                ),
            )
        ]
    )
    fig.update_layout(title_text="R-values over epoch", title_x=0.5, width=700)

    wandb.log({"R-vals": wandb.Html(fig.to_html())})

    # NOTE: Code to plot final rwork/rfree as function of epoch

    epochs = rvals_df["epoch"][1:].cast(int).to_list()
    plt.hlines(
        y=rvals_df["Rwork_final"].cast(float)[0],
        xmin=0,
        xmax=49,
        linestyle="--",
        label="Ref Final R-work",
        color="black",
    )
    plt.hlines(
        y=rvals_df["Rfree_final"].cast(float)[0],
        xmin=0,
        xmax=49,
        linestyle="--",
        label="Ref Final R-free",
        color="blue",
    )
    plt.plot(
        epochs,
        rvals_df["Rfree_final"][1:].cast(float).to_list(),
        label="Final R-free",
        color="black",
    )
    plt.plot(
        epochs,
        rvals_df["Rwork_final"][1:].cast(float),
        label="Final R-work",
        color="blue",
    )
    plt.ylabel("R value")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()

    wandb.log({"R-vals vs epoch": wandb.Image(plt.gcf())})
    plt.clf()

    # -
    # TODO: plots for weighted cc half
    cc_weighted = []
    epochs = []

    for h in data.html_files:
        df = pd.read_html(h)
        epochs.append(int(h.parents[2].name.split("_")[-1]))
        cc_half = [float(s.strip("*")) for s in df[1]["CC½"].to_numpy()]
        n_obs = df[1]["N(obs)"].to_numpy()
        cc_weighted.append((cc_half * n_obs).sum() / n_obs.sum())

    df = pl.DataFrame({"epochs": epochs, "cc_weighted": cc_weighted}).sort("epochs")

    bets_cc_weighted = df["epochs"][df["cc_weighted"].arg_max()]

    ref_df = pd.read_html(data.reference_html)
    n_obs_ref = ref_df[1]["N(obs)"].to_numpy()
    ref_cchalf = [float(s.strip("*")) for s in ref_df[1]["CC½"].to_numpy()]

    cc_weighted = df["cc_weighted"].to_list()
    ref_cc_weighted = (n_obs_ref * ref_cchalf).sum() / n_obs_ref.sum()

    merging_stat_df = merging_stat_df.hstack(df.select("cc_weighted"))

    x_axis = np.linspace(1, len(cc_weighted), len(cc_weighted))
    plt.plot(x_axis, cc_weighted, color="black", label="Model")
    plt.axhline(y=ref_cc_weighted, color="red", label="DIALS")
    plt.legend()
    plt.title(f"Weighted CC_half\nbets_cc_weighted epoch: {bets_cc_weighted}")
    plt.ylabel("weighted CC_half")
    plt.xlabel("epoch")
    plt.xticks(x_axis, df["epochs"].cast(str))
    plt.grid()

    wandb.log({"Weighted CC_half": wandb.Image(plt.gcf())})
    plt.clf()

    # plt.show()

    # -
    # NOTE: Plot DIALS vs Network for 'best' epoch
    best_epoch = f"epoch_{str(bets_cc_weighted)}"

    dfs = []
    for file in data.peak_files:
        dfs.append(pl.DataFrame(pd.read_csv(file)))

    preds = torch.load(list(path.glob(f"**/{best_epoch}/*.pt"))[0], weights_only=False)

    pred_df = pl.DataFrame(preds)
    pred_df = pred_df.explode(pred_df.columns).sort("refl_ids")
    pred_df = pred_df.hstack(data.labels).drop("dataid")

    x_val = pred_df["intensity_mean"].filter(pred_df["label"] == "val")
    y_val = pred_df["dials_I_prf_value"].filter(pred_df["label"] == "val")

    x_train = pred_df["intensity_mean"].filter(pred_df["label"] == "train")
    y_train = pred_df["dials_I_prf_value"].filter(pred_df["label"] == "train")

    plt.scatter(x_train, y_train, s=2, color="#005d5d", label="train", alpha=0.2)
    plt.scatter(x_val, y_val, s=2, color="#6929c4", label="val", alpha=0.2)
    plt.ylabel("intensity.prf.value")
    plt.xlabel("qI.mean")
    plt.title(f"DIALS vs {best_epoch}\n")
    plt.yscale("symlog")
    plt.xscale("symlog")
    plt.legend()
    plt.grid()

    wandb.log({"DIALS vs Network": wandb.Image(plt.gcf())})
    plt.clf()
    # plt.show()

    # -
    # TODO: Model report
    # Aggregate a summary of the model performance
    summary_df = pl.DataFrame(
        {
            "wandb_id": data.wandb_id,
            "wandb_log_path": data.path.as_posix(),
            "qi": data.config["components"]["qi"]["name"],
            "qbg": data.config["components"]["qbg"]["name"],
            "qp": data.config["components"]["qp"]["name"],
            "pprf_name": data.config["components"]["loss"]["args"]["pprf_name"],
            "pprf_weight": data.config["components"]["loss"]["args"]["pprf_weight"],
            "pi_name": data.config["components"]["loss"]["args"]["pi_name"],
            "pi_weight": data.config["components"]["loss"]["args"]["pi_weight"],
            "pi_params": [
                [k, str(v)]
                for k, v in data.config["components"]["loss"]["args"][
                    "pi_params"
                ].items()
            ],
            "pbg_name": data.config["components"]["loss"]["args"]["pbg_name"],
            "pbg_weight": data.config["components"]["loss"]["args"]["pbg_weight"],
            "pbg_params": [
                [k, str(v)]
                for k, v in data.config["components"]["loss"]["args"][
                    "pbg_params"
                ].items()
            ],
            "config_path": data.config_path.as_posix(),
            #            "min_nll_val": data.avg_val_metrics["avg_nll"].min(),
            #            "min_nll_epoch_val": data.avg_val_metrics[
            #                data.avg_val_metrics["avg_nll"].arg_min()
            #            ]["epoch"].item(),
            #            "min_nll_train": data.avg_train_metrics["avg_nll"].min(),
            #            "min_nll_epoch_train": data.avg_train_metrics[
            #                data.avg_train_metrics["avg_nll"].arg_min()
            #            ]["epoch"].item(),
            "val_avg_kl": [data.plot_log["val_avg_kl"]],
            "val_avg_nll": [data.plot_log["val_avg_nll"]],
            "val_avg_loss": [data.plot_log["val_avg_loss"]],
            "train_avg_kl": [data.plot_log["train_avg_kl"]],
            "train_avg_nll": [data.plot_log["train_avg_nll"]],
            "train_avg_loss": [data.plot_log["train_avg_loss"]],
            "max_peak_epoch": max_peak_epoch,
            "max_peak": max_peak,
            "avg_loss_val_train_gap": [data.plot_log["avg_loss_val_train_gap"]],
            "avg_kl_val_train_gap": [data.plot_log["avg_kl_val_train_gap"]],
            "avg_nll_val_train_gap": [data.plot_log["avg_nll_val_train_gap"]],
            "val_epochs": [data.val_epochs.tolist()],
            "train_epochs": [data.train_epochs.tolist()],
            "train_label_paths": data.path_to_train_labels.as_posix(),
            "val_label_paths": data.path_to_val_labels.as_posix(),
            "cc_weighted": [cc_weighted],
            "best_cc_weighted_epoch": bets_cc_weighted,
            "start_rwork": [rvals_df["Rwork_start"].cast(float).to_list()],
            "final_rwork": [rvals_df["Rwork_final"].cast(float).to_list()],
            "start_rfree": [rvals_df["Rfree_start"].cast(float).to_list()],
            "final_rfree": [rvals_df["Rfree_final"].cast(float).to_list()],
            "final_rfree-rwork": [rvals_df["Rfree-Rwork"].cast(float).to_list()],
        }
    )

    # write out a summary json file
    summary_df.write_json(f"{data.path}/summary_{data.wandb_id}.json")

    artifact = wandb.Artifact(name="model_summary", type="DataFrame")
    artifact.add_file(
        local_path=f"{data.path}/summary_{data.wandb_id}.json", name="model_summary"
    )
    artifact.save()

    # -

    pred_list = list(path.glob("**/preds.pt"))

    pred_dfs = []
    for p in pred_list:
        epoch = re.findall(r"epoch_\d*", p.as_posix())[0].split("_")[-1]
        preds = torch.load(p, weights_only=False)
        sample_dict = dict()
        for c in (
            pl.DataFrame(
                {
                    "intensity_mean": np.concatenate(preds["intensity_mean"]),
                    "intensity_var": np.concatenate(preds["intensity_var"]),
                    "dials_I_sum_value": np.concatenate(preds["dials_I_sum_value"]),
                    "dials_I_sum_var": np.concatenate(preds["dials_I_sum_var"]),
                    "dials_I_prf_value": np.concatenate(preds["dials_I_prf_value"]),
                    "dials_I_prf_var": np.concatenate(preds["dials_I_prf_var"]),
                    "qbg_mean": np.concatenate(preds["qbg_mean"]),
                    "d": np.concatenate(preds["qbg_mean"]),
                }
            )
            .sample(10000)
            .iter_columns()
        ):
            sample_dict["epoch"] = int(epoch)
            sample_dict[c.name] = [c.to_list()]

        pred_dfs.append(pl.DataFrame(sample_dict))

    merged_df = merging_stat_df.hstack(
        pl.concat(pred_dfs)
        .sort("epoch")
        .select(
            "intensity_mean",
            "intensity_var",
            "dials_I_sum_value",
            "dials_I_sum_var",
            "dials_I_prf_value",
            "dials_I_prf_var",
            "qbg_mean",
            "d",
        )
    )

    merged_df.write_json(f"{data.path}/pred_summary_{data.wandb_id}.json")
    # -

    reference_df = read_refl_file(data.config["output"]["refl_file"])

    res_bins = reference_df.res_bins

    nn_refl_tbl = read_refl_file(
        list(path.glob(f"**/{best_epoch}/reflections/dials_out_posterior/*.refl"))[
            0
        ].as_posix()
    )

    nn_refl_tbl = nn_refl_tbl.join(
        data.labels.rename({"dataid": "refl_ids"}), on="refl_ids"
    )

    merged_df = reference_df.join(nn_refl_tbl, on="refl_ids", suffix="_pred")

    fig, axes = plt.subplots(4, 5, figsize=(20, 15))
    axes = axes.ravel()
    scale = "log"

    for i, bin in enumerate(res_bins):
        filtered_df = merged_df.filter((pl.col("d") > bin[1]) & (pl.col("d") < bin[0]))
        filtered_df_val = filtered_df.filter(pl.col("label") == "val")
        filtered_df_train = filtered_df.filter(pl.col("label") == "train")

        axes[i].scatter(
            filtered_df_train["intensity.prf.value_pred"],
            filtered_df_train["intensity.prf.value"],
            s=1.0,
            alpha=0.2,
            color="#005d5d",
            label="train",
        )
        axes[i].scatter(
            filtered_df_val["intensity.prf.value_pred"],
            filtered_df_val["intensity.prf.value"],
            s=1.0,
            alpha=0.2,
            color="#6929c4",
            label="val",
        )

        axes[i].set_xscale(scale)
        axes[i].set_yscale(scale)
        axes[i].legend()
        axes[i].grid()
        axes[i].set_title(f"bin: {bin[0]:.2f} - {bin[1]:.2f}")

    plt.suptitle("Intensity comparisons per resolution bin", fontsize=14)
    plt.tight_layout()

    wandb.log({"DIALS vs Network: per solution": wandb.Image(plt.gcf())})
    run.finish()
