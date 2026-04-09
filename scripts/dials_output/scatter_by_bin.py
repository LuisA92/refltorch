"""Scatter plots of model vs DIALS quantities, faceted by group_label.

Uses the group_label column written during prediction (same resolution bins
the model trained with) rather than re-binning.  Falls back to binning by
d-spacing if group_label is not in the parquets.

Usage:
    python scatter_by_bin.py \
        --run-dir /path/to/run_dir/ \
        --save-dir /path/to/output/ \
        [--epoch 74]
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from refltorch.io import load_config

SCATTER_DEFS = [
    {
        "x": "qi_mean",
        "y": "intensity.prf.value",
        "xlabel": "Model I",
        "ylabel": "DIALS I",
        "tag": "intensity",
    },
    {
        "x": "qbg_mean",
        "y": "background.mean",
        "xlabel": "Model bg",
        "ylabel": "DIALS bg",
        "tag": "background",
    },
    {
        "x": "qi_var",
        "y": "intensity.prf.variance",
        "xlabel": "Model var(I)",
        "ylabel": "DIALS var(I)",
        "tag": "variance",
    },
]


def _plot_scatter_grid(
    df: pl.DataFrame,
    bin_col: str,
    bin_values: list,
    bin_labels: list[str],
    x: str, y: str,
    xlabel: str, ylabel: str,
    title: str,
    save_path: Path,
    wilson_means: dict | None = None,
):
    n_bins = len(bin_values)
    ncols = min(4, n_bins)
    nrows = (n_bins + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3 * ncols, 3 * nrows),
        squeeze=False,
    )

    for idx, (bval, blabel) in enumerate(zip(bin_values, bin_labels)):
        ax = axes[idx // ncols][idx % ncols]
        sub = df.filter(pl.col(bin_col) == bval)

        # Filter positive values for log-log
        sub = sub.filter((pl.col(x) > 0) & (pl.col(y) > 0))

        if len(sub) == 0:
            ax.set_title(f"{blabel}\n(n=0)")
            ax.set_visible(False)
            continue

        ax.scatter(
            sub[x].to_numpy(),
            sub[y].to_numpy(),
            s=2, alpha=0.1, c="black", edgecolors="none",
        )

        # x=y line
        vals = np.concatenate([sub[x].to_numpy(), sub[y].to_numpy()])
        pos = vals[vals > 0]
        lo, hi = pos.min() * 0.5, pos.max() * 2
        ax.plot([lo, hi], [lo, hi], c="red", alpha=0.5, lw=1)

        # Wilson expected mean as vertical line
        subtitle = f"n={len(sub):,}"
        if wilson_means is not None and bval in wilson_means:
            wm = wilson_means[bval]
            ax.axvline(wm, color="blue", alpha=0.6, lw=1, ls="--")
            subtitle += f" | E[I]={wm:.1f}"

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{blabel}\n{subtitle}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(n_bins, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(
        save_path, dpi=150, facecolor="white",
        bbox_inches="tight", pad_inches=0.05,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to run directory containing run_metadata.yaml",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=None,
        help="Output directory for plots (default: <wandb_log>/plots/scatter_by_bin)",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch to plot (default: latest)",
    )
    args = parser.parse_args()

    # Resolve paths from run_metadata.yaml
    run_metadata = list(args.run_dir.glob("run_metadata.yaml"))[0]
    config = load_config(run_metadata)
    wandb_log = Path(config["wandb"]["log_dir"]).parent
    preds_dir = wandb_log / "predictions"

    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = wandb_log / "plots" / "scatter_by_bin"

    # Load predictions
    parquets = sorted(preds_dir.glob("**/preds_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {preds_dir}")
    print(f"Found {len(parquets)} parquet files")

    df = pl.read_parquet(parquets)
    print(f"Columns: {df.columns}")

    # Select epoch
    if "epoch" in df.columns:
        epochs = sorted(df["epoch"].unique().to_list())
        epoch = args.epoch if args.epoch is not None else max(epochs)
        print(f"Plotting epoch {epoch} (available: {epochs})")
        df = df.filter(pl.col("epoch") == epoch)
    else:
        epoch = 0

    # Filter positive variance
    if "intensity.prf.variance" in df.columns:
        n_total = len(df)
        df = df.filter(pl.col("intensity.prf.variance") > 0)
        print(f"Filtered to {len(df):,} / {n_total:,} reflections (var > 0)")

    # Get bin column
    if "group_label" in df.columns:
        bin_col = "group_label"
        groups = sorted(df[bin_col].unique().to_list())
        if "d" in df.columns:
            bin_labels = []
            for g in groups:
                sub_d = df.filter(pl.col(bin_col) == g)["d"]
                bin_labels.append(f"bin {g} (d: {sub_d.min():.2f}-{sub_d.max():.2f})")
        else:
            bin_labels = [f"bin {g}" for g in groups]
        print(f"Using group_label with {len(groups)} bins")
    elif "d" in df.columns:
        # Fallback: quantile bins
        n_bins = 10
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.quantile(df["d"].to_numpy(), quantiles).tolist()
        bin_labels = [f"< {edges[0]:.2f}"]
        for lo, hi in zip(edges[:-1], edges[1:]):
            bin_labels.append(f"{lo:.2f}-{hi:.2f}")
        bin_labels.append(f"> {edges[-1]:.2f}")
        df = df.with_columns(
            pl.col("d").cut(edges, labels=bin_labels).alias("d_bin")
        )
        bin_col = "d_bin"
        groups = bin_labels
        print(f"Fallback: binned d into {len(groups)} quantile shells")
    else:
        raise ValueError("Need 'group_label' or 'd' column in parquets")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Compute Wilson expected mean per bin from tau_per_refl: E[I] = 1/tau
    wilson_means = None
    if "tau_per_refl" in df.columns and bin_col == "group_label":
        wilson_means = {}
        for g in groups:
            tau = df.filter(pl.col(bin_col) == g)["tau_per_refl"].mean()
            if tau is not None and tau > 0:
                wilson_means[g] = 1.0 / tau
        print(f"Wilson E[I] per bin: {[f'{wilson_means.get(g, 0):.1f}' for g in groups]}")

    # Filter scatter defs to available columns
    for sdef in SCATTER_DEFS:
        if sdef["x"] not in df.columns or sdef["y"] not in df.columns:
            print(f"  Skipping {sdef['tag']}: missing {sdef['x']} or {sdef['y']}")
            continue

        # Only show Wilson mean on intensity plot
        wm = wilson_means if sdef["tag"] == "intensity" else None

        save_path = save_dir / f"scatter_{sdef['tag']}_by_bin_epoch_{epoch}.png"
        print(f"  Plotting {sdef['tag']}...")
        _plot_scatter_grid(
            df=df,
            bin_col=bin_col,
            bin_values=groups,
            bin_labels=bin_labels,
            x=sdef["x"],
            y=sdef["y"],
            xlabel=sdef["xlabel"],
            ylabel=sdef["ylabel"],
            title=f"{sdef['xlabel']} vs {sdef['ylabel']} | Epoch {epoch}",
            save_path=save_path,
            wilson_means=wm,
        )
        print(f"    Saved: {save_path}")

    print("Done")


if __name__ == "__main__":
    main()
