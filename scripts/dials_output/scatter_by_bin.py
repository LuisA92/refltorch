"""Scatter plots of model vs DIALS quantities, faceted by resolution bin.

Usage:
    python scatter_by_bin.py \
        --preds-dir /path/to/predictions/ \
        --save-dir  /path/to/output/ \
        [--epoch 74] \
        [--edges 1.1 1.3 1.5 1.8 2.0 2.5 3.0]

If --epoch is omitted, uses the latest epoch found.
If --edges is omitted, uses the DIALS_EDGES_9B7C default.
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

DIALS_EDGES_9B7C = [
    1.1, 1.14, 1.18, 1.23, 1.30, 1.49, 1.64, 2.06, 2.97,
]

REQUIRED_COLS = [
    "epoch", "d",
    "qi_mean", "qi_var", "qbg_mean",
    "intensity.prf.value", "intensity.prf.variance", "background.mean",
]

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


def _make_bin_labels(edges: list[float]) -> list[str]:
    labels = [f"< {edges[0]:.2f}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo:.2f}-{hi:.2f}")
    labels.append(f"> {edges[-1]:.2f}")
    return labels


def _assign_bins(df: pl.DataFrame, edges: list[float]) -> pl.DataFrame:
    labels = _make_bin_labels(edges)
    return df.with_columns(
        pl.col("d")
        .cut(edges, labels=labels)
        .alias("d_bin")
    )


def _plot_scatter_grid(
    df: pl.DataFrame,
    bins: list[str],
    x: str, y: str,
    xlabel: str, ylabel: str,
    title: str,
    save_path: Path,
):
    n_bins = len(bins)
    ncols = min(4, n_bins)
    nrows = (n_bins + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
    )

    for idx, bname in enumerate(bins):
        ax = axes[idx // ncols][idx % ncols]
        sub = df.filter(pl.col("d_bin") == bname)

        # Filter positive values for log-log
        sub = sub.filter((pl.col(x) > 0) & (pl.col(y) > 0))

        if len(sub) == 0:
            ax.set_title(f"{bname}\n(n=0)")
            ax.set_visible(False)
            continue

        ax.scatter(
            sub[x].to_numpy(),
            sub[y].to_numpy(),
            s=2, alpha=0.1, c="black", edgecolors="none",
        )

        # x=y line
        vals = np.concatenate([sub[x].to_numpy(), sub[y].to_numpy()])
        lo, hi = vals[vals > 0].min() * 0.5, vals.max() * 2
        ax.plot([lo, hi], [lo, hi], c="red", alpha=0.5, lw=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"d: {bname}\n(n={len(sub):,})", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(n_bins, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(
        save_path, dpi=300, facecolor="white",
        bbox_inches="tight", pad_inches=0.05,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preds-dir", type=Path, required=True,
        help="Directory containing epoch_*/preds_*.parquet files",
    )
    parser.add_argument(
        "--save-dir", type=Path, required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch to plot (default: latest)",
    )
    parser.add_argument(
        "--edges", type=float, nargs="+", default=None,
        help="Resolution bin edges in Angstroms (default: DIALS_EDGES_9B7C)",
    )
    args = parser.parse_args()

    edges = args.edges if args.edges is not None else DIALS_EDGES_9B7C

    # Load predictions
    parquets = sorted(args.preds_dir.glob("**/preds_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {args.preds_dir}")
    print(f"Found {len(parquets)} parquet files")

    df = pl.read_parquet(parquets)

    # Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available: {df.columns}\n"
            f"Run add_d_to_parquet.py to add missing metadata columns."
        )

    # Select epoch
    epochs = sorted(df["epoch"].unique().to_list()) if "epoch" in df.columns else [0]
    epoch = args.epoch if args.epoch is not None else max(epochs)
    print(f"Plotting epoch {epoch} (available: {epochs})")
    df = df.filter(pl.col("epoch") == epoch)

    # Filter positive variance
    n_total = len(df)
    df = df.filter(pl.col("intensity.prf.variance") > 0)
    n_filtered = len(df)
    print(f"Filtered to {n_filtered:,} / {n_total:,} reflections (var > 0)")

    # Assign resolution bins
    df = _assign_bins(df, edges)
    bins = _make_bin_labels(edges)

    # Check which bins have data
    bins_with_data = df["d_bin"].unique().to_list()
    bins = [b for b in bins if b in bins_with_data]
    print(f"Resolution bins with data: {len(bins)}")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    for sdef in SCATTER_DEFS:
        save_path = args.save_dir / f"scatter_{sdef['tag']}_by_dbin_epoch_{epoch}.png"
        print(f"  Plotting {sdef['tag']}...")
        _plot_scatter_grid(
            df=df,
            bins=bins,
            x=sdef["x"],
            y=sdef["y"],
            xlabel=sdef["xlabel"],
            ylabel=sdef["ylabel"],
            title=f"{sdef['xlabel']} vs {sdef['ylabel']} | Epoch {epoch}",
            save_path=save_path,
        )
        print(f"    Saved: {save_path}")

    print("Done")


if __name__ == "__main__":
    main()
