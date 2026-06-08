"""Training and validation loss curves.

Functions take long-format loss frames and return a `(fig, ax)` pair so the
caller controls saving.
"""

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from ._shared import get_palette, set_figsize


def plot_loss_gap(loss_gap_df, *, metric, hue_key="label", palette="Dark2", title=None):
    """Plot a train/val gap metric over epochs, one line per series.

    Args:
        loss_gap_df: Long-format frame with `epoch`, `hue_key`, and the
            `metric` column (e.g. `kl_gap`, `nll_gap`, `elbo_gap`).
        metric: Column holding the gap values to plot.
        hue_key: Column used to color the lines.
        palette: Seaborn palette name or mapping passed to `sns.lineplot`.
        title: Optional axis title; defaults to `Train/Val {metric} gap`.

    Returns:
        Tuple of (fig, ax).
    """
    fig, ax = plt.subplots(figsize=set_figsize())
    sns.lineplot(
        data=loss_gap_df,
        x="epoch",
        y=metric,
        hue=hue_key,
        palette=palette,
        ax=ax,
    )
    ax.grid()
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_title(title if title is not None else f"Train/Val {metric} gap")
    return fig, ax


def plot_train_val_metric(
    long_loss_df,
    *,
    variables,
    palette=None,
    dashes=None,
    hue_key="label",
    style_key="Stage",
    title=None,
    ylabel=None,
    ymin=None,
    ymax=None,
    linewidth=2,
):
    """Plot a single loss metric for train and val stages over epochs.

    Lines are colored by `hue_key` and styled (solid vs dashed) by
    `style_key`, so train and val curves of the same series share a color.

    Args:
        long_loss_df: Long-format frame with `epoch`, `value`, `variable`,
            `hue_key`, and `style_key` columns.
        variables: The two `variable` values to keep (train and val of one
            metric, e.g. `("train nll", "val nll")`).
        palette: Optional color mapping. Defaults to a palette over the
            unique `hue_key` values.
        dashes: Optional style-to-dash mapping. Defaults to dashed val,
            solid train.
        hue_key: Column used to color the lines.
        style_key: Column used to set line style (train vs val).
        title: Optional axis title.
        ylabel: Optional y axis label.
        ymin: Lower y limit. Defaults to the minimum plotted value.
        ymax: Optional upper y limit.
        linewidth: Line width.

    Returns:
        Tuple of (fig, ax).
    """
    if palette is None:
        palette = get_palette(long_loss_df[hue_key].unique().to_list())
    if dashes is None:
        dashes = {"val": (2, 1), "train": ""}

    plot_df = long_loss_df.filter(pl.col("variable").is_in(list(variables)))
    if ymin is None:
        ymin = plot_df["value"].min()

    fig, ax = plt.subplots(figsize=set_figsize())
    sns.lineplot(
        data=plot_df,
        x="epoch",
        y="value",
        hue=hue_key,
        style=style_key,
        palette=palette,
        dashes=dashes,
        linewidth=linewidth,
        ax=ax,
    )

    if title is not None:
        ax.set_title(title)
    ax.grid()
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)
    ax.set_ylim(ymin=ymin)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    return fig, ax
