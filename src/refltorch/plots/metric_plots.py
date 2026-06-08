"""Line plots of metrics across bins, epochs, and models.

These functions are data-shape agnostic: callers pass already-binned
polars frames (with `bin_labels` columns) or plain (x, y) series and get
back a `(fig, ax)` pair to save or further customize.
"""

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from ._shared import get_palette, hide_odd_xticklabels, set_figsize
from .cmaps import get_cube_helix_map


def plot_binned_metric(
    df_map,
    base_df,
    *,
    series_ids,
    labels=None,
    palette=None,
    x_key="bin_id",
    y_key="mean_qi_var",
    x_label="bin",
    y_label="value",
    title=None,
    y_log=False,
):
    """Plot one binned metric line per series on a shared bin axis.

    Each series is left-joined onto `base_df` so every bin appears even
    when a series has no data for it.

    Args:
        df_map: Mapping from series id to a binned polars DataFrame that
            contains a `bin_labels` column plus `x_key`/`y_key`.
        base_df: Bin reference frame with `bin_labels` and `bin_id`
            columns, defining the bin order and tick labels.
        series_ids: Iterable of ids selecting and ordering the series.
        labels: Optional mapping from series id to legend label. Defaults
            to the string form of the id.
        palette: Optional mapping from series id to color. Defaults to
            `get_palette(series_ids)`.
        x_key: Column plotted on the x axis.
        y_key: Column plotted on the y axis.
        x_label: X axis label.
        y_label: Y axis label.
        title: Optional axis title.
        y_log: Whether to use a log y scale.

    Returns:
        Tuple of (fig, ax).
    """
    series_ids = list(series_ids)
    fig, ax = plt.subplots(figsize=set_figsize(ratio=0.6, textwidth_pt=452.9679))
    tick_labels = base_df["bin_labels"].to_list()
    ticks = base_df["bin_id"].to_list()

    if palette is None:
        palette = get_palette(series_ids)

    for sid in series_ids:
        df = base_df.join(df_map[sid], on="bin_labels", how="left").sort("bin_id")
        label = labels[sid] if labels is not None else str(sid)
        ax.plot(df[x_key], df[y_key], label=label, color=palette[sid])

    ax.set_xlabel(x_label)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel(y_label)

    hide_odd_xticklabels(ax)

    if title is not None:
        ax.set_title(title)
    if y_log:
        ax.set_yscale("log")
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    return fig, ax


def plot_binned_metric_by_epoch(
    df,
    base_df,
    *,
    epochs,
    x_key="bin_id",
    y_key="mean_qi_var",
    x_label="bin",
    y_label="value",
    title=None,
    y_log=False,
    x_tick_labels=None,
):
    """Plot one line per epoch of a binned metric, colored by epoch.

    Lines are colored with the project cubehelix ramp from the first to
    the last epoch.

    Args:
        df: Binned polars DataFrame with `epoch`, `bin_labels`, and the
            `x_key`/`y_key` columns.
        base_df: Bin reference frame with `bin_labels` and `bin_id`.
        epochs: Ordered iterable of epochs to draw (also sets line count).
        x_key: Column plotted on the x axis.
        y_key: Column plotted on the y axis.
        x_label: X axis label.
        y_label: Y axis label.
        title: Optional axis title.
        y_log: Whether to use a log y scale.
        x_tick_labels: Optional explicit tick labels; when given, ticks are
            placed at `base_df["bin_id"]`.

    Returns:
        Tuple of (fig, ax).
    """
    epochs = list(epochs)
    colors = get_cube_helix_map(len(epochs))

    fig, ax = plt.subplots(figsize=set_figsize())
    for c, e in zip(colors, epochs):
        df_ = df.filter(pl.col("epoch") == e)
        df_ = base_df.join(df_, on="bin_labels", how="left").sort("bin_id")
        ax.plot(df_[x_key], df_[y_key], c=c)

    ax.grid()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_tick_labels is not None:
        ax.set_xticks(base_df["bin_id"].to_list())
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    if title is not None:
        ax.set_title(title)
    if y_log:
        ax.set_yscale("log")
    return fig, ax


def plot_metric_over_epoch(
    series,
    *,
    ref_value=None,
    ref_label="DIALS",
    x_label="epoch",
    y_label="value",
    title=None,
):
    """Plot one metric line per series over epochs with an optional reference.

    Args:
        series: Iterable of `(label, x, y, color)` tuples, one per line.
        ref_value: Optional constant drawn as a horizontal reference line.
        ref_label: Legend label for the reference line.
        x_label: X axis label.
        y_label: Y axis label.
        title: Optional axis title.

    Returns:
        Tuple of (fig, ax).
    """
    fig, ax = plt.subplots(figsize=set_figsize())
    for label, x, y, color in series:
        ax.plot(x, y, label=label, color=color)
    if ref_value is not None:
        ax.axhline(ref_value, c="red", label=ref_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    return fig, ax
