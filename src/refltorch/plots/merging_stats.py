"""Merging-statistic curves versus resolution, colored by epoch.

Each call plots a single statistic (CChalf, Rpim, I/sig(I), CCanom, ...)
across resolution bins with one line per epoch and an attached epoch
colorbar, optionally overlaying a reference (e.g. DIALS) curve.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ._shared import cubehelix_cmap, hide_odd_xticklabels, set_figsize


def plot_stat_vs_resolution(
    df,
    *,
    y_key,
    x_key="resolution",
    hue_key="epoch",
    ref=None,
    ref_label="DIALS",
    xlabel="Resolution bin",
    ylabel=None,
    title=None,
    yticks=None,
    cmap=None,
    fraction=0.6,
):
    """Plot a merging statistic across resolution bins, one line per epoch.

    Args:
        df: Long-format frame with `x_key`, `y_key`, and `hue_key` columns.
        y_key: Statistic column plotted on the y axis.
        x_key: Resolution-bin column plotted on the x axis.
        hue_key: Column mapped to the epoch color ramp.
        ref: Optional reference series plotted against bin index (e.g.
            `ref_df[y_key]`); drawn in red and labeled `ref_label`.
        ref_label: Legend label for the reference curve.
        xlabel: X axis label.
        ylabel: Optional y axis label.
        title: Optional axis title.
        yticks: Optional explicit y tick positions.
        cmap: Colormap for the epoch ramp. Defaults to `cubehelix_cmap()`.
        fraction: Text-width fraction passed to `set_figsize`.

    Returns:
        Tuple of (fig, ax).
    """
    if cmap is None:
        cmap = cubehelix_cmap()
    norm = mpl.colors.Normalize(vmin=df[hue_key].min(), vmax=df[hue_key].max())
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=set_figsize(fraction=fraction))
    sns.lineplot(
        data=df,
        x=x_key,
        y=y_key,
        hue=hue_key,
        palette=cmap,
        hue_norm=norm,
        legend=False,
        ax=ax,
    )
    if ref is not None:
        ax.plot(ref, label=ref_label, color="red")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(yticks)
    hide_odd_xticklabels(ax)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Epoch", rotation=90)
    if ref is not None:
        ax.legend()
    ax.grid(alpha=0.5)
    if title is not None:
        ax.set_title(title)
    return fig, ax
