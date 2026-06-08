"""Correlation curves across bins, colored by epoch.

Each call plots a single correlation column (e.g. intensity, background,
variance) across a categorical bin axis with one line per epoch and an
attached epoch colorbar.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ._shared import cubehelix_cmap, hide_odd_xticklabels, set_figsize


def plot_correlation_vs_bin(
    df,
    *,
    y_key,
    x_key="d_bins",
    hue_key="epoch",
    cmap=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xtick_rotation=50,
):
    """Plot a correlation column across bins, one line per epoch.

    Args:
        df: Frame with `x_key`, `y_key`, and `hue_key` columns.
        y_key: Correlation column plotted on the y axis.
        x_key: Categorical bin column plotted on the x axis.
        hue_key: Column mapped to the epoch color ramp.
        cmap: Colormap for the epoch ramp. Defaults to `cubehelix_cmap()`.
        xlabel: Optional x axis label.
        ylabel: Optional y axis label.
        title: Optional axis title.
        xtick_rotation: Rotation in degrees for the x tick labels.

    Returns:
        Tuple of (fig, ax).
    """
    if cmap is None:
        cmap = cubehelix_cmap()
    norm = mpl.colors.Normalize(vmin=df[hue_key].min(), vmax=df[hue_key].max())
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=set_figsize())
    sns.lineplot(
        data=df,
        x=x_key,
        y=y_key,
        hue=hue_key,
        hue_norm=norm,
        palette=cmap,
        legend=False,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    ax.grid()

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Epoch", rotation=90)
    hide_odd_xticklabels(ax)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return fig, ax
