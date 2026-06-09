"""Hexbin maps of per-reflection statistics over the detector face.

Bins reflections by their detector (x, y) position and colors each hexagon
by a reduction (mean by default) of a per-reflection statistic. Useful for
spotting spatial structure in background, intensity, or model-minus-DIALS
residuals that a per-reflection scatter would hide.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_hexbin_detector(
    x,
    y,
    c,
    *,
    reduce=np.mean,
    gridsize=60,
    mincnt=1,
    cmap=None,
    vmin=None,
    vmax=None,
    norm=None,
    xlabel="detector x (px)",
    ylabel="detector y (px)",
    clabel=None,
    title=None,
    figsize=(7, 6),
):
    """Hexbin a per-reflection statistic over detector positions.

    Args:
        x: Detector x position per reflection.
        y: Detector y position per reflection.
        c: Per-reflection statistic aggregated within each hexagon.
        reduce: Reduction applied to `c` within each hexagon (e.g. `np.mean`
            or `np.median`).
        gridsize: Number of hexagons across the x axis.
        mincnt: Minimum reflections per hexagon for it to be drawn.
        cmap: Colormap. For residual maps pass a diverging map (e.g.
            `RdBu_r`) with symmetric `vmin`/`vmax`.
        vmin: Lower color limit. Ignored when `norm` is given.
        vmax: Upper color limit. Ignored when `norm` is given.
        norm: Optional matplotlib norm (e.g. `LogNorm`); overrides
            `vmin`/`vmax`.
        xlabel: X axis label.
        ylabel: Y axis label.
        clabel: Colorbar label.
        title: Optional axis title.
        figsize: Figure size in inches.

    Returns:
        Tuple of (fig, ax).
    """
    hexbin_kwargs = dict(
        C=c,
        reduce_C_function=reduce,
        gridsize=gridsize,
        mincnt=mincnt,
        cmap=cmap,
    )
    if norm is not None:
        hexbin_kwargs["norm"] = norm
    else:
        hexbin_kwargs["vmin"] = vmin
        hexbin_kwargs["vmax"] = vmax

    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(x, y, **hexbin_kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(hb, ax=ax)
    if clabel is not None:
        cbar.set_label(clabel, rotation=90)
    return fig, ax
