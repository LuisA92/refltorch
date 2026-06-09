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


def _shared_limits(panels, lo=1, hi=99) -> tuple[float | None, float | None]:
    """Robust (lo, hi) percentile limits pooled across all panels' values."""
    finite = []
    for *_, c in panels:
        arr = np.asarray(c, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite.append(arr)
    if not finite:
        return None, None
    pooled = np.concatenate(finite)
    return float(np.percentile(pooled, lo)), float(np.percentile(pooled, hi))


def plot_hexbin_detector_grid(
    panels,
    *,
    reduce=np.mean,
    gridsize=60,
    mincnt=1,
    cmap=None,
    vmin=None,
    vmax=None,
    norm=None,
    ncols=None,
    xlabel="detector x (px)",
    ylabel="detector y (px)",
    clabel=None,
    suptitle=None,
    figsize=None,
    panel_size=(4.0, 3.6),
):
    """Grid of detector hexbins sharing one color scale and colorbar.

    Every panel is drawn with the same `cmap` and color limits so the
    hexbins are directly comparable, with a single colorbar for the figure.
    Pass a diverging `cmap` with symmetric `vmin`/`vmax` for difference maps.

    Args:
        panels: Iterable of `(title, x, y, c)` tuples, one per subplot.
        reduce: Reduction applied to `c` within each hexagon (e.g. `np.mean`,
            `np.median`, `np.min`, `np.max`).
        gridsize: Number of hexagons across the x axis.
        mincnt: Minimum reflections per hexagon for it to be drawn.
        cmap: Colormap shared by all panels.
        vmin: Lower color limit. Pooled robust limit when None (and no
            `norm`).
        vmax: Upper color limit. Pooled robust limit when None (and no
            `norm`).
        norm: Optional matplotlib norm shared by all panels; overrides
            `vmin`/`vmax`.
        ncols: Number of columns. Defaults to `min(len(panels), 3)`.
        xlabel: X axis label (applied to every panel).
        ylabel: Y axis label (applied to every panel).
        clabel: Shared colorbar label.
        suptitle: Optional figure title.
        figsize: Figure size in inches. Derived from `panel_size` when None.
        panel_size: Per-panel `(width, height)` used to size the figure.

    Returns:
        Tuple of (fig, axes), where axes is the 2D array from `subplots`.
    """
    panels = list(panels)
    n = len(panels)
    if ncols is None:
        ncols = min(n, 3) if n else 1
    nrows = (n + ncols - 1) // ncols if n else 1

    if norm is None and (vmin is None or vmax is None):
        lo, hi = _shared_limits(panels)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, layout="constrained"
    )
    flat = axes.ravel()

    hb = None
    for ax, (title, x, y, c) in zip(flat, panels):
        kwargs = dict(
            C=c,
            reduce_C_function=reduce,
            gridsize=gridsize,
            mincnt=mincnt,
            cmap=cmap,
        )
        if norm is not None:
            kwargs["norm"] = norm
        else:
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
        hb = ax.hexbin(x, y, **kwargs)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

    for ax in flat[n:]:
        ax.set_visible(False)

    if hb is not None:
        cbar = fig.colorbar(hb, ax=axes.ravel().tolist())
        if clabel is not None:
            cbar.set_label(clabel, rotation=90)
    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig, axes
