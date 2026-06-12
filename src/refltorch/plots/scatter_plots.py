"""Scatter plots comparing two quantities against the identity line.

Useful for model-vs-reference comparisons where the y=x line marks perfect
agreement. Scales, limits, and the identity-line extent are all caller
controlled so the same primitive covers linear, log-log, and symlog views.
"""

import matplotlib.pyplot as plt


def plot_scatter_identity(
    x,
    y,
    *,
    identity=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    xlim=None,
    ylim=None,
    figsize=(8, 8),
    color="black",
    alpha=0.15,
    s=3,
    identity_color="red",
    identity_alpha=0.5,
    grid_alpha=0.3,
):
    """Scatter `x` against `y` with an optional y=x reference line.

    Args:
        x: Values for the x axis.
        y: Values for the y axis.
        identity: Optional `(lo, hi)` endpoints for the identity line. No
            line is drawn when None.
        xlabel: X axis label.
        ylabel: Y axis label.
        title: Optional axis title.
        xscale: X axis scale (`linear`, `log`, or `symlog`).
        yscale: Y axis scale (`linear`, `log`, or `symlog`).
        xlim: Optional `(left, right)` x limits, applied after the scale is
            set so a None bound autoscales on the final axis. Either bound may
            be None to leave that side automatic.
        ylim: Optional `(bottom, top)` y limits, applied after the scale is
            set so a None bound autoscales on the final axis. Either bound may
            be None to leave that side automatic.
        figsize: Figure size in inches.
        color: Marker color.
        alpha: Marker alpha.
        s: Marker size.
        identity_color: Color of the identity line.
        identity_alpha: Alpha of the identity line.
        grid_alpha: Grid alpha.

    Returns:
        Tuple of (fig, ax).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, alpha=alpha, c=color, s=s, edgecolors="none")
    if identity is not None:
        lo, hi = identity
        ax.plot([lo, hi], [lo, hi], c=identity_color, alpha=identity_alpha)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xscale != "linear":
        ax.set_xscale(xscale)
    if yscale != "linear":
        ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=grid_alpha)
    return fig, ax
