"""Shared helpers for the reusable plotting modules.

These utilities are small building blocks (figure sizing, categorical
palettes, bin-axis construction, colormaps) that the individual plotting
modules in this package depend on. They are intentionally generic so any
analysis script can reuse them directly.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

CATEGORICAL_HEX_COLORS = {n: COLORS[:n] for n in range(1, len(COLORS) + 1)}


def get_palette(ids) -> dict:
    """Map a sequence of ids to colors.

    Uses the curated categorical palette when the number of ids is small
    enough, otherwise falls back to seaborn's `Dark2` palette.

    Args:
        ids: Iterable of hashable identifiers (e.g. run ids or labels).

    Returns:
        Mapping from each id to a color.
    """
    ids = list(ids)
    if len(ids) in CATEGORICAL_HEX_COLORS:
        hex_colors = CATEGORICAL_HEX_COLORS[len(ids)]
        return dict(zip(ids, hex_colors))
    c_palette = sns.color_palette("Dark2")
    return {id_: c for id_, c in zip(ids, c_palette)}


def set_figsize(
    fraction=0.6,
    ratio=0.6,
    textwidth_pt=452.9679,
    paper="a4",
):
    """Compute a typographically scaled figure size in inches.

    Args:
        fraction: Fraction of the text width the figure should span.
        ratio: Height-to-width ratio of the figure.
        textwidth_pt: Text width in points. If None, derived from `paper`.
        paper: Paper type used when `textwidth_pt` is None (`a4` or `letter`).

    Returns:
        Tuple of (width, height) in inches.
    """
    if textwidth_pt is None:
        if paper.lower() == "a4":
            textwidth_pt = 452.97  # ~6.26 in
        elif paper.lower() == "letter":
            textwidth_pt = 468.0  # ~6.48 in
        else:
            raise ValueError(f"Unknown paper type: {paper}")

    inches_per_pt = 1.0 / 72.27

    fig_width = textwidth_pt * inches_per_pt * fraction
    fig_height = fig_width * ratio

    return fig_width, fig_height


def set_mpl_fonts(base_pt):
    """Set matplotlib font sizes relative to a base point size.

    Args:
        base_pt: Base font size in points used for body text.
    """
    mpl.rcParams.update(
        {
            "font.size": base_pt,
            "axes.labelsize": base_pt,
            "xtick.labelsize": base_pt - 1,
            "ytick.labelsize": base_pt - 1,
            "legend.fontsize": base_pt - 1,
            "axes.titlesize": base_pt + 1,
        }
    )


def get_bins(edges: list) -> tuple[list, pl.LazyFrame]:
    """Build bin labels and an ordered bin-id frame from numeric edges.

    The labels include open-ended `<first` and `>last` bins so values
    outside the edge range still receive a label. The returned frame maps
    each label to a `bin_id`, ordered from the highest bin down, which is
    convenient for resolution axes.

    Args:
        edges: Sorted list of numeric bin edges.

    Returns:
        Tuple of (bin_labels, base_df) where base_df is a LazyFrame with
        `bin_labels` (Categorical) and `bin_id` (Int32) columns.
    """
    bin_labels = [f"{a} - {b}" for a, b in zip(edges[:-1], edges[1:])]
    bin_labels.insert(0, f"<{edges[0]}")
    bin_labels.append(f">{edges[-1]}")

    reversed_labels = list(reversed(bin_labels))

    base_df = pl.DataFrame(
        {
            "bin_labels": reversed_labels,
            "bin_id": list(range(len(reversed_labels))),
        },
        schema={
            "bin_labels": pl.Categorical,
            "bin_id": pl.Int32,
        },
    ).lazy()

    return bin_labels, base_df


def cubehelix_cmap(start=0.5, rot=-0.55, dark=0, light=0.8):
    """Return the project's standard cubehelix colormap object.

    Args:
        start: Cubehelix start hue.
        rot: Number of hue rotations.
        dark: Lightness of the darkest end (0 to 1).
        light: Lightness of the lightest end (0 to 1).

    Returns:
        A matplotlib colormap suitable for `palette`/`cmap` arguments.
    """
    return sns.cubehelix_palette(
        start=start, rot=rot, dark=dark, light=light, as_cmap=True
    )


def add_epoch_colorbar(ax, vmin, vmax, cmap=None, label="Epoch"):
    """Attach a colorbar spanning a value range to an axis.

    Args:
        ax: Axis to attach the colorbar to.
        vmin: Minimum value of the color scale.
        vmax: Maximum value of the color scale.
        cmap: Colormap object. Defaults to `cubehelix_cmap()`.
        label: Colorbar label.

    Returns:
        Tuple of (colorbar, ScalarMappable).
    """
    if cmap is None:
        cmap = cubehelix_cmap()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(label, rotation=90)
    return cbar, sm


def hide_odd_xticklabels(ax):
    """Hide every other x tick label to reduce crowding.

    Args:
        ax: Axis whose x tick labels should be thinned.
    """
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 != 0:
            label.set_visible(False)


def save_figure(
    fig,
    path,
    *,
    dpi: int = 300,
    transparent: bool = True,
    facecolor: str = "white",
    bbox_inches: str = "tight",
    pad_inches: float = 0.02,
    close: bool = True,
):
    """Save a figure with the project's standard options.

    Args:
        fig: Figure to save.
        path: Output path.
        dpi: Output resolution.
        transparent: Whether to use a transparent background.
        facecolor: Figure face color.
        bbox_inches: Bounding-box mode passed to `savefig`.
        pad_inches: Padding around the figure.
        close: Whether to close the figure after saving.
    """
    fig.savefig(
        path,
        dpi=dpi,
        transparent=transparent,
        facecolor=facecolor,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    if close:
        plt.close(fig)
