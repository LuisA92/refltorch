"""
plotsizes.py
============
Typographically correct figure sizing and font scaling
for LaTeX documents using the specified geometry.

Default geometry (letter):
    letterpaper  = 8.5 x 11 in
    left=1.25in, right=1.0in  → textwidth = 6.25in
    top=0.9in, bottom=1.0in, headheight=14pt → textheight ≈ 9.1in

Use ``set_paper("a4")`` or ``paper_context("b5")`` to switch geometry.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib.pyplot as plt

_MM = 1 / 25.4  # millimetres → inches


# ============================================================
#  Paper geometry
# ============================================================

@dataclass(frozen=True)
class PaperGeometry:
    """Page geometry for a LaTeX paper size (all values in inches)."""

    page_width: float
    page_height: float
    margin_left: float
    margin_right: float
    margin_top: float
    margin_bottom: float
    headheight_pt: float = 14.0  # \headheight in points
    headsep_pt: float = 0.0     # \headsep in points
    base_font_pt: float = 11.0  # \normalsize in points

    @property
    def text_width(self) -> float:
        return self.page_width - self.margin_left - self.margin_right

    @property
    def text_height(self) -> float:
        return (
            self.page_height
            - self.margin_top
            - self.margin_bottom
            - (self.headheight_pt + self.headsep_pt) / 72.27
        )


PAPER_SIZES: dict[str, PaperGeometry] = {
    # letterpaper — matches the original hardcoded geometry
    "letter": PaperGeometry(
        page_width=8.5,
        page_height=11.0,
        margin_left=1.25,
        margin_right=1.0,
        margin_top=0.9,
        margin_bottom=1.0,
    ),
    # A4 — 210 × 297 mm, 25 mm symmetric margins
    "a4": PaperGeometry(
        page_width=210 * _MM,
        page_height=297 * _MM,
        margin_left=25 * _MM,
        margin_right=25 * _MM,
        margin_top=25 * _MM,
        margin_bottom=25 * _MM,
    ),
    # ISO B5 — 176 × 250 mm
    "b5": PaperGeometry(
        page_width=176 * _MM,
        page_height=250 * _MM,
        margin_left=24 * _MM,
        margin_right=24 * _MM,
        margin_top=24 * _MM,
        margin_bottom=26 * _MM,
        headheight_pt=14.0,
        headsep_pt=10.0,
        base_font_pt=10.0,
    ),
}

# ============================================================
#  Active geometry (module-level globals, updated by set_paper)
# ============================================================

_active_paper: str = "letter"

# Initialise from the default paper
TEXT_WIDTH: float = PAPER_SIZES["letter"].text_width    # 6.25 in
TEXT_HEIGHT: float = PAPER_SIZES["letter"].text_height  # ~9.1 in


def set_paper(name: str) -> PaperGeometry:
    """Set the active paper size, updating TEXT_WIDTH, TEXT_HEIGHT, and FONT_SIZES.

    Args:
        name: one of ``"letter"``, ``"a4"``, ``"b5"`` (or any key in
              ``PAPER_SIZES``).

    Returns:
        The selected :class:`PaperGeometry`.

    Example::

        plots.set_paper("b5")
        fig, ax = plt.subplots(figsize=plots.figsize("half"))
    """
    global TEXT_WIDTH, TEXT_HEIGHT, _active_paper, FONT_SIZES
    geo = PAPER_SIZES[name]
    TEXT_WIDTH = geo.text_width
    TEXT_HEIGHT = geo.text_height
    _active_paper = name
    FONT_SIZES = _make_font_sizes(geo.base_font_pt)
    return geo


@contextmanager
def paper_context(name: str):
    """Context manager: temporarily switch to a different paper geometry.

    Restores the previous geometry (including font sizes) on exit.

    Example::

        with plots.paper_context("b5"):
            fig, ax = plt.subplots(figsize=plots.figsize("full"))
            ...
    """
    global TEXT_WIDTH, TEXT_HEIGHT, _active_paper, FONT_SIZES
    saved = (_active_paper, TEXT_WIDTH, TEXT_HEIGHT, FONT_SIZES)
    set_paper(name)
    try:
        yield PAPER_SIZES[name]
    finally:
        _active_paper, TEXT_WIDTH, TEXT_HEIGHT, FONT_SIZES = saved

# ============================================================
#  Font sizes — derived from the active paper's base_font_pt
#
#  All sizes are offsets from \normalsize so they stay
#  proportional regardless of paper / font size choice.
#
#  Offsets calibrated against 11pt Minion Pro on letter paper:
#    title      ≈ normalsize + 0   (labels match body text)
#    axis_label ≈ normalsize − 1
#    tick_label ≈ normalsize − 2.5
#    legend     ≈ normalsize − 2.5  (slightly below tick labels)
#    annotation ≈ normalsize − 3
#    suptitle   ≈ normalsize + 1
# ============================================================


@dataclass(frozen=True)
class FigureFontSizes:
    """Font sizes in pt for matplotlib figures."""

    title: float
    axis_label: float
    tick_label: float
    legend: float
    annotation: float
    suptitle: float


# Offsets from base_font_pt for each figure-width preset
_FONT_OFFSETS: dict[str, dict[str, float]] = {
    # legend is 0.5pt below tick_label — it's supplementary, not structural
    "full":    {"title":  0.0, "axis_label": -1.0, "tick_label": -2.5, "legend": -3.0, "annotation": -3.0, "suptitle": +1.0},
    "half":    {"title": -1.0, "axis_label": -2.0, "tick_label": -3.0, "legend": -3.5, "annotation": -3.5, "suptitle":  0.0},
    "quarter": {"title": -2.0, "axis_label": -2.5, "tick_label": -3.5, "legend": -4.0, "annotation": -4.0, "suptitle": -1.0},
    "margin":  {"title": -2.0, "axis_label": -3.0, "tick_label": -4.0, "legend": -4.5, "annotation": -4.5, "suptitle": -2.0},
}


def _make_font_sizes(base_pt: float) -> dict[str, FigureFontSizes]:
    """Build font-size presets scaled to *base_pt* (\normalsize)."""
    return {
        preset: FigureFontSizes(**{k: base_pt + v for k, v in offsets.items()})
        for preset, offsets in _FONT_OFFSETS.items()
    }


# Active presets — updated by set_paper()
FONT_SIZES: dict[str, FigureFontSizes] = _make_font_sizes(
    PAPER_SIZES["letter"].base_font_pt
)



def figsize(
    width: str | float = "full",
    aspect: float = 0.618,  # golden ratio
    height: float | None = None,
) -> tuple[float, float]:
    """Compute figure dimensions in inches.

    Args:
        width: "full", "half", "third", "quarter", or a float (fraction of textwidth)
        aspect: height / width ratio (ignored if height is given)
        height: explicit height in inches

    Returns:
        (width_in, height_in)
    """
    fractions = {
        "full": 1.0,
        "half": 0.48,  # slightly under 0.5 for gap
        "third": 0.31,
        "quarter": 0.23,
        "margin": 0.35,
    }

    if isinstance(width, str):
        w = TEXT_WIDTH * fractions[width]
    else:
        w = TEXT_WIDTH * width

    h = height if height is not None else w * aspect
    return (w, h)


def fontsizes(width: str | float = "full") -> FigureFontSizes:
    """Get appropriate font sizes for the given figure width.

    Args:
        width: "full", "half", "quarter", "margin", or a float.
               Floats are mapped to the nearest preset.
    """
    if isinstance(width, str):
        return FONT_SIZES.get(width, FONT_SIZES["full"])

    # Map float to nearest preset
    if width >= 0.7:
        return FONT_SIZES["full"]
    elif width >= 0.4:
        return FONT_SIZES["half"]
    elif width >= 0.25:
        return FONT_SIZES["quarter"]
    else:
        return FONT_SIZES["margin"]


def apply_fontsizes(
    width: str | float = "full",
    ax=None,
    fig=None,
):
    """Apply correct font sizes to a figure/axes.

    Args:
        width: figure width preset or fraction
        ax: single axes or array of axes (if None, uses current)
        fig: figure (if None, uses current)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    sizes = fontsizes(width)

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.get_axes()

    # Normalize to iterable
    axes = np.atleast_1d(ax).ravel()

    for a in axes:
        a.title.set_fontsize(sizes.title)
        a.xaxis.label.set_fontsize(sizes.axis_label)
        a.yaxis.label.set_fontsize(sizes.axis_label)
        label_color = plt.rcParams.get("xtick.labelcolor", "inherit")
        if label_color == "inherit":
            label_color = plt.rcParams.get("text.color", "black")
        a.tick_params(axis="both", labelsize=sizes.tick_label, labelcolor=label_color)

        legend = a.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(sizes.legend)

    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(sizes.suptitle)


def rc_fontsizes(width: str | float = "full") -> dict:
    """Return an rcParams dict for the given width.

    Usage:
        with plt.rc_context(rc_fontsizes("half")):
            fig, ax = plt.subplots()
            ...
    """
    sizes = fontsizes(width)
    return {
        "axes.titlesize": sizes.title,
        "axes.labelsize": sizes.axis_label,
        "xtick.labelsize": sizes.tick_label,
        "ytick.labelsize": sizes.tick_label,
        "legend.fontsize": sizes.legend,
        "figure.titlesize": sizes.suptitle,
    }


# ============================================================
#  Default padding (inches) for panel layouts
# ============================================================
PAD_BOTTOM = 0.40  # room for tick labels
PAD_TOP_1LINE = 0.35  # room for 1-line titles
PAD_TOP_2LINE = 0.55  # room for 2-line titles
PAD_LEFT = 0.45  # room for y-axis tick labels
PAD_RIGHT = 0.10  # small safety margin on the right
AX_HEIGHT = 1.0  # default axes height (= width when aspect="equal")


def panel_figsize(
    pad_top: float = PAD_TOP_1LINE,
    ax_height: float = AX_HEIGHT,
    pad_bottom: float = PAD_BOTTOM,
) -> tuple[float, float]:
    """Figure size for panel layouts at full text width.

    Always returns TEXT_WIDTH as the figure width.  Height is computed
    from the desired axes height plus padding for titles and tick labels.

    Use with ``panel_adjust`` and ``save(..., tight=False)`` to guarantee
    identical panel sizes across figures with different panel counts.

    Args:
        pad_top: inches reserved above the axes (title space).
            Use ``PAD_TOP_2LINE`` for two-line titles.
        ax_height: desired axes box height in inches.
        pad_bottom: inches reserved below the axes (tick labels).

    Returns:
        (width_in, height_in) tuple suitable for ``plt.subplots(figsize=...)``.
    """
    return (TEXT_WIDTH, ax_height + pad_top + pad_bottom)


def panel_adjust(
    fig,
    pad_top: float = PAD_TOP_1LINE,
    pad_bottom: float = PAD_BOTTOM,
    pad_left: float = PAD_LEFT,
    pad_right: float = PAD_RIGHT,
    wspace: float = 0.35,
):
    """Apply fixed subplot margins so axes height is deterministic.

    Converts absolute padding (inches) to figure-fraction, ensuring
    panels have a consistent size regardless of content (title lines,
    number of panels, etc.).

    Args:
        fig: matplotlib Figure.
        pad_top: inches reserved for titles (use ``PAD_TOP_2LINE`` for two-line).
        pad_bottom: inches reserved for tick labels.
        pad_left: inches reserved on the left (y-axis labels).
        pad_right: inches reserved on the right.
        wspace: horizontal space between subplots (fraction of axes width).
    """
    fig_w, fig_h = fig.get_size_inches()
    fig.subplots_adjust(
        left=pad_left / fig_w,
        right=1 - pad_right / fig_w,
        wspace=wspace,
        bottom=pad_bottom / fig_h,
        top=1 - pad_top / fig_h,
    )


def _measure_axes_padding(fig, renderer, exclude=None):
    """Measure the decoration extent beyond each axes box (inches).

    Returns (need_left, need_right, need_bottom, need_top) — the maximum
    space that tick labels, axis labels, and titles require on each side
    across all axes (excluding those in *exclude*).
    """
    fig_w, fig_h = fig.get_size_inches()
    dpi = fig.dpi

    need_left = 0.0
    need_right = 0.0
    need_bottom = 0.0
    need_top = 0.0
    exclude = set(exclude or [])

    for ax in fig.get_axes():
        if ax in exclude:
            continue
        tight_bb = ax.get_tightbbox(renderer)
        if tight_bb is None:
            continue

        # Tight bbox in inches (from figure origin)
        t_left = tight_bb.x0 / dpi
        t_right = tight_bb.x1 / dpi
        t_bottom = tight_bb.y0 / dpi
        t_top = tight_bb.y1 / dpi

        # Axes box in inches (from figure origin)
        pos = ax.get_position()
        a_left = pos.x0 * fig_w
        a_right = pos.x1 * fig_w
        a_bottom = pos.y0 * fig_h
        a_top = pos.y1 * fig_h

        # Space decorations need beyond the axes box on each side
        need_left = max(need_left, a_left - t_left)
        need_right = max(need_right, t_right - a_right)
        need_bottom = max(need_bottom, a_bottom - t_bottom)
        need_top = max(need_top, t_top - a_top)

    return need_left, need_right, need_bottom, need_top


def auto_adjust(fig, pad_extra: float = 0.05, wspace: float | None = None):
    """Automatically compute and apply subplot margins from rendered content.

    Measures the actual extent of tick labels, axis labels, and titles
    on each side, then sets subplot margins so nothing is clipped.
    Use with ``save(..., tight=False)`` for deterministic figure sizing.

    Args:
        fig: matplotlib Figure.
        pad_extra: extra safety margin in inches added to each side.
        wspace: horizontal space between subplots (passed through if set).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w, fig_h = fig.get_size_inches()

    need_left, need_right, need_bottom, need_top = _measure_axes_padding(
        fig, renderer
    )

    # Guarantee room for a title even if it hasn't been set yet
    need_top = max(need_top, PAD_TOP_1LINE)

    kwargs = dict(
        left=(need_left + pad_extra) / fig_w,
        right=1 - (need_right + pad_extra) / fig_w,
        bottom=(need_bottom + pad_extra) / fig_h,
        top=1 - (need_top + pad_extra) / fig_h,
    )
    if wspace is not None:
        kwargs["wspace"] = wspace

    fig.subplots_adjust(**kwargs)


def auto_colorbar_adjust(
    fig,
    mappable,
    pad_extra: float = 0.05,
    cbar_width: float = 0.03,
    cbar_gap: float = 0.04,
    **cbar_kwargs,
):
    """Like :func:`auto_adjust` but for figures with a colorbar.

    Creates a colorbar, then measures the rendered content (including
    colorbar tick labels) and repositions everything so nothing is
    clipped.

    Call this **after** all plotting and labelling is done.

    Args:
        fig: matplotlib Figure (should contain only the main subplot axes).
        mappable: the mappable (image, ScalarMappable, etc.) for the colorbar.
        pad_extra: extra safety margin in inches on every side.
        cbar_width: colorbar width as a fraction of figure width (default 0.03).
        cbar_gap: gap between the main axes and the colorbar as a fraction
            of figure width (default 0.04).
        **cbar_kwargs: forwarded to ``fig.colorbar`` (e.g. ``label=...``).

    Returns:
        cax: the colorbar :class:`~matplotlib.axes.Axes`.

    Example::

        fig, ax = plt.subplots(figsize=reflplots.figsize(0.48, aspect=0.7))
        im = ax.pcolormesh(...)
        ax.set_title("Title")
        cax = reflplots.auto_colorbar_adjust(fig, im)
        reflplots.save(fig, name, tight=False)
    """
    fig_w, fig_h = fig.get_size_inches()
    cbar_w_frac = cbar_width
    cbar_g_frac = cbar_gap

    # --- pass 1: initial layout with generous colorbar space -----
    cbar_left = 1 - 0.15 - cbar_w_frac          # generous initial right margin
    main_right = cbar_left - cbar_g_frac
    fig.subplots_adjust(left=0.15, right=main_right, bottom=0.15, top=0.90)
    cax = fig.add_axes((cbar_left, 0.15, cbar_w_frac, 0.75))
    fig.colorbar(mappable, cax=cax, **cbar_kwargs)

    # --- pass 2: measure everything including colorbar labels ----
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Measure main axes (exclude the colorbar axes)
    need_left, _, need_bottom, need_top = _measure_axes_padding(
        fig, renderer, exclude=[cax],
    )

    # Measure colorbar's right-side extent (its tick labels)
    cbar_bb = cax.get_tightbbox(renderer)
    dpi = fig.dpi
    if cbar_bb is not None:
        cbar_pos = cax.get_position()
        need_cbar_right = cbar_bb.x1 / dpi - cbar_pos.x1 * fig_w
    else:
        need_cbar_right = 0.2

    # --- final layout --------------------------------------------
    # Guarantee room for a title even if it hasn't been set yet
    need_top = max(need_top, PAD_TOP_1LINE)

    pad_l = (need_left + pad_extra) / fig_w
    pad_b = (need_bottom + pad_extra) / fig_h
    pad_t = (need_top + pad_extra) / fig_h

    cbar_right_edge = 1 - (need_cbar_right + pad_extra) / fig_w
    cbar_left = cbar_right_edge - cbar_w_frac
    main_right = cbar_left - cbar_g_frac

    fig.subplots_adjust(left=pad_l, right=main_right, bottom=pad_b, top=1 - pad_t)
    cax.set_position((cbar_left, pad_b, cbar_w_frac, 1 - pad_t - pad_b))
    return cax


def colorbar_adjust(
    fig,
    left: float = 0.16,
    bottom: float = 0.42,
    right: float = 0.78,
    cbar_left: float = 0.82,
    cbar_width: float = 0.03,
    pad_top: float = PAD_TOP_1LINE,
):
    """Apply subplots_adjust and add a colorbar axes aligned to the plot area.

    Computes ``top`` from an absolute inch pad so the axes title always fits
    regardless of figure height or paper size.

    Args:
        fig: matplotlib Figure.
        left, bottom, right: figure-fraction margins for the main axes.
        cbar_left: figure-fraction left edge of the colorbar axes.
        cbar_width: figure-fraction width of the colorbar axes.
        pad_top: inches reserved above the axes for the title.
            Use ``PAD_TOP_2LINE`` for two-line titles.

    Returns:
        cax: the colorbar :class:`~matplotlib.axes.Axes`.

    Example::

        fig, ax = plt.subplots(figsize=reflplots.figsize(0.48, aspect=0.7))
        ...
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cax = reflplots.colorbar_adjust(fig)
        fig.colorbar(sm, cax=cax)
    """
    fig_h = fig.get_size_inches()[1]
    top = 1.0 - pad_top / fig_h
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    return fig.add_axes((cbar_left, bottom, cbar_width, top - bottom))


def figsize_panels(
    n_panels: int,
    max_panels: int = 4,  # your widest layout
    panel_aspect: float = 1.0,
    gap: float = 0.15,
) -> tuple[float, float]:
    """Always returns the same figure width regardless of n_panels.
    Fewer panels = more whitespace, but panels stay the same size.
    """
    panel_width = (TEXT_WIDTH - (max_panels - 1) * gap) / max_panels
    fig_width = TEXT_WIDTH  # always full text width
    fig_height = panel_width * panel_aspect
    return (fig_width, fig_height)


def make_centered_subplots(
    n_panels: int,
    max_panels: int = 4,
    panel_aspect: float = 1.0,
    gap: float = 0.15,
    pad_top: float = 0.45,  # inches reserved for titles
    pad_bottom: float = 0.4,  # inches for x tick labels
    pad_left: float = 0.4,  # inches for y tick labels
    pad_inner: float = 0.15,  # inches between y-label and next panel
):
    """Create n_panels centered in a full-width figure.
    Axes boxes are identical in absolute inches regardless of n_panels.
    """
    panel_width = (TEXT_WIDTH - (max_panels - 1) * gap) / max_panels
    ax_size = panel_width - pad_left - pad_inner  # actual axes box width
    ax_height = ax_size * panel_aspect

    fig_width = TEXT_WIDTH
    fig_height = ax_height + pad_top + pad_bottom

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Total width used by n panels (in figure fraction)
    total_axes = n_panels * ax_size + (n_panels - 1) * gap
    total_with_pad = pad_left + total_axes + pad_inner
    left_start = (fig_width - total_with_pad) / 2 + pad_left

    axes = []
    for i in range(n_panels):
        x = left_start + i * (ax_size + gap)
        # Convert inches to figure fraction
        rect = [
            x / fig_width,
            pad_bottom / fig_height,
            ax_size / fig_width,
            ax_height / fig_height,
        ]
        axes.append(fig.add_axes(rect))

    return fig, axes
