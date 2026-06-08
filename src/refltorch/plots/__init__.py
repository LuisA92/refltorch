from ._shared import (
    cubehelix_cmap,
    get_bins,
    get_palette,
    save_figure,
    set_figsize,
)
from .cmaps import get_cube_helix_map
from .mpl_config import setup_mpl_config
from .plotsizes import (
    AX_HEIGHT,
    PAD_BOTTOM,
    PAD_LEFT,
    PAD_RIGHT,
    PAD_TOP_1LINE,
    PAD_TOP_2LINE,
    PAPER_SIZES,
    PaperGeometry,
    apply_fontsizes,
    auto_adjust,
    auto_colorbar_adjust,
    colorbar_adjust,
    figsize,
    figsize_panels,
    make_centered_subplots,
    panel_adjust,
    panel_figsize,
    paper_context,
    set_paper,
)
from .plotstyle import apply as apply_plotstyle
from .plotstyle import save

__all__ = [
    "setup_mpl_config",
    "get_cube_helix_map",
    "cubehelix_cmap",
    "get_bins",
    "get_palette",
    "save_figure",
    "set_figsize",
    "apply_plotstyle",
    "apply_fontsizes",
    "auto_adjust",
    "auto_colorbar_adjust",
    "colorbar_adjust",
    "figsize",
    "figsize_panels",
    "save",
    "make_centered_subplots",
    "panel_figsize",
    "panel_adjust",
    "AX_HEIGHT",
    "PAD_BOTTOM",
    "PAD_LEFT",
    "PAD_RIGHT",
    "PAD_TOP_1LINE",
    "PAD_TOP_2LINE",
    # paper geometry
    "PAPER_SIZES",
    "PaperGeometry",
    "set_paper",
    "paper_context",
]
