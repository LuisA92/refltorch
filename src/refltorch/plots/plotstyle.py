"""
plotstyle.py — updated with explicit font registration
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Colors ---
COLORS = {
    "slate": "#334E68",
    "dark": "#1F2933",
    "mid": "#616E7C",
    "light": "#9AA5B4",
    "bg": "#F5F7FA",
}

# --- Font paths (adjust if yours differ) ---
FONT_DIR = Path.home() / "Library" / "Fonts"

_FONT_FILES = {
    # Minion Pro
    "MinionPro-Regular": FONT_DIR / "MinionPro-Regular.otf",
    "MinionPro-Bold": FONT_DIR / "MinionPro-Bold.otf",
    "MinionPro-It": FONT_DIR / "MinionPro-It.otf",
    "MinionPro-BoldIt": FONT_DIR / "MinionPro-BoldIt.otf",
    # Myriad Pro
    "MyriadPro-Regular": FONT_DIR / "MyriadPro-Regular.otf",
    "MyriadPro-Bold": FONT_DIR / "MyriadPro-Bold.otf",
    "MyriadPro-It": FONT_DIR / "MyriadPro-It.otf",
    "MyriadPro-BoldIt": FONT_DIR / "MyriadPro-BoldIt.otf",
    # Source Code Pro (Adobe, not Powerline)
    "SourceCodePro-Regular": FONT_DIR / "SourceCodePro-Regular.otf",
    "SourceCodePro-Bold": FONT_DIR / "SourceCodePro-Bold.otf",
    "SourceCodePro-It": FONT_DIR / "SourceCodePro-It.otf",
    "SourceCodePro-BoldIt": FONT_DIR / "SourceCodePro-BoldIt.otf",
}


def _register_fonts():
    """Register font files with matplotlib's font manager."""
    for name, path in _FONT_FILES.items():
        if path.exists():
            fm.fontManager.addfont(str(path))
        else:
            print(f"plotstyle: font not found: {path}")


def apply(
    font_scale: float = 1.0,
    context: str = "paper",
    palette: str = "Dark2",
    usetex: bool = True,
):
    """Apply the full plot style."""

    # Register fonts with matplotlib
    _register_fonts()

    # Seaborn base
    sns.set_context(context, font_scale=font_scale)
    sns.set_palette(palette)

    if usetex:
        _mtpro2_opts = r"\usepackage{amsthm}\usepackage[uprightGreek,uprightoperators,mtpcal,mtpscr,mtpfrak,mtphrb,amssymbols,]{mtpro2}"
        _pgf_preamble = "\n".join(
            [
                r"\usepackage{fontspec}",
                # Minion Pro — match reportstyle.sty optical sizes
                r"\setmainfont{Minion Pro}[",
                r"  Ligatures={TeX, Common},",
                r"  Kerning=On,",
                r"  Numbers=OldStyle,",
                r"  Scale=MatchLowercase,",
                r"  SizeFeatures={",
                r"    {Size={-8.4},   Font=MinionPro-Capt},",
                r"    {Size={8.4-13}, Font=MinionPro-Regular},",
                r"    {Size={13-19.9},Font=MinionPro-Subh},",
                r"    {Size={19.9-},  Font=MinionPro-Disp},",
                r"  },",
                r"  BoldFont=MinionPro-Bold,",
                r"  ItalicFont=MinionPro-It,",
                r"  BoldItalicFont=MinionPro-BoldIt,",
                r"]",
                # Source Code Pro — match reportstyle.sty
                r"\setmonofont{Source Code Pro}[",
                r"  Extension=.otf,",
                r"  UprightFont=SourceCodePro-Regular,",
                r"  BoldFont=SourceCodePro-Bold,",
                r"  ItalicFont=SourceCodePro-It,",
                r"  BoldItalicFont=SourceCodePro-BoldIt,",
                r"  Scale=MatchLowercase,",
                r"  Numbers=Lining,",
                r"]",
                # Myriad Pro — match reportstyle.sty
                r"\setsansfont{Myriad Pro}[",
                r"  Ligatures={TeX, Common},",
                r"  Kerning=On,",
                r"  Numbers=Lining,",
                r"  Scale=MatchLowercase,",
                r"  BoldFont=MyriadPro-Bold,",
                r"  ItalicFont=MyriadPro-It,",
                r"  BoldItalicFont=MyriadPro-BoldIt,",
                r"]",
                # Math fonts
                _mtpro2_opts,
            ]
        )
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\providecommand{\mathdefault}[1]{#1}" + "\n" + _mtpro2_opts,
                "pgf.texsystem": "lualatex",
                "pgf.preamble": _pgf_preamble,
                "font.family": "serif",
                "font.serif": ["Minion Pro"],
                "font.sans-serif": ["Myriad Pro"],
                "font.monospace": ["Source Code Pro"],
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["Minion Pro", "Times New Roman", "DejaVu Serif"],
                "font.sans-serif": ["Myriad Pro", "Helvetica", "DejaVu Sans"],
                "font.monospace": ["Source Code Pro", "DejaVu Sans Mono"],
            }
        )

    # Figure defaults
    plt.rcParams.update(
        {
            "figure.figsize": (5.5, 3.5),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.05,
            "axes.linewidth": 0.6,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titlepad": 8,
            "axes.labelpad": 4,
            "axes.labelcolor": COLORS["dark"],
            "axes.titlecolor": COLORS["dark"],
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["mid"],
            "axes.grid": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.labelcolor": COLORS["dark"],
            "ytick.labelcolor": COLORS["dark"],
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.color": COLORS["mid"],
            "ytick.color": COLORS["mid"],
            "grid.linewidth": 0.4,
            "grid.alpha": 0.3,
            "grid.color": COLORS["light"],
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "legend.borderpad": 0.3,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "patch.linewidth": 0.5,
        }
    )


def figsize(width_ratio: float = 1.0, aspect: float = 0.618):
    textwidth = 5.5
    w = textwidth * width_ratio
    return (w, w * aspect)


def _fix_pgf_imgpaths(pgf_path, latex_root):
    """Rewrite bare *-imgN.png references in a pgf to paths relative to latex_root.

    Matplotlib's pgf backend writes companion raster images with bare filenames
    (e.g. ``name-img0.png``), but LaTeX resolves includegraphics paths relative
    to the compile directory, not the pgf file's location.  This patches the
    saved pgf so the path is correct when compiling from latex_root.
    """
    import os

    pgf_path = str(pgf_path)
    pgf_dir = os.path.dirname(os.path.abspath(pgf_path))
    root = os.path.abspath(str(latex_root))
    rel_dir = os.path.relpath(pgf_dir, root).replace(os.sep, "/")
    stem = os.path.splitext(os.path.basename(pgf_path))[0]
    with open(pgf_path) as f:
        content = f.read()
    patched = content.replace(f"{{{stem}-img", f"{{{rel_dir}/{stem}-img")
    if patched != content:
        with open(pgf_path, "w") as f:
            f.write(patched)


def save(fig, name, formats=("pdf", "pgf", "png"), tight=True, latex_root=None):
    """Save figure in one or more formats.

    Parameters
    ----------
    latex_root:
        Path to the LaTeX compile directory (e.g. ``project_root / "report"``).
        When set, bare ``*-imgN.png`` references in the saved ``.pgf`` are
        rewritten to paths relative to that directory so rasterized elements
        are found at compile time.
    """
    kw = {"pad_inches": 0.05}
    if tight:
        kw["bbox_inches"] = "tight"
    for fmt in formats:
        fig.savefig(f"{name}.{fmt}", backend="pgf", **kw)
        if fmt == "pgf" and latex_root is not None:
            _fix_pgf_imgpaths(f"{name}.pgf", latex_root)
