"""Refinement R-value curves over epochs.

Plots R-work and R-free for a single run over epochs, distinguishing the
two metrics by line style and optionally overlaying reference R-values.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from ._shared import get_palette, set_figsize


def plot_r_values(
    run_df,
    *,
    palette=None,
    dashes=None,
    ref_vals=None,
    hue_key="Label",
    style_key="Metric",
    linewidth=1,
    title=None,
):
    """Plot R-work/R-free over epochs for one run.

    Args:
        run_df: Long-format frame for a single run with `epoch`, `value`,
            `hue_key`, and `style_key` columns. Filter to the desired
            refinement stage (e.g. `final`) before calling.
        palette: Optional color mapping over `hue_key` values. Pass a
            palette built from all runs to keep colors consistent.
        dashes: Optional style-to-dash mapping. Defaults to dashed r_free,
            solid r_work.
        ref_vals: Optional mapping with `r_work_final` and `r_free_final`
            keys, drawn as horizontal reference lines.
        hue_key: Column used to color the lines.
        style_key: Column used to set line style (r_work vs r_free).
        linewidth: Line width.
        title: Optional axis title.

    Returns:
        Tuple of (fig, ax).
    """
    if palette is None:
        palette = get_palette(run_df[hue_key].unique().to_list())
    if dashes is None:
        dashes = {"r_free": (2, 1), "r_work": ""}

    fig, ax = plt.subplots(figsize=set_figsize())
    sns.lineplot(
        data=run_df,
        x="epoch",
        y="value",
        hue=hue_key,
        style=style_key,
        palette=palette,
        dashes=dashes,
        linewidth=linewidth,
        ax=ax,
    )

    if ref_vals is not None:
        ax.axhline(
            ref_vals["r_work_final"],
            color="#cc0000",
            label="DIALS rwork final",
        )
        ax.axhline(
            ref_vals["r_free_final"],
            color="#cc0000",
            linestyle="--",
            label="DIALS rfree final",
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("r value")
    if title is not None:
        ax.set_title(title)
    ax.grid()
    ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))
    return fig, ax
