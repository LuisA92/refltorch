"""Parsers for DIALS analysis artifacts.

The merged HTML produced by `dials.merge`/`dials.scale` contains a
per-resolution-bin merging-statistics table. `parse_dials_merging_stats`
turns one such table (already read via `pandas.read_html`) into a tidy
polars frame, optionally tagged with epoch/run/model identifiers.
"""

import polars as pl

DIALS_MERGING_KEYS = (
    "resolution",
    "n_refls",
    "n_unique",
    "multiplicity",
    "completeness",
    "mean_i",
    "meani_sigi",
    "rmerge",
    "rmeas",
    "rpim",
    "ranom",
    "cchalf",
    "ccanom",
)


def parse_dials_merging_stats(
    tbl,
    epoch: int | None = None,
    run_id: str | None = None,
    model_name: str | None = None,
    keys: tuple[str, ...] = DIALS_MERGING_KEYS,
) -> pl.LazyFrame:
    """Parse a DIALS merging-statistics table into a polars LazyFrame.

    The `cchalf` and `ccanom` columns may carry a trailing `*` (DIALS marks
    the resolution at which the statistic falls below a threshold); it is
    stripped before casting to float.

    Args:
        tbl: A pandas DataFrame for one merging-stats table (one row per
            resolution bin), e.g. the second table from
            `pandas.read_html(merged_html)`.
        epoch: Optional epoch to attach as a literal column.
        run_id: Optional run id to attach as a literal column.
        model_name: Optional model name to attach as a literal column.
        keys: Output column names, positionally matched to `tbl.columns`.

    Returns:
        A LazyFrame with one column per key plus any attached identifiers.
        Call `.collect()` for an eager DataFrame.
    """
    if len(keys) != len(tbl.columns):
        raise ValueError("keys must match number of columns")

    data = {}
    for key, col in zip(keys, tbl.columns):
        values = tbl[col].tolist()
        if key in ("cchalf", "ccanom"):
            values = [float(str(v).strip("*")) for v in values]
        data[key] = values

    lf = pl.LazyFrame(data)
    if epoch is not None:
        lf = lf.with_columns(epoch=pl.lit(epoch))
    if run_id is not None:
        lf = lf.with_columns(run_id=pl.lit(run_id))
    if model_name is not None:
        lf = lf.with_columns(model_name=pl.lit(model_name))

    return lf
