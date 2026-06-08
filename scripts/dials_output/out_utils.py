"""Shared workflow assembly for the dials_output analysis scripts.

Holds constants and helpers specific to this script family's directory and
config conventions: resolution/intensity bin edges, the run-prediction
filename schema, the dials_phenix_cfg dataclass, and a peaks.csv reader.

Generic, broadly reusable helpers live in `refltorch` instead:
- reference-config helpers: `refltorch.tools` (get_reference_*).
- DIALS/PHENIX parsers: `refltorch.tools` (parse_dials_merging_stats,
  parse_phenix_r_values).
- run-path resolution: `refltorch.io` (open_run, glob_predictions,
  epoch_from_path).
- bin-axis + figure helpers: `refltorch.plots._shared`.
"""

from dataclasses import dataclass

import polars as pl

INTENSITY_EDGES = [0, 10, 25, 50, 100, 300, 600, 1000, 1500, 2500, 5000, 10000]
DIALS_EDGES_9B7C = [
    0,
    1.1,
    1.11,
    1.14,
    1.16,
    1.18,
    1.21,
    1.23,
    1.27,
    1.30,
    1.34,
    1.49,
    1.56,
    1.64,
    1.74,
    2.06,
    2.36,
    2.97,
]


@dataclass
class DialsPhenixConfig:
    """Schema for `dials_phenix_cfg.yaml` (written by create_config.py).

    Used both to write the config and to read it back (e.g. in
    process_single_refl.py), so it is the single source of truth for the
    field set. Optional fields carry defaults so older configs that omit
    them still load.
    """

    refl_files: list
    expt_file: str
    dials_env: str
    phenix_env: str
    phenix_eff: str
    pdb: str
    paired_ref_eff: str
    paired_model_eff: str
    columns: str = "intensity"
    merged_mtz: list | None = None
    user_selected_labels: str = ""
    french_wilson_scale: bool = True


def add_run_epoch_cols(
    lf: pl.LazyFrame,
    *,
    path_col: str = "filenames",
    epoch_pattern: str = r"/epoch_(\d+)/",
) -> pl.LazyFrame:
    """Add `run_id` and `epoch` columns extracted from a path column.

    Args:
        lf: LazyFrame scanned with `include_file_paths=path_col`.
        path_col: Column holding the source file path of each row.
        epoch_pattern: Regex (one capture group) for the epoch number;
            override for layouts like `/train_epoch_(\\d+).parquet`.

    Returns:
        The LazyFrame with `run_id` (str) and `epoch` (Int32) columns.
    """
    return lf.with_columns(
        [
            pl.col(path_col).str.extract(r"/run-[^/]+-([^/]+)/", 1).alias("run_id"),
            pl.col(path_col)
            .str.extract(epoch_pattern, 1)
            .cast(pl.Int32)
            .alias("epoch"),
        ]
    )


def read_peaks_csv(path) -> pl.DataFrame:
    """Read an anomalous peaks.csv into a polars DataFrame.

    Args:
        path: Path to a peaks.csv with `seqid`, `residue`, `peakz` columns.

    Returns:
        DataFrame sorted by `seqid`.
    """
    return pl.read_csv(
        path,
        columns=["seqid", "residue", "peakz"],
        schema_overrides={
            "seqid": pl.Int64,
            "residue": pl.String,
            "peakz": pl.Float64,
        },
    ).sort("seqid")
