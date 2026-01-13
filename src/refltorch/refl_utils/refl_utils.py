from pathlib import Path

import reciprocalspaceship as rs
import reciprocalspaceship.io as io
import torch

SCALAR_DTYPES = {
    "zeta": "double",
    "qe": "double",
    "profile.correlation": "double",
    "partiality": "double",
    "partial_id": "std::size_t",
    "panel": "std::size_t",
    "flags": "std::size_t",
    "num_pixels.valid": "int",
    "num_pixels.foreground": "int",
    "num_pixels.background_used": "int",
    "num_pixels.background": "int",
    "lp": "double",
    "intensity.prf.value": "double",
    "intensity.prf.variance": "double",
    "intensity.sum.value": "double",
    "intensity.sum.variance": "double",
    "imageset_id": "int",
    "entering": "bool",
    "d": "double",
    "background.mean": "double",
    "background.sum.value": "double",
    "background.sum.variance": "double",
    "refl_ids": "int",
}


VECTOR_COLUMNS = {
    "bbox": ("int6", 6),
    "s1": ("vec3<double>", 3),
    "xyzcal.mm": ("vec3<double>", 3),
    "xyzcal.px": ("vec3<double>", 3),
    "xyzobs.mm.value": ("vec3<double>", 3),
    "xyzobs.mm.variance": ("vec3<double>", 3),
    "xyzobs.px.value": ("vec3<double>", 3),
    "xyzobs.px.variance": ("vec3<double>", 3),
    "miller_index": ("cctbx::miller::index<>", 3),
}

DEFAULT_REFL_COLS = list(SCALAR_DTYPES.keys()) + list(VECTOR_COLUMNS.keys())

# Default columns to exclude
DEFAULT_EXCLUDED_COLS = ["BATCH", "PARTIAL"]


def refl_as_pt(
    refl,
    column_names: list[str] = DEFAULT_REFL_COLS,
    excluded_columns: list[str] = DEFAULT_EXCLUDED_COLS,
    out_dir: Path | None = None,
    out_fname: str = "metadata.pt",
) -> dict:
    ds = io.read_dials_stills(
        refl,
        extra_cols=column_names,
    )
    assert isinstance(ds, rs.DataSet)

    data = {}
    for k, v in ds.items():
        if k not in excluded_columns:
            data[k] = torch.tensor(v, dtype=torch.float32)

    # write to output directory if specified
    if out_dir is not None:
        fname = out_dir / out_fname
    else:
        fname = out_fname

    torch.save(data, fname)
    return data
