"""Extract shoeboxes from laue-dials integrated.refl + integrated.expt.

Differences from refltorch.mksbox (regular DIALS):
- Each experiment is a single-frame still (no rotation scan); xyzcal.px.z
  is ~0 for all refls, so routing goes through refl["id"] (the per-image
  experiment index) instead.

Run:
    refltorch.mksbox-laue \\
        --data-dir /n/.../laue-dials \\
        --refl integrated.refl \\
        --expt integrated.expt \\
        --out-dir /n/.../pytorch_data \\
        --w 21 --h 21 --d 1 \\
        --max-images 1000 \\
        --save-as-pt
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from refltorch.refl_utils import refl_as_pt

_TRAILING_INT_RE = re.compile(r"_(\d+)\.[A-Za-z0-9]+$")


def _fast_beam_center_px(expt_path: Path) -> tuple[float, float]:
    """Extract beam center in pixels from .expt JSON without loading images."""
    import json

    with open(expt_path) as f:
        data = json.load(f)

    s0 = np.array(data["beam"][0]["direction"])
    panel = data["detector"][0]["panels"][0]
    origin = np.array(panel["origin"])
    fast = np.array(panel["fast_axis"])
    slow = np.array(panel["slow_axis"])
    pix = panel["pixel_size"]

    normal = np.cross(fast, slow)
    t = -origin.dot(normal) / s0.dot(normal)
    bc_mm = t * s0 - origin
    cx = bc_mm.dot(fast) / pix[0]
    cy = bc_mm.dot(slow) / pix[1]
    return (cx, cy)


def _path_to_image_num(path: str) -> int:
    """Parse the trailing integer in a laue-dials image filename.

    e.g. HEWL_NaI_3_2_0001.mccd -> 0   (1-indexed on disk -> 0-indexed here)
    """
    m = _TRAILING_INT_RE.search(path)
    if m is None:
        raise ValueError(f"could not parse image number from filename: {path}")
    return int(m.group(1)) - 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract shoeboxes from a laue-dials integrated.refl + integrated.expt.",
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--refl", type=str, required=True)
    parser.add_argument("--expt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="out_dir")
    parser.add_argument("--w", type=int, default=21, help="shoebox width (odd)")
    parser.add_argument("--h", type=int, default=21, help="shoebox height (odd)")
    parser.add_argument(
        "--d",
        type=int,
        default=1,
        help="shoebox depth (must be 1 for laue stills)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        required=True,
        help="keep only reflections whose image_num is < this value",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="number of images per worker chunk",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="max parallel workers (capped by os.cpu_count())",
    )
    parser.add_argument(
        "--refl-fname",
        type=str,
        default="reflections_.refl",
        help="filename of output reflection table",
    )
    parser.add_argument(
        "--counts-fname",
        type=str,
        default="counts.npy",
    )
    parser.add_argument(
        "--masks-fname",
        type=str,
        default="masks.npy",
    )
    parser.add_argument(
        "--counts-dtype",
        type=str,
        default="uint16",
        choices=["uint16", "int32", "float32"],
    )
    parser.add_argument(
        "--shoebox-format",
        type=str,
        default="npy",
        choices=["npy", "pt"],
        help="storage format for counts and masks. npy uses streaming memmap "
        "writes (low RAM, recommended for large datasets); pt converts each "
        "memmap to a .pt tensor at the end (full RAM materialization).",
    )
    parser.add_argument(
        "--save-as-pt",
        action="store_true",
        help="also write stats.pt, anscombe_stats.pt, concentration.pt",
    )
    parser.add_argument("--stats-chunk", type=int, default=10_000)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="fraction of reflections to flag as is_test (random per-image)",
    )
    return parser.parse_args()


def _shift_bbox_xy(_x, _y, nx, ny, dx_det, dy_det):
    """Shift the (x, y) range so the full window stays on the detector.

    Falls back to clipping only when the requested window is wider than the
    detector itself — degenerate case. Mirrors generate_shoeboxes_ld.py.
    """
    fw_x = 2 * nx + 1
    fw_y = 2 * ny + 1

    x0_full = _x - nx
    x1_full = _x + nx + 1
    if x0_full < 0:
        x0 = 0
        x1 = min(dx_det, x0 + fw_x)
        if x1 - x0 < fw_x:
            x1 = min(dx_det, x0_full + fw_x)
    elif x1_full >= dx_det:
        x1 = dx_det
        x0 = max(0, x1 - fw_x)
        if x1 - x0 < fw_x:
            x0 = max(0, x1_full - fw_x)
    else:
        x0 = x0_full
        x1 = x1_full

    y0_full = _y - ny
    y1_full = _y + ny + 1
    if y0_full < 0:
        y0 = 0
        y1 = min(dy_det, y0 + fw_y)
        if y1 - y0 < fw_y:
            y1 = min(dy_det, y0_full + fw_y)
    elif y1_full >= dy_det:
        y1 = dy_det
        y0 = max(0, y1 - fw_y)
        if y1 - y0 < fw_y:
            y0 = max(0, y1_full - fw_y)
    else:
        y0 = y0_full
        y1 = y1_full

    return x0, x1, y0, y1


def _process_image_chunk(
    image_records,
    expt_path,
    dz,
    dy,
    dx,
    counts_path,
    masks_path,
    counts_dtype_str,
):
    """Worker.

    Opens the pre-allocated `counts.npy` / `masks.npy` memmaps in r+ mode and
    writes its slice of refls directly. Each refl_id is owned by exactly one
    worker (refls are grouped by experiment, experiments by chunk), so the
    worker writes are at disjoint rows.

    image_records: list of dicts, one per image to process. Each dict has:
        - "expt_idx": int, position into the .expt's experiment list
        - "panels":   (n,) int array
        - "bboxes":   (n, 6) int array, z range = (0, 1)
        - "refl_ids": (n,) int array

    Returns a small summary dict; no large arrays travel back to main.
    """
    import numpy as np  # noqa: F811  (re-import for fork safety)
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    experiments = ExperimentListFactory.from_json_file(expt_path, check_format=True)
    counts_dtype = np.dtype(counts_dtype_str)
    dtype_max = (
        np.iinfo(counts_dtype).max if np.issubdtype(counts_dtype, np.integer) else None
    )

    counts_mm = np.load(counts_path, mmap_mode="r+")
    masks_mm = np.load(masks_path, mmap_mode="r+")

    n_done = 0
    n_clipped = 0
    for rec in image_records:
        expt_idx = rec["expt_idx"]
        panels = rec["panels"]
        bboxes = rec["bboxes"]
        refl_ids = rec["refl_ids"]
        n = len(refl_ids)

        subset = flex.reflection_table()
        subset["panel"] = flex.size_t(panels.astype(np.int64))
        bbox_col = flex.int6(n)
        for j in range(n):
            bbox_col[j] = (
                int(bboxes[j, 0]),
                int(bboxes[j, 1]),
                int(bboxes[j, 2]),
                int(bboxes[j, 3]),
                int(bboxes[j, 4]),
                int(bboxes[j, 5]),
            )
        subset["bbox"] = bbox_col
        subset["shoebox"] = flex.shoebox(subset["panel"], subset["bbox"], allocate=True)

        imageset = experiments[expt_idx].imageset
        subset.extract_shoeboxes(imageset)

        counts = np.zeros((n, dz, dy, dx), dtype=np.int32)
        masks = np.zeros((n, dz, dy, dx), dtype=bool)
        for i, sb in enumerate(subset["shoebox"]):
            counts[i] = sb.data.as_numpy_array()
            masks[i] = (sb.mask.as_numpy_array() & 1).astype(bool)

        counts_flat = counts.reshape(n, -1)
        masks_flat = masks.reshape(n, -1)

        if dtype_max is not None:
            over = counts_flat > dtype_max
            if over.any():
                n_clipped += int(over.sum())
                counts_flat = np.clip(counts_flat, 0, dtype_max)

        # Direct write to disk-backed memmap; no pickling back to main.
        counts_mm[refl_ids] = counts_flat.astype(counts_dtype, copy=False)
        masks_mm[refl_ids] = masks_flat
        n_done += n

    counts_mm.flush()
    masks_mm.flush()
    return {"n_done": n_done, "n_clipped": n_clipped}


def _save_stats_from_memmap(
    counts_path: Path,
    masks_path: Path,
    out_dir: Path,
    chunk: int = 10_000,
    ans_fname: str = "anscombe_stats.pt",
    stats_fname: str = "stats.pt",
):
    counts = np.load(counts_path, mmap_mode="r")
    masks = np.load(masks_path, mmap_mode="r")
    n, _ = counts.shape

    sum_c = sumsq_c = sum_a = sumsq_a = 0.0
    nel = 0
    for i in range(0, n, chunk):
        c = counts[i : i + chunk].astype(np.float64)
        m = masks[i : i + chunk]
        c = c * m
        a = 2.0 * np.sqrt(c + 0.375)
        sum_c += c.sum()
        sumsq_c += (c * c).sum()
        sum_a += a.sum()
        sumsq_a += (a * a).sum()
        nel += c.size

    mean_c = sum_c / nel
    var_c = sumsq_c / nel - mean_c * mean_c
    mean_a = sum_a / nel
    var_a = sumsq_a / nel - mean_a * mean_a

    torch.save(
        torch.tensor([mean_c, var_c], dtype=torch.float32), out_dir / stats_fname
    )
    torch.save(torch.tensor([mean_a, var_a], dtype=torch.float32), out_dir / ans_fname)


def _save_concentration_from_memmap(
    counts_path: Path,
    out_dir: Path,
    chunk: int = 10_000,
    out_fname: str = "concentration.pt",
):
    counts = np.load(counts_path, mmap_mode="r")
    n = counts.shape[0]
    conc = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        c = counts[i : i + chunk].astype(np.float32)
        conc[i : i + chunk] = c.mean(axis=1)
    torch.save(torch.from_numpy(conc), out_dir / out_fname)


def _convert_npy_memmap_to_pt(npy_path: Path) -> Path:
    """Wriate a .npy file into a .pt tensor.

    Reads the file fully into RAM in one shot (torch.save does not stream).
    Returns the new .pt path.
    """
    arr = np.load(npy_path)
    pt_path = npy_path.with_suffix(".pt")
    torch.save(torch.from_numpy(arr), pt_path)
    del arr
    npy_path.unlink()
    return pt_path


def main():
    import yaml
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    args = parse_args()

    if args.w % 2 == 0 or args.h % 2 == 0:
        raise ValueError(f"--w and --h must be odd (got w={args.w}, h={args.h})")
    if args.d != 1:
        raise ValueError(
            f"--d must be 1 for laue extraction (got {args.d}); multi-frame "
            "stills would need to span adjacent imagesets, which is not supported."
        )

    nx = args.w // 2
    ny = args.h // 2
    dz, dy, dx = args.d, args.h, args.w

    data_dir = Path(args.data_dir)
    refl_path_in = data_dir / args.refl
    expt_path_in = data_dir / args.expt
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {refl_path_in}")
    reflections = flex.reflection_table.from_file(str(refl_path_in))
    print(f"  {len(reflections)} reflections")

    print(f"loading {expt_path_in}")
    experiments = ExperimentListFactory.from_json_file(
        str(expt_path_in),
        check_format=False,
    )
    print(f"  {len(experiments)} experiments")

    #  expt_idx -> image_num map (laue-dials filename convention)
    expt_to_img = np.empty(len(experiments), dtype=np.int64)
    for i in range(len(experiments)):
        expt_to_img[i] = _path_to_image_num(experiments[i].imageset.get_path(0))
    if len(np.unique(expt_to_img)) != len(expt_to_img):
        raise ValueError("duplicate image_num across experiments; check filenames")
    print(
        f"  image_num range: {expt_to_img.min()}-{expt_to_img.max()} "
        f"({len(expt_to_img)} images)"
    )

    # detector size (single-panel, taken from experiment 0)
    det0 = experiments[0].detector[0]
    dx_det, dy_det = det0.get_image_size()

    #  per-refl image_num via id (stills convention)
    expt_idx_per_refl = np.array(reflections["id"]).astype(np.int64)
    if expt_idx_per_refl.min() < 0 or expt_idx_per_refl.max() >= len(experiments):
        raise ValueError(
            f"refl['id'] out of bounds: range=[{expt_idx_per_refl.min()}, "
            f"{expt_idx_per_refl.max()}], len(experiments)={len(experiments)}"
        )
    image_num_per_refl = expt_to_img[expt_idx_per_refl]

    # filter: --max-images on derived image_num
    keep_np = image_num_per_refl < args.max_images
    keep_mask = flex.bool(keep_np.tolist())
    n_kept = int(keep_np.sum())
    print(
        f"keeping {n_kept} / {len(reflections)} refls (image_num < {args.max_images})"
    )
    reflections = reflections.select(keep_mask)
    expt_idx_per_refl = expt_idx_per_refl[keep_np]
    image_num_per_refl = image_num_per_refl[keep_np]

    # wavelength: read from refl table (laue-dials writes it)
    if "wavelength" not in reflections:
        raise ValueError(
            "refl table has no 'wavelength' column; this CLI expects laue-dials "
            "integrated output. (Older un-integrated tables can use 1/|s1| "
            "instead — open a ticket if you need that path.)"
        )

    # bbox column with shift logic; z fixed to (0, 1) per single-frame
    x_int = flex.floor(reflections["xyzcal.px"].parts()[0]).iround()
    y_int = flex.floor(reflections["xyzcal.px"].parts()[1]).iround()

    bbox = flex.int6(len(reflections))
    for j in range(len(reflections)):
        x0, x1, y0, y1 = _shift_bbox_xy(x_int[j], y_int[j], nx, ny, dx_det, dy_det)
        bbox[j] = (x0, x1, y0, y1, 0, 1)
    reflections["bbox"] = bbox

    # refl_ids + image_num + is_test
    n_refl = len(reflections)
    reflections["refl_ids"] = flex.int(np.arange(n_refl, dtype=np.int32))
    reflections["image_num"] = flex.int(image_num_per_refl.astype(np.int32))
    rng = np.random.default_rng(42)
    is_test = rng.random(n_refl) < args.test_fraction
    reflections["is_test"] = flex.bool(is_test.tolist())
    print(f"  is_test: {is_test.sum()} / {n_refl} ({100 * is_test.mean():.1f}%)")

    # d-spacing per refl via per-experiment unit cell
    reflections.compute_d(experiments)
    d_arr = np.array(reflections["d"])
    print(f"  d range: {d_arr.min():.3f} - {d_arr.max():.3f} Å")

    # save the reflection table (now also carries d)
    refl_path_out = out_dir / args.refl_fname
    reflections.as_file(str(refl_path_out))
    print(f"wrote refl with bbox/wavelength/refl_ids/d -> {refl_path_out}")

    # write identifiers for traceability
    identifier = dict(reflections.experiment_identifiers())
    (out_dir / "identifiers.yaml").write_text(yaml.safe_dump(identifier))

    # crystal metadata: cell + spacegroup + beam center
    # All per-image experiments are copies of the same refined crystal model,
    # so the first one is representative.
    crystal0 = experiments[0].crystal
    cell_params = crystal0.get_unit_cell().parameters()
    sg_info = crystal0.get_space_group().info()

    beam_center_px = _fast_beam_center_px(expt_path_in)

    crystal_meta = {
        "cell": [float(x) for x in cell_params],
        "space_group": sg_info.symbol_and_number(),
        "space_group_number": int(sg_info.type().number()),
        "beam_center_px": [float(beam_center_px[0]), float(beam_center_px[1])],
    }
    (out_dir / "crystal.yaml").write_text(yaml.safe_dump(crystal_meta))
    print(
        f"wrote crystal metadata -> {out_dir / 'crystal.yaml'}: "
        f"cell={tuple(round(c, 3) for c in crystal_meta['cell'])}, "
        f"sg={crystal_meta['space_group']}, "
        f"beam_center_px=({beam_center_px[0]:.1f}, {beam_center_px[1]:.1f})"
    )

    # group refls by expt_idx for parallel extraction
    panels_np = np.array(reflections["panel"])
    bboxes_np = np.stack(
        [b.as_numpy_array() for b in reflections["bbox"].parts()],
        axis=-1,
    )  # (N, 6)
    refl_ids_np = np.array(reflections["refl_ids"])

    image_records: list[dict] = []
    for ei in np.unique(expt_idx_per_refl):
        sel = expt_idx_per_refl == ei
        image_records.append(
            {
                "expt_idx": int(ei),
                "panels": panels_np[sel],
                "bboxes": bboxes_np[sel],
                "refl_ids": refl_ids_np[sel],
            }
        )
    print(f"prepared {len(image_records)} image records for extraction")

    #  chunk and run in parallel
    block_size = args.block_size
    chunks = [
        image_records[i : i + block_size]
        for i in range(0, len(image_records), block_size)
    ]

    max_workers = min(args.max_workers, os.cpu_count() or 1)
    print(f"running {len(chunks)} chunks across {max_workers} workers")

    # Pre-allocate output memmaps and close them
    N = len(refl_ids_np)
    counts_path = out_dir / args.counts_fname
    masks_path = out_dir / args.masks_fname
    counts_dtype = np.dtype(args.counts_dtype)

    counts_mm = open_memmap(
        counts_path,
        mode="w+",
        dtype=counts_dtype,
        shape=(N, dz * dy * dx),
    )
    masks_mm = open_memmap(
        masks_path,
        mode="w+",
        dtype=np.bool_,
        shape=(N, dz * dy * dx),
    )
    counts_mm.flush()
    masks_mm.flush()
    del counts_mm, masks_mm  # close so workers can reopen r+

    n_done = 0
    n_clipped_total = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_image_chunk,
                chunk,
                str(expt_path_in),
                dz,
                dy,
                dx,
                str(counts_path),
                str(masks_path),
                args.counts_dtype,
            )
            for chunk in chunks
        ]
        for f in as_completed(futures):
            res = f.result()
            n_done += res["n_done"]
            n_clipped_total += res["n_clipped"]

    if n_clipped_total > 0:
        print(
            f"WARNING: {n_clipped_total} pixel(s) exceeded {counts_dtype} max "
            f"({np.iinfo(counts_dtype).max}); clipped. Consider --counts-dtype "
            "int32 if overloads matter."
        )
    print(f"extracted {n_done} shoeboxes -> {counts_path}, {masks_path}")

    #  metadata.pt via refl_as_pt
    from refltorch.refl_utils.refl_utils import DEFAULT_REFL_COLS

    cols = list(DEFAULT_REFL_COLS)
    for must_have in ("wavelength", "d", "image_num", "is_test"):
        if must_have not in cols:
            cols.append(must_have)
    refl_as_pt(refl=str(refl_path_out), column_names=cols, out_dir=out_dir)
    print(f"wrote metadata.pt under {out_dir}")

    if args.save_as_pt:
        _save_stats_from_memmap(
            counts_path=counts_path,
            masks_path=masks_path,
            out_dir=out_dir,
            chunk=args.stats_chunk,
        )
        _save_concentration_from_memmap(
            counts_path=counts_path,
            out_dir=out_dir,
            chunk=args.stats_chunk,
        )
        print("wrote stats.pt, anscombe_stats.pt, concentration.pt")

    if args.shoebox_format == "pt":
        nbytes = counts_path.stat().st_size + masks_path.stat().st_size
        print(
            f"converting counts/masks .npy -> .pt "
            f"(loads {nbytes / 1e9:.1f} GB into RAM)"
        )
        new_counts = _convert_npy_memmap_to_pt(counts_path)
        new_masks = _convert_npy_memmap_to_pt(masks_path)
        print(f"  -> {new_counts}, {new_masks}")


if __name__ == "__main__":
    main()
