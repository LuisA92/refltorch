"""Extract shoeboxes from laue-dials integrated.refl + integrated.expt.

Differences from refltorch.mksbox (regular DIALS):
- Each experiment is a single-frame still (no rotation scan).
- refl["id"] is degenerate (all zero); image_num is recovered from the
  trailing integer in the imageset filename (per laue-dials convention).
- Wavelength is recovered as 1/|s1| per reflection.
- Bbox edge handling shifts to preserve full size (vs clip+zero-pad).
- --max-images selects the first N original image numbers (skips images
  that were dropped by laue-dials integration).

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
        default=8,
        help="max parallel workers (capped by os.cpu_count())",
    )
    parser.add_argument(
        "--refl-fname", type=str, default="reflections_.refl",
        help="filename of output reflection table",
    )
    parser.add_argument(
        "--counts-fname", type=str, default="counts.npy",
    )
    parser.add_argument(
        "--masks-fname", type=str, default="masks.npy",
    )
    parser.add_argument(
        "--counts-dtype",
        type=str,
        default="uint16",
        choices=["uint16", "int32", "float32"],
    )
    parser.add_argument(
        "--save-as-pt",
        action="store_true",
        help="also write stats.pt, anscombe_stats.pt, concentration.pt",
    )
    parser.add_argument("--stats-chunk", type=int, default=10_000)
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
):
    """Worker.

    image_records: list of dicts, one per image to process. Each dict has:
        - "expt_idx": int, position into the .expt's experiment list
        - "panels":   (n,) int array
        - "bboxes":   (n, 6) int array, z range = (0, 1)
        - "refl_ids": (n,) int array
    """
    import numpy as np  # noqa: F811  (re-import for fork safety)
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    experiments = ExperimentListFactory.from_json_file(expt_path, check_format=True)

    results = []
    for rec in image_records:
        expt_idx = rec["expt_idx"]
        panels = rec["panels"]
        bboxes = rec["bboxes"]
        refl_ids = rec["refl_ids"]
        n = len(refl_ids)

        # Build a minimal reflection table so DIALS can do the actual extraction.
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
        subset["shoebox"] = flex.shoebox(
            subset["panel"], subset["bbox"], allocate=True
        )

        imageset = experiments[expt_idx].imageset
        subset.extract_shoeboxes(imageset)

        counts = np.zeros((n, dz, dy, dx), dtype=np.int32)
        masks = np.zeros((n, dz, dy, dx), dtype=bool)
        for i, sb in enumerate(subset["shoebox"]):
            data = sb.data.as_numpy_array()         # (dz, dy, dx)
            mraw = sb.mask.as_numpy_array()         # bitfield; bit 1 = Valid
            counts[i] = data
            masks[i] = (mraw & 1).astype(bool)

        results.append(
            {
                "refl_ids": refl_ids,
                "counts": counts.reshape(n, -1),
                "masks": masks.reshape(n, -1),
            }
        )

    return results


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

    torch.save(torch.tensor([mean_c, var_c], dtype=torch.float32), out_dir / stats_fname)
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
        str(expt_path_in), check_format=False,
    )
    print(f"  {len(experiments)} experiments")

    # --- image_num <-> expt_idx map (laue-dials filename convention) ----------
    img_to_expt: dict[int, int] = {}
    for i in range(len(experiments)):
        path = experiments[i].imageset.get_path(0)
        img_num = _path_to_image_num(path)
        if img_num in img_to_expt:
            raise ValueError(
                f"duplicate image_num {img_num} from path {path}; "
                f"earlier expt_idx={img_to_expt[img_num]}, this one={i}"
            )
        img_to_expt[img_num] = i
    print(
        f"  image_num range: {min(img_to_expt)}-{max(img_to_expt)} "
        f"({len(img_to_expt)} images)"
    )

    # --- detector size (single-panel, taken from experiment 0) ----------------
    det0 = experiments[0].detector[0]
    dx_det, dy_det = det0.get_image_size()

    # --- per-refl image number from xyzcal.px.z ------------------------------
    xyzcal = reflections["xyzcal.px"]
    x_px = xyzcal.parts()[0]
    y_px = xyzcal.parts()[1]
    z_px = xyzcal.parts()[2]
    image_num = flex.floor(z_px).iround()

    # filter: drop refls on missing images, and apply --max-images
    keep_mask = flex.bool(len(reflections), False)
    for j in range(len(reflections)):
        n_img = image_num[j]
        if n_img < 0 or n_img >= args.max_images:
            continue
        if n_img not in img_to_expt:
            continue
        keep_mask[j] = True
    n_kept = keep_mask.count(True)
    print(
        f"keeping {n_kept} / {len(reflections)} refls "
        f"(image_num < {args.max_images} AND in expt list)"
    )
    reflections = reflections.select(keep_mask)
    image_num = image_num.select(keep_mask)

    # --- wavelength = 1/|s1| -------------------------------------------------
    s1_np = np.array(reflections["s1"])  # (N, 3), units 1/Å
    wavelength_np = 1.0 / np.linalg.norm(s1_np, axis=1)
    reflections["wavelength"] = flex.double(wavelength_np.astype(np.float64))

    # --- bbox column with shift logic; z fixed to (0, 1) per single-frame ----
    x_int = flex.floor(reflections["xyzcal.px"].parts()[0]).iround()
    y_int = flex.floor(reflections["xyzcal.px"].parts()[1]).iround()

    bbox = flex.int6(len(reflections))
    for j in range(len(reflections)):
        x0, x1, y0, y1 = _shift_bbox_xy(x_int[j], y_int[j], nx, ny, dx_det, dy_det)
        bbox[j] = (x0, x1, y0, y1, 0, 1)
    reflections["bbox"] = bbox

    # --- refl_ids ------------------------------------------------------------
    reflections["refl_ids"] = flex.int(np.arange(len(reflections), dtype=np.int32))

    # --- save the reflection table (with wavelength + bbox + refl_ids) ------
    refl_path_out = out_dir / args.refl_fname
    reflections.as_file(str(refl_path_out))
    print(f"wrote refl with bbox/wavelength/refl_ids -> {refl_path_out}")

    # write identifiers for traceability
    identifier = dict(reflections.experiment_identifiers())
    (out_dir / "identifiers.yaml").write_text(yaml.safe_dump(identifier))

    # --- group refls by image_num for parallel extraction --------------------
    image_num_np = np.array(image_num)
    panels_np = np.array(reflections["panel"])
    bboxes_np = np.stack(
        [b.as_numpy_array() for b in reflections["bbox"].parts()],
        axis=-1,
    )  # (N, 6)
    refl_ids_np = np.array(reflections["refl_ids"])

    image_records: list[dict] = []
    for img_num in np.unique(image_num_np):
        sel = image_num_np == img_num
        image_records.append(
            {
                "expt_idx": img_to_expt[int(img_num)],
                "panels": panels_np[sel],
                "bboxes": bboxes_np[sel],
                "refl_ids": refl_ids_np[sel],
            }
        )
    print(f"prepared {len(image_records)} image records for extraction")

    # --- chunk and run in parallel -------------------------------------------
    block_size = args.block_size
    chunks = [
        image_records[i : i + block_size]
        for i in range(0, len(image_records), block_size)
    ]

    max_workers = min(args.max_workers, os.cpu_count() or 1)
    print(f"running {len(chunks)} chunks across {max_workers} workers")

    # pre-allocate output memmaps
    N = len(refl_ids_np)
    counts_path = out_dir / args.counts_fname
    masks_path = out_dir / args.masks_fname
    counts_dtype = np.dtype(args.counts_dtype)

    counts_mm = open_memmap(
        counts_path, mode="w+", dtype=counts_dtype, shape=(N, dz * dy * dx),
    )
    masks_mm = open_memmap(
        masks_path, mode="w+", dtype=np.bool_, shape=(N, dz * dy * dx),
    )

    dtype_max = (
        np.iinfo(counts_dtype).max if np.issubdtype(counts_dtype, np.integer) else None
    )
    clip_warned = False

    n_done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_image_chunk,
                chunk,
                str(expt_path_in),
                dz, dy, dx,
            )
            for chunk in chunks
        ]
        for f in as_completed(futures):
            for res in f.result():
                ids = res["refl_ids"]
                sbox = res["counts"]

                if dtype_max is not None:
                    over = sbox > dtype_max
                    if over.any():
                        if not clip_warned:
                            print(
                                f"WARNING: {int(over.sum())} pixel(s) exceed "
                                f"{counts_dtype} max ({dtype_max}); clipping. "
                                "Consider --counts-dtype int32 if overloads matter."
                            )
                            clip_warned = True
                        sbox = np.clip(sbox, 0, dtype_max)

                counts_mm[ids] = sbox.astype(counts_dtype, copy=False)
                masks_mm[ids] = res["masks"]
                n_done += len(ids)

    counts_mm.flush()
    masks_mm.flush()
    del counts_mm, masks_mm
    print(f"extracted {n_done} shoeboxes -> {counts_path}, {masks_path}")

    # --- metadata.pt via refl_as_pt (picks up wavelength via SCALAR_DTYPES) --
    refl_as_pt(refl=str(refl_path_out), out_dir=out_dir)
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


if __name__ == "__main__":
    main()
