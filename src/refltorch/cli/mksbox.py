"""
Run with the following command:
'''bash
refltorch.mksbox \
        --data-dir /n/hekstra_lab/people/aldama/integrator_data/hewl_9b7c/dials \
        --refl integrated.refl \
        --expt integrated.expt \
        --out-dir /n/hekstra_lab/people/aldama/integrator_data/hewl_9b7c/pytorch_data \
        --w 21 \
        --h 21 \
        --d 3 \
        --save-as-pt
'''

"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from refltorch.refl_utils import refl_as_pt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract shoeboxes using dials.refl and dials.expt."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to directory containing integrated.refl and integrated.expt files",
    )
    parser.add_argument(
        "--refl",
        type=str,
        help="Path to the dials.refl file",
    )
    parser.add_argument(
        "--refl-fname",
        type=str,
        default="reflections_.refl",
        help="Filename of output reflection file",
    )
    parser.add_argument(
        "--expt",
        type=str,
        help="Path to the dials.expt file",
    )
    parser.add_argument(
        "--w",
        type=int,
        help="Number of pixels along shoebox horizontal axis",
    )
    parser.add_argument(
        "--h",
        type=int,
        help="Number of pixels along shoebox vertical axis",
    )
    parser.add_argument(
        "--d",
        type=int,
        help="Number of pixels along shoebox rotation axis",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out_dir",
        help="Path to output directory",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Number of images to process per job",
    )
    parser.add_argument(
        "--save-as-pt",
        action="store_true",
        help="Also write stats and concentration .pt files alongside counts/masks .npy",
    )
    parser.add_argument(
        "--counts-fname",
        type=str,
        default="counts.npy",
        help="Name of the output counts file (memmap-backed .npy)",
    )

    parser.add_argument(
        "--masks-fname",
        type=str,
        default="masks.npy",
        help="Name of the output masks file (memmap-backed .npy)",
    )

    parser.add_argument(
        "--counts-dtype",
        type=str,
        default="uint16",
        choices=["uint16", "int32", "float32"],
        help="Storage dtype for counts (uint16 halves RAM/disk vs float32)",
    )

    parser.add_argument(
        "--stats-chunk",
        type=int,
        default=10_000,
        help="Reflections per chunk when computing stats from the memmap",
    )
    return parser.parse_args()


def run_all_blocks(
    blocks,
    bboxes,
    panels,
    refl_ids,
    expt_path,
    dz,
    dy,
    dx,
):
    cpu_count = os.cpu_count() or 1
    max_workers = min(8, cpu_count)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_block,
                block,
                bboxes,
                panels,
                refl_ids,
                expt_path,
                dz,
                dy,
                dx,
            )
            for block in blocks
        ]

        for f in as_completed(futures):
            results.append(f.result())

    return results


def process_block(
    block_indices,
    bboxes_full,  # IMPORTANT: these are FULL boxes, may go out of bounds
    panels,
    refl_ids,
    expt_path,
    dz,
    dy,
    dx,
):
    import numpy as np
    from dxtbx.model.experiment_list import ExperimentListFactory

    experiments = ExperimentListFactory.from_json_file(expt_path)
    imageset = experiments[0].imageset

    # detector size (single-panel assumption here; extend if multi-panel)
    det = imageset.get_detector()[0]
    dx_det, dy_det = det.get_image_size()

    block_boxes = bboxes_full[block_indices]
    z0_block = int(block_boxes[:, 4].min())
    z1_block = int(block_boxes[:, 5].max())

    # preload images + masks for needed z
    images = {}
    detmasks = {}

    scan = imageset.get_scan()
    frame0, frame1 = scan.get_array_range()

    z_load0 = max(frame0, z0_block)
    z_load1 = min(frame1, z1_block)

    for z in range(z_load0, z_load1):
        raw = imageset.get_raw_data(z)[0]
        images[z] = raw.as_numpy_array()

        m = imageset.get_mask(z)[0]
        detmasks[z] = m.as_numpy_array().astype(bool)

    n = len(block_indices)

    if images:
        any_z = next(iter(images))
        dtype = images[any_z].dtype
    else:
        dtype = np.float32

    shoeboxes = np.zeros((n, dz, dy, dx), dtype=dtype)
    mask = np.zeros((n, dz, dy, dx), dtype=bool)

    for i, idx in enumerate(block_indices):
        x0f, x1f, y0f, y1f, z0f, z1f = bboxes_full[idx]

        # destination is always [0:dx), [0:dy), [0:dz)
        # source is [x0f:x1f), etc, but may go out of bounds

        for zz in range(z0f, z1f):
            if zz not in images:
                continue

            # clip source range to detector bounds
            xs0 = max(0, x0f)
            xs1 = min(dx_det, x1f)
            ys0 = max(0, y0f)
            ys1 = min(dy_det, y1f)

            if xs0 >= xs1 or ys0 >= ys1:
                continue

            # destination offsets (where clipped source lands inside the full box)
            xd0 = xs0 - x0f
            yd0 = ys0 - y0f
            zd = zz - z0f

            img = images[zz]
            dm = detmasks[zz]

            patch = img[ys0:ys1, xs0:xs1]
            dm_patch = dm[ys0:ys1, xs0:xs1]

            valid = (patch >= 0) & dm_patch

            shoeboxes[i, zd, yd0 : yd0 + patch.shape[0], xd0 : xd0 + patch.shape[1]] = (
                patch
            )

            mask[i, zd, yd0 : yd0 + patch.shape[0], xd0 : xd0 + patch.shape[1]] = valid

    imageset.clear_cache()

    shoeboxes = shoeboxes.reshape(n, dz * dy * dx)
    mask = mask.reshape(n, dz * dy * dx)

    return {
        "shoeboxes": shoeboxes,
        "mask": mask,
        "refl_ids": refl_ids[block_indices],
    }


def _get_bounding_boxes(
    x,
    y,
    z,
    params,
    reflections,
):
    """
    Return FULL, CENTERED bounding boxes.
    These boxes are NOT clipped to the detector or frame range.
    Padding is handled later during extraction.
    """
    from dials.array_family import flex

    bbox = flex.int6(len(reflections))

    for j, (_x, _y, _z) in enumerate(zip(x, y, z)):
        x0 = _x - params.nx
        x1 = _x + params.nx + 1

        y0 = _y - params.ny
        y1 = _y + params.ny + 1

        z0 = _z - params.nz
        z1 = _z + params.nz + 1

        bbox[j] = (x0, x1, y0, y1, z0, z1)

    return bbox


def _get_blocks(
    block_ids,
) -> list:
    # Get blocks
    blocks = []
    start = 0

    for i in range(1, len(block_ids)):
        if block_ids[i] != block_ids[start]:
            blocks.append(np.arange(start, i))
            start = i

    blocks.append(np.arange(start, len(block_ids)))

    return blocks


def anscombe_transform(
    tensor: torch.Tensor,
) -> torch.Tensor:
    return 2 * (tensor + 0.375).sqrt()


def _save_stats_from_memmap(
    counts_path: Path,
    masks_path: Path,
    out_dir: Path,
    chunk: int = 10_000,
    ans_fname: str = "anscombe_stats.pt",
    stats_fname: str = "stats.pt",
):
    """Compute (mean, var) of masked counts and of their Anscombe transform
    by streaming the on-disk memmap in chunks. Never loads the full array."""
    counts = np.load(counts_path, mmap_mode="r")
    masks = np.load(masks_path, mmap_mode="r")
    n, d = counts.shape

    sum_c = 0.0
    sumsq_c = 0.0
    sum_a = 0.0
    sumsq_a = 0.0
    nel = 0

    for i in range(0, n, chunk):
        c = counts[i : i + chunk].astype(np.float64)
        m = masks[i : i + chunk]
        c = c * m  # zero out masked pixels, matches prior semantics
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
        torch.tensor([mean_c, var_c], dtype=torch.float32),
        out_dir / stats_fname,
    )
    torch.save(
        torch.tensor([mean_a, var_a], dtype=torch.float32),
        out_dir / ans_fname,
    )


def _save_concentration_from_memmap(
    counts_path: Path,
    out_dir: Path,
    chunk: int = 10_000,
    out_fname: str = "concentration.pt",
):
    """Per-reflection mean over voxels, computed by streaming the memmap."""
    counts = np.load(counts_path, mmap_mode="r")
    n = counts.shape[0]

    conc = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        c = counts[i : i + chunk].astype(np.float32)
        conc[i : i + chunk] = c.mean(axis=1)

    torch.save(torch.from_numpy(conc), out_dir / out_fname)


def main():
    import torch
    import yaml
    from dials.array_family import flex
    from dials.util import Sorry
    from dials.util.options import (
        ArgumentParser,
        flatten_experiments,
        flatten_reflections,
    )
    from libtbx.phil import parse

    args = parse_args()

    phil_scope = parse(
        """
      nx = 1
        .type = int(value_min=1)
        .help = "+/- x around centroid"

      ny = 1
        .type = int(value_min=1)
        .help = "+/- y around centroid"

      nz = 1
        .type = int(value_min=1)
        .help = "+/- z around centroid"
      output {
        reflections = 'shoeboxes.refl'
          .type = str
          .help = "The integrated output filename"
      }
    """
    )

    parser = ArgumentParser(
        phil=phil_scope,
        read_experiments=True,
        read_reflections=True,
    )

    if args.w % 2 == 1:
        nx = args.w // 2
    else:
        raise ValueError(f"Width must be an odd integer: w={args.w}")

    if args.h % 2 == 1:
        ny = args.h // 2
    else:
        raise ValueError(f"Height must be an odd integer: h={args.h}")

    if args.d % 2 == 1:
        nz = args.d // 2
    else:
        raise ValueError(f"Depth must be an odd integer: d={args.d}")

    data_dir = Path(args.data_dir)
    refl = data_dir / args.refl
    expt = data_dir / args.expt

    # Writing out dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # arguments and options
    params, options = parser.parse_args(
        [
            f"{refl}",
            f"{expt}",
            f"nx={nx}",
            f"ny={ny}",
            f"nz={nz}",
        ]
    )

    # reflections and experiments
    reflections = flatten_reflections(
        params.input.reflections,
    )

    experiments = flatten_experiments(params.input.experiments)

    if not any([experiments, reflections]):
        parser.print_help()
        exit(0)
    elif len(experiments) > 1:
        raise Sorry("More than 1 experiment set")
    elif len(experiments) == 1:
        imageset = experiments[0].imageset

    if len(reflections) != 1:
        raise Sorry("Need 1 reflection table, got %d" % len(reflections))
    else:
        reflections = reflections[0]

    identifier = dict(reflections.experiment_identifiers())

    id_fname = out_dir / "identifiers.yaml"
    id_fname.write_text(yaml.safe_dump(identifier))

    # Check the reflections contain the necessary stuff
    assert "bbox" in reflections
    assert "panel" in reflections

    # Get some models
    detector = imageset.get_detector()
    scan = imageset.get_scan()
    frame0, frame1 = scan.get_array_range()

    x, y, z = reflections["xyzcal.px"].parts()

    x = flex.floor(x).iround()
    y = flex.floor(y).iround()
    z = flex.floor(z).iround()

    bbox = _get_bounding_boxes(
        x=x,
        y=y,
        z=z,
        params=params,
        reflections=reflections,
    )

    # Store bboxes into reflection file
    reflections["bbox"] = bbox

    # assign ids
    reflections["refl_ids"] = flex.int(np.arange(len(reflections)))

    # add z_px and sort
    reflections["z_px"] = reflections["xyzcal.px"].parts()[2]
    reflections.sort("z_px")

    # extract as numpy
    bbox_sorted = reflections["bbox"]
    bboxes = np.stack([x.as_numpy_array() for x in bbox_sorted.parts()]).T
    panels = reflections["panel"].as_numpy_array()
    refl_ids = reflections["refl_ids"].as_numpy_array()

    # image ranges and centroids
    z0 = bboxes[:, 4]
    z1 = bboxes[:, 5]
    zc = (z0 + z1) // 2

    # assign block block ids
    BLOCK_SIZE = args.block_size
    block_ids = zc // BLOCK_SIZE
    reflections["block_ids"] = flex.int(block_ids)

    # Save a copy, but restore original order first
    perm = flex.sort_permutation(reflections["refl_ids"])
    refl_fname = out_dir / args.refl_fname
    reflections.reorder(perm)
    reflections.as_file(refl_fname)

    # Get blocks of block ids
    blocks = _get_blocks(block_ids)

    dz = 2 * params.nz + 1
    dy = 2 * params.ny + 1
    dx = 2 * params.nx + 1

    results = run_all_blocks(
        blocks,
        bboxes,
        panels,
        refl_ids,
        expt_path=expt,
        dz=dz,
        dy=dy,
        dx=dx,
    )

    # aggregate results into on-disk memmaps — never materialize the full
    # (N, dz*dy*dx) array in RAM. This is the fix for the OOM at large shoeboxes.
    N = len(refl_ids)
    counts_path = out_dir / args.counts_fname
    masks_path = out_dir / args.masks_fname

    counts_dtype = np.dtype(args.counts_dtype)
    shoeboxes_all = open_memmap(
        counts_path,
        mode="w+",
        dtype=counts_dtype,
        shape=(N, dz * dy * dx),
    )
    mask_all = open_memmap(
        masks_path,
        mode="w+",
        dtype=np.bool_,
        shape=(N, dz * dy * dx),
    )

    # uint16 overflow guard: warn once if any pixel exceeds its range
    dtype_max = np.iinfo(counts_dtype).max if np.issubdtype(counts_dtype, np.integer) else None
    clip_warned = False

    for res in results:
        ids = res["refl_ids"]
        sbox = res["shoeboxes"]

        if dtype_max is not None:
            over = sbox > dtype_max
            if over.any():
                if not clip_warned:
                    print(
                        f"WARNING: {int(over.sum())} pixel(s) exceed {counts_dtype} max ({dtype_max}); "
                        f"clipping. Consider --counts-dtype int32 if overloads matter to you."
                    )
                    clip_warned = True
                sbox = np.clip(sbox, 0, dtype_max)

        shoeboxes_all[ids] = sbox.astype(counts_dtype, copy=False)
        mask_all[ids] = res["mask"]

    shoeboxes_all.flush()
    mask_all.flush()
    del shoeboxes_all, mask_all

    # save metadata.pt file (reads from refl file, small)
    refl_as_pt(
        refl=refl_fname.as_posix(),
        out_dir=out_dir,
    )

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


if __name__ == "__main__":
    main()
