"""
Run with the following command:
'''bash
python ../refltorch/scripts/mksbox.py \
        --refl integrated.refl \
        --expt integrated.expt \
        --out-dir out_dir \
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
from dials.array_family import flex

from refltorch.refl_utils import refl_as_pt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract shoeboxes using dials.refl and dials.expt."
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
        help="Flag to save as pytorch.pt files",
    )
    parser.add_argument(
        "--counts-fname",
        type=str,
        default="counts.pt",
        help="Name of the output counts.pt file",
    )

    parser.add_argument(
        "--masks-fname",
        type=str,
        default="masks.pt",
        help="Name of the output masks.pt file",
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
    max_workers = min(4, cpu_count)

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


# def process_block(
#     block_indices,
#     bboxes,
#     panels,
#     refl_ids,
#     expt_path,
#     dz,
#     dy,
#     dx,
# ):
#     import numpy as np
#     from dxtbx.model.experiment_list import ExperimentListFactory
#
#     experiments = ExperimentListFactory.from_json_file(expt_path)
#     imageset = experiments[0].imageset
#
#     block_bboxes = bboxes[block_indices]
#     z0_block = int(block_bboxes[:, 4].min())
#     z1_block = int(block_bboxes[:, 5].max())
#
#     images = {}
#
#     for z in range(z0_block, z1_block):
#         raw = imageset.get_raw_data(z)[0]  # flex array
#         images[z] = raw.as_numpy_array()  # 2D numpy
#
#     n = len(block_indices)
#     shoeboxes = np.zeros(
#         (n, dz, dy, dx),
#         dtype=images[z0_block].dtype,
#     )
#     mask = np.zeros((n, dz, dy, dx), dtype=bool)
#
#     for i, idx in enumerate(block_indices):
#         x0, x1, y0, y1, z0, z1 = bboxes[idx]
#
#         z0_full = z0 - (dz - (z1 - z0)) // 2
#         y0_full = y0 - (dy - (y1 - y0)) // 2
#         x0_full = x0 - (dx - (x1 - x0)) // 2
#
#         oz0 = z0 - z0_full
#         oy0 = y0 - y0_full
#         ox0 = x0 - x0_full
#
#         for zz in range(z0, z1):
#             img = images[zz]
#             dz_idx = oz0 + (zz - z0)
#
#             shoeboxes[i, dz_idx, oy0 : oy0 + (y1 - y0), ox0 : ox0 + (x1 - x0)] = img[
#                 y0:y1, x0:x1
#             ]
#
#             mask[i, dz_idx, oy0 : oy0 + (y1 - y0), ox0 : ox0 + (x1 - x0)] = True
#
#     imageset.clear_cache()
#     shoeboxes = shoeboxes.reshape(shoeboxes.shape[0], dz * dy * dx)
#     mask = mask.reshape(mask.shape[0], dz * dy * dx)
#
#     return {
#         "shoeboxes": shoeboxes,
#         "mask": mask,
#         "refl_ids": refl_ids[block_indices],
#     }
#


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

    # for z in range(z0_block, z1_block):
    #     raw = imageset.get_raw_data(z)[0]
    #     images[z] = raw.as_numpy_array()
    #
    #     m = imageset.get_mask(z)[0]
    #     detmasks[z] = m.as_numpy_array().astype(bool)

    n = len(block_indices)
    # shoeboxes = np.zeros((n, dz, dy, dx), dtype=images[z0_block].dtype)
    any_z = next(iter(images))
    shoeboxes = np.zeros((n, dz, dy, dx), dtype=images[any_z].dtype)
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


# def _get_bounding_boxes(
#     x,
#     y,
#     z,
#     params,
#     reflections,
#     detector,
#     frame0,
#     frame1,
# ):
#     bbox = flex.int6(len(reflections))
#     dx_det, dy_det = detector[0].get_image_size()
#
#     for j, (_x, _y, _z) in enumerate(zip(x, y, z)):
#         # Calculate unclipped boundaries
#         x0_full = _x - params.nx
#         x1_full = _x + params.nx + 1
#         y0_full = _y - params.ny
#         y1_full = _y + params.ny + 1
#         z0_full = _z - params.nz
#         z1_full = _z + params.nz + 1
#
#         # Calculate full dimensions
#         full_width_x = x1_full - x0_full
#         full_width_y = y1_full - y0_full
#         full_width_z = z1_full - z0_full
#
#         # Check if any boundary is outside detector/frame and adjust
#         # X dimension adjustment
#         if x0_full < 0:
#             # Left edge is outside detector - shift right
#             x_shift = -x0_full  # Amount to shift right
#             x0 = 0
#             x1 = min(dx_det, x0 + full_width_x)  # Try to maintain width
#             # If still can't maintain full width, pad on left instead
#             if x1 - x0 < full_width_x:
#                 x1 = min(dx_det, x0_full + full_width_x)
#         elif x1_full >= dx_det:
#             # Right edge is outside detector - shift left
#             x_shift = dx_det - x1_full  # Amount to shift left (negative)
#             x1 = dx_det
#             x0 = max(0, x1 - full_width_x)  # Try to maintain width
#             # If still can't maintain full width, pad on right instead
#             if x1 - x0 < full_width_x:
#                 x0 = max(0, x1_full - full_width_x)
#         else:
#             # No adjustment needed
#             x0 = x0_full
#             x1 = x1_full
#
#         # Y dimension adjustment (similar to X)
#         if y0_full < 0:
#             y_shift = -y0_full
#             y0 = 0
#             y1 = min(dy_det, y0 + full_width_y)
#             if y1 - y0 < full_width_y:
#                 y1 = min(dy_det, y0_full + full_width_y)
#         elif y1_full >= dy_det:
#             y_shift = dy_det - y1_full
#             y1 = dy_det
#             y0 = max(0, y1 - full_width_y)
#             if y1 - y0 < full_width_y:
#                 y0 = max(0, y1_full - full_width_y)
#         else:
#             y0 = y0_full
#             y1 = y1_full
#
#         # Z dimension adjustment (similar to X and Y)
#         if z0_full < frame0:
#             z_shift = frame0 - z0_full
#             z0 = frame0
#             z1 = min(frame1, z0 + full_width_z)
#             if z1 - z0 < full_width_z:
#                 z1 = min(frame1, z0_full + full_width_z)
#         elif z1_full >= frame1:
#             z_shift = frame1 - z1_full
#             z1 = frame1
#             z0 = max(frame0, z1 - full_width_z)
#             if z1 - z0 < full_width_z:
#                 z0 = max(frame0, z1_full - full_width_z)
#         else:
#             z0 = z0_full
#             z1 = z1_full
#
#         bbox[j] = (x0, x1, y0, y1, z0, z1)
#
#     return bbox
#
#


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


def _save_as_pt(
    out_dir: str,
    counts: torch.Tensor,
    masks: torch.Tensor,
    counts_fname: str = "counts.pt",
    masks_fname: str = "masks.pt",
):
    out_dir_ = Path(out_dir)
    out_dir_.mkdir(parents=True, exist_ok=True)

    # setting filenames
    c = out_dir_ / counts_fname
    m = out_dir_ / masks_fname

    torch.save(counts, c)
    torch.save(masks, m)


def main():
    import torch
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

    # arguments and options
    params, options = parser.parse_args(
        [
            f"{args.refl}",
            f"{args.expt}",
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
        # detector=detector,
        # frame0=frame0,
        # frame1=frame1,
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

    # # save a reflection file
    # refl_fname = Path(args.out_dir) / "reflections_test.refl"
    # reflections.as_file(refl_fname)

    # Save a copy, but restore original order first
    perm = flex.sort_permutation(reflections["refl_ids"])
    refl_fname = Path(args.out_dir) / "reflections_test.refl"
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
        expt_path="integrated.expt",
        dz=dz,
        dy=dy,
        dx=dx,
    )

    # aggregate results
    N = len(refl_ids)
    shoeboxes_all = np.zeros((N, dz * dy * dx), dtype=np.float32)
    mask_all = np.zeros((N, dz * dy * dx), dtype=bool)

    for res in results:
        ids = res["refl_ids"]
        shoeboxes_all[ids] = res["shoeboxes"]
        mask_all[ids] = res["mask"]

    # to torch
    counts = torch.from_numpy(shoeboxes_all)
    masks = torch.from_numpy(mask_all)

    # save
    if args.save_as_pt:
        # save counts.pt and masks.pt
        _save_as_pt(
            out_dir=args.out_dir,
            counts=counts,
            masks=masks,
            counts_fname=args.counts_fname,
            masks_fname=args.masks_fname,
        )
        # save metadata.pt file
        refl_as_pt(
            refl=refl_fname.as_posix(),
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
