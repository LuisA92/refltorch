"""Post-process refltorch mksbox masks to mark pixels "owned" by a neighbor
reflection rather than the target reflection.

Operates on existing mksbox outputs (counts/masks already extracted). The
pixel-data array is not modified; only the mask is updated. This is a cheap
stage-1 fix for overlap contamination in fixed-size shoebox training data.

How ownership works: when two reflections' bboxes share a pixel, the pixel is
owned by whichever reflection's predicted centroid is closer. This mirrors
dials/algorithms/shoebox/mask_overlapping.h:108-166.

Run (with a DIALS-enabled env so dials.array_family is importable):

    refltorch.add-overlap \
        --refl  /path/to/reflections_.refl \
        --masks /path/to/masks.npy \
        --out   /path/to/masks_overlap.npy \
        --h 33 --w 33 --d 9

Supports both .npy (memmap era) and .pt (legacy) input/output formats; the
output format matches the input format unless --out is given with a different
extension.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _load_masks(path):
    """Load masks from .npy or .pt. Returns (ndarray, fmt)."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p), "npy"
    if p.suffix == ".pt":
        import torch

        return torch.load(p).numpy(), "pt"
    raise ValueError(f"unknown mask extension: {p.suffix}")


def _save_masks(arr, path, fmt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path, arr)
    elif fmt == "pt":
        import torch

        torch.save(torch.from_numpy(arr), path)
    else:
        raise ValueError(f"unknown format: {fmt}")


def compute_overlap_mask(bboxes, centroids, D, H, W):
    """For each reflection i, return a (D, H, W) bool array — True at pixels
    within its fixed shoebox window that are owned by another reflection.

    bboxes:    (N, 6)  ints  — (x0, x1, y0, y1, z0, z1) per reflection
    centroids: (N, 3)  float — (x, y, z) predicted centroid (xyzcal.px)

    Uses per-image batching: groups reflections by z-slice (image),
    then vectorizes the overlap computation within each image.
    """
    N = len(bboxes)
    overlap = np.zeros((N, D, H, W), dtype=bool)

    # Group reflections by image (z0 value) for batch processing
    z0_vals = bboxes[:, 4]
    unique_z = np.unique(z0_vals)

    # Precompute pixel grids relative to shoebox origin
    # For fixed-size shoeboxes, all have the same local grid
    ly = np.arange(H, dtype=np.float32)
    lx = np.arange(W, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ly, lx, indexing="ij")  # (H, W)

    for z in _tqdm(unique_z, desc="overlap detection (per image)"):
        idx = np.where(z0_vals == z)[0]
        n_img = len(idx)
        if n_img < 2:
            continue

        # Centroids for this image (x, y only — z is the same)
        cx = centroids[idx, 0].astype(np.float32)
        cy = centroids[idx, 1].astype(np.float32)
        bx0 = bboxes[idx, 0].astype(np.float32)
        by0 = bboxes[idx, 2].astype(np.float32)

        # Build a 2D KDTree for this image's centroids
        tree_2d = cKDTree(np.column_stack([cx, cy]))
        max_r = float(np.sqrt(W * W + H * H))

        # Query all pairs within range
        pairs = tree_2d.query_pairs(r=max_r, output_type="ndarray")
        if len(pairs) == 0:
            continue

        ii = pairs[:, 0]
        jj = pairs[:, 1]

        # Check bbox overlap for all pairs at once
        # bbox intersection in absolute coords
        ix0 = np.maximum(bx0[ii], bx0[jj])
        ix1 = np.minimum(bx0[ii] + W, bx0[jj] + W)
        iy0 = np.maximum(by0[ii], by0[jj])
        iy1 = np.minimum(by0[ii] + H, by0[jj] + H)

        has_overlap = (ix0 < ix1) & (iy0 < iy1)
        pairs_ov = np.where(has_overlap)[0]

        for p in pairs_ov:
            i_local, j_local = ii[p], jj[p]
            i_global, j_global = idx[i_local], idx[j_local]

            ox0, ox1 = int(ix0[p]), int(ix1[p])
            oy0, oy1 = int(iy0[p]), int(iy1[p])

            # Pixel coords in the overlap region
            xs = np.arange(ox0, ox1, dtype=np.float32)
            ys = np.arange(oy0, oy1, dtype=np.float32)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")

            d_i = (xx - cx[i_local]) ** 2 + (yy - cy[i_local]) ** 2
            d_j = (xx - cx[j_local]) ** 2 + (yy - cy[j_local]) ** 2

            # j owns pixels closer to j — mark them in i's mask
            j_wins = d_j < d_i
            i_y0 = int(by0[i_local])
            i_x0 = int(bx0[i_local])
            overlap[
                i_global, 0,
                oy0 - i_y0 : oy1 - i_y0,
                ox0 - i_x0 : ox1 - i_x0,
            ] |= j_wins

            # i owns pixels closer to i — mark them in j's mask
            j_y0 = int(by0[j_local])
            j_x0 = int(bx0[j_local])
            overlap[
                j_global, 0,
                oy0 - j_y0 : oy1 - j_y0,
                ox0 - j_x0 : ox1 - j_x0,
            ] |= ~j_wins

    return overlap


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--refl", required=True, help="Path to reflections_.refl (has bbox + xyzcal.px)"
    )
    ap.add_argument(
        "--masks", required=True, help="Path to existing masks.npy or masks.pt"
    )
    ap.add_argument(
        "--out", default=None, help="Output path (default: <masks>_overlap.<ext>)"
    )
    ap.add_argument("--h", type=int, required=True, help="Shoebox height")
    ap.add_argument("--w", type=int, required=True, help="Shoebox width")
    ap.add_argument("--d", type=int, required=True, help="Shoebox depth (frames)")
    return ap.parse_args()


def main():
    args = parse_args()

    from dials.array_family import flex

    refl = flex.reflection_table.from_file(args.refl)
    bboxes = np.asarray(refl["bbox"]).reshape(-1, 6).astype(np.int64)
    xyzcal = np.asarray([list(v) for v in refl["xyzcal.px"]], dtype=np.float64)

    N = len(bboxes)
    D, H, W = args.d, args.h, args.w
    V = D * H * W

    print(f"Reflections: {N:,}")
    print(f"Shoebox (D, H, W) = ({D}, {H}, {W}), V = {V:,}")

    # Sanity-check bbox size matches H/W/D
    dx = bboxes[:, 1] - bboxes[:, 0]
    dy = bboxes[:, 3] - bboxes[:, 2]
    dz = bboxes[:, 5] - bboxes[:, 4]
    if not (np.all(dx == W) and np.all(dy == H) and np.all(dz == D)):
        print(
            "WARNING: refl bboxes are not all the requested (W, H, D). "
            f"This file may have DIALS bboxes rather than refltorch's fixed windows. "
            f"dx: min={dx.min()} max={dx.max()} expected={W}; "
            f"dy: min={dy.min()} max={dy.max()} expected={H}; "
            f"dz: min={dz.min()} max={dz.max()} expected={D}."
        )

    masks, fmt = _load_masks(args.masks)
    if masks.ndim != 2 or masks.shape != (N, V):
        raise ValueError(
            f"mask shape {masks.shape} doesn't match ({N}, {V}); "
            "check H/W/D args and the reflection file."
        )

    overlap = compute_overlap_mask(bboxes, xyzcal, D, H, W)
    overlap_flat = overlap.reshape(N, V)

    print("Applying overlap to existing mask...")
    new_masks = masks.astype(bool) & ~overlap_flat

    # Stats
    overlap_frac = overlap_flat.mean(-1)
    old_valid = masks.astype(bool).mean(-1)
    new_valid = new_masks.mean(-1)

    print("\n=== Overlap statistics ===")
    print(f"  Mean overlap per refl:         {overlap_frac.mean() * 100:.2f}%")
    print(f"  Median overlap per refl:       {np.median(overlap_frac) * 100:.2f}%")
    print(
        f"  Refl with any overlap:         {(overlap_frac > 0).sum():,} / {N:,}  "
        f"({(overlap_frac > 0).mean() * 100:.1f}%)"
    )
    print(f"  Refl with >10% overlap:        {(overlap_frac > 0.10).sum():,}")
    print(f"  Refl with >30% overlap:        {(overlap_frac > 0.30).sum():,}")
    print(f"  Refl with >50% overlap:        {(overlap_frac > 0.50).sum():,}")
    print()
    print(f"  Valid-pixel fraction (before): {old_valid.mean() * 100:.2f}%")
    print(f"  Valid-pixel fraction (after):  {new_valid.mean() * 100:.2f}%")

    if args.out is None:
        p = Path(args.masks)
        out_path = p.with_name(f"{p.stem}_overlap{p.suffix}")
    else:
        out_path = Path(args.out)
    _save_masks(new_masks, out_path, fmt)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
