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
    """
    N = len(bboxes)
    # Conservative upper bound on centroid-to-centroid distance at which
    # two bboxes of size (W, H, D) can share a pixel.
    max_r = float(np.sqrt(W * W + H * H + D * D))

    tree = cKDTree(centroids)
    overlap = np.zeros((N, D, H, W), dtype=bool)

    for i in _tqdm(range(N), desc="overlap detection"):
        x0, x1, y0, y1, z0, z1 = bboxes[i]
        my_c = centroids[i]

        for nid in tree.query_ball_point(my_c, r=max_r):
            if nid == i:
                continue
            nb = bboxes[nid]
            nb_c = centroids[nid]

            # Intersection in absolute detector coords
            ix0 = max(x0, nb[0])
            ix1 = min(x1, nb[1])
            iy0 = max(y0, nb[2])
            iy1 = min(y1, nb[3])
            iz0 = max(z0, nb[4])
            iz1 = min(z1, nb[5])

            if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
                continue

            # neighbor wins pixels closer to its centroid
            zs = np.arange(iz0, iz1)
            ys = np.arange(iy0, iy1)
            xs = np.arange(ix0, ix1)
            zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")

            d_me = (xx - my_c[0]) ** 2 + (yy - my_c[1]) ** 2 + (zz - my_c[2]) ** 2
            d_nb = (xx - nb_c[0]) ** 2 + (yy - nb_c[1]) ** 2 + (zz - nb_c[2]) ** 2
            nb_owned = d_nb < d_me

            overlap[
                i,
                iz0 - z0 : iz1 - z0,
                iy0 - y0 : iy1 - y0,
                ix0 - x0 : ix1 - x0,
            ] |= nb_owned

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
