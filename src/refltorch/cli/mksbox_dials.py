"""Extract variable-size shoeboxes from DIALS debug output (integration.debug.output=True).

Use this instead of refltorch.mksbox when you want DIALS' full per-pixel masks
(Valid / Foreground / Background / Overlapped) rather than a fixed-size window
extraction. Expects the per-job debug files that DIALS writes when you run:

    dials.integrate refined.expt refined.refl \\
        integration.debug.output=True \\
        integration.debug.separate_files=True \\
        integration.debug.delete_shoeboxes=False

Which produces N `shoeboxes_<job>_<experiment>.refl` files — each a reflection
table for one integration job with a `shoebox` column included.

Outputs (into --out-dir):
    chunks/chunk_000.npz, chunk_001.npz, ...   — ragged per-chunk shoebox data
    reflections.refl                            — consolidated metadata (no shoeboxes), with refl_ids
    identifiers.yaml                            — experiment identifiers (matches mksbox)
    manifest.yaml                               — N reflections per chunk + MaskCode bit layout

Each chunk .npz contains:
    data:        uint16, flat concatenated pixel data across the chunk's reflections
    mask:        bool,   Valid & ~Overlapped  — use as training validity mask
    foreground:  bool,   Foreground bit only (useful for background-aware losses)
    background:  bool,   Background bit only
    overlapped:  bool,   Overlapped bit only  (pure, for debugging)
    offsets:     int64,  (n_refl + 1,)  — cumulative voxel offset into flat arrays
    shapes:      int32,  (n_refl, 3)    — (D, H, W) per reflection
    bboxes:      int32,  (n_refl, 6)    — (x0, x1, y0, y1, z0, z1) in detector coords
    refl_ids:    int64,  (n_refl,)      — globally assigned IDs, unique across chunks

Run:
    refltorch.mksbox-dials \\
        --shoebox-glob '/path/to/shoeboxes_*_*.refl' \\
        --out-dir /path/to/pytorch_data_dials
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import yaml


# MaskCode bits, from dials/src/dials/model/data/mask_code.h
MC_VALID = 1 << 0
MC_BACKGROUND = 1 << 1
MC_FOREGROUND = 1 << 2
MC_STRONG = 1 << 3
MC_BG_USED = 1 << 4
MC_OVERLAPPED = 1 << 5

MASKCODE_DOC = {
    "Valid": MC_VALID,
    "Background": MC_BACKGROUND,
    "Foreground": MC_FOREGROUND,
    "Strong": MC_STRONG,
    "BackgroundUsed": MC_BG_USED,
    "Overlapped": MC_OVERLAPPED,
}


def _tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--shoebox-glob",
        required=True,
        help="Glob pattern matching the shoebox files, e.g. 'path/to/shoeboxes_*_*.refl'",
    )
    ap.add_argument(
        "--out-dir", required=True, help="Output directory (will be created)"
    )
    ap.add_argument(
        "--counts-dtype",
        default="uint16",
        choices=["uint16", "int32"],
        help="Storage dtype for pixel counts",
    )
    ap.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of resolution bins for group_labels (used by hierarchical priors).",
    )
    ap.add_argument(
        "--min-per-bin",
        type=int,
        default=50,
        help="Minimum reflections per bin; n-bins is auto-reduced if below.",
    )
    return ap.parse_args()


def _bin_by_resolution(d, n_bins: int, min_per_bin: int = 50):
    """Quantile-bin reflections by resolution (mirrors prepare_priors._bin_by_resolution).

    Returns (group_labels, bin_edges, n_bins_actual).
    """
    import torch

    d = d.float()
    while n_bins > 1:
        quantiles = torch.linspace(0, 1, n_bins + 1)
        bin_edges = torch.quantile(d, quantiles)
        group_labels = torch.searchsorted(bin_edges[1:-1], d).long()
        counts_per_bin = torch.bincount(group_labels, minlength=n_bins)
        if counts_per_bin.min() >= min_per_bin:
            return group_labels, bin_edges, n_bins
        old_n = n_bins
        n_bins = max(1, n_bins - 1)
        print(f"  [bin] shrinking n_bins {old_n} -> {n_bins} (min bin < {min_per_bin})")
    group_labels = torch.zeros(len(d), dtype=torch.long)
    bin_edges = torch.tensor([d.min().item(), d.max().item()])
    return group_labels, bin_edges, 1


def _streaming_stats_from_chunks(chunk_paths):
    """Compute mean/var of raw counts, Anscombe counts, AND log1p counts
    across all chunks, restricted to mask-valid voxels. Returns a dict
    with 'raw', 'anscombe', and 'log1p' each mapping to (mean, var)
    float tensors plus an 'n_valid' count."""
    import torch

    sum_c = sumsq_c = 0.0
    sum_a = sumsq_a = 0.0
    sum_l = sumsq_l = 0.0
    n_valid = 0

    for p in chunk_paths:
        with np.load(p) as npz:
            data = npz["data"].astype(np.float64)      # may be uint16 / int32
            mask = npz["mask"]                          # bool
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        v = data[mask]                                  # (n_valid_chunk,)
        if v.size == 0:
            continue
        # Clip pre-transform to non-negative (raw can have -1 sentinels for
        # out-of-trusted-range pixels; the mask already excludes them).
        np.clip(v, 0, None, out=v)
        a = 2.0 * np.sqrt(v + 0.375)
        l = np.log1p(v)
        sum_c += float(v.sum())
        sumsq_c += float((v * v).sum())
        sum_a += float(a.sum())
        sumsq_a += float((a * a).sum())
        sum_l += float(l.sum())
        sumsq_l += float((l * l).sum())
        n_valid += int(v.size)

    if n_valid == 0:
        raise RuntimeError("No valid voxels found across chunks; cannot compute stats.")

    def _mv(s, ss):
        m = s / n_valid
        return m, max(ss / n_valid - m * m, 0.0)

    return {
        "raw": tuple(torch.tensor(x, dtype=torch.float32) for x in _mv(sum_c, sumsq_c)),
        "anscombe": tuple(torch.tensor(x, dtype=torch.float32) for x in _mv(sum_a, sumsq_a)),
        "log1p": tuple(torch.tensor(x, dtype=torch.float32) for x in _mv(sum_l, sumsq_l)),
        "n_valid": n_valid,
    }


def process_one_file(shoebox_path, chunk_idx, global_offset, out_chunks_dir, counts_dtype):
    """Read one shoeboxes_*_*.refl, write one chunk .npz, return (metadata_refl, n).

    metadata_refl: DIALS reflection_table with the `shoebox` column removed
                   and `refl_ids` added (global IDs starting from global_offset)
    n:             number of reflections in this chunk
    """
    from dials.array_family import flex

    refl = flex.reflection_table.from_file(shoebox_path)
    n = len(refl)
    if n == 0:
        return None, 0

    if "shoebox" not in refl:
        raise ValueError(
            f"{shoebox_path}: no 'shoebox' column — did you run dials.integrate "
            "with integration.debug.output=True?"
        )

    refl_ids = np.arange(global_offset, global_offset + n, dtype=np.int64)

    # Buffers for this chunk
    data_list = []
    valid_list = []
    fg_list = []
    bg_list = []
    ov_list = []
    shapes = np.empty((n, 3), dtype=np.int32)
    bboxes = np.empty((n, 6), dtype=np.int32)
    total_voxels = 0

    dtype_np = np.dtype(counts_dtype)
    dtype_max = np.iinfo(dtype_np).max
    clip_warned = False

    sbs = refl["shoebox"]
    for i, sb in enumerate(_tqdm(sbs, desc=f"chunk {chunk_idx:03d}")):
        # Derive (D, H, W) from the bbox — do NOT rely on np.asarray(sb.data).shape,
        # which flattens flex arrays to 1D. bbox is (x0, x1, y0, y1, z0, z1).
        x0, x1, y0, y1, z0, z1 = sb.bbox
        D = int(z1 - z0)
        H = int(y1 - y0)
        W = int(x1 - x0)
        V = D * H * W

        d_flat = np.asarray(sb.data).ravel()
        m_flat = np.asarray(sb.mask).ravel().astype(np.int32)

        if d_flat.size != V:
            raise RuntimeError(
                f"shoebox {i} at bbox ({x0},{x1},{y0},{y1},{z0},{z1}): "
                f"data has {d_flat.size} voxels, expected D*H*W = {V}"
            )

        # Reshape to 3D in (D, H, W) order — DIALS flex accessor is (nz, ny, nx)
        d = d_flat.reshape(D, H, W)
        m = m_flat.reshape(D, H, W)

        # Handle overflow on downcast to uint16
        if d.max() > dtype_max:
            if not clip_warned:
                print(
                    f"  [chunk {chunk_idx}] WARNING: pixel(s) above {counts_dtype} max "
                    f"({dtype_max}); clipping. Consider --counts-dtype int32."
                )
                clip_warned = True
            d = np.clip(d, 0, dtype_max)
        d = d.astype(dtype_np, copy=False)

        valid = (m & MC_VALID) != 0
        fg = (m & MC_FOREGROUND) != 0
        bg = (m & MC_BACKGROUND) != 0
        ov = (m & MC_OVERLAPPED) != 0
        train_mask = valid & ~ov

        data_list.append(d.ravel())
        valid_list.append(train_mask.ravel())
        fg_list.append(fg.ravel())
        bg_list.append(bg.ravel())
        ov_list.append(ov.ravel())
        shapes[i] = (D, H, W)
        bboxes[i] = [x0, x1, y0, y1, z0, z1]
        total_voxels += V

    # Concatenate to flat arrays + cumulative offsets
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum([d.size for d in data_list], out=offsets[1:])

    data_flat = np.concatenate(data_list) if data_list else np.empty(0, dtype=dtype_np)
    mask_flat = np.concatenate(valid_list) if valid_list else np.empty(0, dtype=bool)
    fg_flat = np.concatenate(fg_list) if fg_list else np.empty(0, dtype=bool)
    bg_flat = np.concatenate(bg_list) if bg_list else np.empty(0, dtype=bool)
    ov_flat = np.concatenate(ov_list) if ov_list else np.empty(0, dtype=bool)

    chunk_path = out_chunks_dir / f"chunk_{chunk_idx:03d}.npz"
    np.savez(
        chunk_path,
        data=data_flat,
        mask=mask_flat,
        foreground=fg_flat,
        background=bg_flat,
        overlapped=ov_flat,
        offsets=offsets,
        shapes=shapes,
        bboxes=bboxes,
        refl_ids=refl_ids,
    )

    # Metadata: drop shoebox column, add refl_ids
    del refl["shoebox"]
    refl["refl_ids"] = flex.int(refl_ids.tolist())

    return refl, n


def main():
    args = parse_args()

    from dials.array_family import flex

    shoebox_files = sorted(glob.glob(args.shoebox_glob))
    if not shoebox_files:
        raise SystemExit(f"No files matched: {args.shoebox_glob}")

    out_dir = Path(args.out_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(shoebox_files)} shoebox file(s):")
    for f in shoebox_files:
        print(f"  {f}")
    print(f"Output: {out_dir}")
    print(f"Chunks: {chunks_dir}")

    combined_refl = flex.reflection_table()
    per_chunk_counts = []
    global_offset = 0
    identifiers = {}

    for chunk_idx, f in enumerate(shoebox_files):
        meta_refl, n = process_one_file(
            f, chunk_idx, global_offset, chunks_dir, args.counts_dtype
        )
        if meta_refl is None:
            continue
        # Capture experiment identifiers (same as mksbox does)
        identifiers.update(dict(meta_refl.experiment_identifiers()))
        combined_refl.extend(meta_refl)
        per_chunk_counts.append({"chunk": chunk_idx, "source": f, "n": int(n)})
        global_offset += n

    # Consolidated metadata reflection file (no shoeboxes, refl_ids added)
    refl_out = out_dir / "reflections.refl"
    combined_refl.as_file(refl_out)
    print(f"\nWrote {refl_out}  ({len(combined_refl):,} reflections, no shoeboxes)")

    # Identifiers
    (out_dir / "identifiers.yaml").write_text(yaml.safe_dump(identifiers))

    # Manifest
    manifest = {
        "total_reflections": int(global_offset),
        "n_chunks": len(per_chunk_counts),
        "chunks": per_chunk_counts,
        "counts_dtype": args.counts_dtype,
        "maskcode_bits": MASKCODE_DOC,
        "mask_semantics": {
            "mask": "Valid & ~Overlapped (recommended training mask)",
            "foreground": "DIALS Foreground bit (profile interior)",
            "background": "DIALS Background bit (profile exterior, used for bg fit)",
            "overlapped": "DIALS Overlapped bit (belongs to a neighbor)",
        },
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"Wrote {out_dir / 'manifest.yaml'}")

    # ------------------------------------------------------------
    # metadata.pt — per-reflection scalars (d, miller_index, flags, etc.)
    # ------------------------------------------------------------
    from refltorch.refl_utils import refl_as_pt
    print("\nWriting metadata.pt from consolidated reflection file...")
    metadata = refl_as_pt(refl=str(refl_out), out_dir=out_dir)
    print(f"  metadata keys: {sorted(metadata.keys())}")
    if "d" not in metadata:
        print("  WARNING: metadata has no 'd' column — WilsonLoss will fail.")
    else:
        d = metadata["d"]
        print(f"  d: shape={tuple(d.shape)}, min={float(d.min()):.3f}, max={float(d.max()):.3f}")

    # ------------------------------------------------------------
    # stats.pt + anscombe_stats.pt — for encoder-input standardization
    # ------------------------------------------------------------
    import torch as _torch
    print("\nComputing dataset stats (streaming over chunks)...")
    chunk_paths = sorted(chunks_dir.glob("chunk_*.npz"))
    stats = _streaming_stats_from_chunks(chunk_paths)
    raw_mean, raw_var = stats["raw"]
    ans_mean, ans_var = stats["anscombe"]
    log_mean, log_var = stats["log1p"]
    _torch.save(_torch.tensor([raw_mean, raw_var], dtype=_torch.float32), out_dir / "stats.pt")
    _torch.save(_torch.tensor([ans_mean, ans_var], dtype=_torch.float32), out_dir / "anscombe_stats.pt")
    _torch.save(_torch.tensor([log_mean, log_var], dtype=_torch.float32), out_dir / "log1p_stats.pt")
    print(f"  raw    mean={raw_mean:.3f}, var={raw_var:.3f}  (over {stats['n_valid']:,} valid voxels)")
    print(f"  ans    mean={ans_mean:.3f}, var={ans_var:.3f}")
    print(f"  log1p  mean={log_mean:.3f}, var={log_var:.3f}")

    # ------------------------------------------------------------
    # group_labels — resolution bins used by hierarchical priors
    # ------------------------------------------------------------
    if "d" in metadata and args.n_bins > 0:
        print(f"\nBinning by resolution into {args.n_bins} group(s)...")
        group_labels, bin_edges, n_bins_actual = _bin_by_resolution(
            metadata["d"], args.n_bins, args.min_per_bin
        )
        gl_name = f"group_labels_{n_bins_actual}.pt"
        edges_name = f"bin_edges_{n_bins_actual}.pt"
        _torch.save(group_labels, out_dir / gl_name)
        _torch.save(bin_edges, out_dir / edges_name)
        print(f"  wrote {gl_name} (final n_bins={n_bins_actual})")
        print(f"  wrote {edges_name}")
        counts_per_bin = _torch.bincount(group_labels, minlength=n_bins_actual).tolist()
        print(f"  refl per bin: {counts_per_bin}")

    print("\nDone.")


if __name__ == "__main__":
    main()
