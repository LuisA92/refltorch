"""Merge lattice and coset mksbox outputs into a single dataset.

Concatenates counts, masks, and metadata from two mksbox output directories
(lattice reflections + coset/background-only reflections), adds an `is_coset`
flag, recomputes Anscombe/stats/concentration over the merged data, and
reassigns contiguous refl_ids.

Usage:
    uv run python -m refltorch.cli.merge_coset \
        --lattice /path/to/lattice_pytorch_data \
        --coset   /path/to/coset_pytorch_data \
        --out     /path/to/merged_pytorch_data
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap


def parse_args():
    p = argparse.ArgumentParser(description="Merge lattice + coset mksbox outputs")
    p.add_argument("--lattice", type=Path, required=True, help="Lattice mksbox output dir")
    p.add_argument("--coset", type=Path, required=True, help="Coset mksbox output dir")
    p.add_argument("--out", type=Path, required=True, help="Merged output dir")
    p.add_argument(
        "--counts-fname", type=str, default="counts.npy",
        help="Name of counts file in each input dir",
    )
    p.add_argument(
        "--masks-fname", type=str, default="masks.npy",
        help="Name of masks file in each input dir",
    )
    p.add_argument("--chunk", type=int, default=10_000, help="Chunk size for streaming")
    p.add_argument(
        "--test-fraction", type=float, default=0.1,
        help="Test fraction to assign if is_test is missing from an input",
    )
    return p.parse_args()


def stream_copy(src_path: Path, dst_mm: np.ndarray, offset: int, chunk: int):
    """Copy src memmap into dst memmap starting at offset, in chunks."""
    src = np.load(src_path, mmap_mode="r")
    n = src.shape[0]
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        dst_mm[offset + i : offset + end] = src[i:end]
    return n


def compute_stats(counts_path: Path, masks_path: Path, chunk: int):
    """Streaming Anscombe + raw stats over merged memmap."""
    counts = np.load(counts_path, mmap_mode="r")
    masks = np.load(masks_path, mmap_mode="r")
    n = counts.shape[0]

    sum_c = 0.0
    sumsq_c = 0.0
    sum_a = 0.0
    sumsq_a = 0.0
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

    return (
        torch.tensor([mean_c, var_c], dtype=torch.float32),
        torch.tensor([mean_a, var_a], dtype=torch.float32),
    )


def compute_concentration(counts_path: Path, chunk: int):
    counts = np.load(counts_path, mmap_mode="r")
    n = counts.shape[0]
    conc = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        conc[i:end] = counts[i:end].astype(np.float32).mean(axis=1)
    return torch.from_numpy(conc)


def merge_metadata(
    lattice_dir: Path, coset_dir: Path, n_lattice: int, n_coset: int,
    test_fraction: float = 0.1,
):
    meta_lat = torch.load(lattice_dir / "metadata.pt", map_location="cpu", weights_only=False)
    meta_cos = torch.load(coset_dir / "metadata.pt", map_location="cpu", weights_only=False)

    rng = np.random.default_rng(42)

    # Assign is_test if missing
    if "is_test" not in meta_lat:
        meta_lat["is_test"] = torch.from_numpy(rng.random(n_lattice) < test_fraction)
        print(f"  assigned is_test to lattice: {meta_lat['is_test'].sum()}/{n_lattice}")
    if "is_test" not in meta_cos:
        meta_cos["is_test"] = torch.from_numpy(rng.random(n_coset) < test_fraction)
        print(f"  assigned is_test to coset: {meta_cos['is_test'].sum()}/{n_coset}")

    all_keys = set(meta_lat.keys()) | set(meta_cos.keys())
    merged = {}

    for k in sorted(all_keys):
        if k in meta_lat and k in meta_cos:
            merged[k] = torch.cat([meta_lat[k], meta_cos[k]], dim=0)
        elif k in meta_lat:
            fill = torch.zeros(n_coset, dtype=meta_lat[k].dtype)
            merged[k] = torch.cat([meta_lat[k], fill], dim=0)
        else:
            fill = torch.zeros(n_lattice, dtype=meta_cos[k].dtype)
            merged[k] = torch.cat([fill, meta_cos[k]], dim=0)

    n_total = n_lattice + n_coset
    merged["refl_ids"] = torch.arange(n_total, dtype=torch.float32)
    merged["is_coset"] = torch.cat([
        torch.zeros(n_lattice, dtype=torch.bool),
        torch.ones(n_coset, dtype=torch.bool),
    ])

    return merged


def main():
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_counts = np.load(args.lattice / args.counts_fname, mmap_mode="r")
    cos_counts = np.load(args.coset / args.counts_fname, mmap_mode="r")

    n_lat, d = lat_counts.shape
    n_cos, d_cos = cos_counts.shape
    assert d == d_cos, f"Shoebox size mismatch: lattice={d}, coset={d_cos}"
    n_total = n_lat + n_cos

    print(f"Lattice: {n_lat} reflections, Coset: {n_cos} reflections, Total: {n_total}")

    counts_dtype = lat_counts.dtype
    del lat_counts, cos_counts

    # Merge counts
    counts_path = out_dir / args.counts_fname
    counts_mm = open_memmap(counts_path, mode="w+", dtype=counts_dtype, shape=(n_total, d))
    print("Copying lattice counts...")
    stream_copy(args.lattice / args.counts_fname, counts_mm, 0, args.chunk)
    print("Copying coset counts...")
    stream_copy(args.coset / args.counts_fname, counts_mm, n_lat, args.chunk)
    counts_mm.flush()
    del counts_mm

    # Merge masks
    masks_path = out_dir / args.masks_fname
    masks_mm = open_memmap(masks_path, mode="w+", dtype=np.bool_, shape=(n_total, d))
    print("Copying lattice masks...")
    stream_copy(args.lattice / args.masks_fname, masks_mm, 0, args.chunk)
    print("Copying coset masks...")
    stream_copy(args.coset / args.masks_fname, masks_mm, n_lat, args.chunk)
    masks_mm.flush()
    del masks_mm

    # Merge metadata
    print("Merging metadata...")
    merged_meta = merge_metadata(args.lattice, args.coset, n_lat, n_cos, args.test_fraction)
    torch.save(merged_meta, out_dir / "metadata.pt")

    # Recompute stats
    print("Computing stats over merged data...")
    raw_stats, ans_stats = compute_stats(counts_path, masks_path, args.chunk)
    torch.save(raw_stats, out_dir / "stats.pt")
    torch.save(ans_stats, out_dir / "anscombe_stats.pt")
    print(f"  raw stats: mean={raw_stats[0]:.4f}, var={raw_stats[1]:.4f}")
    print(f"  anscombe:  mean={ans_stats[0]:.4f}, var={ans_stats[1]:.4f}")

    # Recompute concentration
    print("Computing concentration...")
    conc = compute_concentration(counts_path, args.chunk)
    torch.save(conc, out_dir / "concentration.pt")

    # Copy reflections_.refl from lattice (coset doesn't have meaningful DIALS integration)
    lat_refl = args.lattice / "reflections_.refl"
    if lat_refl.exists():
        import shutil
        shutil.copy2(lat_refl, out_dir / "reflections_.refl")
        print(f"Copied {lat_refl.name} from lattice dir")

    # Copy identifiers from lattice
    lat_ids = args.lattice / "identifiers.yaml"
    if lat_ids.exists():
        import shutil
        shutil.copy2(lat_ids, out_dir / "identifiers.yaml")

    print(f"\nMerged dataset written to {out_dir}")
    print(f"  {n_total} reflections ({n_lat} lattice + {n_cos} coset)")
    print(f"  is_coset flag added to metadata.pt")


if __name__ == "__main__":
    main()
