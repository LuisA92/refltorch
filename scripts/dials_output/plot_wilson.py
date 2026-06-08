"""Plot learned Wilson K and B factors, and per-bin tau diagnostics.

Loads the model checkpoint to extract the variational posterior q(log K)
and q(log B), then compares the Wilson-predicted tau per resolution bin
against the empirical tau.

Usage:
    python plot_wilson.py \
        --run-dir /path/to/run_dir/ \
        [--save-dir /path/to/output/] \
        [--epoch 74]
"""

import argparse
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from refltorch.io import open_run
from refltorch.plots import save_figure


def _load_checkpoint(log_dir: Path, epoch: int | None) -> dict:
    """Load a checkpoint, defaulting to the latest epoch."""
    ckpts = sorted(log_dir.glob("**/epoch*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")

    epoch_re = re.compile(r"epoch=(\d+)")
    ckpt_map = {}
    for c in ckpts:
        m = epoch_re.search(c.name)
        if m:
            ckpt_map[int(m.group(1))] = c

    if epoch is not None and epoch in ckpt_map:
        ckpt_path = ckpt_map[epoch]
    else:
        latest = max(ckpt_map.keys())
        ckpt_path = ckpt_map[latest]
        epoch = latest

    print(f"Loading checkpoint: {ckpt_path.name} (epoch {epoch})")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return state, epoch


def _extract_wilson_params(state_dict: dict) -> dict:
    """Extract Wilson parameters from checkpoint state_dict."""
    # Keys are prefixed with 'loss.' in the lightning state_dict
    prefix = "loss."
    keys = {
        "q_log_K_loc": None,
        "q_log_K_log_scale": None,
        "q_log_B_loc": None,
        "q_log_B_log_scale": None,
        "s_squared_per_group": None,
        "tau_per_group": None,
        "bg_rate_per_group": None,
        "concentration_per_group": None,
        "hp_log_K_loc": None,
        "hp_log_K_scale": None,
        "hp_log_B_loc": None,
        "hp_log_B_scale": None,
    }

    for k in keys:
        full_key = prefix + k
        if full_key in state_dict:
            keys[k] = state_dict[full_key]

    # Check we have the Wilson params
    if keys["q_log_K_loc"] is None:
        raise ValueError(
            "No Wilson parameters found in checkpoint. "
            "Is this a wilson_per_bin model?"
        )

    return keys


def _compute_wilson_stats(params: dict) -> dict:
    """Compute K, B means/stds and Wilson-predicted tau."""
    mu_K = params["q_log_K_loc"].item()
    mu_B = params["q_log_B_loc"].item()
    s_K = F.softplus(params["q_log_K_log_scale"]).item()
    s_B = F.softplus(params["q_log_B_log_scale"]).item()

    # LogNormal moments
    K_mean = np.exp(mu_K + 0.5 * s_K**2)
    B_mean = np.exp(mu_B + 0.5 * s_B**2)
    K_std = K_mean * np.sqrt(np.expm1(s_K**2))
    B_std = B_mean * np.sqrt(np.expm1(s_B**2))

    # Wilson-predicted tau per bin: tau = (1/K) * exp(2B * s^2)
    s_sq = params["s_squared_per_group"].numpy()
    tau_empirical = params["tau_per_group"].numpy() if params["tau_per_group"] is not None else None
    bg_rate = params["bg_rate_per_group"].numpy()

    # Use mean K, B for point estimate of tau
    tau_wilson = (1.0 / K_mean) * np.exp(2.0 * B_mean * s_sq)

    # d-spacing from s^2: d = 1 / (2 * sqrt(s^2))
    d_per_bin = 1.0 / (2.0 * np.sqrt(s_sq + 1e-12))

    return {
        "K_mean": K_mean, "K_std": K_std,
        "B_mean": B_mean, "B_std": B_std,
        "mu_K": mu_K, "s_K": s_K,
        "mu_B": mu_B, "s_B": s_B,
        "s_sq": s_sq,
        "d_per_bin": d_per_bin,
        "tau_empirical": tau_empirical,
        "tau_wilson": tau_wilson,
        "bg_rate": bg_rate,
        # Hyperprior
        "hp_K_loc": params["hp_log_K_loc"].item() if params["hp_log_K_loc"] is not None else None,
        "hp_K_scale": params["hp_log_K_scale"].item() if params["hp_log_K_scale"] is not None else None,
        "hp_B_loc": params["hp_log_B_loc"].item() if params["hp_log_B_loc"] is not None else None,
        "hp_B_scale": params["hp_log_B_scale"].item() if params["hp_log_B_scale"] is not None else None,
    }


def _plot_tau_comparison(stats: dict, epoch: int, save_dir: Path):
    """Plot empirical vs Wilson-predicted tau as a function of resolution."""
    s_sq = stats["s_sq"]
    tau_w = stats["tau_wilson"]
    tau_e = stats["tau_empirical"]

    sort_idx = np.argsort(s_sq)
    s_sq = s_sq[sort_idx]
    tau_w = tau_w[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(s_sq, tau_w, "o-", color="blue", label="Wilson predicted", markersize=4)
    if tau_e is not None:
        tau_e = tau_e[sort_idx]
        ax.plot(s_sq, tau_e, "s--", color="red", label="Empirical", markersize=4)

    ax.set_xlabel("$s^2 = 1/(4d^2)$ ($\\AA^{-2}$)")
    ax.set_ylabel("$\\tau$ (intensity prior rate)")
    ax.set_title(
        f"Wilson tau per resolution bin | Epoch {epoch}\n"
        f"K = {stats['K_mean']:.4f} $\\pm$ {stats['K_std']:.4f}, "
        f"B = {stats['B_mean']:.2f} $\\pm$ {stats['B_std']:.2f}"
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add d-spacing as secondary x-axis with fixed tick positions
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    d_ticks = np.array([1.0, 1.2, 1.5, 2.0, 3.0, 5.0])
    s2_ticks = 1.0 / (4.0 * d_ticks**2)
    s2_lo, s2_hi = ax.get_xlim()
    mask = (s2_ticks >= s2_lo) & (s2_ticks <= s2_hi)
    ax2.set_xticks(s2_ticks[mask])
    ax2.set_xticklabels([f"{d:.1f}" for d in d_ticks[mask]])
    ax2.set_xlabel("d-spacing ($\\AA$)")

    save_figure(
        fig,
        save_dir / f"wilson_tau_epoch_{epoch}.png",
        dpi=150, pad_inches=0.05, transparent=False,
    )


def _plot_wilson_curve(stats: dict, epoch: int, save_dir: Path):
    """Plot the Wilson curve: expected intensity Sigma = K * exp(-2B*s^2)."""
    s_sq_bins = stats["s_sq"]

    # Dense curve in s^2 space
    s_sq_dense = np.linspace(s_sq_bins.min() * 0.9, s_sq_bins.max() * 1.1, 200)
    sigma_dense = stats["K_mean"] * np.exp(-2.0 * stats["B_mean"] * s_sq_dense)

    # Per-bin values
    sigma_bins = stats["K_mean"] * np.exp(-2.0 * stats["B_mean"] * s_sq_bins)

    sort_idx = np.argsort(s_sq_bins)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_sq_dense, sigma_dense, "-", color="blue", alpha=0.7, label="Wilson curve")
    ax.plot(
        s_sq_bins[sort_idx], sigma_bins[sort_idx],
        "o", color="red", markersize=5, label="Bin centers",
    )

    # If empirical tau available, plot 1/tau as empirical Sigma
    if stats["tau_empirical"] is not None:
        sigma_emp = 1.0 / stats["tau_empirical"]
        ax.plot(
            s_sq_bins[sort_idx], sigma_emp[sort_idx],
            "s", color="green", markersize=4,
            label="Empirical $\\Sigma$ (1/$\\tau$)",
        )

    ax.set_xlabel("$s^2 = 1/(4d^2)$ ($\\AA^{-2}$)")
    ax.set_ylabel("$\\Sigma$ (expected intensity)")
    ax.set_title(
        f"Wilson plot | Epoch {epoch}\n"
        f"$\\Sigma = K \\cdot e^{{-2Bs^2}}$, "
        f"K = {stats['K_mean']:.4f}, B = {stats['B_mean']:.2f}"
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add d-spacing as secondary x-axis with fixed tick positions
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    d_ticks = np.array([1.0, 1.2, 1.5, 2.0, 3.0, 5.0])
    s2_ticks = 1.0 / (4.0 * d_ticks**2)
    s2_lo, s2_hi = ax.get_xlim()
    mask = (s2_ticks >= s2_lo) & (s2_ticks <= s2_hi)
    ax2.set_xticks(s2_ticks[mask])
    ax2.set_xticklabels([f"{d:.1f}" for d in d_ticks[mask]])
    ax2.set_xlabel("d-spacing ($\\AA$)")

    save_figure(
        fig,
        save_dir / f"wilson_curve_epoch_{epoch}.png",
        dpi=150, pad_inches=0.05, transparent=False,
    )


def _plot_bg_rate(stats: dict, epoch: int, save_dir: Path):
    """Plot background rate per resolution bin."""
    s_sq = stats["s_sq"]
    bg = stats["bg_rate"]

    sort_idx = np.argsort(s_sq)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_sq[sort_idx], bg[sort_idx], "o-", color="purple", markersize=4)
    ax.set_xlabel("$s^2 = 1/(4d^2)$ ($\\AA^{-2}$)")
    ax.set_ylabel("Background rate (per bin)")
    ax.set_title(f"Background prior rate per resolution bin | Epoch {epoch}")
    ax.grid(True, alpha=0.3)

    ax2 = ax.secondary_xaxis("top", functions=(
        lambda s2: 1.0 / (2.0 * np.sqrt(np.maximum(s2, 1e-12))),
        lambda d: 1.0 / (4.0 * d**2),
    ))
    ax2.set_xlabel("d-spacing ($\\AA$)")

    save_figure(
        fig,
        save_dir / f"bg_rate_per_bin_epoch_{epoch}.png",
        dpi=150, pad_inches=0.05, transparent=False,
    )


def _plot_posterior_summary(stats: dict, epoch: int, save_dir: Path):
    """Plot posterior distributions of K and B."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # q(log K) -> K is LogNormal
    for ax, name, mu, s, mean, std in [
        (axes[0], "K", stats["mu_K"], stats["s_K"], stats["K_mean"], stats["K_std"]),
        (axes[1], "B", stats["mu_B"], stats["s_B"], stats["B_mean"], stats["B_std"]),
    ]:
        # Sample from LogNormal
        log_samples = np.random.normal(mu, s, size=10000)
        samples = np.exp(log_samples)

        # Clip for display
        q99 = np.percentile(samples, 99.5)
        samples_clip = samples[samples < q99]

        ax.hist(samples_clip, bins=80, density=True, alpha=0.7, color="steelblue")
        ax.axvline(mean, color="red", lw=2, label=f"Mean = {mean:.4f}")
        ax.axvline(mean - std, color="red", lw=1, ls="--", alpha=0.5)
        ax.axvline(mean + std, color="red", lw=1, ls="--", alpha=0.5,
                   label=f"Std = {std:.4f}")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"q({name}) posterior | Epoch {epoch}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_figure(
        fig,
        save_dir / f"wilson_KB_posterior_epoch_{epoch}.png",
        dpi=150, pad_inches=0.05, transparent=False,
    )


def _print_summary(stats: dict, epoch: int):
    """Print a text summary of Wilson parameters."""
    print(f"\n{'='*60}")
    print(f"Wilson Parameter Summary (Epoch {epoch})")
    print(f"{'='*60}")
    print(f"  K = {stats['K_mean']:.6f} +/- {stats['K_std']:.6f}")
    print(f"  B = {stats['B_mean']:.4f} +/- {stats['B_std']:.4f}")
    print(f"  log K: mu={stats['mu_K']:.4f}, sigma={stats['s_K']:.4f}")
    print(f"  log B: mu={stats['mu_B']:.4f}, sigma={stats['s_B']:.4f}")
    if stats["hp_K_loc"] is not None:
        print(f"\n  Hyperprior p(log K) ~ N({stats['hp_K_loc']:.2f}, {stats['hp_K_scale']:.2f})")
        print(f"  Hyperprior p(log B) ~ N({stats['hp_B_loc']:.2f}, {stats['hp_B_scale']:.2f})")
    print(f"\n  Resolution bins: {len(stats['d_per_bin'])}")
    print(f"  d range: {stats['d_per_bin'].min():.3f} - {stats['d_per_bin'].max():.3f} A")
    print(f"  s^2 range: {stats['s_sq'].min():.4f} - {stats['s_sq'].max():.4f}")
    if stats["tau_empirical"] is not None:
        print(f"\n  Empirical tau range: {stats['tau_empirical'].min():.6f} - {stats['tau_empirical'].max():.6f}")
    print(f"  Wilson tau range:    {stats['tau_wilson'].min():.6f} - {stats['tau_wilson'].max():.6f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to run directory containing run_metadata.yaml",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=None,
        help="Output directory (default: <wandb_log>/plots/wilson)",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch to analyze (default: latest checkpoint)",
    )
    args = parser.parse_args()

    # Resolve paths
    config, wandb_log = open_run(args.run_dir)

    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = wandb_log / "plots" / "wilson"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    state, epoch = _load_checkpoint(wandb_log, args.epoch)
    state_dict = state["state_dict"]

    # Extract Wilson params
    params = _extract_wilson_params(state_dict)
    stats = _compute_wilson_stats(params)

    # Print summary
    _print_summary(stats, epoch)

    # Generate plots
    _plot_tau_comparison(stats, epoch, save_dir)
    print(f"Saved: {save_dir}/wilson_tau_epoch_{epoch}.png")

    _plot_wilson_curve(stats, epoch, save_dir)
    print(f"Saved: {save_dir}/wilson_curve_epoch_{epoch}.png")

    _plot_bg_rate(stats, epoch, save_dir)
    print(f"Saved: {save_dir}/bg_rate_per_bin_epoch_{epoch}.png")

    _plot_posterior_summary(stats, epoch, save_dir)
    print(f"Saved: {save_dir}/wilson_KB_posterior_epoch_{epoch}.png")

    print("Done")


if __name__ == "__main__":
    main()
