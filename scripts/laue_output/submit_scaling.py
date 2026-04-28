"""Submit Careless scaling jobs for all epochs in a run directory.

Follows the same pattern as refltorch/scripts/dials_output/submit_jobs.py:
reads run_metadata.yaml, finds prediction dirs, submits SLURM jobs via
subprocess.

Usage:
    python scripts/submit_scaling.py --run-dir <run_dir>
    python scripts/submit_scaling.py --run-dir <run_dir> --configs 1 4 6
    python scripts/submit_scaling.py --run-dir <run_dir> --last-epoch-only
"""

import argparse
import subprocess
from pathlib import Path

import yaml


CARELESS_CONFIGS = {
    1: "baseline (anomalous, no pos-encoding)",
    2: "friedel split, double-wilson",
    3: "friedel split, double-wilson, pos-encoding",
    4: "anomalous, pos-encoding",
    5: "anomalous, image-layers=2, 6k iters",
    6: "anomalous, image-layers=2, pos-encoding",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit Careless scaling SLURM jobs for integrator predictions",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing run_metadata.yaml",
    )
    parser.add_argument(
        "--configs",
        type=int,
        nargs="+",
        default=list(CARELESS_CONFIGS.keys()),
        help="Which Careless configs to run (default: all 1-6)",
    )
    parser.add_argument(
        "--last-epoch-only",
        action="store_true",
        help="Only scale the last epoch checkpoint",
    )
    parser.add_argument(
        "--dmin",
        type=float,
        default=1.5,
        help="Resolution cutoff for Careless (default: 1.5)",
    )
    parser.add_argument(
        "--dependency",
        type=str,
        default=None,
        help="SLURM job ID(s) to depend on (afterany:ID[:ID...])",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without submitting",
    )
    return parser.parse_args()


def get_prediction_dir(run_dir: Path) -> Path:
    meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    log_dir = Path(meta["wandb"]["log_dir"])
    return log_dir.parent / "predictions"


def get_epoch_dirs(pred_dir: Path, last_only: bool = False) -> list[Path]:
    epoch_dirs = sorted(
        d for d in pred_dir.glob("epoch_*")
        if (d / "preds.mtz").exists()
    )
    if not epoch_dirs:
        raise FileNotFoundError(
            f"No epoch dirs with preds.mtz found under {pred_dir}"
        )
    if last_only:
        epoch_dirs = [epoch_dirs[-1]]
    return epoch_dirs


def submit_scaling_job(
    epoch_dir: Path,
    config: int,
    scale_script: Path,
    dmin: float,
    dependency: str | None = None,
    dry_run: bool = False,
) -> str | None:
    cmd = [
        "sbatch",
        "--parsable",
    ]
    if dependency:
        cmd.append(f"--dependency=afterany:{dependency}")
    cmd += [
        str(scale_script),
        str(epoch_dir),
        str(config),
        str(dmin),
    ]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return None
    job_id = subprocess.check_output(cmd, text=True).strip()
    return job_id


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    scale_script = Path(__file__).parent / "careless_scale.sh"
    if not scale_script.exists():
        raise FileNotFoundError(f"careless_scale.sh not found at {scale_script}")

    pred_dir = get_prediction_dir(run_dir)
    epoch_dirs = get_epoch_dirs(pred_dir, last_only=args.last_epoch_only)

    print(f"run dir:     {run_dir}")
    print(f"pred dir:    {pred_dir}")
    print(f"epochs:      {len(epoch_dirs)}")
    print(f"configs:     {args.configs}")
    print(f"dmin:        {args.dmin}")
    if args.last_epoch_only:
        print(f"  (last epoch only: {epoch_dirs[0].name})")
    print()

    job_ids = []
    for epoch_dir in epoch_dirs:
        for cfg in args.configs:
            label = CARELESS_CONFIGS.get(cfg, "unknown")
            print(f"  {epoch_dir.name} config{cfg} ({label})")
            job_id = submit_scaling_job(
                epoch_dir, cfg, scale_script, args.dmin,
                dependency=args.dependency, dry_run=args.dry_run,
            )
            if job_id:
                print(f"    → job {job_id}")
                job_ids.append(job_id)

    n = len(job_ids) if not args.dry_run else len(epoch_dirs) * len(args.configs)
    print(f"\nsubmitted {n} scaling jobs")

    if job_ids:
        print(f"monitor: squeue -u $USER --name=careless")
        # Print colon-separated IDs for downstream dependency chaining
        print(f"SCALING_JOB_IDS={':'.join(job_ids)}")


if __name__ == "__main__":
    main()
