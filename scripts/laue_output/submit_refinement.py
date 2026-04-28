"""Submit phenix.refine jobs on Careless scaling outputs.

Handles:
- Configs 1,4,5,6: refines directly on configN_0.mtz
- Configs 2,3: unfriedelizes configN_0.mtz + configN_1.mtz first, then refines

Usage:
    python scripts/submit_refinement.py --run-dir <run_dir>
    python scripts/submit_refinement.py --run-dir <run_dir> --configs 1 4 --last-epoch-only
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

import yaml


REFERENCE_PDB = Path(
    "/n/holylabs/hekstra_lab/Users/laldama/integrato_refac/integrator"
    "/reference_data/shaken_9b7c.pdb"
)
REFINEMENT_EFF_1 = Path(
    "/n/holylabs/hekstra_lab/Users/laldama/laue_analysis"
    "/custom_refinement_param_1_.eff"
)
REFINEMENT_EFF_2 = Path(
    "/n/holylabs/hekstra_lab/Users/laldama/laue_analysis"
    "/custom_refinement_param_2_.eff"
)

SINGLE_MTZ_CONFIGS = {1, 4, 5, 6}
FRIEDEL_SPLIT_CONFIGS = {2, 3}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit phenix refinement on Careless scaling outputs",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--configs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument("--last-epoch-only", action="store_true")
    parser.add_argument("--pdb", type=Path, default=REFERENCE_PDB)
    parser.add_argument("--eff1", type=Path, default=REFINEMENT_EFF_1)
    parser.add_argument("--eff2", type=Path, default=REFINEMENT_EFF_2)
    parser.add_argument(
        "--dependency",
        type=str,
        default=None,
        help="SLURM job ID(s) to depend on (afterany:ID[:ID...])",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_prediction_dir(run_dir: Path) -> Path:
    meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    log_dir = Path(meta["wandb"]["log_dir"])
    return log_dir.parent / "predictions"


def get_epoch_dirs(pred_dir: Path, last_only: bool = False) -> list[Path]:
    epoch_dirs = sorted(
        d for d in pred_dir.glob("epoch_*") if d.is_dir()
    )
    if last_only:
        epoch_dirs = [epoch_dirs[-1]]
    return epoch_dirs


def unfriedelize(plus_mtz: Path, minus_mtz: Path, out_mtz: Path):
    """Combine Friedel-split Careless outputs into a single anomalous MTZ."""
    import reciprocalspaceship as rs

    plus = rs.read_mtz(str(plus_mtz))
    minus = rs.read_mtz(str(minus_mtz))
    anom_keys = ["F(+)", "SigF(+)", "F(-)", "SigF(-)", "N(+)", "N(-)"]
    out = rs.concat([
        plus,
        minus.apply_symop("-x,-y,-z"),
    ]).unstack_anomalous()[anom_keys]
    out.write_mtz(str(out_mtz))
    print(f"  unfriedelized → {out_mtz.name} ({len(out)} refls)")


def get_mtz_for_config(scaling_dir: Path, config: int) -> Path | None:
    """Find or create the MTZ to refine for a given config."""
    if config in SINGLE_MTZ_CONFIGS:
        mtz = scaling_dir / f"config{config}_0.mtz"
        if mtz.exists():
            return mtz
        return None

    if config in FRIEDEL_SPLIT_CONFIGS:
        plus = scaling_dir / f"config{config}_0.mtz"
        minus = scaling_dir / f"config{config}_1.mtz"
        merged = scaling_dir / f"config{config}_merged_anom.mtz"
        if not plus.exists() or not minus.exists():
            return None
        if not merged.exists():
            unfriedelize(plus, minus, merged)
        return merged

    return None


def submit_refinement(
    mtz: Path,
    refine_dir: Path,
    pdb: Path,
    eff1: Path,
    eff2: Path,
    dependency: str | None = None,
    dry_run: bool = False,
) -> str | None:
    refine_dir.mkdir(parents=True, exist_ok=True)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        source /n/hekstra_lab_tier0/Lab/garden/phenix_1_20/phenix-1.20.1-4487/phenix_env.sh

        refine1="{refine_dir}/refine1"
        refine2="{refine_dir}/refine2"
        mkdir -p "$refine1" "$refine2"

        cd "$refine1"
        phenix.refine "{eff1}" \\
            refinement.input.xray_data.file_name="{mtz}" \\
            refinement.input.pdb.file_name="{pdb}" \\
            refinement.output.prefix=refined \\
            --overwrite

        cd "$refine2"
        phenix.refine "{eff2}" \\
            refinement.input.xray_data.file_name="${{refine1}}/refined_data.mtz" \\
            refinement.input.pdb.file_name="${{refine1}}/refined_1.pdb" \\
            refinement.input.xray_data.r_free_flags.file_name="${{refine1}}/refined_data.mtz" \\
            refinement.output.prefix=refined \\
            --overwrite
    """)

    script_path = refine_dir / "refine.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    cmd = [
        "sbatch",
        "--parsable",
        "--job-name=phenix_refine",
        f"--output={refine_dir}/refine_%j.out",
        f"--error={refine_dir}/refine_%j.err",
        "--time=01:00:00",
        "--mem=32G",
        "--partition=shared,seas_compute",
        "--cpus-per-task=4",
    ]
    if dependency:
        cmd.append(f"--dependency=afterany:{dependency}")
    cmd.append(str(script_path))

    if dry_run:
        print(f"    [dry-run] {' '.join(cmd)}")
        return None

    job_id = subprocess.check_output(cmd, text=True).strip()
    return job_id


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    pred_dir = get_prediction_dir(run_dir)
    epoch_dirs = get_epoch_dirs(pred_dir, last_only=args.last_epoch_only)

    print(f"run dir:  {run_dir}")
    print(f"epochs:   {len(epoch_dirs)}")
    print(f"configs:  {args.configs}")
    print(f"pdb:      {args.pdb}")
    print()

    job_ids = []
    for epoch_dir in epoch_dirs:
        scaling_dir = epoch_dir / "scaling"
        if not scaling_dir.exists():
            print(f"  {epoch_dir.name}: no scaling dir, skipping")
            continue

        for cfg in args.configs:
            mtz = get_mtz_for_config(scaling_dir, cfg)
            if mtz is None:
                print(f"  {epoch_dir.name} config{cfg}: MTZ not found, skipping")
                continue

            refine_dir = scaling_dir / f"config{cfg}_refine"
            print(f"  {epoch_dir.name} config{cfg}: {mtz.name}")

            job_id = submit_refinement(
                mtz, refine_dir, args.pdb, args.eff1, args.eff2,
                dependency=args.dependency, dry_run=args.dry_run,
            )
            if job_id:
                print(f"    → job {job_id}")
                job_ids.append(job_id)

    print(f"\nsubmitted {len(job_ids)} refinement jobs")
    if job_ids:
        print(f"REFINEMENT_JOB_IDS={':'.join(job_ids)}")


if __name__ == "__main__":
    main()
