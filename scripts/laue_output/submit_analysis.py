"""Submit anomalous peak height + CC_anom analysis on refined outputs.

Runs after submit_refinement.py completes. For each (epoch, config):
1. Computes anomalous peak heights from the refined MTZ
2. Runs careless.ccanom on the cross-validation MTZ

Usage:
    python scripts/submit_analysis.py --run-dir <run_dir>
    python scripts/submit_analysis.py --run-dir <run_dir> --configs 1 4 --last-epoch-only
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

import yaml


REFERENCE_PDB = Path(
    "/n/holylabs/hekstra_lab/Users/laldama/integrato_refac/integrator"
    "/reference_data/9b7c.pdb"
)
ANOMALOUS_ELEMENTS = "[S,I]"

SINGLE_MTZ_CONFIGS = {1, 4, 5, 6}
FRIEDEL_SPLIT_CONFIGS = {2, 3}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit anomalous analysis on refined outputs",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--configs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument("--last-epoch-only", action="store_true")
    parser.add_argument("--pdb", type=Path, default=REFERENCE_PDB)
    parser.add_argument("--elements", type=str, default=ANOMALOUS_ELEMENTS)
    parser.add_argument(
        "--peak-script",
        type=str,
        default=str(Path(__file__).parent / "anomalous_peak_heights.py"),
        help="path to anomalous_peak_heights.py",
    )
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
    epoch_dirs = sorted(d for d in pred_dir.glob("epoch_*") if d.is_dir())
    if last_only:
        epoch_dirs = [epoch_dirs[-1]]
    return epoch_dirs


def build_analysis_script(
    scaling_dir: Path,
    config: int,
    pdb: Path,
    elements: str,
    peak_script: str,
) -> str | None:
    prefix = f"config{config}"
    refine_dir = scaling_dir / f"{prefix}_refine"
    refined_mtz = refine_dir / "refine2" / "refined_2.mtz"

    if not refined_mtz.exists():
        return None

    peaks_csv = scaling_dir / f"{prefix}_peaks.csv"

    # Determine which xval MTZ to use for ccanom
    if config in FRIEDEL_SPLIT_CONFIGS:
        xval_plus = scaling_dir / f"{prefix}_xval_0.mtz"
        xval_minus = scaling_dir / f"{prefix}_xval_1.mtz"
        xval_merged = scaling_dir / f"{prefix}_xval_merged.mtz"
        if not xval_plus.exists() or not xval_minus.exists():
            xval_section = f'echo "xval MTZ not found for {prefix}, skipping ccanom"'
        else:
            xval_section = textwrap.dedent(f"""\
                # Unfriedelize xval for ccanom
                source /n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
                micromamba activate crls

                python3 -c "
import reciprocalspaceship as rs

plus = rs.read_mtz('{xval_plus}')
minus = rs.read_mtz('{xval_minus}')
half_repeats = []
merged = rs.concat([plus, minus.apply_symop('-x,-y,-z')])
for repeat in merged.repeat.unique():
    for half in range(2):
        sel = merged.loc[
            (merged.repeat == repeat) & (merged.half == half),
            ['F', 'SigF', 'I', 'SigI', 'N', 'high', 'loc', 'low', 'scale'],
        ]
        sel = sel.unstack_anomalous()
        sel['half'] = half
        sel['half'] = sel['half'].astype('MTZInt')
        sel['repeat'] = repeat
        sel['repeat'] = sel['repeat'].astype('MTZInt')
        half_repeats.append(sel)
out = rs.concat(half_repeats)
out.write_mtz('{xval_merged}')
print(f'wrote {{len(out)}} refls to {xval_merged.name}')
"
                careless.ccanom "{xval_merged}" -o "{scaling_dir}/{prefix}_ccanom.csv" -i "{scaling_dir}/{prefix}_ccanom.png"
            """)
    else:
        xval_mtz = scaling_dir / f"{prefix}_xval_0.mtz"
        if not xval_mtz.exists():
            xval_section = f'echo "xval MTZ not found for {prefix}, skipping ccanom"'
        else:
            xval_section = textwrap.dedent(f"""\
                source /n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
                micromamba activate crls
                careless.ccanom "{xval_mtz}" -o "{scaling_dir}/{prefix}_ccanom.csv" -i "{scaling_dir}/{prefix}_ccanom.png"
            """)

    script = f"""#!/bin/bash
source /n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh

# Peak heights (integrator env)
micromamba activate integrator
python "{peak_script}" \\
    "{refined_mtz}" \\
    "{pdb}" \\
    {elements} \\
    "{peaks_csv}"

# CC_anom (crls env)
{xval_section}
"""
    return script


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    pred_dir = get_prediction_dir(run_dir)
    epoch_dirs = get_epoch_dirs(pred_dir, last_only=args.last_epoch_only)

    print(f"run dir:  {run_dir}")
    print(f"epochs:   {len(epoch_dirs)}")
    print(f"configs:  {args.configs}")
    print()

    job_ids = []
    for epoch_dir in epoch_dirs:
        scaling_dir = epoch_dir / "scaling"
        if not scaling_dir.exists():
            continue

        for cfg in args.configs:
            script_content = build_analysis_script(
                scaling_dir, cfg, args.pdb, args.elements, args.peak_script,
            )
            if script_content is None:
                print(f"  {epoch_dir.name} config{cfg}: refined MTZ not found, skipping")
                continue

            script_path = scaling_dir / f"config{cfg}_analysis.sh"
            script_path.write_text(script_content)
            script_path.chmod(0o755)

            cmd = [
                "sbatch",
                "--parsable",
                "--job-name=anom_analysis",
                f"--output={scaling_dir}/config{cfg}_analysis_%j.out",
                f"--error={scaling_dir}/config{cfg}_analysis_%j.err",
                "--time=00:30:00",
                "--mem=16G",
                "--partition=shared,seas_compute",
                "--cpus-per-task=1",
            ]
            if args.dependency:
                cmd.append(f"--dependency=afterany:{args.dependency}")
            cmd.append(str(script_path))

            print(f"  {epoch_dir.name} config{cfg}")
            if args.dry_run:
                print(f"    [dry-run] {' '.join(cmd)}")
            else:
                job_id = subprocess.check_output(cmd, text=True).strip()
                print(f"    → job {job_id}")
                job_ids.append(job_id)

    print(f"\nsubmitted {len(job_ids)} analysis jobs")


if __name__ == "__main__":
    main()
