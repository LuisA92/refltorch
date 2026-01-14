import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare dials_phenix_cfg.yaml file to run DIALS/PHENIX",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory; located where integrator.pred was called",
    )

    return parser.parse_args()


@dataclass
class Config:
    refl_files: list
    phenix_eff: str
    phenix_env: str
    dials_env: str
    expt_file: str
    pdb: str
    paired_ref_eff: str
    paired_model_eff: str


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # Reading in pred.yaml
    meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    wandb_info = meta["wandb"]

    path = Path(wandb_info["log_dir"]).parent / "predictions/"
    refl_files = {
        "refl_files": [x.as_posix() for x in list(path.glob("**/preds*.refl"))]
    }

    # Configuration for a dials env
    config = Config(
        refl_files=refl_files["refl_files"],
        phenix_eff="/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/reference_data/phenix.eff",
        dials_env="/n/hekstra_lab/people/aldama/software/dials-v3-16-1/dials_env.sh",
        phenix_env="/n/hekstra_lab/garden_backup/phenix-1.21/phenix-1.21.1-5286/phenix_env.sh",
        expt_file="/n/hekstra_lab/people/aldama/integrator_data/hewl_9b7c/dials/integrated.expt",
        pdb="/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/reference_data/9b7c.pdb",
        paired_ref_eff="/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/reference_data/paired_refinement_ref.eff",
        paired_model_eff="/n/holylabs/LABS/hekstra_lab/Users/laldama/integrato_refac/integrator/reference_data/paired_refinement_model.eff",
    )

    config = asdict(config)
    cfg_fname = run_dir / "dials_phenix_cfg.yaml"
    (cfg_fname).write_text(yaml.safe_dump(config))


if __name__ == "__main__":
    main()
