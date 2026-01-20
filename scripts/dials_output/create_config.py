import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from refltorch.cli.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare dials_phenix_cfg.yaml file to run DIALS/PHENIX",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory; located where integrator.pred was called",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
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
    setup_logging(args.verbose)

    # Run directory
    run_dir = Path(args.run_dir)
    logger.info(f"Run directory: {run_dir.as_posix()}")

    # Reading in pred.yaml
    cfg_path = run_dir / "run_metadata.yaml"
    meta = yaml.safe_load(cfg_path.read_text())
    wandb_info = meta["wandb"]
    logger.info(f"Run metadata file: {cfg_path.as_posix()}")

    log_dir = Path(wandb_info["log_dir"]).parent
    path = log_dir / "predictions/"
    refl_files = {
        "refl_files": [x.as_posix() for x in list(path.glob("**/preds*.refl"))]
    }
    logger.info(f"W&B log directory: {log_dir.as_posix()}")

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
    logger.info(f"Wrote DIALS/PHENIX config to: {cfg_fname.as_posix()}")


if __name__ == "__main__":
    main()
