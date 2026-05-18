"""Prepare dials_phenix_cfg.yaml from the training config + environment.

Reads dataset-specific paths (expt_file, pdb, phenix_eff, etc.) from
the training config's ``output`` section (config_copy.yaml in run_dir).
Tool paths (DIALS, PHENIX) come from environment variables, with CLI
overrides available for everything.

Usage:
    # Minimal — everything from training config + env vars
    export DIALS_ENV=/n/hekstra_lab/people/aldama/micromamba/envs/ld/bin/activate
    export PHENIX_ENV=/n/.../phenix_env.sh
    python create_config.py --run-dir /path/to/run_dir

    # Override a specific path
    python create_config.py --run-dir /path/to/run_dir --expt-file /path/to/integrated.expt

Training config ``output`` section (add to your YAML):
    output:
      refl_file: /path/to/reflections_.refl
      expt_file: /path/to/integrated.expt
      pdb: /path/to/reference.pdb
      phenix_eff: /path/to/phenix.eff
      paired_ref_eff: /path/to/paired_refinement_ref.eff     # optional
      paired_model_eff: /path/to/paired_refinement_model.eff  # optional
"""

import argparse
import logging
import os
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
        type=Path,
        required=True,
        help="Run directory (contains run_metadata.yaml and config_copy.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
    )

    # Dataset paths — override what's in config_copy.yaml
    parser.add_argument("--expt-file", type=str, default=None)
    parser.add_argument("--pdb", type=str, default=None)
    parser.add_argument("--phenix-eff", type=str, default=None)
    parser.add_argument("--paired-ref-eff", type=str, default=None)
    parser.add_argument("--paired-model-eff", type=str, default=None)

    # Tool paths — override env vars DIALS_ENV / PHENIX_ENV
    parser.add_argument("--dials-env", type=str, default=None)
    parser.add_argument("--phenix-env", type=str, default=None)
    return parser.parse_args()


def _resolve(cli_val, cfg_val, env_var, label):
    """Pick the first non-None source: CLI > config > env var."""
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return str(cfg_val)
    env = os.environ.get(env_var) if env_var else None
    if env is not None:
        return env
    raise ValueError(
        f"Missing '{label}'. Set it in the training config output section, "
        f"pass --{label.replace('_', '-')}, "
        + (f"or set ${env_var}" if env_var else "")
    )


def _resolve_optional(cli_val, cfg_val):
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return str(cfg_val)
    return ""


@dataclass
class Config:
    refl_files: list
    expt_file: str
    dials_env: str
    phenix_env: str
    phenix_eff: str
    pdb: str
    paired_ref_eff: str
    paired_model_eff: str


def main():
    args = parse_args()
    setup_logging(args.verbose)

    run_dir = args.run_dir.resolve()
    logger.info(f"Run directory: {run_dir}")

    # Load run metadata for wandb log dir
    meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    wandb_info = meta["wandb"]
    log_dir = Path(wandb_info["log_dir"]).parent
    logger.info(f"W&B log directory: {log_dir}")

    # Discover prediction .refl files
    pred_dir = log_dir / "predictions"
    refl_files = sorted(x.as_posix() for x in pred_dir.glob("**/preds*.refl"))
    if not refl_files:
        raise FileNotFoundError(f"No preds*.refl found in {pred_dir}")
    logger.info(f"Found {len(refl_files)} .refl files")

    # Load training config for dataset-specific paths
    cfg_path = run_dir / "config_copy.yaml"
    if cfg_path.exists():
        train_cfg = yaml.safe_load(cfg_path.read_text())
        output_cfg = train_cfg.get("output", {})
    else:
        logger.warning(f"No config_copy.yaml in {run_dir}, using CLI/env only")
        output_cfg = {}

    config = Config(
        refl_files=refl_files,
        expt_file=_resolve(
            args.expt_file,
            output_cfg.get("expt_file"),
            None,
            "expt_file",
        ),
        dials_env=_resolve(
            args.dials_env,
            None,
            "DIALS_ENV",
            "dials_env",
        ),
        phenix_env=_resolve(
            args.phenix_env,
            None,
            "PHENIX_ENV",
            "phenix_env",
        ),
        phenix_eff=_resolve(
            args.phenix_eff,
            output_cfg.get("phenix_eff"),
            None,
            "phenix_eff",
        ),
        pdb=_resolve(
            args.pdb,
            output_cfg.get("pdb"),
            None,
            "pdb",
        ),
        paired_ref_eff=_resolve_optional(
            args.paired_ref_eff,
            output_cfg.get("paired_ref_eff"),
        ),
        paired_model_eff=_resolve_optional(
            args.paired_model_eff,
            output_cfg.get("paired_model_eff"),
        ),
    )

    cfg_fname = run_dir / "dials_phenix_cfg.yaml"
    cfg_fname.write_text(yaml.safe_dump(asdict(config)))
    logger.info(f"Wrote config to: {cfg_fname}")


if __name__ == "__main__":
    main()
