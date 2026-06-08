"""Prepare dials_phenix_cfg.yaml from the training config + environment.
Reads dataset-specific paths (expt_file, pdb, phenix_eff, etc.) from
the training config's ``output`` section (config_copy.yaml in run_dir).
Tool paths (DIALS, PHENIX) come from environment variables, with CLI
overrides available for everything.

Two intensity/amplitude modes (``--columns``):
    intensity (default)  I(+)/SIGI(+)/...  with French-Wilson (phenix default).
                         Used by the .refl -> dials.scale -> dials.merge path.
    amplitude            F(+)/SIGF(+)/...  with French-Wilson OFF. Used when the
                         model already wrote a merged MTZ with posterior
                         amplitudes (integrator.pred --write-merged-mtz). In this
                         mode a phenix.eff is *rendered* from the template with
                         the amplitude labels and french_wilson_scale=False, so
                         phenix consumes the model's F directly (no second I->F).

Usage:
    # Intensity (.refl) path -- everything from training config + env vars
    export DIALS_ENV=/n/hekstra_lab/people/aldama/micromamba/envs/ld/bin/activate
    export PHENIX_ENV=/n/.../phenix_env.sh
    python create_config.py --run-dir /path/to/run_dir

    # Amplitude (merged-MTZ) path -- F(+)/F(-), no French-Wilson
    python create_config.py --run-dir /path/to/run_dir --columns amplitude

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
import re
from dataclasses import asdict
from pathlib import Path

import yaml
from out_utils import DialsPhenixConfig as Config

from refltorch.cli.utils import setup_logging
from refltorch.io import open_run

logger = logging.getLogger(__name__)

# (columns -> (miller labels, array_type "*" token)). French-Wilson defaults to
# OFF for amplitude (the F are already posterior amplitudes) and ON for
# intensity (phenix's standard I->F), unless --french-wilson overrides.
_COLUMN_SPECS = {
    "intensity": ("I(+),SIGI(+),I(-),SIGI(-)", "intensity"),
    "amplitude": ("F(+),SIGF(+),F(-),SIGF(-)", "amplitude"),
}


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

    # Intensity vs amplitude (merged-MTZ) mode.
    parser.add_argument(
        "--columns",
        choices=["intensity", "amplitude"],
        default="intensity",
        help="Which merged columns phenix should read. 'amplitude' uses "
        "F(+)/F(-) from the model's merged MTZ and turns French-Wilson off.",
    )
    parser.add_argument(
        "--french-wilson",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override french_wilson_scale. 'auto' = off for amplitude, on for "
        "intensity.",
    )
    parser.add_argument(
        "--merged-mtz",
        type=str,
        default=None,
        help="Explicit merged MTZ path (amplitude mode). Default: discover "
        "predictions/**/merged.mtz under the run's log dir.",
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


# ---------------------------------------------------------------------------
# phenix.eff rendering (ported from integrator's diagnose_merging.py)
# ---------------------------------------------------------------------------


def _set_array_type_star(line: str, star_token: str) -> str:
    """Move the `*` in an array_type listing to the requested token."""
    tokens = re.findall(r"\S+", line)
    out_tokens = []
    for t in tokens:
        bare = t.lstrip("*")
        out_tokens.append(f"*{bare}" if bare == star_token else bare)
    leading = line[: len(line) - len(line.lstrip())]
    return leading + " ".join(out_tokens) + "\n"


def render_eff(
    template: str,
    labels: str,
    star_token: str,
    fw_scale: bool,
) -> str:
    """Render a phenix.eff from the template for the chosen columns.

    Modifies only the *first* miller_array block (the data file): sets its
    `name` and `user_selected_labels` to `labels`, moves the array_type `*` to
    `star_token` (intensity|amplitude), and flips `french_wilson_scale`
    everywhere. The Rfree block is left intact. The MTZ itself is passed to
    phenix on the command line, so no $MTZFILE substitution is needed here.
    """
    miller_array_count = 0
    in_data_block = False
    array_type_pending = False
    out_lines = []

    for line in template.splitlines(keepends=True):
        stripped = line.strip()

        if stripped.startswith("miller_array"):
            miller_array_count += 1
            in_data_block = miller_array_count == 1
            out_lines.append(line)
            continue

        if in_data_block and stripped.startswith("name ="):
            out_lines.append(
                re.sub(r'name = "[^"]*"', f'name = "{labels}"', line)
            )
            array_type_pending = True
            continue

        if in_data_block and array_type_pending:
            out_lines.append(_set_array_type_star(line, star_token))
            if not line.rstrip().endswith("\\"):
                array_type_pending = False
            continue

        if in_data_block and stripped.startswith("user_selected_labels"):
            out_lines.append(
                re.sub(
                    r'user_selected_labels = "[^"]*"',
                    f'user_selected_labels = "{labels}"',
                    line,
                )
            )
            in_data_block = False
            continue

        if "french_wilson_scale" in line:
            out_lines.append(
                re.sub(
                    r"french_wilson_scale\s*=\s*\w+",
                    f"french_wilson_scale = {fw_scale}",
                    line,
                )
            )
            continue

        out_lines.append(line)

    return "".join(out_lines)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    run_dir = args.run_dir.resolve()
    logger.info(f"Run directory: {run_dir}")

    # Load run metadata for wandb log dir
    _, log_dir = open_run(run_dir)
    logger.info(f"W&B log directory: {log_dir}")

    pred_dir = log_dir / "predictions"

    # Discover per-observation .refl files (intensity path) and/or merged MTZs
    # (amplitude path). The merged-only workflow (--write-merged-mtz) produces
    # no preds*.refl, so missing refl is a warning, not an error.
    refl_files = sorted(x.as_posix() for x in pred_dir.glob("**/preds*.refl"))
    if args.merged_mtz is not None:
        merged_mtzs = [str(Path(args.merged_mtz).resolve())]
    else:
        merged_mtzs = sorted(
            x.as_posix() for x in pred_dir.glob("**/merged.mtz")
        )
    logger.info(
        f"Found {len(refl_files)} .refl files, {len(merged_mtzs)} merged MTZs"
    )
    if not refl_files and not merged_mtzs:
        raise FileNotFoundError(
            f"No preds*.refl and no merged.mtz found in {pred_dir}"
        )

    # Resolve French-Wilson: 'auto' -> off for amplitude, on for intensity.
    if args.french_wilson == "auto":
        fw_scale = args.columns == "intensity"
    else:
        fw_scale = args.french_wilson == "true"
    labels, star_token = _COLUMN_SPECS[args.columns]

    # Load training config for dataset-specific paths
    cfg_path = run_dir / "config_copy.yaml"
    if cfg_path.exists():
        train_cfg = yaml.safe_load(cfg_path.read_text())
        output_cfg = train_cfg.get("output", {})
    else:
        logger.warning(f"No config_copy.yaml in {run_dir}, using CLI/env only")
        output_cfg = {}

    phenix_eff_template = _resolve(
        args.phenix_eff,
        output_cfg.get("phenix_eff"),
        None,
        "phenix_eff",
    )

    # Amplitude mode (or any non-default FW): render a dedicated eff so phenix
    # reads the model's F(+)/F(-) directly with the right french_wilson_scale.
    phenix_eff = phenix_eff_template
    if args.columns == "amplitude" or not fw_scale:
        if not merged_mtzs:
            logger.warning(
                "columns=%s but no merged.mtz found; phenix has no amplitude "
                "data to read.",
                args.columns,
            )
        template_text = Path(phenix_eff_template).read_text()
        rendered = render_eff(template_text, labels, star_token, fw_scale)
        rendered_path = run_dir / f"phenix_{args.columns}.eff"
        rendered_path.write_text(rendered)
        phenix_eff = str(rendered_path)
        logger.info(
            "Rendered %s (labels=%s, array_type=%s, french_wilson_scale=%s)",
            rendered_path,
            labels,
            star_token,
            fw_scale,
        )

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
        phenix_eff=phenix_eff,
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
        columns=args.columns,
        merged_mtz=merged_mtzs,
        user_selected_labels=labels,
        french_wilson_scale=fw_scale,
    )

    cfg_fname = run_dir / "dials_phenix_cfg.yaml"
    cfg_fname.write_text(yaml.safe_dump(asdict(config)))
    logger.info(f"Wrote config to: {cfg_fname}")


if __name__ == "__main__":
    main()
