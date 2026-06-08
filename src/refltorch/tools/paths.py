from pathlib import Path

from refltorch.io import load_config


def get_reference_data_path(run_config: dict) -> Path:
    """Resolve the reference-data directory for a run.

    Args:
        run_config: Parsed `run_metadata.yaml` with a `config` path.

    Returns:
        The directory holding the run's reference data.
    """
    cfg = load_config(run_config["config"])
    return Path(cfg["global_vars"]["data_dir"]).parent


def get_reference_data(ref_data_path: Path) -> tuple[Path, ...]:
    """Locate the reference peaks CSV and merged HTML for a run.

    Args:
        ref_data_path: Directory holding a `reference_data/` subdirectory.

    Returns:
        Tuple of (reference peaks CSV path, reference merged HTML path).
    """
    ref_peaks = list(ref_data_path.glob("reference_data/*peaks.csv"))[0]
    ref_merged_html = list(ref_data_path.glob("reference_data/*merged*.html"))[0]
    return ref_peaks, ref_merged_html


def get_reference_metadata(run_config: dict) -> dict:
    """Summarize a run's model/loss configuration.

    Uses `.get()` for the optional loss prior configs so runs that omit
    them yield None rather than raising.

    Args:
        run_config: Parsed `run_metadata.yaml` with a `config` path.

    Returns:
        Mapping of surrogate names, max epochs, integrator name, and the
        pbg/pi/pprf loss prior configs.
    """
    cfg = load_config(run_config["config"])
    loss_args = cfg["loss"]["args"]
    out = {
        "qbg_name": cfg["surrogates"]["qbg"]["name"],
        "qi_name": cfg["surrogates"]["qi"]["name"],
        "max_epochs": cfg["trainer"]["max_epochs"],
        "integrator_name": cfg["integrator"]["name"],
        "pbg": loss_args.get("pbg_cfg"),
        "pi": loss_args.get("pi_cfg"),
        "pprf": loss_args.get("pprf_cfg"),
    }
    return out
