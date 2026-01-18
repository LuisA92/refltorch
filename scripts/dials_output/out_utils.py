from pathlib import Path

from refltorch.io import load_config


def _get_reference_data_path(run_config: dict) -> Path:
    cfg = load_config(run_config["config"])
    return Path(cfg["global_vars"]["data_dir"]).parent


def _get_reference_metadata(run_config: dict) -> dict:
    cfg = load_config(run_config["config"])
    out = {
        "qbg_name": cfg["surrogates"]["qbg"]["name"],
        "qi_name": cfg["surrogates"]["qi"]["name"],
        "max_epochs": cfg["trainer"]["max_epochs"],
        "integrator_name": cfg["integrator"]["name"],
        "pbg": cfg["loss"]["args"]["pbg_cfg"],
        "pi": cfg["loss"]["args"]["pi_cfg"],
        "pprf": cfg["loss"]["args"]["pprf_cfg"],
    }
    return out
