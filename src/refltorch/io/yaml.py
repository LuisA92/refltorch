from importlib.resources import as_file
from pathlib import Path

import yaml


def load_config(resource: str | Path) -> dict:
    """resource is a Traversable from get_configs()."""

    if isinstance(resource, str):
        resource = Path(resource)

    with as_file(resource) as p:
        with open(Path(p), encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    return raw
