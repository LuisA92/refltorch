"""Parsers for PHENIX refinement output.

`parse_phenix_r_values` extracts the start and final R-work/R-free from a
`phenix.refine` log.
"""

import re
from pathlib import Path

_START = re.compile(r"Start R-work")
_FINAL = re.compile(r"Final R-work")
_NUMBER = re.compile(r"\d+\.\d+")


def parse_phenix_r_values(log_path) -> dict[str, float]:
    """Parse start/final R-work and R-free from a phenix.refine log.

    The number regex allows a multi-digit integer part so R values >= 1.0
    are parsed correctly.

    Args:
        log_path: Path to a `phenix.refine` log file.

    Returns:
        Mapping with `r_work_start`, `r_free_start`, `r_work_final`, and
        `r_free_final`.
    """
    with Path(log_path).open("r") as file:
        lines = file.readlines()

    start = [line.strip() for line in lines if _START.search(line)]
    final = [line.strip() for line in lines if _FINAL.search(line)]

    return {
        "r_work_start": float(_NUMBER.findall(start[0])[0]),
        "r_free_start": float(_NUMBER.findall(start[0])[1]),
        "r_work_final": float(_NUMBER.findall(final[0])[0]),
        "r_free_final": float(_NUMBER.findall(final[0])[1]),
    }
