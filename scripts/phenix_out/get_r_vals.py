"""
Script to write phenix.log r-values as a .csv.
Used for polychromatic analysis where the structure is
/config*/refine2/*refine*.log
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Write phenix.log r-values as a .csv.")

    parser.add_argument(
        "--log-dir", type=str, help="Path to directory containing log files"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Name of output file",
    )

    return parser.parse_args()


def main():
    import csv
    import re
    from pathlib import Path

    args = parse_args()
    log_dir = args.log_dir

    log_dir = Path(log_dir)

    # NOTE: Assuming the following structure
    # /config*/refine2/*refine*.log
    rows = []
    for f in log_dir.glob("**/refine2/*refine*.log"):
        data = {"config": f.parts[-3]}
        for line in f.read_text().splitlines():
            m = re.match(
                r"(Start|Final) R-work\s*=\s*(\S+),\s*R-free\s*=\s*(\S+)", line
            )
            if m:
                data[f"{m.group(1)}_R-work"] = float(m.group(2).strip(","))
                data[f"{m.group(1)}_R-free"] = float(m.group(3).strip(","))
        if "Start_R-work" in data and "Final_R-work" in data:
            data["delta_Rwork"] = data["Start_R-work"] - data["Final_R-work"]
            data["delta_Rfree"] = data.get("Start_R-free", 0) - data.get(
                "Final_R-free", 0
            )
            rows.append(data)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
