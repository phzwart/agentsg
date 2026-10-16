"""CSV entry point for :mod:`agentsg.cell.serial_index`."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .serial_index import run, summary


def _load(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        cells = []
        groups = []
        for row in reader:
            cells.append([
                float(row["a"]), float(row["b"]), float(row["c"]),
                float(row["alpha"]), float(row["beta"]), float(row["gamma"]),
            ])
            groups.append(row.get("sg_hm") or None)
    return cells, groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--sigma-deg", type=float, required=True)
    parser.add_argument("--sigma-len", type=float, default=0.0)
    parser.add_argument("--symmetry", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    cells, groups = _load(args.input)
    state = run(
        cells, groups,
        sigma_deg=args.sigma_deg,
        sigma_len=args.sigma_len,
        symmetry=args.symmetry,
    )
    text = summary(state)
    print(text)
    if args.out is not None:
        args.out.write_text(state.to_json())
    return state


if __name__ == "__main__":
    main()
