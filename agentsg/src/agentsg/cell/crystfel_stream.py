"""Parse CrystFEL stream files (serial crystallography indexing results).

A CrystFEL ``.stream`` file records, for every indexed diffraction pattern, the
per-crystal indexed unit cell, the reciprocal-lattice orientation vectors
(astar/bstar/cstar), and the integrated reflections. This module extracts the
per-crystal cells and orientations -- the raw indexed-cell *distribution* that
merging averages away -- with zero third-party dependencies.

Cell parameters in a stream are in nm; they are converted to Angstrom here.
"""
import re

_CELL = re.compile(
    r"Cell parameters\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+nm,"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+deg")
_STAR = re.compile(r"([abc])star\s*=\s*([+\-\d.]+)\s+([+\-\d.]+)\s+([+\-\d.]+)")


def parse_stream(path, with_orientation=False):
    """Yield one dict per indexed crystal.

    Each dict has ``cell`` = (a, b, c, alpha, beta, gamma) in Angstrom/degrees.
    If ``with_orientation`` is True, also ``astar``/``bstar``/``cstar`` as
    3-lists in nm^-1 (the reciprocal-lattice basis vectors in the lab frame).
    """
    cur = {}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = _CELL.search(line)
            if m:
                a, b, c, al, be, ga = (float(x) for x in m.groups())
                cur = {"cell": (a * 10, b * 10, c * 10, al, be, ga)}
                if not with_orientation:
                    yield cur
                    cur = {}
                continue
            if with_orientation:
                sm = _STAR.search(line)
                if sm and "cell" in cur:
                    cur[sm.group(1) + "star"] = [float(sm.group(i)) for i in (2, 3, 4)]
                    if {"astar", "bstar", "cstar"} <= cur.keys():
                        yield cur
                        cur = {}


def read_cells(path):
    """Return a list of (a, b, c, alpha, beta, gamma) tuples (Angstrom) for every
    indexed crystal in the stream -- the per-pattern indexed-cell distribution."""
    return [d["cell"] for d in parse_stream(path)]


def stream_summary(path):
    """One-pass summary: chunk (frame) count, indexed-crystal count, indexing
    rate, and the mean/std of each cell parameter."""
    n_chunks = 0
    cells = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if "Begin chunk" in line:
                n_chunks += 1
                continue
            m = _CELL.search(line)
            if m:
                a, b, c, al, be, ga = (float(x) for x in m.groups())
                cells.append((a * 10, b * 10, c * 10, al, be, ga))
    n = len(cells)
    means = [sum(c[k] for c in cells) / n for k in range(6)] if n else [0] * 6
    stds = [(sum((c[k] - means[k]) ** 2 for c in cells) / n) ** 0.5 for k in range(6)] if n else [0] * 6
    return {
        "n_frames": n_chunks,
        "n_indexed": n,
        "indexing_rate": (n / n_chunks) if n_chunks else 0.0,
        "cell_mean": tuple(means),
        "cell_std": tuple(stds),
    }
