#!/usr/bin/env python3
"""Progressive metric perturbation across crystal systems.

For each ideal holohedry cell, apply seeded random edge/angle noise at several
levels, then report:

  1. Le Page–gated assignment (``lattice_symmetry``, max_delta=3°)
  2. Per-holohedry spectrum: max Le Page δ of that system's two-folds, and
     Kurlin root distance to the Reynolds-symmetrised metric

Run::

    python examples/perturb_metric_symmetry.py
    python examples/perturb_metric_symmetry.py --write examples/perturb_metric_symmetry.md
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from agentsg.cell.reduction import niggli_reduce
from agentsg.cell.g6 import kurlin_distance_to_symmetry
from agentsg.lattice_symmetry import lattice_symmetry, le_page_delta

# Ideal representatives (same set as the lattice-symmetry tests).
IDEAL = {
    "cubic": (5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
    "tetragonal": (5.0, 5.0, 8.0, 90.0, 90.0, 90.0),
    "hexagonal": (5.0, 5.0, 8.0, 90.0, 90.0, 120.0),
    "trigonal": (5.0, 5.0, 5.0, 70.0, 70.0, 70.0),
    "orthorhombic": (5.0, 6.0, 7.0, 90.0, 90.0, 90.0),
    "monoclinic": (5.0, 6.0, 7.0, 90.0, 95.0, 90.0),
    "triclinic": (5.0, 6.0, 7.0, 80.0, 85.0, 95.0),
}

SYSTEMS = list(IDEAL.keys())
# Noise ladder: edge fraction + angle degrees (applied as N(0, σ) draws).
NOISE_LEVELS = [
    (0.0, 0.0),
    (0.002, 0.15),
    (0.005, 0.40),
    (0.010, 0.80),
    (0.020, 1.50),
    (0.040, 3.00),
]
MAX_DELTA = 3.0


def _reduce(cell):
    red, _ = niggli_reduce(*cell)
    return tuple(float(x) for x in red)


def _build_references():
    """Holohedry ops + two-fold axes for each ideal system.

    Candidate operators are built in the **conventional (unreduced)** setting of
    each ideal cell. This is deliberate: Niggli reduction can flip a cell into an
    equivalent but differently-oriented setting (e.g. hexagonal γ 120°↔60°), and
    scoring an unreduced-setting operator set against a reduced cell in the other
    setting injects a spurious symmetry-deficiency baseline. Perturbations and
    spectrum scoring therefore both stay in the conventional setting, where the
    fixed operator set and the cell always share one basis. The Le Page
    *assignment* still reduces first (the Lebedev argument needs a reduced cell).
    """
    refs = {}
    for name, cell in IDEAL.items():
        conv = tuple(float(x) for x in cell)
        ls = lattice_symmetry(conv, max_delta=0.5)
        folds = [
            (s.matrix, s.direct_axis, s.reciprocal_axis)
            for s in ls.two_fold_scores
        ]
        refs[name] = {
            "cell": conv,          # conventional setting (used for scoring)
            "reduced": _reduce(conv),
            "ops": ls.operations,
            "two_folds": folds,
            "order": ls.order,
        }
    return refs


def perturb(cell, edge_frac, angle_deg, rng: random.Random):
    """Perturb a cell in its conventional setting (no reduction — see notes)."""
    a, b, c, al, be, ga = cell
    edges = [
        max(a * (1.0 + rng.gauss(0.0, edge_frac)), 1e-3),
        max(b * (1.0 + rng.gauss(0.0, edge_frac)), 1e-3),
        max(c * (1.0 + rng.gauss(0.0, edge_frac)), 1e-3),
    ]
    angs = [
        al + rng.gauss(0.0, angle_deg),
        be + rng.gauss(0.0, angle_deg),
        ga + rng.gauss(0.0, angle_deg),
    ]
    # Keep angles in an open crystallographic range.
    angs = [min(179.0, max(1.0, x)) for x in angs]
    return (*edges, *angs)


def max_le_page(cell, two_folds) -> float:
    if not two_folds:
        return 0.0
    return max(le_page_delta(cell, u, h) for _, u, h in two_folds)


def md_table(headers, rows) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt(x, nd=4):
    if abs(x) < 5e-7:
        return "0"
    return f"{x:.{nd}g}"


def run(seed: int = 0) -> str:
    refs = _build_references()
    out: list[str] = []
    out.append("# Progressive metric perturbation vs Le Page / Kurlin")
    out.append("")
    out.append(
        "Ideal cells from each crystal system are randomly perturbed "
        f"(seed={seed}) at increasing edge/angle noise. Assignment uses "
        f"`lattice_symmetry(..., max_delta={MAX_DELTA}°)` on the Niggli-reduced "
        "cell (Le Page gate). Spectrum tables report, for every candidate "
        "holohedry, the **max Le Page δ** over that system's two-folds and the "
        "**Kurlin** root-invariant distance to its Reynolds-symmetrised metric, "
        "both scored in the **conventional (unreduced) setting** so the fixed "
        "candidate operators and the cell always share one basis (avoiding the "
        "Niggli reduction-flip that otherwise injects a spurious baseline)."
    )
    out.append("")
    out.append(
        "> Note: the `triclinic` column is 0 at every level by construction — "
        "triclinic symmetrisation is just inversion, which leaves the metric "
        "unchanged. It carries no discriminating information; rank candidates by "
        "symmetry order among those passing the δ gate, not by raw Kurlin."
    )
    out.append("")
    out.append("## Ideal cells (conventional setting)")
    out.append("")
    out.append(md_table(
        ["system", "order", "a", "b", "c", "α", "β", "γ"],
        [
            [
                name,
                str(refs[name]["order"]),
                *[f"{x:.4g}" for x in refs[name]["cell"]],
            ]
            for name in SYSTEMS
        ],
    ))
    out.append("")
    out.append("## Noise ladder")
    out.append("")
    out.append(md_table(
        ["level", "σ_edge (fraction)", "σ_angle (°)"],
        [[str(i), f"{e:g}", f"{a:g}"] for i, (e, a) in enumerate(NOISE_LEVELS)],
    ))
    out.append("")
    out.append(f"## Le Page assignment (`max_delta={MAX_DELTA}°`)")
    out.append("")
    assign_rows = []
    spectra: dict[tuple[str, int], dict] = {}

    for base in SYSTEMS:
        for level, (ef, ad) in enumerate(NOISE_LEVELS):
            rng = random.Random((seed << 16) ^ (hash(base) & 0xFFFF) ^ (level * 9973))
            # Conventional-setting cell: fixed candidate operators and the cell
            # share one basis (no reduction flip). Spectrum scoring uses this.
            cell = refs[base]["cell"] if level == 0 else perturb(
                refs[base]["cell"], ef, ad, rng,
            )
            # Assignment reduces first (Lebedev/Le Page domain is the reduced cell).
            reduced = _reduce(cell)
            ls = lattice_symmetry(reduced, max_delta=MAX_DELTA)
            assign_rows.append([
                base,
                str(level),
                f"{ef:g}",
                f"{ad:g}",
                ls.crystal_system,
                str(ls.order),
                " / ".join(f"{x:.4g}" for x in cell),
            ])
            # Spectrum vs all holohedries, scored in the conventional setting.
            spec = {}
            for sys in SYSTEMS:
                kurlin = kurlin_distance_to_symmetry(cell, refs[sys]["ops"])
                lp = max_le_page(cell, refs[sys]["two_folds"])
                spec[sys] = (lp, kurlin)
            spectra[(base, level)] = {"cell": cell, "assigned": ls, "spec": spec}

    out.append(md_table(
        ["base", "level", "σ_edge", "σ_angle°", "assigned", "order", "cell"],
        assign_rows,
    ))
    out.append("")

    # Wide Kurlin tables per base system
    out.append("## Kurlin distance to each holohedry (Å)")
    out.append("")
    out.append(
        "Rows = noise level for a fixed base ideal cell; columns = candidate "
        "holohedry. Values are Kurlin root distances (0 = exact match)."
    )
    out.append("")
    for base in SYSTEMS:
        out.append(f"### Base: {base}")
        out.append("")
        rows = []
        for level, _ in enumerate(NOISE_LEVELS):
            spec = spectra[(base, level)]["spec"]
            rows.append(
                [str(level)] + [fmt(spec[sys][1]) for sys in SYSTEMS]
            )
        out.append(md_table(["level", *SYSTEMS], rows))
        out.append("")

    out.append("## Max Le Page δ to each holohedry (°)")
    out.append("")
    out.append(
        "Same layout; entry is the largest Le Page delta among the two-folds "
        "of that candidate holohedry (empty two-fold set → 0)."
    )
    out.append("")
    for base in SYSTEMS:
        out.append(f"### Base: {base}")
        out.append("")
        rows = []
        for level, _ in enumerate(NOISE_LEVELS):
            spec = spectra[(base, level)]["spec"]
            rows.append(
                [str(level)] + [fmt(spec[sys][0]) for sys in SYSTEMS]
            )
        out.append(md_table(["level", *SYSTEMS], rows))
        out.append("")

    # Combined detailed tables for a few representative bases
    out.append("## Combined spectra (selected bases)")
    out.append("")
    for base in ("cubic", "tetragonal", "monoclinic"):
        out.append(f"### {base}")
        out.append("")
        for level in (0, 2, 4, 5):
            meta = spectra[(base, level)]
            out.append(
                f"**level {level}** → assigned "
                f"`{meta['assigned'].crystal_system}` "
                f"(order {meta['assigned'].order})"
            )
            out.append("")
            rows = [
                [sys, fmt(lp), fmt(k)]
                for sys, (lp, k) in meta["spec"].items()
            ]
            out.append(md_table(
                ["holohedry", "max Le Page δ (°)", "Kurlin (Å)"],
                rows,
            ))
            out.append("")

    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--write", type=Path, default=None,
        help="optional path to write the markdown report",
    )
    args = ap.parse_args()
    text = run(seed=args.seed)
    print(text)
    if args.write is not None:
        args.write.write_text(text)
        print(f"[wrote {args.write}]", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
