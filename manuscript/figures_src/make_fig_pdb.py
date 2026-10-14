"""Regenerate Figure 2 (figures/fig_pdb.pdf): 3-D SVD of the PDB key cloud
with the exact Bravais skeleton overlaid in the same projection.

Run from the repository root with the PDB snapshot in place:

    python manuscript/figures_src/make_fig_pdb.py data/pdb_cells.duckdb

Method (Section 4.1, SI entry z:embedding):
  - sorted linear key per primitive cell, divided by V^(1/3);
  - uncentred SVD of the key matrix; display = leading three right
    singular directions (deterministic linear projection of the flat cone);
  - colour afterwards by crystal system (from the stored space group);
  - overlay the skeleton of Table tab:nexus computed from cells exactly as
    in z:skeleton and projected with the same singular vectors.
"""
from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "agentsg/src")
from agentsg.cell.rootform import sorted_linear_key  # noqa: E402
from agentsg.cell.selling_cob import _primitive_or_self  # noqa: E402

SYSTEM_OF = {  # sg number ranges -> crystal system
    (1, 2): "triclinic", (3, 15): "monoclinic", (16, 74): "orthorhombic",
    (75, 142): "tetragonal", (143, 167): "trigonal",
    (168, 194): "hexagonal", (195, 230): "cubic",
}
ORDER = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
         "trigonal", "hexagonal", "cubic"]
PALETTE = {  # fixed assignment, categorical
    "triclinic": "#B9C2CF", "monoclinic": "#8A93A6",
    "orthorhombic": "#2F5FE8", "tetragonal": "#C96A1F",
    "trigonal": "#2E8B6E", "hexagonal": "#7A4FB8", "cubic": "#1B1F27",
}


def system(sgno):
    for (lo, hi), name in SYSTEM_OF.items():
        if lo <= sgno <= hi:
            return name
    return "triclinic"


def nkey(cell, sg=None):
    prim = _primitive_or_self(cell, sg) if sg else cell
    k = np.array(sorted_linear_key(prim))
    a, b, c, al, be, ga = prim
    r = math.radians
    arg = (1 - math.cos(r(al)) ** 2 - math.cos(r(be)) ** 2
           - math.cos(r(ga)) ** 2
           + 2 * math.cos(r(al)) * math.cos(r(be)) * math.cos(r(ga)))
    V = a * b * c * math.sqrt(max(arg, 1e-300))
    return k / V ** (1.0 / 3.0)


def skeleton():
    """Curves and points of Table tab:nexus, as normalised keys."""
    out = {}
    out["cP"] = nkey((4, 4, 4, 90, 90, 90))[None, :]
    out["cI"] = nkey((4, 4, 4, 90, 90, 90), "I 2 3")[None, :]
    out["cF"] = nkey((4, 4, 4, 90, 90, 90), "F 2 3")[None, :]
    rat = np.exp(np.linspace(math.log(0.25), math.log(6.0), 120))
    out["tP"] = np.array([nkey((4, 4, 4 * r, 90, 90, 90)) for r in rat])
    out["tI"] = np.array([nkey((4, 4, 4 * r, 90, 90, 90), "I 4")
                          for r in rat])
    out["hP"] = np.array([nkey((4, 4, 4 * r, 90, 90, 120)) for r in rat])
    al = np.linspace(40.0, 115.0, 120)
    out["hR"] = np.array([nkey((4, 4, 4, a, a, a)) for a in al])
    # the oC(a=b)=tP edge is tP itself; draw the oP cube-corner path instead
    out["oP(a=b)"] = np.array([nkey((4, 4, 4 * r, 90, 90, 90)) for r in rat])
    return out


def main(db_path):
    import duckdb  # requires: pip install -e ".[db]"

    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(
        "select a, b, c, alpha, beta, gamma, sg_number, sg_hm "
        "from cells").fetchall()
    keys, systems = [], []
    for a, b, c, al, be, ga, sgno, hm in rows:
        try:
            keys.append(nkey((a, b, c, al, be, ga), hm))
            systems.append(system(int(sgno)))
        except Exception:
            continue
    X = np.array(keys)
    print("cells with keys:", len(X))

    # deterministic, uncentred SVD; display frame = right singular vectors
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt.T[:, :3]
    print("singular values:", np.round(S[:4], 2))

    skel = {name: pts @ Vt.T[:, :3] for name, pts in skeleton().items()}

    fig = plt.figure(figsize=(11, 5.2), facecolor="white")
    views = [(0, 1), (0, 2)]
    rng = np.random.default_rng(0)
    draw = rng.choice(len(X), size=min(30000, len(X)), replace=False)
    for pi, (i, j) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, pi)
        for name in ORDER:
            m = np.array([systems[d] == name for d in draw])
            ax.scatter(P[draw][m, i], P[draw][m, j], s=2.0,
                       color=PALETTE[name], alpha=0.35, edgecolors="none",
                       label=name if pi == 1 else None)
        for name, pts in skel.items():
            if len(pts) == 1:
                ax.scatter(pts[0, i], pts[0, j], s=34, marker="o",
                           facecolors="white", edgecolors="#1B1F27",
                           linewidths=1.2, zorder=5)
                ax.annotate(name, (pts[0, i], pts[0, j]), fontsize=8,
                            textcoords="offset points", xytext=(5, 4))
            else:
                ax.plot(pts[:, i], pts[:, j], color="#1B1F27",
                        linewidth=1.0, alpha=0.85, zorder=4)
        ax.set_xlabel(f"SV{i+1}")
        ax.set_ylabel(f"SV{j+1}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("Volume-normalised key cloud, leading singular directions, "
                 "with the exact Bravais skeleton", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(f"manuscript/submission/figures/fig_pdb.{ext}", dpi=170)
    print("wrote figures/fig_pdb.{pdf,png}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/pdb_cells.duckdb")
