"""Figure 4: non-isomorphism against the price of pseudo-symmetry in the PDB.

Cumulative distributions of (blue) the price between deposited cells of one crystal form
(adjacent ids, same space group, below 20 %; manuscript/supplement/benchmark/pdb_nonisomorphism.py)
and (orange) the price of the cheapest pseudo-symmetric alternative of each entry that
Le Page's 3 degrees finds (manuscript/supplement/benchmark/pdb_lepage.jsonl, priced cases).

    PYTHONPATH=agentsg/src python manuscript/selling_reduction/figures/src/fig_pdb_prices.py
"""
import json
import os

import duckdb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from agentsg.cell.metric import metric_tensor                       # noqa: E402
from agentsg.cell.strain_closure import _price                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..", "..")
BENCH = os.path.join(ROOT, "manuscript", "supplement", "benchmark")
OUT = os.environ.get("SELLINGPROOFS_OUT", os.path.join(HERE, ".."))
INK, MUTED, GRID = "#1f1f1d", "#6b6a66", "#d7d5cf"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def nonisomorphism():
    con = duckdb.connect(os.path.join(os.environ.get("SELLINGPROOFS_DATA", os.path.join(ROOT, "data")), "pdb_cells.duckdb"), read_only=True)
    rows = con.execute("select pdb_id, a, b, c, alpha, beta, gamma, sg_hm from cells order by pdb_id").fetchall()
    out = []
    for r1, r2 in zip(rows, rows[1:]):
        if r1[0][:3] == r2[0][:3] and r1[7] == r2[7]:
            G1 = np.array(metric_tensor(tuple(r1[1:7])), float)
            G2 = np.array(metric_tensor(tuple(r2[1:7])), float)
            if _price(G2, G1, fit_scale=False)[0] < 0.2:                       # one crystal form, as in
                out.append(_price(G2, G1)[0])                                   # pdb_nonisomorphism.py
    return np.array(out)


def pseudo():
    out = []
    for line in open(os.path.join(BENCH, "pdb_lepage.jsonl")):
        r = json.loads(line)
        if r.get("extra"):
            m = min(x for v in r["extra"].values() for x in v)
            if m >= 1e-4:
                out.append(m)
    return np.array(out)


def ecdf(x):
    x = np.sort(x)
    return x, np.arange(1, len(x) + 1) / len(x)


def main():
    ni, ps = nonisomorphism(), pseudo()
    q90, q95 = np.percentile(ni, [90, 95])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for x in (100 * q90, 100 * q95, 14.82):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    ax.text(100 * q90, 1.02, "90 %", color=MUTED, ha="center", va="bottom", fontsize=7)
    ax.text(100 * q95, 1.02, "95 %", color=MUTED, ha="center", va="bottom", fontsize=7)
    ax.text(14.82, 1.02, "Le Page 3°", color=MUTED, ha="center", va="bottom", fontsize=7)
    for data, col, lab in ((ni, BLUE, f"one crystal form ({len(ni):,} pairs)"),
                           (ps, ORANGE, f"cheapest pseudo-symmetry ({len(ps):,} entries)")):
        x, y = ecdf(100 * data)
        ax.plot(x, y, color=col, lw=2, label=lab)
    ax.set_xscale("log"); ax.set_xlim(0.05, 20); ax.set_ylim(0, 1)
    ax.set_xlabel("price (%)", color=INK); ax.set_ylabel("cumulative fraction", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7, labelcolor=INK)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_pdb_prices.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
