"""Figure B1 (and Fig. S1). Generate the inputs with sort_steps.py <cell> steps_worked.json / steps_lysozyme.json.
Two-panel Delone sort graph for the manuscript: worked 92-degree cell and lysozyme (P1 indexing)."""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "..", "supplement", "src")]
from sellingproofs.claims.test_z50_specialisations import EXPECTED  # noqa: E402
OUT = os.environ.get("SELLINGPROOFS_OUT", os.path.join(HERE, ".."))
H = HERE + "/"
BR = dict(C1="cI", C3="cF", C5="cP", H4="hP", T1="tI", T2="tI", T5="tP", R1="hR", R3="hR", O1A="oF", O1B="oI", O2="oI",
          O3="oI", O4="oS", O5="oP", M1A="mC", M1B="mC", M2A="mC", M2B="mC", M3="mC", M4="mP", A1="aP", A2="aP", A3="aP")
rows = {6: ["A1"], 5: ["A2"], 4: ["M2A", "M1A", "M1B", "A3", "M4"], 3: ["O2", "O1A", "O1B", "M2B", "M3", "O4", "O5"],
        2: ["T2", "T1", "O3", "R3", "H4", "T5", "R1"], 1: ["C3", "C1", "C5"]}
xs = {6: [3.0], 5: [3.0], 4: [0.4, 1.7, 3.0, 4.3, 5.6], 3: [-0.5, 0.6, 1.7, 2.8, 3.9, 5.0, 6.1],
      2: [-0.5, 0.6, 1.7, 2.8, 3.9, 5.0, 6.1], 1: [1.0, 3.2, 5.4]}
pos = {s: (xs[k][i], k * 1.35) for k, v in rows.items() for i, s in enumerate(v)}
ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("blue", ramp); norm = LogNorm(0.01, 100)
INK, MUTED, SURF, OFF = "#1f1f1d", "#6b6a66", "#ffffff", "#c9c7c1"

def fmt(v):
    if v > 25:
        return ">25"
    return "0" if v < 0.0005 else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}" if v < 1 else (f"{v:.1f}" if v < 10 else f"{v:.0f}"))

FAR, FARFILL = 25.0, "#ebeae6"          # sorts at least 25 % away: grey, no number


def panel(ax, path, label):
    D = json.load(open(path)); R = D["rows"]; d0 = R["measured"]; placed = []
    for (a, b) in EXPECTED:
        step = 100 * R[a][b]; da, db = 100 * d0[a], 100 * d0[b]
        on = abs(np.hypot(da, step) - db) <= 0.03 * max(db, 0.05) and db < 10
        (x1, y1), (x2, y2) = pos[a], pos[b]
        far = db > FAR
        ax.plot([x1, x2], [y1, y2], color=INK if on else ("#e2e0db" if far else OFF), lw=1.4 if on else 0.8, zorder=1,
                ls=(0, (3, 2)) if BR[a] == BR[b] else "-")
        if far or step > FAR:
            continue
        for t in (0.5, 0.36, 0.64, 0.28, 0.72):
            px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            if all(np.hypot(px - qx, py - qy) > 0.42 for qx, qy in placed):
                break
        placed.append((px, py))
        ax.text(px, py, fmt(step), fontsize=5.5, color=INK if on else MUTED, ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.1", fc=SURF, ec="none"))
    for s, (x, y) in pos.items():
        v = 100 * d0[s]
        fc = FARFILL if v > FAR else (cmap(norm(max(v, 0.01))) if v > 0.0005 else "#f4f3f0")
        ax.scatter([x], [y], s=330, color=fc, edgecolors=INK, linewidths=0.4, zorder=2)
        ax.text(x, y, s, ha="center", va="center", fontsize=6, weight="bold", zorder=4,
                color="#ffffff" if FAR >= v > 0.0005 and norm(v) > 0.55 else (MUTED if v > FAR else INK))
        ax.text(x, y - 0.36, BR[s] if v > FAR else f"{BR[s]} {fmt(v)}", ha="center", va="center", fontsize=5.5, color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.08", fc=SURF, ec="none"))
    for k in rows:
        ax.text(-1.3, k * 1.35, str(k), ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(-1.65, 3.5 * 1.35, "free conorm values", ha="center", va="center", rotation=90, fontsize=6.5, color=MUTED)
    c = D["cell"]
    ax.set_title(f"{label} {c[0]:g} × {c[1]:g} × {c[2]:g} Å, {c[3]:g}°, {c[4]:g}°, {c[5]:g}°", fontsize=8, loc="left", color=INK)
    ax.set_xlim(-1.9, 6.6); ax.set_ylim(0.8, 8.9); ax.axis("off")

plt.rcParams["font.family"] = "DejaVu Sans"
for name, src, label in (("figS1_sort_graph", "steps_worked.json", "worked cell"),
                         ("figS2_sort_graph_lysozyme", "steps_lysozyme.json", "lysozyme indexed in P1:")):
    fig, ax = plt.subplots(1, 1, figsize=(6.7, 4.9))
    panel(ax, H + src, label)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.7)
    cb.set_label("distance to the sort (% strain); grey: at least 25 %", fontsize=7, color=MUTED); cb.ax.tick_params(labelsize=6, colors=MUTED); cb.outline.set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
