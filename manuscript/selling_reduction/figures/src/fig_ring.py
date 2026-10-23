"""Figure 1: the closure of the monoclinic 30 x 40 x 50 A, beta = 100 deg cell (Table 1)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), "..")
INK, MUTED, ACC = "#1f1f1d", "#6b6a66", "#b2342b"
SHAPE = {"M0": 0, "M4": 0, "M1": 1, "M3": 1, "M2": 2, "M5": 2}
COL = ["#2a78d6", "#e08a2b", "#3a9a5b"]
LAB = ["30, 40, 50 Å, β = 100°", "30, 40, 53.7 Å, β = 113.4°", "50, 40, 53.7 Å, β = 146.6°"]
ring = ["M0", "M1", "M5", "M4", "M3", "M2"]                      # the ring M0-M1-M5-M4-M3-M2-M0
ang = {m: np.pi / 2 - k * np.pi / 3 for k, m in enumerate(ring)}
pos = {m: (np.cos(a), np.sin(a)) for m, a in ang.items()}
path = {("M0", "M1"): "1: ab", ("M1", "M5"): "2: bc", ("M5", "M4"): "3: db"}   # the word ab.bc.db, each flip read
other = {}                                                                      # in the labelling current at that step
fig, ax = plt.subplots(figsize=(3.4, 3.6))
for i, m in enumerate(ring):
    n = ring[(i + 1) % 6]
    on = (m, n) in path
    (x1, y1), (x2, y2) = pos[m], pos[n]
    ax.plot([x1, x2], [y1, y2], color=ACC if on else "#9a9893", lw=2.2 if on else 1.0, zorder=1)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    lab = path.get((m, n)) or other.get((m, n))
    if lab:
        ax.text(mx * 1.18, my * 1.18, lab, fontsize=8, style="italic", color=ACC if on else MUTED, ha="center", va="center")
(x0, y0), (x4, y4) = pos["M0"], pos["M4"]
ax.annotate("", xy=(x4, y4 + 0.17), xytext=(x0, y0 - 0.17),
            arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.2, ls=(0, (4, 2))))
ax.text(0.08, 0.05, "2, m", color=ACC, fontsize=8, ha="left")
for m, (x, y) in pos.items():
    ax.scatter([x], [y], s=520, color=COL[SHAPE[m]], edgecolors="white", linewidths=1.5, zorder=2)
    ax.text(x, y, f"$M_{m[1]}$", ha="center", va="center", color="white", fontsize=9, weight="bold", zorder=3)
for k, l in enumerate(LAB):
    ax.scatter([-1.15], [-1.55 - 0.2 * k], s=40, color=COL[k])
    ax.text(-1.02, -1.55 - 0.2 * k, l, fontsize=7, va="center", color=INK)
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-2.05, 1.3); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(os.path.join(OUT, "fig1_ring.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_ring.png"), bbox_inches="tight", dpi=200)
