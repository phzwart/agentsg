"""Figure: the sixteen members of the idealised closure of the measured worked cell by
their strain depth under the measured metric (z60); the band selects among them."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), "..")
ROWS = [("e", 0.0, 0), ("bc", 0.28, 0), ("ab", 0.49, 0), ("bc.db", 0.62, 0), ("ab.bc", 0.78, 0), ("ab.bc.db", 0.78, 0),
        ("da", 3.76, 2), ("ab.bc.ac", 4.56, 2), ("da.ab", 5.06, 2), ("da.ab.ac", 5.08, 2), ("ab.ac", 5.08, 2),
        ("da.ab.bc", 5.09, 2), ("bc.db.dc", 5.09, 2), ("ab.ac.bc.ac", 5.11, 2), ("da.ab.bc.ac", 5.11, 2), ("ab.ac.bc", 5.11, 2)]
COL = ["#2a78d6", "#9a9893", "#e08a2b"]
fig, ax = plt.subplots(figsize=(5.0, 3.6))
for i, (w, d, k) in enumerate(ROWS[::-1]):
    ax.barh(i, max(d, 0.02), color=COL[k], height=0.72)
    ax.text(max(d, 0.02) + 0.06, i, w, va="center", fontsize=6.5, family="monospace")
ax.axvline(7.4, color="#b2342b", lw=1, ls=(0, (4, 2)))
ax.text(7.3, 15.6, "3° (0.074)", color="#b2342b", fontsize=7, ha="right", va="bottom")
ax.set_yticks([]); ax.set_xlim(0, 8); ax.set_xlabel("depth under the measured metric (% strain)", fontsize=8)
ax.tick_params(labelsize=7)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=COL[0], label="the six members of the ring"),
                   Patch(color=COL[2], label="added by the orthorhombic idealisation")], fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0.3, 1.0))
fig.savefig(os.path.join(OUT, "fig2_depths.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_depths.png"), bbox_inches="tight", dpi=200)
