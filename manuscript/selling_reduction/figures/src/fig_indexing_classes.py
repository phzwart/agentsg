"""Figure 4: distinct indexings as the classes of the Laue group in the band's symmetry coset."""
import numpy as np, matplotlib, textwrap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fractions import Fraction
from matplotlib.colors import LinearSegmentedColormap, Normalize
from agentsg.cell.strain_closure import widened_closure, indexing_classes
from agentsg.cell.metric import UnitCell, params_from_metric

INK, MUTED, SURF = "#1f1f1d", "#6b6a66", "#ffffff"
ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("blue", ramp); norm = Normalize(0, 3.0)

def arr(M): return np.array([[float(x) for x in r] for r in M.rows])

def basis_label(P):
    """new basis vectors in terms of the reference's a, b, c (columns of P)."""
    out = []
    for j in range(3):
        terms = []
        for i, n in enumerate("abc"):
            v = Fraction(P.rows[i][j])
            if v == 0: continue
            s = "-" if v < 0 else "+"
            mag = abs(v)
            terms.append((s, ("" if mag == 1 else str(mag)) + n))
        txt = "".join(f"{s}{t}" for s, t in terms).lstrip("+")
        out.append(txt or "0")
    return ", ".join(out)

def op_symbol(h):
    """rotation order and axis [uvw] of the proper part of a lattice operator."""
    d = round(np.linalg.det(h)); R = h * d
    n = {3: 1, -1: 2, 0: 3, 1: 4, 2: 6}[int(round(np.trace(R)))]
    if n == 1: return "1"
    w, V = np.linalg.eig(R)
    v = np.real(V[:, np.argmin(abs(w - 1))])
    v = v / np.max(np.abs(v)); 
    for k in range(1, 7):
        if np.allclose(k * v, np.round(k * v), atol=1e-6): v = np.round(k * v).astype(int); break
    g = np.gcd.reduce(np.abs(v[v != 0])) if np.any(v) else 1
    v = v // g
    if v[np.nonzero(v)[0][0]] < 0: v = -v
    return f"{n} [{''.join(str(x) if x >= 0 else str(-x) + chr(0x0305) for x in v)}]"

def simplest_twin(ci, cj):
    """the reference-frame operator h with Pj = h Pi, choosing within the classes the simplest proper one."""
    best = None
    for Pi in ci.operators[:1]:
        Pi_inv = np.linalg.inv(arr(Pi))
        for Pj in cj.operators:
            h = np.round(arr(Pj) @ Pi_inv).astype(int)
            if round(np.linalg.det(h)) != 1: continue
            key = (int(np.abs(h).sum()), op_symbol(h))
            if best is None or key < best[0]: best = (key, h)
    return op_symbol(best[1])

cases = [("P 3, a = b = 50, c = 80 Å", (50.0, 50.0, 80.0, 90, 90, 120), "P 3"),
         ("P 2₁3, a = 60 Å", (60.0, 60.0, 60.0, 90, 90, 90), "P 21 3"),
         ("P 1, worked cell", (30.02, 39.97, 50.03, 90.25, 92.05, 89.80), "P 1")]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), dpi=160); fig.patch.set_facecolor(SURF)
for ax, (title, cell, sg) in zip(axes.ravel(), cases):
    from agentsg.cell.strain_closure import LEPAGE_STRAIN, band_for
    ref = widened_closure(cell, sg, eps=band_for(LEPAGE_STRAIN))
    G = np.array(UnitCell(*cell).metric_tensor(), float)
    new = params_from_metric((G * 1.002).tolist())          # the reference measured again, 0.1 % error in the length scale
    r = indexing_classes(ref, new, sg, accept=LEPAGE_STRAIN)
    def simplest(c):
        """display operator: the one nearest the identity (proper first, then fewest changes)."""
        return min(c.operators, key=lambda P: (round(np.linalg.det(arr(P))) != 1, float(np.abs(arr(P) - np.eye(3)).sum())))
    C = sorted(r.classes, key=lambda c: (round(c.representative.price, 6), float(np.abs(arr(simplest(c)) - np.eye(3)).sum())))
    n = len(C)
    ang = np.pi / 2 + 2 * np.pi * np.arange(n) / max(n, 1)
    rad = 0 if n == 1 else (1.35 if n == 4 else 1.0)
    pos = [(rad * np.cos(a), rad * np.sin(a)) for a in ang]
    for i in range(n):
        for j in range(i + 1, n):
            (x1, y1), (x2, y2) = pos[i], pos[j]
            ax.plot([x1, x2], [y1, y2], color=INK, lw=1.6, zorder=1)
            lab = simplest_twin(C[i], C[j])
            diag = n == 4 and abs(i - j) == 2
            t = 0.5 if not diag else (0.28 if i == 0 else 0.72)
            if n == 4 and not diag:
                t = 0.62 if (i == 0 or j == 2) else 0.5
            ax.text(x1 + t * (x2 - x1), y1 + t * (y2 - y1), lab, fontsize=7, ha="center", va="center", color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="#d7d5cf"), zorder=3)
    for k, ((x, y), c) in enumerate(zip(pos, C)):
        v = 100 * c.representative.price
        ax.scatter([x], [y], s=1500, color=cmap(norm(v)), edgecolors=SURF, linewidths=2, zorder=2)
        ax.text(x, y + 0.03, f"{k + 1}", ha="center", va="center", fontsize=10, weight="bold",
                color="#ffffff" if norm(v) > 0.5 else INK, zorder=4)
        ax.text(x, y - 0.3, f"({basis_label(simplest(c))})\n{v:.2f}% · {c.size} op{'s' if c.size > 1 else ''}",
                ha="center", va="top", fontsize=6.8, color=INK, zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none"))
    ax.set_title(f"({'abc'[cases.index((title, cell, sg))]}) {title}\n|H$_ε$| = {r.symmetry_order}, |$\\mathcal{{L}}^+$| = {r.group_order // 2}: {n} distinct indexing{'s' if n > 1 else ''}",
                 fontsize=8, color=INK, loc="left")
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.0 if n == 4 else -1.55, 1.75 if n == 4 else 1.35); ax.set_aspect("equal"); ax.axis("off")
import os
OUT = os.environ.get("SELLINGPROOFS_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
fig.tight_layout(rect=(0, 0, 0.965, 1), w_pad=0.2)
cax = fig.add_axes((0.955, 0.2, 0.01, 0.55))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label("class price (% strain)", fontsize=7, color=INK); cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig3_indexing_classes.{ext}"), facecolor=SURF, bbox_inches="tight")
