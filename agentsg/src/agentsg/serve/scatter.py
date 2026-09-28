"""Scatter plot of a PDB-search hit set in the SVD basis of its Kurlin roots."""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np


def root_svd(roots) -> dict[str, Any]:
    """Mean-centred SVD of an (n, 6) Kurlin-root matrix.

    Returns scores on the first two right singular vectors, the column mean,
    ``Vt``, and a JSON-ready spectrum. ``n`` must be at least 2.
    """
    X = np.asarray(roots, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] != 6:
        raise ValueError("need at least two 6-vectors of Kurlin roots")
    mean = X.mean(axis=0)
    centred = X - mean
    _, sigma, vt = np.linalg.svd(centred, full_matrices=False)
    take = min(2, vt.shape[0])
    scores = centred @ vt[:take].T
    if take == 1:
        scores = np.column_stack([scores.ravel(), np.zeros(len(scores))])
    ss = float(np.sum(sigma ** 2))
    frac = (sigma ** 2 / ss).tolist() if ss > 0 else [0.0] * len(sigma)
    cum = np.cumsum(frac).tolist()
    positive = sigma[sigma > 0]
    if positive.size:
        p = positive / positive.sum()
        effective = float(np.exp(-(p * np.log(p)).sum()))
    else:
        effective = 0.0
    return {
        "scores": scores,
        "mean": mean,
        "vt": vt,
        "svd": {
            "n": int(X.shape[0]),
            "centered": True,
            "feature": "root_invariant r0..r5",
            "singular_values": [float(s) for s in sigma],
            "variance_frac": [float(v) for v in frac],
            "variance_cum": [float(v) for v in cum],
            "effective_rank": effective,
        },
    }


def project_query(query_root, mean, vt) -> list[float]:
    """Place the query cell in the hit-set SVD basis (it is not part of the fit)."""
    q = np.asarray(query_root, dtype=np.float64).reshape(6) - mean
    take = min(2, vt.shape[0])
    xy = q @ vt[:take].T
    if take == 1:
        return [float(xy), 0.0]
    return [float(xy[0]), float(xy[1])]


def render_scatter_png(
    scores,
    labels: list[str],
    color,
    *,
    variance_frac: list[float],
    query_xy: list[float] | None = None,
) -> bytes:
    """PNG of the PC1–PC2 scatter. Raises ImportError without matplotlib."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    xy = np.asarray(scores, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=color, s=36, cmap="viridis",
        linewidths=0.4, edgecolors="black", alpha=0.9, zorder=2,
    )
    if len(labels) <= 40:
        for x, y, lab in zip(xy[:, 0], xy[:, 1], labels):
            ax.annotate(
                lab, (x, y), textcoords="offset points", xytext=(4, 3),
                fontsize=7, color="0.15",
            )
    if query_xy is not None:
        ax.scatter(
            [query_xy[0]], [query_xy[1]], marker="*", s=180, c="#c0392b",
            linewidths=0.4, edgecolors="black", zorder=4, label="query",
        )
        ax.legend(frameon=False, loc="best")
    pc1 = 100.0 * variance_frac[0] if variance_frac else 0.0
    pc2 = 100.0 * variance_frac[1] if len(variance_frac) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({pc1:.1f}% of root variance)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% of root variance)")
    ax.set_title("Kurlin roots of the search hits")
    ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("root distance (Å)")
    fig.tight_layout()
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=140)
    finally:
        plt.close(fig)
    return buf.getvalue()


def scatter_payload(
    roots,
    labels: list[str],
    distances: list[float],
    query_root,
) -> dict[str, Any]:
    """SVD plus a base64 PNG. ``png_base64`` is null if matplotlib is missing."""
    fit = root_svd(roots)
    xy = [[float(a), float(b)] for a, b in fit["scores"]]
    query_xy = project_query(query_root, fit["mean"], fit["vt"])
    png_b64 = None
    png_error = None
    try:
        raw = render_scatter_png(
            fit["scores"], labels, distances,
            variance_frac=fit["svd"]["variance_frac"],
            query_xy=query_xy,
        )
        png_b64 = base64.b64encode(raw).decode("ascii")
    except ImportError:
        png_error = "matplotlib is not installed"
    out = {
        "svd": fit["svd"],
        "xy": xy,
        "query_xy": query_xy,
        "png_base64": png_b64,
    }
    if png_error:
        out["png_error"] = png_error
    return out
