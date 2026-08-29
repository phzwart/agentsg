#!/usr/bin/env python3
"""At each latent-grid center, find the smallest radius where entropy erank = 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
Z_PATH = ROOT / "analysis/data/pdb_embedding_similarity_full.npy"
META_PATH = ROOT / "analysis/data/pdb_embedding_explorer_meta.npz"
FIG_PATH = ROOT / "analysis/figures/pdb_embedding_radius_erank2.png"
NPZ_PATH = ROOT / "analysis/data/pdb_embedding_radius_erank2.npz"

TARGET = 2.0
R_MAX = 4.0
N_RADII = 64


def entropy_effective_rank(sig: np.ndarray) -> float:
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig[sig > 0]
    if sig.size == 0:
        return 0.0
    p = sig / sig.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def local_dim(S_local: np.ndarray) -> float:
    if S_local.shape[0] < 2:
        return 0.0
    X = S_local - S_local.mean(axis=0, keepdims=True)
    sig = np.linalg.svd(X, compute_uv=False)
    if float(np.sum(sig**2)) <= 0:
        return 0.0
    return entropy_effective_rank(sig)


def radius_at_target(
    d_sorted: np.ndarray,
    S_sorted: np.ndarray,
    target: float = TARGET,
    r_max: float = R_MAX,
    n_radii: int = N_RADII,
) -> tuple[float, float, int]:
    """Smallest radius where erank reaches `target`.

    Returns (r_star, erank_at_r_max, n_at_r_max). NaN if never reaches target.
    """
    m = d_sorted <= r_max
    d = d_sorted[m]
    S = S_sorted[m]
    n_max = int(d.size)
    if n_max < 2:
        return float("nan"), 0.0, n_max

    e_max = local_dim(S)
    r_lo = float(max(d[1], 1e-8))  # need ≥2 points
    radii = np.unique(np.geomspace(r_lo, r_max, n_radii))

    prev_r: float | None = None
    prev_e: float | None = None
    for r in radii:
        n = int(np.searchsorted(d, r, side="right"))
        if n < 2:
            continue
        e = local_dim(S[:n])
        if e >= target:
            if prev_e is None:
                return float(r), e_max, n_max
            if e <= prev_e:  # flat / non-monotone: take first crossing radius
                return float(r), e_max, n_max
            t = (target - prev_e) / (e - prev_e)
            t = float(np.clip(t, 0.0, 1.0))
            return float(prev_r + t * (r - prev_r)), e_max, n_max
        prev_r, prev_e = float(r), e

    return float("nan"), e_max, n_max


def main() -> None:
    Z = np.load(Z_PATH).astype(np.float64)
    meta = np.load(META_PATH, allow_pickle=True)
    S = meta["S"].astype(np.float64)
    assert len(Z) == len(S)

    spacing = 1.0
    x0 = np.arange(np.floor(Z[:, 0].min()), np.ceil(Z[:, 0].max()) + 1e-9, spacing)
    x1 = np.arange(np.floor(Z[:, 1].min()), np.ceil(Z[:, 1].max()) + 1e-9, spacing)

    tree = cKDTree(Z)
    r_star = np.full((len(x1), len(x0)), np.nan, dtype=np.float64)
    e_at_rmax = np.zeros_like(r_star)
    n_at_rmax = np.zeros_like(r_star)

    for i, cy in enumerate(x1):
        for j, cx in enumerate(x0):
            idxs = tree.query_ball_point([cx, cy], r=R_MAX)
            if len(idxs) < 2:
                continue
            idxs = np.asarray(idxs, dtype=np.int64)
            d = np.linalg.norm(Z[idxs] - np.array([cx, cy]), axis=1)
            order = np.argsort(d)
            r_star[i, j], e_at_rmax[i, j], n_at_rmax[i, j] = radius_at_target(
                d[order], S[idxs][order]
            )

    hit = np.isfinite(r_star)
    vals = r_star[hit]
    print(
        f"grid {len(x0)}×{len(x1)}; reached erank≥{TARGET:g}: {hit.sum()}/{r_star.size}; "
        f"r* med={np.median(vals):.4g} mean={vals.mean():.4g} "
        f"[{vals.min():.4g},{vals.max():.4g}]"
    )
    print(
        f"never reached (but n≥2 in R_max): "
        f"{((~hit) & (n_at_rmax >= 2)).sum()}; "
        f"empty: {(n_at_rmax < 2).sum()}"
    )

    # plot: show r*; mask never-reached / empty
    plot = np.ma.array(r_star, mask=~hit)
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [
        x0[0] - spacing / 2,
        x0[-1] + spacing / 2,
        x1[0] - spacing / 2,
        x1[-1] + spacing / 2,
    ]
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#e8e8e8")
    im = ax.imshow(
        plot,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        interpolation="nearest",
        norm=LogNorm(
            vmin=max(float(vals.min()), 1e-4),
            vmax=max(float(vals.max()), 1e-3),
        ),
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(rf"radius $r^\star$ where entropy erank $= {TARGET:g}$")
    ax.set_xlabel(r"$z_0$")
    ax.set_ylabel(r"$z_1$")
    ax.set_title(
        rf"Radius to entropy effective rank ${TARGET:g}$"
        f"\nunit grid · search ≤ {R_MAX:g} · grey = empty / never reaches {TARGET:g}"
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=220)
    print("wrote", FIG_PATH)

    np.savez(
        NPZ_PATH,
        x0=x0,
        x1=x1,
        r_star=r_star,
        e_at_rmax=e_at_rmax,
        n_at_rmax=n_at_rmax,
        spacing=np.array([spacing]),
        target=np.array([TARGET]),
        r_max=np.array([R_MAX]),
        definition=np.array(["entropy_roy_vetterli"]),
    )
    print("wrote", NPZ_PATH)

    # quick histogram stats
    qs = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95])
    print("r* quantiles [5,25,50,75,95]:", " ".join(f"{q:.4g}" for q in qs))


if __name__ == "__main__":
    main()
