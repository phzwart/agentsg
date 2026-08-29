#!/usr/bin/env python3
"""Local entropy effective-rank map of Kurlin similarity roots on the leanmap plane."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Z_PATH = ROOT / "analysis/data/pdb_embedding_similarity_full.npy"
META_PATH = ROOT / "analysis/data/pdb_embedding_explorer_meta.npz"
FIG_PATH = ROOT / "analysis/figures/pdb_embedding_local_svd_dim.png"
NPZ_PATH = ROOT / "analysis/data/pdb_embedding_local_svd_dim.npz"


def entropy_effective_rank(sig: np.ndarray) -> float:
    """Roy–Vetterli effective rank: exp(H), H = -Σ p ln p, p = σ / Σσ."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig[sig > 0]
    if sig.size == 0:
        return 0.0
    p = sig / sig.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def local_dim(S_local: np.ndarray) -> float:
    """Entropy effective rank from mean-centred SVD; 0 if <2 points."""
    if S_local.shape[0] < 2:
        return 0.0
    X = S_local - S_local.mean(axis=0, keepdims=True)
    sig = np.linalg.svd(X, compute_uv=False)
    if float(np.sum(sig**2)) <= 0:
        return 0.0
    return entropy_effective_rank(sig)


def main() -> None:
    Z = np.load(Z_PATH).astype(np.float64)
    meta = np.load(META_PATH, allow_pickle=True)
    S = meta["S"].astype(np.float64)
    assert len(Z) == len(S)

    spacing = 1.0
    radius = 0.5
    x0 = np.arange(np.floor(Z[:, 0].min()), np.ceil(Z[:, 0].max()) + 1e-9, spacing)
    x1 = np.arange(np.floor(Z[:, 1].min()), np.ceil(Z[:, 1].max()) + 1e-9, spacing)

    dim = np.zeros((len(x1), len(x0)), dtype=np.float64)
    counts = np.zeros_like(dim)
    r2 = radius * radius
    for i, cy in enumerate(x1):
        for j, cx in enumerate(x0):
            m = (Z[:, 0] - cx) ** 2 + (Z[:, 1] - cy) ** 2 <= r2
            n = int(m.sum())
            counts[i, j] = n
            dim[i, j] = local_dim(S[m]) if n >= 2 else 0.0

    occ = dim[dim > 0]
    print(
        f"grid {len(x0)}×{len(x1)}; occupied n≥2: {(counts >= 2).sum()}/{dim.size}; "
        f"erank med={np.median(occ):.3f} mean={occ.mean():.3f} "
        f"[{occ.min():.3f},{occ.max():.3f}]"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [
        x0[0] - spacing / 2,
        x0[-1] + spacing / 2,
        x1[0] - spacing / 2,
        x1[-1] + spacing / 2,
    ]
    im = ax.imshow(
        dim,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=max(float(dim.max()), 1e-6),
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"entropy effective rank  $\exp(-\sum p\ln p)$,  $p=\sigma/\sum\sigma$")
    ax.set_xlabel(r"$z_0$")
    ax.set_ylabel(r"$z_1$")
    ax.set_title(
        "Local Kurlin-root dimensionality on latent grid\n"
        f"centers every {spacing:g}, ball radius {radius:g}  ·  0 = empty / <2 points"
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=220)
    print("wrote", FIG_PATH)

    np.savez(
        NPZ_PATH,
        x0=x0,
        x1=x1,
        dim=dim,
        counts=counts,
        spacing=np.array([spacing]),
        radius=np.array([radius]),
        definition=np.array(["entropy_roy_vetterli"]),
    )
    print("wrote", NPZ_PATH)


if __name__ == "__main__":
    main()
