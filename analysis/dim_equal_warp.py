#!/usr/bin/env python3
"""Warp the leanmap plane so area is equalised by local dimensionality scale r*.

Each local ball of radius r*(z) (where entropy erank first hits 2) has area ~ π r*².
We treat ρ = 1/r*² as a density of “2D patches” and push it to uniform area via a
Knothe–Rosenblatt (triangular) map. After the warp, equal display area ≈ equal
dimensional content.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
Z_PATH = ROOT / "analysis/data/pdb_embedding_similarity_full.npy"
RSTAR_PATH = ROOT / "analysis/data/pdb_embedding_radius_erank2.npz"
META_PATH = ROOT / "analysis/data/pdb_embedding_explorer_meta.npz"
FIG_PATH = ROOT / "analysis/figures/pdb_embedding_dim_warp.png"
NPZ_PATH = ROOT / "analysis/data/pdb_embedding_dim_warp.npz"

# floor so ultra-small r* do not explode the density
R_FLOOR = 0.05
R_CEIL = 4.0


def crystal_system_ids(sg: np.ndarray) -> np.ndarray:
    # mirror embedding_explorer buckets
    out = np.full(len(sg), -1, dtype=np.int8)
    # triclinic 1-2, mono 3-15, ortho 16-74, tetra 75-142, trig 143-167,
    # hex 168-194, cubic 195-230
    bins = [
        (1, 2, 0),
        (3, 15, 1),
        (16, 74, 2),
        (75, 142, 3),
        (143, 167, 4),
        (168, 194, 5),
        (195, 230, 6),
    ]
    for lo, hi, code in bins:
        out[(sg >= lo) & (sg <= hi)] = code
    return out


SYSTEMS = ("triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic")
SYS_COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d")


def fill_rstar(r: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill of NaNs on the grid (iterative)."""
    out = r.copy()
    for _ in range(32):
        nan = ~np.isfinite(out)
        if not nan.any():
            break
        filled = out.copy()
        for i, j in zip(*np.where(nan)):
            nb = []
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < out.shape[0] and 0 <= jj < out.shape[1] and np.isfinite(out[ii, jj]):
                    nb.append(out[ii, jj])
            if nb:
                filled[i, j] = float(np.median(nb))
        out = filled
    # any remaining → ceiling (low dimensional weight)
    out[~np.isfinite(out)] = R_CEIL
    return np.clip(out, R_FLOOR, R_CEIL)


def knothe_rosenblatt(x0: np.ndarray, x1: np.ndarray, rho: np.ndarray):
    """Triangular OT map pushing density ρ on the grid to the unit square.

    Returns callables mapping (z0, z1) → (u, v) via bilinear interpolation of
    the discrete CDF map, and the warped grid coordinates (U, V).
    """
    # rho shape (ny, nx) with axes (x1, x0)
    ny, nx = rho.shape
    dx = float(x0[1] - x0[0]) if nx > 1 else 1.0
    dy = float(x1[1] - x1[0]) if ny > 1 else 1.0
    mass = rho * dx * dy
    total = mass.sum()
    if total <= 0:
        raise RuntimeError("zero density")
    mass = mass / total

    # marginal on y (rows)
    marg_y = mass.sum(axis=1)  # (ny,)
    cdf_y = np.cumsum(marg_y)
    cdf_y = np.clip(cdf_y / cdf_y[-1], 0.0, 1.0)

    # conditional CDF in x for each y-row
    U = np.zeros_like(mass)
    V = np.zeros_like(mass)
    for i in range(ny):
        row = mass[i]
        s = row.sum()
        if s <= 0:
            U[i] = np.linspace(0.0, 1.0, nx)
        else:
            U[i] = np.clip(np.cumsum(row) / s, 0.0, 1.0)
        V[i] = cdf_y[i]

    # interpolators: map original (z0,z1) → (u,v)
    # RegularGridInterpolator expects (x1, x0) order matching rho layout
    fu = RegularGridInterpolator(
        (x1, x0), U, bounds_error=False, fill_value=None
    )
    fv = RegularGridInterpolator(
        (x1, x0), V, bounds_error=False, fill_value=None
    )

    def warp_xy(z0: np.ndarray, z1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.column_stack([z1, z0])
        u = np.clip(fu(pts), 0.0, 1.0)
        v = np.clip(fv(pts), 0.0, 1.0)
        return u, v

    return warp_xy, U, V, mass


def upsample_grid(
    x0: np.ndarray, x1: np.ndarray, field: np.ndarray, factor: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear upsample of a grid field for a smoother Knothe map."""
    from scipy.interpolate import RegularGridInterpolator as RGI

    fine0 = np.linspace(x0[0], x0[-1], (len(x0) - 1) * factor + 1)
    fine1 = np.linspace(x1[0], x1[-1], (len(x1) - 1) * factor + 1)
    f = RGI((x1, x0), field, bounds_error=False, fill_value=None)
    YY, XX = np.meshgrid(fine1, fine0, indexing="ij")
    pts = np.column_stack([YY.ravel(), XX.ravel()])
    out = f(pts).reshape(YY.shape)
    return fine0, fine1, out


def main() -> None:
    Z = np.load(Z_PATH).astype(np.float64)
    meta = np.load(META_PATH, allow_pickle=True)
    sg = meta["sg_number"]
    sys_id = crystal_system_ids(sg)

    pack = np.load(RSTAR_PATH)
    x0 = pack["x0"].astype(np.float64)
    x1 = pack["x1"].astype(np.float64)
    r_raw = pack["r_star"].astype(np.float64)
    hit = np.isfinite(r_raw)
    r_star = fill_rstar(r_raw)

    x0f, x1f, r_fine = upsample_grid(x0, x1, r_star, factor=4)
    # upsample hit mask (nearest) so empty grid cells carry no patch-mass
    _, _, hit_fine = upsample_grid(x0, x1, hit.astype(np.float64), factor=4)
    hit_fine = hit_fine >= 0.5
    rho = np.where(hit_fine, 1.0 / (r_fine ** 2), 0.0)
    if float(rho.sum()) <= 0:
        rho = 1.0 / (r_fine ** 2)

    warp_xy, U, V, mass = knothe_rosenblatt(x0f, x1f, rho)
    u, v = warp_xy(Z[:, 0], Z[:, 1])

    span0 = np.ptp(Z[:, 0])
    span1 = np.ptp(Z[:, 1])
    aspect = span0 / max(span1, 1e-9)
    W = np.column_stack([u * aspect, v])
    W -= W.mean(axis=0)
    W += Z.mean(axis=0)

    fr = RegularGridInterpolator(
        (x1, x0), r_star, bounds_error=False, fill_value=R_CEIL
    )
    r_pt = fr(np.column_stack([Z[:, 1], Z[:, 0]]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    rng = np.random.default_rng(0)
    show = rng.choice(len(Z), size=min(80_000, len(Z)), replace=False)
    vmin = max(R_FLOOR, float(np.nanpercentile(r_pt, 5)))
    vmax = float(np.nanpercentile(r_pt, 95))

    ax = axes[0]
    sc = ax.scatter(
        Z[show, 0], Z[show, 1], c=r_pt[show], s=1.5, cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(r"$r^\star$ (erank$=2$)")
    ax.set_title("original leanmap")
    ax.set_xlabel(r"$z_0$")
    ax.set_ylabel(r"$z_1$")
    ax.set_aspect("equal")

    ax = axes[1]
    sc = ax.scatter(
        W[show, 0], W[show, 1], c=r_pt[show], s=1.5, cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label(r"$r^\star$ (erank$=2$)")
    ax.set_title(r"dim-equalised warp  ($\rho\propto 1/r^{\star 2}$)")
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$w_1$")
    ax.set_aspect("equal")

    ax = axes[2]
    for code, name, col in zip(range(7), SYSTEMS, SYS_COLORS):
        m = sys_id[show] == code
        if not np.any(m):
            continue
        ax.scatter(
            W[show][m, 0], W[show][m, 1], c=col, s=1.5,
            label=name, rasterized=True, alpha=0.7,
        )
    ax.legend(markerscale=4, fontsize=7, loc="best", frameon=False)
    ax.set_title("warped · crystal system")
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$w_1$")
    ax.set_aspect("equal")

    fig.suptitle(
        r"Area warp: equal display area ≈ equal local 2D-patch mass $\mathrm{d}A/r^{\star 2}$",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    print("wrote", FIG_PATH)

    np.savez(
        NPZ_PATH,
        Z=Z.astype(np.float32),
        W=W.astype(np.float32),
        r_star_point=r_pt.astype(np.float32),
        x0=x0f,
        x1=x1f,
        r_star_grid=r_fine,
        rho=rho,
        U=U,
        V=V,
        mass=mass,
        r_floor=np.array([R_FLOOR]),
        r_ceil=np.array([R_CEIL]),
    )
    print("wrote", NPZ_PATH)

    def patch_mass_cv(X: np.ndarray, r: np.ndarray, nbin: int = 12) -> float:
        xedges = np.linspace(X[:, 0].min(), X[:, 0].max(), nbin + 1)
        yedges = np.linspace(X[:, 1].min(), X[:, 1].max(), nbin + 1)
        w = 1.0 / np.maximum(r, R_FLOOR) ** 2
        H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[xedges, yedges], weights=w)
        occ = H[H > 0]
        return float(occ.std() / occ.mean()) if occ.size else float("nan")

    cv0 = patch_mass_cv(Z[show], r_pt[show])
    cv1 = patch_mass_cv(W[show], r_pt[show])
    print(f"patch-mass CV over 12×12 display bins: original={cv0:.3f}  warped={cv1:.3f}")


if __name__ == "__main__":
    main()
