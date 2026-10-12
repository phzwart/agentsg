"""deformation: the certified, symmetrised deformation distance (the ideal).

``deformation_distance(cell_A, cell_B)`` computes

    d(A -> B) = min over integer M with det +-1 of  || G_A - M^T G_B M ||_F,

the smallest Frobenius-norm change to A's metric tensor, in A's own reduced
frame, that makes A congruent to B; the reported distance is the symmetrised
``max(d(A -> B), d(B -> A))``. A candidate of value D forces

    ||M||_2^2 <= (lambda_max(G_A) + D) / lambda_min(G_B),

so enumerating the unimodular matrices with entries inside that radius
certifies the global minimum in one pass. Candidates are seeded from the
Selling-closure match (``closure_distance`` over a small widening schedule),
which also supplies the congruence-test special case: the ideal thresholded
at the noise scale. The minimiser returns ``DeltaG`` and the strain tensor
``eps = sym(G_A^{-1} DeltaG) / 2`` whose principal values set the resolution
to which the two lattices carry mutual information without rescaling.
"""
from __future__ import annotations

import math

from .canonical import closure_distance
from .metric import UnitCell, params_from_metric

_DEFAULT_SCHEDULE = (0.0, 0.06, 0.12, 0.2)


def _np():
    import numpy

    return numpy


def _metric(cell):
    return _np().array(UnitCell(*cell).metric_tensor(), dtype=float)


def _frob(M):
    np = _np()
    return float(np.sqrt(np.sum(np.asarray(M, dtype=float) ** 2)))


def _unimodular_ball(r):
    """All integer 3x3 matrices with entries in [-r, r] and det +-1."""
    np = _np()
    vals = np.arange(-r, r + 1, dtype=np.int64)
    grids = np.meshgrid(*([vals] * 9), indexing="ij")
    M = np.stack([g.ravel() for g in grids], axis=1).reshape(-1, 3, 3)
    det = (
        M[:, 0, 0] * (M[:, 1, 1] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 1])
        - M[:, 0, 1] * (M[:, 1, 0] * M[:, 2, 2] - M[:, 1, 2] * M[:, 2, 0])
        + M[:, 0, 2] * (M[:, 1, 0] * M[:, 2, 1] - M[:, 1, 1] * M[:, 2, 0])
    )
    return M[(det == 1) | (det == -1)]


def _ball_min(G_anchor, G_other, ball):
    """Min over the ball of ||G_anchor - M^T G_other M||_F and its argmin."""
    np = _np()
    Mf = ball.astype(float)
    conj = np.einsum("nji,jk,nkl->nil", Mf, G_other, Mf)
    vals = np.sqrt(np.sum((conj - G_anchor[None, :, :]) ** 2, axis=(1, 2)))
    i = int(np.argmin(vals))
    return float(vals[i]), ball[i]


def _radius(G_anchor, G_other, D):
    np = _np()
    lam_max = float(np.linalg.eigvalsh(G_anchor)[-1])
    lam_min = float(np.linalg.eigvalsh(G_other)[0])
    return math.sqrt(max(lam_max + D, 0.0) / lam_min)


def deformation_distance(cell_A, cell_B, schedule=_DEFAULT_SCHEDULE, rmax=2):
    """Certified symmetrised deformation distance; see the module docstring.

    Returns a dict with ``d`` (symmetrised), ``d_AB``, ``d_BA``, ``M_AB``,
    ``DeltaG`` (A-frame), ``Delta_cell``, ``strain``, ``principal_strains``,
    ``certified`` (both directions), ``radius_bound`` and ``radius_used``.
    """
    np = _np()
    GA, GB = _metric(cell_A), _metric(cell_B)

    # seed each direction from the Selling-closure match
    best = {"AB": (float("inf"), None), "BA": (float("inf"), None)}
    for br in schedule:
        P_ab, _, _ = closure_distance(cell_A, cell_B, boundary_rel=br)
        P_ba, _, _ = closure_distance(cell_B, cell_A, boundary_rel=br)
        for key, P, Ganc, Goth in (("AB", P_ab, GA, GB), ("BA", P_ba, GB, GA)):
            M = np.array(P, dtype=np.int64)
            v = _frob(Ganc - M.T @ Goth @ M)
            if v < best[key][0]:
                best[key] = (v, M)

    # one certified deepening pass over the data-sized integer ball
    R_ab = _radius(GA, GB, best["AB"][0])
    R_ba = _radius(GB, GA, best["BA"][0])
    r_needed = max(int(math.floor(R_ab)), int(math.floor(R_ba)), 1)
    r_used = min(r_needed, int(rmax))
    ball = _unimodular_ball(r_used)
    v_ab, M_ab = _ball_min(GA, GB, ball)
    v_ba, M_ba = _ball_min(GB, GA, ball)
    if v_ab < best["AB"][0]:
        best["AB"] = (v_ab, M_ab)
    if v_ba < best["BA"][0]:
        best["BA"] = (v_ba, M_ba)

    # the minima only shrink, so the required radius only shrinks with them
    cert_ab = r_used >= math.floor(_radius(GA, GB, best["AB"][0]))
    cert_ba = r_used >= math.floor(_radius(GB, GA, best["BA"][0]))

    d_ab, M = best["AB"]
    d_ba, _ = best["BA"]
    M = np.array(M, dtype=np.int64)
    dG = M.T @ GB @ M - GA
    eps = 0.5 * np.linalg.solve(GA, dG)
    eps = 0.5 * (eps + eps.T)
    delta_cell = tuple(
        float(x - y)
        for x, y in zip(params_from_metric((GA + dG).tolist()), cell_A)
    )
    return {
        "d": max(d_ab, d_ba),
        "d_AB": d_ab,
        "d_BA": d_ba,
        "M_AB": [[int(x) for x in row] for row in M],
        "DeltaG": dG.tolist(),
        "Delta_cell": delta_cell,
        "strain": eps.tolist(),
        "principal_strains": [float(x) for x in np.linalg.eigvalsh(eps)],
        "strain_norm": _frob(eps),
        "certified": bool(cert_ab and cert_ba),
        "radius_bound": max(R_ab, R_ba),
        "radius_used": r_used,
    }
