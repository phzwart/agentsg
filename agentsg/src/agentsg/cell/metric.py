"""
Unit-cell metric math (phase 2) -- real-valued, floating point.

This module is deliberately numeric: a unit cell is a measured object with real
(a, b, c, alpha, beta, gamma), unlike the exact-rational symmetry algebra in the
rest of the package. It does NOT import the symmetry modules; the single
sanctioned crossing of the exact/numeric boundary lives in
``agentsg.cell.constraints`` (see docs/DESIGN.md).

Conventions:
  * angles in degrees on input, radians internally;
  * metric tensor G_ij = a_i . a_j, so for a column vector of fractional
    coordinates x, the squared length is x^T G x;
  * orthogonalisation matrix M (Cartesian = M @ fractional) uses the standard
    crystallographic convention with a along x and b in the xy-plane.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import cos, sin, sqrt, radians, degrees, acos, asin, pi
from typing import Sequence

Vec = Sequence[float]


def _mat_vec(M, v):
    return [sum(M[i][j] * v[j] for j in range(3)) for i in range(3)]


def _mat_mat(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]


def _det3(M):
    (a, b, c), (d, e, f), (g, h, i) = M
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _inv3(M):
    det = _det3(M)
    if abs(det) < 1e-300:
        raise ValueError("singular matrix")
    (a, b, c), (d, e, f), (g, h, i) = M
    cof = [
        [(e * i - f * h), -(d * i - f * g), (d * h - e * g)],
        [-(b * i - c * h), (a * i - c * g), -(a * h - b * g)],
        [(b * f - c * e), -(a * f - c * d), (a * e - b * d)],
    ]
    adj = _transpose(cof)
    return [[adj[i][j] / det for j in range(3)] for i in range(3)]


def metric_tensor(cell: Sequence[float]) -> list[list[float]]:
    """Build the direct-space metric tensor G from cell parameters.

    ``cell`` is ``(a, b, c, alpha, beta, gamma)`` with angles in degrees.
    ``G_ij = a_i · a_j``.
    """
    a, b, c, alpha, beta, gamma = (float(x) for x in cell)
    ca, cb, cg = cos(radians(alpha)), cos(radians(beta)), cos(radians(gamma))
    return [
        [a * a, a * b * cg, a * c * cb],
        [a * b * cg, b * b, b * c * ca],
        [a * c * cb, b * c * ca, c * c],
    ]


def params_from_metric(G: Sequence[Sequence[float]]) -> tuple[float, float, float, float, float, float]:
    """Recover ``(a, b, c, alpha, beta, gamma)`` from a metric tensor G.

    Clamping policy (single package-wide rule for cell ↔ G conversion):
      * raise ``ValueError`` on non-positive edge lengths (``sqrt(G_ii)``);
      * clamp cosines to ``[-1, 1]`` before ``acos`` so float noise on near-
        orthogonal / near-degenerate angles does not raise.

    Prefer raising over silently zeroing invalid edges.
    """
    a = sqrt(max(G[0][0], 0.0))
    b = sqrt(max(G[1][1], 0.0))
    c = sqrt(max(G[2][2], 0.0))
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("non-positive cell edge in metric")

    def ang(x: float) -> float:
        return degrees(acos(max(-1.0, min(1.0, x))))

    return (
        a, b, c,
        ang(G[1][2] / (b * c)),
        ang(G[0][2] / (a * c)),
        ang(G[0][1] / (a * b)),
    )


# Alias preferred by some call sites; identical to params_from_metric.
cell_from_metric = params_from_metric


@dataclass(frozen=True)
class UnitCell:
    """A crystallographic unit cell defined by its six parameters.

    Lengths in arbitrary (consistent) units, angles in degrees.
    """
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    # --- metric tensor and volume ---
    def metric_tensor(self) -> list[list[float]]:
        return metric_tensor((self.a, self.b, self.c, self.alpha, self.beta, self.gamma))

    def volume(self) -> float:
        a, b, c = self.a, self.b, self.c
        ca, cb, cg = cos(radians(self.alpha)), cos(radians(self.beta)), cos(radians(self.gamma))
        return a * b * c * sqrt(max(0.0, 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg))

    # --- reciprocal cell ---
    def reciprocal_metric_tensor(self) -> list[list[float]]:
        return _inv3(self.metric_tensor())

    def reciprocal(self) -> "UnitCell":
        """The reciprocal unit cell (its parameters). Reciprocal lengths are in
        1/length units; a*·a = 1 etc."""
        return UnitCell(*params_from_metric(self.reciprocal_metric_tensor()))

    # --- orthogonalisation (fractional -> Cartesian) ---
    def orthogonalization_matrix(self) -> list[list[float]]:
        """Matrix M with Cartesian = M @ fractional (a along x, b in xy-plane)."""
        a, b, c = self.a, self.b, self.c
        al, be, ga = radians(self.alpha), radians(self.beta), radians(self.gamma)
        ca, cb, cg, sg = cos(al), cos(be), cos(ga), sin(ga)
        V = self.volume()
        # standard PDB/crystallographic convention
        return [
            [a, b * cg, c * cb],
            [0.0, b * sg, c * (ca - cb * cg) / sg],
            [0.0, 0.0, V / (a * b * sg)],
        ]

    def fractionalization_matrix(self) -> list[list[float]]:
        """Matrix F with fractional = F @ Cartesian (inverse of orthogonalisation)."""
        return _inv3(self.orthogonalization_matrix())

    def orthogonalize(self, frac: Vec) -> list[float]:
        return _mat_vec(self.orthogonalization_matrix(), list(frac))

    def fractionalize(self, cart: Vec) -> list[float]:
        return _mat_vec(self.fractionalization_matrix(), list(cart))

    # --- d-spacings and reflection geometry ---
    def d_spacing(self, hkl: Sequence[int]) -> float:
        """Interplanar spacing d(hkl) = 1 / |h·a* + k·b* + l·c*|."""
        h, k, l = hkl
        Gs = self.reciprocal_metric_tensor()
        hv = [h, k, l]
        inv_d2 = sum(hv[i] * Gs[i][j] * hv[j] for i in range(3) for j in range(3))
        if inv_d2 <= 0:
            return float("inf")
        return 1.0 / sqrt(inv_d2)

    def d_star_sq(self, hkl: Sequence[int]) -> float:
        """1/d^2 for a reflection (the reciprocal-space squared length)."""
        h, k, l = hkl
        Gs = self.reciprocal_metric_tensor()
        hv = [h, k, l]
        return sum(hv[i] * Gs[i][j] * hv[j] for i in range(3) for j in range(3))

    def two_theta(self, hkl: Sequence[int], wavelength: float) -> float:
        """Bragg 2-theta (degrees) for a reflection at a given wavelength."""
        d = self.d_spacing(hkl)
        s = wavelength / (2 * d)
        if s > 1:
            raise ValueError("reflection beyond the limiting sphere for this wavelength")
        return degrees(2 * asin(s))

    def __repr__(self):
        return (f"UnitCell(a={self.a:g}, b={self.b:g}, c={self.c:g}, "
                f"alpha={self.alpha:g}, beta={self.beta:g}, gamma={self.gamma:g})")
