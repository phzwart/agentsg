"""
Unit-cell comparison ("exploring metric symmetry").

Implements the cell-comparison algorithm of Zwart, Grosse-Kunstleve & Adams,
"Exploring Metric Symmetry" (2006), sec. 2.4 / 3.2. Given two unit cells, decide
whether one is (approximately) a sublattice of the other and report the exact
integer transform relating them.

Algorithm (the "lego / target" construction):

  1. Niggli-reduce both cells. The smaller-volume reduced cell is the *lego*
     building block; the larger is the *target*.
  2. The integer volume ratio r = round(V_target / V_lego) is the sublattice
     index to search. (If the ratio is not near-integer within tolerance there
     is no exact sublattice relation.)
  3. For every index-r sublattice matrix M of the lego (Hermite normal form,
     agentsg.cell.sublattice), transform the lego metric by G' = M^T G_lego M and
     Niggli-reduce the result.
  4. Accept M as a solution when the reduced transformed cell matches the target
     Niggli cell within the length tolerance (percent) and angle tolerance
     (degrees). Niggli reduction already resolves the permutation/sign ambiguity,
     so a direct parameter comparison suffices; the Niggli change-of-basis of
     step 3 is recorded as the "additional transform".

Each solution reports the sublattice matrix M, the additional Niggli transform,
the resulting cell, and the per-parameter deviations (percent on lengths,
degrees on angles) exactly as in the paper's output.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import acos, degrees, sqrt

from .metric import UnitCell
from .reduction import niggli_reduce
from .sublattice import generate_sublattices


@dataclass(frozen=True)
class CellMatch:
    """One solution of a cell comparison."""
    index: int                       # sublattice index (volume factor)
    M: tuple                         # 3x3 integer sublattice matrix (lego -> supercell)
    niggli_transform: tuple          # additional Niggli change-of-basis (integer 3x3)
    resulting_cell: tuple            # (a,b,c,alpha,beta,gamma) after transform+reduce
    deviations: tuple                # (da%,db%,dc%, dalpha,dbeta,dgamma) vs target
    max_length_dev: float            # max |percent| on lengths
    max_angle_dev: float             # max |deg| on angles

    def __repr__(self):
        c = tuple(round(x, 1) for x in self.resulting_cell)
        return (f"CellMatch(index={self.index}, M={self.M}, cell={c}, "
                f"max_len_dev={self.max_length_dev:.2f}%, "
                f"max_ang_dev={self.max_angle_dev:.2f}deg)")


def _metric(cell):
    return UnitCell(*cell).metric_tensor()


def _params_from_G(G):
    a = sqrt(G[0][0]); b = sqrt(G[1][1]); c = sqrt(G[2][2])
    def ang(x): return degrees(acos(max(-1.0, min(1.0, x))))
    return (a, b, c, ang(G[1][2] / (b * c)), ang(G[0][2] / (a * c)), ang(G[0][1] / (a * b)))


def _apply_M(G, M):
    MtG = [[sum(M[k][i] * G[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return [[sum(MtG[i][k] * M[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _deviations(cell, target):
    dl = tuple((cell[i] - target[i]) / target[i] * 100.0 for i in range(3))
    da = tuple(cell[3 + i] - target[3 + i] for i in range(3))
    return dl + da


def compare_cells(cell_A, cell_B, length_tol_pct: float = 3.0,
                  angle_tol_deg: float = 5.0, max_index: int | None = None):
    """Compare two unit cells and return all sublattice relations between them.

    Parameters
    ----------
    cell_A, cell_B : (a, b, c, alpha, beta, gamma), angles in degrees.
    length_tol_pct : tolerance on edge-length deviation, in percent.
    angle_tol_deg  : tolerance on angle deviation, in degrees.
    max_index      : optional cap on the sublattice index searched; defaults to
        the rounded volume ratio.

    Returns a dict with the reduced lego and target cells, the volume ratio, and
    a list of :class:`CellMatch` solutions (sorted by combined deviation).
    """
    rA, _ = niggli_reduce(*cell_A)
    rB, _ = niggli_reduce(*cell_B)
    VA = UnitCell(*rA).volume()
    VB = UnitCell(*rB).volume()
    if VA <= VB:
        lego, target = rA, rB
    else:
        lego, target = rB, rA
    Vlego = UnitCell(*lego).volume()
    Vtarget = UnitCell(*target).volume()
    ratio = Vtarget / Vlego
    r = int(round(ratio))
    solutions = []
    if r >= 1 and abs(ratio - r) <= 0.05 * r + 1e-6:
        Glego = _metric(lego)
        indices = range(1, (max_index or r) + 1)
        for idx in indices:
            for M in generate_sublattices(idx):
                Gt = _apply_M(Glego, M)
                enlarged = _params_from_G(Gt)
                reduced, T = niggli_reduce(*enlarged)
                dev = _deviations(reduced, target)
                mlen = max(abs(d) for d in dev[:3])
                mang = max(abs(d) for d in dev[3:])
                if mlen <= length_tol_pct and mang <= angle_tol_deg:
                    solutions.append(CellMatch(
                        index=idx,
                        M=tuple(tuple(row) for row in M),
                        niggli_transform=tuple(tuple(row) for row in T),
                        resulting_cell=tuple(reduced),
                        deviations=tuple(dev),
                        max_length_dev=mlen,
                        max_angle_dev=mang,
                    ))
    solutions.sort(key=lambda s: s.max_length_dev + s.max_angle_dev)
    return {
        "lego_cell": tuple(lego),
        "target_cell": tuple(target),
        "volume_ratio": ratio,
        "index": r,
        "solutions": solutions,
    }
