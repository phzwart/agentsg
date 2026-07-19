"""Conventional-to-primitive cell reduction for centred lattices.

Kurlin's root invariant is an invariant of the *lattice* -- the full group of
translations. A deposited unit cell is the *conventional* cell, whose corner
lattice for a centred Bravais type (A, B, C, I, F, R) is only a sublattice of
the true crystal lattice: it omits the centring nodes. Selling-reducing the
conventional basis therefore describes the wrong lattice, and its root invariant
is wrong (by tens of Angstrom in root-product units for common centred groups).

This module supplies the standard primitive basis transformations P (columns =
primitive basis vectors expressed in the conventional basis; International Tables
for Crystallography Vol. A, Table 5.1.3.1), with ``det(P) = 1/m`` where ``m`` is
the centring multiplicity. The primitive metric tensor is ``G_P = Pᵀ G P``.

The matrices are validated at import against the centring translation vectors in
:mod:`agentsg.hall` (``LATTICE_CENTERING``): every centring vector must be an
integer combination of the primitive columns (i.e. ``P⁻¹ · t`` is integral).
Nothing here is a transcription an oracle must be trusted for -- it is checked
against the package's own centring table.
"""
from __future__ import annotations
import math

_H = 0.5
_T = 1.0 / 3.0
_TT = 2.0 / 3.0

# columns = primitive basis vectors in conventional coordinates
_PRIM_P = {
    "P": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "A": [[1, 0, 0], [0, _H, -_H], [0, _H, _H]],
    "B": [[_H, 0, -_H], [0, 1, 0], [_H, 0, _H]],
    "C": [[_H, _H, 0], [-_H, _H, 0], [0, 0, 1]],
    "I": [[-_H, _H, _H], [_H, -_H, _H], [_H, _H, -_H]],
    "F": [[0, _H, _H], [_H, 0, _H], [_H, _H, 0]],
    # rhombohedral, obverse setting on hexagonal axes (R and its "H" alias):
    "R": [[_TT, -_T, -_T], [_T, _T, -_TT], [_T, _T, _T]],
}
# "H" is the hexagonal-axes label used for R-centred groups in the PDB.
_LETTER_ALIAS = {"H": "R"}

CENTRING_MULTIPLICITY = {"P": 1, "A": 2, "B": 2, "C": 2,
                         "I": 2, "F": 4, "R": 3, "H": 3}


def lattice_letter(symbol):
    """Extract the Bravais lattice letter from an H-M or Hall symbol.

    Returns one of P,A,B,C,I,F,R (H is normalised to R). Raises ValueError on
    an unrecognised leading character.
    """
    s = symbol.strip().lstrip("-")
    if not s:
        raise ValueError(f"empty space-group symbol: {symbol!r}")
    L = s[0].upper()
    L = _LETTER_ALIAS.get(L, L)
    if L not in _PRIM_P:
        raise ValueError(f"unrecognised lattice letter {L!r} in {symbol!r}")
    return L


def _metric(cell):
    a, b, c, al, be, ga = cell
    al, be, ga = math.radians(al), math.radians(be), math.radians(ga)
    return [[a * a, a * b * math.cos(ga), a * c * math.cos(be)],
            [a * b * math.cos(ga), b * b, b * c * math.cos(al)],
            [a * c * math.cos(be), b * c * math.cos(al), c * c]]


def _cell_from_metric(G):
    a = math.sqrt(G[0][0]); b = math.sqrt(G[1][1]); c = math.sqrt(G[2][2])

    def ang(x):
        return math.degrees(math.acos(max(-1.0, min(1.0, x))))
    al = ang(G[1][2] / (b * c))
    be = ang(G[0][2] / (a * c))
    ga = ang(G[0][1] / (a * b))
    return (a, b, c, al, be, ga)


def _matT_G_M(P, G):
    # Pᵀ G P for 3x3 lists
    PT = [[P[j][i] for j in range(3)] for i in range(3)]
    GP = [[sum(G[i][k] * P[k][j] for k in range(3)) for j in range(3)]
          for i in range(3)]
    return [[sum(PT[i][k] * GP[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def primitive_transform(symbol):
    """Return the primitive basis matrix P (3x3 list) for a space-group symbol.

    Columns of P are the primitive basis vectors in conventional coordinates;
    det(P) = 1/multiplicity.
    """
    return [row[:] for row in _PRIM_P[lattice_letter(symbol)]]


def primitive_cell(cell, symbol):
    """Reduce a conventional cell to its primitive cell for the given symbol.

    Parameters
    ----------
    cell : (a,b,c,alpha,beta,gamma)
        The conventional (deposited) unit cell.
    symbol : str
        Space-group H-M or Hall symbol; only the lattice letter is used.

    Returns
    -------
    (a,b,c,alpha,beta,gamma)
        The primitive cell. For a primitive lattice (P, and H/R already handled)
        this is the input up to numerical round-off.
    """
    P = primitive_transform(symbol)
    Gp = _matT_G_M(P, _metric(cell))
    return _cell_from_metric(Gp)


def _validate():
    """Check each P against the centring vectors in hall.LATTICE_CENTERING."""
    from ..hall import LATTICE_CENTERING

    def inv3(M):
        det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
               - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
               + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
        cof = [[(M[(i + 1) % 3][(j + 1) % 3] * M[(i + 2) % 3][(j + 2) % 3]
                 - M[(i + 1) % 3][(j + 2) % 3] * M[(i + 2) % 3][(j + 1) % 3])
                for j in range(3)] for i in range(3)]
        return [[cof[j][i] / det for j in range(3)] for i in range(3)]

    for letter, P in _PRIM_P.items():
        if letter not in LATTICE_CENTERING:
            continue
        Pinv = inv3(P)
        for t in LATTICE_CENTERING[letter]:
            # P^{-1} t must be integral (t is a lattice node of the primitive cell)
            for i in range(3):
                v = sum(Pinv[i][j] * t[j] for j in range(3))
                if abs(v - round(v)) > 1e-9:
                    raise AssertionError(
                        f"primitive P for {letter} fails on centring {t}")
    return True


_validate()
