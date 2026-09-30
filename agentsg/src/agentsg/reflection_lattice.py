"""
Reflection conditions as sublattices in projection.

The absence rule is: ``h`` is systematically absent iff some operation
``(W, w)`` fixes ``h`` in reciprocal space (``h W = h``) with non-integral
phase ``h . w``. Everything in this module follows from reading that rule as a
statement about lattices rather than about integers on a grid.

**Strata.** Let ``P`` be the point group acting on reciprocal space by
``h -> h W``. Every ``h`` has a stabiliser ``S_h = {W : h W = h}``; the set of
``h`` with a given stabiliser is a *stratum*, and each stratum is the generic
part of a rational subspace ``V_S = {h : h W = h for all W in S}``. The strata
are exactly the reflection classes of International Tables: the trivial
stabiliser is ``hkl``, a mirror stabiliser is a zone (``0kl``, ``hhl``,
``h-hl`` ...), an axial stabiliser is a row (``h00``, ``00l``, ``hh0`` ...).
They are computed here as the closure under intersection of the fixed
subspaces of the individual ``W`` -- nothing is hard-coded, so hexagonal
``h-hl`` (ITA ``h-h0l``) and cubic ``hhh`` appear on their own.

**Condition on a stratum.** For ``h`` in ``V_S`` and ``W`` in ``S``,
``h . w`` depends only on ``w`` mod the lattice, so ``h`` is present iff it
annihilates the translation lattice

    Lambda_S = Z^3 + < w : (W, w) in G, W in S >.

Hence the present reflections of the stratum are ``V_S ∩ Z^3 ∩ Lambda_S^*``:
a sublattice of the saturated lattice ``L_V = V_S ∩ Z^3``. Writing ``h`` in a
basis of ``L_V`` (whose coefficients are the class letters -- ``k, l`` for
``0kl``, ``h, l`` for ``hhl``) each generator ``g`` of ``Lambda_S`` gives one
congruence ``(m g) . h ≡ 0 (mod m)``; a minimal independent set of those is
the printed condition. The index of the sublattice is the product of the
moduli.

The sublattice on a stratum is computed by one Smith normal form of the
projected generators (:func:`dual_lattice`) -- the same routine that
:mod:`agentsg.semi_invariants` applies to the allowed-origin group -- which
gives its invariant factors, index and a minimal congruence system directly.

Exact rational arithmetic throughout; no sampling, no bound on the modulus.
"""
from __future__ import annotations

from fractions import Fraction
from fractions import Fraction as Fr
from functools import reduce
from math import gcd, lcm
from typing import Iterable, Sequence

from .linalg import Vector3
from .symmetry_op import SymmetryOp

# --- small exact integer linear algebra ------------------------------------------


def _int_kernel(A: Sequence[Sequence[int]], n: int) -> list[list[int]]:
    """Basis of ``{x in Z^n : A x = 0}`` (saturated) by unimodular column
    reduction: find U with ``A U = [H | 0]``; the kernel is the trailing
    columns of U."""
    A = [list(map(int, row)) for row in A]
    m = len(A)
    U = [[int(i == j) for j in range(n)] for i in range(n)]
    col = 0
    for i in range(m):
        if col >= n:
            break
        while True:
            nz = [j for j in range(col, n) if A[i][j] != 0]
            if len(nz) <= 1:
                break
            j0 = min(nz, key=lambda j: abs(A[i][j]))
            for j in nz:
                if j == j0:
                    continue
                q = A[i][j] // A[i][j0]
                for r in range(m):
                    A[r][j] -= q * A[r][j0]
                for r in range(n):
                    U[r][j] -= q * U[r][j0]
        nz = [j for j in range(col, n) if A[i][j] != 0]
        if nz:
            j0 = nz[0]
            for r in range(m):
                A[r][j0], A[r][col] = A[r][col], A[r][j0]
            for r in range(n):
                U[r][j0], U[r][col] = U[r][col], U[r][j0]
            col += 1
    return [[U[r][j] for r in range(n)] for j in range(col, n)]


def _row_hnf(rows: Sequence[Sequence[int]]) -> list[list[int]]:
    """Row Hermite normal form (pivots positive, entries above pivots reduced
    into ``[0, pivot)``); zero rows dropped. Canonical for a lattice."""
    M = [list(map(int, r)) for r in rows if any(r)]
    if not M:
        return []
    n = len(M[0])
    r = 0
    for c in range(n):
        while True:
            nz = [i for i in range(r, len(M)) if M[i][c] != 0]
            if len(nz) <= 1:
                break
            i0 = min(nz, key=lambda i: abs(M[i][c]))
            for i in nz:
                if i == i0:
                    continue
                q = M[i][c] // M[i0][c]
                M[i] = [a - q * b for a, b in zip(M[i], M[i0])]
        nz = [i for i in range(r, len(M)) if M[i][c] != 0]
        if not nz:
            continue
        i0 = nz[0]
        M[r], M[i0] = M[i0], M[r]
        if M[r][c] < 0:
            M[r] = [-a for a in M[r]]
        p = M[r][c]
        for i in range(r):
            q = M[i][c] // p
            M[i] = [a - q * b for a, b in zip(M[i], M[r])]
        r += 1
        if r == len(M):
            break
    return [row for row in M[:r] if any(row)]


def _det(M: Sequence[Sequence[int]]) -> int:
    """Integer determinant by Fraction Gaussian elimination (tiny matrices)."""
    n = len(M)
    A = [[Fraction(x) for x in row] for row in M]
    d = Fraction(1)
    for c in range(n):
        p = next((i for i in range(c, n) if A[i][c] != 0), None)
        if p is None:
            return 0
        if p != c:
            A[c], A[p] = A[p], A[c]
            d = -d
        d *= A[c][c]
        for i in range(c + 1, n):
            f = A[i][c] / A[c][c]
            if f:
                A[i] = [a - f * b for a, b in zip(A[i], A[c])]
    return int(d)


# --- exact integer Smith normal form ------------------------------------------------


def smith_normal_form(
    A: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int], list[list[int]]]:
    """Smith normal form ``U A V = S`` of an integer matrix.

    Returns ``(U, s, V)`` with ``U`` (n×n) and ``V`` (m×m) unimodular and
    ``s`` the diagonal of ``S`` (non-negative, each dividing the next, zeros
    last). Elementary row operations are accumulated in ``U`` and column
    operations in ``V``; the divisibility chain is enforced so the ``s_i``
    are the invariant factors.
    """
    S = [list(map(int, r)) for r in A]
    n = len(S)
    m = len(S[0]) if n else 0
    U = [[int(i == j) for j in range(n)] for i in range(n)]
    V = [[int(i == j) for j in range(m)] for i in range(m)]

    def swap_rows(i: int, j: int) -> None:
        S[i], S[j] = S[j], S[i]
        U[i], U[j] = U[j], U[i]

    def sub_row(i: int, j: int, q: int) -> None:  # row_i -= q row_j
        S[i] = [a - q * b for a, b in zip(S[i], S[j])]
        U[i] = [a - q * b for a, b in zip(U[i], U[j])]

    def swap_cols(i: int, j: int) -> None:
        for r in S:
            r[i], r[j] = r[j], r[i]
        for r in V:
            r[i], r[j] = r[j], r[i]

    def sub_col(i: int, j: int, q: int) -> None:  # col_i -= q col_j
        for r in S:
            r[i] -= q * r[j]
        for r in V:
            r[i] -= q * r[j]

    t = 0
    while t < min(n, m):
        cand = [(abs(S[i][j]), i, j) for i in range(t, n) for j in range(t, m) if S[i][j]]
        if not cand:
            break
        _, i, j = min(cand)
        swap_rows(t, i)
        swap_cols(t, j)
        while True:
            # Clear column t below the pivot and row t right of it; if a
            # remainder is smaller than the pivot it becomes the new pivot.
            moved = False
            for i in range(t + 1, n):
                if S[i][t]:
                    sub_row(i, t, S[i][t] // S[t][t])
                    if S[i][t]:
                        swap_rows(t, i)
                        moved = True
            for j in range(t + 1, m):
                if S[t][j]:
                    sub_col(j, t, S[t][j] // S[t][t])
                    if S[t][j]:
                        swap_cols(t, j)
                        moved = True
            if moved:
                continue
            bad = next(
                ((i, j) for i in range(t + 1, n) for j in range(t + 1, m)
                 if S[i][j] % S[t][t]),
                None,
            )
            if bad is None:
                break
            sub_row(t, bad[0], -1)  # pull the offending row into the pivot row
        if S[t][t] < 0:
            S[t] = [-a for a in S[t]]
            U[t] = [-a for a in U[t]]
        t += 1
    s = [S[i][i] if i < m else 0 for i in range(min(n, m))]
    return U, s, V


def _unimodular_inverse(M: Sequence[Sequence[int]]) -> list[list[int]]:
    """Exact integer inverse of a unimodular matrix (via Fraction elimination)."""
    n = len(M)
    A = [[Fr(x) for x in row] + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = next(i for i in range(c, n) if A[i][c] != 0)
        A[c], A[p] = A[p], A[c]
        piv = A[c][c]
        A[c] = [x / piv for x in A[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
    inv = [[A[i][n + j] for j in range(n)] for i in range(n)]
    assert all(x.denominator == 1 for row in inv for x in row)
    return [[int(x) for x in row] for row in inv]



def _dot(a: Sequence, b: Sequence) -> Fraction:
    """Exact rational dot product of two sequences."""
    return sum((Fraction(x) * Fraction(y) for x, y in zip(a, b)), Fraction(0))


# --- strata ---------------------------------------------------------------------


def _W_rows(op: SymmetryOp) -> tuple[tuple[int, ...], ...]:
    """Integer rows of a SymmetryOp rotation part."""
    return tuple(tuple(int(x) for x in row) for row in op.W.rows)


def _fixed_lattice(W: Sequence[Sequence[int]]) -> list[list[int]]:
    """Saturated basis of ``{h in Z^3 : h W = h}`` (row-vector action)."""
    # h W = h  <=>  (W - I)^T h^T = 0
    A = [[W[j][i] - (1 if i == j else 0) for j in range(3)] for i in range(3)]
    return _row_hnf(_int_kernel(A, 3))


def _annihilator(basis: Sequence[Sequence[int]]) -> list[list[int]]:
    """Integer vectors y with b . y = 0 for every basis row b."""
    if not basis:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return _int_kernel(basis, 3)


def _intersect(b1, b2) -> list[list[int]]:
    """Saturated basis of ``span(b1) ∩ span(b2) ∩ Z^3``."""
    cons = _annihilator(b1) + _annihilator(b2)
    if not cons:
        return _row_hnf(b1)
    return _row_hnf(_int_kernel(cons, 3))


def _hkey(basis) -> tuple:
    """Canonical row-HNF tuple key for a basis."""
    return tuple(tuple(r) for r in _row_hnf(basis))


def _transform_basis(basis, W):
    """Image of the lattice under ``h -> h W``."""
    return _row_hnf([[sum(b[i] * W[i][j] for i in range(3)) for j in range(3)]
                     for b in basis])


def strata(operations: Iterable[SymmetryOp]) -> list[dict]:
    """The reflection strata of a space group.

    Returns one dict per stratum (dimension 1..3; the origin is not a
    reflection) with keys ``basis`` (saturated integer basis of ``L_V``, HNF),
    ``dim``, ``stabiliser`` (the ``W`` rows fixing every ``h`` of ``V``), and
    ``orbit`` (the HNF keys of the point-group images of ``V``).
    """
    ops = list(operations)
    P = {}
    for op in ops:
        P.setdefault(_W_rows(op), op)
    Ws = list(P)
    seen: dict[tuple, list[list[int]]] = {}
    full = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    seen[_hkey(full)] = full
    for W in Ws:
        b = _fixed_lattice(W)
        if b:
            seen.setdefault(_hkey(b), b)
    # closure under intersection
    changed = True
    while changed:
        changed = False
        keys = list(seen)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                b = _intersect(seen[keys[i]], seen[keys[j]])
                if b and _hkey(b) not in seen:
                    seen[_hkey(b)] = b
                    changed = True
    out = []
    for key, basis in seen.items():
        stab = [W for W in Ws
                if all(all(sum(b[i] * W[i][j] for i in range(3)) == b[j]
                           for j in range(3)) for b in basis)]
        orbit = {_hkey(_transform_basis(basis, W)) for W in Ws}
        out.append({"basis": basis, "dim": len(basis), "stabiliser": stab,
                    "orbit": orbit, "key": key})
    return out


# --- the sublattice of present reflections on a stratum ----------------------------


def _canon_congruence(f: Sequence[int], m: int):
    """Canonical ``(f, m)`` for ``f . c ≡ 0 (mod m)``: content removed,
    coefficients reduced into ``(-m/2, m/2]``, sign chosen ITA-style (prefer
    more positive coefficients, then first nonzero positive). None if trivial."""
    f = [int(x) for x in f]
    g = reduce(gcd, f + [m])
    if g > 1:
        f = [x // g for x in f]
        m //= g
    if m <= 1:
        return None

    def reduce_mod(v):
        """Reduce integer components into (-m/2, m/2]."""
        out = []
        for x in v:
            r = x % m
            if 2 * r > m:
                r -= m
            out.append(r)
        return out

    f = reduce_mod(f)
    if not any(f):
        return None
    neg = reduce_mod([-x for x in f])

    def rank(v):
        """Tie-breaker score preferring positive terms in canonical ITA order."""
        # ITA: -h+k+l rather than h-k-l (more positive terms); -h+k rather
        # than h-k (the later letters stay positive)
        return (sum(v), [1 if x > 0 else (-1 if x < 0 else 0) for x in reversed(v)])

    f = max([f, neg], key=rank)
    return tuple(f), m


def dual_lattice(vectors: Sequence[Sequence[Fraction]], d: int) -> dict:
    """The sublattice ``{c in Z^d : c . v in Z for every v}`` of rational vectors.

    One Smith normal form does everything: with ``D`` the common denominator
    and ``G`` the integer ``d x k`` matrix of columns ``D v``, ``c . v ∈ Z``
    for all ``v`` reads ``cᵀ G ≡ 0 (mod D)``. Writing ``U G V = S`` and
    ``z = U⁻ᵀ c`` this is ``z_i ≡ 0 (mod m_i)`` with ``m_i = D / gcd(D, s_i)``.

    Returns ``congruences`` (canonical ``(f, m)`` with ``f`` = column ``i`` of
    ``U⁻¹``, one per non-trivial invariant factor -- a minimal system),
    ``invariant_factors`` (the ``m_i > 1``, ascending), ``index`` (their product) and
    ``basis`` (row-HNF basis of the sublattice, rows ``m_i U[i]``).
    """
    vecs = [[Fraction(x) for x in v] for v in vectors]
    D = reduce(lcm, (x.denominator for v in vecs for x in v), 1)
    if not vecs or D == 1:
        return {"congruences": [], "invariant_factors": [], "index": 1,
                "basis": [[int(i == j) for j in range(d)] for i in range(d)]}
    G = [[int(v[i] * D) for v in vecs] for i in range(d)]
    U, s, V = smith_normal_form(G)
    s = list(s) + [0] * (d - len(s))
    Uinv = _unimodular_inverse(U)
    congs, factors, rows = [], [], []
    for i in range(d):
        m = D // gcd(D, s[i])
        rows.append([m * x for x in U[i]])
        if m > 1:
            factors.append(m)
            c = _canon_congruence([Uinv[j][i] for j in range(d)], m)
            if c is not None:
                congs.append(c)
    index = reduce(lambda a, b: a * b, factors, 1)
    return {"congruences": congs, "invariant_factors": sorted(factors),
            "index": index, "basis": _row_hnf(rows)}


def _congruence_index(congs: Sequence[tuple[tuple[int, ...], int]], d: int) -> int:
    """Index of ``{c : f . c ≡ 0 (mod m) for all (f, m)}`` -- each congruence is
    the rational vector ``f / m`` in :func:`dual_lattice`."""
    return dual_lattice([[Fraction(x, m) for x in f] for f, m in congs], d)["index"]


def stratum_lattice(stratum: dict, operations: Sequence[SymmetryOp]) -> dict:
    """The sublattice of present reflections on one stratum, from one Smith
    normal form of the translation generators of ``Lambda_S`` projected onto
    the class coefficients (see :func:`dual_lattice`)."""
    basis = stratum["basis"]
    stab = set(stratum["stabiliser"])
    gens = [tuple(Fraction(x) for x in op.w.v) for op in operations
            if _W_rows(op) in stab]
    return dual_lattice([[_dot(b, g) for b in basis] for g in gens], len(basis))


def stratum_conditions(stratum: dict, operations: Sequence[SymmetryOp]):
    """Minimal congruences on the class coefficients for one stratum.

    Returns ``(congruences, index)``: a list of canonical ``(f, m)`` meaning
    ``f . c ≡ 0 (mod m)`` for ``h = c . basis``, and the index of the
    sublattice of present reflections in ``L_V``.

    The lattice itself (and its index) comes from :func:`stratum_lattice`.
    The congruences reported are the ITA-style ones read off the individual
    screw/glide/centring generators, pruned to a minimal subset that still
    generates exactly that lattice -- so ``0kl: l = 2n; k+l = 4n`` rather than
    whatever basis the Smith form happens to pick.
    """
    basis = stratum["basis"]
    d = len(basis)
    lat = stratum_lattice(stratum, operations)
    target = lat["index"]
    stab = set(stratum["stabiliser"])
    gens = [tuple(Fraction(x) for x in op.w.v) for op in operations
            if _W_rows(op) in stab]
    congs = set()
    for g in gens:
        v = [_dot(b, g) for b in basis]           # coefficient of each letter
        m = reduce(lcm, (x.denominator for x in v), 1)
        if m == 1:
            continue
        f = [int(x * m) for x in v]
        c = _canon_congruence(f, m)
        if c is not None:
            congs.add(c)
    # merge identical forms: l = 2n and l = 3n -> l = 6n
    by_form: dict[tuple, int] = {}
    for f, m in congs:
        by_form[f] = lcm(by_form.get(f, 1), m)
    congs = [_canon_congruence(f, m) for f, m in by_form.items()]
    congs = [c for c in congs if c is not None]
    assert _congruence_index(congs, d) == target
    minimal = list(congs)
    # try to remove the most complex forms first (largest coefficients, highest
    # modulus, then forms with negative terms), so what survives is the
    # simplest equivalent statement -- k+l = 4n rather than -k+l = 4n
    for c in sorted(congs, key=lambda t: (-sum(abs(x) for x in t[0]), -t[1],
                                          sum(t[0]))):
        trial = [g for g in minimal if g != c]
        if _congruence_index(trial, d) == target:
            minimal = trial
    return sorted(minimal, key=lambda t: (t[1], t[0])), target


# --- naming and formatting ------------------------------------------------------------


_LETTERS = "hkl"


def _letters(basis) -> list[str]:
    """Class letter for each basis row: the letter of its first nonzero index
    (``0kl`` -> k, l; ``hhl`` -> h, l; ``h00`` -> h)."""
    return [_LETTERS[next(i for i, x in enumerate(b) if x)] for b in basis]


def _term(coef: int, sym: str, first: bool) -> str:
    """Format one algebraic term with sign and magnitude (e.g. '+2k', '-h')."""
    if coef == 0:
        return ""
    mag = "" if abs(coef) == 1 else str(abs(coef))
    if first:
        return ("-" if coef < 0 else "") + mag + sym
    return ("-" if coef < 0 else "+") + mag + sym


def class_name(basis) -> str:
    """ITA-style class name from the saturated basis of the stratum."""
    d = len(basis)
    if d == 3:
        return "hkl"
    # express each of h,k,l as a combination of the class letters
    letters = _letters(basis)
    cols = []
    for j in range(3):
        parts = ""
        for b, L in zip(basis, letters):
            parts += _term(b[j], L, first=(parts == ""))
        cols.append(parts or "0")
    return "".join(c if len(c) == 1 or c.startswith("-") and len(c) == 2
                   else f"({c})" for c in cols)


def crystal_family(operations: Iterable[SymmetryOp]) -> tuple[str, int]:
    """Lattice family and principal-axis index derived from the point group.

    Returns ``(family, axis)`` with family in {'triclinic', 'monoclinic',
    'orthorhombic', 'tetragonal', 'hexagonal', 'cubic'} and ``axis`` the index
    (0=a, 1=b, 2=c) of the unique / principal axis (2 when not applicable).
    Decided from the rotation orders and the directions of their fixed
    lattices -- no symbol parsing.
    """
    Ws = list({_W_rows(op) for op in operations})
    axes = {}          # (order, axis-direction key) -> present
    for W in Ws:
        # proper part
        det = _det(W)
        M = W if det > 0 else tuple(tuple(-x for x in r) for r in W)
        n = _matrix_order(M)
        if n < 2:
            continue
        fix = _fixed_lattice(M)
        if len(fix) != 1:
            continue
        axes.setdefault(tuple(fix[0]), 0)
        axes[tuple(fix[0])] = max(axes[tuple(fix[0])], n)
    unit = {(1, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 2}
    if any(n == 3 and d not in unit for d, n in axes.items()) and \
            sum(1 for n in axes.values() if n == 3) >= 4:
        return "cubic", 2
    for d, n in axes.items():
        if n in (3, 6) and d in unit:
            return "hexagonal", unit[d]
    for d, n in axes.items():
        if n == 4 and d in unit:
            return "tetragonal", unit[d]
    two = [d for d, n in axes.items() if n == 2 and d in unit]
    if len(two) >= 3:
        return "orthorhombic", 2
    if len(two) == 1:
        return "monoclinic", unit[two[0]]
    return "triclinic", 2


def _matrix_order(W, max_n=6) -> int:
    """Order of integer rotation matrix W (smallest n>=1 with W^n = I)."""
    P = [[int(i == j) for j in range(3)] for i in range(3)]
    for n in range(1, max_n + 1):
        P = [[sum(P[i][k] * W[k][j] for k in range(3)) for j in range(3)]
             for i in range(3)]
        if all(P[i][j] == (1 if i == j else 0) for i in range(3) for j in range(3)):
            return n
    return 0


def _std_classes(family: str, axis: int) -> list[list[list[int]]]:
    """The class bases ITA tabulates for a lattice family (in the standard
    orientation with the principal axis along ``axis``)."""
    e = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    I3 = [e[0], e[1], e[2]]
    p = axis
    q, r = [i for i in range(3) if i != p]      # the two other axes, in order
    if family == "triclinic":
        return [I3]
    if family in ("monoclinic", "orthorhombic"):
        return [I3, [e[1], e[2]], [e[0], e[2]], [e[0], e[1]], [e[0]], [e[1]], [e[2]]]
    diag = [1 if i in (q, r) else 0 for i in range(3)]          # a+b
    anti = [1 if i == q else (-1 if i == r else 0) for i in range(3)]  # a-b
    if family == "tetragonal":
        return [I3, [e[q], e[r]], [e[r], e[p]], [diag, e[p]], [e[p]], [e[q]], [diag]]
    if family == "hexagonal":
        return [I3, [e[q], e[r]], [anti, e[p]], [diag, e[p]], [e[p]], [anti], [diag]]
    return [I3, [e[1], e[2]], [[1, 1, 0], e[2]], [e[0]]]            # cubic


_NAME_RANK = {"hkl": 0,
              "0kl": 10, "h0l": 11, "hk0": 12, "hhl": 13, "h-hl": 14,
              "h00": 20, "0k0": 21, "00l": 22, "hh0": 23, "h-h0": 24,
              "hhh": 25}


def _name_rank(name: str, hexagonal: bool = False) -> tuple:
    """Sort key prioritizing standard ITA reflection-class ordering."""
    r = _NAME_RANK.get(name, 50 + len(name))
    if hexagonal and name in ("h-hl", "h-h0"):
        r -= 5          # ITA names the h-h0l / h-h00 zones first in hexagonal groups
    return (r, name)


def format_conditions(congs, letters) -> str:
    """Format a list of ((c1, c2, ...), m) congruences as an ITA condition string."""
    by_mod: dict[int, list[str]] = {}
    # ITA lists single-letter forms in h, k, l order: "h, k = 2n"
    order = {L: i for i, L in enumerate(_LETTERS)}
    congs = sorted(congs, key=lambda t: (t[1], -sum(1 for x in t[0] if x),
                                         [order[L] for c, L in zip(t[0], letters) if c]))
    for f, m in congs:
        expr = ""
        for coef, L in zip(f, letters):
            expr += _term(coef, L, first=(expr == ""))
        by_mod.setdefault(m, []).append(expr)
    return "; ".join(", ".join(forms) + f" = {m}n" for m, forms in sorted(by_mod.items()))


def _stabiliser(basis, Ws):
    """Subgroup of point-group matrices that fix every vector in the stratum."""
    return [W for W in Ws
            if all(all(sum(b[i] * W[i][j] for i in range(3)) == b[j]
                       for j in range(3)) for b in basis)]


def reflection_conditions(operations: Iterable[SymmetryOp],
                          ita_classes: bool = True) -> dict[str, str]:
    """ITA-style general reflection conditions, derived as sublattices.

    Every stratum (reflection class with its own stabiliser) that carries a
    non-trivial condition is reported once per point-group orbit, under the
    representative ITA names (``0kl`` rather than ``h0l`` in a tetragonal
    group, ``h-hl`` -- ITA ``h-h0l`` -- in a hexagonal one).

    With ``ita_classes`` (default) the classes that International Tables
    tabulate for the lattice family are reported as well even when they are
    not strata of their own -- e.g. ``hk0: -h+k = 3n`` for an R lattice,
    which is just the integral condition restricted to ``l = 0``. Those are
    the same sublattice computation with the parent stratum's stabiliser.
    """
    ops = list(operations)
    Ws = list({_W_rows(op) for op in ops})
    family, axis = crystal_family(ops)
    hexa = family == "hexagonal"
    out: dict[str, str] = {}
    done: set[tuple] = set()
    reported: set[tuple] = set()
    for st in sorted(strata(ops), key=lambda s: -s["dim"]):
        if st["key"] in done:
            continue
        done |= st["orbit"]
        best = min((_transform_basis(st["basis"], W) for W in Ws),
                   key=lambda b: _name_rank(class_name(b), hexa))
        rep = {"basis": best, "stabiliser": _stabiliser(best, Ws)}
        congs, index = stratum_conditions(rep, ops)
        reported.add(_hkey(best))
        if index > 1:
            out[class_name(best)] = format_conditions(congs, _letters(best))
    if ita_classes:
        for basis in _std_classes(family, axis):
            basis = _row_hnf(basis)
            if _hkey(basis) in reported:
                continue
            reported.add(_hkey(basis))
            rep = {"basis": basis, "stabiliser": _stabiliser(basis, Ws)}
            congs, index = stratum_conditions(rep, ops)
            if index > 1:
                out[class_name(basis)] = format_conditions(congs, _letters(basis))
    return out


def present_lattices(operations: Iterable[SymmetryOp]) -> list[dict]:
    """Every stratum with its sublattice of present reflections -- the exact
    object behind :func:`reflection_conditions`, for checking against the
    per-reflection rule."""
    ops = list(operations)
    out = []
    for st in strata(ops):
        congs, index = stratum_conditions(st, ops)
        lat = stratum_lattice(st, ops)
        out.append({"basis": st["basis"], "stabiliser": st["stabiliser"],
                    "congruences": congs, "index": index,
                    "invariant_factors": lat["invariant_factors"],
                    "smith_congruences": lat["congruences"],
                    "sublattice": lat["basis"],
                    "name": class_name(st["basis"])})
    return out


def is_absent_by_lattice(hkl, lattices: Sequence[dict]) -> bool:
    """Absence of an integer ``hkl`` decided purely from the strata lattices:
    find the stratum whose stabiliser is exactly the stabiliser of ``hkl``
    (the smallest stratum containing it), express ``hkl`` in its basis and
    test the congruences."""
    h = [int(x) for x in (hkl.v if isinstance(hkl, Vector3) else hkl)]
    best = None
    for st in lattices:
        basis = st["basis"]
        # is h in span(basis)?  solve c . basis = h over Q
        cons = _annihilator(basis)
        if any(sum(h[i] * y[i] for i in range(3)) != 0 for y in cons):
            continue
        if best is None or len(basis) < len(best["basis"]):
            best = st
    basis = best["basis"]
    # coefficients c: basis is HNF with distinct pivots, solve by substitution
    d = len(basis)
    piv = [next(i for i, x in enumerate(b) if x) for b in basis]
    c = [Fraction(0)] * d
    rem = [Fraction(x) for x in h]
    for a in range(d):
        c[a] = rem[piv[a]] / basis[a][piv[a]]
        rem = [r - c[a] * bb for r, bb in zip(rem, basis[a])]
    assert all(r == 0 for r in rem) and all(x.denominator == 1 for x in c)
    congs = best.get("smith_congruences", best["congruences"])
    return any(sum(int(x) * fi for x, fi in zip(c, f)) % m != 0
               for f, m in congs)
