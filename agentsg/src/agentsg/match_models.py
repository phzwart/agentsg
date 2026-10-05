"""
Match two copies of the same atoms under the Euclidean normalizer.

The inputs are corresponding atoms, in the same order: fractional coordinates
in the conventional basis of ``sg``, or Cartesian coordinates when
``cartesian`` is set. No PDB parser is required for that path. A thin
``gemmi`` helper, imported only when called, pairs atoms by
``(chain, resseq, icode, name)`` and tries chain permutations.

**Search.** Candidates are the proper coset representatives of
:meth:`agentsg.normalizer.EuclideanNormalizer.proper_coset_reps` composed
with every operator of ``G``. Det −1 representatives are added only when
``allow_improper`` is set. Inversion (det −1) maps a chiral molecule to its
mirror image; ``enantiomorph_flag`` records when one of those would have
beaten the best proper match.

**Placement.** For a candidate ``(M, m) (W, w)``, transform B to ``x'_B`` and
take the mean of ``x_A − x'_B``. Each atom is reduced on the torus into
``(−1/2, 1/2]`` relative to the first atom before the mean, via
:func:`agentsg.asu._wrap_half`, so a molecule that straddles the cell
boundary does not split the average. The floating part of that mean is
removed with :func:`agentsg.semi_invariants.pin_floating_origin`. The
remainder is snapped to the nearest element of ``T' + L`` in the Smith
basis of the origin lattice: component ``i`` is rounded to a multiple of
``1/d_i``. A component whose residual reaches ``1/(2 d_i)`` sits on the
boundary between two allowed origins and is marked ``ambiguous_snap``.

The RMSD is the Cartesian deviation, in the basis of ``cell_a``, after the
torus mean translation has been removed.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import sqrt
from typing import Sequence

from .asu import _wrap_half
from .cell.metric import UnitCell
from .linalg import Matrix3, Vector3
from .normalizer import EuclideanNormalizer, euclidean_normalizer
from .semi_invariants import pin_floating_origin
from .symmetry_op import SymmetryOp


def _as_rows(xyz: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    rows = []
    for point in xyz:
        if len(point) != 3:
            raise ValueError(f"{name} must be a list of xyz triples")
        rows.append([float(point[0]), float(point[1]), float(point[2])])
    if not rows:
        raise ValueError(f"{name} must contain at least one atom")
    return rows


def _to_fractional(rows: list[list[float]], cell, cartesian: bool) -> list[list[float]]:
    if not cartesian:
        return rows
    uc = UnitCell(*cell)
    return [list(uc.fractionalize(row)) for row in rows]


def _apply(W: Matrix3, w: Vector3, rows: list[list[float]]) -> list[list[float]]:
    linear = [[float(x) for x in row] for row in W.rows]
    shift = [float(x) for x in w.v]
    out = []
    for xyz in rows:
        out.append([
            linear[i][0] * xyz[0] + linear[i][1] * xyz[1] + linear[i][2] * xyz[2] + shift[i]
            for i in range(3)
        ])
    return out


def _torus_mean(deltas: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Mean of fractional deltas, each wrapped relative to the first atom.

    Returns the mean and the per-atom residual from that mean, wrapped into
    ``(−1/2, 1/2]`` by :func:`agentsg.asu._wrap_half`.
    """
    base = deltas[0]
    shifted = [
        [base[k] + _wrap_half(delta[k] - base[k]) for k in range(3)]
        for delta in deltas
    ]
    n = len(shifted)
    mean = [sum(row[k] for row in shifted) / n for k in range(3)]
    residual = [
        [_wrap_half(row[k] - mean[k]) for k in range(3)]
        for row in shifted
    ]
    return mean, residual


def _cartesian_rmsd(residuals: list[list[float]], cell) -> float:
    uc = UnitCell(*cell)
    acc = 0.0
    for row in residuals:
        cart = uc.orthogonalize(row)
        acc += cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2]
    return sqrt(acc / len(residuals))


def _snap_origin(mean: list[float], norm: EuclideanNormalizer, ops):
    """Split ``mean`` into a floating shift and a snapped element of ``T'``.

    The snap is done in Smith coordinates. Component ``i`` with invariant
    factor ``d_i > 0`` is rounded to a multiple of ``1/d_i``. The floating
    directions are those :func:`pin_floating_origin` removes; they are not
    snapped. A residual of at least ``1/(2 d_i)`` is an ambiguous snap.
    """
    lat = norm.origin_lattice
    mean_v = Vector3(mean)
    pinned = pin_floating_origin(mean_v, ops)
    # ``pin`` before the mod-1 reduction is the exact floating complement.
    floating = Vector3(a - b for a, b in zip(mean_v.v, lat.pin(mean_v).v))
    y = lat._y(pinned)
    snapped_y = [y.v[0], y.v[1], y.v[2]]
    residual = [0.0, 0.0, 0.0]
    ambiguous = False
    for i, factor in enumerate(lat._s):
        if factor == 0:
            residual[i] = 0.0
            continue
        scaled = float(y.v[i] * factor)
        nearest = round(scaled)
        # Halfway values sit on the boundary between two allowed origins.
        if abs(scaled - nearest) >= 0.5 - 1e-9:
            ambiguous = True
        snapped_y[i] = nearest / factor
        residual[i] = float(y.v[i]) - (nearest / factor)
    snapped = lat.reduce(lat._from_y(snapped_y))
    return snapped, floating, tuple(residual), ambiguous


@dataclass(frozen=True)
class MatchCandidate:
    """One placement of model B onto model A.

    ``reindex`` is the linear normalizer matrix ``M``. ``discrete_origin`` is
    the translation of that normalizer element after the mean difference has
    been snapped onto ``T'``. ``floating_shift`` is the continuous part
    :func:`pin_floating_origin` removed. Inversion (``det == -1``) maps a
    chiral molecule to its mirror image.
    """

    reindex: Matrix3
    operator: SymmetryOp
    discrete_origin: Vector3
    floating_shift: Vector3
    rmsd: float
    snap_residual: tuple[float, float, float]
    ambiguous_snap: bool
    det: int

    def linear_map(self) -> Matrix3:
        """Rotation of the composed map ``(M, m) (W, w)``."""
        return self.reindex @ self.operator.W

    def image(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        """Apply the composed map plus the fitted origin and floating shift."""
        pts = [list(map(float, row)) for row in rows]
        mid = _apply(self.operator.W, self.operator.w, pts)
        placed = _apply(self.reindex, self.discrete_origin, mid)
        shift = [float(x) for x in self.floating_shift.v]
        return [[p[i] + shift[i] for i in range(3)] for p in placed]


@dataclass(frozen=True)
class MatchResult:
    """Best placement, every scored candidate, and the enantiomorph flag.

    ``enantiomorph_flag`` is true when a det −1 representative has a smaller
    RMSD than every proper candidate. That representative maps a chiral
    molecule to its mirror image.
    """

    best: MatchCandidate
    ranked: tuple[MatchCandidate, ...]
    enantiomorph_flag: bool
    n_atoms: int


def _candidate(M, m, det, op, xyz_a, xyz_b, cell_a, norm, ops) -> MatchCandidate:
    transformed = _apply(M, m, _apply(op.W, op.w, xyz_b))
    deltas = [
        [a[k] - b[k] for k in range(3)]
        for a, b in zip(xyz_a, transformed)
    ]
    mean, residuals = _torus_mean(deltas)
    snapped, floating, snap_residual, ambiguous = _snap_origin(mean, norm, ops)
    # ``m`` is the coset translation. The reported discrete origin is that
    # translation plus the snapped remainder of the mean.
    discrete = norm.origin_lattice.reduce(m + snapped)
    return MatchCandidate(
        reindex=M,
        operator=op,
        discrete_origin=discrete,
        floating_shift=floating,
        rmsd=_cartesian_rmsd(residuals, cell_a),
        snap_residual=snap_residual,
        ambiguous_snap=ambiguous,
        det=det,
    )


def match_models(
    xyz_a,
    xyz_b,
    sg,
    cell_a,
    cell_b=None,
    allow_improper: bool = False,
    *,
    cartesian: bool = False,
) -> MatchResult:
    """Match corresponding atoms of B onto A under ``N_E(G)``.

    ``xyz_a`` and ``xyz_b`` have the same length and the same atom order.
    They are fractional coordinates of ``sg`` unless ``cartesian`` is set,
    in which case A is fractionalized with ``cell_a`` and B with ``cell_b``
    (or ``cell_a`` when ``cell_b`` is omitted). The RMSD uses ``cell_a``.
    Inversion (det −1) maps a chiral molecule to its mirror image and is
    searched only when ``allow_improper`` is true.
    """
    if cell_a is None or len(tuple(cell_a)) != 6:
        raise ValueError("cell_a must be (a, b, c, alpha, beta, gamma)")
    cell_a_t = tuple(float(x) for x in cell_a)
    cell_b_t = cell_a_t if cell_b is None else tuple(float(x) for x in cell_b)
    if len(cell_b_t) != 6:
        raise ValueError("cell_b must be (a, b, c, alpha, beta, gamma)")
    frac_a = _to_fractional(_as_rows(xyz_a, "xyz_a"), cell_a_t, cartesian)
    frac_b = _to_fractional(_as_rows(xyz_b, "xyz_b"), cell_b_t, cartesian)
    if len(frac_a) != len(frac_b):
        raise ValueError("xyz_a and xyz_b must have the same number of atoms")
    norm = euclidean_normalizer(sg, cell_a_t)
    _ops, _system, rec = _group(sg)
    ops = list(rec.operations()) if hasattr(rec, "operations") else _ops
    proper = [rep for rep in norm.coset_reps() if rep[2] == 1]
    improper = [rep for rep in norm.coset_reps() if rep[2] == -1]
    search = proper + (improper if allow_improper else [])
    if not search:
        raise RuntimeError("the normalizer has no coset representatives to search")

    def score(reps) -> list[MatchCandidate]:
        found = []
        for M, m, det in reps:
            for op in ops:
                found.append(_candidate(M, m, det, op, frac_a, frac_b, cell_a_t, norm, ops))
        return found

    ranked_search = score(search)
    ranked_search.sort(key=lambda item: (item.rmsd, item.ambiguous_snap, item.det))
    proper_scored = [item for item in ranked_search if item.det == 1]
    if allow_improper:
        improper_scored = [item for item in ranked_search if item.det == -1]
    else:
        improper_scored = score(improper)
    best_proper = min(proper_scored, key=lambda item: item.rmsd)
    best_improper = min(improper_scored, key=lambda item: item.rmsd) if improper_scored else None
    flag = (
        best_improper is not None
        and best_improper.rmsd < best_proper.rmsd - 1e-9
    )
    if allow_improper and best_improper is not None and best_improper.rmsd <= best_proper.rmsd:
        best = best_improper
    else:
        best = best_proper
    if allow_improper:
        ranked = tuple(ranked_search)
    else:
        # Improper scores exist only for the flag. The returned list is the
        # proper search the caller asked for.
        ranked = tuple(item for item in ranked_search if item.det == 1)
    return MatchResult(
        best=best,
        ranked=ranked,
        enantiomorph_flag=flag,
        n_atoms=len(frac_a),
    )


def _group(sg):
    from .normalizer import _ops_and_system
    return _ops_and_system(sg)


def _gemmi():
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError(
            "match_models_pdb requires gemmi (pip install gemmi)"
        ) from exc
    return gemmi


def _structure_atoms(structure):
    """Atoms as ``(chain, resseq, icode, name, xyz)``, first altloc only."""
    model = structure[0]
    rows = []
    seen = set()
    for chain in model:
        for residue in chain:
            seq = residue.seqid.num
            icode = (residue.seqid.icode or "").strip()
            for atom in residue:
                alt = atom.altloc or ""
                if alt not in ("", "A", "\0"):
                    continue
                key = (chain.name, seq, icode, atom.name)
                if key in seen:
                    continue
                seen.add(key)
                pos = atom.pos
                rows.append((chain.name, seq, icode, atom.name, [pos.x, pos.y, pos.z]))
    return rows


def _pair_chains(atoms_a, atoms_b):
    """Pair atoms by residue and name inside a chain permutation.

    Chain names are kept when they match. Otherwise every bijection of the
    chains in B onto the chains in A is tried, and the bijection that pairs
    the most atoms is kept. Chains are matched one at a time.
    """
    def by_chain(atoms):
        grouped: dict[str, dict[tuple, list[float]]] = {}
        order = []
        for chain, seq, icode, name, xyz in atoms:
            if chain not in grouped:
                grouped[chain] = {}
                order.append(chain)
            grouped[chain][(seq, icode, name)] = xyz
        return order, grouped

    order_a, map_a = by_chain(atoms_a)
    order_b, map_b = by_chain(atoms_b)
    if len(order_a) != len(order_b):
        raise ValueError(
            f"chain counts differ ({len(order_a)} vs {len(order_b)}); "
            "cannot pair models chain by chain"
        )
    if len(order_a) > 6:
        # Named chains pair with each other. Differing ids at this size are
        # left to the caller, who can pass coordinates already paired.
        if set(order_a) == set(order_b):
            candidates = [tuple(name for name in order_a)]
        else:
            raise ValueError(
                "more than 6 chains and the chain ids differ; "
                "pass coordinates already paired, or rename the chains"
            )
    else:
        candidates = list(itertools.permutations(order_b))

    best = None
    best_key = None
    for perm in candidates:
        paired_a = []
        paired_b = []
        for chain_a, chain_b in zip(order_a, perm):
            bucket = map_b[chain_b]
            for key, xyz_a in map_a[chain_a].items():
                xyz_b = bucket.get(key)
                if xyz_b is None:
                    continue
                paired_a.append(xyz_a)
                paired_b.append(xyz_b)
        rank = (len(paired_a), tuple(perm) == tuple(order_a))
        if best is None or rank > best_key:
            best_key = rank
            best = (paired_a, paired_b, tuple(zip(order_a, perm)))
    if best is None or not best[0]:
        raise ValueError("no atoms paired by (chain, resseq, icode, name)")
    return best


def _cell_of(structure) -> tuple[float, float, float, float, float, float]:
    cell = structure.cell
    return (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)


def match_models_pdb(
    path_a,
    path_b,
    sg,
    cell_a=None,
    cell_b=None,
    allow_improper: bool = False,
):
    """Pair two PDB/mmCIF models with ``gemmi`` and call :func:`match_models`.

    Atoms are paired by ``(resseq, icode, name)`` inside each chain, after a
    chain permutation. Coordinates are Cartesian. ``cell_a`` and ``cell_b``
    default to the cells written in the files.
    """
    gemmi = _gemmi()
    struct_a = gemmi.read_structure(str(path_a))
    struct_b = gemmi.read_structure(str(path_b))
    if cell_a is None:
        cell_a = _cell_of(struct_a)
    if cell_b is None:
        cell_b = _cell_of(struct_b)
    paired_a, paired_b, chain_map = _pair_chains(
        _structure_atoms(struct_a), _structure_atoms(struct_b),
    )
    result = match_models(
        paired_a, paired_b, sg, cell_a, cell_b,
        allow_improper=allow_improper, cartesian=True,
    )
    return result, chain_map
