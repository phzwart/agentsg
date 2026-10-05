"""Round-trip model matching under a random proper normalizer coset."""
from __future__ import annotations

import random

import pytest

from agentsg.cell.metric import UnitCell
from agentsg.group import point_group
from agentsg.linalg import Vector3
from agentsg.match_models import _apply, _snap_origin, _torus_mean, match_models
from agentsg.normalizer import euclidean_normalizer
from agentsg.space_groups import space_group

# Floating (P2₁), primitive orthorhombic, tetragonal enantiomorph, hexagonal
# screw, C-centred monoclinic, and body-centred cubic.
_CASES = [
    (4, (12.0, 20.0, 15.0, 90.0, 100.0, 90.0)),
    (19, (10.0, 15.0, 22.0, 90.0, 90.0, 90.0)),
    (96, (10.0, 10.0, 20.0, 90.0, 90.0, 90.0)),
    (152, (10.0, 10.0, 18.0, 90.0, 90.0, 120.0)),
    (5, (12.0, 20.0, 15.0, 90.0, 100.0, 90.0)),
    (199, (15.0, 15.0, 15.0, 90.0, 90.0, 90.0)),
]


def _cartesian_rmsd(a_rows, b_rows, cell) -> float:
    _mean, residuals = _torus_mean([
        [a[k] - b[k] for k in range(3)]
        for a, b in zip(a_rows, b_rows)
    ])
    uc = UnitCell(*cell)
    acc = 0.0
    for row in residuals:
        cart = uc.orthogonalize(row)
        acc += cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2]
    return (acc / len(residuals)) ** 0.5


def _floating_shift(norm, rng: random.Random) -> Vector3:
    shift = Vector3((0, 0, 0))
    for direction in norm.origin_lattice.floating:
        coef = rng.uniform(-0.3, 0.3)
        shift = shift + Vector3(coef * float(component) for component in direction.v)
    return shift


@pytest.mark.parametrize("number,cell", _CASES)
def test_match_recovers_a_noisy_proper_placement(number, cell):
    rng = random.Random(number)
    norm = euclidean_normalizer(number, cell)
    ops = list(space_group(number).operations())
    matrix, shift, det = rng.choice(norm.proper_coset_reps())
    assert det == 1
    operator = rng.choice(ops)
    origin = rng.choice(norm.origin_lattice.discrete_origins())
    floating = _floating_shift(norm, rng)
    lattice = Vector3((rng.randrange(3), rng.randrange(3), rng.randrange(3)))
    lattice = lattice + rng.choice(tuple(norm.origin_lattice.centering))
    total = shift + origin + floating + lattice

    xyz_b = [[rng.random() for _ in range(3)] for _ in range(12)]
    clean = _apply(matrix, total, _apply(operator.W, operator.w, xyz_b))
    uc = UnitCell(*cell)
    xyz_a = []
    for point in clean:
        cart = uc.orthogonalize(point)
        noisy = [coord + rng.gauss(0.0, 0.1) for coord in cart]
        xyz_a.append(list(uc.fractionalize(noisy)))

    result = match_models(xyz_a, xyz_b, number, cell)
    assert result.n_atoms == 12
    assert result.best.rmsd < 0.45
    assert result.best.det == 1
    assert result.best.ambiguous_snap is False
    assert result.enantiomorph_flag is False
    assert result.ranked[0].rmsd == min(item.rmsd for item in result.ranked)

    predicted = result.best.image(xyz_b)
    assert _cartesian_rmsd(predicted, clean, cell) < 0.45
    # The recorded factorization may absorb an element of G. The composed
    # linear map, modulo that point group, is the one that was applied.
    applied = matrix @ operator.W
    recovered = result.best.linear_map()
    quotient = recovered.inverse() @ applied
    assert quotient in point_group(ops)


def test_improper_placement_sets_the_enantiomorph_flag():
    number, cell = 19, (10.0, 15.0, 22.0, 90.0, 90.0, 90.0)
    rng = random.Random(7)
    norm = euclidean_normalizer(number, cell)
    improper = [rep for rep in norm.coset_reps() if rep[2] == -1]
    matrix, shift, det = rng.choice(improper)
    assert det == -1
    operator = rng.choice(list(space_group(number).operations()))
    xyz_b = [[rng.random() for _ in range(3)] for _ in range(12)]
    clean = _apply(matrix, shift, _apply(operator.W, operator.w, xyz_b))
    hidden = match_models(clean, xyz_b, number, cell, allow_improper=False)
    assert hidden.enantiomorph_flag is True
    assert hidden.best.det == 1
    shown = match_models(clean, xyz_b, number, cell, allow_improper=True)
    assert shown.best.det == -1
    assert shown.best.rmsd < 1e-6
    assert _cartesian_rmsd(shown.best.image(xyz_b), clean, cell) < 1e-6


def test_halfway_smith_component_is_an_ambiguous_snap():
    norm = euclidean_normalizer(19, (10.0, 15.0, 22.0, 90.0, 90.0, 90.0))
    lattice = norm.origin_lattice
    index = next(i for i, factor in enumerate(lattice._s) if factor > 1)
    coords = [0, 0, 0]
    coords[index] = 1 / (2 * lattice._s[index])
    halfway = lattice._from_y(coords)
    _snapped, _floating, _residual, ambiguous = _snap_origin(
        [float(component) for component in halfway.v],
        norm,
        list(space_group(19).operations()),
    )
    assert ambiguous is True
