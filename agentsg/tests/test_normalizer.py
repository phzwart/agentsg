"""Euclidean normalizer: Smith solve versus conjugation on a 1/24 grid.

Nothing here is copied from an International Tables normalizer list. A matrix
is a normalizer element when conjugating the operator set lands back in the
set, modulo the translation lattice. The grid search is that test.
"""
from __future__ import annotations

from fractions import Fraction as Fr
from itertools import product
from math import lcm

import pytest

from agentsg.cell.reindex import twin_laws
from agentsg.group import point_group
from agentsg.linalg import Matrix3
from agentsg.normalizer import (
    _mat_key,
    conjugates_into_group,
    euclidean_normalizer,
)
from agentsg.semi_invariants import _WmI, _integer_matrix, origin_lattice
from agentsg.space_groups import SPACE_GROUPS, space_group
from agentsg.tolerances import METRIC_ANGLE_TOL_DEG, METRIC_LENGTH_TOL_PCT

_NEG_I = Matrix3([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
_SWAP_HK = Matrix3([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
_DEN = 24
_FAST = (1, 4, 19, 76, 144)


def _to_den(x, den=_DEN) -> int:
    value = Fr(x) * den
    if value.denominator != 1:
        raise AssertionError(f"{x} is not on a 1/{den} grid")
    return int(value)


def _matmul(A, B):
    return tuple(
        tuple(A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] for j in range(3))
        for i in range(3)
    )


def _matvec(A, v):
    return (
        A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
        A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
        A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
    )


def _inv(M):
    det = (
        M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
        - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
        + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
    )
    if det not in (1, -1):
        raise AssertionError(det)
    cof = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            rows = [r for r in range(3) if r != i]
            cols = [c for c in range(3) if c != j]
            minor = (
                M[rows[0]][cols[0]] * M[rows[1]][cols[1]]
                - M[rows[0]][cols[1]] * M[rows[1]][cols[0]]
            )
            cof[j][i] = ((-1) ** (i + j)) * minor
    return tuple(tuple(cof[i][j] * det for j in range(3)) for i in range(3))


def _in_lattice_mod(lat, den=_DEN):
    common = 1
    for row in lat._Bt_inv.rows:
        for entry in row:
            common = lcm(common, entry.denominator)
    lifted = [[int(entry * common) for entry in row] for row in lat._Bt_inv.rows]
    modulus = common * den

    def contains(diff) -> bool:
        for i in range(3):
            total = (
                lifted[i][0] * diff[0]
                + lifted[i][1] * diff[1]
                + lifted[i][2] * diff[2]
            )
            if total % modulus:
                return False
        return True

    return contains


def _grid_normalizer_matrices(number: int) -> tuple[set, set, object]:
    """Matrices in H that conjugate G into itself for some 1/24-grid shift."""
    ops = list(space_group(number).operations())
    norm = euclidean_normalizer(number)
    contains = _in_lattice_mod(norm.origin_lattice)
    encoded = []
    by_rotation = {}
    for op in ops:
        rotation = _mat_key(op.W)
        shift = tuple(_to_den(component) for component in op.w.v)
        encoded.append((rotation, shift))
        by_rotation.setdefault(rotation, []).append(shift)

    def normalizes(rotation, shift) -> bool:
        inverse = _inv(rotation)
        for left, translation in encoded:
            image = _matmul(_matmul(rotation, left), inverse)
            moved = _matvec(rotation, translation)
            spun = _matvec(image, shift)
            updated = (
                moved[0] + shift[0] - spun[0],
                moved[1] + shift[1] - spun[1],
                moved[2] + shift[2] - spun[2],
            )
            bucket = by_rotation.get(image)
            if not bucket:
                return False
            if not any(
                contains((updated[0] - old[0], updated[1] - old[1], updated[2] - old[2]))
                for old in bucket
            ):
                return False
        return True

    accepted = set()
    for matrix, _reason in (
        *[(M, None) for M, _, _ in norm.linear_reps],
        *list(norm.rejected),
    ):
        key = _mat_key(matrix)
        # A point-group failure moves some rotation out of G for every shift.
        # The grid is the independent check for the translation solve, which
        # is the only place the Smith form could disagree.
        if any(key == _mat_key(M) and reason == "point_group" for M, reason in norm.rejected):
            if normalizes(key, (0, 0, 0)):
                accepted.add(key)
            continue
        for shift in product(range(_DEN), repeat=3):
            if normalizes(key, shift):
                accepted.add(key)
                break
    return accepted, {_mat_key(M) for M, _, _ in norm.linear_reps}, norm


def _assert_grid_agrees(number: int) -> None:
    ops = list(space_group(number).operations())
    found, solved, norm = _grid_normalizer_matrices(number)
    assert found == solved, (
        f"sg {number}: grid and Smith solve disagree; "
        f"only grid {sorted(found - solved)}; only solve {sorted(solved - found)}"
    )
    for matrix, shift, _det in norm.linear_reps:
        assert conjugates_into_group(ops, matrix, shift, norm.origin_lattice), (
            f"sg {number}: returned {( _mat_key(matrix), shift)} does not normalize"
        )
    rotations = {_mat_key(rotation) for rotation in point_group(ops)}
    for matrix, reason in norm.rejected:
        if reason != "point_group":
            continue
        key = _mat_key(matrix)
        images = {_matmul(_matmul(key, rotation), _inv(key)) for rotation in rotations}
        assert images != rotations, (
            f"sg {number}: {_mat_key(matrix)} was rejected as point_group "
            "but conjugation preserves the rotation set"
        )


def test_stored_smith_factor_is_the_origin_lattice_one():
    """U kept on the origin lattice is the factorization of the stacked blocks."""
    lattice = origin_lattice(space_group(19).operations())
    stacked = []
    for rotation in lattice._stack_W:
        block = _integer_matrix(lattice._Bt_inv @ _WmI(rotation) @ lattice._Bt)
        stacked.extend(block)
    assert len(lattice._U) == len(stacked)
    width = 3
    mixed = [
        [
            sum(lattice._U[i][k] * stacked[k][j] for k in range(len(stacked)))
            for j in range(width)
        ]
        for i in range(len(lattice._U))
    ]
    product_rows = [
        [
            sum(mixed[i][k] * lattice._V[k][j] for k in range(width))
            for j in range(width)
        ]
        for i in range(len(mixed))
    ]
    for i, row in enumerate(product_rows):
        factor = lattice._s[i] if i < len(lattice._s) else 0
        for j, entry in enumerate(row):
            assert entry == (factor if i == j else 0)


def test_p212121_accepts_inversion_and_eight_origins():
    norm = euclidean_normalizer(19, (10.0, 15.0, 22.0, 90.0, 90.0, 90.0))
    assert norm.origin_lattice.n_alternative_origins == 8
    assert any(matrix == _NEG_I for matrix, _shift, _det in norm.linear_reps)
    assert norm.metric_specialized is False
    assert norm.length_tol_pct == METRIC_LENGTH_TOL_PCT
    assert norm.angle_tol_deg == METRIC_ANGLE_TOL_DEG
    quotes = dict(norm.gate_quotes)
    assert "Percent edge tolerance" in quotes["METRIC_LENGTH_TOL_PCT"]
    assert "Angle tolerance in degrees" in quotes["METRIC_ANGLE_TOL_DEG"]


def test_p41_rejects_inversion_for_translation():
    norm = euclidean_normalizer(76, (10.0, 10.0, 20.0, 90.0, 90.0, 90.0))
    reasons = [reason for matrix, reason in norm.rejected if matrix == _NEG_I]
    assert reasons == ["translation"]
    assert all(matrix != _NEG_I for matrix, _shift, _det in norm.linear_reps)


def test_p21_accepts_inversion_and_floats_along_b():
    from agentsg.linalg import Vector3
    norm = euclidean_normalizer(4, (12.0, 20.0, 15.0, 90.0, 100.0, 90.0))
    assert any(matrix == _NEG_I for matrix, _shift, _det in norm.linear_reps)
    assert norm.origin_lattice.floating == (Vector3((0, 1, 0)),)
    assert norm.metric_specialized is False


def test_p31_accepts_the_hk_reindex_twofold():
    norm = euclidean_normalizer(144, (10.0, 10.0, 20.0, 90.0, 90.0, 120.0))
    assert any(matrix == _SWAP_HK for matrix, _shift, _det in norm.linear_reps)
    assert norm.metric_specialized is False


def test_p1_linear_part_is_inversion_with_three_floating_axes():
    norm = euclidean_normalizer(1)
    mats = {_mat_key(matrix) for matrix, _shift, _det in norm.linear_reps}
    assert mats == {
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
    }
    assert len(norm.origin_lattice.floating) == 3
    assert norm.index() == 2


def test_metric_specialization_tracks_the_length_gate():
    close = euclidean_normalizer(19, (10.0, 10.1, 30.0, 90.0, 90.0, 90.0))
    far = euclidean_normalizer(19, (10.0, 15.0, 30.0, 90.0, 90.0, 90.0))
    assert close.metric_specialized is True
    assert any(matrix == _SWAP_HK for matrix, _shift, _det in close.linear_reps)
    assert far.metric_specialized is False
    assert all(matrix != _SWAP_HK for matrix, _shift, _det in far.linear_reps)


def test_twin_laws_classify_does_not_change_the_default_list():
    cell = (50.0, 50.0, 70.0, 90.0, 90.0, 90.0)
    plain = twin_laws("P4", cell)
    labeled = twin_laws("P4", cell, classify=True)
    assert [matrix for matrix, _role in labeled] == plain
    assert labeled[0] == (((1, 0, 0), (0, 1, 0), (0, 0, 1)), "normalizer")
    norm = euclidean_normalizer("P4", cell)
    accepted = {_mat_key(matrix) for matrix, _shift, _det in norm.linear_reps}
    for matrix, role in labeled:
        assert role == ("normalizer" if matrix in accepted else "twin")


def test_p31_reindex_coset_can_be_a_normalizer():
    cell = (50.0, 50.0, 70.0, 90.0, 90.0, 120.0)
    labeled = twin_laws(144, cell, classify=True)
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert labeled[0][0] == identity
    assert labeled[0][1] == "normalizer"
    swap = ((0, 1, 0), (1, 0, 0), (0, 0, -1))
    roles = {matrix: role for matrix, role in labeled}
    norm = euclidean_normalizer(144, cell)
    accepted = {_mat_key(matrix) for matrix, _shift, _det in norm.linear_reps}
    for matrix, role in labeled:
        assert role == ("normalizer" if matrix in accepted else "twin")
    assert swap in accepted
    assert any(role == "twin" for _matrix, role in labeled)
    assert any(matrix != identity and role == "normalizer" for matrix, role in labeled)
    if swap in roles:
        assert roles[swap] == "normalizer"


@pytest.mark.parametrize("number", _FAST)
def test_grid_agrees_on_the_fast_subset(number):
    _assert_grid_agrees(number)


@pytest.mark.slow
def test_grid_agrees_for_every_standard_setting():
    disagreements = []
    for number, _hm, _hall, _system in SPACE_GROUPS:
        try:
            _assert_grid_agrees(number)
        except AssertionError as exc:
            disagreements.append(str(exc))
    assert not disagreements, "\n".join(disagreements)
