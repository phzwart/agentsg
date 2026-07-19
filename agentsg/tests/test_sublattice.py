"""Sublattice generation (Hermite normal form) tests, per Zwart/GK/Adams 2006 sec 2.4."""
import pytest
from agentsg.cell.sublattice import (
    generate_sublattices, sublattice_count, is_hermite_normal_form,
    apply_to_cell, diagonal_triples,
)
from agentsg.cell.metric import UnitCell


def _det(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


# OEIS A001001: number of sublattices of index n in Z^3
_OEIS = {1: 1, 2: 7, 3: 13, 4: 35, 5: 31, 6: 91, 7: 57, 8: 155, 9: 130, 10: 217}


@pytest.mark.parametrize("n,expected", list(_OEIS.items()))
def test_counts_match_oeis(n, expected):
    mats = generate_sublattices(n)
    assert len(mats) == expected
    assert sublattice_count(n) == expected


@pytest.mark.parametrize("n", range(1, 11))
def test_all_matrices_have_correct_determinant(n):
    for M in generate_sublattices(n):
        assert _det(M) == n


@pytest.mark.parametrize("n", range(1, 11))
def test_all_matrices_are_valid_hnf(n):
    for M in generate_sublattices(n):
        assert is_hermite_normal_form(M)


@pytest.mark.parametrize("n", range(1, 8))
def test_matrices_are_distinct(n):
    mats = generate_sublattices(n)
    seen = {tuple(tuple(r) for r in M) for M in mats}
    assert len(seen) == len(mats)


def test_diagonal_triples_multiply_to_index():
    for n in range(1, 13):
        for (a, d, f) in diagonal_triples(n):
            assert a * d * f == n


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_apply_to_cell_scales_volume(n):
    base = (5, 6, 7, 88, 92, 97)
    V0 = UnitCell(*base).volume()
    M = generate_sublattices(n)[0]
    enlarged = apply_to_cell(base, M)
    assert abs(UnitCell(*enlarged).volume() / V0 - n) < 1e-9


def test_index_one_is_identity_only():
    mats = generate_sublattices(1)
    assert mats == [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]


def test_rejects_nonpositive_index():
    with pytest.raises(ValueError):
        generate_sublattices(0)
