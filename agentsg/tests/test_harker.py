"""Algebraic Harker sections from space-group operators."""
from __future__ import annotations
from fractions import Fraction as Fr

import pytest

from agentsg import space_group
from agentsg.harker import (
    harker_sections, harker_vector, site_from_harker, HarkerLocus,
)
from agentsg.linalg import Vector3, IDENTITY3


def _strs(ops):
    return [str(L) for L in harker_sections(ops)]


def test_p21_section():
    loci = harker_sections(space_group(4).operations())
    assert len(loci) == 1
    assert loci[0].kind == "section"
    assert str(loci[0]) == "v = 1/2"


def test_p2_zero_section():
    loci = harker_sections(space_group(3).operations())
    assert [str(L) for L in loci] == ["v = 0"]


def test_p212121_three_sections():
    assert set(_strs(space_group(19).operations())) == {
        "u = 1/2", "v = 1/2", "w = 1/2",
    }


def test_pm_harker_line():
    loci = harker_sections(space_group(6).operations())
    assert len(loci) == 1
    assert loci[0].kind == "line"
    assert str(loci[0]) == "u = 0; w = 0"


def test_pbar1_no_reduced_locus():
    assert harker_sections(space_group(2).operations()) == []


def test_p21c_section_and_line():
    kinds = {L.kind for L in harker_sections(space_group(14).operations())}
    assert kinds == {"section", "line"}
    assert "v = 1/2" in _strs(space_group(14).operations())


def test_harker_vector_lies_on_locus():
    ops = list(space_group(19).operations())
    loci = harker_sections(ops)
    x = Vector3((Fr(1, 7), Fr(2, 9), Fr(3, 11)))
    for op in ops:
        if op.W == IDENTITY3:
            continue
        u = harker_vector(op, x)
        # every non-identity op with a reduced locus must land on some locus
        matching = [L for L in loci if L.contains(u)]
        if any(op in L.operations for L in loci):
            assert matching


def test_site_from_harker_roundtrip():
    ops = list(space_group(4).operations())
    op = next(o for o in ops if o.W != IDENTITY3)
    x = Vector3((Fr(1, 5), Fr(1, 7), Fr(1, 3)))
    u = harker_vector(op, x)
    sol = site_from_harker(op, u)
    assert sol is not None
    particular, basis = sol
    # particular + α·axis should recover x for some α
    assert len(basis) == 1
    # x and particular differ only along the unique axis (y)
    assert particular.v[0] == x.v[0]
    assert particular.v[2] == x.v[2]


@pytest.mark.parametrize("n", (4, 14, 19, 76, 96, 198))
def test_every_self_vector_on_some_locus_or_full_rank(n):
    ops = list(space_group(n).operations())
    loci = harker_sections(ops)
    x = Vector3((Fr(1, 8), Fr(3, 11), Fr(5, 13)))
    for op in ops:
        if op.W == IDENTITY3:
            continue
        from agentsg.harker import _ImW, _matrix_rank
        if _matrix_rank(_ImW(op.W)) == 3:
            continue
        u = harker_vector(op, x)
        assert any(L.contains(u) for L in loci)
