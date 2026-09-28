"""Regression tests for the ITA-plate element engine (audit J1–J4)."""
from collections import defaultdict

import pytest

from agentsg.cell.diagrams import _element_copies
from agentsg.cell.symmetry_elements import _glide_symbol
from agentsg.serve.handlers import _plate_copies
from agentsg.space_groups import space_group


def _els(num):
    return _plate_copies(_element_copies(space_group(num)))


def _axis(el):
    if not el["axis"]:
        return None
    return tuple(int(round(c)) for c in el["axis"])


def _same_normal(el, normal):
    a = _axis(el)
    if a is None:
        return False
    return a == normal or a == tuple(-c for c in normal)


def _offset(el, normal):
    return sum(n * x for n, x in zip(normal, el["location"]))


@pytest.mark.parametrize("num", range(1, 231))
def test_unique_locus_and_symbol(num):
    """No two elements share a locus and a symbol."""
    seen = set()
    for el in _els(num):
        loc = tuple(round(float(c), 6) for c in el["location"])
        key = (el["symbol"], _axis(el), loc)
        assert key not in seen, (num, key)
        seen.add(key)


@pytest.mark.parametrize("num", range(1, 231))
def test_glide_letter_matches_classifier(num):
    raw = _element_copies(space_group(num))
    hexagonal = space_group(num).crystal_system in ("trigonal", "hexagonal")
    for el in raw:
        if el["type"] != "glide":
            continue
        assert el["symbol"] in {"a", "b", "c", "n", "d", "g", "e"}
        again = _glide_symbol(el["intrinsic_exact"], el["W_exact"], hexagonal)
        if el["symbol"] == "e":
            # e is two axial glides on one plane, not a single reduced letter.
            assert again in {"a", "b", "c", "n"}
        else:
            assert again == el["symbol"], (
                num, el["symbol"], again, el["intrinsic_exact"])


@pytest.mark.parametrize("num", range(1, 231))
def test_intrinsic_is_shortest(num):
    """No fixed lattice vector makes the intrinsic part shorter."""
    from agentsg.cell.diagrams import _element_lattice
    from agentsg.cell.symmetry_elements import _dot, _sub

    sg = space_group(num)
    lattice = _element_lattice(sg)
    for el in _element_copies(sg):
        intr = el["intrinsic_exact"]
        W = el["W_exact"]
        base = _dot(intr, intr)
        for L in lattice.fixed(W):
            cand = _sub(intr, L)
            assert _dot(cand, cand) + 0 >= base


def test_p4mm_mirrors_and_glides():
    """P4mm: m through the origin on both diagonals; g only at half-diagonals."""
    els = _els(99)
    for normal, half in (((1, -1, 0), lambda x, y: x - y),
                         ((1, 1, 0), lambda x, y: x + y)):
        mirrors = [e for e in els if e["symbol"] == "m" and _same_normal(e, normal)]
        glides = [e for e in els if e["symbol"] == "g" and _same_normal(e, normal)]
        assert mirrors and glides
        for e in mirrors:
            assert abs(_offset(e, normal) - round(_offset(e, normal))) < 1e-6
        for e in glides:
            assert abs(abs(half(*e["location"][:2])) - 0.5) < 1e-6
            assert abs(_offset(e, normal)) > 1e-6


def test_p4cc_body_diagonal_is_n():
    """P4cc: glide (1/2, 1/2, 1/2) on x-y = 1/2 is an n glide, not g."""
    els = _els(103)
    diag = [e for e in els if _same_normal(e, (1, -1, 0))]
    symbols = {e["symbol"] for e in diag}
    assert "g" not in symbols
    assert "n" in symbols and "c" in symbols
    for e in diag:
        if e["symbol"] == "n":
            assert abs(abs(e["location"][0] - e["location"][1]) - 0.5) < 1e-6


def test_p622_diagonal_axes_are_half_screws():
    """P622: <210> axes are integer directions, half of them 2_1, at rationals."""
    els = _els(177)
    wanted = []
    for e in els:
        a = _axis(e)
        if a is None:
            continue
        aa = tuple(sorted(abs(c) for c in a))
        if aa in ((0, 1, 2),):
            wanted.append(e)
            for c in e["location"]:
                assert abs(c * 12 - round(c * 12)) < 1e-6
    symbols = {e["symbol"] for e in wanted}
    assert symbols == {"2", "2_1"}
    assert any(e["symbol"] == "2" for e in wanted)
    assert any(e["symbol"] == "2_1" for e in wanted)


def test_i23_threefolds_on_distinct_loci():
    by = defaultdict(set)
    for e in _els(197):
        if not str(e["symbol"]).startswith("3"):
            continue
        key = (_axis(e), tuple(round(c, 5) for c in e["location"]))
        by[key].add(e["symbol"])
    assert by
    assert all(len(v) == 1 for v in by.values())
    flat = {s for v in by.values() for s in v}
    assert "3" in flat
    assert "3_1" in flat or "3_2" in flat


def test_fm3m_no_stacked_110_screws():
    by = defaultdict(set)
    for e in _els(225):
        a = _axis(e)
        if a is None or e["symbol"] not in ("2", "2_1"):
            continue
        if sorted(abs(c) for c in a) != [0, 1, 1]:
            continue
        key = (a, tuple(round(c, 5) for c in e["location"]))
        by[key].add(e["symbol"])
    assert by
    assert all(v != {"2", "2_1"} for v in by.values())
