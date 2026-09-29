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
        if el["type"] != "glide" or el.get("contained_in"):
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
        if el.get("contained_in"):
            continue
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


def _plane_offset(el):
    """Signed offset of a plane, reduced into ``[0, g)`` for primitive normal."""
    import math
    from fractions import Fraction
    n = tuple(int(c) for c in el["axis_exact"])
    loc = el["location_exact"]
    c = sum(n[i] * loc[i] for i in range(3))
    g = 0
    for a in n:
        g = abs(a) if g == 0 else math.gcd(g, abs(a))
    q = Fraction(c) / g
    return c - g * (q.numerator // q.denominator)


def _exact_normal(el, normal):
    ax = tuple(int(c) for c in el["axis_exact"])
    return ax == normal or ax == tuple(-c for c in normal)


def test_p3m1_half_offset_glide_is_not_on_the_mirror():
    """P3m1: m on x=2y, and g (1, 1/2, 0) on the parallel plane x-2y = 1/2."""
    from fractions import Fraction
    els = _element_copies(156)
    normal = (1, -2, 0)
    mirrors = [e for e in els if e["symbol"] == "m" and _exact_normal(e, normal)]
    glides = [e for e in els if e["symbol"] == "g" and _exact_normal(e, normal)]
    assert mirrors and glides
    assert any(_plane_offset(e) == 0 for e in mirrors)
    assert any(_plane_offset(e) == Fraction(1, 2) for e in glides)
    glide = next(e for e in glides if _plane_offset(e) == Fraction(1, 2))
    got = glide["intrinsic_exact"]
    # (1, 1/2, 0) and (-1, -1/2, 0) differ by a lattice vector.
    delta = tuple(got[i] - Fraction((1, Fraction(1, 2), 0)[i]) for i in range(3))
    assert all(d.denominator == 1 for d in delta)


def test_p31m_axial_glides_sit_on_the_half_planes():
    """P31m: a lies on y=1/2 and b on x=1/2."""
    from fractions import Fraction
    els = _element_copies(157)
    a = [e for e in els if e["symbol"] == "a" and _exact_normal(e, (0, 1, 0))]
    b = [e for e in els if e["symbol"] == "b" and _exact_normal(e, (1, 0, 0))]
    assert a and b
    assert any(_plane_offset(e) == Fraction(1, 2) for e in a)
    assert any(_plane_offset(e) == Fraction(1, 2) for e in b)


def test_p3c1_c_glide_is_in_the_2x_family():
    """P3c1: the c glide is the family parallel to 2x, x, z (normal 1, -2, 0)."""
    els = _element_copies(158)
    assert any(e["symbol"] == "c" and _exact_normal(e, (1, -2, 0)) for e in els)


def test_p63mmc_c_glides_on_the_axes_and_the_half_planes():
    """P6₃/mmc: c at x=0 and y=0, and on the parallel planes at 1/2."""
    from fractions import Fraction
    els = _element_copies(194)
    for normal in ((1, 0, 0), (0, 1, 0)):
        planes = [e for e in els if e["symbol"] == "c" and _exact_normal(e, normal)]
        offs = {_plane_offset(e) for e in planes}
        assert 0 in offs and Fraction(1, 2) in offs


@pytest.mark.parametrize("num", range(143, 195))
def test_hexagonal_elements_regenerate_the_group(num):
    """Every element rebuilds an operator, and every operator is a listed element."""
    from agentsg.cell.diagrams import _element_lattice, _hexagonal_family
    from agentsg.cell.symmetry_elements import classify_element

    sg = space_group(num)
    lattice = _element_lattice(sg)
    hexagonal = _hexagonal_family(sg)
    els = _element_copies(sg)
    ops = {}
    for op in sg.operations():
        W = tuple(tuple(int(c) for c in row) for row in op.W.rows)
        w = tuple(comp % 1 for comp in op.w.v)
        ops.setdefault(W, set()).add(w)
    for el in els:
        if el["type"] in ("identity", "translation"):
            continue
        W = tuple(tuple(int(c) for c in row) for row in el["W_exact"])
        loc = el["location_exact"]
        intr = el.get("intrinsic_raw_exact") or el["intrinsic_exact"]
        w = []
        for i in range(3):
            acc = intr[i]
            for j in range(3):
                acc -= (W[i][j] - (1 if i == j else 0)) * loc[j]
            w.append(acc % 1)
        assert tuple(w) in ops.get(W, ()), (num, el["symbol"], tuple(w))
    listed = set()
    for el in els:
        listed.add((
            el["type"], el["symbol"], el.get("axis_exact"), el.get("location_exact"),
        ))
    for op in sg.operations():
        W = [[int(c) for c in row] for row in op.W.rows]
        el = classify_element(W, tuple(op.w.v), lattice=lattice, hexagonal=hexagonal)
        if el["type"] in ("identity", "translation"):
            continue
        key = (el["type"], el["symbol"], el.get("axis_exact"), el.get("location_exact"))
        assert key in listed, (num, op.as_xyz(), el["symbol"])


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
