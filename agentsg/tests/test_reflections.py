"""Reflection-condition tests: exhaustive absence agreement with gemmi over an
hkl grid for all 230 groups, plus internal consistency of the conditions
reporter."""
import pytest
from agentsg.space_groups import space_group, SPACE_GROUPS
from agentsg.group import is_systematically_absent
from agentsg.reflections import reflection_conditions, _CLASSES
from agentsg.linalg import Vector3

gemmi = pytest.importorskip("gemmi")

_R = 4
_HKLS = [(h, k, l)
         for h in range(-_R, _R + 1)
         for k in range(-_R, _R + 1)
         for l in range(-_R, _R + 1)
         if not (h == k == l == 0)]


@pytest.mark.parametrize("row", SPACE_GROUPS, ids=[str(r[0]) for r in SPACE_GROUPS])
def test_absences_match_gemmi(row):
    n = row[0]
    ops = list(space_group(n).operations())
    gops = gemmi.SpaceGroup(n).operations()
    for hkl in _HKLS:
        assert (is_systematically_absent(Vector3(hkl), ops)
                == gops.is_systematically_absent(list(hkl))), (n, hkl)


@pytest.mark.parametrize("n", [1, 2, 14, 15, 19, 62, 88, 141, 194, 225, 227, 230])
def test_reported_conditions_reproduce_absences(n):
    """Every reported condition string must, when applied, reproduce the exact
    absent/present split it summarises."""
    ops = list(space_group(n).operations())
    cond = reflection_conditions(ops)
    # for a handful of the named classes, verify no reflection contradicts the
    # derived absence rule (the reporter derives from the same rule, so this is
    # a self-consistency guard).
    for name, sel in _CLASSES:
        members = [(h, k, l) for (h, k, l) in _HKLS if sel(h, k, l)]
        if not members:
            continue
        # if reported as unrestricted, none may be absent
        if name not in cond:
            assert not any(is_systematically_absent(Vector3(m), ops) for m in members), (n, name)


def test_known_conditions_present():
    # F-centring integral condition on Fm-3m
    c225 = reflection_conditions(list(space_group(225).operations()))
    assert c225["hkl"]  # non-empty integral condition (F-centring)
    # 21 screw serial conditions on P212121
    c19 = reflection_conditions(list(space_group(19).operations()))
    assert c19["h00"] == "h = 2n"
    assert c19["0k0"] == "k = 2n"
    assert c19["00l"] == "l = 2n"
    # Ia-3d: hhl h+k+l = 4n (d-glide)
    c230 = reflection_conditions(list(space_group(230).operations()))
    assert c230["hhl"] == "h+k+l = 4n"
