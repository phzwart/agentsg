"""Reflection conditions as sublattices in projection -- checks.

Two independent oracles: the per-reflection rule ``is_systematically_absent``
(itself verified against gemmi over a full grid in test_reflections.py), and
gemmi directly. Plus the published ITA tables for a spread of groups.
"""
import pytest

from agentsg.space_groups import space_group
from agentsg.group import is_systematically_absent
from agentsg.linalg import Vector3
from agentsg import reflection_lattice as RL

_R = 4
_HKLS = [(h, k, l)
         for h in range(-_R, _R + 1)
         for k in range(-_R, _R + 1)
         for l in range(-_R, _R + 1)
         if not (h == k == l == 0)]


# --- the sublattices decide absence exactly as the per-reflection rule ----------

@pytest.mark.parametrize("n", range(1, 231))
def test_lattice_absence_matches_operator_rule(n):
    ops = list(space_group(n).operations())
    lat = RL.present_lattices(ops)
    for hkl in _HKLS:
        assert RL.is_absent_by_lattice(hkl, lat) == \
            is_systematically_absent(Vector3(hkl), ops), (n, hkl)


@pytest.mark.parametrize("n", [5, 15, 62, 70, 88, 142, 146, 167, 169, 186,
                               194, 220, 225, 227, 230])
def test_lattice_absence_matches_gemmi(n):
    gemmi = pytest.importorskip("gemmi")
    g = gemmi.find_spacegroup_by_number(n).operations()
    lat = RL.present_lattices(space_group(n).operations())
    for hkl in _HKLS:
        assert RL.is_absent_by_lattice(hkl, lat) == g.is_systematically_absent(hkl)


# --- strata are the ITA reflection classes ----------------------------------------

def _names(n):
    return {RL.class_name(s["basis"]) for s in RL.strata(space_group(n).operations())}


def test_strata_orthorhombic():
    assert _names(47) == {"hkl", "0kl", "h0l", "hk0", "h00", "0k0", "00l"}


def test_strata_tetragonal_has_diagonal_zones():
    s = _names(123)
    assert {"hkl", "0kl", "h0l", "hk0", "hhl", "h-hl", "00l", "h00", "0k0",
            "hh0", "h-h0"} <= s


def test_strata_hexagonal_has_both_prism_zones():
    # 6/mmm: the h-h0l and hh2-hl zones (3-index h-hl, hhl) are distinct strata
    s = _names(191)
    assert "h-hl" in s and "hhl" in s and "00l" in s


def test_strata_cubic_body_diagonal():
    assert "hhh" in _names(221)


def test_strata_monoclinic_only_unique_axis():
    # P2/m, unique axis b: the b-axis row and the a-c zone, nothing else
    assert _names(10) == {"hkl", "h0l", "0k0"}


@pytest.mark.parametrize("n", range(1, 231))
def test_crystal_family_from_operations(n):
    sg = space_group(n)
    fam, axis = RL.crystal_family(sg.operations())
    want = sg.crystal_system
    if want in ("trigonal", "hexagonal"):
        want = "hexagonal"
    assert fam == want, (n, fam, want)
    if fam == "monoclinic":
        assert axis == 1        # standard settings are unique axis b


# --- published ITA conditions ----------------------------------------------------------

ITA = {
    4: {"0k0": "k = 2n"},
    14: {"h0l": "l = 2n", "0k0": "k = 2n", "00l": "l = 2n"},
    15: {"hkl": "h+k = 2n", "h0l": "h, l = 2n", "0kl": "k = 2n",
         "hk0": "h+k = 2n", "h00": "h = 2n", "0k0": "k = 2n", "00l": "l = 2n"},
    19: {"h00": "h = 2n", "0k0": "k = 2n", "00l": "l = 2n"},
    62: {"0kl": "k+l = 2n", "hk0": "h = 2n", "h00": "h = 2n", "0k0": "k = 2n",
         "00l": "l = 2n"},
    64: {"hkl": "h+k = 2n", "0kl": "k = 2n", "h0l": "h, l = 2n", "hk0": "h, k = 2n",
         "h00": "h = 2n", "0k0": "k = 2n", "00l": "l = 2n"},
    70: {"hkl": "h+k, h+l = 2n", "0kl": "l = 2n; k+l = 4n", "h0l": "l = 2n; h+l = 4n",
         "hk0": "k = 2n; h+k = 4n", "h00": "h = 4n", "0k0": "k = 4n", "00l": "l = 4n"},
    88: {"hkl": "h+k+l = 2n", "hk0": "h, k = 2n", "0kl": "k+l = 2n", "hhl": "l = 2n",
         "00l": "l = 4n", "h00": "h = 2n", "hh0": "h = 2n"},
    133: {"0kl": "k = 2n", "hk0": "h+k = 2n", "hhl": "l = 2n", "00l": "l = 2n",
          "h00": "h = 2n"},
    142: {"hkl": "h+k+l = 2n", "0kl": "k, l = 2n", "hk0": "h, k = 2n", "hhl": "2h+l = 4n",
          "00l": "l = 4n", "h00": "h = 2n", "hh0": "h = 2n"},
    146: {"hkl": "-h+k+l = 3n", "hk0": "-h+k = 3n", "h-hl": "h+l = 3n", "hhl": "l = 3n",
          "00l": "l = 3n", "h-h0": "h = 3n"},
    161: {"hkl": "-h+k+l = 3n", "hk0": "-h+k = 3n", "h-hl": "l = 2n; h+l = 3n",
          "hhl": "l = 3n", "00l": "l = 6n", "h-h0": "h = 3n"},
    169: {"00l": "l = 6n"},
    186: {"hhl": "l = 2n", "00l": "l = 2n"},
    225: {"hkl": "h+k, h+l = 2n", "0kl": "k, l = 2n", "hhl": "h+l = 2n", "h00": "h = 2n"},
    227: {"hkl": "h+k, h+l = 2n", "0kl": "l = 2n; k+l = 4n", "hhl": "h+l = 2n",
          "h00": "h = 4n"},
    230: {"hkl": "h+k+l = 2n", "0kl": "k, l = 2n", "hhl": "2h+l = 4n", "h00": "h = 4n"},
}


@pytest.mark.parametrize("n", sorted(ITA))
def test_ita_conditions(n):
    """Exact agreement on ITA's tabulated classes (minimal form: an implied
    'k, l = 2n' is not repeated next to 'k+l = 4n; l = 2n'). Extra strata that
    ITA does not tabulate (cubic hh0, hhh) are allowed alongside."""
    got = RL.reflection_conditions(space_group(n).operations())
    for cls, cond in ITA[n].items():
        assert got.get(cls) == cond, (n, cls, got.get(cls), cond)
    ita_std = {RL.class_name(RL._row_hnf(b))
               for b in RL._std_classes(*RL.crystal_family(space_group(n).operations()))}
    for cls in got:
        if cls in ita_std:
            assert cls in ITA[n], (n, cls, got[cls])


def test_no_modulus_bound():
    """No enumeration, so nothing limits the modulus: a synthetic 12_1-like
    screw (translation 1/12 along c) gives l = 12n straight away."""
    from agentsg.symmetry_op import SymmetryOp
    from fractions import Fraction as Fr
    from agentsg.linalg import Matrix3
    ops = [SymmetryOp(Matrix3(((1, 0, 0), (0, 1, 0), (0, 0, 1))), Vector3((0, 0, 0))),
           SymmetryOp(Matrix3(((1, 0, 0), (0, 1, 0), (0, 0, 1))), Vector3((0, 0, Fr(1, 12))))]
    # the pure translation by c/12 is a lattice statement: all hkl need l = 12n
    assert RL.reflection_conditions(ops)["hkl"] == "l = 12n"
