"""Tests for the 230-group standard-setting table + lookup API."""
import pytest
from agentsg.space_groups import SPACE_GROUPS, space_group


def test_table_has_230_unique_numbers():
    assert len(SPACE_GROUPS) == 230
    assert sorted(r[0] for r in SPACE_GROUPS) == list(range(1, 231))


def test_lookup_by_number():
    assert space_group(1).hermann_mauguin.replace(" ", "") == "P1"
    assert space_group(230).number == 230


def test_lookup_by_hall_and_hm():
    assert space_group("P 2ac 2ab").number == 19        # by Hall
    assert space_group("P 21 21 21").number == 19        # by HM
    assert space_group("Fm-3m").number == 225            # dashed HM
    assert space_group("fm3m").number == 225             # dash-insensitive


def test_dash_ambiguous_lookup_still_works_for_exact_symbol():
    # exact HM 'P 1' resolves to 1; 'P -1' to 2 -- no ambiguity when the bar
    # is written explicitly
    assert space_group("P 1").number == 1
    assert space_group("P -1").number == 2


def test_unknown_symbol_raises():
    with pytest.raises(KeyError):
        space_group("not a group")
    with pytest.raises(KeyError):
        space_group(999)


@pytest.mark.parametrize("num,order", [(1, 1), (2, 2), (19, 4), (225, 192), (227, 192), (230, 96)])
def test_group_orders(num, order):
    assert space_group(num).order() == order


def test_crystal_systems_present():
    systems = {r[3] for r in SPACE_GROUPS}
    assert systems == {
        "triclinic", "monoclinic", "orthorhombic",
        "tetragonal", "trigonal", "hexagonal", "cubic",
    }
