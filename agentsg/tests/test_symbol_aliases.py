"""Short monoclinic names, parenthesised screws, and ITA number:setting codes.

Comparisons are sets of exact Seitz operators. No coordinate grid.
"""
from agentsg.ita_settings import ITA_SETTINGS, hm_short, symbol_resolution
from agentsg.serve.handlers import setting_info, space_group_info
from agentsg.setting import SpaceGroupSetting
from agentsg.space_groups import space_group


def _ops(key):
    return frozenset(space_group(key).operations())


def test_p21_n_matches_full_symbol_and_hall():
    ops = _ops("P21/n")
    assert ops == _ops("P 21/n")
    assert ops == _ops("P 1 21/n 1")
    assert ops == _ops("-P 2yn")
    assert ops == _ops("14:b2")
    info = space_group_info({"sg": "P21/n"})
    assert info["resolved_from"] == "P21/n"
    assert info["assumed"] == "unique axis b"
    assert info["sg_number"] == 14
    assert info["hall"] == "-P 2yn"


def test_number_setting_codes():
    assert _ops("14:c1") == _ops("P 1 1 21/a")
    assert _ops("68:1") == _ops("C c c a")
    origin2 = space_group("68:2")
    assert origin2.number == 68
    assert origin2.hall == "-C 2a 2ac"
    assert _ops("68:2") == _ops("-C 2a 2ac")


def test_parenthesised_screws():
    assert _ops("P2(1)/c") == _ops("P 1 21/c 1")
    assert _ops("P2(1)2(1)2(1)") == _ops("P 21 21 21")
    assert SpaceGroupSetting.parse("P2(1)/c").base.hall == space_group("P 1 21/c 1").hall


def test_listed_short_names_match_unique_axis_b():
    # Each short name is the compression of a unique-axis-b row. C2/a and In
    # are not among those rows, so they are not invented.
    names = [
        "P21/n", "P21/a", "P2/n", "P2/a", "Pn", "Pa",
        "C2/n", "I2/a", "I2/c", "A2/n", "Cn", "Ia", "Ic",
        "A2", "I2", "Am", "Im",
    ]
    for name in names:
        info = space_group_info({"sg": name})
        assert info["assumed"] == "unique axis b"
        assert _ops(name) == _ops(info["hall"])
    present = {hm_short(row) for row in ITA_SETTINGS}
    assert "C2/a" not in present
    assert "In" not in present
    for missing in ("C2/a", "In"):
        try:
            space_group(missing)
        except KeyError:
            continue
        raise AssertionError(f"{missing} is not an ITA short name")


def test_short_round_trip_over_every_setting():
    checked = 0
    for row in ITA_SETTINGS:
        short = hm_short(row)
        if short is None:
            continue
        got = space_group(short)
        assert frozenset(got.operations()) == _ops(row[2])
        assert got.hall == row[2]
        checked += 1
    assert checked > 0


def test_setting_parser_accepts_short_and_code():
    st = setting_info({"setting": "P 21/n"})
    assert st["base_sg_number"] == 14
    assert st["base_sg_hm"] == "P 1 21/n 1"
    assert st["resolved_from"] == "P 21/n"
    assert st["assumed"] == "unique axis b"
    assert frozenset(SpaceGroupSetting.parse("14:b2").operations()) == _ops("-P 2yn")
    meta = symbol_resolution("P 1 21/n 1")
    assert meta is None
