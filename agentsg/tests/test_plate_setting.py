"""ita_plate setting forms: a change of basis is not a Hermann–Mauguin symbol."""
import pytest

from agentsg.serve.handlers import ita_plate_json, resolve_plate_sg
from agentsg.serve.http import HttpError


_COB = "a/2+b/2,-a/2+b/2,c"
_FORMS = [
    {"sg": "C 2 2 21", "setting": _COB},
    {"sg": "C 2 2 21", "setting": f"({_COB})"},
    {"setting": f"C 2 2 21 ({_COB})"},
    {"sg": f"C 2 2 21 ({_COB})"},
]


def _inventory(data):
    meta = ita_plate_json(data, png_query="sg=20")
    return meta["n_total"], meta["elements"]


def test_four_setting_forms_share_the_primitive_inventory():
    inventories = [_inventory(form) for form in _FORMS]
    assert all(inv == inventories[0] for inv in inventories)
    for form in _FORMS:
        assert resolve_plate_sg(form).order() == 4


def test_symbol_in_setting_must_match_sg():
    with pytest.raises(HttpError, match="No. 1"):
        resolve_plate_sg({"sg": "C 2 2 21", "setting": "P"})
    ok = resolve_plate_sg({"sg": "P 21 21 21", "setting": "P 21 21 21"})
    assert ok.base.number == 19
    with pytest.raises(HttpError, match="No. 18"):
        resolve_plate_sg({"sg": "P 21 21 21", "setting": "P 21 21 2"})


def test_bare_lattice_letter_is_not_a_setting():
    with pytest.raises(HttpError, match="lattice letter"):
        resolve_plate_sg({"setting": "C"})
    with pytest.raises(HttpError, match="lattice letter"):
        resolve_plate_sg({"sg": "P 1", "setting": "P"})


def test_r_guard_still_rejects_a_non_rhombohedral_group():
    meta = ita_plate_json({"sg": 146, "setting": "R"}, png_query="sg=146")
    assert meta["sg_number"] == 146
    with pytest.raises(ValueError, match="rhombohedral groups"):
        resolve_plate_sg({"sg": 16, "setting": "R"})
