"""Centring flags follow the translation coset, not det(P)."""
from agentsg.serve.handlers import setting_info


def test_c2221_primitive_cob_removes_centring():
    st = setting_info({"setting": "C 2 2 21 (a/2+b/2,-a/2+b/2,c)"})
    assert st["det"] == "1/2"
    assert st["centring_from"] == "C"
    assert st["centring_to"] == "P"
    assert st["centring_changed"] is True
    assert st["added_centering"] is False
    assert "deprecated" in st["note"]


def test_axis_permutation_keeps_det_and_moves_c_to_a():
    st = setting_info({"setting": "C 2 2 21 (c,a,b)"})
    assert st["det"] == "1"
    assert st["centring_from"] == "C"
    assert st["centring_to"] == "A"
    assert st["centring_changed"] is True
    assert st["added_centering"] is False


def test_primitive_supercell_that_adds_c_centring():
    # Inverse of the C → primitive cob. The new translation coset is C.
    st = setting_info({"setting": "P 2 2 2 (a+b,-a+b,c)"})
    assert st["centring_from"] == "P"
    assert st["centring_to"] == "C"
    assert st["centring_changed"] is True
    assert st["added_centering"] is True


def test_half_a_supercell_is_not_a_bravais_centring():
    st = setting_info({"setting": "P 2 2 2 (2a,b,c)"})
    assert st["centring_from"] == "P"
    assert st["centring_to"] is None
    assert st["centring_changed"] is True
    assert st["added_centering"] is True
    assert st["centring_translations"] == [["0", "0", "0"], ["1/2", "0", "0"]]
