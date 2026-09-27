"""Derived t- and k-subgroup graphs from operators."""
from agentsg.subgroups import subgroup_edges, subgroup_graph


def _children(sg, *, kind="both"):
    return {(e.type, e.child, e.kind) for e in subgroup_edges(sg, kind=kind)}


def test_p212121_t_includes_p21():
    kids = {e.child for e in subgroup_edges(19, kind="t")}
    assert 4 in kids


def test_p43212_t_includes_p43_and_222():
    kids = {e.child for e in subgroup_edges(96, kind="t")}
    assert 78 in kids
    # axis-aligned 222 in this cell is 19; the diagonal 222 is 18 after 45°.
    assert 19 in kids or 18 in kids


def test_p21c_t_includes_p21_pc_and_pbar1():
    kids = {e.child for e in subgroup_edges(14, kind="t")}
    assert {2, 4, 7} <= kids


def test_fm3m_k_iia_is_pm3m():
    edges = subgroup_edges(225, kind="k")
    iia = [e for e in edges if e.kind == "IIa"]
    assert any(e.child == 221 for e in iia)
    assert all(e.type == "k" for e in iia)


def test_fm3m_t_includes_cubic_children():
    kids = {e.child for e in subgroup_edges(225, kind="t")}
    assert {202, 209, 216} <= kids


def test_p1_has_no_proper_t():
    assert subgroup_edges(1, kind="t") == []


def test_graph_labels_t_vs_k():
    g = subgroup_graph(96)
    assert g["sg_number"] == 96
    assert g["maximal"] is True
    kinds = {e["type"] for e in g["edges"]}
    assert "t" in kinds
    t_to = {e["to"] for e in g["edges"] if e["type"] == "t"}
    assert 78 in t_to
    for e in g["edges"]:
        assert e["type"] in ("t", "k")
        assert e["kind"] in ("I", "IIa", "IIb")


def test_kind_filter_and_p212121_k_iib():
    t_only = subgroup_edges(19, kind="t")
    k_only = subgroup_edges(19, kind="k")
    assert t_only and all(e.type == "t" for e in t_only)
    assert any(e.kind == "IIb" for e in k_only)
