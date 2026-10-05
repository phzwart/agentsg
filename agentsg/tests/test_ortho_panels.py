"""Orthorhombic plate panels use the ITA symbol of the panel's change of basis."""
from agentsg.cell.diagrams import _element_panel_title, orthorhombic_panel
from agentsg.ita_settings import display_hm
from agentsg.setting import SpaceGroupSetting
from agentsg.space_groups import space_group


def _ops(sg):
    return frozenset(sg.operations())


def test_every_orthorhombic_panel_matches_the_transformed_group():
    for number in range(16, 75):
        sg = space_group(number)
        for projection in ("c", "a", "b"):
            title, cob_text, ops, row = orthorhombic_panel(sg, projection)
            assert title == _element_panel_title(sg, projection, "orthorhombic")
            assert title.startswith(f"along {projection}: ")
            assert cob_text in title
            named = display_hm(row[1], row[3], row[4])
            assert _ops(space_group(row[2])) == ops
            assert ops == frozenset(SpaceGroupSetting.parse(
                f"{sg.hermann_mauguin} {cob_text}"
            ).operations())


# (projection) -> (displayed HM, cob). 2016 e-glide spelling when the row has one.
_SPOTS = {
    20: {
        "c": ("C 2 2 21", "(a,b,c)"),
        "a": ("B 2 21 2", "(b,c,a)"),
        "b": ("A 21 2 2", "(c,a,b)"),
    },
    64: {
        "c": ("C m c e", "(a,b,c)"),
        "a": ("B b e m", "(b,c,a)"),
        "b": ("A e m a", "(c,a,b)"),
    },
    68: {
        "c": ("C c c e", "(a,b,c)"),
        "a": ("B b e b", "(b,c,a)"),
        "b": ("A e a a", "(c,a,b)"),
    },
    43: {
        "c": ("F d d 2", "(a,b,c)"),
        "a": ("F d 2 d", "(b,c,a)"),
        "b": ("F 2 d d", "(c,a,b)"),
    },
    46: {
        "c": ("I m a 2", "(a,b,c)"),
        "a": ("I c 2 m", "(b,c,a)"),
        "b": ("I 2 m b", "(c,a,b)"),
    },
    62: {
        "c": ("P n m a", "(a,b,c)"),
        "a": ("P m c n", "(b,c,a)"),
        "b": ("P b n m", "(c,a,b)"),
    },
}


def test_spot_checked_orthorhombic_symbols():
    for number, projections in _SPOTS.items():
        sg = space_group(number)
        for projection, (hm, cob_text) in projections.items():
            title, got_cob, ops, row = orthorhombic_panel(sg, projection)
            shown = display_hm(row[1], row[3], row[4])
            assert shown == hm
            assert got_cob == cob_text
            assert cob_text in title
            assert title.startswith(f"along {projection}: ")
            assert _ops(space_group(row[2])) == ops
    c2221 = space_group(20)
    along_b, _cob, _panel_ops, row = orthorhombic_panel(c2221, "b")
    assert row[1].split()[0] == "A"
    assert "C 2 2 21" not in along_b
