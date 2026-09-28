"""General-position representative and split circles (audit P1, P2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle

from agentsg.cell import diagrams as D
from agentsg.space_groups import space_group


def _projection(num):
    sg = space_group(num)
    if sg.crystal_system == "monoclinic":
        return "b"
    return "c"


def _metrics(num):
    projection = _projection(num)
    sg = space_group(num)
    point = D.best_general_point(sg, projection=projection)
    As, ts = D._projection_maps(sg, projection)
    sigs = [D._map_signature(A, t) for A, t in zip(As, ts)]
    same = np.array([[a == b for b in sigs] for a in sigs])
    imgs = D._projected_images(point, As, ts)
    sep = D._min_projected_separation(imgs, same)
    edge = D._edge_distance(imgs)
    elem = D._trace_clearance(imgs, D._element_traces(sg, projection))
    distinct = {tuple(np.round(q, 3)) for q in imgs}
    return sep, edge, elem, len(distinct), len(imgs)


@pytest.mark.parametrize("num", range(1, 231))
def test_projected_orbit_is_separated(num):
    sep, edge, elem, _distinct, _n = _metrics(num)
    assert sep >= 0.02, (num, sep)
    assert edge >= 0.018, (num, edge)
    assert elem >= 0.008, (num, elem)


def _distinct_xy(num):
    _sep, _edge, _elem, distinct, n = _metrics(num)
    return distinct, n


def test_p222_and_p21212_do_not_collapse():
    d16, n16 = _distinct_xy(16)
    d18, n18 = _distinct_xy(18)
    assert d16 == n16 == 4
    assert d18 == n18 == 4
    # Same multiplicity, different representative, so the panels are not copies.
    assert D.best_general_point(16) != D.best_general_point(18)


def test_ima2_pma_and_i42m_keep_full_orbits():
    assert _distinct_xy(46) == (8, 8)
    assert _distinct_xy(51) == (8, 8)
    assert _distinct_xy(121) == (16, 16)


def test_mirror_parallel_to_page_is_a_split_circle():
    """Pm down b: the two heights share one projected point and one circle."""
    fig, ax = plt.subplots()
    D.general_position_diagram(6, ax=ax, projection="b")
    circles = [p for p in ax.patches if isinstance(p, Circle)]
    assert len(circles) == 1
    plt.close(fig)
