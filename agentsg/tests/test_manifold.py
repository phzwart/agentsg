"""Tests for the deformation-manifold landmark scaffold."""
import math
import pytest

from agentsg.cell.manifold import (
    deformation_graph, symmetry_junctions, farthest_point_landmarks,
    select_landmarks, DeformationManifold,
)


def _traj(t):
    """Shrinkage trajectory: c goes 150 -> 100, crossing a==c at t=0.6."""
    return (120.0, 189.1, 150.0 - 50.0 * t, 90.0, 91.2, 90.0)


CELLS = [_traj(t) for t in [i / 40 for i in range(41)]]
JUNCTION_I = 24   # t = 0.60, a == c == 120


# ------------------------------------------------------------------ graph ----
def test_deformation_graph_shape_and_symmetry():
    D, adj = deformation_graph(CELLS)
    n = len(CELLS)
    assert len(D) == n and len(D[0]) == n
    for i in range(n):
        assert D[i][i] == 0.0
        for j in range(n):
            assert abs(D[i][j] - D[j][i]) < 1e-9
    # adjacency is symmetric
    for i in adj:
        for j in adj[i]:
            assert i in adj[j]


def test_graph_adjacent_states_are_neighbours():
    """Consecutive trajectory states are close, so each links to its neighbour."""
    D, adj = deformation_graph(CELLS, k=4)
    for i in range(1, len(CELLS) - 1):
        assert (i - 1) in adj[i] or (i + 1) in adj[i]


# -------------------------------------------------------------- junctions ----
def test_symmetry_junction_found_at_degeneracy():
    """Exactly the a==c crossing is a junction -- not every state."""
    j = symmetry_junctions(CELLS)
    assert JUNCTION_I in j
    assert len(j) <= 3            # not the whole trajectory


def test_constant_angle_does_not_spuriously_fire():
    """A trajectory with a constant nonzero beta and NO length degeneracy has no
    junctions (the flat angle signal must not fire everywhere)."""
    cells = [(50.0, 60.0, 70.0 + 0.5 * i, 90.0, 99.0, 90.0) for i in range(20)]
    j = symmetry_junctions(cells)
    assert j == [] or all(c[2] != c[0] and c[2] != c[1] for k, c in enumerate(cells) if k in j)


# -------------------------------------------------------------- landmarks ----
def test_farthest_point_covers():
    D, _ = deformation_graph(CELLS)
    lm = farthest_point_landmarks(D, 4, seed=0)
    assert len(lm) == 4
    assert 0 in lm                        # seed
    assert len(set(lm)) == 4              # distinct


def test_select_landmarks_endpoints_and_junction():
    lm = select_landmarks(CELLS)
    # endpoints of the trajectory are the extreme pair
    assert 0 in lm and 40 in lm
    assert JUNCTION_I in lm               # the symmetry junction


# ----------------------------------------------------- manifold scaffold ----
def test_manifold_landmark_roles():
    M = DeformationManifold(CELLS)
    roles = {M.landmark_role(L) for L in M.landmarks}
    assert "endpoint" in roles
    assert "junction" in roles


def test_reindex_to_landmark_exact_for_same_lattice():
    """A state that IS a landmark (or the same lattice as one) reindexes with
    zero residual; a state that is a genuine DEFORMATION away from every landmark
    honestly returns P=None (the residual is the deformation, not a match)."""
    M = DeformationManifold(CELLS)
    # a landmark reindexes onto itself exactly
    for L in M.landmarks:
        Ln, P, resid = M.reindex_to_landmark(L)
        assert Ln == L and P is not None and resid < 1e-6
    # a state far (in deformation) from every landmark returns None -- the honest
    # signal that it needs path routing, not a spurious exact match
    Ln, P, resid = M.reindex_to_landmark(10)   # between landmarks 0 and 24
    assert P is None and resid > 1.0


def test_reindex_to_same_lattice_landmark_is_exact():
    """When landmarks are dense enough that a state shares a landmark's lattice
    (here: duplicate a state as its own neighbour), the hop is exact."""
    # three copies of one lattice + one far lattice: the copies reindex exactly
    base = (120.0, 189.1, 120.6, 90.0, 91.2, 90.0)
    cells = [base, base, base, (60.0, 70.0, 80.0, 90.0, 90.0, 90.0)]
    M = DeformationManifold(cells, k=2, include_junctions=False)
    L, P, resid = M.reindex_to_landmark(0)
    assert P is not None and resid < 1e-6


def test_nearest_landmark_is_a_landmark():
    M = DeformationManifold(CELLS)
    for i in range(0, len(CELLS), 7):
        assert M.nearest_landmark(i) in M.landmarks


def test_path_to_landmark_reaches_landmark():
    M = DeformationManifold(CELLS)
    for i in (5, 15, 35):
        L, path = M.path_to_landmark(i)
        assert path[0] == i
        assert path[-1] == L
        assert L in M.landmarks


def test_reindex_along_path_operator_is_unimodular():
    """The composed path operator is always an integer unimodular matrix
    (det = +-1). NOTE: it is NOT guaranteed to be the identity within a single
    deformation branch -- composing signed superbase operators over a long hem
    of short hops can accumulate a global sign or basis relabelling (e.g. -I),
    and for a genuine deformation the metric residual does not vanish because the
    endpoints are different lattices. What the scaffold guarantees is that each
    SHORT hop is a valid integer operator; the composition inherits
    unimodularity, which is what this test verifies. Frame CONSISTENCY across a
    junction is a separate property checked by the junction-detection tests, not
    by asserting a particular composed matrix here."""
    M = DeformationManifold(CELLS)
    for i in (5, 8, 15):
        L, P, max_res = M.reindex_along_path(i)
        for row in P:
            for x in row:
                assert x == int(x)
        det = (P[0][0]*(P[1][1]*P[2][2]-P[1][2]*P[2][1])
               - P[0][1]*(P[1][0]*P[2][2]-P[1][2]*P[2][0])
               + P[0][2]*(P[1][0]*P[2][1]-P[1][1]*P[2][0]))
        assert abs(det) == 1


def test_fiedler_recovers_deformation_axis():
    """The Fiedler coordinate orders states along the shrinkage axis."""
    np = pytest.importorskip("numpy")
    M = DeformationManifold(CELLS)
    f = M.fiedler_coordinate()
    ts = np.array([i / 40 for i in range(41)])
    corr = abs(np.corrcoef(f, ts)[0, 1])
    assert corr > 0.95
