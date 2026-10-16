"""Smoke tests for the cell-neighbour examples."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from linear_knn_isomap import linear_knn_isomap  # noqa: E402
from monoclinic_modes import A_MEANS, BETA_MEANS, monoclinic_cloud  # noqa: E402
from monoclinic_landmarks import gaussian_landmarks, pairwise_closure_distances, closure_conorms  # noqa: E402
from primitive_batch import primitive_cells  # noqa: E402
from reindex_cosets import reindex_cosets  # noqa: E402
from agentsg.cell.metric import UnitCell  # noqa: E402
from agentsg.cell.reduction import niggli_reduce  # noqa: E402
from selling_steps import combine_cobs  # noqa: E402


def test_i_centred_volume_halves():
    conv = np.array([[10.0, 12.0, 14.0, 90.0, 90.0, 90.0]])
    by_letter = primitive_cells(conv, ["I"])
    by_symbol = primitive_cells(conv, ["I 2 2 2"])
    assert by_symbol == pytest.approx(by_letter)
    kept = primitive_cells(conv, [None])
    assert kept == pytest.approx(conv)
    v_conv = UnitCell(*conv[0]).volume()
    v_prim = UnitCell(*by_letter[0]).volume()
    assert v_prim == pytest.approx(v_conv / 2.0, rel=1e-9)


def test_repeated_cell_closure_distance_coset_and_embedding():
    cell = (5.0, 6.0, 7.0, 90.0, 90.0, 90.0)
    cells = np.array([cell, cell, cell])
    graph = linear_knn_isomap(cells, k=2, boundary_rel=0.0)
    assert graph["xy"].shape == (3, 2)
    assert graph["indices"].shape == (3, 2)
    assert graph["distances"].shape == (3, 2)
    assert np.all(graph["distances"] < 1e-6)
    for i, row in enumerate(graph["indices"]):
        assert i not in row.tolist()
    result = reindex_cosets(cell, cells[:1])
    assert result["closure_size"] >= 1
    assert result["rows"][0]["coset"]
    assert result["rows"][0]["linear_distance"] == pytest.approx(0.0, abs=1e-6)


def test_niggli_cell_matches_without_selling_reduction():
    """A Niggli cell and an acute cell of the same lattice need no Selling step."""
    acute = (8.0, 6.0, 11.0, 90.0, 70.0, 90.0)
    niggli = tuple(float(x) for x in niggli_reduce(*acute)[0])
    assert niggli != pytest.approx(acute, abs=1e-6)
    cells = np.array([acute, niggli])
    graph = linear_knn_isomap(cells, k=1, boundary_rel=0.0)
    assert graph["primitive"] == pytest.approx(cells)
    assert np.all(graph["distances"] < 1e-6)
    result = reindex_cosets(niggli, cells[:1])
    assert result["reference"] == pytest.approx(niggli)
    assert result["rows"][0]["primitive"] == pytest.approx(acute)
    assert result["rows"][0]["coset"]
    assert result["rows"][0]["linear_distance"] == pytest.approx(0.0, abs=1e-6)
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    acute_selling = result["rows"][0]["selling"]["cob"]
    acute_closure = result["rows"][0]["closure_cobs"]
    assert acute_selling != identity
    assert identity in acute_closure
    assert acute_selling not in acute_closure
    assert combine_cobs(acute_selling, identity) == acute_selling
    assert graph["selling"][0]["cob"] == acute_selling
    assert identity in graph["closure_cobs"][0]
    assert acute_selling not in graph["closure_cobs"][0]


def test_monoclinic_cloud_has_four_modes():
    cells, labels = monoclinic_cloud(4000, seed=0)
    assert cells.shape == (4000, 6)
    assert set(labels.tolist()) == {0, 1, 2, 3}
    for lab in range(4):
        block = cells[labels == lab]
        ia, ib = divmod(lab, 2)
        assert block[:, 0].mean() == pytest.approx(A_MEANS[ia], abs=0.15)
        assert block[:, 4].mean() == pytest.approx(BETA_MEANS[ib], abs=0.1)
        assert block[:, 1].mean() == pytest.approx(40.0)
        assert np.all(np.abs(block[:, 3] - 90.0) < 1e-12)
        assert np.all(np.abs(block[:, 5] - 90.0) < 1e-12)


def test_gaussian_landmarks_follow_covariance_axes():
    rng = np.random.default_rng(0)
    keys = np.zeros((400, 6))
    keys[:, 0] = 3.0 * rng.normal(size=400)
    keys[:, 1] = 0.4 * rng.normal(size=400)
    points, info = gaussian_landmarks(keys, order=3)
    assert info["kept"] == 2
    assert len(points) == 9
    assert np.min(np.linalg.norm(points - keys.mean(axis=0), axis=1)) < 1e-8


def test_gpu_closure_distance_matches_reference():
    pytest.importorskip("cupy")
    from agentsg.cell.canonical import closure_distance

    cells = [
        (30.0, 40.0, 55.0, 90.0, 89.2, 90.0),
        (30.4, 40.0, 55.0, 90.0, 91.1, 90.0),
        (35.2, 40.0, 55.0, 90.0, 89.4, 90.0),
    ]
    got = pairwise_closure_distances([closure_conorms(cell) for cell in cells])
    for i in range(3):
        for j in range(i + 1, 3):
            ref = closure_distance(cells[i], cells[j], boundary_rel=0.0)[1]
            assert got[i, j] == pytest.approx(ref, abs=1e-6)
