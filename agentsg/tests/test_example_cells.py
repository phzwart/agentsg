"""Smoke tests for the cell-neighbour examples."""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from linear_knn_isomap import linear_knn_isomap  # noqa: E402
from smiley_landmarks import EYE_SIGMA, NOSE_SIGMA, sample_smiley, smiley_cells  # noqa: E402
from smiley_workflow import (  # noqa: E402
    _metrics_from_sorted_keys,
    assign_stream,
    draw_stream_keys,
    extract_landmarks,
    niggli_table,
    read_cell_csv,
    score_collection,
    write_cell_csv,
)
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


def test_smiley_eyes_are_one_landmark_and_the_smile_is_a_chain():
    result = sample_smiley()
    assert len(result["clusters"]) == 4
    by_size = {}
    for info in result["clusters"]:
        count = sum(mark.cluster_id == info.cluster_id for mark in result["landmarks"])
        by_size.setdefault(len(info.members), []).append(count)
    assert sorted(by_size) == [80, 360, 500]
    assert all(count >= 1 for count in by_size[80])
    assert by_size[500][0] >= 2
    assert by_size[360][0] > 1
    for info in result["clusters"]:
        pts = np.asarray([result["scores"][i] for i in info.members], dtype=np.float64)
        marks_xy = np.asarray([
            result["scores"][mark.cell_index]
            for mark in result["landmarks"]
            if mark.cluster_id == info.cluster_id
        ], dtype=np.float64)
        dist = np.linalg.norm(pts[:, None, :] - marks_xy[None, :, :], axis=-1).min(axis=1)
        assert float(dist.max()) <= result["radii"][info.cluster_id] + 1e-4
    assert min(NOSE_SIGMA) >= 3.0 * EYE_SIGMA
    assert NOSE_SIGMA[0] != NOSE_SIGMA[1]
    nose = next(info for info in result["clusters"] if len(info.members) == 500)
    nose_xy = result["scores"][list(nose.members)]
    eye_xy = result["scores"][result["labels"] == 0]
    eye_sigma = float(np.std(eye_xy - eye_xy.mean(axis=0)))
    centred = nose_xy - nose_xy.mean(axis=0)
    nose_sigma = np.linalg.svd(centred, compute_uv=False) / np.sqrt(len(nose_xy))
    assert float(nose_sigma.min()) >= 2.5 * eye_sigma
    assert float(nose_sigma.max()) > float(nose_sigma.min())


def test_smiley_workflow_assigns_the_stream_to_the_reference(tmp_path):
    generated, labels = smiley_cells(seed=0)
    rows = niggli_table(generated)
    csv_path = write_cell_csv(tmp_path / "niggli.csv", rows[:2])
    reread, cobs = read_cell_csv(csv_path)
    assert reread[0] == pytest.approx(rows[0][0])
    assert cobs[0] == rows[0][1]

    catalog = extract_landmarks([cell for cell, _cob in rows])
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert len(catalog["components"]) == 4
    for component in catalog["components"]:
        reference = next(mark for mark in component["landmarks"] if mark["is_reference"])
        assert reference["cell_index"] == component["reference_index"]
        assert reference["cell_index"] in component["members"]
        assert identity in reference["closure_cobs"]
        assert len(reference["widened_conorms"]) == 6
        assert all(
            widened >= raw
            for raw, widened in zip(reference["conorms"], reference["widened_conorms"])
        )

        assert reference["to_reference"] == identity
        assert all(mark["to_reference"] is not None for mark in component["landmarks"])
        for mark in component["landmarks"]:
            chosen = sum(
                (mark["to_reference"][i][j] - identity[i][j]) ** 2
                for i in range(3) for j in range(3)
            )
            others = [
                sum((op[i][j] - identity[i][j]) ** 2 for i in range(3) for j in range(3))
                for op in mark["to_reference_coset"]
            ]
            assert chosen == min(others)

    for link in catalog["links"]:
        assert link["operator"] is not None
        chosen = sum(
            (link["operator"][i][j] - identity[i][j]) ** 2
            for i in range(3) for j in range(3)
        )
        others = [
            sum((op[i][j] - identity[i][j]) ** 2 for i in range(3) for j in range(3))
            for op in link["coset"]
        ]
        assert chosen == min(others)

    stream_generated, stream_labels = smiley_cells(seed=1)
    pick = [int(np.flatnonzero(stream_labels == label)[0]) for label in range(4)]
    stream = [cell for cell, _cob in niggli_table(stream_generated[pick])]
    assigned = assign_stream(catalog, stream)
    for item, label in zip(assigned, range(4)):
        component = next(
            entry for entry in catalog["components"]
            if entry["cluster_id"] == item["cluster_id"]
        )
        majority = int(np.bincount(labels[list(component["members"])]).argmax())
        assert majority == label
        assert identity in item["to_reference"]
        assert item["reference_index"] in component["members"]
        assert len(item["hits"]) == sum(len(entry["landmarks"]) for entry in catalog["components"])
        for hit in item["hits"]:
            assert len(hit["operators"]) == len(hit["frobenius"])
            assert hit["frobenius"]
            assert min(hit["frobenius"]) >= 0.0
        same = [hit for hit in item["hits"] if hit["cluster_id"] == item["cluster_id"]]
        other = [hit for hit in item["hits"] if hit["cluster_id"] != item["cluster_id"]]
        assert min(min(hit["frobenius"]) for hit in same) <= min(
            min(hit["frobenius"]) for hit in other
        )


def test_collection_score_matches_frobenius_norm():
    generated, _labels = smiley_cells(seed=0)
    catalog = extract_landmarks([cell for cell, _cob in niggli_table(generated[:40])])
    keys, _part = draw_stream_keys(32, seed=3)
    scored = score_collection(catalog, keys, chunk=16)
    assert scored["n"] == 32
    assert scored["landmarks"] == sum(len(c["landmarks"]) for c in catalog["components"])
    scored = score_collection(catalog, keys, chunk=16)
    assert scored["n"] == 32
    assert scored["closures"] >= 2
    assert 0.0 <= scored["cluster_agreement"] <= 1.0
    mark_metrics = _metrics_from_sorted_keys(np.asarray([
        catalog["ingest"].keys[mark["cell_index"]]
        for mark in scored["records"]
    ], dtype=np.float64))
    gram = _metrics_from_sorted_keys(keys[:1])[0]
    winner = int(scored["best_index"][0])
    change = scored["operators"][int(scored["best_operator"][0])]
    diff = change.T @ gram @ change - mark_metrics[winner]
    assert scored["best_frobenius"][0] == pytest.approx(np.linalg.norm(diff, ord="fro"), abs=1e-5)
    identity = np.linalg.norm(gram - mark_metrics[winner], ord="fro")
    assert scored["best_frobenius"][0] <= identity + 1e-5
