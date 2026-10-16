"""Seeded checks for the serial-indexing pipeline."""
import json
from pathlib import Path

import numpy as np

from agentsg.cell.canonical import reindexing_via_canonical
from agentsg.cell.metric import params_from_metric
from agentsg.cell.serial_index import LANDMARK_EXCLUSION_FACTOR, _eta_key, cluster, ingest, landmarks, run
from agentsg.cell.serial_index_cli import main as cli_main


def _cell(vals):
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    products = {pairs[i]: float(vals[i]) for i in range(6)}

    def value(i, j):
        return products[(i, j) if i < j else (j, i)]

    metric = [[0.0] * 3 for _ in range(3)]
    for i in range(1, 4):
        metric[i - 1][i - 1] = sum(value(i, j) for j in range(4) if j != i)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            metric[i - 1][j - 1] = metric[j - 1][i - 1] = -value(i, j)
    return params_from_metric(metric)


def _family(base, n, sigma_deg, seed, edge=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        cell = list(base)
        for index in range(3):
            cell[index] *= 1.0 + float(rng.normal(0.0, edge))
        for index in range(3, 6):
            cell[index] += float(rng.normal(0.0, sigma_deg))
        rows.append(cell)
    return np.asarray(rows, dtype=np.float64)


def test_single_chamber_is_exact_and_unambiguous():
    base = _cell((10, 14, 18, 22, 26, 30))
    cells = _family(base, 500, sigma_deg=0.01, seed=1)
    state = run(cells, sigma_deg=0.01)
    assert state.certify.certified
    assert len(state.clusters) == 1
    info = state.clusters[0]
    assert info.isometric
    assert state.certify.discrepancies == ()
    medoid = state.ingest.cells[info.medoid]
    expected = len(reindexing_via_canonical(medoid, medoid, boundary_rel=0.0))
    sizes = {len(item.operators) for item in state.cosets.self_cosets}
    assert sizes == {expected}
    assert expected == 2
    assert all(item.status == "unambiguous" for item in state.assignments)
    json.loads(state.to_json())


def test_near_tie_is_aligned_once():
    rows = np.array([
        _cell((10, 10.5, 20, 30, 40, 50)),
        _cell((11, 10, 20, 30, 40, 50)),
    ])
    state = run(rows, sigma_deg=0.01)
    assert any(info.align_once for info in state.clusters)
    assert any(item.bin == "UNDER" and item.remediated for item in state.certify.discrepancies)
    assert state.certify.certified


def test_fibre_collision_splits_into_two_forms():
    form_a = _cell((1, 2, 3, 4, 5, 6))
    form_b = _cell((1, 2, 3, 4, 6, 5))
    rows = np.vstack([
        _family(form_a, 16, sigma_deg=0.002, seed=4, edge=0.002),
        _family(form_b, 16, sigma_deg=0.002, seed=5, edge=0.002),
    ])
    rows[:16, 0] *= 1.002
    rows[16:, 1] *= 1.002
    state = run(rows, sigma_deg=0.002)
    assert any(item.bin == "COLLISION" for item in state.certify.discrepancies)
    assert len(state.clusters) == 2
    assert state.certify.certified
    forms = []
    for cluster in state.clusters:
        edges = [state.ingest.cells[i][0] for i in cluster.members]
        forms.append(float(np.median(edges)))
    assert abs(forms[0] - forms[1]) > 0.05


def test_pseudo_monoclinic_branches():
    base = (10.0, 15.0, 22.0, 90.0, 90.2, 90.0)
    cells = _family(base, 12, sigma_deg=0.05, seed=6)
    state = run(cells, sigma_deg=0.5, symmetry="P 1 21 1")
    assert state.cosets.self_cosets
    assert len(state.cosets.self_cosets[0].operators) == 8
    assert state.cosets.self_cosets[0].group_closed
    assert len(state.cosets.self_cosets[0].branches) == 2


def test_p63_widened_coset_and_branches():
    base = (41.8, 41.8, 233.0, 90.0, 90.0, 120.0)
    cells = _family(base, 8, sigma_deg=0.02, seed=7)
    state = run(cells, sigma_deg=0.05, symmetry="P 63")
    assert len(state.cosets.self_cosets[0].operators) == 24
    assert state.cosets.self_cosets[0].group_closed
    assert len(state.cosets.self_cosets[0].branches) == 2


def test_poisson_disc_covers_a_line_of_cells():
    edges = np.linspace(30.0, 50.0, 40)
    cells = np.column_stack([
        edges,
        np.full(40, 40.0),
        np.full(40, 55.0),
        np.full(40, 90.0),
        np.full(40, 90.0),
        np.full(40, 90.0),
    ])
    ingested = ingest(cells, sigma_deg=0.05)
    infos, _svd = cluster(ingested)
    assert len(infos) == 1
    marks = landmarks(infos, ingested)
    assert len({mark.reference_cell for mark in marks}) == 1
    members = set(infos[0].members)
    assert all(mark.cell_index in members for mark in marks)
    assert all(mark.cell == ingested.cells[mark.cell_index] for mark in marks)
    assert all(mark.reference_cell == ingested.cells[infos[0].medoid] for mark in marks)
    assert all(mark.to_center == ((1, 0, 0), (0, 1, 0), (0, 0, 1)) for mark in marks)
    radius = LANDMARK_EXCLUSION_FACTOR * _eta_key(ingested.cells[infos[0].medoid], 0.05, 0.0)
    chosen = np.asarray([mark.key for mark in marks], dtype=np.float64)
    members = np.asarray([ingested.keys[i] for i in infos[0].members], dtype=np.float64)
    separation = np.linalg.norm(chosen[:, None, :] - chosen[None, :, :], axis=-1)
    np.fill_diagonal(separation, np.inf)
    assert len(marks) > 1
    assert float(separation.min()) >= radius - 1e-6
    nearest = np.linalg.norm(members[:, None, :] - chosen[None, :, :], axis=-1).min(axis=1)
    assert float(nearest.max()) <= radius + 1e-6


def test_reindex_to_cluster_center_is_identity():
    base = (40.0, 50.0, 60.0, 90.0, 90.0, 90.0)
    cells = _family(base, 20, sigma_deg=0.05, seed=9)
    cells[0, 0] = base[0] + 2.0
    cells[1, 1] = base[1] - 2.0
    state = run(cells, sigma_deg=0.05)
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert len({mark.reference_cell for mark in state.landmarks}) == 1
    assigned = [item for item in state.assignments if item.status == "unambiguous"]
    assert assigned
    assert all(item.operator == identity for item in assigned)
    assert all(mark.to_center == identity for mark in state.landmarks)


def test_cli_json_round_trip(tmp_path: Path):
    base = _cell((10, 14, 18, 22, 26, 30))
    cells = _family(base, 30, sigma_deg=0.01, seed=8)
    csv_path = tmp_path / "cells.csv"
    with csv_path.open("w") as handle:
        handle.write("a,b,c,alpha,beta,gamma\n")
        for row in cells:
            handle.write(",".join(str(float(x)) for x in row) + "\n")
    out = tmp_path / "report.json"
    cli_main([str(csv_path), "--sigma-deg", "0.01", "--out", str(out)])
    payload = json.loads(out.read_text())
    assert payload["certify"]["certified"] is True
    assert payload["clusters"][0]["eta"] > 0.0
