"""Two-pass landmark catalog: subsample, then distances and cosets."""
import json
from pathlib import Path

import numpy as np

from agentsg.cell.landmark_workflow import (
    LandmarkCatalog,
    as_cell_rows,
    build_landmarks,
    main as cli_main,
    reindex_hkl,
    score_cells,
)


IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


class _UnitCell:
    """Stand-in for a cctbx ``unit_cell``."""

    def __init__(self, parameters):
        self._parameters = tuple(parameters)

    def parameters(self):
        return self._parameters


def _family(base, n, seed):
    rng = np.random.default_rng(seed)
    rows = np.tile(np.asarray(base, dtype=np.float64), (n, 1))
    rows[:, 3:] += rng.normal(0.0, 0.02, size=(n, 3))
    return rows


def test_cctbx_parameters_match_rows():
    raw = (40.0, 50.0, 60.0, 90.0, 90.0, 90.0)
    assert np.allclose(as_cell_rows(_UnitCell(raw)), np.asarray(raw).reshape(1, 6))
    assert as_cell_rows([_UnitCell(raw), _UnitCell(raw)]).shape == (2, 6)


def test_subsample_then_score_identity_coset():
    family_a = _family((40.0, 50.0, 60.0, 90.0, 90.0, 90.0), 8, seed=1)
    family_b = _family((70.0, 50.0, 60.0, 90.0, 90.0, 90.0), 8, seed=2)
    cells = np.vstack([family_a, family_b])
    catalog = build_landmarks(
        cells, sigma_deg=0.05, subsample=list(range(0, 6)) + list(range(8, 14)),
    )
    assert catalog.subsample == tuple(list(range(0, 6)) + list(range(8, 14)))
    assert len(catalog.clusters) == 2
    assert catalog.landmarks
    by_cluster = {}
    for mark in catalog.landmarks:
        assert mark.to_center == IDENTITY
        by_cluster.setdefault(mark.cluster_id, set()).add(mark.reference_cell)
    for references in by_cluster.values():
        assert len(references) == 1

    query = family_a[0].copy()
    query[0] += 2.0
    batch = score_cells(catalog, [_UnitCell(tuple(query))])
    hits = batch.queries[0].hits
    assert len(hits) == len(catalog.landmarks)
    assert [hit.key_distance for hit in hits] == sorted(hit.key_distance for hit in hits)
    nearest = hits[0]
    assert nearest.to_center == IDENTITY
    assert nearest.operators
    stored = {mark.landmark_id: mark for mark in catalog.landmarks}[nearest.landmark_id]
    assert nearest.operators == stored.operators
    assert nearest.branches == stored.branches
    assert abs(nearest.reference_cell[0] - 40.0) < abs(nearest.reference_cell[0] - 70.0)
    assert reindex_hkl((1, 2, 3), nearest.to_center) == (1, 2, 3)

    text = catalog.to_json()
    restored = LandmarkCatalog.from_json(text)
    again = score_cells(restored, [query])
    assert again.queries[0].hits[0].to_center == IDENTITY
    assert again.queries[0].hits[0].operators == nearest.operators


def test_symmetry_branches_round_trip_on_cli(tmp_path: Path):
    base = (41.8, 41.8, 233.0, 90.0, 90.0, 120.0)
    cells = _family(base, 6, seed=7)
    csv_path = tmp_path / "cells.csv"
    with csv_path.open("w") as handle:
        handle.write("a,b,c,alpha,beta,gamma\n")
        for row in cells:
            handle.write(",".join(str(float(x)) for x in row) + "\n")
    catalog_path = tmp_path / "landmarks.json"
    cli_main([
        "build", str(csv_path),
        "--sigma-deg", "0.05",
        "--symmetry", "P 63",
        "--subsample", "6",
        "--out", str(catalog_path),
    ])
    payload = json.loads(catalog_path.read_text())
    assert payload["symmetry"] == "P 63"
    assert payload["landmarks"]
    assert any(len(mark["branches"]) == 2 for mark in payload["landmarks"])
    assert all(mark["to_center"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]] for mark in payload["landmarks"])

    query_path = tmp_path / "queries.csv"
    query_path.write_text(
        "a,b,c,alpha,beta,gamma\n"
        "43.8,41.8,233,90,90,120\n"
    )
    score_path = tmp_path / "scores.json"
    cli_main(["score", str(catalog_path), str(query_path), "--out", str(score_path)])
    scored = json.loads(score_path.read_text())
    nearest = scored["queries"][0]["hits"][0]
    assert nearest["to_center"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert len(nearest["branches"]) == 2
    assert nearest["key_distance"] >= 0.0
