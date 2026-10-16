"""Sample two-pass landmark workflow.

Pass 1 clusters a subsample. Pass 2 scores the remaining cells: the sorted
linear-key distance to every landmark, and that landmark's reindexing coset.
Landmarks may be different cells. The operator onto the cluster center is
``100,010,001``.

cctbx unit cells are accepted anywhere an ``(n, 6)`` array is::

    from agentsg.cell.landmark_workflow import build_landmarks, score_cells, reindex_hkl

    catalog = build_landmarks(unit_cells, sigma_deg=0.05, symmetry="P 1 21 1", subsample=200)
    scored = score_cells(catalog, later_unit_cells)
    nearest = scored.queries[0].hits[0]
    print(reindex_hkl((1, 0, 0), nearest.to_center))

The same two steps from CSV::

    python -m agentsg.cell.landmark_workflow build cells.csv --sigma-deg 0.05 --subsample 200 --out landmarks.json
    python -m agentsg.cell.landmark_workflow score landmarks.json queries.csv --out scores.json

Run::

    python agentsg/examples/landmark_two_pass.py
"""
from __future__ import annotations

import numpy as np

from agentsg.cell.landmark_workflow import (
    build_landmarks,
    catalog_summary,
    reindex_hkl,
    score_cells,
    score_summary,
)


def _cloud(base, n, seed):
    rng = np.random.default_rng(seed)
    rows = np.tile(np.asarray(base, dtype=np.float64), (n, 1))
    rows[:, 3:] += rng.normal(0.0, 0.05, size=(n, 3))
    return rows


def main():
    family_a = _cloud((40.0, 50.0, 60.0, 90.0, 90.0, 90.0), 10, seed=1)
    family_b = _cloud((55.0, 50.0, 60.0, 90.0, 90.0, 90.0), 10, seed=2)
    cells = np.vstack([family_a, family_b])
    # Eight rows from each lattice. A random draw can drop one lattice.
    catalog = build_landmarks(
        cells, sigma_deg=0.05, subsample=list(range(0, 8)) + list(range(10, 18)),
    )
    print(catalog_summary(catalog))
    held = [index for index in range(len(cells)) if index not in set(catalog.subsample)]
    # A cell two ångström off the first family still uses the identity.
    shifted = cells[held[0]].copy()
    shifted[0] += 2.0
    batch = score_cells(catalog, np.vstack([cells[held[:4]], shifted.reshape(1, 6)]))
    print(score_summary(batch))
    nearest = batch.queries[0].hits[0]
    print("hkl", reindex_hkl((1, 2, 3), nearest.to_center))


if __name__ == "__main__":
    main()
