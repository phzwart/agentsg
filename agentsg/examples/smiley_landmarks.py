"""A smiley face in the SVD of the sorted linear key, and its landmarks.

Two tight isotropic clouds sit above a curved band, with a fat nose between
them. The nose is anisotropic, and each of its axes is at least three times
the eye sigma. All four are drawn in two components of the sorted linear key,
with the other four held fixed and far enough apart that the sort never
flips. The SVD plane is that drawing, because the remaining key components
do not move.

Landmarks are the Poisson-disc sample already used by the serial index.
The exclusion radius is one times the medoid's linear-key noise. At the
declared ``sigma_deg`` of 0.05°, that ball covers an eye, steps along the
smile, and steps down the long axis of the nose.

Run::

    python agentsg/examples/smiley_landmarks.py

The same cells, Niggli-reduced and then assigned, are the workflow in
``smiley_workflow.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from agentsg.cell.metric import params_from_metric
from agentsg.cell.serial_index import LANDMARK_EXCLUSION_FACTOR, _eta_key, cluster, ingest, landmarks

_BASE = np.array([2.0, 5.0, 8.0, 11.0, 14.0, 17.0], dtype=np.float64)
_PAIR_INDEX = {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5}
SIGMA_DEG = 0.05
EYE_SIGMA = 0.035
# Both nose axes are at least three eye-sigmas. The vertical axis is longer.
NOSE_SIGMA = (3.0 * EYE_SIGMA, 5.0 * EYE_SIGMA)


def cell_from_sorted_key(key):
    """Unit cell whose sorted linear key is ``key``.

    ``p = q * sum(q)`` undoes ``q = p / sqrt(sum p)``. The six numbers must
    already be in increasing order.
    """
    q = np.asarray(key, dtype=np.float64)
    conorm = q * float(q.sum())

    def pair(i, j):
        a, b = (i, j) if i < j else (j, i)
        return float(conorm[_PAIR_INDEX[(a, b)]])

    metric = np.zeros((3, 3), dtype=np.float64)
    for i in range(1, 4):
        metric[i - 1, i - 1] = sum(pair(i, j) for j in range(4) if j != i)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            metric[i - 1, j - 1] = metric[j - 1, i - 1] = -pair(i, j)
    return params_from_metric(metric)


def _eyes(rng):
    left = rng.normal(0.0, EYE_SIGMA, size=(80, 2)) + np.array([-0.90, 0.45])
    right = rng.normal(0.0, EYE_SIGMA, size=(80, 2)) + np.array([0.90, 0.45])
    return np.vstack([left, right])


def _nose(rng):
    """A filled ellipse. Both standard deviations are at least three eye-sigmas.

    A Gaussian with those widths grows tails that touch the eyes and the
    smile. The uniform ellipse keeps the fat shape and stays one cluster.
    For a uniform ellipse the semi-axis is twice the standard deviation.
    """
    count = 500
    theta = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=count))
    xy = np.column_stack([
        2.0 * NOSE_SIGMA[0] * radius * np.cos(theta),
        2.0 * NOSE_SIGMA[1] * radius * np.sin(theta),
    ])
    return xy + np.array([0.0, 0.12])


def _smile(rng):
    """A parabolic banana, thin compared with the exclusion radius."""
    t = np.linspace(-1.0, 1.0, 360)
    curve = np.column_stack([1.15 * t, -0.55 + 0.28 * t ** 2])
    tangent = np.column_stack([np.full(t.shape, 1.15), 0.56 * t])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return curve + normal * rng.normal(0.0, 0.02, size=t.shape)[:, None]


def smiley_cells(seed=0):
    """Cells, and labels 0..3 for left eye, right eye, smile, and nose.

    The first two sorted-key components carry the drawing. A point ``(x, y)``
    has key ``(2+x, 5+y, 8, 11, 14, 17)``.
    """
    rng = np.random.default_rng(seed)
    eyes = _eyes(rng)
    smile = _smile(rng)
    nose = _nose(rng)
    plane = np.vstack([eyes, smile, nose])
    labels = np.array(
        [0] * 80 + [1] * 80 + [2] * len(smile) + [3] * len(nose), dtype=np.int64,
    )
    keys = _BASE + np.column_stack([plane, np.zeros((len(plane), 4))])
    cells = np.asarray([cell_from_sorted_key(key) for key in keys], dtype=np.float64)
    return cells, labels


def _orient(scores, labels):
    """Put the eyes above the smile, and the left eye on the left."""
    xy = np.array(scores, dtype=np.float64, copy=True)
    eyes = labels < 2
    if xy[eyes, 1].mean() < xy[~eyes, 1].mean():
        xy[:, 1] *= -1.0
    if xy[labels == 0, 0].mean() > xy[labels == 1, 0].mean():
        xy[:, 0] *= -1.0
    return xy


def sample_smiley(sigma_deg=SIGMA_DEG, seed=0):
    """Cluster the face and place Poisson-disc landmarks."""
    cells, labels = smiley_cells(seed)
    ingested = ingest(cells, sigma_deg=sigma_deg)
    infos, _svd = cluster(ingested)
    marks = landmarks(infos, ingested)
    keys = np.asarray(ingested.keys, dtype=np.float64)
    centred = keys - keys.mean(axis=0)
    _u, values, vt = np.linalg.svd(centred, full_matrices=False)
    scores = _orient(centred @ vt[:2].T, labels)
    spread = float(np.sum(values ** 2)) or 1.0
    fraction = (values ** 2 / spread)[:2]
    radii = {
        info.cluster_id: LANDMARK_EXCLUSION_FACTOR * _eta_key(
            ingested.cells[info.medoid], ingested.sigma_deg, ingested.sigma_len,
        )
        for info in infos
    }
    return {
        "cells": cells,
        "labels": labels,
        "scores": scores,
        "variance_frac": fraction,
        "clusters": infos,
        "landmarks": marks,
        "radii": radii,
    }


def write_png(result, path):
    """The face in the SVD plane, landmarks, and their exclusion discs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = result["scores"]
    labels = result["labels"]
    fraction = result["variance_frac"]
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    colours = {0: "#2c3e50", 1: "#2c3e50", 2: "#d35400", 3: "#117a65"}
    names = {0: "left eye", 1: "right eye", 2: "smile", 3: "nose"}
    for label in (2, 3, 0, 1):
        mask = labels == label
        ax.scatter(
            scores[mask, 0], scores[mask, 1],
            s=16, c=colours[label], alpha=0.55, linewidths=0,
            label=names[label], zorder=2,
        )
    labelled = False
    for mark in result["landmarks"]:
        xy = scores[mark.cell_index]
        radius = result["radii"][mark.cluster_id]
        disc = plt.Circle(
            xy, radius, facecolor="#1a1a1a", edgecolor="#1a1a1a",
            alpha=0.08, lw=0.6, zorder=3,
        )
        ax.add_patch(disc)
        ax.scatter(
            [xy[0]], [xy[1]], s=36, c="white", edgecolors="#1a1a1a",
            linewidths=1.0, zorder=4,
            label=None if labelled else "landmark",
        )
        labelled = True
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"PC1 ({100.0 * fraction[0]:.1f}% of linear-key variance, Å)")
    ax.set_ylabel(f"PC2 ({100.0 * fraction[1]:.1f}% of linear-key variance, Å)")
    ax.set_title("Poisson-disc landmarks on a linear-key smiley")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    result = sample_smiley()
    counts = {}
    for info in result["clusters"]:
        counts[info.cluster_id] = sum(
            mark.cluster_id == info.cluster_id for mark in result["landmarks"]
        )
    print(f"cells {len(result['cells'])}")
    print(f"clusters {len(result['clusters'])}")
    for info in result["clusters"]:
        radius = result["radii"][info.cluster_id]
        print(
            f"cluster {info.cluster_id} size {len(info.members)} "
            f"landmarks {counts[info.cluster_id]} exclusion {radius:.4g} Å"
        )
    path = write_png(result, Path(__file__).resolve().parent / "out" / "smiley_landmarks.png")
    print(path)
    return result


if __name__ == "__main__":
    main()
