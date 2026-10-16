"""From the smiley ensemble to a reference setting and a stream assignment.

Three passes:

1. Generate the cells, Niggli-reduce each one, and write that reduced cell
   to CSV. The Niggli change of basis is stored beside the cell and is not
   folded into anything later.
2. Landmark extraction. Connected components come from the linear-key
   clusters. Each landmark is a member of the CSV. For that member the pass
   stores a Selling-reduced cell, the conorms widened by the noise floor,
   the one change of basis onto the Selling cell, and the closure changes
   of basis in that reduced basis. Those two changes of basis stay uncombined.
3. A stream of further cells is identified by its nearest landmark. The
   change of basis that is assigned takes the stream cell onto the canonical
   reference of that landmark's component.

The reference of a component is its medoid. Every landmark in the component
is carried onto that Selling cell by the coset operator closest to the
identity, so the component has one setting. Between disconnected components
the stored operator is likewise the coset member closest to ``100,010,001``.

A stream cell keeps the possible coset onto every landmark. For each
operator the Frobenius norm ``||P^T G(cell) P - G(landmark)||_F`` is stored.

Run::

    python agentsg/examples/smiley_workflow.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from agentsg.cell.reduction import niggli_gk
from agentsg.cell.rootform import conorms, noise_floor
from agentsg.cell.serial_index import cluster, ingest, landmarks

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - optional accelerator
    njit = None

try:
    import torch
except ImportError:  # pragma: no cover - optional accelerator
    torch = None

from selling_steps import closure_cobs, combine_coset, to_selling
from smiley_landmarks import EYE_SIGMA, NOSE_SIGMA, SIGMA_DEG, _BASE, smiley_cells

_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_OUT = Path(__file__).resolve().parent / "out"
_CELL_FIELDS = ("a", "b", "c", "alpha", "beta", "gamma")
_COB_FIELDS = tuple(f"n{i}{j}" for i in range(3) for j in range(3))


def _as_cob(matrix):
    if hasattr(matrix, "rows"):
        matrix = matrix.rows
    return tuple(tuple(int(value) for value in row) for row in matrix)


def niggli_table(cells):
    """Niggli cell and its change of basis, one row per input cell."""
    rows = []
    for cell in np.asarray(cells, dtype=np.float64):
        reduced, cob = niggli_gk(tuple(float(value) for value in cell))
        rows.append((tuple(float(value) for value in reduced), _as_cob(cob)))
    return rows


def write_cell_csv(path, rows):
    """Write Niggli parameters and the Niggli change of basis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CELL_FIELDS + _COB_FIELDS)
        writer.writeheader()
        for cell, cob in rows:
            payload = {name: cell[i] for i, name in enumerate(_CELL_FIELDS)}
            flat = [cob[i][j] for i in range(3) for j in range(3)]
            payload.update(zip(_COB_FIELDS, flat))
            writer.writerow(payload)
    return path


def read_cell_csv(path):
    """Niggli cells and changes of basis from :func:`write_cell_csv`."""
    cells = []
    cobs = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            cells.append(tuple(float(row[name]) for name in _CELL_FIELDS))
            cobs.append(tuple(
                tuple(int(row[f"n{i}{j}"]) for j in range(3))
                for i in range(3)
            ))
    return cells, cobs


def _widened_conorms(cell, sigma_deg):
    """Six conorms, and the same six with the noise floor lifted under them."""
    raw_map = conorms(cell)
    raw = tuple(float(raw_map[pair]) for pair in _PAIRS)
    floor = float(noise_floor(cell, sigma_deg))
    widened = tuple(max(value, floor) for value in raw)
    return raw, widened, floor


def selling_record(cell, sigma_deg=SIGMA_DEG):
    """Selling cell, widened conorms, and the two uncombined changes of basis.

    ``to_selling`` is the change of basis from the submitted Niggli cell.
    ``closure_cobs`` is the widened closure, written in the Selling basis.
    """
    selling = to_selling(cell)
    raw, widened, floor = _widened_conorms(selling["reduced"], sigma_deg)
    cobs = tuple(closure_cobs(
        selling["reduced"], boundary_rel=0.0, angle_sigma=sigma_deg,
    ))
    return {
        "submitted": tuple(float(value) for value in cell),
        "selling_cell": selling["reduced"],
        "to_selling": selling["cob"],
        "conorms": raw,
        "widened_conorms": widened,
        "floor": floor,
        "closure_cobs": cobs,
    }


def extract_landmarks(cells, sigma_deg=SIGMA_DEG):
    """Connected landmarks, each with a Selling record, and one reference each.

    The reference of a component is the medoid. It is a landmark. The other
    landmarks are not given an operator to the reference or to each other.
    """
    ingested = ingest(cells, sigma_deg=sigma_deg)
    infos, _svd = cluster(ingested)
    marks = landmarks(infos, ingested)
    by_cluster = {}
    for mark in marks:
        by_cluster.setdefault(mark.cluster_id, []).append(mark)
    components = []
    for info in infos:
        group = by_cluster[info.cluster_id]
        reference = next(mark for mark in group if mark.cell_index == info.medoid)
        recorded = []
        for mark in group:
            record = selling_record(ingested.cells[mark.cell_index], sigma_deg)
            record["cell_index"] = mark.cell_index
            record["is_reference"] = mark.cell_index == info.medoid
            recorded.append(record)
        components.append({
            "cluster_id": info.cluster_id,
            "members": info.members,
            "reference_index": info.medoid,
            "landmarks": recorded,
        })
    return {
        "ingest": ingested,
        "components": components,
        "links": anchor_settings(components),
    }


def _reference_record(component):
    return next(mark for mark in component["landmarks"] if mark["is_reference"])


def _det(operator):
    (a, b, c), (d, e, f), (g, h, i) = operator
    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def _mul(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _inv(operator):
    det = _det(operator)
    if det not in (1, -1):
        raise ValueError("operator is not unimodular")
    (a, b, c), (d, e, f), (g, h, i) = operator
    adj = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    return tuple(tuple(adj[r][c] * det for c in range(3)) for r in range(3))


def _distance_to_identity(operator):
    return float(np.linalg.norm(np.asarray(operator, dtype=np.float64) - np.eye(3)))


def _closest_to_identity(operators):
    """The coset member nearest ``100,010,001``."""
    return min(operators, key=lambda cob: (_distance_to_identity(cob), cob))


def _selling_view(record):
    return {
        "cell": record["submitted"],
        "reduced": record["selling_cell"],
        "cob": record["to_selling"],
    }


def _coset(query, reference, band_rel=1e-3):
    return tuple(combine_coset(
        _selling_view(query), query["closure_cobs"],
        _selling_view(reference), reference["closure_cobs"],
        band_rel=band_rel,
    ))


def _transformed(metric, operator):
    change = np.asarray(operator, dtype=np.float64)
    gram = np.asarray(metric, dtype=np.float64)
    return change.T @ gram @ change


def frobenius_norm(cell, operator, landmark_cell):
    """``||P^T G(cell) P - G(landmark)||_F``."""
    from agentsg.cell.metric import metric_tensor

    diff = _transformed(metric_tensor(cell), operator) - np.asarray(
        metric_tensor(landmark_cell), dtype=np.float64,
    )
    return float(np.linalg.norm(diff, ord="fro"))


def anchor_settings(components, band_rel=1e-3):
    """One setting per component, and the near-identity link between them.

    Inside a component every landmark stores the coset onto the medoid and
    the member of that coset closest to the identity. Between components the
    stored operator is the coset member closest to the identity.
    """
    for component in components:
        reference = _reference_record(component)
        for mark in component["landmarks"]:
            if mark["is_reference"]:
                mark["to_reference_coset"] = (_IDENTITY,)
                mark["to_reference"] = _IDENTITY
                continue
            coset = _coset(mark, reference, band_rel)
            mark["to_reference_coset"] = coset
            mark["to_reference"] = _closest_to_identity(coset) if coset else None
    links = []
    owned = [(component, _reference_record(component)) for component in components]
    for left_component, left in owned:
        for right_component, right in owned:
            if left_component["cluster_id"] == right_component["cluster_id"]:
                continue
            coset = _coset(left, right, band_rel)
            chosen = _closest_to_identity(coset) if coset else None
            links.append({
                "from_cluster": left_component["cluster_id"],
                "to_cluster": right_component["cluster_id"],
                "coset": coset,
                "operator": chosen,
                "distance_to_identity": (
                    None if chosen is None else _distance_to_identity(chosen)
                ),
            })
    return links


def _link_operator(links, source, target):
    if source == target:
        return _IDENTITY
    for link in links:
        if link["from_cluster"] == source and link["to_cluster"] == target:
            return link["operator"]
    raise KeyError(f"no link from component {source} to {target}")


def _compose_coset(stream_ops, link, landmark_ops):
    found = []
    seen = set()
    for stream_op in stream_ops:
        for landmark_op in landmark_ops:
            operator = _mul(_mul(stream_op, link), _inv(landmark_op))
            if operator in seen:
                continue
            seen.add(operator)
            found.append(operator)
    return tuple(found)


def assign_stream(catalog, stream_cells, sigma_deg=SIGMA_DEG):
    """Index each stream cell onto every landmark.

    The nearest landmark selects the component. The coset onto that
    component's reference is composed with each landmark's operator, and
    with the near-identity link when the landmark belongs to another
    component. Each operator stores the Frobenius norm of the reindexed
    metric against the landmark.
    """
    from agentsg.cell.rootform import sorted_linear_key

    catalog_keys = np.asarray(catalog["ingest"].keys, dtype=np.float64)
    stream_rows = [selling_record(cell, sigma_deg) for cell in stream_cells]
    stream_keys = np.asarray(
        [sorted_linear_key(cell) for cell in stream_cells], dtype=np.float64,
    )
    landmark_index = []
    landmark_keys = []
    owners = []
    catalog_marks = []
    for component in catalog["components"]:
        for mark in component["landmarks"]:
            landmark_index.append(mark["cell_index"])
            landmark_keys.append(catalog_keys[mark["cell_index"]])
            owners.append(component)
            catalog_marks.append((component["cluster_id"], mark))
    landmark_keys = np.asarray(landmark_keys, dtype=np.float64)
    distance = np.linalg.norm(
        stream_keys[:, None, :] - landmark_keys[None, :, :], axis=-1,
    )
    links = catalog["links"]
    assigned = []
    for row, record in enumerate(stream_rows):
        slot = int(np.argmin(distance[row]))
        component = owners[slot]
        reference = _reference_record(component)
        to_reference = _coset(record, reference)
        hits = []
        for cluster_id, mark in catalog_marks:
            link = _link_operator(links, component["cluster_id"], cluster_id)
            if not to_reference or link is None or not mark["to_reference_coset"]:
                operators = ()
            else:
                operators = _compose_coset(to_reference, link, mark["to_reference_coset"])
            norms = tuple(
                frobenius_norm(record["submitted"], operator, mark["submitted"])
                for operator in operators
            )
            hits.append({
                "cell_index": mark["cell_index"],
                "cluster_id": cluster_id,
                "operators": operators,
                "frobenius": norms,
            })
        assigned.append({
            "stream_index": row,
            "landmark_index": landmark_index[slot],
            "cluster_id": component["cluster_id"],
            "reference_index": component["reference_index"],
            "key_distance": float(distance[row, slot]),
            "to_selling": record["to_selling"],
            "to_reference": to_reference,
            "hits": tuple(hits),
        })
    return assigned


def _metrics_from_sorted_keys(keys):
    """Gram matrices of sorted linear keys, shape ``(n, 3, 3)``."""
    conorm = np.asarray(keys, dtype=np.float64) * np.asarray(keys, dtype=np.float64).sum(axis=1, keepdims=True)
    metric = np.zeros((len(conorm), 3, 3), dtype=np.float64)
    metric[:, 0, 0] = conorm[:, 0] + conorm[:, 3] + conorm[:, 4]
    metric[:, 1, 1] = conorm[:, 1] + conorm[:, 3] + conorm[:, 5]
    metric[:, 2, 2] = conorm[:, 2] + conorm[:, 4] + conorm[:, 5]
    metric[:, 0, 1] = metric[:, 1, 0] = -conorm[:, 3]
    metric[:, 0, 2] = metric[:, 2, 0] = -conorm[:, 4]
    metric[:, 1, 2] = metric[:, 2, 1] = -conorm[:, 5]
    return metric


def draw_stream_keys(n, seed=2):
    """``n`` sorted linear keys from the same four-part face as the catalog."""
    rng = np.random.default_rng(seed)
    weights = np.array([80, 80, 360, 500], dtype=np.float64)
    weights /= weights.sum()
    part = rng.choice(4, size=n, p=weights)
    plane = np.empty((n, 2), dtype=np.float64)
    for label, centre in ((0, (-0.90, 0.45)), (1, (0.90, 0.45))):
        mask = part == label
        plane[mask] = rng.normal(0.0, EYE_SIGMA, size=(int(mask.sum()), 2)) + centre
    mask = part == 2
    count = int(mask.sum())
    t = rng.uniform(-1.0, 1.0, size=count)
    tangent = np.column_stack([np.full(count, 1.15), 0.56 * t])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    plane[mask] = (
        np.column_stack([1.15 * t, -0.55 + 0.28 * t ** 2])
        + normal * rng.normal(0.0, 0.02, size=count)[:, None]
    )
    mask = part == 3
    count = int(mask.sum())
    theta = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=count))
    plane[mask, 0] = 2.0 * NOSE_SIGMA[0] * radius * np.cos(theta)
    plane[mask, 1] = 2.0 * NOSE_SIGMA[1] * radius * np.sin(theta) + 0.12
    keys = _BASE + np.column_stack([plane, np.zeros((n, 4), dtype=np.float64)])
    return keys, part


def _closure_bank(catalog):
    """Widened closure of every landmark, as operators in the submitted basis."""
    from selling_steps import combine_cobs

    records = []
    clusters = []
    keys = []
    listed = []
    index = {}
    table = []
    for component in catalog["components"]:
        for mark in component["landmarks"]:
            records.append(mark)
            clusters.append(component["cluster_id"])
            keys.append(catalog["ingest"].keys[mark["cell_index"]])
            ids = []
            for cob in mark["closure_cobs"]:
                full = combine_cobs(mark["to_selling"], cob)
                if full not in index:
                    index[full] = len(table)
                    table.append(full)
                ids.append(index[full])
            listed.append(np.asarray(ids, dtype=np.int64))
    operators = np.stack([np.asarray(cob, dtype=np.float64) for cob in table])
    member = np.zeros((len(records), len(table)), dtype=bool)
    for row, ids in enumerate(listed):
        member[row, ids] = True
    landmark_keys = np.asarray(keys, dtype=np.float64)
    return {
        "records": records,
        "clusters": np.asarray(clusters, dtype=np.int64),
        "keys": landmark_keys,
        "metrics": _metrics_from_sorted_keys(landmark_keys),
        "operators": operators,
        "member": member,
    }


if njit is not None:
    @njit(parallel=True)
    def _closure_best(grams, metrics, operators, member):
        """Min squared Frobenius of P^T G P - H over the widened closure."""
        n = grams.shape[0]
        landmarks = metrics.shape[0]
        closures = operators.shape[0]
        best_f = np.empty(n)
        best_i = np.empty(n, np.int64)
        best_k = np.empty(n, np.int64)
        for i in prange(n):
            bf = 1.0e300
            bi = 0
            bk = 0
            g00 = grams[i, 0, 0]
            g01 = grams[i, 0, 1]
            g02 = grams[i, 0, 2]
            g10 = grams[i, 1, 0]
            g11 = grams[i, 1, 1]
            g12 = grams[i, 1, 2]
            g20 = grams[i, 2, 0]
            g21 = grams[i, 2, 1]
            g22 = grams[i, 2, 2]
            for k in range(closures):
                p00 = operators[k, 0, 0]
                p01 = operators[k, 0, 1]
                p02 = operators[k, 0, 2]
                p10 = operators[k, 1, 0]
                p11 = operators[k, 1, 1]
                p12 = operators[k, 1, 2]
                p20 = operators[k, 2, 0]
                p21 = operators[k, 2, 1]
                p22 = operators[k, 2, 2]
                t00 = g00 * p00 + g01 * p10 + g02 * p20
                t01 = g00 * p01 + g01 * p11 + g02 * p21
                t02 = g00 * p02 + g01 * p12 + g02 * p22
                t10 = g10 * p00 + g11 * p10 + g12 * p20
                t11 = g10 * p01 + g11 * p11 + g12 * p21
                t12 = g10 * p02 + g11 * p12 + g12 * p22
                t20 = g20 * p00 + g21 * p10 + g22 * p20
                t21 = g20 * p01 + g21 * p11 + g22 * p21
                t22 = g20 * p02 + g21 * p12 + g22 * p22
                r00 = p00 * t00 + p10 * t10 + p20 * t20
                r01 = p00 * t01 + p10 * t11 + p20 * t21
                r02 = p00 * t02 + p10 * t12 + p20 * t22
                r10 = p01 * t00 + p11 * t10 + p21 * t20
                r11 = p01 * t01 + p11 * t11 + p21 * t21
                r12 = p01 * t02 + p11 * t12 + p21 * t22
                r20 = p02 * t00 + p12 * t10 + p22 * t20
                r21 = p02 * t01 + p12 * t11 + p22 * t21
                r22 = p02 * t02 + p12 * t12 + p22 * t22
                for j in range(landmarks):
                    if not member[j, k]:
                        continue
                    d00 = r00 - metrics[j, 0, 0]
                    d01 = r01 - metrics[j, 0, 1]
                    d02 = r02 - metrics[j, 0, 2]
                    d10 = r10 - metrics[j, 1, 0]
                    d11 = r11 - metrics[j, 1, 1]
                    d12 = r12 - metrics[j, 1, 2]
                    d20 = r20 - metrics[j, 2, 0]
                    d21 = r21 - metrics[j, 2, 1]
                    d22 = r22 - metrics[j, 2, 2]
                    resid = (
                        d00 * d00 + d01 * d01 + d02 * d02
                        + d10 * d10 + d11 * d11 + d12 * d12
                        + d20 * d20 + d21 * d21 + d22 * d22
                    )
                    if resid < bf:
                        bf = resid
                        bi = j
                        bk = k
            best_f[i] = bf
            best_i[i] = bi
            best_k[i] = bk
        return best_f, best_i, best_k
else:
    _closure_best = None


def _torch_device():
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.device("cuda")


def _grams_torch(keys):
    conorm = keys * keys.sum(dim=1, keepdim=True)
    metric = torch.zeros(keys.shape[0], 3, 3, dtype=keys.dtype, device=keys.device)
    metric[:, 0, 0] = conorm[:, 0] + conorm[:, 3] + conorm[:, 4]
    metric[:, 1, 1] = conorm[:, 1] + conorm[:, 3] + conorm[:, 5]
    metric[:, 2, 2] = conorm[:, 2] + conorm[:, 4] + conorm[:, 5]
    metric[:, 0, 1] = metric[:, 1, 0] = -conorm[:, 3]
    metric[:, 0, 2] = metric[:, 2, 0] = -conorm[:, 4]
    metric[:, 1, 2] = metric[:, 2, 1] = -conorm[:, 5]
    return metric


class _GpuClosure:
    """Landmark keys, metrics, and widened-closure operators on one GPU."""

    def __init__(self, bank):
        self.device = _torch_device()
        self.keys = torch.as_tensor(bank["keys"], dtype=torch.float64, device=self.device)
        self.metrics = torch.as_tensor(bank["metrics"], dtype=torch.float64, device=self.device)
        self.operators = torch.as_tensor(bank["operators"], dtype=torch.float64, device=self.device)
        self.member = torch.as_tensor(bank["member"], dtype=torch.bool, device=self.device)

    def score(self, block):
        keys = torch.as_tensor(block, dtype=torch.float64, device=self.device)
        distance = torch.cdist(keys, self.keys)
        key_winner = torch.argmin(distance, dim=1)
        key_distance = distance.gather(1, key_winner[:, None]).squeeze(1)
        grams = _grams_torch(keys)
        # P^T G P for every closure operator and every cell: (K, N, 3, 3).
        changed = torch.einsum("kba,nbc,kcd->knad", self.operators, grams, self.operators)
        cell_norm = (changed * changed).sum(dim=(-1, -2))
        inner = torch.einsum("knij,lij->knl", changed, self.metrics)
        landmark_norm = (self.metrics * self.metrics).sum(dim=(-1, -2))
        residual = cell_norm[:, :, None] + landmark_norm[None, None, :] - 2.0 * inner
        residual = residual.permute(1, 2, 0)
        residual = residual.masked_fill(~self.member[None, :, :], float("inf"))
        best, which = torch.min(residual, dim=-1)
        winner = torch.argmin(best, dim=1)
        rows = torch.arange(keys.shape[0], device=self.device)
        return (
            key_distance.detach().cpu().numpy(),
            key_winner.detach().cpu().numpy(),
            best[rows, winner].detach().cpu().numpy(),
            winner.detach().cpu().numpy(),
            which[rows, winner].detach().cpu().numpy(),
        )


def score_collection(catalog, keys, chunk=20000, backend="auto"):
    """Best widened-closure reindex of every cell against every landmark.

    For each landmark the loop tries every operator in its widened Selling
    closure and keeps the smallest Frobenius norm of the reindexed metric.
    The linear key is scored as well. A cell's key-nearest landmark names a
    cluster; that cluster is compared with the cluster of the best closure
    score, which is the gate check.

    ``backend`` is ``"torch"`` for the GPU match, ``"numba"`` for the
    threaded CPU kernel, or ``"numpy"``. ``"auto"`` uses the GPU when
    CUDA is available.
    """
    if backend == "auto":
        if _torch_device() is not None:
            backend = "torch"
        elif _closure_best is not None:
            backend = "numba"
        else:
            backend = "numpy"
    bank = _closure_bank(catalog)
    gpu = _GpuClosure(bank) if backend == "torch" else None
    landmark_keys = bank["keys"]
    landmark_metrics = bank["metrics"]
    operators = bank["operators"]
    member = bank["member"]
    clusters = bank["clusters"]
    n = len(keys)
    if backend == "numba" and _closure_best is not None and n:
        _closure_best(
            _metrics_from_sorted_keys(np.asarray(keys[:1], dtype=np.float64)),
            landmark_metrics, operators, member,
        )
    if gpu is not None and n:
        gpu.score(np.asarray(keys[:1], dtype=np.float64))
    landmark_norm = np.sum(landmark_metrics * landmark_metrics, axis=(1, 2))
    best_distance = np.empty(n, dtype=np.float64)
    key_index = np.empty(n, dtype=np.int64)
    best_frobenius = np.empty(n, dtype=np.float64)
    best_index = np.empty(n, dtype=np.int64)
    best_operator = np.empty(n, dtype=np.int64)
    started = time.perf_counter()
    for begin in range(0, n, chunk):
        block = np.asarray(keys[begin:begin + chunk], dtype=np.float64)
        count = len(block)
        rows = np.arange(count)
        if gpu is not None:
            key_distance, key_winner, residual, winner, operator = gpu.score(block)
            key_index[begin:begin + count] = key_winner
            best_distance[begin:begin + count] = key_distance
        else:
            distance = np.linalg.norm(block[:, None, :] - landmark_keys[None, :, :], axis=-1)
            key_winner = np.argmin(distance, axis=1)
            key_index[begin:begin + count] = key_winner
            best_distance[begin:begin + count] = distance[rows, key_winner]
            grams = _metrics_from_sorted_keys(block)
            if backend == "numba":
                residual, winner, operator = _closure_best(
                    grams, landmark_metrics, operators, member,
                )
            else:
                best = np.full((count, len(landmark_keys)), np.inf, dtype=np.float64)
                which = np.full((count, len(landmark_keys)), -1, dtype=np.int64)
                for slot in range(len(operators)):
                    change = operators[slot]
                    changed = np.einsum("ba,nbc,cd->nad", change, grams, change, optimize=True)
                    cell_norm = np.sum(changed * changed, axis=(1, 2))
                    inner = np.einsum("nij,lij->nl", changed, landmark_metrics, optimize=True)
                    score = cell_norm[:, None] + landmark_norm[None, :] - 2.0 * inner
                    score[:, ~member[:, slot]] = np.inf
                    better = score < best
                    best[better] = score[better]
                    which[better] = slot
                winner = np.argmin(best, axis=1)
                residual = best[rows, winner]
                operator = which[rows, winner]
        best_frobenius[begin:begin + count] = np.sqrt(np.maximum(residual, 0.0))
        best_index[begin:begin + count] = winner
        best_operator[begin:begin + count] = operator
    elapsed = time.perf_counter() - started
    key_cluster = clusters[key_index]
    closure_cluster = clusters[best_index]
    identity = np.eye(3)
    chosen = operators[best_operator]
    return {
        "n": n,
        "landmarks": len(bank["records"]),
        "closures": len(operators),
        "backend": backend,
        "seconds": elapsed,
        "per_cell_us": 1e6 * elapsed / n if n else 0.0,
        "cells_per_second": n / elapsed if elapsed else 0.0,
        "best_distance": best_distance,
        "best_frobenius": best_frobenius,
        "best_index": best_index,
        "key_index": key_index,
        "key_cluster": key_cluster,
        "closure_cluster": closure_cluster,
        "cluster_agreement": float(np.mean(key_cluster == closure_cluster)) if n else 0.0,
        "identity_alignment": np.all(np.abs(chosen - identity) < 1e-8, axis=(1, 2)),
        "operators": operators,
        "best_operator": best_operator,
        "records": bank["records"],
        "clusters": clusters,
    }


def score_summary(result):
    """Percentiles of the key gate and the best closure score."""
    distance = result["best_distance"]
    frobenius = result["best_frobenius"]
    percent = (50, 90, 99)
    lines = [
        f"scored {result['n']} cells against {result['landmarks']} landmarks "
        f"and {result['closures']} closure operators on {result['backend']}",
        f"seconds {result['seconds']:.3f}",
        f"per_cell_us {result['per_cell_us']:.3f}",
        f"cells_per_second {result['cells_per_second']:.3g}",
        (
            "best key distance Å "
            + " ".join(f"p{p} {np.percentile(distance, p):.4g}" for p in percent)
        ),
        (
            "best closure frobenius "
            + " ".join(f"p{p} {np.percentile(frobenius, p):.4g}" for p in percent)
        ),
        f"identity alignment {int(result['identity_alignment'].sum())}/{result['n']}",
        (
            "key cluster agrees with closure cluster "
            f"{int(round(result['cluster_agreement'] * result['n']))}/{result['n']}"
        ),
    ]
    return "\n".join(lines)


def _fmt_cob(cob):
    return ",".join("".join(str(int(value)) for value in row) for row in cob)


def report(catalog, assigned):
    """Text summary of the anchored settings and the stream Frobenius norms."""
    lines = [f"components {len(catalog['components'])}"]
    for component in catalog["components"]:
        reference = _reference_record(component)
        aligned = sum(mark["to_reference"] == _IDENTITY for mark in component["landmarks"])
        lines.append(
            f"component {component['cluster_id']} "
            f"members {len(component['members'])} "
            f"landmarks {len(component['landmarks'])} "
            f"reference row {component['reference_index']} "
            f"to_reference {_fmt_cob(reference['to_reference'])} "
            f"identity {aligned}/{len(component['landmarks'])}"
        )
    for link in catalog["links"]:
        if link["from_cluster"] > link["to_cluster"]:
            continue
        lines.append(
            f"link {link['from_cluster']}->{link['to_cluster']} "
            f"{_fmt_cob(link['operator'])} "
            f"distance {link['distance_to_identity']:.4g}"
        )
    lines.append(f"stream {len(assigned)}")
    for item in assigned:
        same = [
            hit for hit in item["hits"]
            if hit["cluster_id"] == item["cluster_id"] and hit["frobenius"]
        ]
        best = min(same, key=lambda hit: min(hit["frobenius"]))
        lines.append(
            f"stream {item['stream_index']} "
            f"landmark {item['landmark_index']} "
            f"component {item['cluster_id']} "
            f"reference {item['reference_index']} "
            f"distance {item['key_distance']:.4g} "
            f"to_reference {len(item['to_reference'])} "
            f"identity {_IDENTITY in item['to_reference']} "
            f"best landmark {best['cell_index']} "
            f"frobenius {min(best['frobenius']):.4g}"
        )
    return "\n".join(lines)


def main():
    generated, _labels = smiley_cells(seed=0)
    catalog_rows = niggli_table(generated)
    catalog_path = write_cell_csv(_OUT / "smiley_niggli.csv", catalog_rows)
    stream_generated, stream_labels = smiley_cells(seed=1)
    pick = []
    for label in range(4):
        hits = np.flatnonzero(stream_labels == label)[:4]
        pick.extend(int(index) for index in hits)
    stream_rows = niggli_table(stream_generated[pick])
    stream_path = write_cell_csv(_OUT / "smiley_stream.csv", stream_rows)
    cells, _cobs = read_cell_csv(catalog_path)
    catalog = extract_landmarks(cells)
    stream_cells, _stream_cobs = read_cell_csv(stream_path)
    assigned = assign_stream(catalog, stream_cells)
    text = report(catalog, assigned)
    print(catalog_path)
    print(stream_path)
    print(text)
    keys, _part = draw_stream_keys(10**6, seed=2)
    scored = score_collection(catalog, keys)
    print(score_summary(scored))
    return catalog, assigned, scored


if __name__ == "__main__":
    main()
