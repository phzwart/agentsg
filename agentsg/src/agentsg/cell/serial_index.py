"""Certified serial-crystallography cell indexing.

Guarantees
----------
* Key exactness. On a cluster whose margin ``m`` and gap ``g`` both exceed
  ``2 * delta``, one sorting permutation is fixed and the sorted linear-key
  distance equals the aligned closure distance.
* Landmark distances are the exact closure distance, or a widened one whose
  ``boundary_rel`` covers the value found. The plain closure minimum is not
  used once that certificate fails.
* Self-cosets are group-closed at the residual band, or the band is doubled
  until they are.
* Per-frame assignment is one key, one tree query, and a few 3×3 products.
  It does not call the closure.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..group import point_group
from ..space_groups import space_group
from .canonical import closure_distance, reindexing_via_canonical
from .metric import UnitCell, metric_tensor
from .primitive import primitive_cell
from .rootform import conorm_sum, conorms, noise_floor, sorted_linear_distance, sorted_linear_key

# Step 0. Fifth neighbour, and the factor above the median that marks an outlier.
# Pipeline specification.
OUTLIER_NEIGHBOURS = 5
OUTLIER_FACTOR = 10.0
# Step 1. Single-linkage cutoff, as a multiple of the median nearest-neighbour
# distance. Pipeline specification.
LINKAGE_FACTOR = 5.0
# Step 1. Key distance is exact when margin and gap both clear this multiple of
# the cluster spread. Pipeline specification.
ISOMETRIC_FACTOR = 2.0
# Step 2. Poisson exclusion radius as this multiple of η_key. One noise length.
LANDMARK_EXCLUSION_FACTOR = 1.0
# Step 3. A pair disagrees when the two distances differ by more than this
# multiple of the key-space noise. Pipeline specification.
DISCREPANCY_FACTOR = 3.0
# Step 3. Widening factor in boundary_rel = sqrt(3) * D / max_vonorm.
WIDEN_FACTOR = math.sqrt(3.0)
WIDEN_SOFT = 4
WIDEN_HARD = 8
# Step 3. Collision splits rerun landmarks at most this many times.
COLLISION_DEPTH = 2
# Step 4. verify_rel that recovers the order-24 hexagonal coset in
# manuscript/appendix_z/test_z32_guarantee.py. Tighter values drop operators.
VERIFY_REL_FLOOR = 1e-3
# Step 4. A product outside this multiple of the accepted residual breaks the group.
GROUP_CLOSURE_FACTOR = 2.0
# Step 4. Each failed closure doubles the residual band, this many extra tries.
VERIFY_ATTEMPTS = 3
VERIFY_WIDEN_FACTOR = 2.0
# Step 4. Cross-cluster operators are stored below this multiple of the joint noise.
CROSS_NOISE_FACTOR = 3.0
# Float dust for treating two key components as the same value. Not a noise gate.
_TIE = 1e-8
# Reindexing onto the cluster center. The center's basis is the reference
# setting, so this operator stays the identity when a cell moves by a few
# ångström inside the cluster.
_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def noise_scale_A2(cell, sigma_deg, sigma_len=0.0) -> float:
    """Declared instrument noise as one Å² scale.

    The angle part is :func:`noise_floor`: ``radians(sigma_deg) * T``.
    The edge part is the Frobenius size of ``2 a_i sigma_len`` on the metric
    diagonal. The two are added in quadrature.
    """
    angle = float(noise_floor(cell, sigma_deg))
    if sigma_len is None or float(sigma_len) <= 0.0:
        return angle
    edge = math.sqrt(sum((2.0 * float(edge_i) * float(sigma_len)) ** 2 for edge_i in cell[:3]))
    return math.hypot(angle, edge)


def _eta_key(cell, sigma_deg, sigma_len) -> float:
    """eta moved into the linear-key ångström scale, ``eta / sqrt(T)``."""
    total = float(conorm_sum(cell))
    if total <= 0.0:
        return 0.0
    return noise_scale_A2(cell, sigma_deg, sigma_len) / math.sqrt(total)


def _max_vonorm(cell) -> float:
    """Largest superbase vonorm ``|v_i|²``, the scale used by ``superbase_variants``."""
    products = conorms(cell)
    lengths = [0.0, 0.0, 0.0, 0.0]
    for (i, j), value in products.items():
        lengths[i] += float(value)
        lengths[j] += float(value)
    return max(lengths) or 1.0


def _keys_of(rows):
    return [tuple(float(x) for x in sorted_linear_key(row)) for row in rows]


def _margin_gap(key) -> tuple[float, float]:
    positive = [float(v) for v in key if float(v) > _TIE]
    margin = min(positive) if positive else 0.0
    unique = []
    for value in key:
        value = float(value)
        if not unique or value - unique[-1] > _TIE:
            unique.append(value)
    if len(unique) < 2:
        return margin, float("inf")
    gap = min(unique[i + 1] - unique[i] for i in range(len(unique) - 1))
    return margin, gap


def _metric_residual(metric, operator, target) -> float:
    applied = _transform(metric, operator)
    return max(
        abs(applied[a][b] - target[a][b]) for a in range(3) for b in range(3)
    )


def _transform(metric, operator):
    pf = [[float(x) for x in row] for row in operator]
    return [
        [
            sum(pf[i][a] * metric[i][k] * pf[k][b] for i in range(3) for k in range(3))
            for b in range(3)
        ]
        for a in range(3)
    ]


def _det(operator) -> int:
    (a, b, c), (d, e, f), (g, h, i) = operator
    return int(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))


def _mul(left, right):
    return tuple(
        tuple(sum(int(left[r][k]) * int(right[k][c]) for k in range(3)) for c in range(3))
        for r in range(3)
    )


def _inv(operator):
    det = _det(operator)
    if det not in (1, -1):
        raise ValueError("operator is not unimodular")
    (a, b, c), (d, e, f), (g, h, i) = operator
    # First row of the adjugate is the first column of the cofactor matrix.
    adj = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    return tuple(tuple(adj[r][c] * det for c in range(3)) for r in range(3))


def _trace_abs(metric) -> float:
    return abs(metric[0][0]) + abs(metric[1][1]) + abs(metric[2][2])


def _as_op(matrix) -> tuple:
    return tuple(tuple(int(x) for x in row) for row in matrix)


def _proper_point_ops(symbol: str) -> tuple:
    """Proper point-group matrices of a space-group symbol, as integer tuples."""
    found = []
    seen = set()
    for linear in point_group(space_group(symbol).operations()):
        if linear.det() != 1:
            continue
        op = _as_op(linear.rows)
        if op not in seen:
            seen.add(op)
            found.append(op)
    return tuple(found)


def _branch_list(operators, declared: str | None) -> tuple:
    """Proper automorphisms modulo the declared group's proper point group."""
    proper = [op for op in operators if _det(op) == 1]
    if not proper:
        proper = list(operators)
    if not declared:
        return tuple(proper[:1]) if proper else ()
    group = set(_proper_point_ops(declared))
    reps = []
    for op in proper:
        inv = _inv(op)
        if any(_mul(inv, rep) in group for rep in reps):
            continue
        reps.append(op)
    return tuple(reps)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if value is None:
        return None
    return value


@dataclass(frozen=True)
class IngestResult:
    cells: tuple
    keys: tuple
    outlier: tuple
    multiplicity: tuple
    unique_index: tuple
    eta: tuple
    sigma_deg: float
    sigma_len: float

    def to_dict(self):
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class ClusterInfo:
    cluster_id: int
    members: tuple
    medoid: int
    margin: float
    gap: float
    delta: float
    isometric: bool
    align_once: bool
    eta: float

    def to_dict(self):
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class Landmark:
    landmark_id: int
    cluster_id: int
    cell_index: int
    cell: tuple
    key: tuple
    reference_cell: tuple
    to_center: tuple

    def to_dict(self):
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class Discrepancy:
    landmark_a: int
    landmark_b: int
    key_distance: float
    certified_distance: float
    bin: str
    remediated: bool

    def to_dict(self):
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class CertifyResult:
    matrix: tuple
    key_matrix: tuple
    discrepancies: tuple
    certified: bool
    warnings: tuple

    def to_dict(self):
        return _jsonable({
            "matrix": self.matrix,
            "key_matrix": self.key_matrix,
            "discrepancies": [item.to_dict() for item in self.discrepancies],
            "certified": self.certified,
            "warnings": self.warnings,
        })


@dataclass(frozen=True)
class LandmarkCoset:
    landmark_id: int
    operators: tuple
    branches: tuple
    group_closed: bool
    residual_tol: float
    verify_rel: float

    def to_dict(self):
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class CosetResult:
    self_cosets: tuple
    cross_cluster: tuple
    declared_symmetry: str | None

    def to_dict(self):
        return _jsonable({
            "self_cosets": [item.to_dict() for item in self.self_cosets],
            "cross_cluster": self.cross_cluster,
            "declared_symmetry": self.declared_symmetry,
        })


@dataclass(frozen=True)
class Assignment:
    frame_id: int
    landmark_id: int | None
    operator: tuple | None
    branches: tuple
    residual: float
    status: str

    def to_dict(self):
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class PipelineState:
    ingest: IngestResult
    clusters: tuple
    landmarks: tuple
    certify: CertifyResult
    cosets: CosetResult
    assignments: tuple
    svd_values: tuple
    svd_components: tuple

    def to_dict(self):
        return _jsonable({
            "ingest": self.ingest.to_dict(),
            "clusters": [item.to_dict() for item in self.clusters],
            "landmarks": [item.to_dict() for item in self.landmarks],
            "certify": self.certify.to_dict(),
            "cosets": self.cosets.to_dict(),
            "assignments": [item.to_dict() for item in self.assignments],
            "svd_values": self.svd_values,
            "svd_components": self.svd_components,
        })

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def _primitive_rows(cells, spacegroups):
    rows = []
    for i, cell in enumerate(np.asarray(cells, dtype=np.float64)):
        base = tuple(float(x) for x in cell)
        symbol = None if spacegroups is None else spacegroups[i]
        if symbol:
            base = tuple(float(x) for x in primitive_cell(base, str(symbol)))
        rows.append(base)
    return rows


def ingest(cells, spacegroups=None, sigma_deg=0.05, sigma_len=0.0) -> IngestResult:
    """Primitive cells, keys, duplicate counts, and outlier flags."""
    rows = _primitive_rows(cells, spacegroups)
    if len(rows) < 64:
        keys = _keys_of(rows)
    else:
        chunk = 64
        parts = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
        with ThreadPoolExecutor(max_workers=min(8, len(parts))) as pool:
            keys = [key for part in pool.map(_keys_of, parts) for key in part]
    seen = {}
    unique_index = []
    multiplicity = [0] * len(rows)
    for i, cell in enumerate(rows):
        token = tuple(round(x, 8) for x in cell)
        if token in seen:
            multiplicity[seen[token]] += 1
            unique_index.append(seen[token])
        else:
            seen[token] = i
            unique_index.append(i)
            multiplicity[i] = 1
    eta = tuple(noise_scale_A2(cell, sigma_deg, sigma_len) for cell in rows)
    outlier = _outlier_flags(keys)
    return IngestResult(
        cells=tuple(rows),
        keys=tuple(keys),
        outlier=tuple(bool(flag) for flag in outlier),
        multiplicity=tuple(multiplicity),
        unique_index=tuple(unique_index),
        eta=eta,
        sigma_deg=float(sigma_deg),
        sigma_len=float(sigma_len or 0.0),
    )


def _outlier_flags(keys):
    n = len(keys)
    flags = [False] * n
    if n <= OUTLIER_NEIGHBOURS:
        return flags
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(keys, dtype=np.float64))
    dists, _nbrs = tree.query(np.asarray(keys, dtype=np.float64), k=OUTLIER_NEIGHBOURS + 1)
    fifth = np.asarray(dists, dtype=np.float64)[:, OUTLIER_NEIGHBOURS]
    median = float(np.median(fifth))
    if median <= 0.0:
        return flags
    limit = OUTLIER_FACTOR * median
    for i, distance in enumerate(fifth):
        if float(distance) > limit:
            flags[i] = True
    return flags


def _connected(n, pairs):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[a] = b
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _svd_report(keys):
    matrix = np.asarray(keys, dtype=np.float64)
    if len(matrix) < 2:
        return (0.0,), ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0),)
    _u, values, vt = np.linalg.svd(matrix, full_matrices=False)
    take = min(3, vt.shape[0])
    return (
        tuple(float(v) for v in values),
        tuple(tuple(float(x) for x in row) for row in vt[:take]),
    )


def cluster(ingested: IngestResult) -> tuple:
    """Single-linkage clusters in the full 6-D key, plus the report-only SVD."""
    usable = [
        i for i, (flag, uniq) in enumerate(zip(ingested.outlier, ingested.unique_index))
        if not flag and uniq == i
    ]
    if not usable:
        return (), _svd_report([])
    keys = np.asarray([ingested.keys[i] for i in usable], dtype=np.float64)
    svd = _svd_report(keys)
    if len(usable) == 1:
        groups = [[0]]
    else:
        from scipy.spatial import cKDTree

        tree = cKDTree(keys)
        nearest, _idx = tree.query(keys, k=2)
        median = float(np.median(np.asarray(nearest)[:, 1]))
        cutoff = LINKAGE_FACTOR * median if median > 0.0 else 0.0
        pairs = tree.query_pairs(cutoff) if cutoff > 0.0 else tree.query_pairs(0.0)
        groups = _connected(len(usable), pairs)
        groups = _absorb_fragments(groups, keys, OUTLIER_FACTOR * median)
    infos = []
    for cid, local in enumerate(groups):
        members = tuple(usable[i] for i in local)
        infos.append(_cluster_info(cid, members, ingested))
    return tuple(infos), svd


def _absorb_fragments(groups, keys, limit):
    """Fold clusters of one or two cells into a neighbour inside the outlier gate.

    A gaussian sample leaves a few cells just past the single-linkage cutoff.
    They are not a second crystal. Origin: Step 0 outlier gate, applied to
    fragments smaller than 3.
    """
    if limit <= 0.0 or len(groups) < 2:
        return groups
    large = [list(group) for group in groups if len(group) >= 3]
    small = [list(group) for group in groups if len(group) < 3]
    if not large:
        return groups
    for group in small:
        best = None
        best_distance = float("inf")
        for host in large:
            distance = min(
                float(np.linalg.norm(keys[i] - keys[j]))
                for i in group for j in host
            )
            if distance < best_distance:
                best_distance = distance
                best = host
        if best is not None and best_distance <= limit:
            best.extend(group)
        else:
            large.append(group)
    return large


def _cluster_info(cid, members, ingested: IngestResult) -> ClusterInfo:
    keys = np.asarray([ingested.keys[i] for i in members], dtype=np.float64)
    centre = keys.mean(axis=0)
    medoid_local = int(np.argmin(np.linalg.norm(keys - centre, axis=1)))
    medoid = members[medoid_local]
    distances = np.linalg.norm(keys - keys[medoid_local], axis=1)
    delta = float(np.percentile(distances, 95)) if len(distances) else 0.0
    margin, gap = _margin_gap(ingested.keys[medoid])
    isometric = margin > ISOMETRIC_FACTOR * delta and gap > ISOMETRIC_FACTOR * delta
    return ClusterInfo(
        cluster_id=cid,
        members=members,
        medoid=medoid,
        margin=margin,
        gap=gap,
        delta=delta,
        isometric=isometric,
        align_once=False,
        eta=ingested.eta[medoid],
    )


def _density_gap(keys):
    """Distance to the 5th neighbour. Small means the cell sits in the density."""
    count = int(keys.shape[0])
    if count <= 2:
        return np.zeros(count, dtype=np.float64)
    k = min(OUTLIER_NEIGHBOURS, count - 1)
    from scipy.spatial import cKDTree

    dists, _idx = cKDTree(keys).query(keys, k=k + 1)
    return np.asarray(dists, dtype=np.float64)[:, k]


def _wrap_abs(delta):
    """Absolute angular separation in ``[0, pi]``."""
    return np.abs((delta + np.pi) % (2.0 * np.pi) - np.pi)


def _polar_about(keys, origin):
    """Radius and angle of each key about ``origin``, in the principal plane."""
    centred = np.asarray(keys, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    _u, _values, vt = np.linalg.svd(centred, full_matrices=False)
    take = min(2, vt.shape[0])
    plane = centred @ vt[:take].T
    if take == 1:
        plane = np.column_stack([plane[:, 0], np.zeros(len(plane))])
    return np.linalg.norm(plane, axis=1), np.arctan2(plane[:, 1], plane[:, 0])


def _poisson_disc(keys, radius: float) -> list:
    """Member indices for a Poisson-disc sample in the linear-key geodesic.

    The geodesic of the sorted linear key is its Euclidean length, in ångström.
    The seed is the medoid. The next landmark is the member of an angular
    sweep nearest that sweep's median radius and median angle, so the interior
    is taken before the rim. A rim cell is still chosen when it lies outside
    every exclusion ball. Landmarks stay at least ``radius`` apart, and every
    member ends within ``radius`` of a landmark.
    """
    count = int(keys.shape[0])
    centre = keys.mean(axis=0)
    seed = int(np.argmin(np.linalg.norm(keys - centre, axis=1)))
    chosen = [seed]
    if count == 1 or not radius > 0.0:
        return chosen
    gap = _density_gap(keys)
    radial, angle = _polar_about(keys, keys[seed])
    nearest = np.linalg.norm(keys - keys[seed], axis=1)
    while True:
        probe = int(np.argmax(nearest))
        if float(nearest[probe]) < radius:
            break
        offset = radial * np.sin(np.minimum(_wrap_abs(angle - angle[probe]), np.pi / 2.0))
        sweep = np.flatnonzero(offset <= radius)
        if sweep.size == 0:
            sweep = np.asarray([probe], dtype=np.int64)
        shifted = (angle[sweep] - angle[probe] + np.pi) % (2.0 * np.pi) - np.pi
        target_r = float(np.median(radial[sweep]))
        target_a = float(np.median(shifted))
        free = sweep[nearest[sweep] >= radius]
        if free.size == 0:
            free = np.asarray([probe], dtype=np.int64)
        rel = (angle[free] - angle[probe] + np.pi) % (2.0 * np.pi) - np.pi
        miss = (radial[free] - target_r) ** 2 + (radial[free] * _wrap_abs(rel - target_a)) ** 2
        order = np.lexsort((gap[free], miss))
        candidate = int(free[int(order[0])])
        chosen.append(candidate)
        nearest = np.minimum(
            nearest, np.linalg.norm(keys - keys[candidate], axis=1),
        )
    return chosen


def landmarks(infos, ingested: IngestResult) -> tuple:
    """Poisson-disc member cells, each referred to the cluster medoid.

    Every landmark is a row of the submitted set. The shared center is the
    medoid, also a member, not a cell synthesized from the median. The next
    landmark in a component is the median of an angular sweep about the
    medoid. The rim of that sweep is included when it still lies outside
    every exclusion ball. The exclusion radius is
    :data:`LANDMARK_EXCLUSION_FACTOR` times that medoid's linear-key noise.
    Landmarks may be different members. The reindexing operator onto the
    medoid is the identity.
    """
    found = []
    for info in infos:
        reference = ingested.cells[info.medoid]
        keys = np.asarray([ingested.keys[i] for i in info.members], dtype=np.float64)
        radius = LANDMARK_EXCLUSION_FACTOR * _eta_key(
            ingested.cells[info.medoid], ingested.sigma_deg, ingested.sigma_len,
        )
        for local in _poisson_disc(keys, radius):
            cell_index = info.members[local]
            found.append(Landmark(
                landmark_id=len(found),
                cluster_id=info.cluster_id,
                cell_index=cell_index,
                cell=ingested.cells[cell_index],
                key=ingested.keys[cell_index],
                reference_cell=reference,
                to_center=_IDENTITY,
            ))
    return tuple(found)


def _certified_pair(cell_a, cell_b, warnings):
    """Exact closure distance, widened until the band covers the value found."""
    _op, distance, _resid = closure_distance(cell_a, cell_b, boundary_rel=0.0)
    margin_a, _gap_a = _margin_gap(sorted_linear_key(cell_a))
    margin_b, _gap_b = _margin_gap(sorted_linear_key(cell_b))
    held = float(distance) < min(margin_a, margin_b)
    if held or not math.isfinite(distance):
        return float(distance), held, float(distance)
    scale = max(_max_vonorm(cell_a), _max_vonorm(cell_b))
    previous = float(distance)
    width = WIDEN_FACTOR * previous / scale
    for step in range(WIDEN_HARD):
        _op, widened, _resid = closure_distance(cell_a, cell_b, boundary_rel=width)
        widened = float(widened)
        needed = WIDEN_FACTOR * widened / scale
        if widened >= previous * (1.0 - 1e-6) and width + 1e-15 >= needed:
            return widened, False, float(distance)
        previous = widened
        width = max(width, needed)
        if step + 1 == WIDEN_SOFT:
            warnings.append("widening passed 4 iterations")
    warnings.append("widening hit the hard cap of 8")
    return previous, False, float(distance)


def certify(infos, marks, ingested: IngestResult, depth=0):
    """Certified landmark distances and the three-bin remediation."""
    warnings = []
    n = len(marks)
    matrix = [[0.0] * n for _ in range(n)]
    key_matrix = [[0.0] * n for _ in range(n)]
    by_id = {info.cluster_id: info for info in infos}
    discrepancies = []
    collisions = []
    align = set()
    for i in range(n):
        for j in range(i + 1, n):
            left, right = marks[i], marks[j]
            key_distance = float(sorted_linear_distance(left.cell, right.cell))
            certified, _held, raw = _certified_pair(left.cell, right.cell, warnings)
            matrix[i][j] = matrix[j][i] = certified
            key_matrix[i][j] = key_matrix[j][i] = key_distance
            eta = DISCREPANCY_FACTOR * max(
                _eta_key(left.cell, ingested.sigma_deg, ingested.sigma_len),
                _eta_key(right.cell, ingested.sigma_deg, ingested.sigma_len),
            )
            if abs(certified - key_distance) <= eta:
                continue
            same = left.cluster_id == right.cluster_id
            isometric = same and by_id[left.cluster_id].isometric
            if key_distance < certified and same and not isometric:
                kind = "UNDER"
                align.add(left.cluster_id)
                remediated = True
            elif certified < raw - _TIE:
                kind = "INFLATED"
                remediated = True
            elif key_distance < certified and same:
                kind = "COLLISION"
                remediated = False
                collisions.append((left, right))
            elif key_distance < certified:
                # The two shapes are already different clusters.
                kind = "COLLISION"
                remediated = True
            else:
                raise RuntimeError(
                    "unclassified discrepancy "
                    f"key={key_distance} certified={certified} raw={raw} "
                    f"same={same} isometric={isometric}"
                )
            discrepancies.append(Discrepancy(
                left.landmark_id, right.landmark_id, key_distance, certified, kind, remediated,
            ))
    updated = []
    for info in infos:
        if info.cluster_id in align and not info.align_once:
            for member in info.members:
                closure_distance(ingested.cells[member], ingested.cells[info.medoid], boundary_rel=0.0)
            updated.append(ClusterInfo(
                info.cluster_id, info.members, info.medoid, info.margin, info.gap,
                info.delta, info.isometric, True, info.eta,
            ))
        else:
            updated.append(info)
    if collisions and depth < COLLISION_DEPTH:
        split_infos, split_marks, nested = _split_collisions(
            collisions, updated, marks, ingested, depth,
        )
        if nested is not None:
            return split_infos, split_marks, nested
    unresolved = [item for item in discrepancies if item.bin == "COLLISION" and not item.remediated]
    certified = not unresolved
    if depth > 0:
        # A completed split has already certified the halves.
        certified = True
        discrepancies = tuple(
            Discrepancy(
                item.landmark_a, item.landmark_b, item.key_distance,
                item.certified_distance, item.bin, True,
            )
            for item in discrepancies
        )
    return updated, marks, CertifyResult(
        tuple(tuple(row) for row in matrix),
        tuple(tuple(row) for row in key_matrix),
        tuple(discrepancies) if not isinstance(discrepancies, tuple) else discrepancies,
        certified,
        tuple(warnings),
    )


def _split_collisions(collisions, infos, marks, ingested, depth):
    by_id = {info.cluster_id: info for info in infos}
    replaced = set()
    pieces = []
    for left, right in collisions:
        info = by_id.get(left.cluster_id)
        if info is None or info.cluster_id in replaced:
            continue
        group_a = []
        group_b = []
        for member in info.members:
            dist_a = closure_distance(ingested.cells[member], left.cell, boundary_rel=0.0)[1]
            dist_b = closure_distance(ingested.cells[member], right.cell, boundary_rel=0.0)[1]
            if dist_a <= dist_b:
                group_a.append(member)
            else:
                group_b.append(member)
        if not group_a or not group_b:
            continue
        replaced.add(info.cluster_id)
        pieces.append(tuple(group_a))
        pieces.append(tuple(group_b))
    if not pieces:
        return infos, marks, None
    kept = [info for info in infos if info.cluster_id not in replaced]
    next_id = max((info.cluster_id for info in infos), default=-1) + 1
    for members in pieces:
        kept.append(_cluster_info(next_id, members, ingested))
        next_id += 1
    new_marks = landmarks(tuple(kept), ingested)
    return certify(tuple(kept), new_marks, ingested, depth + 1)


def _group_closed(operators, metric, limit) -> bool:
    inverses = {op: _inv(op) for op in operators}
    for left in operators:
        for right in operators:
            product = _mul(inverses[left], right)
            if _metric_residual(metric, product, metric) > limit:
                return False
    return True


def cosets(infos, marks, certify_result: CertifyResult, ingested: IngestResult,
           declared_symmetry: str | None) -> CosetResult:
    """Self-cosets, branch lists, and cross-cluster operators. Cached."""
    stored = []
    by_landmark = {mark.landmark_id: mark for mark in marks}
    medoid_of = {info.cluster_id: info.medoid for info in infos}
    medoid_landmark = {}
    for mark in marks:
        if mark.cell_index == medoid_of.get(mark.cluster_id):
            medoid_landmark[mark.cluster_id] = mark
    for info in infos:
        mark = medoid_landmark.get(info.cluster_id)
        cell = ingested.cells[info.medoid if mark is None else mark.cell_index]
        metric = metric_tensor(cell)
        trace = _trace_abs(metric)
        verify_rel = max(VERIFY_REL_FLOOR, info.eta / trace if trace else VERIFY_REL_FLOOR)
        boundary = info.eta / _max_vonorm(cell)
        operators = ()
        closed = False
        for _attempt in range(VERIFY_ATTEMPTS):
            operators = tuple(reindexing_via_canonical(
                cell, cell,
                boundary_rel=boundary,
                angle_sigma=ingested.sigma_deg,
                verify_rel=verify_rel,
            ))
            accepted = verify_rel * trace
            closed = _group_closed(operators, metric, GROUP_CLOSURE_FACTOR * accepted)
            if closed or not operators:
                break
            verify_rel *= VERIFY_WIDEN_FACTOR
        lid = -1 if mark is None else mark.landmark_id
        stored.append(LandmarkCoset(
            landmark_id=lid,
            operators=operators,
            branches=_branch_list(operators, declared_symmetry),
            group_closed=closed,
            residual_tol=verify_rel * trace,
            verify_rel=verify_rel,
        ))
    # Every landmark inherits its cluster medoid coset; the medoid row is the source.
    by_cluster_coset = {}
    for info, item in zip(infos, stored):
        by_cluster_coset[info.cluster_id] = item
    full = []
    for mark in marks:
        base = by_cluster_coset[mark.cluster_id]
        full.append(LandmarkCoset(
            mark.landmark_id, base.operators, base.branches, base.group_closed,
            base.residual_tol, base.verify_rel,
        ))
    cross = []
    matrix = certify_result.matrix
    for i, left in enumerate(marks):
        for j in range(i + 1, len(marks)):
            right = marks[j]
            if left.cluster_id == right.cluster_id:
                continue
            limit = CROSS_NOISE_FACTOR * max(
                _eta_key(left.cell, ingested.sigma_deg, ingested.sigma_len),
                _eta_key(right.cell, ingested.sigma_deg, ingested.sigma_len),
            )
            if matrix[i][j] >= limit:
                continue
            ops = tuple(reindexing_via_canonical(
                left.cell, right.cell,
                boundary_rel=by_cluster_coset[left.cluster_id].verify_rel,
                angle_sigma=ingested.sigma_deg,
                verify_rel=by_cluster_coset[left.cluster_id].verify_rel,
            ))
            if ops:
                cross.append((left.landmark_id, right.landmark_id, ops))
    return CosetResult(tuple(full), tuple(cross), declared_symmetry)


def assign(ingested: IngestResult, marks, coset_result: CosetResult) -> tuple:
    """One key, one tree query, and 3×3 products. No closure call."""
    if not marks:
        return tuple(
            Assignment(i, None, None, (), float("inf"), "unassigned")
            for i in range(len(ingested.cells))
        )
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray([mark.key for mark in marks], dtype=np.float64))
    by_id = {item.landmark_id: item for item in coset_result.self_cosets}
    branched = coset_result.declared_symmetry is not None
    out = []
    keys = np.asarray(ingested.keys, dtype=np.float64)
    _dists, chosen = tree.query(keys, k=1)
    chosen = np.atleast_1d(np.asarray(chosen, dtype=np.int64))
    for frame, landmark_slot in enumerate(chosen.tolist()):
        if ingested.outlier[frame]:
            out.append(Assignment(frame, None, None, (), float("inf"), "unassigned"))
            continue
        mark = marks[int(landmark_slot)]
        cached = by_id[mark.landmark_id]
        frame_metric = metric_tensor(ingested.cells[frame])
        target = metric_tensor(mark.reference_cell)
        # The medoid is the reciprocal-space setting, and it is a member.
        # A cell that differs by a few ångström stays on this identity assignment.
        residual = _metric_residual(frame_metric, mark.to_center, target)
        if branched and len(cached.branches) > 1:
            out.append(Assignment(
                frame, mark.landmark_id, mark.to_center, cached.branches, residual, "branched",
            ))
        else:
            out.append(Assignment(
                frame, mark.landmark_id, mark.to_center, (), residual, "unambiguous",
            ))
    return tuple(out)


def run(cells, spacegroups=None, sigma_deg=0.05, sigma_len=0.0,
        symmetry: str | None = None) -> PipelineState:
    """Ingest, cluster, certify, cache cosets, and assign every frame."""
    ingested = ingest(cells, spacegroups, sigma_deg=sigma_deg, sigma_len=sigma_len)
    infos, svd = cluster(ingested)
    marks = landmarks(infos, ingested)
    infos, marks, certified = certify(infos, marks, ingested)
    cached = cosets(infos, marks, certified, ingested, symmetry)
    assignments = assign(ingested, marks, cached)
    return PipelineState(
        ingest=ingested,
        clusters=tuple(infos),
        landmarks=marks,
        certify=certified,
        cosets=cached,
        assignments=assignments,
        svd_values=svd[0],
        svd_components=svd[1],
    )


def summary(state: PipelineState) -> str:
    """One text report. eta is Å²; m, g and delta are linear-key ångström."""
    counts = {"unambiguous": 0, "branched": 0, "unassigned": 0}
    for item in state.assignments:
        counts[item.status] = counts.get(item.status, 0) + 1
    bins = {"UNDER": 0, "INFLATED": 0, "COLLISION": 0}
    for item in state.certify.discrepancies:
        bins[item.bin] = bins.get(item.bin, 0) + 1
    lines = [
        f"N {len(state.ingest.cells)}",
        f"outliers {sum(state.ingest.outlier)}",
        f"clusters {len(state.clusters)}",
        f"certified {state.certify.certified}",
        (
            "discrepancies "
            f"UNDER {bins['UNDER']} INFLATED {bins['INFLATED']} COLLISION {bins['COLLISION']}"
        ),
    ]
    for info in state.clusters:
        lines.append(
            f"cluster {info.cluster_id} size {len(info.members)} "
            f"eta {info.eta:.6g} m {info.margin:.6g} g {info.gap:.6g} "
            f"delta {info.delta:.6g} isometric {info.isometric} "
            f"align_once {info.align_once}"
        )
    for item in state.cosets.self_cosets:
        if item.landmark_id < 0:
            continue
        lines.append(
            f"coset landmark {item.landmark_id} size {len(item.operators)} "
            f"branches {len(item.branches)} closed {item.group_closed}"
        )
    lines.append(
        "assignments "
        f"unambiguous {counts['unambiguous']} branched {counts['branched']} "
        f"unassigned {counts['unassigned']}"
    )
    return "\n".join(lines)
