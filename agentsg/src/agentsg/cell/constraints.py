"""
The ONE sanctioned crossing of the exact/numeric boundary (see docs/DESIGN.md).

Everywhere else, ``agentsg.cell`` is real-valued and knows nothing about the
exact symmetry algebra. Here -- and only here -- we import exact point-group
operators (integer W matrices) and use them to constrain or symmetrise a numeric
metric tensor. The governing identity is

    W^T G W = G   for every point-group operation W,

i.e. the metric tensor is invariant under the point group. This is what forces
a cubic cell to have a=b=c, alpha=beta=gamma=90, a hexagonal cell to have
gamma=120, and so on -- the crystal-system restrictions are a *consequence* of
this equation, not a separate table.
"""
from __future__ import annotations


def _matT_G_mat(W, G):
    """Compute W^T G W for integer W (list of rows) and numeric G."""
    # (W^T G W)_{ij} = sum_{k,l} W_{ki} G_{kl} W_{lj}
    out = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                for l in range(3):
                    s += W[k][i] * G[k][l] * W[l][j]
            out[i][j] = s
    return out


def _int_rows(W):
    """Extract integer rotation rows from a agentsg Matrix3 or a plain nested list."""
    rows = getattr(W, "rows", W)
    return [[int(rows[i][j]) for j in range(3)] for i in range(3)]


def metric_is_invariant(G, point_group_ops, tol: float = 1e-6) -> bool:
    """True iff W^T G W == G (within tol) for every rotation W in the point group.

    ``point_group_ops`` may be agentsg Matrix3 rotation parts (e.g. from
    ``agentsg.group.point_group``) or plain 3x3 integer lists.
    """
    for W in point_group_ops:
        Wr = _int_rows(W)
        GG = _matT_G_mat(Wr, G)
        for i in range(3):
            for j in range(3):
                if abs(GG[i][j] - G[i][j]) > tol * (1 + abs(G[i][j])):
                    return False
    return True


def symmetrize_metric(G, point_group_ops):
    """Project a numeric metric tensor onto the point-group-invariant subspace by
    Reynolds averaging:  G_sym = (1/|PG|) sum_W W^T G W.

    The result exactly satisfies W^T G_sym W = G_sym for all W in the group, and
    is the closest invariant metric to G in the Frobenius sense. This is how a
    measured (noisy) cell is snapped onto its ideal crystal-system metric.
    """
    ops = list(point_group_ops)
    if not ops:
        raise ValueError("empty point group")
    acc = [[0.0] * 3 for _ in range(3)]
    for W in ops:
        Wr = _int_rows(W)
        GG = _matT_G_mat(Wr, G)
        for i in range(3):
            for j in range(3):
                acc[i][j] += GG[i][j]
    n = len(ops)
    return [[acc[i][j] / n for j in range(3)] for i in range(3)]


def free_metric_parameters(point_group_ops, tol: float = 1e-9) -> int:
    """Number of independent free parameters in a metric tensor invariant under
    the given point group -- i.e. the dimension of the space of allowed cells.

    Computed by symmetrising each of the 6 independent basis metric tensors and
    counting the rank of the resulting (symmetric) images. This *derives* the
    familiar counts: triclinic 6, monoclinic 4, orthorhombic 3, tetragonal &
    hexagonal & trigonal 2, cubic 1.
    """
    ops = list(point_group_ops)
    # basis of symmetric 3x3 matrices (6 of them)
    basis_idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    images = []
    for (a, b) in basis_idx:
        E = [[0.0] * 3 for _ in range(3)]
        E[a][b] = 1.0
        E[b][a] = 1.0
        S = symmetrize_metric(E, ops)
        # flatten to the 6-vector (upper triangle, off-diagonals doubled weight
        # is fine as long as consistent) -- use all 6 unique entries
        images.append([S[0][0], S[1][1], S[2][2], S[0][1], S[0][2], S[1][2]])
    # rank of the 6x6 matrix of images = number of free parameters
    return _rank(images, tol)


def _rank(rows, tol):
    """Numeric rank of a small matrix via Gaussian elimination with pivoting."""
    M = [r[:] for r in rows]
    n = len(M)
    m = len(M[0]) if n else 0
    rank = 0
    used = [False] * n
    for col in range(m):
        piv = -1
        best = tol
        for i in range(n):
            if not used[i] and abs(M[i][col]) > best:
                best = abs(M[i][col]); piv = i
        if piv < 0:
            continue
        used[piv] = True
        rank += 1
        pv = M[piv][col]
        for i in range(n):
            if i != piv and abs(M[i][col]) > tol:
                f = M[i][col] / pv
                M[i] = [M[i][j] - f * M[piv][j] for j in range(m)]
    return rank
