"""Drop centring from a batch of conventional cells.

Each row is any unit cell: ``a, b, c`` in Å and angles in degrees. It does
not need to be Selling-reduced. A Niggli-reduced cell is the usual input.
This helper only applies the centring matrix. It does not Niggli-reduce or
Selling-reduce the metric.

A centering letter (``P``, ``A``, ``B``, ``C``, ``I``, ``F``, ``R``) and a
space-group symbol are the same kind of argument: :func:`primitive_cell`
reads only the lattice letter. A missing entry is left unchanged.
"""
from __future__ import annotations

import numpy as np

from agentsg.cell.primitive import primitive_cell


def primitive_cells(cells, symbols=None) -> np.ndarray:
    """Return primitive cells for a conventional ``(n, 6)`` array.

    Parameters
    ----------
    cells :
        Any cells, shape ``(n, 6)``: ``a, b, c`` in Å and angles in degrees.
        A Niggli cell is the usual case. Selling reduction is not required.
        A single six-vector is accepted as one row.
    symbols :
        Optional sequence of length ``n``. Each entry is a centering letter,
        a Hermann–Mauguin symbol, or ``None`` / ``""`` to keep that row.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 6)``, float64. Centring removed. The metric is not
        Niggli-reduced or Selling-reduced.
    """
    arr = np.asarray(cells, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("cells must have shape (n, 6)")
    n = arr.shape[0]
    if symbols is None:
        flags = [None] * n
    else:
        flags = list(symbols)
        if len(flags) != n:
            raise ValueError(f"symbols has length {len(flags)}, expected {n}")
    out = np.empty((n, 6), dtype=np.float64)
    for i, flag in enumerate(flags):
        cell = tuple(float(x) for x in arr[i])
        if flag is None or str(flag).strip() == "":
            out[i] = cell
        else:
            out[i] = primitive_cell(cell, str(flag))
    return out
