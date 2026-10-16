"""Reindexing cosets of a cell list against one reference lattice.

Each cell may be any unit cell. A Niggli-reduced cell is the usual input.
Two changes of basis are stored separately:

1. :func:`to_selling` — the submitted cell to one obtuse cell, and that COB.
2. :func:`closure_cobs` — the Selling closure of the obtuse cell, every COB
   written in the reduced basis.

The coset is a later step. It multiplies those matrices into the basis that
was submitted, and only when the sorted linear distance is inside the volume
gate. A cell past the gate comes back with an empty coset and the measured
linear distance.

Run::

    python agentsg/examples/reindex_cosets.py
"""
from __future__ import annotations

import numpy as np

from agentsg.cell.rootform import sorted_linear_distance, symmetry_cutoff
from agentsg.cell.sublattice import apply_to_cell
from agentsg.tolerances import REINDEX_BAND_REL, REINDEX_BOUNDARY_REL, VOLUME_FRAC

from primitive_batch import primitive_cells
from selling_steps import closure_cobs, combine_coset, to_selling


def _prepared(cell, boundary_rel):
    """Step 1 and step 2, stored separately."""
    selling = to_selling(cell)
    closure = closure_cobs(selling["reduced"], boundary_rel=boundary_rel)
    return selling, closure


def reindex_cosets(
    reference,
    cells,
    symbols=None,
    *,
    reference_symbol=None,
    max_volume_frac=VOLUME_FRAC,
    boundary_rel=REINDEX_BOUNDARY_REL,
):
    """Selling COB, closure COBs, and the later coset onto the reference.

    Parameters
    ----------
    reference :
        Any one cell, six numbers: ``a, b, c`` in Å and angles in degrees.
        A Niggli cell is the usual case. Selling reduction is not required.
    cells :
        Any cells to reindex onto the reference, shape ``(n, 6)``.
        Same setting rule as ``reference``.
    symbols :
        Centering letters or space-group strings for ``cells``.
    reference_symbol :
        Centering letter or space-group string for the reference.
        ``None`` keeps the reference cell as given.
    max_volume_frac :
        Sorted-linear-key gate as a fractional volume change. Default
        ``VOLUME_FRAC`` (0.05). A cell beyond the gate returns an empty coset.
    boundary_rel :
        Near-obtuse width for the closure (step 2). Default
        ``REINDEX_BOUNDARY_REL``. It is not folded into the step-1 COB.

    Returns
    -------
    dict
        ``reference`` is the primitive reference cell.
        ``selling`` is step 1. ``closure_cobs`` is step 2, in the reduced
        basis, and ``closure_size`` is its length.
        ``rows`` has one dict per input cell: ``primitive`` (submitted setting,
        centring removed), ``selling``, ``closure_cobs``, ``linear_distance``
        in Å, and ``coset`` (the two COBs multiplied into the submitted basis;
        empty when the gate rejects).
    """
    ref = tuple(float(x) for x in primitive_cells([reference], [reference_symbol])[0])
    ref_selling, ref_closure = _prepared(ref, boundary_rel)
    prim = primitive_cells(cells, symbols)
    rows = []
    for row in prim:
        cell = tuple(float(x) for x in row)
        selling, closure = _prepared(cell, boundary_rel)
        dist = float(sorted_linear_distance(cell, ref))
        gate = symmetry_cutoff(cell, volume_tol=max_volume_frac)
        if dist > gate:
            coset = []
        else:
            coset = combine_coset(
                selling, closure, ref_selling, ref_closure, band_rel=REINDEX_BAND_REL,
            )
        rows.append({
            "primitive": cell,
            "selling": selling,
            "closure_cobs": closure,
            "linear_distance": dist,
            "coset": coset,
        })
    return {
        "reference": ref,
        "selling": ref_selling,
        "closure_cobs": ref_closure,
        "closure_size": len(ref_closure),
        "rows": rows,
    }


def _fmt_op(op):
    return "[" + "; ".join(" ".join(str(int(x)) for x in row) for row in op) + "]"


def main():
    reference = (5.0, 6.0, 7.0, 90.0, 90.0, 90.0)
    swapped = apply_to_cell(reference, ((0, 0, 1), (0, 1, 0), (1, 0, 0)))
    other = (8.0, 9.0, 15.0, 80.0, 85.0, 95.0)
    cells = np.asarray([reference, swapped, other], dtype=np.float64)
    result = reindex_cosets(reference, cells)
    print(f"to-Selling COB: {_fmt_op(result['selling']['cob'])}")
    print(f"closure COBs, reduced basis: {result['closure_size']}")
    names = ("same setting", "a<->c setting", "unrelated cell")
    for name, row in zip(names, result["rows"]):
        ops = row["coset"]
        head = _fmt_op(ops[0]) if ops else "empty"
        print(
            f"{name}: linear {row['linear_distance']:.4g} Å, "
            f"coset {len(ops)}, first {head}"
        )
    return result


if __name__ == "__main__":
    main()
