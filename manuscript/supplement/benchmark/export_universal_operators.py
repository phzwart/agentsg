"""Write the 6096 elements of U as integer column operators W (P = B_R W B_N^-1 between two
Selling-reduced bases B_R, B_N, columns being basis vectors) to universal_operators_columns.json,
with the determinant of each. A lookup keeps the 3048 whose determinant equals
det(B_R) det(B_N), so that P is proper whatever the hands of the two reduced bases.

    PYTHONPATH=agentsg/src:manuscript/supplement/src python manuscript/supplement/benchmark/export_universal_operators.py
"""
import json
import os

import numpy as np

from agentsg.cell.strain_closure import universal_operators

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SELLINGPROOFS_OUT", HERE)          # output directory (default: here)

if __name__ == "__main__":
    U = universal_operators()
    det = np.round(np.linalg.det(U)).astype(int)
    meta = {
        "title": "The universal operator set U of the Selling closure",
        "reference": "P. H. Zwart, The Selling closure: a complete operator set for reindexing, "
                     "and the price of pseudo-symmetry (Acta Cryst. A), Section 3.3 and Appendix A",
        "software": "agentsg 0.5.0, https://github.com/phzwart/agentsg",
        "count": int(len(U)),
        "contents": "every unimodular integer matrix that carries one obtuse (Selling-reduced) "
                    "superbase of a lattice to another obtuse superbase of the same lattice, for "
                    "any lattice; 6096 = 48 x 127 matrices with entries in {-3, ..., 3}, closed under "
                    "inversion and under the 48 relabellings of a reduced superbase",
        "reduced_basis": "a Selling-reduced basis (a, b, c) is the last three vectors of an obtuse "
                         "superbase (d, a, b, c), d = -a-b-c, all six pairwise dot products <= 0",
        "W": "W[n] is a 3 x 3 integer matrix given as a list of rows; column j of W[n] is the j-th "
             "new basis vector written in the old reduced basis (a, b, c)",
        "det": "det[n] is the determinant of W[n], +1 or -1; 3048 of each",
        "usage": "with B_R and B_N the reduced bases of a reference and a new cell (columns are "
                 "basis vectors, in each cell's input basis), the candidate operators are "
                 "P = B_R W B_N^-1, which satisfy G_N ~ s P^T G_R P for metric tensors G and a "
                 "scale s; keep the W with det(W) = det(B_R) det(B_N) for proper P, price each "
                 "candidate and keep those within the band. The list contains the true operator "
                 "whenever the reference's widened set lies in the first layer (Remark A8).",
    }
    json.dump({"description": meta, "W": U.tolist(), "det": det.tolist()},
              open(os.path.join(OUT, "universal_operators_columns.json"), "w"), indent=None)
    print(len(U), "operators;", int((det == 1).sum()), "of determinant +1,", int((det == -1).sum()), "of -1")
