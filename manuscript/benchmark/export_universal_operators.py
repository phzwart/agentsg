"""Write the 6096 elements of U as integer column operators W (P = B_R W B_N^-1 between two
Selling-reduced bases B_R, B_N, columns being basis vectors) to universal_operators_columns.json,
with the determinant of each. A lookup keeps the 3048 whose determinant equals
det(B_R) det(B_N), so that P is proper whatever the hands of the two reduced bases.

    PYTHONPATH=agentsg/src python manuscript/benchmark/export_universal_operators.py
"""
import json
import os

import numpy as np

from agentsg.cell.strain_closure import universal_operators

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    U = universal_operators()
    det = np.round(np.linalg.det(U)).astype(int)
    json.dump({"convention": "columns are new basis vectors in the old basis; P = B_R W B_N^-1",
               "W": U.tolist(), "det": det.tolist()},
              open(os.path.join(HERE, "universal_operators_columns.json"), "w"))
    print(len(U), "operators;", int((det == 1).sum()), "of determinant +1,", int((det == -1).sum()), "of -1")
