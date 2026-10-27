"""z51 — the seven sample cells distributed with Oishi-Tomiyasu's BLDConograph
(github.com/rtomiyasu, commit 93b58ed): the cells, their Bravais types and BLDConograph's
minimal distances, used by the checks of SI Table S5 (z60, z66).  No claim of its own.
"""
from __future__ import annotations

# name, primitive cell as distributed (input.py), Bravais type in the sample
# folder name, holohedry order, BLDConograph minimal distance for that type
SAMPLES = [
    ("HERMES_Sr327_250K", (3.8417, 3.8524, 10.389, 100.41, 100.53, 90.046), "tI", 16, 0.5672),
    ("1339prbasrcuo_bs1", (3.8792, 3.8830, 11.654, 90.018, 90.039, 90.057), "tP", 16, 0.05641),
    ("sample3", (16.332, 16.348, 16.357, 109.41, 109.52, 109.34), "cI", 48, 2.0738),
    ("CeO2_matsushita", (3.7250, 3.7269, 3.7286, 90.100, 119.96, 119.91), "cF", 48, 0.084427),
    ("M402-pf-119673", (10.654, 22.895, 23.003, 60.600, 89.377, 89.119), "hP", 24, 20.549),
    ("Acrinol-B-D8CuKa1", (9.1568, 9.1712, 20.946, 81.634, 81.751, 80.086), "mC", 4, 0.71525),
    ("Alumina676", (4.7428, 4.7802, 5.1093, 116.89, 90.170, 119.69), "hR", 12, 1.9315),
]
