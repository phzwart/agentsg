"""Where the supplement's files are.  The package is used from a checkout of the deposited
repository (``pip install -e manuscript/supplement``); the PDB snapshot and other external
data go in the repository's ``data/`` directory, or in ``$SELLINGPROOFS_DATA``."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SUPPLEMENT = PACKAGE.parents[1]                  # manuscript/supplement
REPO = SUPPLEMENT.parents[1]                     # the agentsg repository
BENCH = SUPPLEMENT / "benchmark"                 # scripts, deposited inputs and results
FORMAL = SUPPLEMENT / "formal"                   # the Lean development
PAPER = REPO / "manuscript" / "selling_reduction"
DATA = Path(os.environ.get("SELLINGPROOFS_DATA", REPO / "data"))

PDB_DB = DATA / "pdb_cells.duckdb"
PDB_ROOTS = REPO / "analysis" / "data" / "pdb_roots.npz"
XFEL_STREAM = DATA / "blac_new_v0_nomulti.stream"
U_COLUMNS = BENCH / "universal_operators_columns.json"
