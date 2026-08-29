"""Paths and skip helpers for Appendix Z checks."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PDB_DB = REPO / "data" / "pdb_cells.duckdb"
PDB_ROOTS = REPO / "analysis" / "data" / "pdb_roots.npz"
XFEL_STREAM = REPO / "data" / "blac_new_v0_nomulti.stream"


@pytest.fixture(scope="session")
def repo_root():
    return REPO


@pytest.fixture(scope="session")
def pdb_db_path():
    if not PDB_DB.is_file():
        pytest.skip(f"PDB DuckDB not found: {PDB_DB}")
    return PDB_DB


@pytest.fixture(scope="session")
def pdb_roots_path():
    if not PDB_ROOTS.is_file():
        pytest.skip(f"PDB roots npz not found: {PDB_ROOTS}")
    return PDB_ROOTS


@pytest.fixture(scope="session")
def xfel_stream_path():
    if not XFEL_STREAM.is_file():
        pytest.skip(f"XFEL stream not found: {XFEL_STREAM}")
    return XFEL_STREAM
