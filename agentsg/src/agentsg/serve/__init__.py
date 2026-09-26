"""Public HTTP API for Muse (and any OpenAPI client).

Stdlib only. Optional extras: ``agentsg[db]`` for PDB search,
``agentsg[plot]`` for ITA plates.
"""
from .app import run_server

__all__ = ["run_server"]
