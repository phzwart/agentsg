"""The concept graph stays tied to the source tree and to the ledger."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentsg.kg.graph import ConceptGraph
from agentsg.serve.handlers import TOOL_CONCEPTS

ROOT = Path(__file__).resolve().parents[2]
KG = ROOT / "agentsg" / "src" / "agentsg" / "kg"
SRC = ROOT / "agentsg" / "src"
OUT = KG / "out"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_anchors_rebuild_has_zero_problems(tmp_path):
    dest = tmp_path / "anchors.json"
    proc = subprocess.run(
        [sys.executable, str(KG / "anchors.py"), str(SRC), str(dest)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "0 problems" in proc.stdout
    payload = json.loads(dest.read_text())
    assert payload["anchors"]


def test_code_evidence_sha256_matches_files():
    payload = json.loads((OUT / "agentsg_concepts.json").read_text())
    seen: dict[str, str] = {}
    for concept in payload["concepts"]:
        assert concept["code_evidence"], concept["id"]
        assert concept["references"], concept["id"]
        for anchor in concept["code_evidence"]:
            module = anchor["module"]
            digest = _sha(SRC / module)
            assert anchor["sha256"] == digest, module
            seen.setdefault(module, digest)
    assert seen


def test_relation_targets_and_csv_endpoints_exist():
    payload = json.loads((OUT / "agentsg_concepts.json").read_text())
    ids = {c["id"] for c in payload["concepts"]}
    for concept in payload["concepts"]:
        for rel in concept["relations"]:
            assert rel["target"] in ids, f"{concept['id']} -> {rel['target']}"
    neo = OUT / "neo4j"
    with (neo / "concept_relations.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            assert row[":START_ID(Concept)"] in ids
            assert row[":END_ID(Concept)"] in ids
    evidence = set()
    with (neo / "code_evidence.csv").open(newline="") as handle:
        evidence = {row["id:ID(Evidence)"] for row in csv.DictReader(handle)}
    with (neo / "code_evidence_edges.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            assert row[":START_ID(Concept)"] in ids
            assert row[":END_ID(Evidence)"] in evidence
    references = set()
    with (neo / "references.csv").open(newline="") as handle:
        references = {row["id:ID(Reference)"] for row in csv.DictReader(handle)}
    with (neo / "reference_edges.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            assert row[":START_ID(Concept)"] in ids
            assert row[":END_ID(Reference)"] in references


def test_ledger_validates():
    if importlib.util.find_spec("pygrits") is None:
        pytest.skip("pygrits is not installed")
    import pygrits
    ledger = json.loads((OUT / "agentsg_kg.grits.jsonld").read_text())
    pygrits.validate(pygrits.load(ledger))


def test_tool_concept_ids_exist_in_the_payload():
    graph = ConceptGraph.load(OUT)
    for tool, ids in TOOL_CONCEPTS.items():
        assert ids, tool
        for ident in ids:
            assert ident in graph.by_id, f"{tool} -> {ident}"


def test_search_neighbors_and_receipt():
    graph = ConceptGraph.load(OUT)
    assert graph.resolve("Hall notation") == "hall_symbol"
    hits = graph.search("a-glide")
    assert hits["hits"][0]["id"] == "glide_a"
    neighbours = graph.neighbors("glide_a", relation="SPECIALIZES", depth=1)
    assert {row["id"] for row in neighbours["neighbors"]} == {"glide_plane"}
    receipt = graph.receipt("ent:concept:space_group")
    assert receipt["node"]["@id"] == "ent:concept:space_group"


def test_handler_source_parses():
    ast.parse((SRC / "agentsg" / "serve" / "handlers.py").read_text())
