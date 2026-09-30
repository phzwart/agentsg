"""Concept graph search, cards, and receipts."""
import urllib.error

from agentsg.serve.app import make_handler
from agentsg.serve.concepts import get_graph
from agentsg.serve.mcp_app import _mcp_playbook

from test_serve import _fetch, _server


def test_search_smith_and_allowed_origin():
    graph = get_graph()
    smith = graph.search("Smith normal form")
    assert smith["hits"][0]["id"] == "smith_normal_form"
    assert smith["snapshot"]
    allowed = graph.search("allowed origin")
    assert allowed["hits"][0]["id"] == "allowed_origins"


def test_card_uses_module_and_receipt():
    graph = get_graph()
    card = graph.card("smith_normal_form")
    assert card["definition_receipt"] == "ent:concept:smith_normal_form"
    assert card["code_evidence"]
    assert card["code_evidence"][0]["receipt"].startswith("evi:code:smith_normal_form:")
    assert card["code_evidence"][0]["uri"].startswith("https://github.com/phzwart/agentsg/")
    uses = graph.uses("reflection_conditions", 3)
    assert uses["uses"]
    assert all(1 <= row["depth"] <= 3 for row in uses["uses"])
    assert uses["uses"] == sorted(uses["uses"], key=lambda row: (row["depth"], row["id"]))
    mods = graph.module("semi_invariants.py")
    assert any(row["id"] for row in mods["concepts"])
    assert all("semi_invariants.py" in a["module"] for row in mods["concepts"] for a in row["anchors"])
    receipt = graph.receipt("ent:concept:allowed_origins")
    assert receipt["node"]["how"] == "derived"
    assert "rationale" in receipt["node"]


def test_http_concept_routes_and_skill_questions():
    state, httpd, base = _server()
    try:
        status, health = _fetch(f"{base}/health")
        assert status == 200
        assert health["concepts"] > 100
        status, found = _fetch(f"{base}/v1/concept?q=Smith+normal+form")
        assert status == 200
        assert found["hits"][0]["id"] == "smith_normal_form"
        status, card = _fetch(f"{base}/v1/concept?id=allowed_origins")
        assert status == 200
        assert card["definition"]
        assert any(ref["source"] == "iucr" and ref["quote"] for ref in card["references"])
        status, receipt = _fetch(
            f"{base}/v1/concept/receipt?id=ent:concept:allowed_origins",
        )
        assert status == 200
        assert receipt["node"]["@id"] == "ent:concept:allowed_origins"
        status, mods = _fetch(
            f"{base}/v1/concept/module?module=agentsg/semi_invariants.py",
        )
        assert status == 200
        assert mods["concepts"]
        status, md = _fetch(f"{base}/skill.md")
        text = md.decode()
        assert "Where is the Smith normal form implemented?" in text
        assert "Concept questions" in text
        assert "ent:concept:allowed_origins" in text
        try:
            _fetch(f"{base}/v1/concept?id=not_a_concept")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        httpd.shutdown()
        if state.db:
            state.db.close()


def test_playbook_names_tools_and_sample_questions():
    book = _mcp_playbook("https://sg-mcp.mxagents.org")
    assert "Where is the Smith normal form implemented?" in book
    assert "What does “allowed origin” mean here" in book
    assert 'concept with q="Smith normal form"' in book
    assert "concept_uses" in book
    assert "concept_module" in book
    assert "concept_receipt" in book
    assert "GET /v1/concept" not in book
    assert "bearer" not in book.lower()
    assert "authorization" not in book.lower()
