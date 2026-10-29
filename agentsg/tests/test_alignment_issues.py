"""The alignment-issue ontology is well formed and covers every records flag and status."""
import pytest

pytest.importorskip("yaml")

from agentsg.cell.alignment_issues import KINDS, STATUSES, ancestors, by_flag, load_issues


def test_tree_is_well_formed():
    terms = load_issues()
    assert "alignment_issue" in terms
    for t, d in terms.items():
        assert d.get("definition"), t
        if t == "alignment_issue":
            continue
        assert d.get("parent") in terms, t
        assert ancestors(t)[-1] == "alignment_issue", t
        assert d.get("kind") in KINDS, t
        if "status" in d:
            assert d["status"] in STATUSES, t


@pytest.mark.parametrize("name", ["low_margin", "far_same_structure", "far_different_structure",
                                  "different_space_group", "no_shared_accession",
                                  "too_few_paired_atoms"])
def test_every_flag_and_status_has_a_term(name):
    assert by_flag(name), name
