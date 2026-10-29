"""The ontology of crystal-form alignment issues (``data/alignment_issues.yaml``).

``load_issues()`` returns ``{term_id: dict}``; ``ancestors(term)`` the is_a chain;
``by_flag(flag)`` the terms a records flag or status maps to.
"""
from __future__ import annotations

import os
from functools import lru_cache

_PATH = os.path.join(os.path.dirname(__file__), "data", "alignment_issues.yaml")
KINDS = ("input", "lattice", "placement", "comparison", "biology", "method")
STATUSES = ("handled", "flagged", "open")


@lru_cache(maxsize=1)
def load_issues():
    import yaml
    with open(_PATH) as fh:
        doc = yaml.safe_load(fh)
    return doc["terms"]


def ancestors(term):
    terms = load_issues()
    out = []
    while term in terms and terms[term].get("parent"):
        term = terms[term]["parent"]
        out.append(term)
    return out


def by_flag(flag):
    """Terms whose ``flag`` field or ``signature`` names a records flag or status."""
    return [t for t, d in load_issues().items()
            if d.get("flag") == flag or flag in str(d.get("signature", ""))]
