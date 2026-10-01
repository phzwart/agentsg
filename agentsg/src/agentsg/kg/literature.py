"""Bibliographic pointers promoted from concept literature strings.

A DOI is recorded only after a Crossref work whose authors, year, and
title match the citation. Strings with no confirmed DOI stay as citations.
"""
from __future__ import annotations

# match is a substring of the literature string on the concept.
RECORDS = [
    {
        "match": "Hall, S. R. (1981)",
        "title": "Space-group notation with an explicit origin",
        "doi": "10.1107/S0567739481001228",
    },
    {
        "match": "Le Page (1982)",
        "title": "The derivation of the axes of the conventional unit cell from the dimensions of the Buerger-reduced cell",
        "doi": "10.1107/S0021889882011959",
    },
    {
        "match": "Lebedev, Vagin & Murshudov (2006)",
        "title": "Intensity statistics in twinned crystals with examples from the PDB",
        "doi": "10.1107/S0907444905036759",
    },
    {
        "match": "Grosse-Kunstleve, Sauter & Adams (2004)",
        "title": "Numerically stable algorithms for the computation of reduced unit cells",
        "doi": "10.1107/S010876730302186X",
    },
    {
        "match": "Křivý & Gruber (1976)",
        "title": "A unified algorithm for determining the reduced (Niggli) cell",
        "doi": "10.1107/S0567739476000636",
    },
    {
        "match": "Andrews & Bernstein (1988). Acta Cryst. A44",
        "title": "Lattices and reduced cells as points in 6-space and selection of Bravais lattice type by projections",
        "doi": "10.1107/S0108767388006427",
    },
    {
        "match": "Andrews & Bernstein (2014)",
        "title": "The geometry of Niggli reduction: BGAOL — embedding Niggli reduction and analysis of boundaries",
        "doi": "10.1107/S1600576713031002",
    },
    {
        "match": "Brehm & Diederichs (2014)",
        "title": "Breaking the indexing ambiguity in serial crystallography",
        "doi": "10.1107/S1399004713025431",
    },
    {
        "match": "Billiet & Rolley-Le Coz (1980)",
        "title": "Le groupe P1 et ses sous-groupes. III. Conservation réticulaire",
        "doi": "10.1107/S056773948000157X",
    },
    {
        "match": "arXiv:2201.10543",
        "title": "Mathematics of 2-dimensional and 3-dimensional lattices",
        "url": "https://arxiv.org/abs/2201.10543",
    },
]


def lookup(citation: str) -> dict:
    """Return doi/url/title for a citation string, or just the citation."""
    for rec in RECORDS:
        if rec["match"] in citation:
            out = {"title": rec["title"]}
            if rec.get("doi"):
                out["doi"] = rec["doi"]
                out["url"] = "https://doi.org/" + rec["doi"]
            elif rec.get("url"):
                out["url"] = rec["url"]
            return out
    return {}
