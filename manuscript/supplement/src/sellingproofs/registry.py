"""The claim registry: every check is a function decorated with the z-number of the
statement it verifies and the place in the paper where that statement is made.

    @claim("z52", "main §3.3, Table 4", kind="exhaustive")
    def test_universal_set_has_6096_unimodular_operators():
        "U has 6096 distinct unimodular elements with entries in {-3, ..., 3}."
        ...

The first paragraph of the docstring is the statement as the ledger prints it.  The
decorator records the claim, attaches pytest marks (the z-number, the kind, and
``needs_<resource>`` for every resource in ``needs``), and leaves the function otherwise unchanged, so the claims run as
ordinary pytest tests.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

KINDS = {
    "formal": "machine-checked proof (Lean 4 + Mathlib)",
    "exhaustive": "exhaustive finite computation; the assertion is exact",
    "sampled": "assertion over seeded random samples",
    "certified": "exact rational certificate",
    "benchmark": "number printed in the paper, recomputed on the deposited benchmark",
    "pdb": "number printed in the paper, recomputed on the PDB snapshot",
    "reproduce": "a deposited results file is regenerated identically",
}

NEEDS = {
    "slow": "takes more than about a minute",
    "pdb": "the PDB cell snapshot (Zenodo, doi:10.5281/zenodo.22986222) in data/",
    "cctbx": "cctbx-base",
    "sella": "SELLA (CmdSella)",
    "ncdist": "the NCDist/CS6Dist libraries",
    "lean": "Lean 4 with lake on PATH",
    "xfel": "the XFEL stream data file in data/",
    "sympy": "sympy",
    "lysozyme": "the lysozyme data files in data/",
}


@dataclass(frozen=True)
class Claim:
    z: str
    where: tuple[str, ...]
    kind: str
    statement: str
    function: str
    needs: tuple[str, ...] = field(default=())

    @property
    def module(self) -> str:
        return self.function.split("::")[0]


REGISTRY: list[Claim] = []


def _statement(f) -> str:
    doc = inspect.getdoc(f) or ""
    return " ".join(doc.split("\n\n")[0].split())


def claim(z: str, *where: str, kind: str = "exhaustive", needs: tuple[str, ...] | list[str] = ()):
    """Register a check of the statement numbered ``z`` made at ``where``."""
    if not (len(z) == 3 and z[0] == "z" and z[1:].isdigit()):
        raise ValueError(f"z-number must look like z07, got {z!r}")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; one of {sorted(KINDS)}")
    unknown = set(needs) - set(NEEDS)
    if unknown:
        raise ValueError(f"unknown needs {sorted(unknown)}; one of {sorted(NEEDS)}")
    if not where:
        raise ValueError("give the place in the paper where the statement is made")

    def deco(f):
        c = Claim(z=z, where=tuple(where), kind=kind, statement=_statement(f),
                  function=f"{f.__module__}::{f.__qualname__}", needs=tuple(needs))
        if not c.statement:
            raise ValueError(f"{c.function}: the docstring states the claim; it is empty")
        REGISTRY.append(c)
        f.__claim__ = c
        try:
            import pytest
        except ImportError:                                   # ledger without pytest
            return f
        marks = [getattr(pytest.mark, z), pytest.mark.claim, getattr(pytest.mark, kind)]
        marks += [getattr(pytest.mark, f"needs_{n}") for n in needs]
        for m in marks:
            f = m(f)
        return f
    return deco


def load_all() -> list[Claim]:
    """Import every claim module and return the registry, in z order."""
    import importlib
    import pkgutil

    from . import claims
    for m in pkgutil.iter_modules(claims.__path__):
        if m.name.startswith("test_"):
            importlib.import_module(f"{claims.__name__}.{m.name}")
    seen, out = set(), []
    for c in REGISTRY:
        if c.function not in seen:
            seen.add(c.function)
            out.append(c)
    return sorted(out, key=lambda c: (c.z, c.function))
