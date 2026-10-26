"""z62 — the machine-checked proofs (Lean 4 + Mathlib, ``manuscript/supplement/formal``).

The development is built once (``lake build``); each claim then reads the axioms its
theorem depends on (``#print axioms``) and asserts that they are exactly the three standard
axioms of Lean's logic plus the named compiled evaluations (``native_decide``), so that no
theorem rests on ``sorry`` or on an unlisted assumption.  A further claim regenerates the
Lean copy of U from the deposited table and compares it byte for byte.
"""
from __future__ import annotations

import re
import subprocess
import sys

import pytest

from sellingproofs import paths
from sellingproofs.registry import claim

STANDARD = {"propext", "Classical.choice", "Quot.sound"}

# (Lean name, place in the paper, statement, compiled evaluations it may use)
THEOREMS = [
    ("arrow_mem_U", ("main §3.3", "main Lemma A2"),
     "Completeness of U: every unimodular matrix between two obtuse superbases of any lattice is in the deposited table.",
     {"finite_check"}),
    ("U_arrow", ("main §3.3",),
     "Every element of U is an arrow of every lattice on some face.",
     {"U_realised", "U_unimodular"}),
    ("arrow_zero_pattern", ("main §3.3", "main §3.2"),
     "Arrows depend only on the zero pattern: an arrow for one metric is an arrow for every metric with the same zero conorms.",
     {"finite_check", "zero_pattern_check"}),
    ("box_of_conn", ("main Lemma A2",),
     "Lemma A2 (window): gamma = 0 on a connected support puts a vector in the box [-7, 7]^3.",
     set()),
    ("U_swaps", ("main Remark A8",), "U is closed under the 48 swaps (relabellings of the target cell).",
     {"U_swaps"}),
    ("U_inv", ("main §3.3",), "U is closed under inversion.", {"U_inv"}),
    ("excess_identity", ("main Lemma A1",), "Lemma A1, the class excess identity.", set()),
    ("partial_shortest", ("main Appendix A, Terms",),
     "The partial sums of an obtuse superbase are shortest in their parity classes.", set()),
    ("price_mul", ("main Proposition A10(i)",), "Prices are subadditive under products.", set()),
    ("price_inv", ("main Proposition A10(i)",), "A self-map and its inverse have the same price.", set()),
    ("coset_lower", ("main Proposition A10(ii)",), "H_{eps-p0} P0 lies among the candidates.", set()),
    ("coset_upper", ("main Proposition A10(ii)",), "The candidates lie in H_{eps+p0} P0.", set()),
    ("Hgroup", ("main Proposition A10(iii)",), "With a price gap (eps, 2 eps], H_eps is a group.", set()),
    ("candidates_eq_coset", ("main Proposition A10(iv)",),
     "With a price gap (eps - p0, eps + p0], the candidates are one coset H_eps P0.", set()),
    ("stab_sub_H", ("main Proposition A10(v)",),
     "The point group of a metric at distance nu lies in H_{2 nu}.", set()),
    ("H_sub_H", ("main Proposition A10(v)",), "H_eps(G_R) lies in H_{eps + 2 nu}(G_0).", set()),
    ("twofold_midpoint_fixed", ("main Proposition A9(i)",),
     "For a twofold g, the midpoint of x and g x is fixed by g.", set()),
    ("price_le_two_dist_fixed", ("main Proposition A9(i)",),
     "The price of g is at most twice the distance to the metrics g fixes.", set()),
    ("Niggli.typeII_obtuse", ("main Proposition A11(i)",),
     "A type-II Niggli cell with d = -a-b-c is an obtuse superbase.", set()),
    ("Niggli.typeII_d_conorms", ("main Proposition A11(i)",),
     "For a type-II Niggli cell, the conorm of d with x is at least x.x/2 minus the smaller product of x.", set()),
    ("Niggli.typeII_separation", ("main Proposition A11(i)",),
     "Every conorm on d exceeds every conorm off d that is at most tau, when tau < a.a/4.", set()),
    ("Niggli.typeI_no_orthogonal_obtuse", ("main Proposition A11(ii)",),
     "For a type-I cell none of the four orthogonal superbases is obtuse.", set()),
]


def _lake(*args, timeout=7200):
    return subprocess.run(["lake", *args], cwd=paths.FORMAL, capture_output=True, text=True,
                          timeout=timeout)


@pytest.fixture(scope="module")
def axioms():
    b = _lake("build")
    assert b.returncode == 0, b.stdout[-3000:] + b.stderr[-3000:]
    r = _lake("env", "lean", "audit/axioms.lean", timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = {}
    for m in re.finditer(r"'SellingClosure\.([\w.]+)' depends on axioms: \[([^\]]*)\]", r.stdout):
        out[m.group(1)] = {a.strip() for a in m.group(2).replace("\n", " ").split(",") if a.strip()}
    for m in re.finditer(r"'SellingClosure\.([\w.]+)' does not depend on any axioms", r.stdout):
        out[m.group(1)] = set()
    return out


def _make(name, where, statement, native):
    def check(axioms):
        assert name in axioms, f"{name} missing from the axiom audit"
        got = axioms[name]
        assert "sorryAx" not in got
        nat = {a.split("._native")[0] for a in got if "._native.native_decide" in a}
        assert got - {a for a in got if "._native.native_decide" in a} <= STANDARD, got
        assert nat == native, (name, nat)
    check.__doc__ = statement + " (Lean: " + name + ")"
    fname = "test_lean_" + name.replace(".", "_")
    check.__name__ = check.__qualname__ = fname
    check.__module__ = __name__
    return claim("z62", *where, kind="formal", needs=("lean", "slow"))(check)


for _t in THEOREMS:
    globals()["test_lean_" + _t[0].replace(".", "_")] = _make(*_t)


@claim("z62", "SI §S7", kind="formal")
def test_lean_copy_of_U_is_the_deposited_table():
    """The Lean copy of U (SellingClosure/Data.lean) is regenerated identically from the
    deposited table universal_operators_columns.json, whose SHA-256 it records."""
    r = subprocess.run([sys.executable, "check_data.py"], cwd=paths.FORMAL, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "recorded" in r.stdout and "NOT recorded" not in r.stdout
    assert "identical" in r.stdout and "DIFFERS" not in r.stdout


# the statements as the supporting information typesets them (Table S12)
TEX = {
    "arrow_mem_U": r"Completeness of $\mathcal U$: every unimodular matrix between two obtuse superbases of any lattice is in the deposited table.",
    "U_arrow": r"Every element of $\mathcal U$ is an arrow of every lattice on some face.",
    "arrow_zero_pattern": r"Arrows depend only on the zero pattern: an arrow for one metric is an arrow for every metric with the same zero conorms.",
    "box_of_conn": r"The window of Lemma~A2: $\gamma=0$ on a connected support puts a vector in $[-7,7]^3$.",
    "U_swaps": r"$\mathcal U$ is closed under the 48 swaps.",
    "U_inv": r"$\mathcal U$ is closed under inversion.",
    "excess_identity": r"The class excess identity of Lemma~A1.",
    "partial_shortest": r"The partial sums of an obtuse superbase are shortest in their parity classes.",
    "price_mul": r"$\pi(h_1h_2)\le\pi(h_1)+\pi(h_2)$.",
    "price_inv": r"$\pi(h^{-1})=\pi(h)$.",
    "coset_lower": r"$H_{\epsilon-p_0}\Pop_0\subseteq C_\epsilon$.",
    "coset_upper": r"$C_\epsilon\subseteq H_{\epsilon+p_0}\Pop_0$.",
    "Hgroup": r"With no price in $(\epsilon,2\epsilon]$, $H_\epsilon$ is a group.",
    "candidates_eq_coset": r"With no price in $(\epsilon-p_0,\epsilon+p_0]$, $C_\epsilon=H_\epsilon\Pop_0$.",
    "stab_sub_H": r"$\mathrm{Aut}(\Gmet_0)\subseteq H_{2\nu}(\Gmet_R)$.",
    "H_sub_H": r"$H_\epsilon(\Gmet_R)\subseteq H_{\epsilon+2\nu}(\Gmet_0)$.",
    "twofold_midpoint_fixed": r"For a twofold $g$, the midpoint of $x$ and $g\cdot x$ is fixed by $g$.",
    "price_le_two_dist_fixed": r"The price of $g$ is at most twice the distance to the metrics $g$ fixes.",
    "Niggli.typeII_obtuse": r"A type~II Niggli cell with $d=-a-b-c$ is an obtuse superbase.",
    "Niggli.typeII_d_conorms": r"For a type~II Niggli cell, $p_{dx}\ge\frac12x\cdot x-\min(|x\cdot y|,|x\cdot z|)$.",
    "Niggli.typeII_separation": r"Every conorm on $d$ exceeds every conorm off $d$ that is at most $\tau<\frac14a\cdot a$.",
    "Niggli.typeI_no_orthogonal_obtuse": r"For a type~I cell none of the four orthogonal superbases is obtuse.",
}
assert set(TEX) == {t[0] for t in THEOREMS}
