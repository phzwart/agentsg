"""z:delone-derived — the 24 Delone sorts derived from the closure groupoid.

Input: the zero patterns whose complement connects the four vectors, and
Kurlin's closure of one generic lattice per pattern. On a stratum every
arrow, composed with a relabelling, permutes the conorm slots; a sort is a
tie pattern that is closed (the permutations preserving it have exactly its
blocks as orbits), counted once across the members of its closure. The
result is compared with the Selling patterns of Andrews, Bernstein & Sauter
(2023); the loop groups come out as the holohedries.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import itertools

import numpy as np
import pytest

from . import test_z44_widened_complete as t  # noqa: E402

SL = t.SL; PERM4 = list(itertools.permutations(range(4)))

def connected(es):
    adj = {v: set() for v in range(4)}
    for a, b in es: adj[a].add(b); adj[b].add(a)
    s = {0}; st = [0]
    while st:
        x = st.pop()
        for y in adj[x] - s: s.add(y); st.append(y)
    return len(s) == 4

def slot_perm(pm):
    idx = {tuple(sorted(e)): n for n, e in enumerate(SL)}
    return [idx[tuple(sorted((pm[a], pm[b])))] for a, b in SL]

def sella(c): return [c[5], c[4], c[3], c[0], c[1], c[2]]

def pattern_string(vals, tol):
    out, seen = [], []
    for v in vals:
        if abs(v) < tol: out.append("0"); continue
        for i, u in enumerate(seen):
            if abs(u - v) < tol: out.append("rstuvw"[i]); break
        else: seen.append(v); out.append("rstuvw"[len(seen) - 1])
    return "".join(out)

def closure_data(p):
    G = t._gram(p); tol = 1e-7 * np.abs(p).max(); _, mem = t._generate(p, 0.0, 1e18)
    best, aut, fix = None, 0, 0
    k0 = t._key(np.array(t.S0, float))
    for k, S in mem.items():
        for pm in PERM4:
            s = pattern_string(sella(t._cn(S[list(pm)], G)), tol)
            best = s if best is None or s < best else best
            for sg in (1, -1):
                W = (sg * S[list(pm)])[1:4].T
                if np.allclose(W.T @ G @ W, G, atol=1e-7 * np.abs(G).max()):
                    aut += 1; fix += (k == k0)
    return best, len(mem), aut, fix

def derive(seed=1):
    rng = np.random.default_rng(seed)
    Zs = {}
    for r in range(4):
        for Z in itertools.combinations(range(6), r):
            if not connected([SL[k] for k in range(6) if k not in Z]): continue
            rep = min(tuple(sorted(slot_perm(pm)[z] for z in Z)) for pm in PERM4)
            Zs.setdefault(rep, Z)
    sorts = {}
    for Z in Zs.values():
        free = [k for k in range(6) if k not in Z]
        p = np.zeros(6); p[free] = rng.uniform(300, 2500, len(free))
        G = t._gram(p); tol = 1e-7 * np.abs(p).max(); _, mem = t._generate(p, 0.0, 1e18)
        Pi = set()
        for S in mem.values():
            for pm in PERM4:
                c = t._cn(S[list(pm)], G)
                if not all((abs(c[k]) < tol) == (k in Z) for k in range(6)): continue
                Pi.add(tuple(int(np.argmin(np.abs(p - c[k]))) if k not in Z else k for k in range(6)))
        def partitions(s):
            if not s: yield []; return
            f, rest = s[0], s[1:]
            for P in partitions(rest):
                for i in range(len(P)): yield P[:i] + [[f] + P[i]] + P[i + 1:]
                yield [[f]] + P
        for P in partitions(free):
            block = {k: i for i, b in enumerate(P) for k in b}
            stab = [s for s in Pi if all(block[s[k]] == block[k] for k in free)]
            if any({s[k] for s in stab} != {j for j in free if block[j] == block[k]} for k in free): continue
            q = np.zeros(6); vals = rng.uniform(300, 2500, len(P))
            for i, b in enumerate(P):
                for k in b: q[k] = vals[i]
            can, nmem, aut, fix = closure_data(q)
            sorts.setdefault(can, dict(zeros=len(Z), members=nmem, aut=aut, fix=fix, Pi=len(Pi)))
    return Zs, sorts

SELLA = {"C1":"rrrrrr","C3":"rr0rr0","C5":"000rrr","T1":"rrsrrs","T2":"rr0rrs","T5":"000rrs","R1":"rrrsss","R3":"rr0sr0",
"O1A":"rrsrrt","O1B":"rstrst","O2":"rs0srt","O3":"rs0rs0","O4":"00rsst","O5":"000rst","M1A":"rrsttu","M1B":"rstrsu",
"M2A":"rs0stu","M2B":"rs0rst","M3":"rs0ts0","M4":"00rstu","H4":"00rrrs","A1":"rstuvw","A2":"rs0tuv","A3":"rs0tu0"}

def sella_canonical(seed=2):
    rng = np.random.default_rng(seed); out = {}
    for name, pat in SELLA.items():
        letters = sorted(set(pat) - {"0"}); val = dict(zip(letters, rng.uniform(300, 2500, len(letters))))
        s = [0.0 if ch == "0" else val[ch] for ch in pat]
        out[closure_data(np.array([s[3], s[4], s[5], s[2], s[1], s[0]]))[0]] = name
    return out



@claim("z48", "SI §S5", kind="exhaustive")
def test_delone_sorts_from_the_groupoid():
    r"""Delone's 24 sorts are recovered from the closure groupoid alone: the conorm patterns and their equalities."""
    Zs, sorts = derive(seed=1)
    assert len(Zs) == 6
    assert len(sorts) == 24
    sel = sella_canonical(seed=2)
    assert set(sorts) == set(sel)                     # exactly SELLA's 24
    holo = {"C": 48, "T": 16, "R": 12, "H": 24, "O": 8, "M": 4, "A": 2}
    for can, d in sorts.items():
        assert d["aut"] == holo[sel[can][0]], sel[can]
        assert d["members"] in (1, 2, 3, 6, 16)
