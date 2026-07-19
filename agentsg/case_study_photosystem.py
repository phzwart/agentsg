"""
Case study: Photosystem serial-crystallography cell manifold, reduction flip,
and reindexing to a reference setting -- built entirely on agentsg.

Pipeline
--------
1. Reference monoclinic Photosystem cells 2WSC / 4XK8 (P 1 21 1). Cached from a
   real RCSB fetch by default; set FETCH=True to pull them live via celldb.
2. Simulate serial-crystallography frames: many copies of each crystal with
   per-frame measurement noise, plus genuine reindexing variants (the pipeline
   legitimately picks different valid monoclinic cell-choices per frame) and a
   mis-indexed c-doubled supercell contaminant.
3. Niggli-reduce every frame -> the reduction flips discontinuously under noise
   (multiple change-of-basis matrices for ONE crystal).
4. Compute the root-invariant (Kurlin 2022) distance matrix -- continuous ACROSS
   every flip -- and embed it (t-SNE / MDS on the precomputed metric).
5. Reindex each same-lattice frame back to the reference setting via the
   reindexing coset.

Run: python case_study_photosystem.py   (needs agentsg[db] + scikit-learn)
"""
import math, random, json
import numpy as np
from agentsg.cell.celldb import fetch_pdb_cells
from agentsg.cell.metric import UnitCell
from agentsg.cell.reduction import niggli_reduce
from agentsg.cell.rootform import root_invariant, root_distance
from agentsg.cell.reindex import reindexing_operator
from agentsg.cell.g6 import _transform_metric


def reindex_cell(cell, M):
    G = UnitCell(*cell).metric_tensor()
    Gp = _transform_metric(G, M)
    a = math.sqrt(Gp[0][0]); b = math.sqrt(Gp[1][1]); c = math.sqrt(Gp[2][2])
    ang = lambda x: math.degrees(math.acos(max(-1, min(1, x))))
    return (a, b, c, ang(Gp[1][2] / (b * c)), ang(Gp[0][2] / (a * c)),
            ang(Gp[0][1] / (a * b)))


# Fallback cell parameters for offline runs. These are the RCSB values for the
# two entries; set FETCH=True to pull them live via celldb and verify (the live
# fetch is the source of truth if the PDB is ever revised).
FETCH = False
_CACHED = {"2WSC": (120.7, 189.1, 129.4, 90.0, 91.2, 90.0),   # PSII, P 1 21 1
           "4XK8": (165.6, 192.2, 175.1, 90.0, 91.4, 90.0)}   # PSII, P 1 21 1


def reference_cells():
    """Return (REF_A, REF_B) for 2WSC / 4XK8, fetched live if FETCH else cached."""
    if FETCH:
        got = {c["pdb_id"]: (c["a"], c["b"], c["c"], c["alpha"], c["beta"], c["gamma"])
               for c in fetch_pdb_cells(["2WSC", "4XK8"])}
        return got["2WSC"], got["4XK8"]
    return _CACHED["2WSC"], _CACHED["4XK8"]


def build_dataset(seed=20260717):
    rng = random.Random(seed)
    REF_A, REF_B = reference_cells()
    reindex_ops = [((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                   ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
                   ((-1, 0, 0), (0, 1, 0), (-1, 0, 1))]

    def perturb(cell, ls=0.6, as_=0.3):
        a, b, c, al, be, ga = cell
        f = lambda v: v * (1 + rng.gauss(0, ls / 100))
        g = lambda v: v + rng.gauss(0, as_)
        return (f(a), f(b), f(c), g(al), g(be), g(ga))

    frames = []
    for _ in range(80):
        frames.append((perturb(reindex_cell(REF_A, rng.choice(reindex_ops))), "2WSC (ref)"))
    for _ in range(40):
        frames.append((perturb(reindex_cell(REF_B, rng.choice(reindex_ops))), "4XK8"))
    for _ in range(12):
        frames.append((perturb(reindex_cell(REF_A, ((1, 0, 0), (0, 1, 0), (0, 0, 2)))),
                       "2WSC c-doubled"))
    return frames, REF_A, REF_B


def main():
    frames, REF_A, REF_B = build_dataset()
    reduced, invariants, Ts = [], [], []
    for cell, _ in frames:
        red, T = niggli_reduce(*cell)
        reduced.append(red); invariants.append(root_invariant(red))
        Ts.append(tuple(tuple(r) for r in T))

    n = len(frames)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = root_distance(reduced[i], reduced[j])

    # reindex reference-lattice frames back to REF_A
    ok = 0
    for (cell, label) in frames:
        if label != "2WSC (ref)":
            continue
        P = reindexing_operator(cell, REF_A, length_tol_pct=3.0, angle_tol_deg=3.0)
        ok += P is not None
    print(f"{n} frames; reduction-flip settings in reference crystal: "
          f"{len(set(t for t, (_, l) in zip(Ts, frames) if l == '2WSC (ref)'))}")
    print(f"reindexed back to reference: {ok}/80")

    # embed the distance matrix (t-SNE / MDS on the precomputed root-invariant
    # metric); optional -- only if scikit-learn is available.
    try:
        from sklearn.manifold import TSNE, MDS
        emb = TSNE(n_components=2, metric="precomputed", init="random",
                   perplexity=15, random_state=42).fit_transform(D)
        emb_mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                      normalized_stress="auto").fit_transform(D)
        print(f"embedded: t-SNE {emb.shape}, MDS {emb_mds.shape}")
    except ImportError:
        emb = emb_mds = None
        print("scikit-learn not installed -> skipped embedding")
    return D, invariants, frames, Ts, emb, emb_mds


if __name__ == "__main__":
    main()
