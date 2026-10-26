# sellingproofs — the checks behind *The Selling closure*

This package holds every machine-checked proof and every computation behind the
statements and numbers of P. H. Zwart, *The Selling closure: a complete operator set for
reindexing, and the price of pseudo-symmetry* (Acta Cryst. A) and its supporting information.  Each check is a
Python function registered with the number of the statement it verifies (z-number) and
the place in the paper where that statement is made:

```python
@claim("z52", "main §3.3", "main Table 4", kind="exhaustive")
def test_universal_operator_set():
    """U has 6096 = 48 x 127 distinct unimodular elements with entries in {-3, ..., 3}, ..."""
```

The first paragraph of the docstring is the statement as the paper makes it; the body
asserts it.  All checks are ordinary pytest tests, and every test in the package must be
registered (an unregistered test is an error).

## Install and run

From a checkout of the deposited repository:

```bash
pip install -e .                          # the agentsg library, at the deposited commit
pip install -e manuscript/supplement      # this package
python -m sellingproofs pin               # library version, data hashes, git commit
python -m sellingproofs run --fast        # every claim except the slow ones
python -m sellingproofs run               # everything (z56 alone takes about 25 min)
python -m sellingproofs run z52 z60       # selected statements
python -m sellingproofs run --strict-needs  # fail instead of skipping when data are absent
python -m sellingproofs ledger            # the list of claims (also --format tex|json)
```

Without installing, put both source trees on the path:
`PYTHONPATH=agentsg/src:manuscript/supplement/src python -m sellingproofs ...`.

## Kinds of claim

| kind | meaning |
|---|---|
| formal | machine-checked proof in Lean 4 + Mathlib (`formal/`); the claim builds the development and asserts each theorem's axioms |
| exhaustive | a finite computation over every case; the assertion is exact |
| sampled | an assertion over seeded random samples (spot checks) |
| certified | an exact rational certificate |
| benchmark | a number printed in the paper, recomputed from the deposited benchmark results |
| pdb | a number printed in the paper, recomputed from the PDB results |
| reproduce | a deposited results file is regenerated identically by rerunning its script |

A claim that needs something absent (the PDB snapshot, cctbx, Lean, SELLA, NCDist) is
skipped with the reason, or fails under `--strict-needs`.  The PDB snapshot is on Zenodo
(doi:10.5281/zenodo.22986222); put `pdb_cells.duckdb` in the repository's `data/`, or point
`SELLINGPROOFS_DATA` at its directory.  cctbx must be importable for the claims that compare
with Niggli reduction.  The Lean claims need `lake` on the path (Lean 4.35.0, Mathlib
v4.35.0, pinned in `formal/`).

## Layout

| path | contents |
|---|---|
| `src/sellingproofs/claims/test_zNN_*.py` | the claims, one module per z-number |
| `src/sellingproofs/registry.py` | `@claim` and the ledger |
| `src/sellingproofs/pin.py` | the library version and SHA-256 hashes of the operator tables |
| `benchmark/` | the benchmark, the comparison and PDB scripts, their inputs and their deposited results |
| `formal/` | the Lean development, its data generator and the axiom audit |

## Pinning

The claims call the `agentsg` library of the same repository.  `python -m sellingproofs pin`
prints the commit of the checkout, the library version (pinned to 0.5.0) and whether the
operator tables have the recorded SHA-256 hashes; the deposited archive is a tagged commit
of https://github.com/phzwart/agentsg.

## Use of AI

This work used AI models extensively for coding, proofs, insight and writing.  To ensure
the correctness of the results, every statement and proof that can be computed has a check
here: a machine-checked proof in Lean, or a Python function that verifies it by exhaustive
enumeration or by spot checks on seeded samples.
