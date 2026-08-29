# Appendix Z computational checks

Runnable counterparts to every `\zitem{z:…}` in [`../main _v8.tex`](../main%20_v8.tex)
(and earlier drafts). Each module maps to one check and calls `agentsg.cell`
(no forked lattice math).

## Run

From the repo root:

```bash
PYTHONPATH=agentsg/src pytest manuscript/appendix_z -m "not slow"
PYTHONPATH=agentsg/src pytest manuscript/appendix_z          # include slow / data
```

Or from this directory (pytest.ini sets `pythonpath`):

```bash
pytest -m "not slow"
```

## Markers

| marker | meaning |
|--------|---------|
| `zcheck` | Appendix Z item |
| `slow` | benchmark / large Monte Carlo |
| `needs_pdb` | needs `data/pdb_cells.duckdb` or `analysis/data/pdb_roots.npz` |
| `needs_xfel` | needs `data/blac_new_v0_nomulti.stream` |
| `symbolic` | needs `sympy` |

## RNG / tolerances

- All stochastic draws use named fixed seeds in `helpers.py`.
- Exhaustive claims (closure sizes, fibres, coset orders, Schoenberg sign) assert **exactly**.
- Empirical medians (Table C1, etc.) pin the **manuscript printed value ±10 %** via `assert_within_pct`.

## Map

| label | file |
|-------|------|
| `z:reduction-step` | `test_z01_reduction_step.py` |
| `z:closure-count` | `test_z02_closure_count.py` |
| `z:type-rule` | `test_z03_type_rule.py` |
| `z:pdb-types` | `test_z04_pdb_types.py` |
| `z:one-key` | `test_z05_one_key.py` |
| `z:trajectory` | `test_z06_trajectory.py` |
| `z:lowerbound` | `test_z07_lowerbound.py` |
| `z:euclid` | `test_z08_euclid.py` |
| `z:fibre` | `test_z09_fibre.py` |
| `z:d7` | `test_z10_d7.py` |
| `z:noise` | `test_z11_noise.py` |
| `z:stabilise` | `test_z12_stabilise.py` |
| `z:floor-invariance` | `test_z13_floor_invariance.py` |
| `z:coset` | `test_z14_coset.py` |
| `z:reindex-proc` | `test_z15_reindex_proc.py` |
| `z:verify-tol` | `test_z16_verify_tol.py` |
| `z:noisy-frames` | `test_z17_noisy_frames.py` |
| `z:lepage` | `test_z18_lepage.py` |
| `z:reindex-bench` | `test_z19_reindex_bench.py` |
| `z:timing` | `test_z20_timing.py` |
| `z:embedding` | `test_z21_embedding.py` |
| `z:audit` | `test_z22_audit.py` |
| `z:effrank` | `test_z23_effrank.py` |
| `z:lysozyme` | `test_z24_lysozyme.py` |
| `z:xfel` | `test_z25_xfel.py` |
| `z:brute-complete` | `test_z26_brute_complete.py` |
| `z:coset-reps` | `test_z27_coset_reps.py` |
| `z:pseudo` | `test_z28_pseudo.py` |

`helpers.prepare_reference` / `reindex_frame` mirror the App. D pseudo-code so
`z:reindex-bench` can assert the reference closure is enumerated once.

New in `main_v8`: `z:coset-reps`, `z:pseudo`, `z:brute-complete` (includes the
Niggli CoB identity `N^T G N == G_red`).
