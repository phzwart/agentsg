# Ralph backlog — code-quality fixes (M/S review)

Ordered atomic tasks. Work **top to bottom**. Flip `[ ]` → `[x]` only after
`bash ralph/verify.sh` is green and the commit is made.

Scope: code-quality only. MCP/skill facade (R1–R6) is **out of scope**.

---

## Phase A — foundational refactors

### [x] A1 — Extract public rational solver (M1)

**Goal.** Stop cross-module imports of private `_solve_affine` / `_rref`.

**Do.**
1. Move `_rref` and `_solve_affine` from `src/agentsg/wyckoff.py` into a new
   public module `src/agentsg/rational_solve.py` (preferred) **or** into
   `src/agentsg/linalg.py` if that keeps the surface cleaner.
2. Export the public names (e.g. `rref`, `solve_affine`) from the new module.
3. Keep back-compat aliases `_rref` / `_solve_affine` in `wyckoff.py` that
   re-export from the new home (so existing tests importing from wyckoff still
   work).
4. Update imports in `src/agentsg/harker.py` and `src/agentsg/identify.py` to
   use the public module (not wyckoff's private aliases).
5. Export from `src/agentsg/__init__.py` only if useful; not required.

**Verify.** Targeted: `tests/test_wyckoff.py`, `tests/test_harker.py`,
`tests/test_identify.py`. Then full suite.

**Done when.** No remaining `from .wyckoff import _solve_affine` / `_rref`
outside `wyckoff.py` itself (aliases OK inside wyckoff).

---

### [x] A2 — Consolidate metric ↔ param conversion (M2)

**Goal.** One clamping policy; delete ~6 duplicate converters.

**Do.**
1. In `src/agentsg/cell/metric.py`, add (or promote) module-level helpers:
   - `metric_tensor(cell)` — from `(a,b,c,α,β,γ)` to 3×3 `G`
   - `cell_from_metric(G)` / `params_from_metric(G)` — from `G` to params,
     with **one** documented clamping policy (raise on non-positive edges;
     clamp cosines to `[-1,1]` for angles). Prefer raising over silent zero.
2. Replace duplicates in:
   - `src/agentsg/lattice_symmetry.py` (`_metric_tensor`, `_params_from_metric`,
     `_cell_params`)
   - `src/agentsg/cell/g6.py` (`_cell_from_metric`)
   - `src/agentsg/cell/ambiguity.py` (inline `params()` in `resolve` /
     `surface_geometric_operators` — and any third copy)
3. `UnitCell.metric_tensor()` should call the shared helper (or vice versa)
   so both stay identical.
4. Add `tests/test_metric_helpers.py` (or extend `test_cell_metric.py`) that
   round-trips a few cells and asserts lattice_symmetry / g6 / ambiguity paths
   agree on params for the same `G`.

**Verify.** Targeted: `tests/test_cell_metric.py`, `tests/test_lattice_symmetry.py`,
`tests/test_g6.py`, `tests/test_ambiguity.py`, plus the new consistency test.
Then full suite.

**Done when.** No private reimplementation of cell↔G conversion remains outside
`metric.py` (thin wrappers that call the shared helpers are OK).

---

### [x] A3 — Promote `_discrete_allowed_origins` (M1)

**Goal.** Stop `asu.py` importing a private from `semi_invariants`.

**Do.**
1. Rename / promote `_discrete_allowed_origins` to a public
   `discrete_allowed_origins` in `src/agentsg/semi_invariants.py`.
2. Keep `_discrete_allowed_origins` as a back-compat alias.
3. Update `src/agentsg/asu.py` to import the public name.
4. Optionally export from package `__init__` / `semi_invariants` consumers.

**Verify.** Targeted: `tests/test_semi_invariants.py`, `tests/test_dirichlet_asu.py`.
Then full suite.

---

## Phase B — correctness / robustness

### [ ] B1 — Fix `_cubic_roots` degenerate branch (S4)

**Goal.** SPD inertia tensors must not silently return three identical bogus
eigenvalues when `disc > 0` due to float noise.

**Do.**
1. In `src/agentsg/asu.py` `_cubic_roots` (near the non-three-real branch ~386–392):
   for near-degenerate SPD cases, clamp toward the trigonometric (three-real)
   branch, **or** assert / raise if three distinct real roots cannot be recovered.
2. Add a near-degenerate inertia / sphericity test in `tests/test_dirichlet_asu.py`
   (or a small dedicated test) that exercises a nearly-equal-eigenvalue case.

**Verify.** Targeted: `tests/test_dirichlet_asu.py` (+ new test). Then full suite.

---

### [ ] B2 — Niggli reduction fuzz vs oracles (S5)

**Goal.** Catch rare det-flips / non-idempotent reduction on near-degenerate cells.

**Do.**
1. Add `tests/test_reduction_fuzz.py` that:
   - draws random + near-degenerate cells
   - compares `niggli_reduce` to gemmi and/or spglib where available
   - asserts `det(M) in {-1, +1}`
   - asserts idempotency: reducing the reduced cell is a no-op (params match
     within tolerance; CoB is identity up to sign conventions you document)
2. If fuzz exposes a bug in `src/agentsg/cell/reduction.py` Step 3/4
   (sign-normalization / det=+1 path), fix it.
3. Keep the test bounded (runtime: a few seconds, not minutes).

**Verify.** Targeted: `tests/test_reduction_fuzz.py`, `tests/test_cell_reduction.py`.
Then full suite.

---

### [ ] B3 — Make MC ASU estimators explicit (S2)

**Goal.** Monte-Carlo nature of Dirichlet ASU metrics must be obvious.

**Do.**
1. In `src/agentsg/asu.py`, for `volume_fraction`, `sphericity`,
   `inertia_eigenvalues` (and related score methods): put "Monte-Carlo estimate"
   (or equivalent) in the **first line** of each docstring.
2. Expose `n_samples` clearly (keep existing `n=` as the parameter or alias;
   document it). Do not break call sites.
3. Optionally record `n_samples` in `OptimizedAsu.metrics`.

**Verify.** Targeted: `tests/test_dirichlet_asu.py`. Then full suite.

---

### [ ] B4 — Doc-only: crystal_system sentinel + Harker comment (S3, S6)

**Goal.** Documentation / comment clarity only — no behavior change.

**Do.**
1. In `src/agentsg/lattice_symmetry.py`, document that `crystal_system` may be a
   non-system sentinel like `"order-N"` when closure order is not a known
   holohedry (around `_ORDER_TO_SYSTEM` / `lattice_symmetry` return).
2. In `src/agentsg/harker.py` `_normalize_constraint`, replace the garbled
   derivation comment (~lines 59–63) with a clear explanation of the `D/g`
   scaling of the constant.

**Verify.** Full suite (docs-only; suite must still pass). Spot-check that no
code logic changed beyond comments/docstrings.

---

## Phase C — API surface hygiene

### [ ] C1 — Soften "complete invariant" claim for triclinic V1 (S1)

**Goal.** Public wording must not over-claim Kurlin completeness for generic
triclinic.

**Do.**
1. Soften wording in `src/agentsg/cell/rootform.py` module / `root_invariant`
   docstrings: complete for V2–V5; collision-free-in-practice similarity key for
   generic triclinic (V1).
2. Align the same caveat in `agentsg/README.md` wherever root invariant is
   described as complete.

**Verify.** Full suite (docs-only).

---

### [ ] C2 — Mark g6 deficiency APIs diagnostic vs rootform canonical (M3)

**Goal.** Units and preferred API must be unambiguous (G6 Å² vs Kurlin Å).

**Do.**
1. In `src/agentsg/cell/g6.py`, mark `distance_to_symmetry` /
   `symmetry_deficiency_spectrum` (and related) as diagnostic/legacy in
   docstrings; point callers to Kurlin / `rootform` equivalents for canonical
   deficiency scores.
2. Update `agentsg/README.md` briefly so the preferred path is clear.

**Verify.** Full suite (docs-only). Keep symbols importable.

---

### [ ] C3 — Demote `GENERATOR_TABLE` from public API (M5)

**Goal.** Starter table must not look like the way to get operators.

**Do.**
1. Remove `GENERATOR_TABLE` from `src/agentsg/__init__.py` `__all__` (and from
   the top-level import if it is re-exported there).
2. Keep `from agentsg.generators import GENERATOR_TABLE` working for
   `tests/test_group_closure.py`.
3. Optionally add a one-line note in `generators.py` that Hall/`space_group`
   is authoritative.

**Verify.** Targeted: `tests/test_group_closure.py`. Confirm
`from agentsg import GENERATOR_TABLE` is no longer the advertised public path
(may still work via submodule). Then full suite.
