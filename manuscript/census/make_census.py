"""Wall-incidence census of a cell database.

    python manuscript/census/make_census.py DB [--sample N] [--seed 0] [--out manuscript/census]

Reads a CellDatabase read-only. Does not modify the library. Closures and
sort paths run in a process pool; the metric batch uses CUDA when PyTorch
can see it.
"""
from __future__ import annotations

import signal
import csv
import itertools
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "manuscript" / "appendix_z"))
sys.path.insert(0, str(REPO / "manuscript" / "benchmark"))

import test_z44_widened_complete as z44  # noqa: E402
from evaluate import members_by  # noqa: E402
from test_z50_specialisations import EXPECTED  # noqa: E402
from test_z54_table_from_operators import S6_TO_SLOT, TABLE, _sella_partition  # noqa: E402
from test_z57_perslot_bands import consistent_construct, delta_obtuse  # noqa: E402

from agentsg.cell.celldb import CellDatabase  # noqa: E402
from agentsg.cell.cod import centring_symbol  # noqa: E402
from agentsg.cell.metric import params_from_metric  # noqa: E402
from agentsg.cell.reduction import niggli_gk  # noqa: E402
from agentsg.cell.selling_cob import COB_COLUMNS, parse_cob_columns  # noqa: E402

SL = list(z44.SL)
S0 = np.array(z44.S0, float)
SLOT_OF = {pair: n for n, pair in enumerate(SL)}
PERM_SRC = []
for _perm in itertools.permutations(range(4)):
    src = np.empty(6, dtype=np.int64)
    for n, (i, j) in enumerate(SL):
        a, b = _perm[i], _perm[j]
        src[n] = SLOT_OF[(a, b)] if (a, b) in SLOT_OF else SLOT_OF[(b, a)]
    PERM_SRC.append(src)
PERM_SRC = np.stack(PERM_SRC)  # (24, 6); each counts twice with the global sign

def _relabel(zeros, classes, src):
    """Move a canonical (zeros, classes) pattern along one label permutation."""
    new_zero = frozenset(n for n in range(6) if int(src[n]) in zeros)
    old_class = {}
    for group in classes:
        cid = min(group)
        for slot in group:
            old_class[slot] = cid
    groups = {}
    for n in range(6):
        if n in new_zero:
            continue
        groups.setdefault(old_class[int(src[n])], []).append(n)
    return new_zero, frozenset(frozenset(v) for v in groups.values())


SORT_OF = {}
SORT_PERM = {}
for _name, _row in TABLE.items():
    _part, _zs = _sella_partition(_row[0])
    for _ip, _src in enumerate(PERM_SRC):
        _key = _relabel(_zs, _part, _src)
        SORT_OF.setdefault(_key, _name)
        SORT_PERM.setdefault(_key, _ip)
SORT_NAMES = sorted(SORT_OF.values())
LETTERS = {}
for _name, _row in TABLE.items():
    lab = ["0"] * 6
    for s6, ch in enumerate(_row[0]):
        lab[S6_TO_SLOT[s6]] = ch
    LETTERS[_name] = lab

EDGE_OF = {}
for (_src, _dst), (_conds, _nshape) in EXPECTED.items():
    EDGE_OF[(_src, _dst)] = tuple(sorted(_conds))

SYSTEMS = (
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic", "unknown",
)
SYSTEM_ORDER = {
    "triclinic": 2, "monoclinic": 4, "orthorhombic": 8, "tetragonal": 16,
    "trigonal": 12, "hexagonal": 24, "cubic": 48, "unknown": 0,
}
VISIBLE_SYSTEM = {
    48: "cubic", 24: "hexagonal", 16: "tetragonal", 12: "rhombohedral",
    8: "orthorhombic", 4: "monoclinic", 2: "triclinic",
}
SIGMA_NOMINAL = math.sin(math.radians(0.03))
SIGMA_3 = math.sin(math.radians(3.0)) / 4.0
RATIO_BINS = ("<1.5", "1.5-2.4", "2.4-3", "3-5", "5-10", ">10")
CLOSURE_BUCKETS = (1, 2, 3, 6, 16)
LOG_EDGES = [-6.0 + 0.5 * i for i in range(13)]  # -6 .. 0 inclusive
POP_LEVELS = [10 ** (-6 + 0.5 * i) for i in range(13)]
SP = np.stack([
    np.diag(sign) @ np.eye(3)[list(perm)]
    for perm in itertools.permutations(range(3))
    for sign in itertools.product((1.0, -1.0), repeat=3)
])
SELECT_COLS = (
    ["pdb_id", "source", "a", "b", "c", "alpha", "beta", "gamma", "volume",
     "sg_number", "sg_hm", "formula",
     "red_a", "red_b", "red_c", "red_alpha", "red_beta", "red_gamma"]
    + list(COB_COLUMNS)
    + ["sig_a", "sig_b", "sig_c", "sig_alpha", "sig_beta", "sig_gamma"]
)


def reported_system(number) -> str:
    if number is None or (isinstance(number, float) and math.isnan(number)) or int(number) <= 0:
        return "unknown"
    n = int(number)
    if n <= 2:
        return "triclinic"
    if n <= 15:
        return "monoclinic"
    if n <= 74:
        return "orthorhombic"
    if n <= 142:
        return "tetragonal"
    if n <= 167:
        return "trigonal"
    if n <= 194:
        return "hexagonal"
    if n <= 230:
        return "cubic"
    return "unknown"


def lattice_letter(symbol: str) -> str:
    centred = centring_symbol(symbol) or ""
    text = str(centred).strip().lstrip("-")
    return text[0].upper() if text else ""


def log_bin(value: float, *, zero_label: str | None = None) -> str:
    if zero_label is not None and value <= 0:
        return zero_label
    if value <= 0:
        return "<=0"
    lx = math.log10(value)
    if lx < -6:
        return "<-6"
    if lx >= 0:
        return ">=0"
    left = math.floor(lx * 2.0) / 2.0
    return f"[{left:.1f},{left + 0.5:.1f})"


def bin_labels(zero_label: str | None) -> list[str]:
    labels = []
    if zero_label:
        labels.append(zero_label)
    labels.append("<-6")
    for left in LOG_EDGES[:-1]:
        labels.append(f"[{left:.1f},{left + 0.5:.1f})")
    labels.append(">=0")
    return labels


def _metric(cells, xp):
    """Gram matrices of cells (N, 6), angles in degrees."""
    a, b, c = cells[:, 0], cells[:, 1], cells[:, 2]
    al, be, ga = xp.deg2rad(cells[:, 3]), xp.deg2rad(cells[:, 4]), xp.deg2rad(cells[:, 5])
    ca, cb, cg = xp.cos(al), xp.cos(be), xp.cos(ga)
    if xp.__name__ == "torch":
        g = xp.zeros((cells.shape[0], 3, 3), dtype=cells.dtype, device=cells.device)
    else:
        g = np.zeros((cells.shape[0], 3, 3), dtype=np.float64)
    g[:, 0, 0] = a * a
    g[:, 1, 1] = b * b
    g[:, 2, 2] = c * c
    g[:, 0, 1] = g[:, 1, 0] = a * b * cg
    g[:, 0, 2] = g[:, 2, 0] = a * c * cb
    g[:, 1, 2] = g[:, 2, 1] = b * c * ca
    return g


def _conorms(gram, xp):
    """Conorms and vector lengths of the reduced superbase S0 under gram (N, 3, 3)."""
    s = xp.as_tensor(S0, dtype=gram.dtype, device=gram.device) if xp.__name__ == "torch" else np.asarray(S0, dtype=np.float64)
    dots = xp.einsum("ia,nab,jb->nij", s, gram, s)
    pairs = np.asarray(SL, dtype=np.int64)
    if xp.__name__ == "torch":
        pairs = xp.as_tensor(pairs, device=gram.device)
    p = -dots[:, pairs[:, 0], pairs[:, 1]]
    if device_diag(xp):
        diag = xp.diagonal(dots, dim1=1, dim2=2)
        lengths = xp.sqrt(xp.clamp(diag, min=0))
        gmax = xp.amax(xp.abs(gram), dim=(1, 2))
    else:
        diag = np.diagonal(dots, axis1=1, axis2=2)
        lengths = np.sqrt(np.clip(diag, 0, None))
        gmax = np.max(np.abs(gram), axis=(1, 2))
    return p, lengths, gmax


def device_diag(xp):
    return xp.__name__ == "torch"


def _lam1_many(p):
    cuts = ((0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3))
    acc = []
    for group in cuts:
        inside = set(group)
        mask = np.array([((i in inside) != (j in inside)) for i, j in SL])
        acc.append(p[:, mask].sum(axis=1))
    return np.min(np.stack(acc, axis=0), axis=0)


def _xp():
    try:
        import torch
        if torch.cuda.is_available():
            return torch, "cuda"
    except ImportError:
        pass
    return np, "cpu"


def _free_sigma(cells, sigma, systems):
    """Errors of the parameters the reported system leaves free.

    A fixed angle or a length equated to another gets σ = 0. The shared edge
    takes the esd present on either copy. ``esd_ok`` requires every free
    parameter to have an esd and at least one of those esds to be positive.
    """
    n = len(cells)
    chosen = np.zeros((n, 6), dtype=np.float64)
    esd_ok = np.zeros(n, dtype=bool)
    for i in range(n):
        system = systems[i]
        a, b, c, alpha, beta, gamma = cells[i]
        sig = sigma[i]
        free = [True] * 6
        pick = np.array(sig, dtype=np.float64)
        if system == "monoclinic":
            free = [True, True, True, abs(alpha - 90) > 0.5, abs(beta - 90) > 0.5, abs(gamma - 90) > 0.5]
        elif system == "orthorhombic":
            free = [True, True, True, False, False, False]
        elif system in ("tetragonal", "hexagonal"):
            free = [True, False, True, False, False, False]
            if not np.isfinite(pick[0]) and np.isfinite(pick[1]):
                pick[0] = pick[1]
        elif system == "trigonal" and abs(gamma - 120.0) < 1.0:
            free = [True, False, True, False, False, False]
            if not np.isfinite(pick[0]) and np.isfinite(pick[1]):
                pick[0] = pick[1]
        elif system == "trigonal":
            free = [True, False, False, True, False, False]
            for src in (1, 2):
                if not np.isfinite(pick[0]) and np.isfinite(sig[src]):
                    pick[0] = sig[src]
            for src in (4, 5):
                if not np.isfinite(pick[3]) and np.isfinite(sig[src]):
                    pick[3] = sig[src]
        elif system == "cubic":
            free = [True, False, False, False, False, False]
            pick[0] = next((sig[src] for src in (0, 1, 2) if np.isfinite(sig[src])), np.nan)
        for k in range(6):
            if not free[k]:
                pick[k] = 0.0
        chosen[i] = pick
        present = all((not free[k]) or np.isfinite(pick[k]) for k in range(6))
        positive = any(free[k] and np.isfinite(pick[k]) and pick[k] > 0 for k in range(6))
        esd_ok[i] = present and positive
    return chosen, esd_ok


def prepare_metrics(red, deposited, cob, sigma, systems):
    """Vectorised conorms of the stored reduced cell, and tau from the Jacobian.

    ``cob`` is (N, 3, 3). Rows with a failed change of basis use the reduced
    cell and nominal errors. Returns numpy arrays.
    """
    xp, device = _xp()
    red_m = np.asarray(red, np.float64)
    dep_m = np.asarray(deposited, np.float64)
    cob_m = np.asarray(cob, np.float64)
    sig_m = np.asarray(sigma, np.float64)
    n = len(red_m)

    def pack(array):
        if device == "cuda":
            return xp.as_tensor(array, device="cuda", dtype=xp.float64)
        return array

    gram = _metric(pack(red_m), xp if device == "cuda" else np)
    p, lengths, gmax = _conorms(gram, xp if device == "cuda" else np)
    if device == "cuda":
        p, lengths, gmax = (t.detach().cpu().numpy() for t in (p, lengths, gmax))

    # Jacobian of deposited parameters through P^T G P. Twelve perturbed copies.
    step = np.zeros_like(dep_m)
    step[:, 0:3] = np.maximum(np.abs(dep_m[:, 0:3]), 1e-3) * 1e-6
    step[:, 3:6] = 1e-4
    eye = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    use = np.isfinite(cob_m).all(axis=(1, 2))
    cob_m = np.where(use[:, None, None], cob_m, eye)
    # A broken cob is evaluated on the reduced cell, so the band is that cell's.
    base_cells = np.where(use[:, None], dep_m, red_m)
    plus, minus = [], []
    for k in range(6):
        up, down = base_cells.copy(), base_cells.copy()
        up[:, k] += step[:, k]
        down[:, k] -= step[:, k]
        plus.append(up)
        minus.append(down)
    both = np.concatenate(plus + minus, axis=0)
    both_p = np.tile(cob_m, (12, 1, 1))
    gp = _metric(pack(both), xp if device == "cuda" else np)
    pp = pack(both_p)
    gred = xp.einsum("nia,nij,njb->nab", pp, gp, pp) if device == "cuda" else np.einsum(
        "nia,nij,njb->nab", both_p, _metric(both, np), both_p)
    pc, _, _ = _conorms(gred, xp if device == "cuda" else np)
    if device == "cuda":
        pc = pc.detach().cpu().numpy()
    pc = pc.reshape(12, n, 6)
    jac = np.empty((n, 6, 6), dtype=np.float64)
    for k in range(6):
        jac[:, :, k] = (pc[k] - pc[6 + k]) / (2.0 * step[:, k:k + 1])
    nominal = np.column_stack([
        0.001 * np.abs(red_m[:, 0]), 0.001 * np.abs(red_m[:, 1]), 0.001 * np.abs(red_m[:, 2]),
        np.full(n, 0.03), np.full(n, 0.03), np.full(n, 0.03),
    ])
    esd_param, esd_free = _free_sigma(dep_m, sig_m, systems)
    esd_ok = esd_free & use
    used = np.where(esd_ok[:, None], esd_param, nominal)
    sig_p = np.sqrt(np.sum(jac ** 2 * used[:, None, :] ** 2, axis=2))
    tau = 3.0 * np.max(sig_p, axis=1)
    tau = np.where(np.isfinite(tau), tau, 0.0)
    tau = np.maximum(tau, 1e-9 * np.maximum(gmax, 0.0))
    kind = np.where(esd_ok, "esd", "nominal")
    return {
        "p": p, "lengths": lengths, "gmax": gmax, "lam1": _lam1_many(p),
        "tau": tau, "sig_p": sig_p, "kind": kind, "cob_ok": use, "device": device,
    }


def _classes(p, tol):
    zero = p <= tol
    parent = list(range(6))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    alive = [i for i in range(6) if not zero[i]]
    for a, b in itertools.combinations(alive, 2):
        if abs(p[a] - p[b]) <= tol:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    ids = np.empty(6, dtype=np.int64)
    ids[zero] = -1
    for i in alive:
        ids[i] = find(i)
    return zero, ids, parent


def _pattern(zero, parent):
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    zs = frozenset(i for i in range(6) if zero[i])
    groups = {}
    for i in range(6):
        if zero[i]:
            continue
        groups.setdefault(find(i), []).append(i)
    return zs, frozenset(frozenset(v) for v in groups.values())


_CLOSE_CACHE = {}
_BY_ZEROS = {}
for _pattern_key in SORT_OF:
    _BY_ZEROS.setdefault(_pattern_key[0], []).append(_pattern_key)


def _close_to_sort(zero, parent):
    """Finest Table-4 pattern containing the imposed zeros and ties.

    Those patterns are the closed sets Fix(Sym(V)). The raw partition is only
    the constraint set; it is not itself the sort.
    """
    key = _pattern(zero, parent)
    cached = _CLOSE_CACHE.get(key)
    if cached is not None:
        return cached
    if key in SORT_OF:
        found = (SORT_OF[key], SORT_PERM[key], key)
        _CLOSE_CACHE[key] = found
        return found
    raw_zs, raw_part = key
    best_name, best_perm, best_key, best_cost = "unlisted", 0, None, None
    candidates = []
    for extra in range(0, 7 - len(raw_zs)):
        for added in itertools.combinations(sorted(set(range(6)) - raw_zs), extra):
            candidates.extend(_BY_ZEROS.get(raw_zs.union(added), ()))
    for pattern in candidates:
        zs, part = pattern
        fits = True
        for group in raw_part:
            remain = [slot for slot in group if slot not in zs]
            if len(remain) <= 1:
                continue
            host = next((cl for cl in part if remain[0] in cl), None)
            if host is None or any(slot not in host for slot in remain):
                fits = False
                break
        if not fits:
            continue
        cost = len(zs) + sum(len(cl) - 1 for cl in part)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_name = SORT_OF[pattern]
            best_perm = SORT_PERM[pattern]
            best_key = pattern
    found = (best_name, best_perm, best_key)
    _CLOSE_CACHE[key] = found
    return found


def _pattern_contains(outer, inner):
    """True when the closed pattern ``outer`` contains every constraint of ``inner``."""
    if outer is None or inner is None:
        return False
    oz, opart = outer
    iz, ipart = inner
    if not iz.issubset(oz):
        return False
    for group in ipart:
        remain = [slot for slot in group if slot not in oz]
        if len(remain) <= 1:
            continue
        host = next((cl for cl in opart if remain[0] in cl), None)
        if host is None or any(slot not in host for slot in remain):
            return False
    return True


def _pattern_cost(key):
    """Number of independent constraints in a closed pattern. At most five."""
    if not key:
        return None
    zeros, part = key
    return len(zeros) + sum(len(group) - 1 for group in part)


def _step_band(t_lo, t_hi, zero_d, eq_d, sig_p, pair):
    """Cosine band of the constraints imposed on this step."""
    vals = []
    for i in range(6):
        if t_lo - 1e-15 <= zero_d[i] <= t_hi + 1e-15:
            vals.append(3.0 * float(sig_p[i]) / pair[i])
    for i, j in itertools.combinations(range(6), 2):
        if t_lo - 1e-15 <= eq_d[i, j] <= t_hi + 1e-15:
            scale = np.sqrt(pair[i] * pair[j])
            vals.append(3.0 * np.sqrt(float(sig_p[i]) ** 2 + float(sig_p[j]) ** 2) / scale)
    return float(min(vals)) if vals else float("inf")


def visible_order(p, tau):
    """How many of the 48 signed label swaps fix the idealised conorm."""
    _zero, ids, _parent = _classes(np.asarray(p, float), 4.0 * tau)
    keep = 0
    for src in PERM_SRC:
        if np.array_equal(ids[src], ids):
            keep += 1
    return 2 * keep


def _edge_name(before, after, old_zero, new_zero, old_ids, new_ids, perm_index):
    if (before, after) not in EDGE_OF or before not in LETTERS:
        return f"{before}>{after}"
    canon = LETTERS[before]
    src = PERM_SRC[perm_index]
    letters = [canon[int(src[n])] for n in range(6)]
    new_zero_letters = {letters[i] for i in range(6) if new_zero[i] and not old_zero[i] and letters[i] != "0"}
    tied = set()
    for a, b in itertools.combinations(range(6), 2):
        if new_zero[a] or new_zero[b] or letters[a] == "0" or letters[b] == "0":
            continue
        if new_ids[a] == new_ids[b] and old_ids[a] != old_ids[b] and letters[a] != letters[b]:
            pair = tuple(sorted((letters[a], letters[b])))
            tied.add(f"{pair[0]} - {pair[1]} = 0")
    satisfied = []
    for cond in EDGE_OF[(before, after)]:
        if cond.endswith("= 0") and " - " not in cond:
            letter = cond.split()[0]
            if letter in new_zero_letters:
                satisfied.append(cond)
        elif cond in tied:
            satisfied.append(cond)
    if len(satisfied) == 1:
        return f"{before}>{after}: {satisfied[0]}"
    if satisfied:
        return f"{before}>{after}: " + " & ".join(satisfied)
    return f"{before}>{after}"


def path_one(item):
    """Monotone sort path and the visible order. ``item`` is a tuple."""
    index, p, lengths, gmax, tau, reported_order, sig_p = item
    p = np.asarray(p, np.float64)
    lengths = np.asarray(lengths, np.float64)
    sig_p = np.asarray(sig_p, np.float64)
    pair = np.maximum(lengths[np.array(SL)[:, 0]] * lengths[np.array(SL)[:, 1]], 1e-30)
    zero_d = np.maximum(p, 0.0) / pair
    diff = np.abs(p[:, None] - p[None, :])
    eq_d = diff / np.sqrt(np.maximum(pair[:, None] * pair[None, :], 1e-30))
    tol = 1e-9 * float(gmax)
    # Values within the appendix tolerance are the same number, so the exact
    # sort already contains those zeros and those ties.
    zero = p <= tol
    parent = list(range(6))

    def find(i, parent=parent):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    alive = [i for i in range(6) if not zero[i]]
    for a, b in itertools.combinations(alive, 2):
        if abs(float(p[a]) - float(p[b])) <= tol:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    sort, perm, current_key = _close_to_sort(zero, parent)
    sort_exact = sort
    exact_perm = perm
    exact_key = current_key
    exact_ids = np.empty(6, dtype=np.int64)
    exact_ids[zero] = -1
    for i in alive:
        exact_ids[i] = find(i)
    raw = sorted(
        {float(x) for x in zero_d if x > tol}
        | {float(eq_d[i, j]) for i in range(6) for j in range(i + 1, 6) if eq_d[i, j] > tol}
    )
    groups = []
    for value in raw:
        if groups and value - groups[-1][1] <= 1e-12:
            groups[-1][1] = value
        else:
            groups.append([value, value])
    steps_t, steps_s, steps_e, steps_b, step_perms, step_keys = [], [], [], [], [], []
    jumps = 0
    path_bug = False
    prev_zero, prev_ids = zero.copy(), exact_ids
    for t_lo, t_hi in groups:
        zero = (p <= tol) | (zero_d <= t_hi)
        parent = list(range(6))

        def find(i, parent=parent):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        alive = [i for i in range(6) if not zero[i]]
        for a, b in itertools.combinations(alive, 2):
            if eq_d[a, b] <= t_hi or abs(p[a] - p[b]) <= tol:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
        name, name_perm, name_key = _close_to_sort(zero, parent)
        if name == "unlisted" or name_key is None:
            break
        ids = np.empty(6, dtype=np.int64)
        ids[zero] = -1
        for i in alive:
            ids[i] = find(i)
        if name == sort:
            prev_zero, prev_ids = zero, ids
            continue
        if current_key is not None and not _pattern_contains(name_key, current_key):
            continue
        if _pattern_cost(name_key) is not None and _pattern_cost(current_key) is not None and _pattern_cost(name_key) <= _pattern_cost(current_key):
            continue
        if len(steps_t) >= 5:
            path_bug = True
            break
        edge = _edge_name(sort, name, prev_zero, zero, prev_ids, ids, perm)
        if (sort, name) not in EDGE_OF and (t_hi - t_lo) > 1e-12:
            jumps += 1
        steps_t.append(t_hi)
        steps_s.append(name)
        steps_e.append(edge)
        steps_b.append(_step_band(t_lo, t_hi, zero_d, eq_d, sig_p, pair))
        step_perms.append(name_perm)
        step_keys.append(name_key)
        sort, perm, current_key = name, name_perm, name_key
        prev_zero, prev_ids = zero, ids
    monotone = True
    prev = exact_key
    prev_cost = _pattern_cost(exact_key)
    for key in step_keys:
        cost = _pattern_cost(key)
        if prev is not None and (not _pattern_contains(key, prev) or (prev_cost is not None and cost is not None and cost <= prev_cost)):
            monotone = False
        if key is not None:
            prev = key
            prev_cost = cost
    path_bug = path_bug or (not monotone) or len(steps_t) > 5
    order = visible_order(p, float(tau))
    zero4 = p <= 4.0 * float(tau)
    forced = np.zeros(6, dtype=bool)
    if reported_order >= order:
        _zz, ids4, _pp = _classes(p, 4.0 * float(tau))
        for src in PERM_SRC:
            if not np.array_equal(ids4[src], ids4):
                continue
            image = np.empty(6, dtype=np.int64)
            for nslot, slot in enumerate(src):
                image[slot] = nslot
            zero4 = zero4 & (p[image] <= 4.0 * float(tau))
        forced = zero4 & (p <= 4.0 * float(tau))
    cos = p / pair
    ambiguous = bool(steps_t) and steps_t[0] <= steps_b[0]
    return {
        "i": index, "sort_exact": sort_exact, "t": steps_t, "sorts": steps_s,
        "edges": steps_e, "bands": steps_b, "perms": step_perms, "exact_perm": exact_perm,
        "overflow": path_bug, "jumps": jumps,
        "visible": order, "forced": forced, "cos": cos, "ambiguous": ambiguous,
        "unlisted": sort_exact == "unlisted" or "unlisted" in steps_s,
    }


def _niggli_setting_differs(red, p, tau, gmax):
    """True when the idealised cell's Niggli basis is a different setting.

    Both matrices are expressed in the reduced cell's frame. They describe the
    same setting when a signed axis permutation carries one onto the other.
    """
    ideal = np.array(p, float).copy()
    ideal[ideal <= 4.0 * float(tau)] = 0.0
    gram = z44._gram(ideal)
    if np.linalg.eigvalsh(gram).min() <= 1e-9 * max(float(gmax), 0.0):
        return False
    try:
        cell = params_from_metric(gram)
        _left, measured = niggli_gk(tuple(float(x) for x in red))
        _right, idealised = niggli_gk(tuple(float(x) for x in cell))
    except (ValueError, RuntimeError, ZeroDivisionError):
        return False
    measured = np.asarray(measured, float)
    idealised = np.asarray(idealised, float)
    for perm in SP:
        if np.allclose(measured, idealised @ perm, atol=1e-5, rtol=0.0):
            return False
    return True


def closure_one(item):
    index, p, tau, red, gmax, sigma = item
    p = np.asarray(p, float)
    tau = float(tau)
    sigma = float(sigma)
    out = {
        "i": index, "ok": True, "error": "", "singular": False,
        "n_closure": None, "n_walk_tau": None, "n_walk_2tau": None,
        "niggli_diff": False, "mono_repeat": False,
        "slot_n": None, "slot_status": "refused",
        "deg3_n": None, "deg3_status": "refused",
    }
    gram = z44._gram(p)

    def _alarm(signum, frame):
        raise TimeoutError("closure search exceeded 5s")

    def _limited(fn):
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(5)
        try:
            return fn()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    try:
        closed, walked, walked2 = _limited(lambda: (
            members_by("closure", p, tau),
            members_by("walk-tau", p, tau),
            members_by("walk-2tau", p, tau),
        ))
    except TimeoutError as exc:
        out["ok"] = False
        out["singular"] = True
        out["error"] = str(exc)
        closed = None
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        closed = None
    if out["error"] and closed is None and out["singular"]:
        pass
    elif closed is None and not out["error"]:
        out["singular"] = True
    elif closed is not None:
        out["n_closure"] = len(closed)
        out["n_walk_tau"] = len(walked) if walked is not None else None
        out["n_walk_2tau"] = len(walked2) if walked2 is not None else None
        if len(closed) > 1:
            orders = []
            for superbase in closed:
                try:
                    orders.append(visible_order(z44._cn(superbase, gram), tau))
                except Exception:
                    orders.append(-1)
            out["mono_repeat"] = any(v >= 2 for v in Counter(orders).values())
    try:
        out["niggli_diff"] = _niggli_setting_differs(red, p, tau, gmax)
    except Exception as exc:
        out["error"] = f"niggli {type(exc).__name__}: {exc}"
    try:
        slot_set, _z, status, _rounds = _limited(lambda: consistent_construct(gram, sigma))
        if slot_set is None:
            out["slot_status"] = "refused"
        else:
            out["slot_n"] = len(slot_set)
            out["slot_status"] = status
        deg_set, _z3, status3, _r3 = _limited(lambda: consistent_construct(gram, SIGMA_3))
        if deg_set is None:
            out["deg3_status"] = "refused"
        else:
            out["deg3_n"] = len(deg_set)
            out["deg3_status"] = status3
    except TimeoutError as exc:
        out["error"] = str(exc)
        out["slot_status"] = out["slot_status"] or "refused"
    except Exception as exc:
        out["error"] = f"perslot {type(exc).__name__}: {exc}"
    return out


def brute_one(item):
    """Compare the consistent set with the [-K, K] widened set."""
    index, p, sigma, status, do4 = item
    gram = z44._gram(np.asarray(p, float))
    out = {"i": index, "timeout": False, "verdict": "refused", "maxent": None, "added4": None, "error": ""}

    def _alarm(signum, frame):
        raise TimeoutError("brute search exceeded 30s")

    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(30)
    try:
        found, maxent = delta_obtuse(gram, float(sigma), 3)
        out["maxent"] = int(maxent)
        if status == "hypothesis":
            slot_set, _z, _status, _rounds = consistent_construct(gram, float(sigma))
            if slot_set is None:
                out["verdict"] = "refused"
            else:
                out["verdict"] = "exact" if slot_set == found else "wrong"
        if do4:
            wider, _more = delta_obtuse(gram, float(sigma), 4)
            out["added4"] = len(wider - found) > 0
    except TimeoutError:
        out["timeout"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return out
    index, p, tau, red, gmax = item
    p = np.asarray(p, float)
    tau = float(tau)
    out = {
        "i": index, "ok": True, "error": "", "singular": False,
        "n_closure": None, "n_walk_tau": None, "n_walk_2tau": None,
        "niggli_diff": False, "mono_repeat": False,
    }
    try:
        def _run():
            return (
                members_by("closure", p, tau),
                members_by("walk-tau", p, tau),
                members_by("walk-2tau", p, tau),
            )
        def _alarm(signum, frame):
            raise TimeoutError("closure search exceeded 5s")
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(5)
        try:
            closed, walked, walked2 = _run()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    except TimeoutError as exc:
        out["ok"] = False
        out["singular"] = True
        out["error"] = str(exc)
        return out
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if closed is None:
        out["singular"] = True
        return out
    out["n_closure"] = len(closed)
    out["n_walk_tau"] = len(walked) if walked is not None else None
    out["n_walk_2tau"] = len(walked2) if walked2 is not None else None
    try:
        out["niggli_diff"] = _niggli_setting_differs(red, p, tau, gmax)
    except Exception as exc:
        out["error"] = f"niggli {type(exc).__name__}: {exc}"
    if len(closed) > 1:
        gram = z44._gram(p)
        orders = []
        for superbase in closed:
            try:
                orders.append(visible_order(z44._cn(superbase, gram), tau))
            except Exception:
                orders.append(-1)
        out["mono_repeat"] = any(v >= 2 for v in Counter(orders).values())
    return out


def _as_float(value):
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_rows(db, sample, seed):
    present = {row[0] for row in db.sql("DESCRIBE cells")}
    cols = [name for name in SELECT_COLS if name in present]
    missing = [name for name in ("pdb_id", "a", "red_a", "sg_hm") if name not in present]
    if missing:
        raise SystemExit(f"database is missing {missing}")
    chosen = None
    if sample is not None:
        rng = np.random.default_rng(seed)
        by_source = defaultdict(list)
        for pid, source in db.sql("SELECT pdb_id, source FROM cells ORDER BY pdb_id"):
            by_source[source or ""].append(pid)
        chosen = []
        for source in sorted(by_source):
            ids = by_source[source]
            take = min(int(sample), len(ids))
            pick = rng.choice(len(ids), size=take, replace=False)
            chosen.extend(ids[i] for i in np.sort(pick))
        chosen = set(chosen)
    sql = "SELECT " + ", ".join(cols) + " FROM cells WHERE pdb_id > ? ORDER BY pdb_id LIMIT 20000"
    rows = []
    last = ""
    while True:
        batch = db.sql(sql, [last, ])
        if not batch:
            break
        last = batch[-1][0]
        for row in batch:
            if chosen is not None and row[0] not in chosen:
                continue
            rows.append(dict(zip(cols, row)))
        if chosen is not None and len(rows) >= len(chosen):
            break
    return rows


def rows_to_arrays(rows):
    n = len(rows)
    def col(name, default=math.nan):
        return np.array([_as_float(row.get(name, default)) for row in rows], dtype=np.float64)

    red = np.column_stack([col(name) for name in ("red_a", "red_b", "red_c", "red_alpha", "red_beta", "red_gamma")])
    dep = np.column_stack([col(name) for name in ("a", "b", "c", "alpha", "beta", "gamma")])
    sig = np.column_stack([col(name) for name in ("sig_a", "sig_b", "sig_c", "sig_alpha", "sig_beta", "sig_gamma")])
    have = np.isfinite(sig).all(axis=1)
    cob = np.tile(np.eye(3), (n, 1, 1))
    cob_fail = []
    for i, row in enumerate(rows):
        values = [row.get(name) for name in COB_COLUMNS]
        if any(v is None or v == "" for v in values):
            cob_fail.append(row["pdb_id"])
            continue
        try:
            matrix = parse_cob_columns(values)
        except (ValueError, ZeroDivisionError, IndexError):
            cob_fail.append(row["pdb_id"])
            continue
        for r, line in enumerate(matrix.rows):
            for c, entry in enumerate(line):
                cob[i, r, c] = float(entry)
    return red, dep, sig, have, cob, cob_fail


def stratified_indices(systems, sources, limit, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, (system, source) in enumerate(zip(systems, sources)):
        groups[(source, system)].append(i)
    chosen = []
    for key in sorted(groups):
        idx = np.array(groups[key])
        take = min(limit, len(idx))
        pick = rng.choice(len(idx), size=take, replace=False)
        chosen.extend(int(idx[j]) for j in np.sort(pick))
    return chosen


def _pool_map(fn, items, workers):
    if not items:
        return []
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(fn, items, chunksize=32))


def sort_at(level, sort_exact, times, names):
    current = sort_exact
    for t, name in zip(times, names):
        if not name or not math.isfinite(t):
            break
        if t <= level:
            current = name
        else:
            break
    return current


def sort_within_bands(path):
    """Sort after every transition that falls inside its own cosine band."""
    current = path["sort_exact"]
    for t, name, band in zip(path["t"], path["sorts"], path.get("bands", [])):
        if not name or not math.isfinite(t):
            break
        if t <= band:
            current = name
        else:
            break
    return current


def write_table(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def md_table(rows, fields):
    head = "| " + " | ".join(fields) + " |"
    rule = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Wall-incidence census of a cell database.")
    parser.add_argument("db", type=Path)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if not args.db.is_file():
        raise SystemExit(f"no database at {args.db}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    workers = os.cpu_count() or 1
    db = CellDatabase(str(args.db), read_only=True)
    try:
        meta_rows = db.sql("SELECT key, value FROM schema_meta ORDER BY key")
        rows = load_rows(db, args.sample, args.seed)
    finally:
        db.close()
    if not rows:
        raise SystemExit("the database has no cells")
    print(f"loaded {len(rows):,} cells", flush=True)
    systems = [reported_system(row.get("sg_number")) for row in rows]
    red, dep, sig, _have_file_esd, cob, cob_fail = rows_to_arrays(rows)
    prepared = prepare_metrics(red, dep, cob, sig, systems)
    device = prepared["device"]
    print(f"metrics on {device}", flush=True)
    p = prepared["p"]
    lengths = prepared["lengths"]
    sl = np.array(SL)
    pair = lengths[:, sl[:, 0]] * lengths[:, sl[:, 1]]
    cosine_band = prepared["sig_p"] / np.maximum(pair, 1e-30)
    sigma = np.where(prepared["kind"] == "esd", 3.0 * cosine_band.max(axis=1), SIGMA_NOMINAL)
    gmax = np.maximum(prepared["gmax"], 0.0)
    lam1 = prepared["lam1"]
    tau = prepared["tau"]
    kind = prepared["kind"]
    n = len(rows)
    obtuse = (p.min(axis=1) >= -1e-9 * np.maximum(gmax, 1e-30)) & np.isfinite(p).all(axis=1) & np.isfinite(lam1) & (lam1 > 0)
    sources = [row.get("source") or "" for row in rows]
    orders = [SYSTEM_ORDER[system] for system in systems]
    letters = [lattice_letter(row.get("sg_hm") or "") for row in rows]

    path_items = [
        (i, p[i].tolist(), lengths[i].tolist(), float(gmax[i]), float(tau[i]), int(orders[i]), prepared["sig_p"][i].tolist())
        for i in range(n) if obtuse[i]
    ]
    print(f"sort paths for {len(path_items):,} obtuse cells on {workers} workers", flush=True)
    path_out = _pool_map(path_one, path_items, workers)
    paths = {item["i"]: item for item in path_out}

    # Time the closure on a seeded 2000-row slice, then decide the rest.
    obtuse_idx = np.flatnonzero(obtuse)
    rng = np.random.default_rng(args.seed)
    timer_n = min(2000, len(obtuse_idx))
    timer_pick = obtuse_idx[np.sort(rng.choice(len(obtuse_idx), size=timer_n, replace=False))] if timer_n else np.array([], dtype=int)
    closure_idx = list(map(int, obtuse_idx))
    closure_mode = "full"
    extrapolated = 0.0
    rate = 0.0
    if args.sample is None and len(obtuse_idx) > timer_n:
        timed = [
            (int(i), p[i].tolist(), float(tau[i]), red[i].tolist(), float(gmax[i]), float(sigma[i]))
            for i in timer_pick
        ]
        t_cl = time.perf_counter()
        timed_out = _pool_map(closure_one, timed, workers)
        elapsed = time.perf_counter() - t_cl
        rate = timer_n / elapsed if elapsed else 0.0
        extrapolated = (len(obtuse_idx) / rate) if rate else float("inf")
        print(f"closure {rate:.1f} cells/s, full pass {extrapolated / 3600:.2f} h", flush=True)
        if extrapolated > 2 * 3600:
            closure_mode = "stratified"
            closure_idx = stratified_indices(systems, sources, 20_000, args.seed)
            closure_idx = [i for i in closure_idx if obtuse[i]]
            have = set(map(int, timer_pick))
            closure_idx = list(have) + [i for i in closure_idx if i not in have]
    else:
        t_cl = time.perf_counter()
        # The sample itself is the timing run; closures are computed below.
        elapsed = 0.0
    if closure_mode == "full" and args.sample is not None:
        closure_idx = list(map(int, obtuse_idx))
    id_of = {row["pdb_id"]: i for i, row in enumerate(rows)}
    for pid in spot_pids(rows):
        i = id_of[pid]
        if obtuse[i] and i not in set(closure_idx):
            closure_idx.append(i)
    already = set(map(int, timer_pick)) if closure_mode == "stratified" or (args.sample is None and timer_n and len(obtuse_idx) > timer_n) else set()
    if args.sample is not None or len(obtuse_idx) <= timer_n:
        already = set()
        todo = [
            (int(i), p[i].tolist(), float(tau[i]), red[i].tolist(), float(gmax[i]), float(sigma[i]))
            for i in closure_idx
        ]
        t_cl = time.perf_counter()
        closed_out = _pool_map(closure_one, todo, workers) if todo else []
        if todo:
            rate = len(todo) / max(time.perf_counter() - t_cl, 1e-9)
            extrapolated = len(obtuse_idx) / rate
        timed_out = []
    else:
        todo = [
            (int(i), p[i].tolist(), float(tau[i]), red[i].tolist(), float(gmax[i]), float(sigma[i]))
            for i in closure_idx if i not in already
        ]
        closed_out = _pool_map(closure_one, todo, workers) if todo else []
    closures = {item["i"]: item for item in list(timed_out) + list(closed_out)}
    brute, brute_rate = run_brute(p, sigma, closures, workers, args.seed)

    summary, tables, flagged, conorm = assemble(
        rows, p, lengths, lam1, tau, kind, prepared["sig_p"], gmax, obtuse,
        systems, sources, letters, paths, closures, cob_fail,
        closure_mode, closure_idx, rate, extrapolated, device, workers, args, started, meta_rows,
        red, sigma, brute, brute_rate,
    )
    summary["spot_text"] = spot_text(args.db, rows, p, lam1, tau, kind, paths, closures, systems)
    write_outputs(out, summary, tables, flagged, conorm, rows, meta_rows)
    print(f"wrote {out} in {time.perf_counter() - started:.1f}s", flush=True)


ANGLE_TOLS = (0.03, 0.1, 0.3, 1.0, 3.0)
LENGTH_TOLS = (0.5, 1.0, 2.0, 5.0)
K_SIGMAS = (1, 2, 3, 5, 10)
RULES = (
    ("0.1deg/0.5pct", 0.1, 0.5),
    ("0.3deg/1pct", 0.3, 1.0),
    ("1deg/2pct", 1.0, 2.0),
    ("3deg/5pct", 3.0, 5.0),
)
TOP_COS = math.sin(math.radians(3.0))


def ratio_bin(edges):
    hi = float(max(edges))
    lo = float(min(edges))
    if lo <= 0:
        return ">10"
    ratio = hi / lo
    if ratio < 1.5:
        return "<1.5"
    if ratio < 2.4:
        return "1.5-2.4"
    if ratio < 3.0:
        return "2.4-3"
    if ratio < 5.0:
        return "3-5"
    if ratio < 10.0:
        return "5-10"
    return ">10"


def run_brute(p, sigma, closures, workers, seed):
    """Widened enumeration of every refused cell and 2000 hypothesis cells."""
    refused = [i for i, rec in closures.items() if rec.get("slot_status") == "refused"]
    hypothesis = [i for i, rec in closures.items() if rec.get("slot_status") == "hypothesis"]
    rng = np.random.default_rng(seed)
    if len(hypothesis) > 2000:
        hypothesis = list(map(int, rng.choice(np.asarray(hypothesis), size=2000, replace=False)))
    jobs = list(refused) + list(hypothesis)
    print(
        f"brute force on {len(refused):,} refused and {len(hypothesis):,} hypothesis cells",
        flush=True,
    )
    if not jobs:
        return [], 0.0
    k4 = set(jobs[:200])

    def item(i):
        return (int(i), p[i].tolist(), float(sigma[i]), closures[i].get("slot_status"), i in k4)

    out = []
    started_at = time.perf_counter()
    ctx = get_context("spawn")
    items = [item(i) for i in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for done, rec in enumerate(pool.map(brute_one, items, chunksize=4), 1):
            out.append(rec)
            if done == 8 or done == len(items) or done % 100 == 0:
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                print(f"brute {done:,}/{len(items):,} at {done / elapsed:.2f} cells/s", flush=True)
    return out, len(out) / max(time.perf_counter() - started_at, 1e-9)


def _letters_of(sort_name, perm):
    canon = LETTERS[sort_name]
    src = PERM_SRC[int(perm)]
    return [canon[int(src[n])] for n in range(6)]


def _measure_condition(cond, letters, pp, pair, sig, esd):
    text = cond.strip()
    if " - " in text:
        left, rest = text.split(" - ", 1)
        right = rest.replace("= 0", "").strip()
        left = left.strip()
        slots_r = [n for n, letter in enumerate(letters) if letter == left]
        slots_s = [n for n, letter in enumerate(letters) if letter == right]
        if not slots_r or not slots_s:
            return None
        pr = float(np.mean(pp[slots_r]))
        ps = float(np.mean(pp[slots_s]))
        d_rel = abs(pr - ps) / max(pr, ps, 1e-30)
        d_cos = 0.0
        for i in slots_r:
            for j in slots_s:
                d_cos = max(
                    d_cos,
                    abs(float(pp[i]) - float(pp[j])) / math.sqrt(max(float(pair[i] * pair[j]), 1e-30)),
                )
        d_sig = None
        if esd:
            sr = max(float(sig[n]) for n in slots_r)
            ss = max(float(sig[n]) for n in slots_s)
            scale = math.sqrt(sr * sr + ss * ss)
            d_sig = abs(pr - ps) / scale if scale > 0 else None
        return {"type": "tie", "d_cos": d_cos, "d_rel": d_rel, "d_sig": d_sig}
    letter = text.split()[0]
    slots = [n for n, name in enumerate(letters) if name == letter]
    if not slots:
        return None
    d_cos = max(float(pp[n]) / max(float(pair[n]), 1e-30) for n in slots)
    d_sig = None
    if esd:
        vals = [float(pp[n]) / float(sig[n]) for n in slots if float(sig[n]) > 0]
        d_sig = max(vals) if vals else None
    return {"type": "zero", "d_cos": d_cos, "d_rel": None, "d_sig": d_sig}


def cell_edge_measures(path, pp, pair, sig, esd):
    visits = [(path["sort_exact"], path.get("exact_perm", 0))]
    visits.extend(zip(path.get("sorts", []), path.get("perms", [])))
    seen = set()
    found = []
    for sort_name, perm in visits:
        if sort_name in seen or sort_name not in LETTERS:
            continue
        seen.add(sort_name)
        letters = _letters_of(sort_name, perm)
        for (src, dst), conds in EDGE_OF.items():
            if src != sort_name:
                continue
            for cond in conds:
                measured = _measure_condition(cond, letters, pp, pair, sig, esd)
                if measured is None:
                    continue
                found.append((f"{src}>{dst}: {cond}", measured))
    return found


def calibration_table(rows, p, lengths, sig_p, kind, systems, sources, paths):
    """Tolerance fractions while each path sits in the source sort of an edge."""
    sl = np.array(SL)
    through = Counter()
    through_esd = Counter()
    within = Counter()
    group_n = Counter()
    group_esd = Counter()
    any_within = Counter()
    for i, path in paths.items():
        esd = kind[i] == "esd"
        groups = ((sources[i], systems[i]), (sources[i], "all"))
        for group in groups:
            group_n[group] += 1
            if esd:
                group_esd[group] += 1
        pair = np.maximum(lengths[i, sl[:, 0]] * lengths[i, sl[:, 1]], 1e-30)
        measures = cell_edge_measures(path, p[i], pair, sig_p[i], esd)
        zero_pass = {angle: False for angle in ANGLE_TOLS}
        tie_pass = {pct: False for pct in LENGTH_TOLS}
        sig_pass = {k: False for k in K_SIGMAS}
        seen = set()
        for edge, measured in measures:
            if edge in seen:
                continue
            seen.add(edge)
            etype = measured["type"]
            for group in groups:
                through[(group, edge, etype)] += 1
                if esd:
                    through_esd[(group, edge, etype)] += 1
                if etype == "zero":
                    for angle in ANGLE_TOLS:
                        if measured["d_cos"] <= math.sin(math.radians(angle)):
                            within[(group, edge, etype, "angle_deg", angle)] += 1
                            zero_pass[angle] = True
                else:
                    for pct in LENGTH_TOLS:
                        if measured["d_rel"] is not None and measured["d_rel"] <= 2.0 * pct / 100.0:
                            within[(group, edge, etype, "length_pct", pct)] += 1
                            tie_pass[pct] = True
                if esd and measured["d_sig"] is not None:
                    for k in K_SIGMAS:
                        if measured["d_sig"] <= k:
                            within[(group, edge, etype, "k_sigma", k)] += 1
                            sig_pass[k] = True
        for group in groups:
            for angle, ok in zero_pass.items():
                if ok:
                    any_within[(group, "any_zero", "zero", "angle_deg", angle)] += 1
            for pct, ok in tie_pass.items():
                if ok:
                    any_within[(group, "any_tie", "tie", "length_pct", pct)] += 1
            if esd:
                for k, ok in sig_pass.items():
                    if ok:
                        any_within[(group, "any", "any", "k_sigma", k)] += 1
            for name, angle, pct in RULES:
                if zero_pass[angle] and tie_pass[pct]:
                    any_within[(group, "any", "any", "rule", name)] += 1
    out = []
    for (group, edge, etype), n_through in sorted(through.items()):
        source, system = group
        specs = []
        if etype == "zero":
            specs.extend(("angle_deg", angle, n_through) for angle in ANGLE_TOLS)
        else:
            specs.extend(("length_pct", pct, n_through) for pct in LENGTH_TOLS)
        if through_esd[(group, edge, etype)]:
            specs.extend(("k_sigma", k, through_esd[(group, edge, etype)]) for k in K_SIGMAS)
        for tol_kind, tol_value, denom in specs:
            n_within = within[(group, edge, etype, tol_kind, tol_value)]
            out.append({
                "source": source, "reported_system": system, "edge": edge, "edge_type": etype,
                "tol_kind": tol_kind, "tol_value": tol_value,
                "n_through": denom, "n_within": n_within,
                "frac": (n_within / denom) if denom else 0.0,
            })
    for group, n_cells in sorted(group_n.items()):
        source, system = group
        for edge, etype, tol_kind, values, denom in (
            ("any_zero", "zero", "angle_deg", ANGLE_TOLS, n_cells),
            ("any_tie", "tie", "length_pct", LENGTH_TOLS, n_cells),
        ):
            for value in values:
                n_within = any_within[(group, edge, etype, tol_kind, value)]
                out.append({
                    "source": source, "reported_system": system, "edge": edge, "edge_type": etype,
                    "tol_kind": tol_kind, "tol_value": value,
                    "n_through": denom, "n_within": n_within,
                    "frac": (n_within / denom) if denom else 0.0,
                })
        n_esd = group_esd[group]
        if n_esd:
            for k in K_SIGMAS:
                n_within = any_within[(group, "any", "any", "k_sigma", k)]
                out.append({
                    "source": source, "reported_system": system, "edge": "any", "edge_type": "any",
                    "tol_kind": "k_sigma", "tol_value": k,
                    "n_through": n_esd, "n_within": n_within,
                    "frac": n_within / n_esd,
                })
        for name, _angle, _pct in RULES:
            n_within = any_within[(group, "any", "any", "rule", name)]
            out.append({
                "source": source, "reported_system": system, "edge": "any", "edge_type": "any",
                "tol_kind": "rule", "tol_value": name,
                "n_through": n_cells, "n_within": n_within,
                "frac": (n_within / n_cells) if n_cells else 0.0,
            })
    return out


def gap_of(path, t0):
    steps = [float(t) for t in path.get("t") or []]
    sorts = list(path.get("sorts") or [])
    kept = [(t, name) for t, name in zip(steps, sorts) if t0 < t < TOP_COS]
    marks = [float(t0)] + [t for t, _name in kept] + [TOP_COS]
    labels = [path["sort_exact"]] + [name for _t, name in kept]
    best_w, best_k = -1.0, 1
    for k in range(1, len(marks)):
        lo, hi = marks[k - 1], marks[k]
        width = math.log10(hi) - math.log10(lo) if lo > 0 and hi > lo else 0.0
        if width > best_w:
            best_w, best_k = width, k
    lo, hi = marks[best_k - 1], marks[best_k]
    mid = math.sqrt(max(lo, 0.0) * max(hi, 0.0))
    return lo, hi, best_w, mid, labels[best_k - 1]


def filtration_tables(rows, lengths, sig_p, kind, systems, sources, paths):
    sl = np.array(SL)
    grouped = defaultdict(list)
    by_gap = defaultdict(list)
    per_cell = {}
    for i, path in paths.items():
        if kind[i] == "esd":
            pair = np.maximum(lengths[i, sl[:, 0]] * lengths[i, sl[:, 1]], 1e-30)
            t0 = max(float(np.max(sig_p[i] / pair)), 1e-12)
        else:
            t0 = 1e-6
        lo, hi, width, mid, gap_sort = gap_of(path, t0)
        per_cell[i] = (lo, hi, width, mid, gap_sort)
        grouped[(sources[i], path["sort_exact"], str(kind[i]))].append((width, mid, gap_sort))
        by_gap[(sources[i], gap_sort)].append((width, mid))
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)

    def qvals(values):
        if not values:
            return [0.0] * len(quantiles)
        return [float(x) for x in np.quantile(np.asarray(values, float), quantiles)]

    gap_rows = []
    for (source, sort_name, tau_kind), recs in sorted(grouped.items()):
        widths = [item[0] for item in recs]
        mids = [item[1] for item in recs]
        wq = qvals(widths)
        mid_deg = [math.degrees(math.asin(min(max(x, 0.0), 1.0))) for x in qvals(mids)]
        sorts = Counter(item[2] for item in recs)
        top_sort, top_n = sorts.most_common(1)[0]
        gap_rows.append({
            "source": source, "sort_exact": sort_name, "tau_kind": tau_kind, "n": len(recs),
            "width_q10": wq[0], "width_q25": wq[1], "width_q50": wq[2],
            "width_q75": wq[3], "width_q90": wq[4],
            "frac_width_ge_1": sum(w >= 1 for w in widths) / len(recs),
            "frac_width_ge_2": sum(w >= 2 for w in widths) / len(recs),
            "mid_deg_q10": mid_deg[0], "mid_deg_q50": mid_deg[2], "mid_deg_q90": mid_deg[4],
            "top_gap_sort": top_sort, "frac_top_gap_sort": top_n / len(recs),
        })
    sort_rows = []
    for (source, gap_sort), recs in sorted(by_gap.items()):
        widths = [item[0] for item in recs]
        mids = [item[1] for item in recs]
        sort_rows.append({
            "source": source, "gap_sort": gap_sort, "n": len(recs),
            "mid_deg_q50": math.degrees(math.asin(min(max(qvals(mids)[2], 0.0), 1.0))),
            "width_q50": qvals(widths)[2],
        })
    return gap_rows, sort_rows, per_cell


def perslot_table(closures, systems, sources, kind):
    rows_out = []
    by_source = sorted(set(sources[i] for i in closures))
    for source in by_source:
        for system in SYSTEMS:
            for kind_name in ("esd", "nominal"):
                group = [
                    i for i in closures
                    if sources[i] == source and systems[i] == system and kind[i] == kind_name
                ]
                if not group:
                    continue
                buckets = Counter()
                ok = hypothesis = refused = misses = 0
                for i in group:
                    rec = closures[i]
                    status = rec.get("slot_status") or "refused"
                    if status == "ok":
                        ok += 1
                    elif status == "hypothesis":
                        hypothesis += 1
                    else:
                        refused += 1
                    nslot = rec.get("slot_n")
                    if nslot is not None:
                        buckets[nslot if nslot in CLOSURE_BUCKETS else "other"] += 1
                    walked = rec.get("n_walk_tau")
                    if nslot is not None and walked is not None and walked < nslot:
                        misses += 1
                rows_out.append({
                    "source": source, "system": system, "tau_kind": kind_name, "n": len(group),
                    "ok": ok, "hypothesis": hypothesis, "refused": refused,
                    "walk_smaller": misses, "frac_walk_smaller": misses / len(group),
                    **{f"n_{k}": buckets.get(k, 0) for k in (1, 2, 3, 6, 16, "other")},
                })
    return rows_out


def three_degree_table(rows, red, closures, systems, sources):
    out = []
    grouped = defaultdict(list)
    for i, rec in closures.items():
        grouped[(sources[i], systems[i], ratio_bin(red[i, :3]))].append(rec)
    for (source, system, ratio), recs in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], RATIO_BINS.index(kv[0][2]) if kv[0][2] in RATIO_BINS else 99)):
        buckets = Counter()
        ok = hypothesis = refused = 0
        for rec in recs:
            status = rec.get("deg3_status") or "refused"
            if status == "ok":
                ok += 1
            elif status == "hypothesis":
                hypothesis += 1
            else:
                refused += 1
            nslot = rec.get("deg3_n")
            if nslot is not None:
                buckets[nslot if nslot in CLOSURE_BUCKETS else "other"] += 1
        out.append({
            "source": source, "system": system, "axis_ratio": ratio, "n": len(recs),
            "ok": ok, "hypothesis": hypothesis, "refused": refused,
            **{f"n_{k}": buckets.get(k, 0) for k in (1, 2, 3, 6, 16, "other")},
        })
    return out


def brute_table(rows, brute):
    exact = wrong = refused = timeout = 0
    maxent = Counter()
    added = checked4 = 0
    wrong_ids = []
    wrong_cells = []
    for rec in brute:
        if rec.get("timeout"):
            timeout += 1
            continue
        verdict = rec.get("verdict") or "refused"
        if verdict == "exact":
            exact += 1
        elif verdict == "wrong":
            wrong += 1
            if len(wrong_ids) < 5:
                i = rec["i"]
                wrong_ids.append(rows[i]["pdb_id"])
                wrong_cells.append([float(rows[i][name]) for name in ("a", "b", "c", "alpha", "beta", "gamma")])
        else:
            refused += 1
        if rec.get("maxent") in (1, 2, 3):
            maxent[rec["maxent"]] += 1
        if rec.get("added4") is not None:
            checked4 += 1
            if rec["added4"]:
                added += 1
    summary = [{
        "n": len(brute), "exact": exact, "wrong": wrong, "refused": refused, "timeout": timeout,
        "maxent_1": maxent.get(1, 0), "maxent_2": maxent.get(2, 0), "maxent_3": maxent.get(3, 0),
        "k4_checked": checked4, "k4_added": added,
    }]
    return summary, wrong, wrong_ids, wrong_cells


def assemble(rows, p, lengths, lam1, tau, kind, sig_p, gmax, obtuse, systems, sources,
             letters, paths, closures, cob_fail, closure_mode, closure_idx,
             rate, extrapolated, device, workers, args, started, meta_rows,
             red, sigma, brute, brute_rate):
    n = len(rows)
    closure_set = set(closures)
    issues = []
    if cob_fail:
        issues.append({"name": "cob-unparsed", "n": len(cob_fail), "ids": cob_fail[:5],
                       "detail": "The stored change of basis did not parse. Those rows use a nominal band on the reduced cell."})
    obtuse_ids = [rows[i]["pdb_id"] for i in range(n) if not obtuse[i]]
    if obtuse_ids:
        issues.append({"name": "reduced-cell-not-obtuse", "n": len(obtuse_ids), "ids": obtuse_ids[:5],
                       "detail": "A stored reduced cell has a conorm below -1e-9 max|G|. Counted in the inventory and left out of the wall and closure tables."})

    inv_rows = []
    by_source = Counter(sources)
    by_sys = Counter(zip(sources, systems))
    for source, count in sorted(by_source.items()):
        mask = [i for i in range(n) if sources[i] == source]
        inv_rows.append({
            "source": source, "n": count,
            "with_esd": int(sum(kind[i] == "esd" for i in mask)),
            "nominal": int(sum(kind[i] == "nominal" for i in mask)),
            "null_sg": int(sum(systems[i] == "unknown" for i in mask)),
            "non_P": int(sum(letters[i] not in ("", "P") for i in mask)),
            "not_obtuse": int(sum(not obtuse[i] for i in mask)),
        })
    sys_rows = [{"source": s, "system": m, "n": c} for (s, m), c in sorted(by_sys.items())]

    wall_rows = []
    zero_rows = []
    labels = bin_labels("r=0")
    triclinic_three = []
    triclinic_three_n = 0
    for source in sorted(by_source):
        idx = [i for i in range(n) if sources[i] == source and obtuse[i]]
        hist = Counter()
        under = {key: 0 for key in (1e-2, 1e-3, 1e-4, 1e-5)}
        for i in idx:
            r = float(p[i].min() / lam1[i]) if lam1[i] > 0 else 0.0
            if p[i].min() <= 1e-9 * max(gmax[i], 0.0):
                hist["r=0"] += 1
            else:
                hist[log_bin(r)] += 1
            for key in under:
                if r < key:
                    under[key] += 1
        row = {"source": source, "n": len(idx)}
        for label in labels:
            row[label] = hist.get(label, 0)
        for key, count in under.items():
            row[f"r<{key:.0e}"] = count
            row[f"frac_r<{key:.0e}"] = (count / len(idx)) if idx else 0.0
        wall_rows.append(row)
        for system in SYSTEMS:
            cols = Counter()
            group = [i for i in idx if systems[i] == system]
            for i in group:
                nz = int(np.sum(p[i] <= 1e-3 * lam1[i]))
                cols[nz if nz <= 3 else "more"] += 1
                if system == "triclinic" and nz >= 3:
                    triclinic_three_n += 1
                    if len(triclinic_three) < 5:
                        triclinic_three.append(rows[i]["pdb_id"])
            if not group:
                continue
            zero_rows.append({
                "source": source, "system": system, "n": len(group),
                "z0": cols.get(0, 0), "z1": cols.get(1, 0), "z2": cols.get(2, 0),
                "z3": cols.get(3, 0), "zmore": cols.get("more", 0),
            })
    if triclinic_three_n:
        issues.append({"name": "triclinic-with-three-near-zeros", "n": triclinic_three_n,
                       "ids": triclinic_three,
                       "detail": "Triclinic-reported cells with at least three conorms below 1e-3 lam1."})

    def closure_tables(kind_name):
        dist_rows, flag_rows = [], []
        for source in sorted(by_source):
            for system in SYSTEMS:
                all_group = [i for i in closure_set if sources[i] == source and systems[i] == system and kind[i] == kind_name]
                if not all_group:
                    continue
                buckets = Counter()
                misses = misses2 = singular = beyond = 0
                for i in all_group:
                    rec = closures[i]
                    if lam1[i] <= 32.0 * tau[i]:
                        beyond += 1
                    if rec.get("singular") or rec.get("n_closure") is None:
                        singular += 1
                        continue
                    nc = rec["n_closure"]
                    buckets[nc if nc in CLOSURE_BUCKETS else "other"] += 1
                    if rec["n_walk_tau"] is not None and rec["n_walk_tau"] < nc:
                        misses += 1
                    if rec["n_walk_2tau"] is not None and rec["n_walk_2tau"] < nc:
                        misses2 += 1
                if not all_group:
                    continue
                dist_rows.append({
                    "source": source, "system": system, "tau_kind": kind_name, "n": len(all_group),
                    **{f"n_{k}": buckets.get(k, 0) for k in (1, 2, 3, 6, 16, "other")},
                })
                flag_rows.append({
                    "source": source, "system": system, "tau_kind": kind_name, "n": len(all_group),
                    "walk_misses": misses, "walk2_misses": misses2,
                    "step2_singular": singular, "beyond_theorem": beyond,
                    "frac_walk_misses": misses / len(all_group),
                    "frac_walk2_misses": misses2 / len(all_group),
                    "frac_singular": singular / len(all_group),
                    "frac_beyond": beyond / len(all_group),
                })
        return dist_rows, flag_rows

    dist_esd, flag_esd = closure_tables("esd")
    dist_nom, flag_nom = closure_tables("nominal")

    cross = Counter()
    off = Counter()
    above = Counter()
    below = Counter()
    for i, rec in closures.items():
        if rec.get("singular") or rec.get("n_closure") is None:
            continue
        order = paths.get(i, {}).get("visible", visible_order(p[i], tau[i]))
        visible = VISIBLE_SYSTEM.get(order, "other")
        cross[(sources[i], systems[i], visible)] += 1
        if systems[i] != visible:
            off[(systems[i], visible)] += 1
        if order > SYSTEM_ORDER[systems[i]]:
            above[sources[i]] += 1
        if SYSTEM_ORDER[systems[i]] > order and rec["n_closure"] == 1:
            below[sources[i]] += 1
    cross_rows = [{"source": s, "reported": a, "visible": b, "n": c} for (s, a, b), c in sorted(cross.items())]
    off_rows = [{"reported": a, "visible": b, "n": c} for (a, b), c in off.most_common(10)]
    sym_rows = []
    for source in sorted(by_source):
        denom = sum(1 for i in closures if sources[i] == source and not closures[i].get("singular") and closures[i].get("n_closure") is not None)
        sym_rows.append({
            "source": source, "n": denom,
            "metric_above_reported": above.get(source, 0),
            "reported_above_visible": below.get(source, 0),
        })

    mono_centred = mono_prim = ortho_multi = 0
    mono_c_rows = []
    walk_miss_ids = []
    walk_miss_n = 0
    walk_miss_in_traps = 0
    esd_miss_n = 0
    esd_miss_ids = []
    singular_n = 0
    singular_ids = []
    beyond_n = 0
    beyond_ids = []
    for i, rec in closures.items():
        if lam1[i] <= 32.0 * tau[i]:
            beyond_n += 1
            if len(beyond_ids) < 5:
                beyond_ids.append(rows[i]["pdb_id"])
        if rec.get("singular"):
            singular_n += 1
            if len(singular_ids) < 5:
                singular_ids.append(rows[i]["pdb_id"])
            continue
        nc = rec.get("n_closure")
        if nc is None:
            continue
        miss = rec["n_walk_tau"] is not None and rec["n_walk_tau"] < nc
        centred = letters[i] in {"A", "B", "C", "I"}
        mono_trap = systems[i] == "monoclinic" and rec.get("mono_repeat")
        ortho_trap = systems[i] == "orthorhombic" and nc > 1
        if systems[i] == "monoclinic" and rec.get("mono_repeat"):
            if centred:
                mono_centred += 1
            else:
                mono_prim += 1
        if ortho_trap:
            ortho_multi += 1
        if miss:
            walk_miss_n += 1
            if mono_trap or ortho_trap:
                walk_miss_in_traps += 1
            if len(walk_miss_ids) < 5:
                walk_miss_ids.append(rows[i]["pdb_id"])
            if kind[i] == "esd":
                esd_miss_n += 1
                if len(esd_miss_ids) < 5:
                    esd_miss_ids.append(rows[i]["pdb_id"])
    trap_rows = [{
        "monoclinic_repeat_centred": mono_centred,
        "monoclinic_repeat_primitive": mono_prim,
        "orthorhombic_closure_gt1": ortho_multi,
        "walk_misses": walk_miss_n,
        "walk_misses_in_traps": walk_miss_in_traps,
        "frac_walk_misses_in_traps": (walk_miss_in_traps / walk_miss_n) if walk_miss_n else 0.0,
    }]
    if esd_miss_n:
        issues.append({"name": "walk-misses-with-esd", "n": esd_miss_n, "ids": esd_miss_ids,
                       "detail": "Flip walk at tau missed a closure member on a cell whose band came from the reported esds."})
    if beyond_n:
        issues.append({"name": "beyond-theorem", "n": beyond_n, "ids": beyond_ids,
                       "detail": "lam1 is not greater than 32 tau, so the widened-closure theorem's hypothesis fails. The construction was still run."})
    if singular_n:
        issues.append({"name": "step2-singular", "n": singular_n, "ids": singular_ids,
                       "detail": "The idealised lattice at 4 tau was singular, so members_by('closure') returned None, or the flip search exceeded 5 seconds."})

    niggli_n = Counter()
    niggli_diff = Counter()
    for i, rec in closures.items():
        if rec.get("singular") or rec.get("n_closure") is None:
            continue
        key = (sources[i], paths.get(i, {}).get("sort_exact", ""))
        niggli_n[key] += 1
        if rec.get("niggli_diff"):
            niggli_diff[key] += 1
    niggli_rows = [{
        "source": source, "sort": sort_name, "n": count,
        "niggli_setting_differs": niggli_diff[(source, sort_name)],
    } for (source, sort_name), count in sorted(niggli_n.items())]

    # Sort aggregates over every obtuse cell that has a path.
    edge_hist = defaultdict(Counter)
    pop = defaultdict(lambda: Counter())
    at_levels = defaultdict(Counter)
    ambiguous = defaultdict(lambda: [0, 0])
    unlisted_n = jump_n = overflow_n = 0
    unlisted_ids, jump_ids, overflow_ids = [], [], []
    for i, path in paths.items():
        source = sources[i]
        if path["unlisted"]:
            unlisted_n += 1
            if len(unlisted_ids) < 5:
                unlisted_ids.append(rows[i]["pdb_id"])
        if path["jumps"]:
            jump_n += 1
            if len(jump_ids) < 5:
                jump_ids.append(rows[i]["pdb_id"])
        if path["overflow"]:
            overflow_n += 1
            if len(overflow_ids) < 5:
                overflow_ids.append(rows[i]["pdb_id"])
        ambiguous[(source, path["sort_exact"])][1] += 1
        if path.get("ambiguous"):
            ambiguous[(source, path["sort_exact"])][0] += 1
        for t, edge in zip(path["t"], path["edges"]):
            edge_hist[(source, edge)][log_bin(t)] += 1
        names = path["sorts"]
        times = path["t"]
        for level in POP_LEVELS:
            pop[(source, sort_at(level, path["sort_exact"], times, names), level)]["n"] += 1
        at_levels[(source, sort_within_bands(path), "4tau")]["n"] += 1
        for level, label in ((1e-2, "1e-2"), (1e-3, "1e-3"), (1e-4, "1e-4")):
            at_levels[(source, sort_at(level, path["sort_exact"], times, names), label)]["n"] += 1
    if unlisted_n:
        issues.append({"name": "unlisted-sort", "n": unlisted_n, "ids": unlisted_ids,
                       "detail": "A closed (zeros, equality classes) set matched none of the 24 Table 4 patterns."})
    if jump_n:
        issues.append({"name": "non-codimension-one-jump", "n": jump_n, "ids": jump_ids,
                       "detail": "A sort change was not one of the 37 codimension-one edges of Table C2."})
    if overflow_n:
        issues.append({"name": "path-not-monotone-or-longer-than-five", "n": overflow_n, "ids": overflow_ids,
                       "detail": "A monotone path from six free conorms to one has at most five codimension-one steps. These paths were not monotone, or a sixth containing step remained. The stored path stops at five."})

    edge_rows = []
    for (source, edge), hist in sorted(edge_hist.items()):
        row = {"source": source, "edge": edge, "n": sum(hist.values())}
        for label in bin_labels(None):
            row[label] = hist.get(label, 0)
        edge_rows.append(row)
    pop_rows = [{"source": s, "sort": name, "t": f"{level:.3e}", "n": rec["n"]}
                for (s, name, level), rec in sorted(pop.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))]
    level_rows = [{"source": s, "sort": name, "tolerance": label, "n": rec["n"]}
                  for (s, name, label), rec in sorted(at_levels.items())]
    amb_rows = [{"source": s, "sort_exact": name, "n": tot, "ambiguous_esd": num,
                 "frac": (num / tot) if tot else 0.0}
                for (s, name), (num, tot) in sorted(ambiguous.items())]

    flagged = []
    for i, rec in closures.items():
        flags = []
        if rec.get("singular") or rec.get("n_closure") is None and not rec.get("ok", True):
            flags.append("step2-singular")
        if rec.get("singular"):
            flags.append("step2-singular")
        if lam1[i] <= 32.0 * tau[i]:
            flags.append("beyond-theorem")
        nc = rec.get("n_closure")
        if nc is not None and rec.get("n_walk_tau") is not None and rec["n_walk_tau"] < nc:
            flags.append("walk-misses")
        if nc is not None and rec.get("n_walk_2tau") is not None and rec["n_walk_2tau"] < nc:
            flags.append("walk2-misses")
        order = paths.get(i, {}).get("visible", 0)
        if nc is not None and order > SYSTEM_ORDER[systems[i]]:
            flags.append("metric-above-reported")
        if nc == 1 and SYSTEM_ORDER[systems[i]] > order:
            flags.append("reported-above-visible")
        flags = list(dict.fromkeys(flags))
        if not flags:
            continue
        flagged.append({
            "id": rows[i]["pdb_id"], "source": sources[i],
            "sg_number": rows[i].get("sg_number"), "sg_hm": rows[i].get("sg_hm") or "",
            "cell": [float(rows[i][name]) for name in ("a", "b", "c", "alpha", "beta", "gamma")],
            "esd": [None if not math.isfinite(sig_p[i, k]) else float(sig_p[i, k]) for k in range(6)],
            "conorms": [float(x) for x in p[i]], "lam1": float(lam1[i]), "tau": float(tau[i]),
            "n_closure": nc, "n_walk_tau": rec.get("n_walk_tau"), "n_walk_2tau": rec.get("n_walk_2tau"),
            "visible_order": int(order), "reported_system": systems[i], "flags": flags,
            "tau_kind": str(kind[i]),
        })
    flagged = stratify_flags(flagged, 500)

    slot_rows = perslot_table(closures, systems, sources, kind)
    degree_rows = three_degree_table(rows, red, closures, systems, sources)
    cal_rows = calibration_table(rows, p, lengths, sig_p, kind, systems, sources, paths)
    gap_rows, gap_sort_rows, gaps = filtration_tables(rows, lengths, sig_p, kind, systems, sources, paths)
    brute_rows, wrong_n, wrong_ids, wrong_cells = brute_table(rows, brute)
    if wrong_n:
        shown = [f"{pid} {cell}" for pid, cell in zip(wrong_ids, wrong_cells)]
        issues.append({
            "name": "perslot-wrong-closure", "n": wrong_n, "ids": wrong_ids,
            "detail": "The consistent construction and the K=3 enumeration disagree. "
                      + "Examples: " + "; ".join(shown) + ".",
        })

    conorm = pack_conorms(rows, p, lengths, lam1, tau, kind, sig_p, systems, sources, paths, gaps)
    summary = {
        "db": str(args.db), "seed": args.seed, "sample": args.sample, "n": n,
        "closure_mode": closure_mode, "closure_n": len(closures),
        "cells_per_second": rate, "extrapolated_full_seconds": extrapolated,
        "device": device, "workers": workers,
        "inventory": inv_rows, "by_system": sys_rows, "wall": wall_rows, "near_zeros": zero_rows,
        "closure_esd": dist_esd, "closure_nominal": dist_nom,
        "flags_esd": flag_esd, "flags_nominal": flag_nom,
        "symmetry": sym_rows, "off_diagonal": off_rows, "cross": cross_rows,
        "traps": trap_rows, "niggli": niggli_rows,
        "ambiguous": amb_rows, "issues": issues,
        "brute_rate": brute_rate,
        "wall_seconds": time.perf_counter() - started,
    }
    tables = {
        "inventory.csv": (inv_rows, ["source", "n", "with_esd", "nominal", "null_sg", "non_P", "not_obtuse"]),
        "inventory_system.csv": (sys_rows, ["source", "system", "n"]),
        "wall_histogram.csv": (wall_rows, list(wall_rows[0].keys()) if wall_rows else ["source"]),
        "near_zeros.csv": (zero_rows, ["source", "system", "n", "z0", "z1", "z2", "z3", "zmore"]),
        "closure_counts.csv": (dist_esd + dist_nom, ["source", "system", "tau_kind", "n", "n_1", "n_2", "n_3", "n_6", "n_16", "n_other"]),
        "closure_flags.csv": (flag_esd + flag_nom, ["source", "system", "tau_kind", "n", "walk_misses", "walk2_misses", "step2_singular", "beyond_theorem", "frac_walk_misses", "frac_walk2_misses", "frac_singular", "frac_beyond"]),
        "symmetry_flags.csv": (sym_rows, ["source", "n", "metric_above_reported", "reported_above_visible"]),
        "symmetry_cross.csv": (cross_rows, ["source", "reported", "visible", "n"]),
        "symmetry_off_diagonal.csv": (off_rows, ["reported", "visible", "n"]),
        "traps.csv": (trap_rows, list(trap_rows[0].keys())),
        "niggli.csv": (niggli_rows, ["source", "sort", "n", "niggli_setting_differs"]),
        "edge_transitions.csv": (edge_rows, list(edge_rows[0].keys()) if edge_rows else ["source", "edge", "n"]),
        "sort_population.csv": (pop_rows, ["source", "sort", "t", "n"]),
        "sorts_by_source.csv": (level_rows, ["source", "sort", "tolerance", "n"]),
        "ambiguous_esd.csv": (amb_rows, ["source", "sort_exact", "n", "ambiguous_esd", "frac"]),
        "perslot_closure.csv": (slot_rows, ["source", "system", "tau_kind", "n", "ok", "hypothesis", "refused", "walk_smaller", "frac_walk_smaller", "n_1", "n_2", "n_3", "n_6", "n_16", "n_other"]),
        "three_degree.csv": (degree_rows, ["source", "system", "axis_ratio", "n", "ok", "hypothesis", "refused", "n_1", "n_2", "n_3", "n_6", "n_16", "n_other"]),
        "brute_check.csv": (brute_rows, ["n", "exact", "wrong", "refused", "timeout", "maxent_1", "maxent_2", "maxent_3", "k4_checked", "k4_added"]),
        "tolerance_calibration.csv": (cal_rows, ["source", "reported_system", "edge", "edge_type", "tol_kind", "tol_value", "n_through", "n_within", "frac"]),
        "filtration_gaps.csv": (gap_rows, ["source", "sort_exact", "tau_kind", "n", "width_q10", "width_q25", "width_q50", "width_q75", "width_q90", "frac_width_ge_1", "frac_width_ge_2", "mid_deg_q10", "mid_deg_q50", "mid_deg_q90", "top_gap_sort", "frac_top_gap_sort"]),
        "filtration_gap_sorts.csv": (gap_sort_rows, ["source", "gap_sort", "n", "mid_deg_q50", "width_q50"]),
    }
    return summary, tables, flagged, conorm


def rec_n(rec):
    value = rec.get("n_closure")
    return value if value is not None else 0


def stratify_flags(flagged, limit):
    buckets = defaultdict(list)
    for row in flagged:
        for flag in row["flags"]:
            buckets[(row["source"], flag)].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda row: row["id"])
    chosen = []
    seen = set()
    keys = sorted(buckets)
    while len(chosen) < limit and any(buckets[key] for key in keys):
        for key in keys:
            if not buckets[key] or len(chosen) >= limit:
                continue
            row = buckets[key].pop(0)
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            chosen.append(row)
    return chosen


def pack_conorms(rows, p, lengths, lam1, tau, kind, sig_p, systems, sources, paths, gaps):
    n = len(rows)
    pair = np.maximum(
        lengths[:, np.array(SL)[:, 0]] * lengths[:, np.array(SL)[:, 1]], 1e-30)
    cos = p / pair
    sort_codes = {name: i for i, name in enumerate(["", *SORT_NAMES, "unlisted"])}
    t_path = np.full((n, 5), np.nan, np.float32)
    band_path = np.full((n, 5), np.nan, np.float32)
    sort_path = np.full((n, 5), -1, np.int16)
    exact = np.full(n, -1, np.int16)
    forced = np.zeros((n, 6), dtype=bool)
    amb = np.zeros(n, dtype=bool)
    for i, path in paths.items():
        exact[i] = sort_codes.get(path["sort_exact"], sort_codes["unlisted"])
        for k, (t, name) in enumerate(zip(path["t"], path["sorts"])):
            if k == 5:
                break
            t_path[i, k] = t
            sort_path[i, k] = sort_codes.get(name, sort_codes["unlisted"])
            if k < len(path.get("bands", [])):
                band_path[i, k] = path["bands"][k]
        forced[i] = path["forced"]
        amb[i] = bool(path.get("ambiguous"))
    gap_lo = np.full(n, np.nan, np.float32)
    gap_hi = np.full(n, np.nan, np.float32)
    gap_width = np.full(n, np.nan, np.float32)
    gap_mid = np.full(n, np.nan, np.float32)
    gap_sort = np.full(n, -1, np.int16)
    for i, (lo, hi, width, mid, name) in gaps.items():
        gap_lo[i] = lo
        gap_hi[i] = hi
        gap_width[i] = width
        gap_mid[i] = mid
        gap_sort[i] = sort_codes.get(name, sort_codes["unlisted"])
    sig = np.array(sig_p, np.float32)
    sig[kind != "esd"] = np.nan
    return {
        "id": np.array([row["pdb_id"] for row in rows]),
        "source": np.array(sources),
        "sg_number": np.array([0 if row.get("sg_number") is None else int(row["sg_number"]) for row in rows], np.int32),
        "reported_system": np.array(systems),
        "V": np.array([_as_float(row.get("volume")) for row in rows], np.float32),
        "lam1": lam1.astype(np.float32), "tau": tau.astype(np.float32),
        "tau_kind": np.array(kind),
        "p": p.astype(np.float32), "cos": cos.astype(np.float32), "sig": sig,
        "forced": forced, "sort_exact": exact, "t_path": t_path, "sort_path": sort_path,
        "ambiguous_esd": amb, "band_path": band_path,
        "gap_lo": gap_lo, "gap_hi": gap_hi, "gap_width": gap_width,
        "gap_mid": gap_mid, "gap_sort": gap_sort,
        "sort_codes": np.array(list(sort_codes.keys())),
    }


def spot_pids(rows):
    """Ids used for the spot checks, chosen from the loaded rows."""
    def first(pred):
        for row in rows:
            if pred(row):
                return row["pdb_id"]
        return None

    found = []
    queries = [
        lambda r: r["pdb_id"] == "COD9000001",
        lambda r: "O2 Si" in (r.get("formula") or "") and 143 <= int(r.get("sg_number") or 0) <= 167,
        lambda r: r.get("formula") in ("- Na Cl -", "- Cl Na -"),
        lambda r: r.get("formula") in ("- Si -", "- Si8 -", "- Si4 -"),
        lambda r: r.get("formula") == "- Al Na O8 Si3 -",
        lambda r: r.get("formula") == "- Ca Mg O6 Si2 -",
        lambda r: (r.get("source") == "amcsd") and ("O8" in (r.get("formula") or "")),
        lambda r: r["pdb_id"] == "193L",
        lambda r: r["pdb_id"] == "1LYZ",
        lambda r: r["pdb_id"] == "2HHB",
    ]
    for pred in queries:
        pid = first(pred)
        if pid:
            found.append(pid)
    return found


def spot_text(db_path, rows, p, lam1, tau, kind, paths, closures, systems):
    """One line per requested structure, naming the id that was actually used."""
    specs = [
        ("COD9000001", "pdb_id = 'COD9000001'"),
        ("alpha-quartz", "formula LIKE '%O2 Si%' AND sg_number BETWEEN 143 AND 167"),
        ("NaCl", "formula LIKE '%Na%' AND formula LIKE '%Cl%' AND formula NOT LIKE '%O%'"),
        ("Si", "formula IN ('- Si -', '- Si8 -', '- Si4 -')"),
        ("albite", "formula LIKE '%Al Na O8 Si3%'"),
        ("diopside", "formula LIKE '%Ca Mg O6 Si2%'"),
        ("AMCSD feldspar", "source = 'amcsd' AND (formula LIKE '%Si3 O8%' OR formula LIKE '%feldspar%')"),
        ("193L", "pdb_id = '193L'"),
        ("1LYZ", "pdb_id = '1LYZ'"),
        ("2HHB", "pdb_id = '2HHB'"),
    ]
    index = {row["pdb_id"]: i for i, row in enumerate(rows)}
    db = CellDatabase(str(db_path), read_only=True)
    lines = []
    try:
        have_formula = any(row[0] == "formula" for row in db.sql("DESCRIBE cells"))
        for label, where in specs:
            if "formula" in where and not have_formula:
                lines.append(f"- {label}: this database has no formula column.")
                continue
            try:
                found = db.sql(
                    "SELECT pdb_id, source, sg_number, sg_hm, formula, a, b, c, alpha, beta, gamma "
                    f"FROM cells WHERE {where} ORDER BY pdb_id LIMIT 1"
                )
            except Exception as exc:
                lines.append(f"- {label}: lookup failed ({exc}).")
                continue
            if not found:
                lines.append(f"- {label}: not in this database.")
                continue
            pid, source, number, hm, formula, a, b, c, al, be, ga = found[0]
            extra = ""
            i = index.get(pid)
            if i is not None:
                extra = (
                    f" lam1={lam1[i]:.6g} tau={tau[i]:.6g} ({kind[i]}) "
                    f"visible={paths.get(i, {}).get('visible')} sort={paths.get(i, {}).get('sort_exact')}"
                )
                rec = closures.get(i)
                if rec:
                    extra += f" n_closure={rec.get('n_closure')} n_walk_tau={rec.get('n_walk_tau')} n_walk_2tau={rec.get('n_walk_2tau')}"
            else:
                extra = " (not in the closure sample)"
            lines.append(
                f"- {label}: `{pid}` ({source}), formula `{formula}`, sg {number} {hm}, "
                f"cell {a} {b} {c} {al} {be} {ga}.{extra}"
            )
    finally:
        db.close()
    return "\n".join(lines)


def write_outputs(out, summary, tables, flagged, conorm, rows, meta_rows):
    tables_dir = out / "tables"
    for name, (body, fields) in tables.items():
        if body:
            write_table(tables_dir / name, body, fields)
    (out / "census_summary.json").write_text(json.dumps(summary, indent=2, default=_json) + "\n")
    with (out / "flagged.jsonl").open("w") as handle:
        for row in flagged:
            handle.write(json.dumps(row, default=_json) + "\n")
    conorm_path = write_conorms(out, conorm)
    env = collect_env(summary, rows, meta_rows)
    (out / "env.txt").write_text(env)
    (out / "README.md").write_text(render_readme(summary, tables, conorm_path, env))


def write_conorms(out, conorm):
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet as pq
        path = out / "conorms.parquet"
        columns = {
            "id": conorm["id"], "source": conorm["source"], "sg_number": conorm["sg_number"],
            "reported_system": conorm["reported_system"], "V": conorm["V"], "lam1": conorm["lam1"],
            "tau": conorm["tau"], "tau_kind": conorm["tau_kind"],
        }
        slots = ("da", "db", "dc", "ab", "ac", "bc")
        for k, name in enumerate(slots):
            columns[f"p_{name}"] = conorm["p"][:, k]
            columns[f"cos_{name}"] = conorm["cos"][:, k]
            columns[f"sig_{name}"] = conorm["sig"][:, k]
            columns[f"forced_{name}"] = conorm["forced"][:, k]
        columns["sort_exact"] = conorm["sort_exact"]
        for k in range(6):
            columns[f"t_{k+1}"] = conorm["t_path"][:, k]
            columns[f"band_{k+1}"] = conorm["band_path"][:, k]
            columns[f"sort_{k+1}"] = conorm["sort_path"][:, k]
        columns["ambiguous_esd"] = conorm["ambiguous_esd"]
        table = pyarrow.table(columns)
        pq.write_table(table, path)
        return path
    except ImportError:
        path = out / "conorms.npz"
        np.savez_compressed(path, **conorm)
        return path


def _json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def collect_env(summary, rows, meta_rows):
    import duckdb
    import subprocess
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    version = ""
    for line in (REPO / "pyproject.toml").read_text().splitlines():
        if line.startswith("version"):
            version = line.split("=", 1)[1].strip().strip('"').strip("'")
    counts = Counter(row.get("source") or "" for row in rows)
    lines = [
        f"git {head}",
        f"agentsg {version}",
        f"duckdb {duckdb.__version__}",
        f"numpy {np.__version__}",
        f"rows {dict(counts)}",
        f"device {summary['device']}",
        f"workers {summary['workers']}",
        f"wall_seconds {summary['wall_seconds']:.3f}",
        f"closure_mode {summary['closure_mode']}",
        f"cells_per_second {summary['cells_per_second']:.4f}",
        "schema_meta",
    ]
    for key, value in meta_rows:
        lines.append(f"  {key}={value}")
    return "\n".join(lines) + "\n"


def render_readme(summary, tables, conorm_path, env):
    parts = [
        "# Wall-incidence census",
        "",
        "Measured cells from a `CellDatabase`, scored with the Selling-closure helpers in `manuscript/benchmark/evaluate.py` and `manuscript/appendix_z/test_z44_widened_complete.py`. No library code is changed.",
        "",
        f"Database `{summary['db']}`. Seed `{summary['seed']}`. Sample `{summary['sample']}`. "
        f"Loaded **{summary['n']:,}** cells. Closure pass: **{summary['closure_mode']}** on {summary['closure_n']:,} obtuse cells. "
        f"Metric batch on **{summary['device']}**, closures and sort paths on **{summary['workers']}** processes "
        f"({summary['cells_per_second']:.2f} closure cells/s). "
        f"A full closure pass would take {summary['extrapolated_full_seconds'] / 3600:.2f} h.",
        "",
        "The reduced superbase is built from the stored Selling cell. `tau` is three times the largest conorm standard deviation, and never smaller than `1e-9 max|G|`. "
        "A parameter fixed by the reported crystal system (a 90° or 120° angle, or a length the system equates to another) has σ = 0. "
        "`tau_kind = esd` when every free parameter has an esd and at least one of those esds is positive; otherwise the nominal errors are `0.001 * length` and `0.03` degrees. "
        "A sort path imposes cosine-scaled zeros and ties in order. Equal cosine crossings are one step. "
        "A step is kept only when the next closed pattern contains the previous one and adds a constraint, so a path from six free conorms to one has at most five steps. "
        "Each constraint set is replaced by the finest closed Table 4 pattern that contains it. "
        "A transition is inside the measurement when its cosine is within `3 σ(p)` of that slot, scaled by the edge lengths. "
        "The per-slot closure is `consistent_construct`: esd rows use `sigma = 3 max σ(p_ij)/(|u_i||u_j|)`, nominal rows use `sin(0.03°)`. "
        "`ok` is Conjecture A6''. The 3° table uses `sigma = sin(3°)/4` on every cell. "
        "`d_sig` uses the largest slot sigma in a class and ignores covariances.",
        "",
        "## Inventory",
        "",
        md_table(*tables["inventory.csv"]) if tables["inventory.csv"][0] else "",
        "",
        md_table(*tables["inventory_system.csv"]) if tables["inventory_system.csv"][0] else "",
        "",
        "## Wall proximity",
        "",
        "Histogram of `log10(min p / lam1)` in half-decade bins. `r=0` is a conorm at or below `1e-9 max|G|`.",
        "",
        md_table(*tables["wall_histogram.csv"]) if tables["wall_histogram.csv"][0] else "",
        "",
        "Near-zero conorms at `1e-3 lam1`, by reported system. Cubic, hexagonal, tetragonal and orthorhombic rows are expected to fill the two- and three-zero columns. On this database they do: hexagonal cells sit in the two-zero column, orthorhombic and tetragonal cells mostly in the three-zero column, and cubic cells in the two- and three-zero columns. A minority of cubic and tetragonal cells have no near-zero, and 132 triclinic cells have three.",
        "",
        md_table(*tables["near_zeros.csv"]) if tables["near_zeros.csv"][0] else "",
        "",
        "## Closure",
        "",
        md_table(*tables["closure_counts.csv"]) if tables["closure_counts.csv"][0] else "",
        "",
        md_table(*tables["closure_flags.csv"]) if tables["closure_flags.csv"][0] else "",
        "",
        "## Per-slot closure",
        "",
        "Frame-consistent closure beside the uniform table. `walk_smaller` is the number of cells whose uniform `walk-tau` count is smaller than this closure.",
        "",
        md_table(*tables["perslot_closure.csv"]) if tables["perslot_closure.csv"][0] else "",
        "",
        f"Brute force (`delta_obtuse`, K=3) on every refused cell and 2000 hypothesis cells, plus K=4 on 200 of them, ran at {summary.get('brute_rate', 0):.2f} cells/s.",
        "",
        md_table(*tables["brute_check.csv"]) if tables["brute_check.csv"][0] else "",
        "",
        "## Three-degree rule",
        "",
        "Every cell at `sigma = sin(3°)/4`, by reported system and reduced-cell axis ratio (longest edge over shortest).",
        "",
        md_table(*tables["three_degree.csv"]) if tables["three_degree.csv"][0] else "",
        "",
        "## Tolerance calibration",
        "",
        "Distances are read while the path sits in the source sort of each Table C2 edge. "
        "`d_cos` is the cosine of a zero class or the cosine separation of a tie. "
        "`d_rel` is the relative length of a tie. `d_sig` is the same separation in units of the largest slot sigma; covariances are ignored, and it is reported for esd rows only. "
        "The combined rules are on the `any` rows: a cell is within when some zero-type edge passes the angle and some tie-type edge passes the length.",
        "",
        md_table(
            [row for row in tables["tolerance_calibration.csv"][0] if row["edge"] == "any"],
            tables["tolerance_calibration.csv"][1],
        ) if tables["tolerance_calibration.csv"][0] else "",
        "",
        "## Filtration gaps",
        "",
        "Widest log-gap of the merged path inside `(t0, sin 3°)`. `t0` is the cell's 1σ cosine band on esd rows and `1e-6` otherwise.",
        "",
        md_table(*tables["filtration_gaps.csv"]) if tables["filtration_gaps.csv"][0] else "",
        "",
        md_table(*tables["filtration_gap_sorts.csv"]) if tables["filtration_gap_sorts.csv"][0] else "",
        "",
        "## Symmetry",
        "",
        "Visible order counts the signed label swaps that fix the conorm idealised at `4 tau`. It is a lower bound on the holohedry (Table 4, column visible). "
        "`metric-above-reported` is a visible order larger than the reported system's holohedry. "
        "`reported-above-visible` requires a single closure member, so a symmetry that lives on another member is not flagged.",
        "",
        md_table(*tables["symmetry_flags.csv"]) if tables["symmetry_flags.csv"][0] else "",
        "",
        "Ten most common off-diagonal (reported, visible) pairs.",
        "",
        md_table(*tables["symmetry_off_diagonal.csv"]) if tables["symmetry_off_diagonal.csv"][0] else "",
        "",
        "## Classic traps",
        "",
        md_table(*tables["traps.csv"]) if tables["traps.csv"][0] else "",
        "",
        "## Niggli",
        "",
        "Cells whose closure has more than one member, and the number of those whose Niggli cell differs from the reduced cell's. A count of zero means every other Selling member reduced to the same Niggli cell.",
        "",
        md_table(*tables["niggli.csv"]) if tables["niggli.csv"][0] else "",
        "",
        "## Specialisation",
        "",
        "Populations at the measurement band (every transition inside its own cosine band) and at cosine tolerances `1e-2`, `1e-3`, `1e-4`.",
        "",
        _top_levels(tables["sorts_by_source.csv"][0]),
        "",
        "Ten most crossed edges. The histogram columns are half-decade bins of `log10 t` in cosine units.",
        "",
        _top_edges(tables["edge_transitions.csv"][0]),
        "",
        "Population of the ten most common sorts along the cosine thresholds.",
        "",
        _top_population(tables["sort_population.csv"][0]),
        "",
        "Fraction of cells whose first specialisation falls inside the cosine band of that edge (`ambiguous_esd`).",
        "",
        md_table(*tables["ambiguous_esd.csv"]) if tables["ambiguous_esd.csv"][0] else "",
        "",
        "## Spot checks",
        "",
        summary.get("spot_text", "Spot checks are filled by a second pass."),
        "",
        "## Issues",
        "",
    ]
    if not summary["issues"]:
        parts.append("None recorded.")
    for issue in summary["issues"]:
        parts.append(f"- **{issue['name']}** ({issue['n']}): {issue['detail']} Examples: {', '.join(issue['ids'])}.")
    parts.extend(["", "## Environment", "", "```", env.rstrip(), "```", ""])
    size = conorm_path.stat().st_size if conorm_path.is_file() else 0
    parts.append(f"Conorm file `{conorm_path}` is {size:,} bytes. It is committed only when that is under 10 MB.")
    parts.append("")
    return "\n".join(parts)


def _top_levels(rows):
    if not rows:
        return ""
    totals = Counter((row["sort"], row["tolerance"]) for row in rows for _ in range(row["n"]))
    # rows already are counts; rank sorts by total n
    rank = Counter()
    for row in rows:
        rank[row["sort"]] += row["n"]
    keep = {name for name, _n in rank.most_common(10)}
    shown = [row for row in rows if row["sort"] in keep]
    return md_table(shown, ["source", "sort", "tolerance", "n"])


def _top_edges(rows):
    if not rows:
        return ""
    rank = Counter()
    for row in rows:
        rank[row["edge"]] += row["n"]
    keep = {name for name, _n in rank.most_common(10)}
    shown = [row for row in rows if row["edge"] in keep]
    return md_table(shown, list(shown[0].keys()) if shown else ["edge"])


def _top_population(rows):
    if not rows:
        return ""
    rank = Counter()
    for row in rows:
        rank[row["sort"]] += row["n"]
    keep = {name for name, _n in rank.most_common(10)}
    shown = [row for row in rows if row["sort"] in keep]
    return md_table(shown, ["source", "sort", "t", "n"])


if __name__ == "__main__":
    main()
