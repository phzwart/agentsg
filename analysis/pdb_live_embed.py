#!/usr/bin/env python3
"""Live leanmap embedding of PDB root invariants (ε-net + step-wise scatter).

Opens an interactive matplotlib window that refreshes every ``--every`` training
steps. By default writes a side-by-side PNG: crystal system (left) and centring
P/C/I/F/R (right, with A/B/C merged into C).

Usage (leanmap venv recommended)::

    cd /Users/phzwart/Projects/agentsg
    KMP_DUPLICATE_LIB_OK=TRUE \\
    ../leanmap/.venv/bin/python analysis/pdb_live_embed.py --every 25

Default ε=1 Å (root-invariant L2). Auto ε from 1-NN quantiles is far too small
on PDB roots and yields R≈N — use ``--eps-crawl`` only to explore.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "analysis" / "data"
OUT = REPO / "analysis" / "figures"
DEFAULT_CACHE = DATA / "pdb_roots.npz"
DEFAULT_CACHE_SIM = DATA / "pdb_similarity_roots.npz"
DEFAULT_PDB = REPO / "data" / "pdb_cells.duckdb"
DEFAULT_GRAPH = DATA / "pdb_graph.pt"
DEFAULT_GRAPH_SIM = DATA / "pdb_graph_similarity.pt"
DEFAULT_GRAPH_SIM_PRIMITIVE = DATA / "pdb_graph_similarity_primitive.pt"
DEFAULT_STAGES = DATA / "pdb_graph_stages"
DEFAULT_STAGES_SIM = DATA / "pdb_graph_stages_similarity"
DEFAULT_STAGES_SIM_PRIMITIVE = DATA / "pdb_graph_stages_similarity_primitive"


def _ensure_leanmap():
    root = REPO.parent / "leanmap" / "src"
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import leanmap  # noqa: F401


def _ensure_agentsg():
    src = REPO / "agentsg" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_pdb_roots(
    cache: Path,
    *,
    pdb: Path | None = None,
    similarity: bool = False,
) -> tuple[np.ndarray, dict]:
    if similarity:
        cache = cache if cache != DEFAULT_CACHE else DEFAULT_CACHE_SIM
    if cache.is_file():
        data = np.load(cache)
        key = "S" if similarity and "S" in data else "X"
        X = data[key].astype(np.float32, copy=False)
        meta: dict = {"source": str(cache), "n": len(X), "similarity": similarity}
        for k in ("volume", "sg_number", "sg_hm"):
            if k in data:
                meta[k] = data[k]
        return X, meta

    from agentsg.cell.celldb import CellDatabase

    pdb_path = pdb or DEFAULT_PDB
    if not pdb_path.is_file():
        raise SystemExit(f"missing {pdb_path} — build with pdb_app or pass --cache")
    prefix = "s" if similarity else "r"
    cols = ",".join(f"{prefix}{i}" for i in range(6))
    db = CellDatabase(str(pdb_path))
    rows = db.sql(
        f"SELECT {cols}, volume, sg_number FROM cells "
        f"WHERE {prefix}0 IS NOT NULL ORDER BY pdb_id"
    )
    db.close()
    if not rows:
        kind = "similarity (s0..s5)" if similarity else "root (r0..r5)"
        raise SystemExit(
            f"{pdb_path} has no {kind} columns — run backfill_similarity_invariants()"
        )
    X = np.array([r[:6] for r in rows], dtype=np.float32)
    volume = np.array([r[6] for r in rows], dtype=np.float64)
    sg_number = np.array([r[7] for r in rows], dtype=np.int32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(X=X, volume=volume, sg_number=sg_number)
    if similarity:
        payload["S"] = X
    np.savez_compressed(cache, **payload)
    print(f"cached {len(X)} {'similarity' if similarity else 'root'} features -> {cache}")
    return X, {
        "source": str(pdb_path),
        "n": len(X),
        "volume": volume,
        "sg_number": sg_number,
        "similarity": similarity,
    }


def enrich_labels(meta: dict, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (crystal_system_codes, lattice_letter_codes) as int32."""
    _ensure_agentsg()
    from agentsg.cell.primitive import lattice_letter
    from agentsg.space_groups import SPACE_GROUPS, space_group

    systems = [
        "triclinic", "monoclinic", "orthorhombic", "tetragonal",
        "trigonal", "hexagonal", "cubic",
    ]
    sys_to_id = {s: i for i, s in enumerate(systems)}
    let_to_id = {"P": 0, "A": 1, "B": 2, "C": 3, "I": 4, "F": 5, "R": 6, "H": 6}

    sg_hm = meta.get("sg_hm")
    sg_num = np.asarray(meta.get("sg_number"), dtype=np.int32)
    if idx is not None:
        sg_num = sg_num[idx]
        if sg_hm is not None:
            sg_hm = sg_hm[idx]

    sys_by_num = np.empty(230, dtype=np.int32)
    hm_by_num = np.empty(230, dtype=object)
    for n, hm, _hall, cs in SPACE_GROUPS:
        sys_by_num[n - 1] = sys_to_id.get(cs, 0)
        hm_by_num[n - 1] = hm

    if sg_hm is not None:
        hm = np.asarray(sg_hm, dtype=object)
        sys_ids = np.array(
            [sys_to_id.get(space_group(str(h)).crystal_system, 0) for h in hm],
            dtype=np.int32,
        )
        let_ids = np.array([let_to_id.get(lattice_letter(str(h)), 0) for h in hm], dtype=np.int32)
    else:
        sys_ids = sys_by_num[np.clip(sg_num, 1, 230) - 1]
        hm = hm_by_num[np.clip(sg_num, 1, 230) - 1]
        let_ids = np.array([let_to_id.get(lattice_letter(str(h)), 0) for h in hm], dtype=np.int32)
    return sys_ids, let_ids


def primitive_only_mask(meta: dict) -> np.ndarray:
    """True for P-centred (primitive Bravais) cells."""
    _, let_ids = enrich_labels(meta)
    return let_ids == 0


def apply_row_mask(meta: dict, mask: np.ndarray) -> None:
    n = mask.shape[0]
    for k, v in list(meta.items()):
        if k in ("source", "n", "similarity"):
            continue
        if isinstance(v, np.ndarray) and len(v) == n:
            meta[k] = v[mask]
    meta["n"] = int(mask.sum())


def pick_epsilon(X: np.ndarray, *, n_sample: int, seed: int) -> tuple[float, dict]:
    import torch
    from leanmap.build.resolution import crawl_epsilon, format_epsilon_crawl
    from leanmap.metrics import wrap_metric

    Xt = torch.as_tensor(X, dtype=torch.float32)
    metric = wrap_metric("l2", X=Xt, n_neighbors=15, seed=seed)
    report = crawl_epsilon(
        Xt, metric, n_sample=min(n_sample, len(X)), seed=seed, n_rows=len(X),
    )
    print(format_epsilon_crawl(report))
    rec = report["recommend"]
    eps = float(rec["epsilon"]) if rec else float(report["nn1_quantiles"].get(0.01, 0.01))
    print(f"using epsilon={eps:.6g}  (projected R≈{rec['R_proj']:.0f})")
    return eps, report


def stratified_sample_idx(labels: np.ndarray, n_pick: int, seed: int) -> np.ndarray:
    """Balanced subsample: equal quota per class, then fill to n_pick."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n_pick >= n:
        return np.arange(n, dtype=np.int64)
    classes = np.unique(labels)
    per = max(1, n_pick // len(classes))
    picks = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        k = min(per, len(idx))
        picks.append(rng.choice(idx, size=k, replace=False))
    out = np.sort(np.concatenate(picks))
    if len(out) < n_pick:
        rest = np.setdiff1d(np.arange(n), out, assume_unique=True)
        extra = rng.choice(rest, size=min(n_pick - len(out), len(rest)), replace=False)
        out = np.sort(np.concatenate([out, extra]))
    return out.astype(np.int64)


def training_stratify_labels(
    sys_ids: np.ndarray,
    cent_ids: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Integer class codes for stratified train / epoch edge sampling."""
    if mode == "system":
        return np.asarray(sys_ids, dtype=np.int64)
    if mode == "centring":
        return np.asarray(cent_ids, dtype=np.int64)
    # system × centring — 7 systems, 5 centring types
    return (
        np.asarray(sys_ids, dtype=np.int64) * 10
        + np.asarray(cent_ids, dtype=np.int64)
    )


def stratified_display_idx(labels: np.ndarray, n_display: int, seed: int) -> np.ndarray:
    """Alias kept for callers; see :func:`stratified_sample_idx`."""
    return stratified_sample_idx(labels, n_display, seed)


def collapse_centring_ids(let_ids: np.ndarray) -> np.ndarray:
    """Map Bravais letters to P, C (A/B/C), I, F, R for display."""
    lid = np.asarray(let_ids, dtype=np.int32)
    out = np.zeros_like(lid)
    out[(lid >= 1) & (lid <= 3)] = 1  # A, B, C → C
    out[lid == 4] = 2  # I
    out[lid == 5] = 3  # F
    out[lid == 6] = 4  # R / H
    return out


def _color_spec(color_mode: str) -> tuple[str, float, tuple[str, ...]]:
    if color_mode == "system":
        return "tab10", 6.5, LiveScatter.SYSTEM_NAMES
    if color_mode == "centring":
        return "Set1", 4.5, LiveScatter.CENTRING_NAMES
    return "Set1", 7.5, LiveScatter.LETTER_NAMES


def _legend_for(sc, color: np.ndarray, names: tuple[str, ...]):
    from matplotlib.lines import Line2D

    present = sorted(set(int(c) for c in color))
    return [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=sc.cmap(sc.norm(c)), markersize=6,
            label=names[c] if c < len(names) else str(c),
        )
        for c in present
    ]


def save_scatter_png(
    Z: np.ndarray,
    color: np.ndarray,
    path: Path,
    *,
    color_mode: str,
    title: str,
    subtitle: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=color, s=2, linewidths=0, alpha=0.8)
    cmap, clim, names = _color_spec(color_mode)
    sc.set_cmap(cmap)
    sc.set_clim(-0.5, clim)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title)
    ax.legend(handles=_legend_for(sc, color, names), loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_dual_scatter_png(
    Z: np.ndarray,
    sys_color: np.ndarray,
    cent_color: np.ndarray,
    path: Path,
    *,
    title: str,
    subtitle: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    panels = (
        (axes[0], sys_color, "system", "crystal system"),
        (axes[1], cent_color, "centring", "centring (P, C, I, F, R)"),
    )
    for ax, panel_color, mode, panel_title in panels:
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=panel_color, s=2, linewidths=0, alpha=0.8)
        cmap, clim, names = _color_spec(mode)
        sc.set_cmap(cmap)
        sc.set_clim(-0.5, clim)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel_title)
        ax.legend(handles=_legend_for(sc, panel_color, names), loc="upper right", fontsize=8, framealpha=0.85)
    full_title = title if not subtitle else f"{title}\n{subtitle}"
    fig.suptitle(full_title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


class LiveScatter:
    """Refresh scatter every ``every`` steps; optionally write PNG snapshots."""

    SYSTEM_NAMES = (
        "triclinic", "monoclinic", "orthorhombic", "tetragonal",
        "trigonal", "hexagonal", "cubic",
    )
    LETTER_NAMES = ("P", "A", "B", "C", "I", "F", "R", "H")
    CENTRING_NAMES = ("P", "C", "I", "F", "R")

    def __init__(
        self,
        X: np.ndarray,
        display_idx: np.ndarray,
        color: np.ndarray,
        *,
        color_secondary: np.ndarray | None = None,
        color_mode: str,
        every: int,
        title: str,
        scatter_out: Path | None = None,
        scatter_dir: Path | None = None,
        interactive: bool = False,
    ):
        self.X = X
        self.display_idx = display_idx
        self.color = color[display_idx]
        self.color_secondary = (
            None if color_secondary is None else color_secondary[display_idx]
        )
        self.every = max(1, int(every))
        self.title = title
        self.color_mode = color_mode
        self.scatter_out = scatter_out
        self.scatter_dir = scatter_dir
        self._last = -1
        self._fig = self._ax = self._sc = None

        if interactive:
            import matplotlib.pyplot as plt

            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(8, 7))
            self._sc = self._ax.scatter([], [], c=[], s=3, linewidths=0, alpha=0.75)
            self._ax.set_aspect("equal", adjustable="datalim")
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._fig.canvas.manager.set_window_title("agentsg PDB — leanmap live")
            self._fig.tight_layout()
            plt.show(block=False)

    def _embed(self, model) -> np.ndarray:
        import torch

        was = model.training
        model.eval()
        Xi = torch.as_tensor(self.X[self.display_idx], dtype=torch.float32)
        dev = next(model.parameters()).device
        with torch.no_grad():
            Z, _ = model.embed(Xi.to(dev), return_score=False)
        if was:
            model.train()
        return Z.detach().cpu().numpy()

    def _draw(self, epoch: int, step: int, model, subtitle: str = ""):
        Z = self._embed(model)
        if self.color_mode == "both":
            if self.scatter_out is not None:
                save_dual_scatter_png(
                    Z, self.color, self.color_secondary, self.scatter_out,
                    title=self.title,
                    subtitle=f"epoch {epoch} step {step}  {subtitle}".strip(),
                )
            if self.scatter_dir is not None:
                save_dual_scatter_png(
                    Z, self.color, self.color_secondary,
                    self.scatter_dir / f"epoch_{epoch:03d}_step_{step:05d}.png",
                    title=self.title,
                    subtitle=f"epoch {epoch} step {step}  {subtitle}".strip(),
                )
            return

        if self._sc is not None:
            import matplotlib.pyplot as plt

            self._sc.set_offsets(Z)
            self._sc.set_array(self.color)
            cmap, clim, _ = _color_spec(self.color_mode)
            self._sc.set_cmap(cmap)
            self._sc.set_clim(-0.5, clim)
            self._ax.set_title(f"{self.title}\nepoch {epoch} step {step}  {subtitle}")
            self._fig.canvas.draw_idle()
            plt.pause(0.001)

        if self.scatter_out is not None:
            save_scatter_png(
                Z, self.color, self.scatter_out,
                color_mode=self.color_mode, title=self.title,
                subtitle=f"epoch {epoch} step {step}  {subtitle}".strip(),
            )
        if self.scatter_dir is not None:
            save_scatter_png(
                Z, self.color,
                self.scatter_dir / f"epoch_{epoch:03d}_step_{step:05d}.png",
                color_mode=self.color_mode, title=self.title,
                subtitle=f"epoch {epoch} step {step}  {subtitle}".strip(),
            )

    def on_step(self, epoch: int, step: int, model, metrics: dict):
        if step - self._last < self.every:
            return
        self._last = step
        self._draw(epoch, step, model)

    def __call__(self, epoch: int, model, metrics: dict):
        self._draw(
            epoch, self._last if self._last >= 0 else 0, model,
            subtitle=f"loss geom={metrics.get('geom', float('nan')):.3f}",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=None,
                    help="npz cache (default: pdb_roots.npz or pdb_similarity_roots.npz)")
    ap.add_argument("--pdb", type=Path, default=DEFAULT_PDB,
                    help="DuckDB with precomputed r0..r5 / s0..s5")
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH,
                    help="cache leanmap graph pyramid (build once, reuse)")
    ap.add_argument("--stages", type=Path, default=DEFAULT_STAGES,
                    help="Zarr graph stages for ε-net / kNN spill")
    ap.add_argument("--color", choices=("system", "lattice", "centring", "both"),
                    default="both",
                    help="scatter colour: crystal system, Bravais letter, "
                         "P/C/I/F/R centring, or side-by-side both (default)")
    ap.add_argument("--train-n", type=int, default=0,
                    help="rows to train on (0 = all in cache)")
    ap.add_argument("--display-n", type=int, default=25_000,
                    help="stratified subsample for the live scatter")
    ap.add_argument("--stratify-by", choices=("system", "centring", "both"),
                    default="both",
                    help="stratify train subsample and per-epoch edge draws")
    ap.add_argument("--class-sample-mix", type=float, default=1.0,
                    help="blend edge sampling toward equal stratify-by class "
                         "coverage (0=off, 1=full; needs class labels)")
    ap.add_argument("--pyramid-weights", default="1,1,1,1",
                    help="comma-separated edge attraction weights per pyramid "
                         "level, finest first (default: 1,1,1,1)")
    ap.add_argument("--every", type=int, default=25, help="refresh every N steps")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="ε-net radius in Å (root L2 = Å); 1.0 works well at PDB scale")
    ap.add_argument("--eps-crawl", action="store_true",
                    help="run subsampled ε crawl instead of fixed --epsilon")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lambda-geo", type=float, default=0.0,
                    help="geodesic MDS + Procrustes landmark backbone weight (0=off)")
    ap.add_argument("--geo-ramp", default="0.2,0.45",
                    help="geo loss ramp (start,end) as fraction of training; "
                         "use 0,0 for full weight from epoch 1")
    ap.add_argument("--knn-mode", default="metric_net",
                    choices=("auto", "brute", "nndescent", "ann", "ivf", "metric_net"),
                    help="rep graph: metric_net (ball of 2ε), ann/nndescent/ivf kNN, "
                         "auto, or brute (default: metric_net)")
    ap.add_argument("--neighbor-radius", type=float, default=None,
                    help="metric_net ball radius (default: 2×epsilon)")
    ap.add_argument("--similarity", action="store_true",
                    help="use volume-normalised roots RI/V**(1/3) (shape-only, dimensionless)")
    ap.add_argument("--model-out", type=Path, default=None)
    ap.add_argument("--scatter-out", type=Path, default=None,
                    help="write latest scatter PNG here (updated each epoch/step)")
    ap.add_argument("--scatter-dir", type=Path, default=None,
                    help="optional directory for per-step PNG snapshots")
    ap.add_argument("--interactive", action="store_true",
                    help="open live matplotlib window (needs a display)")
    ap.add_argument("--primitive-only", action="store_true",
                    help="embed P-centred (primitive Bravais) cells only")
    ap.add_argument("--resume", type=Path, default=None,
                    help="load encoder weights from saved .pt and continue training")
    ap.add_argument("--rebuild-graph", action="store_true")
    args = ap.parse_args()

    tag = "similarity" if args.similarity else "root"
    if args.primitive_only:
        tag = f"{tag}_primitive"
    if args.graph == DEFAULT_GRAPH and args.similarity:
        args.graph = DEFAULT_GRAPH_SIM_PRIMITIVE if args.primitive_only else DEFAULT_GRAPH_SIM
    if args.stages == DEFAULT_STAGES and args.similarity:
        args.stages = (
            DEFAULT_STAGES_SIM_PRIMITIVE if args.primitive_only else DEFAULT_STAGES_SIM
        )
    if args.scatter_out is None:
        args.scatter_out = OUT / f"pdb_live_scatter_{tag}.png"
    if args.model_out is None:
        args.model_out = OUT / f"pdb_leanmap_live_{tag}.pt"

    if args.cache is None:
        args.cache = DEFAULT_CACHE_SIM if args.similarity else DEFAULT_CACHE

    _ensure_leanmap()
    _ensure_agentsg()
    import torch
    from leanmap import PLANEConfig, fit

    X, meta = load_pdb_roots(args.cache, pdb=args.pdb, similarity=args.similarity)
    if args.similarity:
        print("features: similarity invariant s0..s5 from DuckDB")
    if args.primitive_only:
        pmask = primitive_only_mask(meta)
        X = X[pmask]
        apply_row_mask(meta, pmask)
        print(f"primitive-only filter: {len(X):,} P-centred cells")
    n_all = len(X)
    sys_ids_all, let_ids_all = enrich_labels(meta)
    cent_ids_all = collapse_centring_ids(let_ids_all)
    strat_labels_all = training_stratify_labels(
        sys_ids_all, cent_ids_all, args.stratify_by,
    )

    if args.train_n and args.train_n < n_all:
        train_idx = stratified_sample_idx(strat_labels_all, args.train_n, args.seed)
        train_idx = np.sort(train_idx)
        X_train = X[train_idx]
        print(f"stratified train subsample: {len(train_idx):,} rows by {args.stratify_by}")
    else:
        train_idx = None
        X_train = X

    sys_ids, let_ids = enrich_labels(meta, train_idx)
    cent_ids = collapse_centring_ids(let_ids)
    strat_labels = training_stratify_labels(sys_ids, cent_ids, args.stratify_by)
    if args.color == "system":
        color_train, color_secondary = sys_ids, None
    elif args.color == "lattice":
        color_train, color_secondary = let_ids, None
    elif args.color == "centring":
        color_train, color_secondary = cent_ids, None
    else:
        color_train, color_secondary = sys_ids, cent_ids

    stratify = strat_labels if args.color == "both" else color_train
    display_idx = stratified_sample_idx(stratify, args.display_n, args.seed)
    feat_label = "similarity invariant" if args.similarity else "root invariant"
    color_desc = {
        "system": "crystal system",
        "lattice": "Bravais letter",
        "centring": "centring P/C/I/F/R",
        "both": "system + centring",
    }[args.color]
    print(f"training on {len(X_train):,} / {n_all:,} PDB {feat_label}s; "
          f"display {len(display_idx):,} points by {color_desc}; "
          f"epoch stratify={args.stratify_by} mix={args.class_sample_mix}")

    eps_report = None
    if args.eps_crawl:
        eps, eps_report = pick_epsilon(X_train, n_sample=10_000, seed=args.seed)
    else:
        eps = float(args.epsilon)
        unit = " (L2, dimensionless)" if args.similarity else " Å (root-invariant L2)"
        print(f"using epsilon={eps}{unit}")

    cfg = PLANEConfig.for_scale(len(X_train))
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.epsilon = eps
    cfg.lambda_geo = args.lambda_geo
    gr = [float(x) for x in str(args.geo_ramp).split(",") if x.strip()]
    cfg.geo_ramp = (gr[0], gr[1]) if len(gr) >= 2 else (0.0, 0.0)
    cfg.knn_mode = args.knn_mode
    if args.neighbor_radius is not None:
        cfg.neighbor_radius = float(args.neighbor_radius)
    cfg.graph_stages_dir = str(args.stages)
    cfg.class_sample_mix = float(args.class_sample_mix)
    pw = [float(x) for x in str(args.pyramid_weights).split(",") if x.strip()]
    cfg.pyramid_level_weights = tuple(pw)
    if args.epochs is not None:
        cfg.epochs = args.epochs

    init_state_dict = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise SystemExit(f"resume model not found: {args.resume}")
        payload = torch.load(str(args.resume), map_location="cpu", weights_only=False)
        init_state_dict = payload["state_dict"]
        saved_cfg = payload.get("config") or {}
        for key in ("epsilon", "knn_mode", "n_landmarks", "n_neighbors", "width", "depth",
                    "pyramid_level_weights", "class_sample_mix", "landmark_sample_mix",
                    "epoch_unit", "landmark_epoch_samples"):
            if key in saved_cfg and getattr(cfg, key, None) != saved_cfg[key]:
                if hasattr(cfg, key):
                    setattr(cfg, key, saved_cfg[key])
        print(f"resuming weights from {args.resume} "
              f"(saved epochs={saved_cfg.get('epochs', '?')})")

    print(f"pyramid_level_weights={cfg.pyramid_level_weights}")

    live = LiveScatter(
        X_train,
        display_idx,
        color_train,
        color_secondary=color_secondary,
        color_mode=args.color,
        every=args.every,
        title=f"PDB {feat_label}s",
        scatter_out=args.scatter_out,
        scatter_dir=args.scatter_dir,
        interactive=args.interactive,
    )

    graph_path = str(args.graph)
    rebuild = args.rebuild_graph or not args.graph.is_file()
    if rebuild:
        if args.graph.is_file():
            args.graph.unlink()
        if args.stages.is_dir():
            shutil.rmtree(args.stages)
        print(f"building graph -> {graph_path} (ε={eps}, stages={args.stages}) …")
    else:
        print(f"reusing graph {graph_path}")

    result = fit(
        X_train,
        dist_fn="l2",
        config=cfg,
        callbacks=[live],
        graph_path=graph_path,
        rebuild_graph=rebuild,
        class_labels=strat_labels if args.class_sample_mix > 0 else None,
        init_state_dict=init_state_dict,
    )

    with torch.no_grad():
        Z, score = result.embed(torch.as_tensor(X_train, dtype=torch.float32))
    Z = Z.detach().cpu().numpy()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(args.model_out))
    np.save(args.model_out.with_suffix(".embedding.npy"), Z)

    stats = {
        "n_train": len(X_train),
        "n_all": n_all,
        "feature": tag,
        "epsilon": eps,
        "color": args.color,
        "stratify_by": args.stratify_by,
        "class_sample_mix": args.class_sample_mix,
        "pyramid_level_weights": list(cfg.pyramid_level_weights),
        "graph": graph_path,
    }
    if eps_report and eps_report.get("recommend"):
        stats["epsilon_crawl"] = eps_report["recommend"]
    with (args.model_out.with_suffix(".json")).open("w") as fh:
        json.dump(stats, fh, indent=2)

    if args.color == "both":
        save_dual_scatter_png(
            Z[display_idx], color_train[display_idx], cent_ids[display_idx],
            args.scatter_out.with_name("pdb_live_scatter_final.png"),
            title=f"PDB {feat_label}s",
            subtitle="final embedding",
        )
    else:
        save_scatter_png(
            Z[display_idx], color_train[display_idx],
            args.scatter_out.with_name("pdb_live_scatter_final.png"),
            color_mode=args.color,
            title=f"PDB {feat_label}s ({args.color})",
            subtitle="final embedding",
        )
    print(f"saved {args.model_out}")
    print(f"saved scatter {args.scatter_out}")
    if args.interactive:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
