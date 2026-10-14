#!/usr/bin/env python3
"""Leanmap plane of the volume-normalised sorted linear key.

The feature is the archive search key divided by the cube root of the
primitive-cell volume::

    sort(p_ij / sqrt(sum p)) / V_primitive**(1/3)

``l0``..``l5`` in the DuckDB are that key on the primitive lattice. The
stored volume is the conventional cell, so the scale is
``volume / centring_multiplicity``. Columns ``s0``..``s5`` are the older
square-root similarity and are not read.

Usage (leanmap venv, agentsg DuckDB bindings on ``PYTHONPATH``)::

    cd /path/to/agentsg
    PYTHONPATH="agentsg/src:.venv/lib/python3.12/site-packages" \\
    KMP_DUPLICATE_LIB_OK=TRUE \\
    ../leanmap/.venv/bin/python linear_embed/embed.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DEFAULT_PDB = REPO / "data" / "pdb_cells.duckdb"
DEFAULT_CACHE = OUT / "linear_similarity.npz"

_MULT = {"P": 1.0, "A": 2.0, "B": 2.0, "C": 2.0, "I": 2.0, "F": 4.0, "R": 3.0, "H": 3.0}


def _ensure_leanmap():
    root = REPO.parent / "leanmap" / "src"
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import leanmap  # noqa: F401


def _letter(symbol: str) -> str:
    text = str(symbol or "P").strip().lstrip("-")
    return text[0].upper() if text else "P"


def primitive_volume_scale(volume: np.ndarray, sg_hm: np.ndarray) -> np.ndarray:
    """Cube root of the primitive-cell volume.

    The linear key is computed on the primitive lattice. ``volume`` in the
    archive is the deposited conventional cell, larger by the centring
    multiplicity.
    """
    mult = np.fromiter((_MULT[_letter(s)] for s in sg_hm), dtype=np.float64, count=len(sg_hm))
    return (np.asarray(volume, dtype=np.float64) / mult) ** (1.0 / 3.0)


def load_features(pdb: Path, cache: Path) -> tuple[np.ndarray, dict]:
    if cache.is_file():
        data = np.load(cache, allow_pickle=True)
        meta = {
            "source": str(data["source"]),
            "n": int(len(data["S"])),
            "volume": data["volume"],
            "sg_number": data["sg_number"],
            "sg_hm": data["sg_hm"],
            "pdb_id": data["pdb_id"],
        }
        return data["S"].astype(np.float32, copy=False), meta

    import duckdb

    if not pdb.is_file():
        raise SystemExit(f"PDB database not found: {pdb}")
    db = duckdb.connect(str(pdb), read_only=True)
    rows = db.execute(
        "SELECT pdb_id, volume, sg_number, sg_hm, "
        "l0, l1, l2, l3, l4, l5 FROM cells "
        "WHERE l0 IS NOT NULL ORDER BY pdb_id"
    ).fetchall()
    db.close()
    if not rows:
        raise SystemExit(f"{pdb} has no l0..l5 — run backfill_linear_keys first")

    pdb_id = np.array([r[0] for r in rows], dtype=object)
    volume = np.array([r[1] for r in rows], dtype=np.float64)
    sg_number = np.array([r[2] for r in rows], dtype=np.int32)
    sg_hm = np.array([r[3] for r in rows], dtype=object)
    linear = np.array([r[4:10] for r in rows], dtype=np.float64)
    scale = primitive_volume_scale(volume, sg_hm)
    S = (linear / scale[:, None]).astype(np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        S=S,
        volume=volume,
        sg_number=sg_number,
        sg_hm=sg_hm,
        pdb_id=pdb_id,
        source=np.array(str(pdb)),
    )
    print(f"cached {len(S):,} volume-normalised linear keys -> {cache}")
    return S, {
        "source": str(pdb),
        "n": len(S),
        "volume": volume,
        "sg_number": sg_number,
        "sg_hm": sg_hm,
        "pdb_id": pdb_id,
    }


def crystal_system_ids(sg_number: np.ndarray) -> np.ndarray:
    src = REPO / "agentsg" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from agentsg.space_groups import SPACE_GROUPS

    systems = (
        "triclinic", "monoclinic", "orthorhombic", "tetragonal",
        "trigonal", "hexagonal", "cubic",
    )
    sys_to_id = {name: i for i, name in enumerate(systems)}
    by_num = np.zeros(230, dtype=np.int32)
    for number, _hm, _hall, crystal in SPACE_GROUPS:
        by_num[number - 1] = sys_to_id[crystal]
    return by_num[np.clip(sg_number, 1, 230) - 1]


SYSTEM_NAMES = (
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic",
)


def _fps_indices(S: np.ndarray, idx: np.ndarray, take: int, seed: int) -> np.ndarray:
    import torch
    from leanmap.distance import EuclideanDistance
    from leanmap.landmarks import fps_init_indices

    take = min(int(take), len(idx))
    X = torch.as_tensor(np.ascontiguousarray(S[idx]), dtype=torch.float32)
    local = fps_init_indices(X, EuclideanDistance(), take, seed=int(seed))
    return idx[local.detach().cpu().numpy()]


def system_landmarks(
    S: np.ndarray,
    sg_number: np.ndarray,
    per_system: int,
    seed: int = 0,
    sg_hm: np.ndarray | None = None,
):
    """Farthest-point landmarks inside each crystal system, capped at ``per_system``.

    Cubic cells of one centring are the same shape, so cubic landmarks are one
    point per centring letter instead of a cap of ``per_system``.
    """
    sys_ids = crystal_system_ids(sg_number)
    chosen = []
    counts = {}
    for sid, name in enumerate(SYSTEM_NAMES):
        idx = np.flatnonzero(sys_ids == sid)
        if len(idx) == 0:
            counts[name] = 0
            continue
        if name == "cubic":
            if sg_hm is None:
                raise ValueError("cubic landmarks per centring need Hermann–Mauguin symbols")
            letters = np.array([_letter(s) for s in sg_hm[idx]])
            for j, let in enumerate(sorted(set(letters.tolist()))):
                sub = idx[letters == let]
                picked = _fps_indices(S, sub, 1, seed + 100 + j)
                chosen.append(picked)
                counts[f"cubic {let}"] = int(len(picked))
                print(f"landmarks cubic {let}: {len(picked)} of {len(sub):,}", flush=True)
            continue
        picked = _fps_indices(S, idx, per_system, seed + sid)
        chosen.append(picked)
        counts[name] = int(len(picked))
        print(f"landmarks {name}: {len(picked)} of {len(idx):,}", flush=True)
    order = np.concatenate(chosen)
    M = np.ascontiguousarray(S[order], dtype=np.float32)
    print(f"landmarks total {len(M):,}", flush=True)
    return M, counts


def centring_ids(sg_hm: np.ndarray) -> np.ndarray:
    """P, C (A/B/C merged), I, F, R."""
    letter_to_id = {"P": 0, "A": 1, "B": 1, "C": 1, "I": 2, "F": 3, "R": 4, "H": 4}
    return np.fromiter(
        (letter_to_id.get(_letter(s), 0) for s in sg_hm),
        dtype=np.int32,
        count=len(sg_hm),
    )


class EpochGif:
    """Append one scatter frame to an animated GIF at the end of each epoch."""

    def __init__(self, S, color, path: Path, *, every: int = 1, seed: int = 0,
                 max_points: int = 80_000):
        self.S = np.asarray(S, dtype=np.float32)
        n = len(self.S)
        if n > max_points:
            rng = np.random.default_rng(seed)
            self.idx = np.sort(rng.choice(n, size=max_points, replace=False))
        else:
            self.idx = None
        self.color = np.asarray(color if self.idx is None else color[self.idx])
        self.path = Path(path)
        self.every = max(1, int(every))
        self.frames = []
        self.lim = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)

    def __call__(self, epoch: int, model, metrics: dict):
        if int(epoch) % self.every:
            return
        Z = self._embed(model)
        self._append(int(epoch), Z, metrics)
        loss = metrics.get("loss", float("nan"))
        print(f"live gif epoch {epoch} loss={loss:.4f} -> {self.path}", flush=True)

    def _embed(self, model) -> np.ndarray:
        import torch

        was = model.training
        model.eval()
        X = self.S if self.idx is None else self.S[self.idx]
        dev = next(model.parameters()).device
        chunks = []
        with torch.no_grad():
            t = torch.as_tensor(X, dtype=torch.float32)
            for start in range(0, len(t), 8192):
                z, _ = model.embed(t[start:start + 8192].to(dev), return_score=False)
                chunks.append(z.detach().cpu().numpy())
        if was:
            model.train()
        return np.concatenate(chunks, axis=0)

    def _limits(self, Z: np.ndarray):
        lo = np.nanpercentile(Z, 0.4, axis=0)
        hi = np.nanpercentile(Z, 99.6, axis=0)
        span = np.maximum(hi - lo, 1e-3)
        pad = 0.06 * span
        lo, hi = lo - pad, hi + pad
        if self.lim is None:
            self.lim = (lo, hi)
        else:
            self.lim = (
                np.minimum(self.lim[0], lo),
                np.maximum(self.lim[1], hi),
            )
        return self.lim

    def _append(self, epoch: int, Z: np.ndarray, metrics: dict):
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from PIL import Image

        names = (
            "triclinic", "monoclinic", "orthorhombic", "tetragonal",
            "trigonal", "hexagonal", "cubic",
        )
        lo, hi = self._limits(Z)
        fig, ax = plt.subplots(figsize=(7.2, 6.6))
        sc = ax.scatter(
            Z[:, 0], Z[:, 1], c=self.color, s=2, linewidths=0,
            alpha=0.75, rasterized=True,
        )
        sc.set_cmap("tab10")
        sc.set_clim(-0.5, 6.5)
        present = sorted(set(int(c) for c in self.color))
        ax.legend(
            handles=[
                Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=sc.cmap(sc.norm(c)), markersize=6,
                    label=names[c],
                )
                for c in present
            ],
            loc="upper right", fontsize=8, framealpha=0.85,
        )
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        loss = metrics.get("loss", float("nan"))
        geo = metrics.get("geo", float("nan"))
        ax.set_title(f"epoch {epoch}   loss {loss:.3f}   geo {geo:.3f}")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        frame = img.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        self.frames.append(frame)
        self.frames[0].save(
            self.path,
            save_all=True,
            append_images=self.frames[1:],
            duration=120,
            loop=0,
            disposal=2,
        )


def save_scatter(Z, sys_ids, cent_ids, path: Path, *, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    path.parent.mkdir(parents=True, exist_ok=True)
    systems = (
        "triclinic", "monoclinic", "orthorhombic", "tetragonal",
        "trigonal", "hexagonal", "cubic",
    )
    centrings = ("P", "C", "I", "F", "R")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    panels = (
        (axes[0], sys_ids, "tab10", 6.5, systems, "crystal system"),
        (axes[1], cent_ids, "Set1", 4.5, centrings, "centring"),
    )
    for ax, color, cmap, clim, names, panel_title in panels:
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=color, s=2, linewidths=0, alpha=0.75)
        sc.set_cmap(cmap)
        sc.set_clim(-0.5, clim)
        present = sorted(set(int(c) for c in color))
        handles = [
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=sc.cmap(sc.norm(c)), markersize=6,
                label=names[c],
            )
            for c in present
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel_title)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--epsilon", type=float, default=0.05,
                    help="target ε on the dimensionless key; the large-N preset "
                         "widens δ when this would keep almost every row")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None,
                    help="AdamW learning rate; omit to keep the large-N preset (1e-3)")
    ap.add_argument("--n-landmarks", type=int, default=None,
                    help="landmark count; omit to keep the large-N preset (512)")
    ap.add_argument("--landmarks-per-system", type=int, default=0,
                    help="farthest-point landmarks inside each crystal system, "
                         "capped at this count (0 = one global set)")
    ap.add_argument("--live-every", type=int, default=0,
                    help="append a frame to embedding.gif every N epochs (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lambda-geo", type=float, default=None,
                    help="geodesic weight; omit to keep the large-N preset (0.5)")
    ap.add_argument("--geo-ramp", default=None,
                    help="start,end fractions for the geodesic weight; "
                         "0,0 turns it on from the first epoch. "
                         "Omit to keep the preset (0.2,0.45)")
    ap.add_argument("--lambda-frame", type=float, default=None,
                    help="local rigidity weight; omit to leave it off")
    ap.add_argument("--lambda-anchor", type=float, default=None,
                    help="Procrustes weight inside the geodesic term; "
                         "0 keeps pairwise distances and leaves the gauge to the frame")
    ap.add_argument("--frame-tangent", action=argparse.BooleanOptionalAction, default=None,
                    help="project the frame loss onto a 2-D tangent; "
                         "--no-frame-tangent matches the key directly")
    ap.add_argument("--min-dist", type=float, default=None,
                    help="how far non-neighbors keep being pushed; "
                         "omit to keep the preset (0.5)")
    ap.add_argument("--subsample", type=int, default=0,
                    help="random subset size for a short trial (0 = all rows)")
    ap.add_argument("--systems", default="",
                    help="comma-separated crystal systems to keep, for example "
                         "tetragonal,trigonal,hexagonal,cubic")
    ap.add_argument("--schedule", choices=("scale", "large"), default="scale",
                    help="large keeps the N>200k recipe (50 epochs, warm start, "
                         "landmark epochs) on a smaller subset")
    args = ap.parse_args()

    _ensure_leanmap()
    import torch
    from leanmap import PLANEConfig, fit

    S, meta = load_features(args.pdb, args.cache)
    system_names = [s.strip().lower() for s in str(args.systems).split(",") if s.strip()]
    if system_names:
        unknown = [s for s in system_names if s not in SYSTEM_NAMES]
        if unknown:
            raise SystemExit(
                "unknown crystal system: " + ", ".join(unknown)
                + ". Known: " + ", ".join(SYSTEM_NAMES)
            )
        sys_ids = crystal_system_ids(meta["sg_number"])
        wanted = np.array([SYSTEM_NAMES.index(s) for s in system_names], dtype=np.int32)
        idx = np.flatnonzero(np.isin(sys_ids, wanted))
        if len(idx) == 0:
            raise SystemExit("no cells in the requested crystal systems")
        S = S[idx]
        for key in ("volume", "sg_number", "sg_hm", "pdb_id"):
            meta[key] = meta[key][idx]
        meta["n"] = len(S)
        kept = crystal_system_ids(meta["sg_number"])
        for sid, name in enumerate(SYSTEM_NAMES):
            n_sys = int((kept == sid).sum())
            if n_sys:
                print(f"kept {name}: {n_sys:,}", flush=True)
    if args.subsample and args.subsample < len(S):
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(S), size=args.subsample, replace=False))
        S = S[idx]
        for key in ("volume", "sg_number", "sg_hm", "pdb_id"):
            meta[key] = meta[key][idx]
        meta["n"] = len(S)

    n = len(S)
    norms = np.linalg.norm(S, axis=1)
    print(
        f"loaded {n:,} keys, dim={S.shape[1]}  "
        f"||s|| median={np.median(norms):.4f} "
        f"[{norms.min():.4f}, {norms.max():.4f}]"
    )

    cfg = PLANEConfig.for_scale(n)
    if args.schedule == "large":
        # The subset is far below the 200k switch, so for_scale would drop to
        # the mid-size preset. Keep the recipe the full-archive runs used.
        cfg.width = 512
        cfg.depth = 3
        cfg.epochs = 50
        cfg.coarse_first_frac = 0.2
        cfg.warm_start_steps = 50
        cfg.warm_start_layout = "auto"
        cfg.epoch_unit = "landmarks"
        cfg.landmark_epoch_samples = 128.0
        cfg.landmark_sample_mix = 0.75
        cfg.delta = "auto"
        cfg.epoch_active_rows = 50_000
        cfg.epoch_overlap = 0.2
        cfg.epoch_cover_visits = 1
    cfg.seed = int(args.seed)
    cfg.device = args.device
    cfg.epsilon = float(args.epsilon)
    if args.lambda_geo is not None:
        cfg.lambda_geo = float(args.lambda_geo)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.lr is not None:
        cfg.lr = float(args.lr)
    if args.n_landmarks is not None:
        cfg.n_landmarks = int(args.n_landmarks)
    if args.geo_ramp is not None:
        parts = [float(x) for x in str(args.geo_ramp).split(",") if x.strip()]
        if len(parts) != 2:
            raise SystemExit("--geo-ramp needs two fractions, for example 0,0")
        cfg.geo_ramp = (parts[0], parts[1])
    if args.lambda_frame is not None:
        cfg.lambda_frame = float(args.lambda_frame)
    if args.lambda_anchor is not None:
        cfg.lambda_anchor = float(args.lambda_anchor)
    if args.frame_tangent is not None:
        cfg.frame_tangent = bool(args.frame_tangent)
    if args.min_dist is not None:
        cfg.min_dist = float(args.min_dist)

    landmarks = None
    landmark_counts = None
    if args.landmarks_per_system:
        landmarks, landmark_counts = system_landmarks(
            S, meta["sg_number"], int(args.landmarks_per_system),
            seed=int(args.seed), sg_hm=meta["sg_hm"],
        )
        cfg.n_landmarks = int(len(landmarks))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "graph.pt"
    print(
        f"leanmap: epochs={cfg.epochs} lr={cfg.lr} n_landmarks={cfg.n_landmarks} "
        f"epsilon={cfg.epsilon} lambda_geo={cfg.lambda_geo} "
        f"geo_ramp={cfg.geo_ramp} lambda_anchor={cfg.lambda_anchor} "
        f"lambda_frame={cfg.lambda_frame} frame_tangent={cfg.frame_tangent} "
        f"min_dist={cfg.min_dist} width={cfg.width} "
        f"epoch_unit={cfg.epoch_unit} "
        f"device={cfg.device} knn_mode={cfg.knn_mode}"
    )

    callbacks = []
    if args.live_every:
        gif = EpochGif(
            S, crystal_system_ids(meta["sg_number"]), out_dir / "embedding.gif",
            every=args.live_every, seed=int(args.seed),
        )
        callbacks.append(gif)
        print(f"live gif every {args.live_every} epoch(s) -> {gif.path}")

    result = fit(
        S,
        dist_fn="l2",
        config=cfg,
        graph_path=str(graph_path),
        rebuild_graph=not graph_path.is_file(),
        callbacks=callbacks,
        precomputed_landmarks=landmarks,
    )
    with torch.no_grad():
        Z, score = result.embed(torch.as_tensor(S, dtype=torch.float32))
    Z = Z.detach().cpu().numpy()
    score = score.detach().cpu().numpy()

    model_path = out_dir / "model.pt"
    result.save(str(model_path))
    np.save(out_dir / "embedding.npy", Z)
    np.save(out_dir / "score.npy", score)

    sys_ids = crystal_system_ids(meta["sg_number"])
    cent_ids = centring_ids(meta["sg_hm"])
    save_scatter(
        Z, sys_ids, cent_ids, out_dir / "embedding_by_system.png",
        title=f"volume-normalised linear key — {n:,} PDB cells",
    )

    saved = torch.load(str(model_path), map_location="cpu", weights_only=False)
    saved_cfg = saved.get("config") or {}
    stats = {
        "feature": "sorted_linear_key / V_primitive**(1/3)",
        "n": n,
        "d_in": int(S.shape[1]),
        "d_out": int(Z.shape[1]),
        "epsilon": float(saved_cfg.get("epsilon", cfg.epsilon)),
        "delta": float(saved_cfg["delta"]) if isinstance(saved_cfg.get("delta"), (int, float)) else None,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "lambda_geo": cfg.lambda_geo,
        "geo_ramp": list(cfg.geo_ramp),
        "lambda_anchor": cfg.lambda_anchor,
        "lambda_frame": cfg.lambda_frame,
        "frame_tangent": cfg.frame_tangent,
        "min_dist": cfg.min_dist,
        "systems": system_names or list(SYSTEM_NAMES),
        "n_landmarks": cfg.n_landmarks,
        "landmarks_per_system": landmark_counts,
        "geodesic_spearman": (getattr(result, "graph_stats", None) or {}).get("geodesic_spearman"),
        "geodesic_stress": (getattr(result, "graph_stats", None) or {}).get("geodesic_stress"),
        "device": cfg.device,
        "source": meta["source"],
        "norm_median": float(np.median(norms)),
        "graph": str(graph_path),
        "model": str(model_path),
    }
    with (out_dir / "stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"saved {Z.shape} -> {out_dir / 'embedding.npy'}")
    print(f"saved {model_path}")


if __name__ == "__main__":
    main()
