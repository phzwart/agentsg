#!/usr/bin/env python3
"""Embed unit-cell manifolds with leanmap (ε-net graph + PLANE).

Features are Kurlin root invariants (6D); the ambient metric is Euclidean L2,
which equals root distance in Å.

Usage::

    # Full PDB (precomputed roots from DuckDB)
    PYTHONPATH=agentsg/src python analysis/embed_cells.py --pdb data/pdb_cells.duckdb

    # Cached root matrix
    PYTHONPATH=agentsg/src python analysis/embed_cells.py --roots analysis/data/pdb_roots.npy

Requires leanmap (sibling checkout or pip install) and matplotlib for plots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis" / "figures"
DATA = REPO / "analysis" / "data"
DEFAULT_PDB = REPO / "data" / "pdb_cells.duckdb"
DEFAULT_STREAM = REPO / "data" / "blac_new_v0_nomulti.stream"


def _ensure_leanmap():
    leanmap_root = REPO.parent / "leanmap" / "src"
    if leanmap_root.is_dir() and str(leanmap_root) not in sys.path:
        sys.path.insert(0, str(leanmap_root))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import leanmap  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "leanmap not found — pip install leanmap or clone ../leanmap and "
            "pip install -e ../leanmap[cpu]"
        ) from exc


def load_roots_from_pdb(
    path: Path,
    cache: Path | None = None,
    *,
    similarity: bool = False,
) -> tuple[np.ndarray, dict]:
    """Load precomputed r0..r5 or s0..s5 from DuckDB (optionally cache to .npz)."""
    cache = cache or (DATA / ("pdb_similarity_roots.npz" if similarity else "pdb_roots.npz"))
    if cache.is_file():
        data = np.load(cache)
        key = "S" if similarity and "S" in data else "X"
        meta = {"source": str(cache), "n": len(data[key]), "similarity": similarity}
        if "volume" in data:
            meta["volume"] = data["volume"]
        if "sg_number" in data:
            meta["sg_number"] = data["sg_number"]
        return data[key].astype(np.float32, copy=False), meta

    from agentsg.cell.celldb import CellDatabase

    prefix = "s" if similarity else "r"
    cols = ",".join(f"{prefix}{i}" for i in range(6))
    db = CellDatabase(str(path))
    rows = db.sql(
        f"SELECT {cols}, volume, sg_number FROM cells "
        f"WHERE {prefix}0 IS NOT NULL ORDER BY pdb_id"
    )
    db.close()
    if not rows:
        raise SystemExit(
            f"{path} missing {prefix}0..{prefix}5 — run backfill on DuckDB first"
        )
    X = np.array([r[:6] for r in rows], dtype=np.float32)
    volume = np.array([r[6] for r in rows], dtype=np.float64)
    sg_number = np.array([r[7] for r in rows], dtype=np.int32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(X=X, volume=volume, sg_number=sg_number)
    if similarity:
        payload["S"] = X
    np.savez_compressed(cache, **payload)
    print(f"cached {len(X)} roots -> {cache}")
    return X, {
        "source": str(path),
        "n": len(X),
        "volume": volume,
        "sg_number": sg_number,
        "similarity": similarity,
    }


def load_roots_from_stream(path: Path) -> tuple[np.ndarray, dict]:
    from agentsg.cell.crystfel_stream import parse_stream
    from agentsg.cell.rootform import root_invariant

    cells, c_vals, astar_z = [], [], []
    for rec in parse_stream(str(path), with_orientation=True):
        cells.append(rec["cell"])
        c_vals.append(rec["cell"][2])
        cstar = np.array(rec["cstar"], dtype=np.float64)
        cstar /= max(np.linalg.norm(cstar), 1e-12)
        astar_z.append(float(cstar[2]))

    X = np.array([root_invariant(c) for c in cells], dtype=np.float32)
    meta = {
        "source": str(path),
        "n": len(cells),
        "c": np.array(c_vals, dtype=np.float64),
        "cstar_z": np.array(astar_z, dtype=np.float64),
    }
    return X, meta


def load_roots(path: Path) -> tuple[np.ndarray, dict]:
    if path.suffix == ".npy":
        X = np.load(path).astype(np.float32, copy=False)
        return X, {"source": str(path), "n": len(X)}
    if path.suffix == ".npz":
        data = np.load(path)
        X = data["X"].astype(np.float32, copy=False)
        meta = {"source": str(path), "n": len(X)}
        for k in ("c", "cstar_z", "labels"):
            if k in data:
                meta[k] = data[k]
        return X, meta
    return load_roots_from_stream(path)


def save_scatter(Z, color, path: Path, *, title: str, label: str = ""):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=color, s=4, cmap="viridis", linewidths=0, alpha=0.85)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", type=Path, default=None,
                    help="DuckDB with precomputed r0..r5 (default: data/pdb_cells.duckdb)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="cache extracted roots as .npz (default: analysis/data/pdb_roots.npz)")
    ap.add_argument("--stream", type=Path, default=None,
                    help="CrystFEL .stream or .npy/.npz root matrix")
    ap.add_argument("--roots", type=Path, default=None,
                    help="precomputed (N,6) root matrix (.npy or .npz)")
    ap.add_argument("-o", "--out-dir", type=Path, default=OUT)
    ap.add_argument("--model", type=Path, default=None, help="save leanmap artefact (.pt)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lambda-geo", type=float, default=0.5,
                    help="geodesic MDS backbone (raise for smooth 1D manifolds)")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="ε-net radius in Å (root L2 = Å); default 1.0 for full PDB")
    ap.add_argument("--similarity", action="store_true",
                    help="use DuckDB s0..s5 similarity invariants instead of r0..r5")
    ap.add_argument("--subsample", type=int, default=0,
                    help="optional random subsample for quick tests")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip scatter plots (useful at full PDB scale)")
    args = ap.parse_args()

    _ensure_leanmap()
    import torch
    from leanmap import PLANEConfig, fit

    if args.roots is not None:
        src = args.roots
        if not src.is_file():
            raise SystemExit(f"input not found: {src}")
        X, meta = load_roots(src)
    elif args.pdb is not None or (args.stream is None and args.roots is None):
        pdb_path = args.pdb or DEFAULT_PDB
        if not pdb_path.is_file():
            raise SystemExit(f"PDB database not found: {pdb_path}")
        X, meta = load_roots_from_pdb(pdb_path, cache=args.cache, similarity=args.similarity)
        src = pdb_path
    else:
        src = args.stream
        if not src.is_file():
            raise SystemExit(f"input not found: {src}")
        X, meta = load_roots(src)
    if args.subsample and args.subsample < len(X):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X), size=args.subsample, replace=False)
        X = X[idx]
        for k in ("c", "cstar_z"):
            if k in meta:
                meta[k] = meta[k][idx]

    n = len(X)
    print(f"loaded {n} points, dim={X.shape[1]} from {meta.get('source', src)}")

    cfg = PLANEConfig.for_scale(n)
    cfg.seed = int(args.seed)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.device is not None:
        cfg.device = args.device
    if args.epsilon is not None:
        cfg.epsilon = float(args.epsilon)
    cfg.knn_mode = str(args.knn_mode)
    cfg.lambda_geo = float(args.lambda_geo)
    cfg.geo_ramp = (0.2, 0.45)

    print(f"leanmap: epochs={cfg.epochs} n_landmarks={cfg.n_landmarks} "
          f"lambda_geo={cfg.lambda_geo} epsilon={cfg.epsilon} knn_mode={cfg.knn_mode}")

    result = fit(X, dist_fn="l2", config=cfg)
    with torch.no_grad():
        Z, score = result.embed(X)
    Z = Z.detach().cpu().numpy()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "pdb_roots" if args.pdb is not None or args.roots is None else src.stem.replace(".", "_")
    np.save(out_dir / f"embedding_{tag}.npy", Z)
    np.save(out_dir / f"embedding_{tag}_score.npy", score.detach().cpu().numpy())

    gs = getattr(result, "graph_stats", None) or {}
    stats = {
        "n": n,
        "d_in": int(X.shape[1]),
        "d_out": int(Z.shape[1]),
        "epsilon": float(gs.get("epsilon", cfg.epsilon or 0)),
        "delta": float(gs.get("delta", gs.get("epsilon", 0))),
        "R_reps": int(gs.get("R", n)),
        "epochs": cfg.epochs,
        "lambda_geo": cfg.lambda_geo,
        "source": meta.get("source", str(src)),
    }
    with (out_dir / f"embedding_{tag}_stats.json").open("w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"graph: R={stats['R_reps']} epsilon={stats['epsilon']:.6g} "
          f"delta={stats['delta']:.6g}")

    if not args.no_plot:
        color = meta.get("volume")
        if color is not None and len(color) == len(Z):
            save_scatter(
                Z, np.log10(color), out_dir / f"embedding_{tag}_by_volume.png",
                title=f"leanmap ε-net — {n:,} PDB cells (colour: log₁₀ V)",
                label="log₁₀ volume",
            )
        save_scatter(
            Z, score.detach().cpu().numpy(), out_dir / f"embedding_{tag}_by_cover.png",
            title=f"leanmap ε-net — {n:,} cells (colour: landmark cover)",
            label="cover score",
        )
    print(f"saved embedding {Z.shape} -> {out_dir / f'embedding_{tag}.npy'}")

    if args.model is not None:
        result.save(str(args.model))
        print(f"saved model {args.model}")


if __name__ == "__main__":
    main()
