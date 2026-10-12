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


def centring_ids(sg_hm: np.ndarray) -> np.ndarray:
    """P, C (A/B/C merged), I, F, R."""
    letter_to_id = {"P": 0, "A": 1, "B": 1, "C": 1, "I": 2, "F": 3, "R": 4, "H": 4}
    return np.fromiter(
        (letter_to_id.get(_letter(s), 0) for s in sg_hm),
        dtype=np.int32,
        count=len(sg_hm),
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lambda-geo", type=float, default=None,
                    help="geodesic weight; omit to keep the large-N preset (0.5)")
    ap.add_argument("--subsample", type=int, default=0,
                    help="random subset size for a short trial (0 = all rows)")
    args = ap.parse_args()

    _ensure_leanmap()
    import torch
    from leanmap import PLANEConfig, fit

    S, meta = load_features(args.pdb, args.cache)
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
    cfg.seed = int(args.seed)
    cfg.device = args.device
    cfg.epsilon = float(args.epsilon)
    if args.lambda_geo is not None:
        cfg.lambda_geo = float(args.lambda_geo)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "graph.pt"
    print(
        f"leanmap: epochs={cfg.epochs} n_landmarks={cfg.n_landmarks} "
        f"epsilon={cfg.epsilon} lambda_geo={cfg.lambda_geo} "
        f"device={cfg.device} knn_mode={cfg.knn_mode}"
    )

    result = fit(
        S,
        dist_fn="l2",
        config=cfg,
        graph_path=str(graph_path),
        rebuild_graph=not graph_path.is_file(),
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
        "lambda_geo": cfg.lambda_geo,
        "n_landmarks": cfg.n_landmarks,
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
