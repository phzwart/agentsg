"""
Phase 2: unit-cell math -- real-valued (floating point).

This subpackage is numeric and, with ONE exception, does not import from
agentsg's exact symmetry modules. The exception is ``constraints.py`` -- the
single sanctioned crossing of the exact/numeric boundary (metric <-> point-group
invariance, W^T G W = G). See docs/DESIGN.md, "The one interface point".

  * metric.py       UnitCell: metric tensor, volume, reciprocal cell,
                    orthogonalisation, d-spacings, two-theta.
  * reduction.py    Niggli reduction (stabilised Grosse-Kunstleve/Sauter/Adams
                    2004 algorithm) with the integer change-of-basis matrix.
  * constraints.py  the exact<->numeric bridge: W^T G W = G, metric
                    symmetrisation, and derived crystal-system free parameters.
  * sublattice.py   Hermite-normal-form enumeration of index-d sublattices.
  * compare.py      compare_cells: the "exploring metric symmetry" lego/target
                    algorithm relating two unit cells by an exact integer transform.
  * ambiguity.py    reindexing / indexing-ambiguity operators for serial
                    crystallography (coset of Laue group in the tolerance lattice
                    group, cached per dataset). A re-implementation of the
                    dials.cosym / Brehm-Diederichs method; see its module
                    docstring for attribution and the merohedral-vs-pseudo scope.
  * g6.py           G6 / S6 lattice embeddings and boundary-aware distances
                    (Andrews-Bernstein): lattices as points on a continuous
                    manifold, distances robust to cell choice and the Niggli
                    reduction-flip, and symmetry as a continuous distance-to-
                    subspace rather than a binary test.
  * rootform.py     Kurlin (2022) root invariant: a COMPLETE, CONTINUOUS
                    isometry invariant -- one ordered vector per lattice from the
                    obtuse superbase, no orbit minimisation, continuous across
                    the reduction-flip boundary. The preferred manifold
                    coordinate; g6.py is kept for comparison / compatibility.
  * neartree.py     Exact metric nearest-neighbour index (Andrews 2001); the
                    `lattice_index` helper keys a NearTree on the root invariant
                    for fast, exact lattice search over a database.
"""
from .metric import UnitCell
from .reduction import niggli_reduce
from .constraints import (
    metric_is_invariant, symmetrize_metric, free_metric_parameters,
)
from .sublattice import (
    generate_sublattices, sublattice_count, is_hermite_normal_form,
    apply_to_cell,
)
from .compare import compare_cells, CellMatch
from .ambiguity import (
    reindexing_ambiguity_operators, ambiguity_index, apply_to_hkl_batch,
    ReindexingReference, AmbiguityResolution,
    surface_geometric_operators, GeometricOperator,
)
from .g6 import (
    g6, s6, g6_distance, s6_distance,
    distance_to_symmetry, kurlin_distance_to_symmetry,
    symmetry_deficiency_spectrum, kurlin_deficiency_spectrum,
)
from .rootform import (
    delaunay_superbase, conorms, root_products, root_invariant, root_distance,
    similarity_invariant, similarity_distance, root_volume_decomposition,
    root_cutoff_for_edge_tolerance,
    root_distance_to_volume_ratio, volume_ratio_to_root_distance,
    symmetry_cutoff,
)
from .neartree import NearTree, build_neartree, lattice_index
from .reindex import reindexing_operators, reindexing_operator, twin_laws
from .canonical import (
    canonical_superbase, superbase_variants, reindexing_via_canonical,
    reindexing_operator_via_canonical, best_reindex_with_residual,
    calibrate_verify_tol,
)
from .crystfel_stream import parse_stream, read_cells, stream_summary
from .primitive import primitive_cell, primitive_transform, lattice_letter
from .manifold import (
    deformation_graph, symmetry_junctions, farthest_point_landmarks,
    select_landmarks, DeformationManifold,
)

__all__ = [
    "UnitCell", "niggli_reduce",
    "metric_is_invariant", "symmetrize_metric", "free_metric_parameters",
    "generate_sublattices", "sublattice_count", "is_hermite_normal_form",
    "apply_to_cell",
    "compare_cells", "CellMatch",
    "reindexing_ambiguity_operators", "ambiguity_index", "apply_to_hkl_batch",
    "ReindexingReference", "AmbiguityResolution",
    "surface_geometric_operators", "GeometricOperator",
    "g6", "s6", "g6_distance", "s6_distance",
    "distance_to_symmetry", "kurlin_distance_to_symmetry",
    "symmetry_deficiency_spectrum", "kurlin_deficiency_spectrum",
    "delaunay_superbase", "conorms", "root_products", "root_invariant",
    "root_distance", "similarity_invariant", "similarity_distance",
    "root_volume_decomposition", "root_cutoff_for_edge_tolerance",
    "root_distance_to_volume_ratio", "volume_ratio_to_root_distance",
    "symmetry_cutoff",
    "NearTree", "build_neartree", "lattice_index",
    "reindexing_operators", "reindexing_operator", "twin_laws",
    "canonical_superbase", "superbase_variants", "reindexing_via_canonical",
    "reindexing_operator_via_canonical", "best_reindex_with_residual",
    "calibrate_verify_tol",
    "parse_stream", "read_cells", "stream_summary",
    "primitive_cell", "primitive_transform", "lattice_letter",
    "deformation_graph", "symmetry_junctions", "farthest_point_landmarks",
    "select_landmarks", "DeformationManifold",
    # DuckDB-backed database + PDB builder (lazy; need agentsg[db]):
    "CellDatabase", "RootIndex", "list_pdb_ids", "fetch_pdb_cells",
    "build_pdb_database",
]

# Lazy exports for the optional DuckDB-backed layer. Importing agentsg.cell must
# not require DuckDB; these resolve only when actually accessed.
_LAZY = {
    "CellDatabase": ("celldb", "CellDatabase"),
    "RootIndex": ("celldb", "RootIndex"),
    "list_pdb_ids": ("celldb", "list_pdb_ids"),
    "fetch_pdb_cells": ("celldb", "fetch_pdb_cells"),
    "build_pdb_database": ("pdb_app", "build_pdb_database"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f".{target[0]}", __name__)
    return getattr(mod, target[1])
