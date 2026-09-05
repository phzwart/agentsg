"""
agentsg: exact-rational space-group algebra for all 230 groups.

Features:
* Exact-rational linear algebra (Fraction-based, no fixed denominators).
* Hall symbol parser and generator table for all 230 standard space groups.
* Group closure, point groups, centring vectors, and systematic absences.
* Change-of-basis operations and arbitrary Hermann-Mauguin / Hall settings.
* Wyckoff positions, site-symmetry stabilisers, and orbits derived from first principles.
* Reflection conditions derived as sublattices in projection.
* Harker sections and self-Patterson vectors.
* Direct and reciprocal asymmetric units (bricks and Dirichlet Voronoi domains).
* Semi-invariants and continuous / discrete origin degrees of freedom.
* Metric lattice symmetry, Le Page delta, and tolerance metric automorphisms.
* Zero runtime dependencies.
"""
from .linalg import Matrix3, Vector3, IDENTITY3, ZERO3, frac_mod1
from .symmetry_op import SymmetryOp
from .group import (
    close_group, point_group, centering_translations, is_systematically_absent,
    transform_hkl, phase_shift, is_centrosymmetric, is_reflection_centric,
    phase_restriction, PhaseRestriction,
)
from .change_of_basis import ChangeOfBasis
from .hall import parse_hall, ops_from_hall, LATTICE_CENTERING
from .space_groups import SpaceGroup, space_group, SPACE_GROUPS
from .setting import (
    SpaceGroupSetting, parse_setting, parse_cob, format_cob,
)
from .lattice_symmetry import (
    lattice_symmetry, LatticeSymmetry, le_page_delta,
    kurlin_distance_to_two_fold, evaluate_two_folds, TwoFoldScore,
    tolerance_metric_symmetry, LEBEDEV_MATRICES, TWO_FOLD_MATRICES,
)
from .identify import IdentifyResult, identify_space_group, hall_from_ops
from .semi_invariants import (
    SemiInvariant, semi_invariants, is_semi_invariant,
    floating_origin_basis, pin_floating_origin, is_allowed_origin,
    discrete_allowed_origins,
)
from .reflections import (
    reflection_conditions, EquivalentHKL,
    equivalent_reflections, are_equivalent_reflections,
    epsilon_factor, reflection_multiplicity, laue_multiplicity,
)
from .asu import (
    ReciprocalAsu, DirectAsuBrick, DirichletAsu, OptimizedAsu,
    build_dirichlet_asu, optimize_asu, laue_class,
)
from .harker import (
    HarkerConstraint, HarkerLocus,
    harker_sections, harker_vector, site_from_harker,
)
from .wyckoff import (
    site_symmetry_ops, site_symmetry_point_group, site_symmetry_order,
    orbit, multiplicity, general_position_multiplicity, fixed_locus,
)

__all__ = [
    "Matrix3", "Vector3", "IDENTITY3", "ZERO3", "frac_mod1",
    "SymmetryOp",
    "close_group", "point_group", "centering_translations", "is_systematically_absent",
    "transform_hkl", "phase_shift", "is_centrosymmetric", "is_reflection_centric",
    "phase_restriction", "PhaseRestriction",
    "ChangeOfBasis",
    "parse_hall", "ops_from_hall", "LATTICE_CENTERING",
    "SpaceGroup", "space_group", "SPACE_GROUPS",
    "SpaceGroupSetting", "parse_setting", "parse_cob", "format_cob",
    "lattice_symmetry", "LatticeSymmetry", "le_page_delta",
    "kurlin_distance_to_two_fold", "evaluate_two_folds", "TwoFoldScore",
    "tolerance_metric_symmetry", "LEBEDEV_MATRICES", "TWO_FOLD_MATRICES",
    "IdentifyResult", "identify_space_group", "hall_from_ops",
    "SemiInvariant", "semi_invariants", "is_semi_invariant",
    "floating_origin_basis", "pin_floating_origin", "is_allowed_origin",
    "discrete_allowed_origins",
    "reflection_conditions", "EquivalentHKL",
    "equivalent_reflections", "are_equivalent_reflections",
    "epsilon_factor", "reflection_multiplicity", "laue_multiplicity",
    "ReciprocalAsu", "DirectAsuBrick", "DirichletAsu", "OptimizedAsu",
    "build_dirichlet_asu", "optimize_asu", "laue_class",
    "HarkerConstraint", "HarkerLocus",
    "harker_sections", "harker_vector", "site_from_harker",
    "site_symmetry_ops", "site_symmetry_point_group", "site_symmetry_order",
    "orbit", "multiplicity", "general_position_multiplicity", "fixed_locus",
]
