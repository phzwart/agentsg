"""Named numeric gates are the defaults the implementations actually use."""
import inspect

from agentsg.cell.canonical import reindex, reindexing_via_canonical, superbase_variants
from agentsg.cell.compare import compare_cells
from agentsg.cell.selling_closure import selling_superbase_closure, _zero_tol
from agentsg.cell.selling_cob import reference_orbit, match_operators
from agentsg.tolerances import (
    BOUNDARY_REL,
    COB_ANGLE_SIGMA_DEG,
    COB_ANGLE_TOL_DEG,
    COB_LENGTH_TOL_PCT,
    COMPARE_ANGLE_TOL_DEG,
    COMPARE_LENGTH_TOL_PCT,
    CONORM_TOL_REL,
    REINDEX_BOUNDARY_REL,
    VERIFY_REL,
    ZERO_NOISE_MULT,
)
from agentsg.lattice_symmetry import lattice_symmetry


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


def test_named_gates_are_the_call_defaults():
    assert _default(lattice_symmetry, "max_delta") == 3.0
    assert _default(compare_cells, "length_tol_pct") == COMPARE_LENGTH_TOL_PCT
    assert _default(compare_cells, "angle_tol_deg") == COMPARE_ANGLE_TOL_DEG
    assert _default(reference_orbit, "angle_sigma") == COB_ANGLE_SIGMA_DEG
    assert _default(reference_orbit, "boundary_rel") == BOUNDARY_REL
    assert _default(match_operators, "length_tol_pct") == COB_LENGTH_TOL_PCT
    assert _default(match_operators, "angle_tol_deg") == COB_ANGLE_TOL_DEG
    assert _default(match_operators, "verify_rel") == VERIFY_REL
    assert _default(selling_superbase_closure, "tol_rel") == CONORM_TOL_REL
    assert _default(_zero_tol, "noise_mult") == ZERO_NOISE_MULT
    assert _default(superbase_variants, "boundary_rel") == BOUNDARY_REL
    assert _default(reindexing_via_canonical, "verify_rel") == VERIFY_REL
    assert _default(reindex, "boundary_rel") == REINDEX_BOUNDARY_REL
