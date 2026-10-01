"""Named numeric gates for lattice matching, Selling classification, and Le Page symmetry.

Function defaults and the HTTP and MCP parameters use these names. The sentence
after each assignment is the quote the concept graph stores for that constant.
"""

REL_EPS = 1e-9
"Relative floor for Niggli, Delaunay, and zero-conorm tests."

CONORM_TOL_REL = REL_EPS
"Relative floor on a Selling conorm before an angle sigma widens zero detection."

ZERO_NOISE_MULT = 3.0
"How many angle-sigma widths make the absolute zero-conorm tolerance."

LE_PAGE_MAX_DELTA_DEG = 3.0
"Le Page angular cutoff in degrees: a two-fold is kept when its delta is at most this."

LE_PAGE_LENGTH_TOL_PCT = 2.0
"Percent length change allowed when a Le Page two-fold is checked against the metric."

METRIC_LENGTH_TOL_PCT = 2.0
"Percent edge tolerance for the metric-automorphism group, twin laws, and geometric reindexing."

METRIC_ANGLE_TOL_DEG = 2.0
"Angle tolerance in degrees for the metric-automorphism group, twin laws, and geometric reindexing."

METRIC_INVARIANT_TOL = 1e-6
"Relative residual at which a rotation is treated as leaving the metric tensor unchanged."

METRIC_SYM_RESIDUAL = 1e-6
"Residual at or below which a geometric reindexing operator is exact metric symmetry."

COMPARE_LENGTH_TOL_PCT = 3.0
"Percent edge tolerance when comparing two cells, including sublattice matches."

COMPARE_ANGLE_TOL_DEG = 5.0
"Angle tolerance in degrees when comparing two cells, including sublattice matches."

VOLUME_FRAC = 0.05
"Fractional volume agreement required for a sublattice index or a supercell match."

COB_LENGTH_TOL_PCT = 0.75
"Percent edge tolerance for a Selling-closure member to match a stored reduced cell."

COB_ANGLE_TOL_DEG = 0.5
"Angle tolerance in degrees for a Selling-closure member to match a stored reduced cell."

COB_ANGLE_SIGMA_DEG = 0.05
"Degrees. Zero-conorm width of the query Selling closure, inside the match angle tolerance."

BOUNDARY_REL = 1e-3
"Relative width of a near-zero Selling conorm that is still treated as a boundary flip."

SUPERBASE_MAX_VARIANTS = 64
"Cap on the near-boundary superbase expander."

REINDEX_BOUNDARY_REL = 6e-2
"Relative conorm width used when reindex searches the Selling closure of a deformed cell."

REINDEX_BAND_REL = 1e-3
"Relative residual band that keeps near-best reindexing operators in the coset."

VERIFY_REL = 1e-6
"Relative metric residual at which a change-of-basis operator is accepted."

NIGGLI_COB_TOL_REL = 1e-6
"Relative tolerance when a Niggli change of basis is checked against the reduced metric."

ROOT_STABILIZE_KAPPA = 2.0
"Exponent of the optional stabilised root product. The archive search key does not use it."

SYMMETRY_CUTOFF_Z = 11.0
"Scale of the edge-noise null at the 95th percentile, used to turn a fractional noise into a root cutoff."

ROOT_SNAP_REL = 1e-6
"Relative size below which a sorted-root component is written as zero."

ROOT_SNAP_DECIMALS = 10
"Decimal places kept in a sorted-root key after the noise snap."

MANIFOLD_HOP_VERIFY_REL = 1e-3
"Relative residual allowed for one deformation-manifold hop onto a landmark."

MANIFOLD_PATH_VERIFY_REL = 1e-2
"Relative residual allowed for one hop when a manifold path is composed."
