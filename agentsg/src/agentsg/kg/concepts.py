"""Concept catalogue for the agentsg knowledge graph.

Each concept: id, label, kind, definition (paraphrase, how=derived), anchors
(module, symbol) -> verbatim docstring quotes are extracted from the source by
build.py, relations to other concepts, and external reference keys:
  iucr  -> page title in the IUCr Online Dictionary of Crystallography (or None)
  wiki  -> English Wikipedia article title (or None)
  refs  -> literature references named in the code (free text)

kind: crystallography | mathematics | algorithm | data-structure
relations: IS_A, PART_OF, USES, DUAL_OF, SPECIALIZES, EQUIVALENT_TO,
           RELATED_TO, DEFINED_BY, INSTANCE_OF, CONTRASTS_WITH
"""

C = []


def c(id, label, kind, definition, anchors, rels=(), iucr=None, wiki=None, refs=(), aliases=()):
    C.append(dict(id=id, label=label, kind=kind, definition=definition,
                  anchors=list(anchors), rels=list(rels), iucr=iucr, wiki=wiki,
                  refs=list(refs), aliases=list(aliases)))


# ---------------------------------------------------------------- symmetry core
c("space_group", "Space group", "crystallography",
  "The group of all symmetry operations (rotation part W with translation w) of a three-dimensional crystal pattern; agentsg builds each of the 230 from Hall-symbol generators by closure.",
  [("agentsg/space_groups.py", ""), ("agentsg/group.py", "close_group"), ("agentsg/__init__.py", "")],
  [("USES", "symmetry_operation"), ("USES", "group_closure"), ("DEFINED_BY", "hall_symbol"), ("PART_OF", "group_theory")],
  iucr="Space group", wiki="Space group", aliases=["230 space groups"])
c("symmetry_operation", "Symmetry operation (Seitz operator)", "crystallography",
  "An affine map x' = W x + w with integer rotation part W (det ±1) and rational translation w, composed and inverted exactly; written as an ITA xyz triplet.",
  [("agentsg/symmetry_op.py", ""), ("agentsg/symmetry_op.py", "SymmetryOp")],
  [("IS_A", "affine_transformation"), ("USES", "exact_rational_arithmetic"), ("PART_OF", "space_group")],
  iucr="Symmetry operation", wiki="Space group", aliases=["Seitz symbol", "(W|w)", "xyz triplet"])
c("point_group", "Point group (rotation parts)", "crystallography",
  "The set of distinct rotation parts W of a space group; agentsg derives it from the operator list rather than a table.",
  [("agentsg/group.py", "point_group")],
  [("PART_OF", "space_group"), ("IS_A", "group_theory"), ("RELATED_TO", "crystallographic_restriction")],
  iucr="Point group", wiki="Crystallographic point group")
c("laue_class", "Laue class / Laue group", "crystallography",
  "The point group obtained by adding inversion; used because Friedel's law makes diffraction centrosymmetric, e.g. for reindexing cosets and reflection multiplicities.",
  [("agentsg/asu.py", "laue_class"), ("agentsg/reflections.py", "laue_multiplicity"), ("agentsg/cell/ambiguity.py", "_laue_matrices")],
  [("SPECIALIZES", "point_group"), ("USES", "friedel_law"), ("USES", "inversion_centre")],
  iucr="Laue class", wiki="Laue group")
c("holohedry", "Holohedry (lattice point group)", "crystallography",
  "The full point group of a lattice: all rotations leaving its metric tensor invariant; determined numerically from the 81 Lebedev two-folds.",
  [("agentsg/lattice_symmetry.py", ""), ("agentsg/lattice_symmetry.py", "lattice_symmetry")],
  [("SPECIALIZES", "point_group"), ("USES", "metric_tensor"), ("USES", "lattice_symmetry_determination")],
  iucr="Holohedry", wiki="Holohedry")
c("centrosymmetry", "Centrosymmetry / inversion centre", "crystallography",
  "Presence of the operation −I (x → −x); tested on the point group.",
  [("agentsg/group.py", "is_centrosymmetric"), ("agentsg/cell/diagrams.py", "draw_inversion")],
  [("IS_A", "symmetry_element")],
  iucr="Symmetry element", wiki="Centrosymmetry", aliases=["inversion centre"])
c("inversion_centre", "Inversion (−1) operation", "crystallography",
  "The symmetry operation W = −I; Laue groups are point groups closed under it.",
  [("agentsg/group.py", "is_centrosymmetric"), ("agentsg/cell/selling_group.py", "inversion_cob")],
  [("IS_A", "symmetry_operation"), ("EQUIVALENT_TO", "centrosymmetry")],
  wiki="Point reflection")
c("group_closure", "Group closure by composition", "algorithm",
  "Generate the full operation set from generators by repeated composition modulo lattice translations; terminates because a space-group order is bounded by 192 in the conventional cell.",
  [("agentsg/group.py", ""), ("agentsg/group.py", "close_group"), ("agentsg/cell/selling_group.py", "expand_group")],
  [("PART_OF", "group_theory"), ("USES", "symmetry_operation"), ("USES", "generators_of_a_group")],
  wiki="Generating set of a group")
c("generators_of_a_group", "Generators of a group", "mathematics",
  "A small set of elements whose products generate the whole group; Hall symbols encode space-group generators.",
  [("agentsg/generators.py", ""), ("agentsg/hall.py", "")],
  [("PART_OF", "group_theory"), ("RELATED_TO", "hall_symbol")],
  wiki="Generating set of a group")
c("hall_symbol", "Hall space-group symbol", "crystallography",
  "Compact notation (Hall 1981) encoding the lattice centring and generator operations of a space group, including an optional origin shift in twelfths; parsed into generators.",
  [("agentsg/hall.py", ""), ("agentsg/hall.py", "parse_hall")],
  [("RELATED_TO", "hermann_mauguin_symbol"), ("USES", "lattice_centring"), ("USES", "origin_shift")],
  wiki="Hall notation", refs=["Hall, S. R. (1981). Acta Cryst. A37, 517", "ITA Vol. B 1.4"],
  aliases=["Hall notation"])
c("hermann_mauguin_symbol", "Hermann–Mauguin symbol", "crystallography",
  "The international (ITA) symbol of a space group or point group; agentsg resolves HM, Hall and number, and prints extended HM symbols with a parenthesised change of basis.",
  [("agentsg/space_groups.py", "space_group"), ("agentsg/ita_settings.py", "display_hm"), ("agentsg/cell/diagrams.py", "_display_hm")],
  [("RELATED_TO", "hall_symbol"), ("RELATED_TO", "ita_setting")],
  iucr="Hermann-Mauguin symbols", wiki="Hermann–Mauguin notation")
c("ita_setting", "ITA settings (origin choice, cell choice, unique axis)", "crystallography",
  "The alternative descriptions of one space-group type tabulated in International Tables Vol. A (origin choices 1/2, hexagonal/rhombohedral axes, unique axes), each with its own Hall symbol.",
  [("agentsg/ita_settings.py", ""), ("agentsg/ita_settings.py", "lookup_setting")],
  [("USES", "change_of_basis"), ("USES", "origin_shift"), ("RELATED_TO", "conventional_cell")],
  iucr="Conventional cell", wiki="Space group", aliases=["origin choice 2", "rhombohedral axes"])
c("extended_setting_notation", "Extended setting notation (symbol + change of basis)", "crystallography",
  "A base space-group symbol followed by a parenthesised change of basis, e.g. 'C 2y (x+y,z,x-y)'; when det P ≠ 1 the transform introduces centring translations.",
  [("agentsg/setting.py", ""), ("agentsg/setting.py", "parse_cob"), ("agentsg/setting.py", "SpaceGroupSetting")],
  [("USES", "change_of_basis"), ("USES", "lattice_centring"), ("USES", "group_closure")],
  iucr="Hermann-Mauguin symbols",
  refs=["Zwart, Grosse-Kunstleve & Adams, Exploring Metric Symmetry, IUCr Comp. Comm. Newsletter 7 (2006)"])
c("change_of_basis", "Change of basis / coordinate transformation (P, p)", "crystallography",
  "A transformation of the crystallographic basis and origin: (a',b',c') = (a,b,c)P, x' = P⁻¹(x − p), W' = P⁻¹WP, and Miller indices transform covariantly h' = hP.",
  [("agentsg/change_of_basis.py", ""), ("agentsg/change_of_basis.py", "ChangeOfBasis")],
  [("USES", "matrix_inverse"), ("USES", "unimodular_matrix"), ("RELATED_TO", "reindexing"), ("RELATED_TO", "covariance_contravariance")],
  iucr="Crystallographic basis", wiki="Change of basis")
c("covariance_contravariance", "Covariant / contravariant transformation", "mathematics",
  "Fractional coordinates transform with P⁻¹ (contravariantly) while Miller indices, dual to the basis, transform with P so that h·x stays invariant.",
  [("agentsg/change_of_basis.py", ""), ("agentsg/group.py", "transform_hkl")],
  [("RELATED_TO", "dual_basis"), ("PART_OF", "change_of_basis")],
  wiki="Covariance and contravariance of vectors")
c("origin_shift", "Origin shift", "crystallography",
  "Translation p of the coordinate origin; conjugates each operator to (W, w + (I − W)p) and multiplies structure factors by the phase exp(2πi h·p) without changing h.",
  [("agentsg/hall.py", ""), ("agentsg/identify.py", "_conjugate_by_origin"), ("agentsg/identify.py", "_origin_shift_to_standard")],
  [("PART_OF", "change_of_basis"), ("RELATED_TO", "allowed_origins"), ("USES", "conjugation")],
  iucr="Conventional cell", wiki="Space group")
c("conjugation", "Conjugation of a group element", "mathematics",
  "Mapping g ↦ h g h⁻¹; applied with a translation h = (I, p) it re-expresses a space group in a shifted origin, and with a basis change it re-expresses W as P⁻¹WP.",
  [("agentsg/identify.py", "_conjugate_by_origin"), ("agentsg/hall.py", "")],
  [("PART_OF", "group_theory"), ("USES", "matrix_inverse")],
  iucr="Conjugacy class", wiki="Conjugacy class")
c("lattice_centring", "Lattice centring (P, A, B, C, I, F, R)", "crystallography",
  "Additional pure translations of a conventional cell; derived in agentsg as the operations with W = I and used for integral reflection conditions and primitive-cell reduction.",
  [("agentsg/group.py", "centering_translations"), ("agentsg/hall.py", ""), ("agentsg/cell/primitive.py", "")],
  [("PART_OF", "bravais_lattice"), ("RELATED_TO", "integral_reflection_conditions"), ("RELATED_TO", "primitive_cell")],
  iucr="Centred lattice", wiki="Bravais lattice")
c("bravais_lattice", "Bravais lattice", "crystallography",
  "One of the 14 translation-lattice types in three dimensions; distinguished by holohedry and centring.",
  [("agentsg/cell/primitive.py", ""), ("agentsg/cell/g6.py", "")],
  [("USES", "lattice_centring"), ("USES", "holohedry"), ("IS_A", "lattice")],
  iucr="Bravais lattice", wiki="Bravais lattice")
c("lattice", "Lattice (translation lattice)", "mathematics",
  "The set of all integer combinations of three basis vectors; the translation subgroup of a space group, including centring vectors.",
  [("agentsg/cell/symmetry_elements.py", "translation_lattice"), ("agentsg/semi_invariants.py", "_primitive_basis")],
  [("IS_A", "group_theory"), ("RELATED_TO", "integer_lattice")],
  iucr="Lattice", wiki="Lattice (group)")
c("integer_lattice", "Integer lattice Zⁿ and sublattices", "mathematics",
  "Z³ as the lattice of fractional coordinates; sublattices, saturated sublattices and their bases are handled with integer (unimodular) column and row reduction.",
  [("agentsg/reflection_lattice.py", "_int_kernel"), ("agentsg/subgroups.py", "_kernel_basis"), ("agentsg/cell/sublattice.py", "")],
  [("USES", "hermite_normal_form"), ("USES", "unimodular_matrix"), ("RELATED_TO", "sublattice")],
  wiki="Integer lattice")
c("crystal_system", "Crystal system", "crystallography",
  "Classification of point groups (triclinic … cubic); in agentsg the metric restrictions of each system follow from WᵀGW = G rather than a table.",
  [("agentsg/space_groups.py", ""), ("agentsg/cell/constraints.py", ""), ("agentsg/cell/constraints.py", "free_metric_parameters")],
  [("RELATED_TO", "crystal_family"), ("RELATED_TO", "metric_invariance")],
  iucr="Crystal system", wiki="Crystal system")
c("crystal_family", "Crystal family / lattice family", "crystallography",
  "The six families (triclinic, monoclinic, orthorhombic, tetragonal, hexagonal, cubic) decided from rotation orders and their fixed axes.",
  [("agentsg/reflection_lattice.py", "crystal_family")],
  [("RELATED_TO", "crystal_system"), ("USES", "point_group")],
  iucr="Crystal family", wiki="Crystal system")
c("crystallographic_restriction", "Crystallographic restriction theorem", "mathematics",
  "Rotations compatible with a lattice have orders 1, 2, 3, 4 or 6 and are integer matrices in a lattice basis; agentsg's order test only searches k ≤ 6.",
  [("agentsg/reflection_lattice.py", "_matrix_order"), ("agentsg/lattice_symmetry.py", "_lebedev_matrices")],
  [("RELATED_TO", "point_group"), ("RELATED_TO", "unimodular_matrix")],
  wiki="Crystallographic restriction theorem")

# ------------------------------------------------------------ symmetry elements
c("symmetry_element", "Symmetry element", "crystallography",
  "The geometric locus (axis, plane, point) together with the intrinsic translation of an operation; agentsg classifies (W, w) exactly, reducing w modulo the lattice.",
  [("agentsg/cell/symmetry_elements.py", ""), ("agentsg/cell/symmetry_elements.py", "classify_element")],
  [("USES", "symmetry_operation"), ("USES", "fixed_point_locus"), ("USES", "intrinsic_translation")],
  iucr="Symmetry element", wiki="Symmetry element")
c("screw_axis", "Screw axis", "crystallography",
  "A rotation combined with a translation along the axis; causes serial reflection conditions such as 00l: l = 2n for 2₁ along c.",
  [("agentsg/hall.py", ""), ("agentsg/cell/symmetry_elements.py", "classify_element"), ("agentsg/reflections.py", "")],
  [("IS_A", "symmetry_element"), ("USES", "intrinsic_translation"), ("RELATED_TO", "serial_reflection_conditions")],
  wiki="Screw axis")
c("glide_plane", "Glide plane", "crystallography",
  "A reflection combined with a translation parallel to the plane (a, b, c, n, d, e glides); causes zonal reflection conditions.",
  [("agentsg/cell/symmetry_elements.py", "_glide_symbol"), ("agentsg/reflections.py", "")],
  [("IS_A", "symmetry_element"), ("USES", "intrinsic_translation"), ("RELATED_TO", "zonal_reflection_conditions")],
  wiki="Glide plane")
c("rotoinversion", "Rotoinversion axis", "crystallography",
  "An improper operation: rotation followed by inversion (−N in Hall symbols).",
  [("agentsg/hall.py", ""), ("agentsg/cell/diagrams.py", "_draw_bar4")],
  [("IS_A", "symmetry_element"), ("USES", "inversion_centre")],
  wiki="Improper rotation")
c("mirror_plane", "Mirror plane", "crystallography",
  "A reflection with zero intrinsic translation; its fixed subspace in reciprocal space is a zone of reflections.",
  [("agentsg/cell/symmetry_elements.py", ""), ("agentsg/reflection_lattice.py", "")],
  [("IS_A", "symmetry_element"), ("RELATED_TO", "reflection_stratum")],
  wiki="Reflection symmetry")
c("intrinsic_translation", "Intrinsic (screw/glide) translation", "crystallography",
  "The component of w parallel to the fixed space of W, reduced to the shortest lattice-equivalent representative; zero for pure rotations and mirrors.",
  [("agentsg/cell/symmetry_elements.py", "_reduce_intrinsic"), ("agentsg/cell/symmetry_elements.py", "")],
  [("PART_OF", "symmetry_element"), ("USES", "fixed_subspace")],
  wiki="Screw axis")
c("fixed_point_locus", "Fixed-point locus of an operation", "mathematics",
  "The affine subspace of points x with W x + w ≡ x (mod lattice), found by solving (W − I)x = −w exactly; its dimension gives point, line or plane.",
  [("agentsg/wyckoff.py", ""), ("agentsg/wyckoff.py", "fixed_locus"), ("agentsg/cell/symmetry_elements.py", "_particular")],
  [("USES", "rational_linear_solve"), ("USES", "affine_subspace"), ("RELATED_TO", "wyckoff_position")],
  wiki="Fixed point (mathematics)")
c("fixed_subspace", "Fixed subspace / +1 eigenspace of W", "mathematics",
  "The kernel of (W − I): directions left invariant by a rotation; in reciprocal space (row action) it defines reflection strata, in direct space floating origins and axis directions.",
  [("agentsg/reflection_lattice.py", "_fixed_lattice"), ("agentsg/lattice_symmetry.py", "_two_fold_axis_direct"), ("agentsg/semi_invariants.py", "floating_origin_basis")],
  [("USES", "kernel_nullspace"), ("USES", "eigenvector"), ("RELATED_TO", "fixed_point_locus")],
  wiki="Eigenvalues and eigenvectors")
c("ita_diagrams", "ITA space-group diagrams (general position, symmetry elements)", "crystallography",
  "The International Tables plates: projected general-position orbit with heights and handedness, and the symmetry-element diagram with graphical symbols; rendered from the derived operations.",
  [("agentsg/cell/diagrams.py", ""), ("agentsg/cell/diagrams.py", "general_position_diagram"), ("agentsg/cell/diagrams.py", "symmetry_element_diagram")],
  [("USES", "symmetry_element"), ("USES", "general_position"), ("USES", "metric_invariance"), ("USES", "convex_hull")],
  wiki="International Tables for Crystallography")

# ---------------------------------------------------------- positions / orbits
c("wyckoff_position", "Wyckoff position", "crystallography",
  "A set of points with conjugate site-symmetry groups; agentsg computes multiplicity, site-symmetry order and locus dimension from the operations (Wyckoff letters are not reproduced).",
  [("agentsg/wyckoff.py", ""), ("agentsg/wyckoff.py", "multiplicity")],
  [("USES", "site_symmetry"), ("USES", "crystallographic_orbit"), ("USES", "fixed_point_locus")],
  iucr="Wyckoff position", wiki="Wyckoff positions")
c("site_symmetry", "Site symmetry (stabilizer of a point)", "crystallography",
  "The subgroup of operations fixing a point modulo lattice translations; its order divides the group order.",
  [("agentsg/wyckoff.py", "site_symmetry_ops"), ("agentsg/wyckoff.py", "site_symmetry_point_group")],
  [("IS_A", "stabilizer"), ("PART_OF", "wyckoff_position")],
  iucr="Site symmetry", wiki="Site symmetry")
c("crystallographic_orbit", "Crystallographic orbit", "crystallography",
  "The set of images of a point under the space group, reduced into the unit cell; its size is |G|/|S(x)| (orbit–stabilizer).",
  [("agentsg/wyckoff.py", "orbit"), ("agentsg/wyckoff.py", "multiplicity")],
  [("IS_A", "orbit"), ("USES", "orbit_stabilizer_theorem")],
  iucr="Crystallographic orbit", wiki="Wyckoff positions")
c("general_position", "General position", "crystallography",
  "A point with trivial site symmetry; its multiplicity equals the order of the space group in the conventional cell.",
  [("agentsg/wyckoff.py", "general_position_multiplicity"), ("agentsg/cell/diagrams.py", "best_general_point")],
  [("SPECIALIZES", "wyckoff_position")],
  iucr="Wyckoff position", wiki="Wyckoff positions")
c("stabilizer", "Stabilizer (group action)", "mathematics",
  "For a group acting on a set, the subgroup fixing a given element; used for sites (direct space) and for reflection strata (reciprocal space).",
  [("agentsg/wyckoff.py", ""), ("agentsg/reflection_lattice.py", "_stabiliser"), ("agentsg/cell/diagrams.py", "_stabilizer")],
  [("PART_OF", "group_action"), ("RELATED_TO", "orbit")],
  iucr="Stabilizer", wiki="Group action")
c("orbit", "Orbit (group action)", "mathematics",
  "The set of images of an element under a group; point-group orbits of Miller indices give symmetry-equivalent reflections.",
  [("agentsg/reflections.py", "_point_group_orbit"), ("agentsg/reflections.py", "equivalent_reflections")],
  [("PART_OF", "group_action"), ("RELATED_TO", "stabilizer")],
  wiki="Group action")
c("orbit_stabilizer_theorem", "Orbit–stabilizer theorem", "mathematics",
  "|orbit| = |G| / |stabilizer|; gives Wyckoff multiplicities and reflection multiplicities (|Laue group| / epsilon).",
  [("agentsg/wyckoff.py", ""), ("agentsg/reflections.py", "reflection_multiplicity")],
  [("PART_OF", "group_action")],
  wiki="Group action")
c("group_action", "Group action", "mathematics",
  "A group acting on a set (points by x ↦ Wx + w, Miller indices by h ↦ hW, lattices by G ↦ MᵀGM).",
  [("agentsg/group.py", "transform_hkl"), ("agentsg/reflection_lattice.py", "")],
  [("PART_OF", "group_theory")],
  wiki="Group action")

# ------------------------------------------------------- reflections / absences
c("miller_indices", "Miller indices", "crystallography",
  "Integer triple (h, k, l) labelling a reflection / lattice plane; treated as a row vector acting by h' = hW.",
  [("agentsg/group.py", "transform_hkl"), ("agentsg/serve/serialize.py", "parse_hkl")],
  [("PART_OF", "reciprocal_lattice"), ("RELATED_TO", "covariance_contravariance")],
  iucr="Miller indices", wiki="Miller index")
c("reciprocal_lattice", "Reciprocal lattice / reciprocal space", "crystallography",
  "The dual lattice of the direct lattice; reflections are its nodes and the reciprocal metric tensor is G⁻¹.",
  [("agentsg/cell/metric.py", "reciprocal_metric_tensor"), ("agentsg/cell/metric.py", "reciprocal")],
  [("DUAL_OF", "lattice"), ("USES", "dual_lattice")],
  iucr="Reciprocal lattice", wiki="Reciprocal lattice")
c("structure_factor_phase", "Structure-factor phase shift under symmetry", "crystallography",
  "An operation (W, w) fixing h multiplies F(h) by exp(2πi h·w); the phase h·w in turns decides absences and phase restrictions.",
  [("agentsg/group.py", "phase_shift"), ("agentsg/group.py", "phase_restriction"), ("agentsg/change_of_basis.py", "")],
  [("USES", "structure_factor"), ("RELATED_TO", "systematic_absences"), ("RELATED_TO", "centric_reflection")],
  iucr="Structure factor", wiki="Structure factor")
c("structure_factor", "Structure factor", "crystallography",
  "F(h) = Σ f_j exp(2πi h·x_j); its symmetry properties (absences, centric phases, semi-invariants) are what agentsg derives from the operators.",
  [("agentsg/group.py", "phase_shift")],
  [("RELATED_TO", "miller_indices")],
  iucr="Structure factor", wiki="Structure factor")
c("systematic_absences", "Systematic absences (extinctions)", "crystallography",
  "A reflection h is absent iff some operation fixes h (hW = h) with non-integral phase h·w; the single rule from which all reflection conditions are derived.",
  [("agentsg/group.py", "is_systematically_absent"), ("agentsg/reflections.py", ""), ("agentsg/reflection_lattice.py", "")],
  [("USES", "structure_factor_phase"), ("USES", "stabilizer"), ("RELATED_TO", "reflection_conditions")],
  iucr="Systematic absences", wiki="Systematic absence", aliases=["extinctions"])
c("reflection_conditions", "Reflection conditions", "crystallography",
  "Generator-derived congruences for which reflections can be present, pruned against the Smith-form lattice of each stratum and reported as integral, zonal, and serial conditions. Not a transcription of the International Tables lists.",
  [("agentsg/reflection_lattice.py", "reflection_conditions"), ("agentsg/reflections.py", "reflection_conditions")],
  [("USES", "systematic_absences"), ("USES", "reflection_stratum"), ("USES", "dual_lattice"), ("USES", "smith_normal_form")],
  iucr="Reflection conditions", wiki="Systematic absence")
c("integral_reflection_conditions", "Integral reflection conditions", "crystallography",
  "Conditions on all hkl arising from lattice centring, e.g. F: h+k, h+l = 2n.",
  [("agentsg/reflections.py", ""), ("agentsg/reflection_lattice.py", "")],
  [("SPECIALIZES", "reflection_conditions"), ("USES", "lattice_centring")],
  iucr="Integral reflection conditions", wiki="Systematic absence")
c("zonal_reflection_conditions", "Zonal reflection conditions", "crystallography",
  "Conditions on a zone (hk0, h0l, 0kl, hhl …) arising from glide planes.",
  [("agentsg/reflections.py", ""), ("agentsg/reflection_lattice.py", "")],
  [("SPECIALIZES", "reflection_conditions"), ("USES", "glide_plane")],
  iucr="Zonal reflection conditions", wiki="Systematic absence")
c("serial_reflection_conditions", "Serial reflection conditions", "crystallography",
  "Conditions on a row (h00, 0k0, 00l …) arising from screw axes.",
  [("agentsg/reflections.py", ""), ("agentsg/reflection_lattice.py", "")],
  [("SPECIALIZES", "reflection_conditions"), ("USES", "screw_axis")],
  iucr="Serial reflection conditions", wiki="Systematic absence")
c("reflection_stratum", "Reflection stratum (class with a given stabilizer)", "crystallography",
  "The set of reciprocal-lattice vectors with the same stabilizer in the point group; the generic part of a rational subspace, computed as the intersection closure of fixed subspaces — exactly the ITA reflection classes hkl, 0kl, h00, hhl….",
  [("agentsg/reflection_lattice.py", ""), ("agentsg/reflection_lattice.py", "strata"), ("agentsg/reflection_lattice.py", "class_name")],
  [("USES", "stabilizer"), ("USES", "fixed_subspace"), ("USES", "saturated_sublattice"), ("PART_OF", "reflection_conditions")],
  iucr="Zone", wiki="Zone axis")
c("augmented_translation_lattice", "Augmented translation lattice Λ_S and its dual", "mathematics",
  "On a stratum the operations with W in the stabilizer act like pure translations, so present reflections are the dual of Λ_S = Z³ + ⟨w⟩; the same construction with allowed origins gives the semi-invariants.",
  [("agentsg/reflection_lattice.py", "stratum_lattice"), ("agentsg/semi_invariants.py", "origin_lattice")],
  [("USES", "dual_lattice"), ("USES", "smith_normal_form"), ("RELATED_TO", "reflection_conditions"), ("RELATED_TO", "structure_seminvariant")],
  wiki="Dual lattice")
c("centric_reflection", "Centric reflection / phase restriction", "crystallography",
  "A reflection mapped to −h by some operation has its phase restricted to two values; agentsg reports absence, centricity and the restricted phase (SgInfo semantics).",
  [("agentsg/group.py", "is_reflection_centric"), ("agentsg/group.py", "PhaseRestriction")],
  [("USES", "structure_factor_phase"), ("RELATED_TO", "friedel_law"), ("RELATED_TO", "wilson_statistics")],
  iucr="Centric and acentric distribution", wiki="Structure factor")
c("friedel_law", "Friedel's law", "crystallography",
  "|F(h)| = |F(−h)| in the absence of anomalous scattering, so the diffraction pattern has the Laue symmetry.",
  [("agentsg/cell/ambiguity.py", ""), ("agentsg/reflections.py", "equivalent_reflections")],
  [("RELATED_TO", "laue_class")],
  iucr="Friedel's law", wiki="Friedel's law")
c("equivalent_reflections", "Symmetry-equivalent reflections", "crystallography",
  "The orbit of h under the point group (with Friedel mates); its size is the reflection multiplicity.",
  [("agentsg/reflections.py", "equivalent_reflections"), ("agentsg/reflections.py", "EquivalentHKL")],
  [("IS_A", "orbit"), ("USES", "laue_class")],
  iucr="Laue indices", wiki="Miller index")
c("epsilon_factor", "Epsilon factor (reflection stabilizer order)", "crystallography",
  "The number of point-group operations fixing h; the expected intensity enhancement factor ε of Wilson statistics.",
  [("agentsg/reflections.py", "epsilon_factor"), ("agentsg/reflections.py", "reflection_multiplicity")],
  [("IS_A", "stabilizer"), ("USES", "orbit_stabilizer_theorem"), ("RELATED_TO", "wilson_statistics")],
  iucr="Centric and acentric distribution", wiki="Wilson plot")
c("reciprocal_asu", "Reciprocal-space asymmetric unit", "crystallography",
  "A set of unique reflections under the Laue group (CCP4/cctbx/gemmi convention); agentsg maps hkl into it.",
  [("agentsg/asu.py", "ReciprocalAsu"), ("agentsg/cell/ambiguity.py", "_map_to_asu")],
  [("SPECIALIZES", "asymmetric_unit"), ("USES", "laue_class")],
  iucr="Asymmetric unit", wiki="Fundamental domain")

# -------------------------------------------------- origins and semi-invariants
c("allowed_origins", "Allowed (permissible) origins", "crystallography",
  "Origin shifts o with (W − I)o ∈ L for all W, which leave the operator set unchanged; they form a group T' ⊇ L whose discrete part is the set of alternative origins.",
  [("agentsg/semi_invariants.py", "OriginLattice"), ("agentsg/semi_invariants.py", "discrete_allowed_origins")],
  [("USES", "smith_normal_form"), ("USES", "primitive_basis"), ("RELATED_TO", "euclidean_normalizer"), ("RELATED_TO", "origin_shift")],
  iucr="Normalizer", wiki="Euclidean normalizer", aliases=["alternative origins", "permissible origins", "Cheshire origins"])
c("floating_origin", "Floating origin (polar direction)", "crystallography",
  "A continuous origin freedom along the common +1 eigenspace of all rotation parts (kernel of the stacked W − I); zero invariant factors of the Smith form.",
  [("agentsg/semi_invariants.py", "floating_origin_basis"), ("agentsg/semi_invariants.py", "pin_floating_origin")],
  [("PART_OF", "allowed_origins"), ("USES", "kernel_nullspace"), ("USES", "smith_normal_form")],
  wiki="Polar point group", aliases=["polar axis"])
c("euclidean_normalizer", "Euclidean normalizer (Cheshire group)", "crystallography",
  "The normalizer of a space group in the Euclidean group; its translation part is exactly the group of allowed origin shifts.",
  [("agentsg/semi_invariants.py", "origin_lattice"), ("agentsg/semi_invariants.py", "OriginLattice")],
  [("USES", "normalizer"), ("RELATED_TO", "allowed_origins")],
  iucr="Normalizer", wiki="Euclidean normalizer", refs=["ITA Vol. A Chapter 3.5 (Euclidean and affine normalizers)"])
c("normalizer", "Normalizer of a subgroup", "mathematics",
  "N_G(H) = {g : gHg⁻¹ = H}; the largest subgroup in which H is normal.",
  [("agentsg/semi_invariants.py", "origin_lattice")],
  [("PART_OF", "group_theory"), ("USES", "conjugation")],
  iucr="Normalizer", wiki="Centralizer and normalizer")
c("structure_seminvariant", "Structure seminvariant", "crystallography",
  "A reflection (or linear combination) whose phase is unchanged by every allowed origin shift: h·o ∈ Z for all o ∈ T'; the dual lattice of the allowed-origin group, reported as congruences (vector, modulus).",
  [("agentsg/semi_invariants.py", "semi_invariants"), ("agentsg/semi_invariants.py", "SemiInvariant"), ("agentsg/semi_invariants.py", "is_semi_invariant")],
  [("DUAL_OF", "allowed_origins"), ("USES", "dual_lattice"), ("USES", "linear_congruence"), ("RELATED_TO", "direct_methods")],
  iucr="Direct methods", wiki="Direct methods (crystallography)", refs=["SgInfo (Grosse-Kunstleve) TabTrial_si", "Giacovazzo, Direct Phasing in Crystallography"])
c("direct_methods", "Direct methods (context)", "crystallography",
  "Phase determination from structure-factor statistics; seminvariants and origin definition are its group-theoretical foundation.",
  [("agentsg/semi_invariants.py", "semi_invariants")],
  [("USES", "structure_seminvariant")],
  iucr="Direct methods", wiki="Direct methods (crystallography)")

# --------------------------------------------------------------- Patterson
c("patterson_function", "Patterson function / vectors", "crystallography",
  "The autocorrelation of the electron density; peaks at interatomic vectors u = x_i − x_j.",
  [("agentsg/harker.py", ""), ("agentsg/harker.py", "harker_vector")],
  [("RELATED_TO", "harker_section")],
  iucr="Patterson vector", wiki="Patterson function")
c("harker_section", "Harker sections and lines", "crystallography",
  "Loci of self-Patterson vectors u = (I − W)x − w; each left-null vector n of (I − W) gives a constraint n·u ≡ −n·w (mod 1) defining a plane, line or point.",
  [("agentsg/harker.py", ""), ("agentsg/harker.py", "harker_sections"), ("agentsg/harker.py", "site_from_harker")],
  [("USES", "patterson_function"), ("USES", "left_nullspace"), ("USES", "linear_congruence"), ("USES", "affine_subspace")],
  iucr="Harker section", wiki="Patterson function")

# ------------------------------------------------------------ asymmetric units
c("asymmetric_unit", "Asymmetric unit (fundamental domain)", "crystallography",
  "A region containing exactly one representative of each orbit; agentsg provides the conventional brick and a metric Dirichlet/Voronoi domain.",
  [("agentsg/asu.py", ""), ("agentsg/asu.py", "DirectAsuBrick")],
  [("USES", "crystallographic_orbit")],
  iucr="Asymmetric unit", wiki="Fundamental domain")
c("dirichlet_domain", "Dirichlet / Voronoi domain of an orbit", "crystallography",
  "The set of points closer to a seed than to any of its orbit mates, bounded by bisecting half-spaces; a metric asymmetric unit whose shape depends on the origin gauge.",
  [("agentsg/asu.py", "DirichletAsu"), ("agentsg/asu.py", "build_dirichlet_asu"), ("agentsg/asu.py", "optimize_asu")],
  [("SPECIALIZES", "asymmetric_unit"), ("IS_A", "voronoi_cell"), ("USES", "half_space"), ("USES", "allowed_origins"), ("USES", "monte_carlo")],
  iucr="Voronoi domain", wiki="Voronoi diagram", aliases=["Wigner–Seitz cell"])
c("voronoi_cell", "Voronoi cell", "mathematics",
  "The region of points nearer to one site of a point set than to any other.",
  [("agentsg/asu.py", "DirichletAsu")],
  [], iucr="Wigner-Seitz cell", wiki="Voronoi diagram")
c("half_space", "Half-space / convex polytope", "mathematics",
  "A region n·x ≤ c; intersections of half-spaces define the Dirichlet ASU facets.",
  [("agentsg/asu.py", "HalfSpace")],
  [("PART_OF", "dirichlet_domain")], wiki="Half-space (geometry)")
c("monte_carlo", "Monte-Carlo estimation", "algorithm",
  "Random sampling used to estimate ASU volume fraction, inertia eigenvalues and sphericity.",
  [("agentsg/asu.py", "DirichletAsu")],
  [("RELATED_TO", "inertia_tensor")], wiki="Monte Carlo method")
c("inertia_tensor", "Second-moment (inertia) tensor and sphericity", "mathematics",
  "The 3×3 second-moment matrix of a region; the ratio λ_min/λ_max of its eigenvalues scores how spherical a domain is.",
  [("agentsg/asu.py", "DirichletAsu"), ("agentsg/asu.py", "_eigh3_sorted")],
  [("USES", "eigenvalue_symmetric"), ("RELATED_TO", "isoperimetric_quotient")],
  wiki="Moment of inertia")
c("isoperimetric_quotient", "Isoperimetric quotient", "mathematics",
  "36πV²/A³, equal to 1 only for a ball; agentsg estimates a proxy for ASU compactness.",
  [("agentsg/asu.py", "DirichletAsu")],
  [], wiki="Isoperimetric inequality")

# ---------------------------------------------------------------- subgroups
c("subgroup", "Subgroup", "mathematics",
  "A subset closed under the group operation; maximal subgroups of a space group are derived from the operator list.",
  [("agentsg/subgroups.py", ""), ("agentsg/subgroups.py", "point_group_subgroups")],
  [("PART_OF", "group_theory")],
  iucr="Subgroup", wiki="Subgroup")
c("t_subgroup", "Translationengleiche (t) subgroup", "crystallography",
  "A subgroup with the same translation lattice and a smaller point group (type I).",
  [("agentsg/subgroups.py", ""), ("agentsg/subgroups.py", "subgroup_edges")],
  [("SPECIALIZES", "subgroup"), ("CONTRASTS_WITH", "k_subgroup")],
  iucr="Translationengleiche subgroup", wiki="Space group")
c("k_subgroup", "Klassengleiche (k) subgroup", "crystallography",
  "A subgroup with the same point group and fewer translations (type II): dropping centring (IIa) or an enlarged cell / invariant sublattice (IIb).",
  [("agentsg/subgroups.py", ""), ("agentsg/subgroups.py", "_k_iib_edges")],
  [("SPECIALIZES", "subgroup"), ("USES", "sublattice"), ("USES", "lattice_centring")],
  iucr="Klassengleiche subgroups", wiki="Space group")
c("index_of_subgroup", "Index of a subgroup / coset", "mathematics",
  "The number of cosets of H in G; index-2 subgroups are kernels of homomorphisms G → C₂ found by nullspaces over GF(2).",
  [("agentsg/subgroups.py", "_index2_subgroups"), ("agentsg/subgroups.py", "_f2_nullspace"), ("agentsg/cell/ambiguity.py", "_coset_reps")],
  [("PART_OF", "group_theory"), ("USES", "coset"), ("USES", "finite_field_gf2")],
  iucr="Coset", wiki="Index of a subgroup")
c("coset", "Coset decomposition", "mathematics",
  "Partition of a group into translates gH of a subgroup; reindexing operators and twin laws are coset representatives of the Laue group in the lattice symmetry group.",
  [("agentsg/cell/ambiguity.py", ""), ("agentsg/cell/ambiguity.py", "_coset_reps"), ("agentsg/cell/reindex.py", "")],
  [("PART_OF", "group_theory"), ("RELATED_TO", "index_of_subgroup")],
  iucr="Coset", wiki="Coset")
c("finite_field_gf2", "Linear algebra over GF(2)", "mathematics",
  "Nullspaces of 0/1 matrices modulo 2, used to enumerate homomorphisms onto C₂ (index-2 subgroups).",
  [("agentsg/subgroups.py", "_f2_nullspace")],
  [("IS_A", "kernel_nullspace")], wiki="GF(2)")
c("group_homomorphism", "Group homomorphism and kernel", "mathematics",
  "A structure-preserving map between groups; its kernel is a normal subgroup (index-2 subgroups = kernels of maps to C₂).",
  [("agentsg/subgroups.py", "_index2_subgroups")],
  [("PART_OF", "group_theory")], iucr="Group homomorphism", wiki="Group homomorphism")
c("group_theory", "Group theory (finite and crystallographic groups)", "mathematics",
  "The algebra of symmetry: closure, subgroups, cosets, normalizers, actions; the foundation of every derived object in agentsg.",
  [("agentsg/__init__.py", ""), ("agentsg/group.py", "")],
  [], iucr="Group", wiki="Group theory")
c("permutation_group_s4", "Symmetric group S₄ (superbase relabellings)", "mathematics",
  "The 24 permutations of the four superbase labels, extended by −I to order 48; realised as integer unimodular changes of basis.",
  [("agentsg/cell/selling_group.py", ""), ("agentsg/cell/selling_group.py", "selling_group")],
  [("IS_A", "group_theory"), ("USES", "unimodular_matrix"), ("PART_OF", "selling_closure")],
  wiki="Symmetric group")

# ------------------------------------------------------------ cells and metric
c("unit_cell", "Unit cell", "crystallography",
  "The parallelepiped (a, b, c, α, β, γ) spanned by a lattice basis; agentsg treats it numerically via its metric tensor.",
  [("agentsg/cell/metric.py", ""), ("agentsg/cell/metric.py", "UnitCell")],
  [("USES", "metric_tensor"), ("RELATED_TO", "lattice")],
  iucr="Unit cell", wiki="Unit cell")
c("metric_tensor", "Metric tensor (Gram matrix)", "mathematics",
  "G_ij = a_i·a_j; squared lengths are xᵀGx, volume² = det G, and a basis change acts by G' = PᵀGP.",
  [("agentsg/cell/metric.py", "metric_tensor"), ("agentsg/cell/reduction.py", "_gram_from_params"), ("agentsg/cell/constraints.py", "")],
  [("IS_A", "gram_matrix"), ("USES", "determinant"), ("PART_OF", "unit_cell")],
  iucr="Metric tensor", wiki="Metric tensor")
c("gram_matrix", "Gram matrix", "mathematics",
  "The matrix of inner products of a set of vectors; symmetric positive definite for a basis.",
  [("agentsg/cell/reduction.py", "_gram_from_params")],
  [], wiki="Gram matrix")
c("reciprocal_metric", "Reciprocal metric tensor and reciprocal cell", "crystallography",
  "G* = G⁻¹; gives d-spacings via 1/d² = hᵀG*h and the reciprocal cell parameters.",
  [("agentsg/cell/metric.py", "UnitCell")],
  [("USES", "matrix_inverse"), ("PART_OF", "reciprocal_lattice"), ("RELATED_TO", "d_spacing")],
  iucr="Reciprocal lattice", wiki="Reciprocal lattice")
c("d_spacing", "Interplanar spacing d(hkl) and Bragg angle", "crystallography",
  "d = 1/|h a* + k b* + l c*|; 2θ from Bragg's law λ = 2d sin θ.",
  [("agentsg/cell/metric.py", "UnitCell")],
  [("USES", "reciprocal_metric"), ("USES", "braggs_law")],
  iucr="Interplanar spacing", wiki="Bragg's law")
c("braggs_law", "Bragg's law", "crystallography",
  "nλ = 2d sin θ, relating diffraction angle to interplanar spacing.",
  [("agentsg/cell/metric.py", "UnitCell")],
  [], iucr="Bragg's law", wiki="Bragg's law")
c("orthogonalization", "Orthogonalization / fractionalization matrix", "crystallography",
  "The matrix M with Cartesian = M·fractional (a along x, b in xy) and its inverse.",
  [("agentsg/cell/metric.py", "UnitCell"), ("agentsg/cell/rootform.py", "_cart_basis")],
  [("USES", "matrix_inverse"), ("PART_OF", "unit_cell")],
  wiki="Fractional coordinates")
c("cell_volume", "Unit-cell volume", "crystallography",
  "V = abc√(1 − cos²α − cos²β − cos²γ + 2cosα cosβ cosγ) = √det G.",
  [("agentsg/cell/metric.py", "UnitCell"), ("agentsg/cell/rootform.py", "_cell_volume")],
  [("USES", "determinant"), ("PART_OF", "unit_cell")],
  iucr="Unit cell", wiki="Unit cell")
c("metric_invariance", "Metric invariance WᵀGW = G", "crystallography",
  "A point-group operation leaves the metric tensor invariant; this single identity yields the crystal-system restrictions, metric symmetrisation and the free-parameter count.",
  [("agentsg/cell/constraints.py", ""), ("agentsg/cell/constraints.py", "metric_is_invariant"), ("agentsg/cell/constraints.py", "free_metric_parameters")],
  [("USES", "metric_tensor"), ("USES", "point_group"), ("RELATED_TO", "reynolds_operator"), ("RELATED_TO", "neumanns_principle")],
  iucr="Metric specialization", wiki="Metric tensor")
c("neumanns_principle", "Neumann's principle (context)", "crystallography",
  "Physical properties (here the metric) are invariant under the point group of the crystal.",
  [("agentsg/cell/constraints.py", "")],
  [("RELATED_TO", "metric_invariance")], iucr="Neumann's principle", wiki="Neumann's principle")
c("reynolds_operator", "Reynolds operator (group averaging)", "mathematics",
  "Projection onto the invariant subspace by averaging over the group: G_sym = (1/|G|) Σ WᵀGW; used to symmetrise metrics and to score two-folds.",
  [("agentsg/cell/constraints.py", "symmetrize_metric"), ("agentsg/cell/g6.py", "_symmetrize_metric"), ("agentsg/lattice_symmetry.py", "_reynolds_two_fold")],
  [("USES", "metric_invariance"), ("IS_A", "projection_operator")],
  wiki="Reynolds operator")
c("projection_operator", "Projection onto a subspace", "mathematics",
  "An idempotent linear map onto a subspace; metric symmetrisation projects onto the point-group-invariant metrics.",
  [("agentsg/cell/constraints.py", "symmetrize_metric")],
  [], wiki="Projection (linear algebra)")
c("primitive_cell", "Primitive cell / conventional-to-primitive transformation", "crystallography",
  "A cell containing one lattice point; ITA Table 5.1.3.1 matrices with det P = 1/m take the conventional centred cell to it (G_P = PᵀGP), validated against the centring vectors.",
  [("agentsg/cell/primitive.py", ""), ("agentsg/cell/primitive.py", "primitive_transform"), ("agentsg/cell/selling_cob.py", "primitive_matrix")],
  [("USES", "lattice_centring"), ("USES", "change_of_basis"), ("CONTRASTS_WITH", "conventional_cell")],
  iucr="Primitive cell", wiki="Primitive cell")
c("primitive_basis", "Primitive basis of a (centred) lattice", "mathematics",
  "A Z-basis of L = Z³ + ⟨centring⟩ obtained by row Hermite normal form; used to write allowed origins in integral coordinates.",
  [("agentsg/semi_invariants.py", "_primitive_basis")],
  [("USES", "hermite_normal_form"), ("RELATED_TO", "primitive_cell")],
  iucr="Primitive basis", wiki="Primitive cell")
c("conventional_cell", "Conventional cell", "crystallography",
  "The standard (possibly centred) cell of ITA whose axes follow the symmetry; identification returns the ITA type in this cell even for primitive inputs.",
  [("agentsg/identify.py", ""), ("agentsg/cell/primitive.py", "")],
  [("RELATED_TO", "lattice_centring")],
  iucr="Conventional cell", wiki="Unit cell")
c("reduced_cell", "Reduced cell", "crystallography",
  "A unique canonical cell of a lattice (Niggli or Delaunay/Selling reduced) that removes the infinite basis ambiguity.",
  [("agentsg/cell/reduction.py", ""), ("agentsg/cell/rootform.py", "selling_reduced_cell")],
  [("USES", "unimodular_matrix")],
  iucr="Reduced cell", wiki="Lattice reduction")
c("niggli_reduction", "Niggli reduction (Křivý–Gruber, Grosse-Kunstleve stabilised)", "algorithm",
  "Iterative reduction of the metric scalars (A, B, C, ξ, η, ζ) with tolerance-aware comparisons, tracking the integer change of basis M with MᵀGM = G_reduced.",
  [("agentsg/cell/reduction.py", ""), ("agentsg/cell/reduction.py", "niggli_gk"), ("agentsg/cell/reduction.py", "niggli_reduce")],
  [("SPECIALIZES", "reduced_cell"), ("USES", "metric_tensor"), ("USES", "unimodular_matrix"), ("USES", "floating_point_tolerance")],
  iucr="Reduced cell", wiki="Lattice reduction",
  refs=["Grosse-Kunstleve, Sauter & Adams (2004). Acta Cryst. A60, 1-6", "Křivý & Gruber (1976). Acta Cryst. A32, 297"])
c("delaunay_selling_reduction", "Delaunay / Selling reduction to an obtuse superbase", "algorithm",
  "Reduce a lattice basis to four zero-sum vectors with all pairwise scalar products ≤ 0 by repeated Selling flips; the reduced superbase is unique up to a finite closure.",
  [("agentsg/cell/rootform.py", "delaunay_superbase"), ("agentsg/cell/selling_closure.py", "_selling_flip"), ("agentsg/cell/g6.py", "")],
  [("SPECIALIZES", "reduced_cell"), ("USES", "obtuse_superbase"), ("RELATED_TO", "niggli_reduction")],
  iucr="Reduced cell", wiki="Lattice reduction", refs=["Delaunay (1933)", "Selling (1874)", "Conway & Sloane, Low-dimensional lattices VI"])
c("obtuse_superbase", "Obtuse superbase", "mathematics",
  "Four vectors v0..v3 summing to zero with all v_i·v_j ≤ 0; its conorms p_ij = −v_i·v_j (six) and vonorms (seven) are lattice invariants.",
  [("agentsg/cell/selling_group.py", ""), ("agentsg/cell/selling_closure.py", "_is_obtuse"), ("agentsg/cell/rootform.py", "conorms")],
  [("PART_OF", "delaunay_selling_reduction"), ("RELATED_TO", "voronoi_type")],
  wiki="Lattice reduction", aliases=["superbase", "conorms", "vonorms"])
c("voronoi_type", "Voronoi (Delaunay) type of a lattice V1–V5", "crystallography",
  "The five combinatorial types of three-dimensional Voronoi cells, classified from the zero conorms of the obtuse superbase; higher types have several non-isometric superbase classes.",
  [("agentsg/cell/selling_closure.py", ""), ("agentsg/cell/selling_closure.py", "voronoi_type_from_superbase")],
  [("USES", "obtuse_superbase"), ("RELATED_TO", "voronoi_cell"), ("PART_OF", "selling_closure")],
  wiki="Plesiohedron", refs=["Kurlin, V. (2022/2026) arXiv:2201.10543, Lemmas 4.1-4.5"])
c("selling_closure", "Selling-superbase closure", "crystallography",
  "The complete finite set of obtuse superbases of one lattice (S₄×{±I} orbit at V1, plus extra classes at V2–V5); enumerated for exact reindexing and identity certification. Zero conorms use a relative floor of 1e-9, widened by three angle-sigma widths (default 0.05 degrees), and a relative boundary width of 1e-3.",
  [("agentsg/cell/selling_closure.py", "selling_superbase_closure"), ("agentsg/cell/canonical.py", ""), ("agentsg/cell/selling_cob.py", "")],
  [("USES", "obtuse_superbase"), ("USES", "permutation_group_s4"), ("USES", "voronoi_type"), ("RELATED_TO", "reindexing")],
  wiki="Lattice reduction", refs=["Zwart, agentsg manuscript: 'The Selling closure for lattice comparison'"])
c("kurlin_root_form", "Kurlin root products / root form (isometry invariant)", "mathematics",
  "Six root products r_ij = √p_ij of the obtuse superbase; Kurlin's ordered 2×3 root form is a complete isometry invariant, agentsg's globally sorted 6-tuple is a continuous (but not complete) Euclidean search key.",
  [("agentsg/cell/rootform.py", ""), ("agentsg/cell/rootform.py", "sorted_root_key"), ("agentsg/cell/rootform.py", "root_products")],
  [("USES", "obtuse_superbase"), ("USES", "rearrangement_inequality"), ("USES", "euclidean_distance"), ("RELATED_TO", "lattice_space")],
  wiki="Lattice (group)", refs=["Kurlin, V. Mathematics of 2-dimensional and 3-dimensional lattices, arXiv:2201.10543"])
c("rearrangement_inequality", "Rearrangement inequality", "mathematics",
  "For sorted vectors ‖sort(x) − sort(y)‖ ≤ min_σ ‖x − σy‖; makes the sorted root key a conservative lower bound on the orbit distance.",
  [("agentsg/cell/rootform.py", "sorted_key_lower_bound"), ("agentsg/cell/rootform.py", "")],
  [], wiki="Rearrangement inequality")
c("similarity_invariant", "Similarity (scale-free) invariant", "mathematics",
  "The sorted root key divided by V^{1/3}, invariant under isotropic scaling; separates shape from volume in lattice comparison.",
  [("agentsg/cell/rootform.py", "similarity_invariant"), ("agentsg/cell/rootform.py", "root_volume_decomposition")],
  [("SPECIALIZES", "kurlin_root_form")], wiki="Similarity (geometry)")
c("lattice_space", "Space of lattices / lattice manifold", "mathematics",
  "Treating lattices as points of a continuous space (G6, S6, root form) so that comparison, symmetry deficiency and nearest-neighbour search become geometric.",
  [("agentsg/cell/g6.py", ""), ("agentsg/cell/manifold.py", ""), ("agentsg/cell/neartree.py", "")],
  [("USES", "g6_s6_embedding"), ("USES", "kurlin_root_form"), ("USES", "nearest_neighbour_search")],
  wiki="Lattice (group)", refs=["Andrews & Bernstein (1988, 2014)", "Kurlin (2022)"])
c("g6_s6_embedding", "G6 / S6 lattice embeddings (Andrews–Bernstein)", "mathematics",
  "G6 = (a·a, b·b, c·c, 2b·c, 2a·c, 2a·b) (Niggli) and S6 = the six Selling scalars; boundary-aware distances minimise over reduction-boundary transforms.",
  [("agentsg/cell/g6.py", ""), ("agentsg/cell/g6.py", "g6_from_metric"), ("agentsg/cell/g6.py", "s6_from_metric")],
  [("USES", "metric_tensor"), ("PART_OF", "lattice_space")],
  wiki="Lattice reduction", refs=["Andrews & Bernstein (1988). Acta Cryst. A44, 1009", "Andrews & Bernstein (2014). J. Appl. Cryst. 47, 346"])
c("symmetry_deficiency", "Symmetry deficiency (distance to a holohedry)", "crystallography",
  "A continuous measure of how far a lattice is from a given point-group subspace: distance from the cell to its Reynolds-symmetrised cell.",
  [("agentsg/cell/g6.py", "distance_to_symmetry"), ("agentsg/cell/g6.py", "kurlin_distance_to_symmetry"), ("agentsg/lattice_symmetry.py", "kurlin_distance_to_two_fold")],
  [("USES", "reynolds_operator"), ("USES", "kurlin_root_form"), ("RELATED_TO", "pseudosymmetry")],
  iucr="Pseudo symmetry", wiki="Lattice (group)")
c("pseudosymmetry", "Pseudo-symmetry / metric pseudo-symmetry", "crystallography",
  "Approximate symmetry of the lattice metric beyond the crystal's true symmetry, quantified by Le Page delta and tolerance.",
  [("agentsg/lattice_symmetry.py", ""), ("agentsg/cell/ambiguity.py", "")],
  [("RELATED_TO", "twinning_merohedry")], iucr="Pseudo symmetry", wiki="Pseudosymmetry")
c("lattice_symmetry_determination", "Metric lattice symmetry (Le Page / Lebedev)", "algorithm",
  "Find the holohedry of a measured cell: test the 81 Lebedev two-folds (integer {−1,0,1} matrices) by the Le Page angle between direct and reciprocal axes, then close the accepted set. The default cutoff is 3 degrees and the length gate is 2 percent.",
  [("agentsg/lattice_symmetry.py", ""), ("agentsg/lattice_symmetry.py", "le_page_delta"), ("agentsg/lattice_symmetry.py", "tolerance_metric_symmetry")],
  [("USES", "metric_invariance"), ("USES", "unimodular_matrix"), ("USES", "group_closure"), ("RELATED_TO", "holohedry")],
  iucr="Metric specialization", wiki="Lattice (group)",
  refs=["Le Page (1982). J. Appl. Cryst. 15, 255", "Lebedev, Vagin & Murshudov (2006). Acta Cryst. D62, 83", "Zwart, Grosse-Kunstleve & Adams (2006)"])
c("cell_comparison", "Cell comparison by sublattice search (lego / target)", "algorithm",
  "Niggli-reduce two cells, take the integer volume ratio r, enumerate index-r sublattices of the smaller cell in Hermite normal form, and accept those whose reduced metric matches the larger cell.",
  [("agentsg/cell/compare.py", ""), ("agentsg/cell/compare.py", "compare_cells"), ("agentsg/cell/celldb.py", "CellDatabase")],
  [("USES", "niggli_reduction"), ("USES", "sublattice"), ("USES", "hermite_normal_form"), ("USES", "floating_point_tolerance")],
  iucr="Sublattice", wiki="Lattice (group)", refs=["Zwart, Grosse-Kunstleve & Adams (2006), sec. 2.4 / 3.2"])
c("sublattice", "Sublattice / superlattice and index", "mathematics",
  "A subgroup of finite index in a lattice; index-d sublattices of Z³ are enumerated as integer matrices of determinant d in Hermite normal form (count Σ_{adf=n} a²d).",
  [("agentsg/cell/sublattice.py", ""), ("agentsg/cell/sublattice.py", "generate_sublattices"), ("agentsg/cell/sublattice.py", "sublattice_count")],
  [("USES", "hermite_normal_form"), ("USES", "determinant"), ("PART_OF", "integer_lattice")],
  iucr="Sublattice", wiki="Lattice (group)", refs=["Billiet & Rolley-Le Coz (1980)", "Rutherford (2006)", "OEIS A001001"])
c("saturated_sublattice", "Saturated sublattice (V ∩ Z³)", "mathematics",
  "The integer points of a rational subspace; obtained by unimodular column reduction (integer kernel) so the basis is primitive.",
  [("agentsg/reflection_lattice.py", "_int_kernel"), ("agentsg/reflection_lattice.py", "_intersect")],
  [("IS_A", "sublattice"), ("USES", "kernel_nullspace")],
  wiki="Lattice (group)")
c("dual_lattice", "Dual lattice", "mathematics",
  "L* = {h : h·t ∈ Z for all t ∈ L}; for augmented lattices the dual is computed from one Smith normal form as invariant factors, index, basis and congruences.",
  [("agentsg/reflection_lattice.py", "dual_lattice")],
  [("USES", "smith_normal_form"), ("RELATED_TO", "reciprocal_lattice"), ("RELATED_TO", "linear_congruence")],
  iucr="Dual basis", wiki="Dual lattice")
c("dual_basis", "Dual basis", "mathematics",
  "The basis a*, b*, c* with a_i·a*_j = δ_ij; Miller indices are coordinates in it.",
  [("agentsg/change_of_basis.py", "")],
  [("RELATED_TO", "dual_lattice")], iucr="Dual basis", wiki="Dual basis")

# ---------------------------------------------------------- reindexing / twins
c("reindexing", "Reindexing operators (coset P·H)", "crystallography",
  "Integer operators relating two settings of the same lattice; never unique — the full answer is the coset P·H of the lattice symmetry group H = {M : MᵀG_AM = G_A}.",
  [("agentsg/cell/reindex.py", ""), ("agentsg/cell/reindex.py", "reindexing_operators"), ("agentsg/cell/canonical.py", "reindexing_via_canonical")],
  [("USES", "coset"), ("USES", "unimodular_matrix"), ("USES", "selling_closure"), ("RELATED_TO", "change_of_basis")],
  iucr="Twin law", wiki="Twinning (crystallography)")
c("indexing_ambiguity", "Indexing ambiguity (serial crystallography)", "crystallography",
  "When the lattice holohedry exceeds the Laue group, frames can be indexed in |M_lattice|/|L_Laue| inequivalent ways; the operators are left-coset representatives, resolved by intensity correlation.",
  [("agentsg/cell/ambiguity.py", ""), ("agentsg/cell/ambiguity.py", "reindexing_ambiguity_operators"), ("agentsg/cell/ambiguity.py", "ReindexingReference")],
  [("USES", "coset"), ("USES", "laue_class"), ("USES", "holohedry"), ("USES", "pearson_correlation"), ("RELATED_TO", "twinning_merohedry")],
  wiki="Twinning (crystallography)", refs=["Brehm & Diederichs (2014). Acta Cryst. D70, 101", "dials.cosym"])
c("twinning_merohedry", "Twinning by (pseudo-)merohedry and twin laws", "crystallography",
  "When the lattice symmetry H is larger than the crystal's Laue group L, the cosets of L in H are the twin domains and their representatives the twin laws.",
  [("agentsg/cell/reindex.py", ""), ("agentsg/cell/reindex.py", "twin_laws")],
  [("USES", "coset"), ("USES", "lattice_symmetry_determination"), ("USES", "laue_class")],
  iucr="Twinning by merohedry", wiki="Twinning (crystallography)")
c("deformation_manifold", "Deformation manifold with landmarks (path-consistent reindexing)", "algorithm",
  "Under continuous lattice deformation reindexing is path-dependent (monodromy); a scaffold of landmark states with a k-NN graph, shortest paths and a Fiedler coordinate keeps the frame consistent.",
  [("agentsg/cell/manifold.py", ""), ("agentsg/cell/manifold.py", "DeformationManifold")],
  [("USES", "knn_graph"), ("USES", "dijkstra"), ("USES", "graph_laplacian"), ("USES", "farthest_point_sampling"), ("USES", "reindexing")],
  wiki="Monodromy")
c("space_group_identification", "Space-group identification from operators", "algorithm",
  "Match a closed operator set to one of the 230 types up to origin shift and an integer change of basis (|det P| ≤ 4), solving (W − I)p = Δw exactly and expanding centring cosets.",
  [("agentsg/identify.py", ""), ("agentsg/identify.py", "identify_space_group")],
  [("USES", "origin_shift"), ("USES", "change_of_basis"), ("USES", "rational_linear_solve"), ("USES", "coset")],
  iucr="Space group", wiki="Space group")
c("selling_settings", "Settings of a space group under the Selling group", "crystallography",
  "The 48 (distinct fewer) reindexed descriptions of one group obtained by sweeping the order-48 Selling change-of-basis group.",
  [("agentsg/cell/selling_settings.py", ""), ("agentsg/cell/selling_settings.py", "selling_settings")],
  [("USES", "permutation_group_s4"), ("USES", "change_of_basis"), ("RELATED_TO", "indexing_ambiguity")],
  wiki="Space group")

# ------------------------------------------------- numeric / search machinery
c("nearest_neighbour_search", "Nearest-neighbour search on lattice keys", "algorithm",
  "Find lattices closest to a query in root-invariant space: exact k-NN and radius queries.",
  [("agentsg/cell/celldb.py", ""), ("agentsg/cell/rootindex.py", ""), ("agentsg/cell/neartree.py", "")],
  [("USES", "kd_tree"), ("USES", "neartree"), ("USES", "euclidean_distance")],
  wiki="Nearest neighbor search")
c("kd_tree", "k-d tree", "data-structure",
  "Space-partitioning tree for Euclidean nearest-neighbour queries (scipy cKDTree over 6-D root keys).",
  [("agentsg/cell/rootindex.py", ""), ("agentsg/cell/rootindex.py", "RootIndex")],
  [("PART_OF", "nearest_neighbour_search")], wiki="K-d tree")
c("neartree", "NearTree (metric-space index)", "data-structure",
  "A tree that prunes searches with the triangle inequality only, valid for any metric (needed for boundary-aware G6/S6 distances that are not Euclidean norms).",
  [("agentsg/cell/neartree.py", ""), ("agentsg/cell/neartree.py", "NearTree")],
  [("PART_OF", "nearest_neighbour_search"), ("USES", "metric_space")],
  wiki="Metric tree", refs=["Andrews, L. C. (2001). J. Appl. Cryst. 34, 663-668"])
c("metric_space", "Metric space and the triangle inequality", "mathematics",
  "A set with a distance satisfying symmetry and d(x,z) ≤ d(x,y) + d(y,z); the property NearTree pruning relies on.",
  [("agentsg/cell/neartree.py", "")],
  [], wiki="Metric space")
c("euclidean_distance", "Euclidean (L2) distance", "mathematics",
  "‖x − y‖₂ on the six sorted root products (Å) or on G6/S6 (Å²).",
  [("agentsg/cell/rootform.py", "sorted_root_distance"), ("agentsg/cell/g6.py", "_euclid")],
  [("IS_A", "metric_space")], wiki="Euclidean distance")
c("knn_graph", "k-nearest-neighbour graph", "data-structure",
  "Graph joining each lattice state to its k nearest states by key distance; the scaffold of the deformation manifold.",
  [("agentsg/cell/manifold.py", "deformation_graph")],
  [("PART_OF", "deformation_manifold")], wiki="Nearest neighbor graph")
c("dijkstra", "Dijkstra's shortest-path algorithm", "algorithm",
  "Shortest weighted paths from a source; routes each state to its nearest landmark.",
  [("agentsg/cell/manifold.py", "_dijkstra")],
  [("PART_OF", "deformation_manifold")], wiki="Dijkstra's algorithm")
c("graph_laplacian", "Graph Laplacian and Fiedler vector", "mathematics",
  "L = D − A; the eigenvector of the second-smallest eigenvalue orders the states along the deformation.",
  [("agentsg/cell/manifold.py", "DeformationManifold")],
  [("USES", "eigenvalue_symmetric"), ("PART_OF", "deformation_manifold")], wiki="Algebraic connectivity")
c("farthest_point_sampling", "Farthest-point (k-center) sampling", "algorithm",
  "Greedy selection of landmarks that maximise the minimum distance to those already chosen.",
  [("agentsg/cell/manifold.py", "farthest_point_landmarks")],
  [("PART_OF", "deformation_manifold")], wiki="Farthest-first traversal")
c("pearson_correlation", "Pearson correlation coefficient", "mathematics",
  "Normalised covariance of two intensity sets; picks the reindexing branch that best matches a reference.",
  [("agentsg/cell/ambiguity.py", "_pearson"), ("agentsg/cell/ambiguity.py", "ReindexingReference")],
  [("PART_OF", "indexing_ambiguity")], wiki="Pearson correlation coefficient")
c("svd_pca", "Singular value decomposition / principal components", "mathematics",
  "Mean-centred SVD of the n×6 matrix of hit-set root keys gives the PC1–PC2 plane used to display a PDB search.",
  [("agentsg/serve/scatter.py", ""), ("agentsg/serve/scatter.py", "root_svd"), ("agentsg/serve/handlers.py", "_attach_root_plot")],
  [("RELATED_TO", "eigenvalue_symmetric")], wiki="Singular value decomposition", aliases=["SVD", "PCA"])
c("convex_hull", "Convex hull", "mathematics",
  "Smallest convex set containing given points; used to outline coordination polyhedra on cubic plates.",
  [("agentsg/cell/diagrams.py", "_hull_polygon")],
  [], wiki="Convex hull")
c("floating_point_tolerance", "Floating-point tolerance / stabilised comparison", "algorithm",
  "Relative epsilons replacing exact equality in numeric reductions and symmetry tests, so near-degenerate cells neither cycle nor misclassify.",
  [("agentsg/cell/reduction.py", ""), ("agentsg/cell/selling_closure.py", "_conorm_tol"), ("agentsg/cell/constraints.py", "metric_is_invariant")],
  [("CONTRASTS_WITH", "exact_rational_arithmetic")], wiki="Machine epsilon")
c("torus_distance", "Periodic (torus) distance in fractional coordinates", "mathematics",
  "Distance on the unit torus obtained by reducing coordinate differences into (−½, ½]; used for stabilisers and diagram layout.",
  [("agentsg/asu.py", "_wrap_half"), ("agentsg/cell/diagrams.py", "_stabilizer")],
  [("USES", "modular_arithmetic")], wiki="Torus")

# -------------------------------------------------------- exact linear algebra
c("exact_rational_arithmetic", "Exact rational arithmetic (no fixed denominator)", "mathematics",
  "All symmetry algebra uses Python Fractions so 1/2, 1/3, 1/12, 1/60 combine exactly; the deliberate alternative to cctbx's fixed translation base factor.",
  [("agentsg/linalg.py", ""), ("agentsg/rational_solve.py", "")],
  [("CONTRASTS_WITH", "floating_point_tolerance")], wiki="Rational number", refs=["cctbx sgtbx tr_vec base factor 12"])
c("determinant", "Determinant", "mathematics",
  "det of a 3×3 matrix (exact by cofactor expansion, integer by elimination); ±1 for crystallographic rotations, index for sublattice matrices, V² for the metric.",
  [("agentsg/linalg.py", "Matrix3"), ("agentsg/reflection_lattice.py", "_det"), ("agentsg/lattice_symmetry.py", "_det3_int")],
  [], wiki="Determinant")
c("matrix_inverse", "Matrix inverse (cofactor / adjugate, unimodular integer inverse)", "mathematics",
  "Exact inverse via cofactors over Q; for unimodular integer matrices the inverse is integer.",
  [("agentsg/linalg.py", "Matrix3"), ("agentsg/cell/reindex.py", "_int_inv"), ("agentsg/reflection_lattice.py", "_unimodular_inverse")],
  [("USES", "determinant"), ("RELATED_TO", "unimodular_matrix")], wiki="Invertible matrix")
c("gaussian_elimination", "Gaussian elimination / reduced row echelon form", "algorithm",
  "Row reduction over the rationals with pivot tracking; the workhorse for solving (W − I)x = −w, rank and nullspaces.",
  [("agentsg/rational_solve.py", "rref"), ("agentsg/cell/symmetry_elements.py", "_row_reduce"), ("agentsg/cell/constraints.py", "_rank")],
  [("USES", "exact_rational_arithmetic")], wiki="Gaussian elimination", aliases=["rref"])
c("rational_linear_solve", "Solving linear and affine systems over Q", "algorithm",
  "M x = rhs and (W − I)x = −w (mod 1) solved exactly, returning particular solutions and free directions.",
  [("agentsg/rational_solve.py", "solve_affine"), ("agentsg/identify.py", ""), ("agentsg/harker.py", "site_from_harker")],
  [("USES", "gaussian_elimination"), ("RELATED_TO", "affine_subspace")], wiki="System of linear equations")
c("kernel_nullspace", "Kernel / nullspace (rational and integer)", "mathematics",
  "{x : Ax = 0}; over Q from rref, over Z as a saturated basis by unimodular column reduction.",
  [("agentsg/reflection_lattice.py", "_int_kernel"), ("agentsg/cell/symmetry_elements.py", "_integer_kernel"), ("agentsg/harker.py", "_left_nullspace")],
  [("USES", "gaussian_elimination"), ("RELATED_TO", "left_nullspace")], wiki="Kernel (linear algebra)")
c("left_nullspace", "Left nullspace", "mathematics",
  "{n : nᵀM = 0}; each left-null vector of (I − W) is a Harker constraint normal.",
  [("agentsg/harker.py", "_left_nullspace")],
  [("IS_A", "kernel_nullspace")], wiki="Kernel (linear algebra)")
c("matrix_rank", "Matrix rank", "mathematics",
  "Number of independent rows/columns, exact over Q or numeric with pivoting; decides plane/line/point loci and free metric parameters.",
  [("agentsg/harker.py", "_matrix_rank"), ("agentsg/cell/constraints.py", "_rank")],
  [("USES", "gaussian_elimination")], wiki="Rank (linear algebra)")
c("affine_subspace", "Affine subspace", "mathematics",
  "A translate of a linear subspace; solution sets of (W − I)x = −w and Harker loci are affine flats.",
  [("agentsg/wyckoff.py", ""), ("agentsg/harker.py", "")],
  [], wiki="Affine space")
c("affine_transformation", "Affine transformation", "mathematics",
  "x ↦ Wx + w; the form of every crystallographic symmetry operation and change of basis.",
  [("agentsg/symmetry_op.py", ""), ("agentsg/cell/diagrams.py", "_projection_maps")],
  [], iucr="Affine mapping", wiki="Affine transformation")
c("eigenvalue_symmetric", "Eigenvalues of a symmetric 3×3 matrix", "mathematics",
  "Real eigenvalues from the characteristic cubic (closed-form roots) for inertia and Laplacian analyses.",
  [("agentsg/asu.py", "_eigh3_sorted"), ("agentsg/asu.py", "_cubic_roots")],
  [("USES", "eigenvector")], wiki="Eigenvalues and eigenvectors")
c("eigenvector", "Eigenvector", "mathematics",
  "A vector mapped to a multiple of itself; the +1 eigenvector of a two-fold is its axis.",
  [("agentsg/lattice_symmetry.py", "_two_fold_axis_direct")],
  [], wiki="Eigenvalues and eigenvectors")
c("unimodular_matrix", "Unimodular integer matrix / GL(3, Z)", "mathematics",
  "Integer matrices with det ±1; lattice-preserving changes of basis; the 480 Lebedev matrices with entries in {−1,0,1} and finite order.",
  [("agentsg/lattice_symmetry.py", "_lebedev_matrices"), ("agentsg/cell/g6.py", "_unimodular_pm1"), ("agentsg/cell/canonical.py", "_inv3_unimod")],
  [("USES", "determinant"), ("RELATED_TO", "change_of_basis")], wiki="Unimodular matrix")
c("hermite_normal_form", "Hermite normal form", "mathematics",
  "Canonical upper-triangular form of an integer matrix under unimodular row (or column) operations; gives a unique basis per lattice and enumerates sublattices.",
  [("agentsg/reflection_lattice.py", "_row_hnf"), ("agentsg/cell/sublattice.py", ""), ("agentsg/cell/sublattice.py", "is_hermite_normal_form")],
  [("USES", "unimodular_matrix"), ("RELATED_TO", "smith_normal_form")], wiki="Hermite normal form")
c("smith_normal_form", "Smith normal form and invariant factors", "mathematics",
  "U A V = diag(s₁, s₂, s₃) with U, V unimodular and s_i | s_{i+1}; the invariant factors classify the quotient of a lattice by a sublattice (zero → free, 1 → trivial, s > 1 → cyclic torsion).",
  [("agentsg/reflection_lattice.py", "smith_normal_form"), ("agentsg/reflection_lattice.py", "dual_lattice")],
  [("USES", "unimodular_matrix"), ("RELATED_TO", "finitely_generated_abelian_group"), ("RELATED_TO", "hermite_normal_form")],
  wiki="Smith normal form")
c("finitely_generated_abelian_group", "Finitely generated abelian groups (torsion and free parts)", "mathematics",
  "Every such group is ⊕Z/s_i ⊕ Z^r; the allowed-origin group modulo the lattice is Z^{floating} ⊕ torsion, read off the Smith form.",
  [("agentsg/semi_invariants.py", "OriginLattice")],
  [("USES", "smith_normal_form")], wiki="Finitely generated abelian group")
c("linear_congruence", "Linear congruence f·c ≡ 0 (mod m)", "mathematics",
  "Integer conditions on index coefficients; the form in which reflection conditions (k+l = 4n), semi-invariants and Harker constraints are stated.",
  [("agentsg/reflection_lattice.py", "_canon_congruence"), ("agentsg/semi_invariants.py", "SemiInvariant"), ("agentsg/harker.py", "HarkerConstraint")],
  [("USES", "modular_arithmetic")], wiki="Modular arithmetic")
c("modular_arithmetic", "Modular arithmetic (mod 1 and mod m)", "mathematics",
  "Translations reduced modulo the lattice (mod 1) and integer congruences modulo m.",
  [("agentsg/linalg.py", "frac_mod1"), ("agentsg/symmetry_op.py", "")],
  [], wiki="Modular arithmetic")
c("gcd_lcm", "Greatest common divisor / least common multiple", "mathematics",
  "Used to clear denominators, remove content from integer vectors and choose moduli.",
  [("agentsg/lattice_symmetry.py", "_igcd3"), ("agentsg/reflection_lattice.py", "_canon_congruence")],
  [("PART_OF", "modular_arithmetic")], wiki="Greatest common divisor")
c("matrix_order", "Order of a matrix / group element", "mathematics",
  "Smallest n with Wⁿ = I; 1, 2, 3, 4 or 6 for crystallographic rotations.",
  [("agentsg/reflection_lattice.py", "_matrix_order"), ("agentsg/lattice_symmetry.py", "_mat_pow_in_set")],
  [("RELATED_TO", "crystallographic_restriction")], iucr="Order (group theory)", wiki="Order (group theory)")

# ---------------------------------------------------- drawn symmetry elements
c("ita_graphical_symbols", "ITA graphical symbols for symmetry elements", "crystallography",
  "The International Tables glyphs for axes and planes: polygon or lens for the rotation order, hooks for the screw index, and a styled line for a plane, drawn from the classified operations.",
  [("agentsg/cell/diagrams.py", "draw_axis_symbol"), ("agentsg/cell/diagrams.py", "draw_plane_symbol")],
  [("PART_OF", "ita_diagrams"), ("USES", "symmetry_element"), ("USES", "screw_axis"), ("USES", "glide_plane")],
  iucr="Symmetry element", wiki="Symmetry element")
c("ita_height_label", "ITA height and fraction labels", "crystallography",
  "Printed heights of general-position images (a fraction of the projected coordinate, or a spelled-out coordinate for an inclined axis) and the single-glyph fractions used on the plate.",
  [("agentsg/cell/diagrams.py", "height_label"), ("agentsg/cell/diagrams.py", "frac_label")],
  [("PART_OF", "ita_diagrams"), ("USES", "general_position")],
  wiki="International Tables for Crystallography")
c("coordination_polyhedron", "Coordination polyhedron on a cubic plate", "crystallography",
  "On a cubic plate the general-position images are joined into orthogonal polyhedra over the closed cell, one circle at each corner, instead of a flat height-labelled orbit.",
  [("agentsg/cell/diagrams.py", "_draw_cubic_polyhedra"), ("agentsg/cell/diagrams.py", "_hull_polygon")],
  [("PART_OF", "ita_diagrams"), ("USES", "general_position"), ("USES", "convex_hull")],
  iucr="Symmetry element")
c("projection_convention", "ITA projection convention", "crystallography",
  "The projected cell outline and which axis is down the page: unique axis for a monoclinic group, otherwise c, taken from the operations rather than a stored plate.",
  [("agentsg/cell/diagrams.py", "cell_frame"), ("agentsg/serve/handlers.py", "default_projection")],
  [("PART_OF", "ita_diagrams"), ("USES", "crystal_system")],
  wiki="International Tables for Crystallography",
  aliases=["unique axis", "monoclinic unique axis"])

def _glide(cid, label, definition):
    c(cid, label, "crystallography", definition,
      [("agentsg/cell/symmetry_elements.py", "_glide_symbol"), ("agentsg/hall.py", "_parse_generator")],
      [("SPECIALIZES", "glide_plane")],
      wiki="Glide plane")

_glide("glide_a", "a-glide",
       "A glide whose intrinsic translation is half the a edge. _glide_symbol returns the letter a for that reduced translation.")
_glide("glide_b", "b-glide",
       "A glide whose intrinsic translation is half the b edge. _glide_symbol returns the letter b for that reduced translation.")
_glide("glide_c", "c-glide",
       "A glide whose intrinsic translation is half the c edge. _glide_symbol returns the letter c for that reduced translation.")
_glide("glide_n", "n-glide",
       "A diagonal glide of half a face or body diagonal on a coordinate plane. _glide_symbol returns n for that case, and g for the hexagonal and some tetragonal diagonals it does not call n.")
_glide("glide_d", "d-glide",
       "A diamond glide: a quarter of a face or body diagonal. _glide_symbol returns d when four times the intrinsic translation is integral and two times is not.")
_glide("glide_e", "e-glide",
       "The double glide IUCr denotes e, used in the 2016 short symbols of five space groups. _glide_symbol never returns e; those and other leftover diagonal glides come back as g, and the Hermann-Mauguin lookup keeps the pre-2016 names.")

def _screw(cid, label, definition):
    c(cid, label, "crystallography", definition,
      [("agentsg/cell/symmetry_elements.py", "_screw_index"), ("agentsg/hall.py", "_parse_generator")],
      [("SPECIALIZES", "screw_axis")],
      wiki="Screw axis")

_screw("screw_21", "2₁ screw axis",
       "A twofold rotation plus half the axis period. _screw_index returns 1 when the intrinsic translation is half the period; the Hall parser reads a screw digit t as translation t/N.")
_screw("screw_31", "3₁ screw axis",
       "A threefold rotation plus one third of the axis period. The screw index is 1.")
_screw("screw_32", "3₂ screw axis",
       "A threefold rotation plus two thirds of the axis period, the opposite hand of 3₁. The screw index is 2. Intensities are not used to choose between them.")
_screw("screw_41", "4₁ screw axis",
       "A fourfold rotation plus one quarter of the axis period. The screw index is 1.")
_screw("screw_42", "4₂ screw axis",
       "A fourfold rotation plus half the axis period. The screw index is 2.")
_screw("screw_43", "4₃ screw axis",
       "A fourfold rotation plus three quarters of the axis period, the opposite hand of 4₁. The screw index is 3. Intensities are not used to choose between them.")
_screw("screw_6", "Hexagonal screws 6₁–6₅",
       "Sixfold screws with index 1 through 5: intrinsic translation k/6 of the axis period. _screw_index returns that k; 6₁ and 6₅, and 6₂ and 6₄, are opposite hands.")

c("chiral_space_group", "Chiral space group (Sohncke group)", "crystallography",
  "A space group with no inversion, mirror, glide, or rotoinversion, so it is not identical to its mirror image. Enantiomorphic pairs such as P4₁ and P4₃ are drawn with a comma on the opposite hand. The plate records handedness; intensities are not used to choose which member of a pair.",
  [("agentsg/cell/diagrams.py", "_draw_position_mark")],
  [("USES", "screw_axis"), ("RELATED_TO", "inversion_centre"), ("RELATED_TO", "glide_plane")],
  iucr="Chiral space group", wiki="Sohncke groups", aliases=["Sohncke groups", "chirality", "enantiomorphic pair"])
c("symmorphic_space_group", "Symmorphic space group", "crystallography",
  "A space group that has an origin at which every operation's intrinsic translation vanishes, so the group is a semidirect product of a point group and the translation lattice. agentsg sees a zero intrinsic part on an operation; it does not emit a symmorphic flag.",
  [("agentsg/hall.py", ""), ("agentsg/cell/symmetry_elements.py", "classify_element")],
  [("SPECIALIZES", "space_group"), ("CONTRASTS_WITH", "nonsymmorphic_space_group"), ("USES", "intrinsic_translation")],
  iucr="Symmorphic space groups")
c("nonsymmorphic_space_group", "Non-symmorphic space group", "crystallography",
  "A space group whose operations include a screw or a glide that cannot all be cleared by an origin shift. agentsg classifies a nonzero intrinsic part; it does not label the group non-symmorphic.",
  [("agentsg/cell/symmetry_elements.py", "_reduce_intrinsic"), ("agentsg/hall.py", "_parse_generator")],
  [("SPECIALIZES", "space_group"), ("CONTRASTS_WITH", "symmorphic_space_group"), ("USES", "screw_axis"), ("USES", "glide_plane")],
  iucr="Symmorphic space groups")
c("geometric_crystal_class", "Geometric crystal class", "crystallography",
  "A conjugacy class of crystallographic point groups in O(3). agentsg derives the point group as the set of rotation parts W; it does not tabulate the 32 geometric classes by name.",
  [("agentsg/group.py", "point_group")],
  [("RELATED_TO", "point_group"), ("RELATED_TO", "crystal_system")],
  iucr="Geometric crystal class", wiki="Crystallographic point group",
  aliases=["crystal class"])
c("arithmetic_crystal_class", "Arithmetic crystal class", "crystallography",
  "A conjugacy class of finite subgroups of GL(3,Z): the point group as integer matrices on a lattice basis. The rotation parts agentsg stores are those integer matrices. No arithmetic-class number is tabulated.",
  [("agentsg/group.py", "point_group"), ("agentsg/symmetry_op.py", "SymmetryOp")],
  [("RELATED_TO", "geometric_crystal_class"), ("RELATED_TO", "unimodular_matrix"), ("USES", "point_group")],
  iucr="Arithmetic crystal class", wiki="Crystallographic point group")
c("bravais_class", "Bravais class", "crystallography",
  "An arithmetic crystal class together with the centring type of the lattice: the classification of which the 14 Bravais lattices are the symmorphic members. agentsg distinguishes holohedry and centring letter; it does not emit a Bravais-class symbol.",
  [("agentsg/lattice_symmetry.py", "lattice_symmetry"), ("agentsg/cell/primitive.py", "lattice_letter")],
  [("USES", "bravais_lattice"), ("USES", "holohedry"), ("USES", "lattice_centring"), ("RELATED_TO", "arithmetic_crystal_class")],
  iucr="Bravais class")
c("wilson_statistics", "Wilson statistics (centric and acentric intensities)", "crystallography",
  "The expected distributions of diffracted intensity for centric and acentric reflections, and the epsilon enhancement where the stabilizer of h is larger than 1. agentsg reports the centric phase restriction and the epsilon factor. It does not fit a Wilson plot.",
  [("agentsg/reflections.py", "epsilon_factor"), ("agentsg/group.py", "is_reflection_centric")],
  [("USES", "epsilon_factor"), ("USES", "centric_reflection")],
  iucr="Centric and acentric distribution", wiki="Wilson plot")
c("tolerance_gated_matching", "Tolerance-gated cell matching", "algorithm",
  "Two cells match when edges and angles stay inside a named tolerance: metric automorphisms, twin laws, and geometric reindexing use 2 percent and 2 degrees; cell comparison uses 3 percent and 5 degrees; a PDB change of basis uses 0.75 percent and 0.5 degrees, and is accepted at a relative metric residual of 1e-6. This is a numeric gate. It is not the exact certificate given by equality over the Selling closure.",
  [("agentsg/cell/reindex.py", "_cell_close"), ("agentsg/cell/compare.py", "_deviations"), ("agentsg/cell/selling_cob.py", "_param_residual"), ("agentsg/cell/canonical.py", "calibrate_verify_tol")],
  [("CONTRASTS_WITH", "selling_closure"), ("USES", "metric_tensor")],
  wiki="Lattice reduction")
c("site_symmetry_stratum", "Site-symmetry stratum (direct-space fixed set)", "crystallography",
  "The locus of points in the unit cell with a given site-symmetry stabilizer: the intersection of the fixed-point loci of those operations. This is the direct-space counterpart of a reflection stratum. The two are analogous stratifications by stabilizer, not a lattice dual of each other.",
  [("agentsg/wyckoff.py", "fixed_locus"), ("agentsg/wyckoff.py", "site_symmetry_ops")],
  [("USES", "site_symmetry"), ("USES", "fixed_point_locus"), ("RELATED_TO", "reflection_stratum"), ("RELATED_TO", "wyckoff_position")],
  iucr="Wyckoff position", wiki="Wyckoff positions")
c("twinning_pseudomerohedry", "Twinning by pseudomerohedry", "crystallography",
  "Twin laws from a lattice holohedry that only approximately contains the crystal Laue group, so the obliquity is small but not zero. twin_laws and the ambiguity cosets enumerate these operators when the cell is within the length and angle tolerances. The obliquity itself is not reported.",
  [("agentsg/cell/reindex.py", "twin_laws"), ("agentsg/cell/ambiguity.py", "reindexing_ambiguity_operators")],
  [("SPECIALIZES", "twinning_merohedry"), ("USES", "tolerance_gated_matching"), ("USES", "coset")],
  iucr="Twinning by pseudomerohedry", wiki="Twinning (crystallography)")
c("twinning_reticular_merohedry", "Twinning by reticular merohedry", "crystallography",
  "Twinning in which only a sublattice is restored by the twin operation. Dictionary context. twin_laws enumerates cosets of the Laue group in the lattice holohedry; it does not build a twin lattice or a twin index.",
  [("agentsg/cell/reindex.py", "twin_laws")],
  [("RELATED_TO", "twinning_merohedry"), ("RELATED_TO", "sublattice")],
  iucr="Twinning by reticular merohedry", wiki="Twinning (crystallography)")
c("twin_index", "Twin index", "crystallography",
  "The index of the twin lattice in the crystal lattice. Dictionary context. The code does not compute it.",
  [("agentsg/cell/reindex.py", "twin_laws")],
  [("RELATED_TO", "twinning_reticular_merohedry"), ("RELATED_TO", "sublattice")],
  iucr="Twin index", wiki="Twinning (crystallography)")
c("twin_obliquity", "Twin obliquity", "crystallography",
  "The angle by which a twin operation fails to be an exact symmetry of the lattice. Dictionary context. Tolerances gate which operators are returned; the angle is not reported.",
  [("agentsg/cell/reindex.py", "_cell_close")],
  [("RELATED_TO", "twinning_pseudomerohedry"), ("RELATED_TO", "tolerance_gated_matching")],
  iucr="Twin obliquity", wiki="Twinning (crystallography)")
c("twin_lattice", "Twin lattice", "crystallography",
  "The sublattice restored by every twin operation of a twin. Dictionary context. The code returns twin-law matrices, not that sublattice.",
  [("agentsg/cell/reindex.py", "twin_laws")],
  [("RELATED_TO", "twinning_reticular_merohedry"), ("RELATED_TO", "sublattice")],
  iucr="Twin lattice", wiki="Twinning (crystallography)")
c("crystfel_stream", "CrystFEL stream", "crystallography",
  "A CrystFEL .stream file: one indexed unit cell, and optionally the reciprocal-lattice orientation, for each crystal in a serial experiment. Cells are converted from nanometres to angstroms.",
  [("agentsg/cell/crystfel_stream.py", ""), ("agentsg/cell/crystfel_stream.py", "parse_stream")],
  [("USES", "indexed_cell_distribution")],
  wiki="Serial femtosecond crystallography")
c("indexed_cell_distribution", "Indexed cell distribution", "crystallography",
  "The set of per-crystal unit cells written by the indexer, before merging averages them. parse_stream yields one cell dict per indexed crystal.",
  [("agentsg/cell/crystfel_stream.py", "parse_stream")],
  [("PART_OF", "crystfel_stream"), ("RELATED_TO", "indexing_ambiguity")],
  wiki="Serial femtosecond crystallography")
c("reference_orbit", "Reference orbit of Selling superbases", "crystallography",
  "One Selling reduction of a deposited cell, the deposited-to-reduced change of basis, and the labeled closure of obtuse superbases used to match a neighbour.",
  [("agentsg/cell/selling_cob.py", "reference_orbit"), ("agentsg/cell/selling_cob.py", "ReferenceOrbit")],
  [("USES", "selling_closure"), ("USES", "change_of_basis"), ("PART_OF", "reindexing")],
  wiki="Lattice reduction")
c("pdb_lattice_search", "PDB lattice search pipeline", "algorithm",
  "The same path for a query and for each stored PDB cell: conventional cell, primitive cell from the centring letter, Selling reduction, Kurlin root form, then a KD-tree radius query on the six root components.",
  [("agentsg/cell/pdb_server.py", ""), ("agentsg/cell/celldb.py", ""), ("agentsg/cell/rootindex.py", "")],
  [("USES", "conventional_cell"), ("USES", "primitive_cell"), ("USES", "delaunay_selling_reduction"), ("USES", "kurlin_root_form"), ("USES", "kd_tree")],
  refs=["Kurlin, V. Mathematics of 2-dimensional and 3-dimensional lattices, arXiv:2201.10543"],
  wiki="K-d tree")
c("k_subgroup_iia", "Klassengleiche subgroup of type IIa", "crystallography",
  "A klassengleiche subgroup on the same conventional cell: the point group stays, and a point-group-invariant proper subset of the centring translations is dropped. Primitive groups have no IIa edge. This is derived from the operators, not from an ITA A1 maximality table.",
  [("agentsg/subgroups.py", "_k_iia_edges")],
  [("SPECIALIZES", "k_subgroup"), ("CONTRASTS_WITH", "k_subgroup_iib"), ("USES", "lattice_centring")],
  iucr="Klassengleiche subgroups", aliases=["IIa", "type IIa"])
c("k_subgroup_iib", "Klassengleiche subgroup of type IIb", "crystallography",
  "A klassengleiche subgroup of a primitive group from an invariant index-2 or index-3 sublattice of Z³. The point group is kept and the cell is enlarged. Infinite isomorphic series beyond those two indices are not enumerated.",
  [("agentsg/subgroups.py", "_k_iib_edges")],
  [("SPECIALIZES", "k_subgroup"), ("CONTRASTS_WITH", "k_subgroup_iia"), ("USES", "sublattice")],
  iucr="Klassengleiche subgroups", aliases=["IIb", "type IIb"])
c("numeric_gate", "Named numeric gate", "algorithm",
  "The angle, length, conorm, and residual cutoffs for Le Page symmetry, cell comparison, reindexing, and a PDB change of basis. Each number is a named constant in tolerances.py. Le Page uses 3 degrees and 2 percent. Metric automorphisms use 2 percent and 2 degrees. Cell comparison uses 3 percent and 5 degrees. A PDB match uses 0.75 percent and 0.5 degrees, with zero conorms inside 0.05 degrees and a relative boundary of 0.001.",
  [("agentsg/tolerances.py", "")],
  [("USES", "tolerance_gated_matching"), ("USES", "lattice_symmetry_determination"),
   ("USES", "selling_closure"), ("USES", "reference_orbit"),
   ("USES", "kurlin_root_form"), ("USES", "floating_point_tolerance")],
  wiki="Machine epsilon",
  aliases=["angle tolerance", "length tolerance", "angle sigma", "boundary_rel", "numeric tolerance"])
c("conorm_noise_floor", "Closure-invariant conorm noise floor", "algorithm",
  "One floor s = c · σ_θ · T applied to every Selling conorm, where T is the sum of the six conorms. Selling flips permute the conorms and leave T fixed, so the floor is the same on every member of the closure. It is not a separate floor on each pair.",
  [("agentsg/cell/rootform.py", "noise_floor"), ("agentsg/cell/rootform.py", "pair_noise_scales")],
  [("USES", "obtuse_superbase"), ("RELATED_TO", "kurlin_root_form"), ("RELATED_TO", "floating_point_tolerance")],
  refs=["Kurlin, V. Mathematics of 2-dimensional and 3-dimensional lattices, arXiv:2201.10543"],
  aliases=["noise floor", "conorm floor"])

CONCEPTS = C

# Secondary code anchors: further routines that embody an already-catalogued concept.
EXTRA_ANCHORS = {
    "allowed_origins": [("agentsg/semi_invariants.py", "is_allowed_origin"), ("agentsg/semi_invariants.py", "n_alternative_origins"), ("agentsg/semi_invariants.py", "origin_lattice"), ("agentsg/semi_invariants.py", "_nicest_generator")],
    "structure_seminvariant": [("agentsg/semi_invariants.py", "_congruence")],
    "harker_section": [("agentsg/harker.py", "HarkerLocus"), ("agentsg/harker.py", "HarkerConstraint"), ("agentsg/harker.py", "_normalize_constraint")],
    "hall_symbol": [("agentsg/hall.py", "ops_from_hall"), ("agentsg/hall.py", "_parse_generator"), ("agentsg/hall.py", "_parse_hall_body"), ("agentsg/ita_settings.py", "hall_for_ops")],
    "space_group_identification": [("agentsg/identify.py", "_identify_by_origin"), ("agentsg/identify.py", "_identify_by_cob"), ("agentsg/identify.py", "hall_from_ops"), ("agentsg/identify.py", "IdentifyResult")],
    "k_subgroup": [("agentsg/subgroups.py", "_kernel_basis"), ("agentsg/subgroups.py", "_closed_primitive_sections"), ("agentsg/subgroups.py", "_invariant_mod_p")],
    "subgroup": [("agentsg/subgroups.py", "subgroup_graph"), ("agentsg/subgroups.py", "SubgroupEdge"), ("agentsg/subgroups.py", "_centering_subgroups"), ("agentsg/subgroups.py", "_is_T_invariant")],
    "lattice_symmetry_determination": [("agentsg/lattice_symmetry.py", "_reciprocal_axis"), ("agentsg/lattice_symmetry.py", "LatticeSymmetry"), ("agentsg/lattice_symmetry.py", "evaluate_two_folds"), ("agentsg/lattice_symmetry.py", "TwoFoldScore"),
        ("agentsg/tolerances.py", "LE_PAGE_MAX_DELTA_DEG"), ("agentsg/tolerances.py", "LE_PAGE_LENGTH_TOL_PCT")],
    "reflection_conditions": [("agentsg/reflection_lattice.py", "present_lattices"), ("agentsg/reflection_lattice.py", "is_absent_by_lattice"), ("agentsg/reflection_lattice.py", "stratum_conditions"), ("agentsg/reflection_lattice.py", "format_conditions"), ("agentsg/reflections.py", "reflection_conditions_grid")],
    "linear_congruence": [("agentsg/reflection_lattice.py", "_congruence_index"), ("agentsg/reflections.py", "_reduce_constraint")],
    "equivalent_reflections": [("agentsg/reflections.py", "are_equivalent_reflections")],
    "extended_setting_notation": [("agentsg/setting.py", "format_cob"), ("agentsg/setting.py", "_lattice_coset_ops"), ("agentsg/setting.py", "parse_setting"), ("agentsg/setting.py", "SpaceGroupSetting")],
    "site_symmetry": [("agentsg/wyckoff.py", "site_symmetry_order")],
    "reindexing": [("agentsg/cell/selling_cob.py", "match_operators"), ("agentsg/cell/selling_cob.py", "deposited_to_reduced"), ("agentsg/cell/selling_cob.py", "selling_matrix"), ("agentsg/cell/selling_cob.py", "reference_orbit"),
                   ("agentsg/cell/canonical.py", "canonical_superbase"), ("agentsg/cell/canonical.py", "reindexing_operator_via_canonical"), ("agentsg/cell/canonical.py", "best_reindex_with_residual"), ("agentsg/cell/canonical.py", "_reindex_coset"), ("agentsg/cell/canonical.py", "reindex"), ("agentsg/cell/canonical.py", "calibrate_verify_tol"),
                   ("agentsg/cell/reindex.py", "reindexing_operator"), ("agentsg/cell/reindex.py", "_find_base_reindex"), ("agentsg/cell/reindex.py", "_lattice_symmetry_matrices")],
    "lattice_centring": [("agentsg/cell/diagrams.py", "_centring_translations"), ("agentsg/cell/diagrams.py", "_draw_centring_markers")],
    "primitive_cell": [("agentsg/cell/primitive.py", "primitive_cell"), ("agentsg/cell/primitive.py", "lattice_letter"), ("agentsg/cell/rootindex.py", "_primitive_for_roots"), ("agentsg/cell/celldb.py", "_primitive_for_roots"), ("agentsg/cell/symmetry_elements.py", "primitive_lattice")],
    "neartree": [("agentsg/cell/neartree.py", "build_neartree")],
    "kd_tree": [("agentsg/cell/rootindex.py", "build_root_index"), ("agentsg/cell/neartree.py", "lattice_index")],
    "obtuse_superbase": [("agentsg/cell/rootform.py", "vonorms"), ("agentsg/cell/rootform.py", "vonorms_from_conorms"), ("agentsg/cell/rootform.py", "conorm_sum"), ("agentsg/cell/rootform.py", "_superbase_lengths"), ("agentsg/cell/selling_closure.py", "_conorm_sum")],
    "kurlin_root_form": [("agentsg/cell/rootform.py", "root_invariant"), ("agentsg/cell/rootform.py", "root_distance"), ("agentsg/cell/rootform.py", "sorted_conorm_key"), ("agentsg/cell/rootform.py", "sorted_vonorm_key"), ("agentsg/cell/rootform.py", "sorted_concat_key"), ("agentsg/cell/rootform.py", "sorted_root_distance"), ("agentsg/cell/canonical.py", "_rootmat"), ("agentsg/cell/canonical.py", "_roots_close"),
        ("agentsg/tolerances.py", "ROOT_SNAP_REL"), ("agentsg/tolerances.py", "ROOT_SNAP_DECIMALS"), ("agentsg/tolerances.py", "ROOT_STABILIZE_KAPPA"), ("agentsg/tolerances.py", "SYMMETRY_CUTOFF_Z")],
    "similarity_invariant": [("agentsg/cell/rootform.py", "root_distance_to_volume_ratio"), ("agentsg/cell/rootform.py", "volume_ratio_to_root_distance"), ("agentsg/cell/rootform.py", "root_cutoff_for_edge_tolerance"), ("agentsg/cell/rootform.py", "symmetry_cutoff"), ("agentsg/cell/rootform.py", "similarity_distance")],
    "niggli_reduction": [("agentsg/cell/reduction.py", "_check_cob_invariant"), ("agentsg/cell/reduction.py", "_sign_matrix"),
        ("agentsg/tolerances.py", "REL_EPS"), ("agentsg/tolerances.py", "NIGGLI_COB_TOL_REL")],
    "metric_tensor": [("agentsg/cell/reduction.py", "_transform_metric"), ("agentsg/cell/g6.py", "_transform_metric"), ("agentsg/cell/canonical.py", "_transform_metric_int"), ("agentsg/cell/metric.py", "params_from_metric"), ("agentsg/cell/primitive.py", "_cell_from_metric"), ("agentsg/cell/sublattice.py", "apply_to_cell"), ("agentsg/lattice_symmetry.py", "_metric_tensor")],
    "selling_settings": [("agentsg/cell/selling_settings.py", "SellingSetting"), ("agentsg/cell/selling_settings.py", "_reduction_cob"), ("agentsg/cell/selling_settings.py", "distinct_settings")],
    "symmetry_element": [("agentsg/cell/symmetry_elements.py", "Lattice"), ("agentsg/cell/symmetry_elements.py", "_canonical_line"), ("agentsg/cell/symmetry_elements.py", "_canonical_plane"), ("agentsg/cell/diagrams.py", "classify_element"), ("agentsg/cell/diagrams.py", "classify_space_group")],
    "intrinsic_translation": [("agentsg/cell/symmetry_elements.py", "_shortest_along")],
    "deformation_manifold": [("agentsg/cell/manifold.py", "_metric_degeneracies"), ("agentsg/cell/manifold.py", "symmetry_junctions"), ("agentsg/cell/manifold.py", "select_landmarks"),
        ("agentsg/tolerances.py", "MANIFOLD_HOP_VERIFY_REL"), ("agentsg/tolerances.py", "MANIFOLD_PATH_VERIFY_REL")],
    "selling_closure": [("agentsg/cell/canonical.py", "superbase_variants"), ("agentsg/cell/canonical.py", "_closure_for_match"), ("agentsg/cell/selling_closure.py", "selling_closure_representatives"), ("agentsg/cell/selling_closure.py", "closure_class_count"), ("agentsg/cell/selling_closure.py", "_class_seeds"), ("agentsg/cell/selling_closure.py", "_v5_even_reps"),
        ("agentsg/tolerances.py", "CONORM_TOL_REL"), ("agentsg/tolerances.py", "ZERO_NOISE_MULT"), ("agentsg/tolerances.py", "COB_ANGLE_SIGMA_DEG"), ("agentsg/tolerances.py", "BOUNDARY_REL"), ("agentsg/tolerances.py", "SUPERBASE_MAX_VARIANTS"), ("agentsg/tolerances.py", "REINDEX_BOUNDARY_REL"), ("agentsg/tolerances.py", "REINDEX_BAND_REL")],
    "voronoi_type": [("agentsg/cell/selling_closure.py", "voronoi_type"), ("agentsg/cell/selling_closure.py", "_zero_pairs"),
        ("agentsg/tolerances.py", "CONORM_TOL_REL"), ("agentsg/tolerances.py", "ZERO_NOISE_MULT"), ("agentsg/tolerances.py", "COB_ANGLE_SIGMA_DEG")],
    "permutation_group_s4": [("agentsg/cell/selling_group.py", "permutation_cob"), ("agentsg/cell/selling_group.py", "selling_generators"), ("agentsg/cell/selling_group.py", "selling_generators_S4"), ("agentsg/cell/selling_group.py", "selling_group_S4"), ("agentsg/cell/selling_group.py", "_perm_matrix")],
    "indexing_ambiguity": [("agentsg/cell/ambiguity.py", "ambiguity_index"), ("agentsg/cell/ambiguity.py", "AmbiguityResolution"), ("agentsg/cell/ambiguity.py", "GeometricOperator"), ("agentsg/cell/ambiguity.py", "surface_geometric_operators"), ("agentsg/cell/ambiguity.py", "apply_to_hkl_batch"), ("agentsg/cell/ambiguity.py", "_cached_ambiguity")],
    "matrix_order": [("agentsg/lattice_symmetry.py", "_mat_pow_in_set")],
    "modular_arithmetic": [("agentsg/wyckoff.py", "_vec_mod1"), ("agentsg/linalg.py", "Vector3"), ("agentsg/asu.py", "_wrap_half")],
    "general_position": [("agentsg/cell/diagrams.py", "general_position_multiplicity"), ("agentsg/cell/diagrams.py", "general_position_images")],
    "inversion_centre": [("agentsg/cell/diagrams.py", "_draw_axis_inversion")],
    "reciprocal_asu": [("agentsg/cell/ambiguity.py", "_map_to_asu_lexmax"), ("agentsg/cell/ambiguity.py", "_merge_to_asu")],
    "unimodular_matrix": [("agentsg/cell/canonical.py", "_int_or_none"), ("agentsg/cell/canonical.py", "_inv3_frac"), ("agentsg/cell/reindex.py", "_int_inv")],
    "coset": [("agentsg/subgroups.py", "_index2_subgroups")],
    "reflection_stratum": [("agentsg/reflections.py", "_generic_members"), ("agentsg/reflections.py", "_constraints_from_ops"), ("agentsg/reflection_lattice.py", "_std_classes"), ("agentsg/reflection_lattice.py", "_transform_basis")],
    "sublattice": [("agentsg/cell/sublattice.py", "diagonal_triples"), ("agentsg/cell/compare.py", "CellMatch")],
    "euclidean_distance": [("agentsg/cell/g6.py", "g6_distance"), ("agentsg/cell/g6.py", "s6_distance"), ("agentsg/cell/rootform.py", "sorted_conorm_distance")],
    "symmetry_deficiency": [("agentsg/cell/g6.py", "symmetry_deficiency_spectrum"), ("agentsg/cell/g6.py", "kurlin_deficiency_spectrum")],
    "orthogonalization": [("agentsg/cell/metric.py", "_inv3")],
    "determinant": [("agentsg/cell/metric.py", "_det3")],
    "eigenvalue_symmetric": [("agentsg/cell/manifold.py", "DeformationManifold")],
    "gaussian_elimination": [("agentsg/rational_solve.py", "")],
    "wyckoff_position": [("agentsg/wyckoff.py", "site_symmetry_order"), ("agentsg/wyckoff.py", "fixed_locus")],
    "laue_class": [("agentsg/asu.py", "_laue_ops"), ("agentsg/cell/ambiguity.py", "_laue_rows")],
    "svd_pca": [("agentsg/serve/scatter.py", "project_query"), ("agentsg/serve/scatter.py", "scatter_payload")],
    "change_of_basis": [("agentsg/cell/selling_cob.py", "cob_column_values"), ("agentsg/cell/selling_cob.py", "parse_cob_columns"), ("agentsg/cell/selling_cob.py", "cob_xyz"), ("agentsg/cell/selling_cob.py", "annotate_search_hits")],
    "screw_axis": [("agentsg/cell/symmetry_elements.py", "_screw_index"), ("agentsg/hall.py", "_parse_generator")],
    "glide_plane": [("agentsg/hall.py", "_parse_generator")],
    "space_group": [("agentsg/space_groups.py", "SpaceGroup")],
    "ita_setting": [("agentsg/ita_settings.py", "settings_for_number"), ("agentsg/ita_settings.py", "match_ops")],
    "g6_s6_embedding": [("agentsg/cell/g6.py", "g6"), ("agentsg/cell/g6.py", "s6")],
    "pdb_lattice_search": [("agentsg/cell/pdb_server.py", "search_compatible")],
    "crystfel_stream": [("agentsg/cell/crystfel_stream.py", "read_cells")],
    "indexed_cell_distribution": [("agentsg/cell/crystfel_stream.py", "stream_summary")],
    "ita_diagrams": [("agentsg/cell/diagrams.py", "ita_plate")],
    "ita_graphical_symbols": [("agentsg/cell/diagrams.py", "draw_parallel_plane_symbol"), ("agentsg/cell/diagrams.py", "symbol_legend"), ("agentsg/cell/diagrams.py", "element_legend")],
    "asymmetric_unit": [("agentsg/asu.py", "AxisBound")],
    "dirichlet_domain": [("agentsg/asu.py", "OptimizedAsu")],
    "projection_convention": [("agentsg/serve/handlers.py", "_unique_axis")],
    "floating_point_tolerance": [("agentsg/tolerances.py", "REL_EPS"), ("agentsg/tolerances.py", "METRIC_INVARIANT_TOL")],
    "tolerance_gated_matching": [("agentsg/tolerances.py", "METRIC_LENGTH_TOL_PCT"), ("agentsg/tolerances.py", "METRIC_ANGLE_TOL_DEG"), ("agentsg/tolerances.py", "COMPARE_LENGTH_TOL_PCT"), ("agentsg/tolerances.py", "COMPARE_ANGLE_TOL_DEG"), ("agentsg/tolerances.py", "COB_LENGTH_TOL_PCT"), ("agentsg/tolerances.py", "COB_ANGLE_TOL_DEG"), ("agentsg/tolerances.py", "VERIFY_REL"), ("agentsg/tolerances.py", "VOLUME_FRAC"), ("agentsg/tolerances.py", "METRIC_SYM_RESIDUAL")],
    "conorm_noise_floor": [("agentsg/tolerances.py", "ZERO_NOISE_MULT"), ("agentsg/tolerances.py", "COB_ANGLE_SIGMA_DEG")],
    "reference_orbit": [("agentsg/tolerances.py", "COB_ANGLE_SIGMA_DEG"), ("agentsg/tolerances.py", "BOUNDARY_REL"), ("agentsg/tolerances.py", "COB_LENGTH_TOL_PCT"), ("agentsg/tolerances.py", "COB_ANGLE_TOL_DEG"), ("agentsg/tolerances.py", "VERIFY_REL")],
    "numeric_gate": [
        ("agentsg/tolerances.py", "REL_EPS"), ("agentsg/tolerances.py", "CONORM_TOL_REL"),
        ("agentsg/tolerances.py", "ZERO_NOISE_MULT"), ("agentsg/tolerances.py", "LE_PAGE_MAX_DELTA_DEG"),
        ("agentsg/tolerances.py", "LE_PAGE_LENGTH_TOL_PCT"), ("agentsg/tolerances.py", "METRIC_LENGTH_TOL_PCT"),
        ("agentsg/tolerances.py", "METRIC_ANGLE_TOL_DEG"), ("agentsg/tolerances.py", "METRIC_INVARIANT_TOL"),
        ("agentsg/tolerances.py", "METRIC_SYM_RESIDUAL"), ("agentsg/tolerances.py", "COMPARE_LENGTH_TOL_PCT"),
        ("agentsg/tolerances.py", "COMPARE_ANGLE_TOL_DEG"), ("agentsg/tolerances.py", "VOLUME_FRAC"),
        ("agentsg/tolerances.py", "COB_LENGTH_TOL_PCT"), ("agentsg/tolerances.py", "COB_ANGLE_TOL_DEG"),
        ("agentsg/tolerances.py", "COB_ANGLE_SIGMA_DEG"), ("agentsg/tolerances.py", "BOUNDARY_REL"),
        ("agentsg/tolerances.py", "SUPERBASE_MAX_VARIANTS"), ("agentsg/tolerances.py", "REINDEX_BOUNDARY_REL"),
        ("agentsg/tolerances.py", "REINDEX_BAND_REL"), ("agentsg/tolerances.py", "VERIFY_REL"),
        ("agentsg/tolerances.py", "NIGGLI_COB_TOL_REL"), ("agentsg/tolerances.py", "ROOT_STABILIZE_KAPPA"),
        ("agentsg/tolerances.py", "SYMMETRY_CUTOFF_Z"), ("agentsg/tolerances.py", "ROOT_SNAP_REL"),
        ("agentsg/tolerances.py", "ROOT_SNAP_DECIMALS"), ("agentsg/tolerances.py", "MANIFOLD_HOP_VERIFY_REL"),
        ("agentsg/tolerances.py", "MANIFOLD_PATH_VERIFY_REL"),
    ],
}
for _cid, _extra in EXTRA_ANCHORS.items():
    for _c in C:
        if _c["id"] == _cid:
            for _a in _extra:
                if _a not in _c["anchors"]:
                    _c["anchors"].append(_a)
            break
    else:
        raise KeyError(_cid)
