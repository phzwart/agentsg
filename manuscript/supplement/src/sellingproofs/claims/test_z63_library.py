"""z63 — the implementation (agentsg.cell.strain_closure), checked by the library's own tests.

The library's tests live in agentsg/tests/test_strain_closure.py of the same repository; each
claim here calls one of them, so the library and the supplement share one copy of each check.
"""
from __future__ import annotations

import importlib.util

from sellingproofs import paths
from sellingproofs.registry import claim

_spec = importlib.util.spec_from_file_location(
    "agentsg_tests_strain_closure", paths.REPO / "agentsg" / "tests" / "test_strain_closure.py")
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)

_CLAIMS = [
    ("test_shipped_operator_data", ("main §4.2",), "exhaustive",
     "The library ships U with 6096 operators and largest entry 3, and the first and second layers of 127 and 2755 superbases, the first inside the second."),
    ("test_lepage_band", ("main §3.4", "main Table 5", "main §4.3", "main Lemma A7"), "exhaustive",
     "Le Page's 3 deg is a strain of 7.4 % (sqrt2 sin 3 deg) and a twofold price of 14.8 %; a price band of 7.4 % widens the reference at 7.7 %."),
    ("test_layers_and_sizes", ("main §3.3", "main Remark A8", "main §5.2"), "exhaustive",
     "The worked cell's widened set lies in the first layer and has 16 members; the needle (56 members) and the plate (52) need the second layer, where an uncertified first-layer lookup misses members."),
    ("test_adjacency_radius", ("main Table 8", "main Remark A8"), "exhaustive",
     "Table 8: r_1 of the needle is 5.57 % and that of the worked cell reaches the 25 % cap."),
    ("test_reindex_recovers_the_true_operator", ("main §4.1", "main §4.2"), "sampled",
     "Reindexing returns the true operator among the candidates for measured cells in random bases."),
    ("test_scale_fit_and_scale_invariance", ("main §4.2",), "sampled",
     "A 2 % change of the length scale is removed by the scale fit (price 0, scale 1.02), and the widened set does not depend on the scale."),
    ("test_sort_spectrum", ("SI §S6",), "exhaustive",
     "Sort spectra of a near-tetragonal cell (T5 at 1.06 %) and a near-cubic F cell (C3 at 0.43 %)."),
    ("test_indexing_classes_reproduce_the_merohedral_ambiguities", ("main §5.3", "main Table 9"), "exhaustive",
     "Table 9: |H_eps|, |L+| and the number of indexing classes for P4_32_12, P3, P3_121, P2_13, P2_12_12_1 and P1, and the anomalous splits."),
    ("test_price_spectrum_and_the_twofold_rule", ("main Proposition A10", "main §5.3"), "exhaustive",
     "The worked cell's self-maps cost 0, 1.55 % and 10.2 %, and a pseudo-twofold's price is twice its sort distance."),
    ("test_strain_from_esds_and_certified_spectrum", ("main §4.3", "SI §S6"), "exhaustive",
     "Standard uncertainties of 0.1 % on the edges and 0.05 deg on the angles of the worked cell give a strain of 0.41 %, and the certified sort spectrum of the needle gives r_1 = 5.57 %."),
    ("test_scale_bound_and_volume_check", ("main §4.2", "main Table 6"), "exhaustive",
     "The fitted scale is bounded by 10 % and primitive cells whose volumes differ beyond 0.9^3 to 1.1^3 are refused with an explanation."),
    ("test_proper_group_order", ("main §5.3", "main Table 9"), "exhaustive",
     "For tetragonal lysozyme in P4_32_12 the self-maps within the band number 16, of which 8 are proper, and the symmetry group has order 8."),
    ("test_centring_and_input_checks", ("main §4.2",), "exhaustive",
     "Centred references and settings (R 3 :R, R 3 :H) reindex consistently, and malformed input is refused."),
    ("test_ball_guard_and_universal_operators", ("main Remark A8", "main §4.2", "main §5.2"), "sampled",
     "A thin needle at a large band is refused quickly, and the 6096 column operators of U, used as B_R W B_N^-1, contain the true operator between two reduced cells."),
    ("test_rhombohedral_axes_ties_and_scale_bounds", ("main §4.2",), "exhaustive",
     "Rhombohedral axes are read as given, ties are ordered nearest the identity first, and the volume check matches the scale bound."),
    ("test_reindex_workflow_matches_the_direct_form", ("SI §S10", "main §4.2"), "sampled",
     "The reindexing workflow returns the same operators, with the same prices, as building the exact operator for every member and relabelling; on the worked cell 384 ordered bases leave 4 to price and 2 operators."),
    ("test_round6_api", ("main §4.2",), "exhaustive",
     "No space group means primitive in both reindexing and indexing classes; impossible R settings raise; an empty result warns."),
]


def _make(name, where, kind, statement):
    f = getattr(lib, name)

    def check():
        f()
    check.__doc__ = statement
    check.__name__ = check.__qualname__ = name
    check.__module__ = __name__
    return claim("z63", *where, kind=kind)(check)


for _c in _CLAIMS:
    globals()[_c[0]] = _make(*_c)

_missing = {n for n in dir(lib) if n.startswith("test_")} - {c[0] for c in _CLAIMS}
assert not _missing, f"library tests not registered as claims: {sorted(_missing)}"
