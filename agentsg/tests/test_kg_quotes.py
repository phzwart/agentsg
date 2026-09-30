"""Docstring sentences and Wikipedia-vs-IUCr page matching."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "concept_kg" / "builder"))

from quotes import first_sentence, is_anaphoric, needles_for, page_matches, pick_sentence  # noqa: E402


def test_first_sentence_joins_wrapped_lines_and_caps():
    doc = (
        "Among the generators ``k g`` (gcd(k, order) = 1) of one cyclic factor and\n"
        "    their coset representatives modulo ``L``, pick the one whose congruence\n"
        "    vector is smallest (fewest/smallest entries, non-negative preferred)."
    )
    sentence = first_sentence(doc)
    assert sentence.startswith("Among the generators")
    assert "their coset representatives" in sentence
    assert "\n" not in sentence
    assert sentence.endswith(".")
    long = "word " * 80
    capped = first_sentence(long)
    assert len(capped) <= 200
    assert capped.endswith("...")


def test_anaphoric_sentence_loses_to_one_that_names_the_concept():
    chosen = pick_sentence(
        [
            "In the second case, the lattice translations are maintained during the transition.",
            "A translationengleiche subgroup keeps the translation lattice of the parent group.",
        ],
        ["Translationengleiche (t) subgroup"],
    )
    assert chosen.startswith("A translationengleiche")
    assert is_anaphoric("In the second case, the lattice translations are maintained.")
    only = pick_sentence(
        ["In the second case, the lattice translations are maintained during the transition."],
        ["Translationengleiche (t) subgroup"],
    )
    assert only == ""


def test_joint_dictionary_page_quotes_the_sentence_that_names_both_terms():
    page = [
        "Subgroups of space groups are often used to describe the structures of a family of compounds which are closely related.",
        "Subgroups of space groups occur in two categories namely klassengleiche subgroups or translationengleiche subgroups.",
        "In the first case, the crystal class of the space group is maintained during the transition from the space group to the subgroup.",
        "In the second case, the lattice translations are maintained during the transition.",
    ]
    chosen = pick_sentence(page, needles_for("Translationengleiche (t) subgroup"))
    assert chosen.startswith("Subgroups of space groups occur in two categories")
    assert "translationengleiche" in chosen
    assert not is_anaphoric(chosen)


def test_wikipedia_space_group_page_is_off_topic_for_t_subgroup():
    assert page_matches("Space group", "Space group", [])
    assert not page_matches(
        "Space group", "Translationengleiche (t) subgroup", [],
    )
    assert not page_matches("Normalizer", "Allowed (permissible) origins", [
        "alternative origins", "permissible origins", "Cheshire origins",
    ])
    assert page_matches("Reciprocal lattice", "Reciprocal lattice / reciprocal space", [])
