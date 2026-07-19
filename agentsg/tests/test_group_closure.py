import pytest
from agentsg.generators import GENERATOR_TABLE
from agentsg.group import close_group, point_group, is_systematically_absent
from agentsg.linalg import Vector3


@pytest.mark.parametrize("name", list(GENERATOR_TABLE))
def test_group_order_matches_known_value(name):
    spec = GENERATOR_TABLE[name]
    ops = close_group(spec["generators"], spec["centering"])
    assert len(ops) == spec["expected_order"], (
        f"{name}: got order {len(ops)}, expected {spec['expected_order']}"
    )


def test_fm3m_point_group_is_order_48():
    spec = GENERATOR_TABLE["Fm-3m"]
    ops = close_group(spec["generators"], spec["centering"])
    assert len(point_group(ops)) == 48  # m-3m (Oh), from 3 generators only


def test_p21_reflection_condition_0k0_k_even():
    spec = GENERATOR_TABLE["P21"]
    ops = close_group(spec["generators"], spec["centering"])
    for k in range(8):
        expected_absent = (k % 2 == 1)
        assert is_systematically_absent(Vector3((0, k, 0)), ops) == expected_absent


def test_bad_generators_raise_instead_of_hanging():
    from agentsg.symmetry_op import SymmetryOp
    # A nonsensical generator set (e.g. det != +/-1 isn't representable as a
    # symmetry op at all, so instead test the max_order guard with something
    # that would blow past it if mis-specified):
    from agentsg.linalg import Vector3 as V
    with pytest.raises(RuntimeError):
        close_group(
            [SymmetryOp.from_xyz("-y,x-y,z")],
            centering_vectors=[V((0, 0, 0))],
            max_order=1,  # deliberately too small
        )
