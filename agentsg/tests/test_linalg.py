import random
from fractions import Fraction as Fr
from agentsg.linalg import Matrix3, Vector3, IDENTITY3, frac_mod1


def test_frac_mod1():
    assert frac_mod1(Fr(3, 2)) == Fr(1, 2)
    assert frac_mod1(Fr(-1, 2)) == Fr(1, 2)
    assert frac_mod1(Fr(2)) == Fr(0)
    assert frac_mod1(Fr(-1, 3)) == Fr(2, 3)


def test_inverse_permutation_matrix_is_transpose():
    M = Matrix3([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert M.inverse() == M.transpose()
    assert M @ M.inverse() == IDENTITY3


def test_inverse_random_round_trip():
    random.seed(1234)
    for _ in range(300):
        while True:
            rows = [[random.randint(-3, 3) for _ in range(3)] for _ in range(3)]
            M = Matrix3(rows)
            if M.det() != 0:
                break
        Minv = M.inverse()
        assert M @ Minv == IDENTITY3
        assert Minv @ M == IDENTITY3


def test_vector_ops():
    a = Vector3((Fr(1, 2), Fr(1, 3), 0))
    b = Vector3((Fr(1, 2), Fr(2, 3), 1))
    assert (a + b) == Vector3((1, 1, 1))
    assert (a + b).mod1() == Vector3((0, 0, 0))
