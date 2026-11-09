from __future__ import annotations

from itertools import combinations
from itertools import product

import numpy as np
import pytest

from .._acquisition import _solve_sdp_relaxation
from .._acquisition import _spin_to_binary
from .._acquisition import _to_spin_matrix
from .._acquisition import maximize_sdp
from .._model import n_coefficients
from .._model import quadratic_form
from .._model import second_order_features


def _all_binary(d: int) -> np.ndarray:
    return np.array(list(product([0.0, 1.0], repeat=d)))


def _random_alpha(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = n_coefficients(d)
    return rng.normal(size=p) * (rng.random(p) < 0.5)


def _brute_force_values(alpha: np.ndarray, d: int, penalty: float = 0.0) -> np.ndarray:
    X = _all_binary(d)
    return second_order_features(X) @ alpha - penalty * X.sum(axis=1)


def test_features_follow_documented_order() -> None:
    x = np.array([[1.0, 0.0, 1.0, 1.0]])
    pairs = [x[0, i] * x[0, j] for i, j in combinations(range(4), 2)]
    expected = np.concatenate([[1.0], x[0], pairs])
    np.testing.assert_array_equal(second_order_features(x)[0], expected)
    assert second_order_features(x).shape[1] == n_coefficients(4)


def test_quadratic_form_reproduces_model() -> None:
    d = 6
    alpha = _random_alpha(d, seed=0)
    constant, b, A = quadratic_form(alpha, d)
    np.testing.assert_array_equal(A, A.T)
    np.testing.assert_array_equal(np.diag(A), np.zeros(d))
    X = _all_binary(d)
    np.testing.assert_allclose(
        constant + X @ b + np.einsum("ni,ij,nj->n", X, A, X), second_order_features(X) @ alpha
    )


def test_spin_matrix_is_exact_on_all_binary_points() -> None:
    d = 5
    _, b, A = quadratic_form(_random_alpha(d, seed=1), d)
    B, offset = _to_spin_matrix(b, A)
    for x in _all_binary(d):
        z = np.append(2.0 * x - 1.0, 1.0)
        assert z @ B @ z + offset == pytest.approx(b @ x + x @ A @ x)


@pytest.mark.parametrize(
    "z, expected",
    [
        ([1.0, -1.0, 1.0, 1.0], [1.0, 0.0, 1.0]),
        # z @ B @ z is invariant to flipping all signs, so y0 = -1 means flip back.
        ([-1.0, 1.0, -1.0, -1.0], [1.0, 0.0, 1.0]),
    ],
)
def test_spin_to_binary_uses_auxiliary_variable_orientation(
    z: list[float], expected: list[float]
) -> None:
    np.testing.assert_array_equal(_spin_to_binary(np.array(z)), np.array(expected))


def test_sdp_relaxation_upper_bounds_binary_optimum() -> None:
    d = 8
    alpha = _random_alpha(d, seed=2)
    _, b, A = quadratic_form(alpha, d)
    B, offset = _to_spin_matrix(b, A)
    _, sdp_value = _solve_sdp_relaxation(B)
    assert sdp_value + offset >= (_brute_force_values(alpha, d) - alpha[0]).max() - 1e-6


def test_maximize_sdp_finds_near_optimal_points() -> None:
    d, n_instances, n_exact = 10, 10, 0
    for seed in range(n_instances):
        alpha = _random_alpha(d, seed)
        values = _brute_force_values(alpha, d)
        x = maximize_sdp(alpha, d, np.random.default_rng(seed))
        assert set(np.unique(x)) <= {0.0, 1.0}
        value = second_order_features(x)[0] @ alpha
        assert (values.max() - value) / (values.max() - values.min()) < 0.01
        n_exact += bool(np.isclose(value, values.max()))
    assert n_exact >= 8


@pytest.mark.parametrize("penalty, expected", [(1e3, 0.0), (-1e3, 1.0)])
def test_penalty_controls_number_of_ones(penalty: float, expected: float) -> None:
    d = 6
    x = maximize_sdp(_random_alpha(d, seed=3), d, np.random.default_rng(0), penalty=penalty)
    np.testing.assert_array_equal(x, np.full(d, expected))


def test_maximize_sdp_is_reproducible() -> None:
    d = 8
    alpha = _random_alpha(d, seed=4)
    first = maximize_sdp(alpha, d, np.random.default_rng(5))
    second = maximize_sdp(alpha, d, np.random.default_rng(5))
    np.testing.assert_array_equal(first, second)


def test_quadratic_form_rejects_wrong_number_of_coefficients() -> None:
    with pytest.raises(ValueError):
        quadratic_form(np.zeros(5), n_vars=4)
