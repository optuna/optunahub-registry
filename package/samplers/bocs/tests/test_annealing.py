from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pytest

from .._annealing import _flip_gains
from .._annealing import maximize_sa
from .._model import n_coefficients
from .._model import quadratic_form
from .._model import second_order_features


def _random_alpha(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = n_coefficients(d)
    return rng.normal(size=p) * (rng.random(p) < 0.5)


def test_flip_gains_match_direct_evaluation() -> None:
    d = 7
    _, b, A = quadratic_form(_random_alpha(d, seed=0), d)
    x = np.random.default_rng(1).integers(0, 2, size=d).astype(float)
    value = b @ x + x @ A @ x
    for i, gain in enumerate(_flip_gains(b, A, x)):
        flipped = x.copy()
        flipped[i] = 1.0 - flipped[i]
        assert gain == pytest.approx(b @ flipped + flipped @ A @ flipped - value)


def test_maximize_sa_finds_near_optimal_points() -> None:
    d, n_instances, n_exact = 10, 10, 0
    all_x = np.array(list(product([0.0, 1.0], repeat=d)))
    for seed in range(n_instances):
        alpha = _random_alpha(d, seed)
        values = second_order_features(all_x) @ alpha
        x = maximize_sa(alpha, d, np.random.default_rng(seed))
        assert set(np.unique(x)) <= {0.0, 1.0}
        value = second_order_features(x)[0] @ alpha
        assert (values.max() - value) / (values.max() - values.min()) < 0.05
        n_exact += bool(np.isclose(value, values.max()))
    assert n_exact >= 7


@pytest.mark.parametrize("penalty, expected", [(1e3, 0.0), (-1e3, 1.0)])
def test_penalty_controls_number_of_ones(penalty: float, expected: float) -> None:
    d = 6
    x = maximize_sa(_random_alpha(d, seed=3), d, np.random.default_rng(0), penalty=penalty)
    np.testing.assert_array_equal(x, np.full(d, expected))


class _AcceptEveryFlip:
    """Random generator whose acceptance draws are always 0, so every flip is taken."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def integers(self, *args: Any, **kwargs: Any) -> Any:
        return self._rng.integers(*args, **kwargs)

    def random(self) -> float:
        return 0.0


def test_returns_best_visited_point_not_last_one() -> None:
    # Accepting every flip turns the walk into a pure random walk that often ends worse
    # than it started. The start point is always a candidate, so the returned point can
    # never be worse than it.
    d = 8
    alpha = _random_alpha(d, seed=6)
    _, b, A = quadratic_form(alpha, d)
    for seed in range(50):
        start = np.random.default_rng(seed).integers(0, 2, size=d).astype(float)
        x = maximize_sa(alpha, d, _AcceptEveryFlip(seed), n_steps=30)  # type: ignore[arg-type]
        assert b @ x + x @ A @ x >= b @ start + start @ A @ start - 1e-12


def test_maximize_sa_is_reproducible() -> None:
    d = 8
    alpha = _random_alpha(d, seed=4)
    first = maximize_sa(alpha, d, np.random.default_rng(5))
    second = maximize_sa(alpha, d, np.random.default_rng(5))
    np.testing.assert_array_equal(first, second)


def test_constant_model_returns_start_point() -> None:
    d = 4
    x = maximize_sa(np.zeros(n_coefficients(d)), d, np.random.default_rng(0))
    assert x.shape == (d,)
