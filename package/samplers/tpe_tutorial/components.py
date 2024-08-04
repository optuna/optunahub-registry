from __future__ import annotations

import numpy as np


class GammaFunc:
    def __init__(self, strategy: str, beta: float):
        strategy_choices = ["linear", "sqrt"]
        if strategy not in strategy_choices:
            raise ValueError(f"strategy must be in {strategy_choices}, but got {strategy}.")

        self._strategy = strategy
        self._beta = beta

    def __call__(self, x: int) -> int:
        if self._strategy == "linear":
            n = int(np.ceil(self._beta * x))
        elif self._strategy == "sqrt":
            n = int(np.ceil(self._beta * np.sqrt(x)))
        else:
            assert "Should not reach."

        return min(n, 25)


class WeightFunc:
    def __init__(self, strategy: str):
        strategy_choices = ["old-decay", "old-drop", "uniform", "EI"]
        if strategy not in strategy_choices:
            raise ValueError(f"strategy must be in {strategy_choices}, but got {strategy}.")

        self._strategy = strategy

    def __call__(self, x: int) -> np.ndarray:
        if x == 0:
            return np.asarray([])
        elif x < 25 or self._strategy == "uniform":
            return np.ones(x)
        elif self._strategy == "old-decay":
            ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
            flat = np.ones(25)
            return np.concatenate([ramp, flat], axis=0)
        elif self._strategy == "old-drop":
            weights = np.ones(x)
            weights[:-25] = 1e-12
            return weights
        elif self._strategy == "EI":
            # For below_trials, weights will be calculated separately.
            return np.ones(x)
        else:
            assert "Should not reach."
