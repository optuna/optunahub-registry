from __future__ import annotations

import random
from typing import Dict
from typing import List

import optuna
import optunahub


class Problem(optunahub.benchmarks.BaseProblem):
    def __init__(
        self,
        n_items: int = 20,
        n_dimensions: int = 2,
        seed: int | None = None,
        max_value: int = 100,
        max_weight: int = 50,
        max_capacity: float = 0.5,
    ):
        self.n_items = n_items
        self.n_dimensions = n_dimensions
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        self.values = [random.randint(1, max_value) for _ in range(n_items)]
        self.weights = [
            [random.randint(1, max_weight) for _ in range(n_dimensions)] for _ in range(n_items)
        ]
        total_weights = [
            sum(self.weights[i][j] for i in range(n_items)) for j in range(n_dimensions)
        ]
        self.capacities = [int(tw * (min(max_capacity, 1.0))) for tw in total_weights]

        self._search_space = {
            f"x{i}": optuna.distributions.CategoricalDistribution([0, 1]) for i in range(n_items)
        }

    @property
    def search_space(self) -> Dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> List[optuna.study.StudyDirection]:
        """Return the optimization directions (maximize value)."""
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: Dict[str, int]) -> float:
        x = [params[f"x{i}"] for i in range(self.n_items)]
        total_value = sum(self.values[i] * x[i] for i in range(self.n_items))
        return total_value

    def evaluate_constraints(self, trial: optuna.trial.FrozenTrial) -> List[float]:
        x = [trial.params[f"x{i}"] for i in range(self.n_items)]
        constraints = []
        for j in range(self.n_dimensions):
            total_weight = sum(self.weights[i][j] * x[i] for i in range(self.n_items))
            constraints.append(total_weight - self.capacities[j])  # Should be <= 0
        return constraints
