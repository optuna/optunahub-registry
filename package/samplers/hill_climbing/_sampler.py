"""Hill Climbing Sampler for Optuna.

This module implements a hill-climbing algorithm as an Optuna sampler.
The sampler is designed for discrete optimization problems (integers and categorical values).
"""

from __future__ import annotations

import random
from typing import Any

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import RandomSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optunahub.samplers import SimpleBaseSampler


class HillClimbingSampler(SimpleBaseSampler):
    """Hill Climbing sampler for discrete optimization problems.

    This sampler implements a hill-climbing algorithm that iteratively improves
    solutions by evaluating neighboring solutions. It supports integer and categorical
    parameters only.

    The algorithm:
    1. Starts from a random point using RandomSampler
    2. Generates neighboring points by modifying current parameters
    3. Moves to the best improvement among neighbors
    4. Restarts when no improvements are found

    Args:
        search_space: A dictionary containing the parameter names and their distributions.
        seed: Seed for random number generator.
        neighbor_size: Number of neighbors to generate per iteration.
        max_restarts: Maximum number of restarts allowed.
    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        *,
        seed: int | None = None,
        neighbor_size: int = 5,
        max_restarts: int = 10,
    ) -> None:
        super().__init__(search_space=search_space, seed=seed)
        self._neighbor_size = neighbor_size
        self._max_restarts = max_restarts
        self._random_sampler = RandomSampler(seed=seed)

        # Algorithm state
        self._current_params: dict[str, Any] | None = None
        self._current_value: float | None = None
        self._neighbors_to_evaluate: list[dict[str, Any]] = []
        self._evaluated_neighbors: set[tuple[tuple[str, Any], ...]] = set()
        self._restart_count = 0
        self._is_initialized = False

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        """Sample parameters using hill-climbing algorithm."""
        # Validate search space contains only supported distributions
        self._validate_search_space(search_space)

        # Initialize if this is the first trial or we need to restart
        if not self._is_initialized or self._should_restart(study):
            return self._initialize_or_restart(study, trial, search_space)

        # If we have neighbors to evaluate, return the next one
        if self._neighbors_to_evaluate:
            return self._neighbors_to_evaluate.pop(0)

        # Generate new neighbors from current position
        return self._generate_neighbors_and_sample(study, trial, search_space)

    def _validate_search_space(self, search_space: dict[str, BaseDistribution]) -> None:
        """Validate that search space contains only supported distributions."""
        for param_name, distribution in search_space.items():
            if not isinstance(distribution, (IntDistribution, CategoricalDistribution)):
                raise ValueError(
                    f"HillClimbingSampler only supports IntDistribution and "
                    f"CategoricalDistribution. Got {type(distribution)} for parameter '{param_name}'."
                )

    def _should_restart(self, study: Study) -> bool:
        """Check if we should restart the search."""
        if self._restart_count >= self._max_restarts:
            # No more restarts allowed, continue with current strategy
            return False

        completed_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )
        if len(completed_trials) < 2:
            return False

        # Check if we haven't improved in the last few trials
        recent_trials = completed_trials[-min(5, len(completed_trials)) :]
        if self._current_value is not None:
            recent_improvements = [
                trial.value
                for trial in recent_trials
                if trial.value is not None and trial.value < self._current_value
            ]
            # Restart if no improvements in recent trials and we still have restarts left
            return len(recent_improvements) == 0

        return False

    def _initialize_or_restart(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        """Initialize or restart the hill-climbing search."""
        if not self._is_initialized:
            self._is_initialized = True
        else:
            self._restart_count += 1

        # Reset state
        self._current_params = None
        self._current_value = None
        self._neighbors_to_evaluate = []
        self._evaluated_neighbors = set()

        # Use RandomSampler for initialization
        return self._random_sampler.sample_relative(study, trial, search_space)

    def _generate_neighbors_and_sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        """Generate neighbors and return the first one to sample."""
        # Update current position based on completed trials
        self._update_current_position(study)

        if self._current_params is None:
            # Fallback to random sampling
            return self._random_sampler.sample_relative(study, trial, search_space)

        # Generate neighbors
        neighbors = self._generate_neighbors(self._current_params, search_space)

        # Filter out already evaluated neighbors
        new_neighbors = []
        for neighbor in neighbors:
            neighbor_key = tuple(sorted(neighbor.items()))
            if neighbor_key not in self._evaluated_neighbors:
                new_neighbors.append(neighbor)
                self._evaluated_neighbors.add(neighbor_key)

        if new_neighbors:
            self._neighbors_to_evaluate = new_neighbors[1:]  # Save rest for later
            return new_neighbors[0]
        else:
            # No new neighbors, restart
            return self._initialize_or_restart(study, trial, search_space)

    def _update_current_position(self, study: Study) -> None:
        """Update current position based on the best trial so far."""
        completed_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )
        if not completed_trials:
            return

        # Find the best trial
        best_trial = min(
            completed_trials, key=lambda t: t.value if t.value is not None else float("inf")
        )
        if best_trial.value is not None:
            self._current_params = best_trial.params.copy()
            self._current_value = best_trial.value

    def _generate_neighbors(
        self, current_params: dict[str, Any], search_space: dict[str, BaseDistribution]
    ) -> list[dict[str, Any]]:
        """Generate neighboring points by modifying parameters."""
        neighbors = []

        for _ in range(self._neighbor_size):
            neighbor = current_params.copy()

            # Randomly select a parameter to modify
            param_name = random.choice(list(search_space.keys()))
            distribution = search_space[param_name]

            if isinstance(distribution, IntDistribution):
                neighbor[param_name] = self._modify_int_parameter(
                    current_params[param_name], distribution
                )
            elif isinstance(distribution, CategoricalDistribution):
                neighbor[param_name] = self._modify_categorical_parameter(
                    current_params[param_name], distribution
                )

            neighbors.append(neighbor)

        return neighbors

    def _modify_int_parameter(self, current_value: int, distribution: IntDistribution) -> int:
        """Modify an integer parameter to create a neighbor."""
        # Generate a small step in either direction
        step_size = max(1, abs(distribution.high - distribution.low) // 20)
        direction = random.choice([-1, 1])
        new_value = current_value + direction * step_size

        # Ensure the new value is within bounds
        return max(distribution.low, min(distribution.high, new_value))

    def _modify_categorical_parameter(
        self, current_value: Any, distribution: CategoricalDistribution
    ) -> Any:
        """Modify a categorical parameter to create a neighbor."""
        # For categorical parameters, randomly select a different value
        choices = [choice for choice in distribution.choices if choice != current_value]
        if choices:
            return random.choice(choices)
        else:
            # If only one choice available, return the current value
            return current_value
