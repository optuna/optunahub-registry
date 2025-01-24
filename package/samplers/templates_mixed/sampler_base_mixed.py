from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna
from optuna.samplers import RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
import optunahub


class Sampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution] | None = None,
        debug: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize the sampler."""
        super().__init__(search_space)
        self.seed = seed
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)
        self.debug = debug
        self.last_time = time.time()
        self.last_trial_count = 0

    def _split_search_space(
        self, search_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> tuple[dict, dict]:
        """Split search space into numerical and categorical parameters."""
        numerical_space = {}
        categorical_space = {}

        for name, dist in search_space.items():
            if isinstance(
                dist,
                (optuna.distributions.FloatDistribution, optuna.distributions.IntDistribution),
            ):
                numerical_space[name] = dist
            else:
                categorical_space[name] = dist

        return numerical_space, categorical_space

    def _sample_numerical_parameters(
        self, numerical_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> dict[str, Any]:
        """Implement your sampler for numerical parameters here"""
        numerical_params = {}

        for param_name, distribution in numerical_space.items():
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                if distribution.log:
                    # Handle log-uniform sampling
                    log_low = np.log(distribution.low)
                    log_high = np.log(distribution.high)
                    val = np.exp(self._rng.rng.uniform(log_low, log_high))
                else:
                    # Linear scale uniform sampling
                    val = self._rng.rng.uniform(distribution.low, distribution.high)
                numerical_params[param_name] = val
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                # Handle integer stepping
                steps = (distribution.high - distribution.low) // distribution.step
                s = self._rng.rng.randint(0, steps + 1)
                val = distribution.low + s * distribution.step
                numerical_params[param_name] = val

        return numerical_params

    def _debug_print(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(message)

    def _calculate_speed(self, n_completed: int) -> None:
        """Calculate and print optimization speed every 100 trials."""
        if not self.debug:
            return

        if n_completed % 100 == 0 and n_completed > 0:
            current_time = time.time()
            elapsed_time = current_time - self.last_time
            trials_processed = n_completed - self.last_trial_count

            if elapsed_time > 0:
                speed = trials_processed / elapsed_time
                print(f"\n[Speed Stats] Trials {self.last_trial_count} to {n_completed}")
                print(f"Speed: {speed:.2f} trials/second")
                print(f"Time elapsed: {elapsed_time:.2f} seconds")
                print("-" * 50)

            self.last_time = current_time
            self.last_trial_count = n_completed

    def reseed_rng(self) -> None:
        """Reseed the random number generator for the sampler."""
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        """Sample parameters using separated numerical and categorical sampling."""
        if len(search_space) == 0:
            return {}

        numerical_space, categorical_space = self._split_search_space(search_space)

        # Sample categorical parameters
        categorical_params = {
            param_name: self._random_sampler.sample_independent(
                study, trial, param_name, distribution
            )
            for param_name, distribution in categorical_space.items()
        }

        # Sample numerical parameters using dedicated method
        numerical_params = self._sample_numerical_parameters(numerical_space)

        self._calculate_speed(trial.number if trial.number is not None else 0)

        return {**numerical_params, **categorical_params}
