from __future__ import annotations

import time
from typing import Any

from llambo.llambo import LLAMBO
import optuna
from optuna.samplers import RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
import optunahub
import pandas as pd


class Sampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution] | None = None,
        debug: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize the sampler with unified parameter handling."""
        super().__init__(search_space)
        self.seed = seed
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)  # Retained but not used in sampling
        self.debug = debug
        self.last_time = time.time()
        self.last_trial_count = 0

        self.n_initial_samples = 5
        self.init_observed_fvals = pd.DataFrame()
        self.init_observed_configs = pd.DataFrame()

        task_context = {
            "model": "RandomForest",
            "task": "classification",
            "tot_feats": 10,
            "cat_feats": 0,
            "num_feats": 10,
            "n_classes": 2,
            "metric": "accuracy",
            "lower_is_better": True,
            "num_samples": 1437,
            "hyperparameter_constraints": {
                "x0": ["float", "linear", [-5.12, 5.12]],  # Constraints for each dimension
                "x1": ["float", "linear", [-5.12, 5.12]],
                "x2": ["float", "linear", [-5.12, 5.12]],
                "x3": ["float", "linear", [-5.12, 5.12]],
                "x4": ["float", "linear", [-5.12, 5.12]],
                "x5": ["float", "linear", [-5.12, 5.12]],
                "x6": ["float", "linear", [-5.12, 5.12]],
                "x7": ["float", "linear", [-5.12, 5.12]],
                "x8": ["float", "linear", [-5.12, 5.12]],
                "x9": ["float", "linear", [-5.12, 5.12]],
            },
        }

        sm_mode = "discriminative"
        if sm_mode == "generative":
            top_pct = 0.25
        else:
            top_pct = None

        self.LLAMBO_instance = LLAMBO(
            task_context,
            sm_mode,
            n_candidates=10,
            n_templates=2,
            n_gens=10,
            alpha=0.1,
            n_initial_samples=self.n_initial_samples,
            n_trials=25,
            top_pct=top_pct,  # only used for generative SM, top percentage of points to consider for generative SM
            key="",
        )

    def _sample_parameters(self) -> dict[str, Any]:
        """Implement your sampler here."""
        print("DEBUG", "begin sampling parameters")
        sampled_configuration = self.LLAMBO_instance.sample_configurations()
        return sampled_configuration

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
        """Reseed the random number generator while preserving RandomSampler."""
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        """Unified sampling method for all parameter types."""
        if len(search_space) == 0:
            return {}

        print("DEBUG trial number", trial.number)

        if trial.number <= self.n_initial_samples:
            if trial.number == 1:
                print("DEBUG", "trigger 1")
                self.init_configs = self.generate_random_samples(
                    search_space, self.n_initial_samples
                )
            print("DEBUG", "sampled parameters", self.init_configs[trial.number - 1])
            return self.init_configs[trial.number - 1]

        print("DEBUG", "trigger init 2")

        if trial.number == self.n_initial_samples + 1:
            # Pass the observed data from initial trials to initialize LLAMBO
            self.LLAMBO_instance._initialize(
                self.init_configs,
                self.LLAMBO_instance.observed_configs,
                self.LLAMBO_instance.observed_fvals,
            )
            print("DEBUG", "trigger init 3")

        parameters = self._sample_parameters()

        # The study will evaluate the parameters, so we don't need to do it here.
        # Just return the suggested parameters.
        return parameters

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: list[float] | None,
    ) -> None:
        """Update the LLAMBO history after a trial is completed."""
        print("DEBUG", "used")
        if state == optuna.trial.TrialState.COMPLETE and values is not None:
            self.LLAMBO_instance.update_history(trial.params, values[0])

    def generate_random_samples(
        self, search_space: dict[str, optuna.distributions.BaseDistribution], num_samples: int = 1
    ) -> list[dict[str, Any]]:
        """
        Generate random samples using the RandomSampler's core logic directly.
        """
        samples = []

        for _ in range(num_samples):
            params = {}
            for param_name, distribution in search_space.items():
                # Use the RandomSampler's actual sampling logic
                params[param_name] = self._random_sampler.sample_independent(
                    study=None,
                    trial=None,  # Not actually used by RandomSampler's implementation
                    param_name=param_name,
                    param_distribution=distribution,
                )
            samples.append(params)

        return samples
