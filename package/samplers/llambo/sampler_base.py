from __future__ import annotations

import time
from typing import Any
from typing import Optional

from llambo.llambo import LLAMBO
import optuna
from optuna.samplers import RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
import optunahub
import pandas as pd


class LLAMBOSampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
        self,
        custom_task_description: Optional[str] = None,
        n_initial_samples: int = 5,
        sm_mode: str = "discriminative",
        num_candidates: int = 10,
        n_templates: int = 2,
        n_gens: int = 10,
        alpha: float = 0.1,
        n_trials: int = 100,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        search_space: Optional[dict[str, optuna.distributions.BaseDistribution]] = None,
        debug: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(search_space)
        self.seed = seed
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)
        self.debug = debug
        self.last_time = time.time()
        self.last_trial_count = 0

        self.custom_task_description = custom_task_description
        self.n_initial_samples = n_initial_samples
        self.sm_mode = sm_mode
        self.num_candidates = num_candidates
        self.n_templates = n_templates
        self.n_gens = n_gens
        self.alpha = alpha
        self.n_trials = n_trials
        self.api_key = api_key
        self.model = model

        self.init_observed_fvals = pd.DataFrame()
        self.init_observed_configs = pd.DataFrame()

        self.LLAMBO_instance = None

    def _split_search_space(
        self, search_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> tuple[
        dict[str, optuna.distributions.BaseDistribution],
        dict[str, optuna.distributions.BaseDistribution],
    ]:
        """
        Split search space into numerical and categorical parts.
        Numerical parameters are FloatDistribution or IntDistribution.
        """
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

    def _initialize_llambo(
        self, numerical_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> None:
        """
        Initialize the LLAMBO instance using only the numerical portion of the search space.
        """
        self.hyperparameter_constraints = {}
        for param_name, distribution in numerical_space.items():
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                dtype = "float"
                dist_type = "log" if distribution.log else "linear"
                bounds = [distribution.low, distribution.high]
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                dtype = "int"
                dist_type = "log" if distribution.log else "linear"
                bounds = [distribution.low, distribution.high]
            else:
                continue  # Ignore any non-numerical parameter.
            self.hyperparameter_constraints[param_name] = [dtype, dist_type, bounds]

        if self.debug:
            print(
                f"Hyperparameter constraints (numerical only): {self.hyperparameter_constraints}"
            )

        # Build a task context that only includes numerical hyperparameters.
        task_context = {
            "custom_task_description": self.custom_task_description,
            "lower_is_better": self.lower_is_better,
            "hyperparameter_constraints": self.hyperparameter_constraints,
        }
        top_pct = 0.25 if self.sm_mode == "generative" else None

        self.LLAMBO_instance = LLAMBO(
            task_context,
            self.sm_mode,
            n_candidates=self.num_candidates,
            n_templates=self.n_templates,
            n_gens=self.n_gens,
            alpha=self.alpha,
            n_initial_samples=self.n_initial_samples,
            n_trials=self.n_trials,
            top_pct=top_pct,
            key=self.api_key,
            model=self.model,
        )

    def _sample_parameters(self) -> dict[str, Any]:
        """
        Sample parameters using the LLAMBO instance.
        Returns a dictionary mapping parameter names to their sampled values.
        """
        sampled_configuration = self.LLAMBO_instance.sample_configurations()
        return sampled_configuration

    def _debug_print(self, message: str) -> None:
        if self.debug:
            print(message)

    def _calculate_speed(self, n_completed: int) -> None:
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
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        """
        Hybrid sampling method:
          - Splits the search space into numerical and categorical parts.
          - Uses LLAMBO (initialized on the numerical space) to sample numerical parameters.
          - Uses RandomSampler to sample categorical parameters.
          - When passing observed configurations to LLAMBO, only numerical columns are retained.
          - Returns the merged configuration.
        """
        if len(search_space) == 0:
            return {}

        # Split search space into numerical and categorical parameters.
        numerical_space, categorical_space = self._split_search_space(search_space)

        # Sample categorical parameters using RandomSampler.
        categorical_params = {}
        for param_name, distribution in categorical_space.items():
            categorical_params[param_name] = self._random_sampler.sample_independent(
                study, trial, param_name, distribution
            )

        # If no numerical parameters exist, return only categorical parameters.
        if not numerical_space:
            return categorical_params

        # For initial trials, generate random samples and initialize LLAMBO.
        if trial.number <= self.n_initial_samples:
            if trial.number == 1:
                self.lower_is_better = (
                    True if study.direction == optuna.study.StudyDirection.MINIMIZE else False
                )
                self.init_configs = self.generate_random_samples(
                    numerical_space, self.n_initial_samples
                )
                self._initialize_llambo(numerical_space)
                self.init_observed_configs = []
                self.init_observed_fvals = []

            config = self.init_configs[trial.number - 1]

            if trial.number > 1:
                completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
                if completed_trials:
                    last_trial = completed_trials[-1]
                    self.init_observed_configs.append(last_trial.params)
                    self.init_observed_fvals.append(last_trial.value)

            return {**config, **categorical_params}

        # For later trials, filter observed configurations to only numerical columns.
        if trial.number == self.n_initial_samples + 1:
            completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
            if completed_trials:
                last_trial = completed_trials[-1]
                self.init_observed_configs.append(last_trial.params)
                self.init_observed_fvals.append(last_trial.value)

            observed_configs_df = pd.DataFrame(self.init_observed_configs)
            # Retain only numerical keys
            observed_configs_df = observed_configs_df[list(numerical_space.keys())]
            observed_fvals_df = pd.DataFrame({"score": self.init_observed_fvals})

            self.LLAMBO_instance._initialize(None, observed_configs_df, observed_fvals_df)

        # Use LLAMBO to sample numerical parameters.
        numerical_params = self.LLAMBO_instance.sample_configurations()

        # Merge the numerical and categorical parameters.
        combined_params = {**numerical_params, **categorical_params}
        return combined_params

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Optional[list[float]] = None,
    ) -> None:
        """
        Update LLAMBO's history using only the numerical portion of the parameters.
        """
        if self.LLAMBO_instance is not None:
            if state == optuna.trial.TrialState.COMPLETE and values is not None:
                # Filter trial.params to keep only numerical keys
                filtered_params = {
                    k: v for k, v in trial.params.items() if k in self.hyperparameter_constraints
                }
                self.LLAMBO_instance.update_history(filtered_params, values[0])

    def generate_random_samples(
        self, search_space: dict[str, optuna.distributions.BaseDistribution], num_samples: int = 1
    ) -> list[dict[str, Any]]:
        """
        Generate random samples for the numerical search space using the RandomSampler.
        """
        samples = []
        for _ in range(num_samples):
            params = {}
            for param_name, distribution in search_space.items():
                params[param_name] = self._random_sampler.sample_independent(
                    study=None,
                    trial=None,  # Not used in this context
                    param_name=param_name,
                    param_distribution=distribution,
                )
            samples.append(params)
        return samples
