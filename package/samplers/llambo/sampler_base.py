"""
LLAMBO-based sampler for optimization combining Bayesian optimization and LLMs.

This module provides a sampler implementation that leverages Language Models
for Bayesian Optimization (LLAMBO) to guide the optimization process.
"""

from __future__ import annotations

import time
from typing import Any
from typing import Optional
from typing import Sequence

from llambo.llambo import LLAMBO
import optuna
from optuna.samplers import RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
import pandas as pd


class LLAMBOSampler(optunahub.samplers.SimpleBaseSampler):
    """
    A sampler that combines Bayesian optimization with Large Language Models.

    This sampler uses LLAMBO (Language Models for Bayesian Optimization) to guide
    the optimization process, incorporating both traditional random sampling and
    LLM-based suggestions.

    Args:
        custom_task_description (Optional[str]): Custom description of the optimization task.
        n_initial_samples (int): Number of initial random samples before using LLAMBO.
        sm_mode (str): Surrogate model mode, either "discriminative" or "generative".
        num_candidates (int): Number of candidate configurations to generate.
        n_templates (int): Number of templates to use for LLM prompts.
        n_gens (int): Number of generations for optimization.
        alpha (float): Exploration-exploitation trade-off parameter.
        n_trials (int): Total number of trials for optimization.
        api_key (str): API key for accessing the LLM service.
        model (str): Name of the LLM model to use.
        search_space (Optional[dict[str, optuna.distributions.BaseDistribution]]): Search space
        definition.
        debug (bool): Whether to enable debug output.
        seed (Optional[int]): Random seed for reproducibility.

    Example:
        >>> search_space = {
        ...     "x": optuna.distributions.FloatDistribution(0, 1),
        ...     "y": optuna.distributions.CategoricalDistribution(["a", "b"]),
        ... }
        >>> sampler = LlamboSampler(
        ...     n_initial_samples=5, search_space=search_space, debug=True
        ... )
    """

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
        """Initialize the sampler with unified parameter handling."""
        super().__init__(search_space)
        self.seed = seed
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)
        self.debug = debug
        self.last_time = time.time()
        self.last_trial_count = 0

        # LLAMBO-specific parameters
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
        self.LLAMBO_instance: Optional[LLAMBO] = None
        self.lower_is_better: bool = True
        self.init_configs: list[dict[str, Any]] = []
        self.search_space: dict[str, optuna.distributions.BaseDistribution] = {}
        self.hyperparameter_constraints: dict[str, list[Any]] = {}

    def _initialize_llambo(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> None:
        """
        Initialize the LLAMBO instance with the given search space.

        Args:
            search_space: Dictionary mapping parameter names to their distributions.

        Raises:
            ValueError: If an unsupported distribution type is encountered.

        Example:
            >>> space = {"x": optuna.distributions.FloatDistribution(0, 1)}
            >>> sampler = LlamboSampler(search_space=space)
            >>> sampler._initialize_llambo(space)
        """
        self.hyperparameter_constraints = {}
        for param_name, distribution in search_space.items():
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                dtype = "float"
                dist_type = "log" if distribution.log else "linear"
                bounds = [float(distribution.low), float(distribution.high)]
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                dtype = "int"
                dist_type = "log" if distribution.log else "linear"
                bounds = [int(distribution.low), int(distribution.high)]
            elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                dtype = "categorical"
                dist_type = "categorical"
                bounds = list(distribution.choices)  # type: ignore[arg-type]
            else:
                raise ValueError(
                    f"Unsupported distribution type {type(distribution)} for parameter "
                    f"{param_name}"
                )
            self.hyperparameter_constraints[param_name] = [dtype, dist_type, bounds]

        self._debug_print(f"Hyperparameter constraints: {self.hyperparameter_constraints}")

        task_context = {
            "custom_task_description": self.custom_task_description,
            "lower_is_better": self.lower_is_better,
            "hyperparameter_constraints": self.hyperparameter_constraints,
        }

        sm_mode = self.sm_mode
        top_pct = 0.25 if sm_mode == "generative" else None

        self.LLAMBO_instance = LLAMBO(
            task_context,
            sm_mode,
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

        Returns:
            dict[str, Any]: Dictionary of sampled parameter values.

        Raises:
            RuntimeError: If LLAMBO instance is not initialized.

        Example:
            >>> sampler = LlamboSampler()
            >>> params = sampler._sample_parameters()
        """
        if self.LLAMBO_instance is None:
            raise RuntimeError("LLAMBO instance not initialized")
        return self.LLAMBO_instance.sample_configurations()

    def _debug_print(self, message: str) -> None:
        """
        Print debug message if debug mode is enabled.

        Args:
            message: Message to print in debug mode.

        Example:
            >>> sampler = LlamboSampler(debug=True)
            >>> sampler._debug_print("Debug information")
        """
        if self.debug:
            print(message)

    def _calculate_speed(self, n_completed: int) -> None:
        """
        Calculate and print optimization speed metrics every 100 trials.

        Args:
            n_completed: Number of completed trials.

        Example:
            >>> sampler = LlamboSampler(debug=True)
            >>> sampler._calculate_speed(100)
        """
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
        """
        Reseed the random number generator while preserving RandomSampler.

        Example:
            >>> sampler = LlamboSampler(seed=42)
            >>> sampler.reseed_rng()
        """
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        """
        Sample parameters relative to the current state of optimization.

        Args:
            study: The Optuna study object.
            trial: The current trial being executed.
            search_space: The search space for parameters.

        Returns:
            dict[str, Any]: Dictionary of sampled parameter values.

        Example:
            >>> study = optuna.create_study()
            >>> trial = optuna.trial.FrozenTrial(...)
            >>> space = {"x": optuna.distributions.FloatDistribution(0, 1)}
            >>> sampler = LlamboSampler()
            >>> params = sampler.sample_relative(study, trial, space)
        """
        if len(search_space) == 0:
            return {}

        self.search_space = search_space

        # Initialize on first trial
        if trial.number <= self.n_initial_samples:
            if trial.number == 1:
                self.lower_is_better = study.direction == optuna.study.StudyDirection.MINIMIZE
                self.init_configs = self.generate_random_samples(
                    search_space,
                    self.n_initial_samples,
                )
                self._initialize_llambo(search_space)

            params = self.init_configs[trial.number - 1]
            # Update LLAMBO instance with the sampled configuration
            if self.LLAMBO_instance is not None:
                completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
                if completed_trials:
                    # Use the actual value from the last completed trial
                    last_value = completed_trials[-1].value
                    if last_value is not None:
                        self.LLAMBO_instance.update_history(params, last_value)

            return params

        # Initialize LLAMBO with collected data at transition point
        if trial.number == self.n_initial_samples + 1 and self.LLAMBO_instance is not None:
            completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
            if len(completed_trials) >= self.n_initial_samples:
                # Get configurations and values from completed initial trials
                configs = [t.params for t in completed_trials[: self.n_initial_samples]]
                values = [
                    t.value
                    for t in completed_trials[: self.n_initial_samples]
                    if t.value is not None
                ]

                # Convert to pandas DataFrames as expected by LLAMBO
                config_df = pd.DataFrame(configs)
                value_df = pd.DataFrame({"score": values, "generalization_score": values})

                self.LLAMBO_instance._initialize(configs, config_df, value_df)

        return self._sample_parameters()

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Update LLAMBO history after a trial is completed.

        Args:
            study: The Optuna study object.
            trial: The completed trial.
            state: The state of the completed trial.
            values: Sequence of objective values from the trial.

        Example:
            >>> study = optuna.create_study()
            >>> trial = optuna.trial.FrozenTrial(...)
            >>> sampler = LlamboSampler()
            >>> sampler.after_trial(
            ...     study, trial, optuna.trial.TrialState.COMPLETE, [0.5]
            ... )
        """
        if self.LLAMBO_instance is not None:
            if state == TrialState.COMPLETE and values is not None:
                self.LLAMBO_instance.update_history(trial.params, values[0])

    def generate_random_samples(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution],
        num_samples: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Generate random samples using the RandomSampler's core logic.

        Args:
            search_space: The search space for parameters.
            num_samples: Number of random samples to generate.

        Returns:
            list[dict[str, Any]]: List of randomly sampled parameter dictionaries.

        Example:
            >>> space = {"x": optuna.distributions.FloatDistribution(0, 1)}
            >>> sampler = LlamboSampler()
            >>> samples = sampler.generate_random_samples(space, 5)
        """
        # Create dummy study and trial for type compatibility
        dummy_study = optuna.create_study()
        dummy_trial = optuna.trial.create_trial(
            state=TrialState.RUNNING,
            value=None,
        )

        samples = []
        for _ in range(num_samples):
            params = {}
            for param_name, distribution in search_space.items():
                params[param_name] = self._random_sampler.sample_independent(
                    study=dummy_study,
                    trial=dummy_trial,
                    param_name=param_name,
                    param_distribution=distribution,
                )
            samples.append(params)
        return samples
