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
    """
    A sampler implementation using LLAMBO (LLM-Augmented Model-Based Optimization).

    This sampler leverages large language models for hyperparameter optimization.
    It starts with random sampling for a few initial trials, then uses LLAMBO to
    intelligently sample parameters based on previous observations.

    Attributes:
        custom_task_description (str): Custom description for the optimization task.
        n_initial_samples (int): Number of initial random samples before using LLAMBO.
        sm_mode (str): Surrogate model mode, either "discriminative" or "generative".
        num_candidates (int): Number of candidate configurations to generate.
        n_templates (int): Number of prompt templates to use.
        n_gens (int): Number of generations for each template.
        alpha (float): Exploration parameter.
        n_trials (int): Total number of trials to run.
        api_key (str): API key for the LLM service.
        model (str): The LLM model to use.
        debug (bool): Whether to enable debug printing.
        seed (Optional[int]): Random seed for reproducibility.

    Example:
        >>> import optuna
        >>> sampler = LLAMBOSampler(
        ...     n_initial_samples=5,
        ...     api_key="your_api_key",
        ...     model="gpt-4o-mini"
        ... )
        >>> study = optuna.create_study(sampler=sampler)
        >>> study.optimize(objective, n_trials=100)
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
        """
        Initialize the sampler with unified parameter handling.

        Args:
            custom_task_description: Custom description for the optimization task.
            n_initial_samples: Number of initial random samples before using LLAMBO.
            sm_mode: Surrogate model mode, either "discriminative" or "generative".
            num_candidates: Number of candidate configurations to generate.
            n_templates: Number of prompt templates to use.
            n_gens: Number of generations for each template.
            alpha: Exploration parameter.
            n_trials: Total number of trials to run.
            api_key: API key for the LLM service.
            model: The LLM model to use.
            search_space: Dictionary mapping parameter names to their distributions.
            debug: Whether to enable debug printing.
            seed: Random seed for reproducibility.
        """
        super().__init__(search_space)
        self.seed = seed
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)  # Retained but not used in sampling
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

        self.LLAMBO_instance = None

    def _initialize_llambo(
        self, search_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> None:
        """
        Initialize the LLAMBO instance with proper hyperparameter constraints.

        Args:
            search_space: Dictionary mapping parameter names to their distributions.

        Raises:
            ValueError: If an unsupported distribution type is encountered.

        Example:
            >>> sampler = LLAMBOSampler()
            >>> search_space = {
            ...     "x": optuna.distributions.FloatDistribution(0, 1),
            ...     "y": optuna.distributions.CategoricalDistribution(["a", "b"])
            ... }
            >>> sampler._initialize_llambo(search_space)
        """
        # Create hyperparameter constraints dictionary from search space
        self.hyperparameter_constraints = {}
        for param_name, distribution in search_space.items():
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                dtype = "float"
                dist_type = "log" if distribution.log else "linear"
                bounds = [distribution.low, distribution.high]
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                dtype = "int"
                dist_type = "log" if distribution.log else "linear"
                bounds = [distribution.low, distribution.high]
            elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                dtype = "categorical"
                dist_type = "categorical"
                bounds = distribution.choices
            else:
                raise ValueError(
                    f"Unsupported distribution type {type(distribution)} for parameter {param_name}"
                )
            self.hyperparameter_constraints[param_name] = [dtype, dist_type, bounds]

        if self.debug:
            print(f"Hyperparameter constraints: {self.hyperparameter_constraints}")

        # Prepare task_context with the constraints
        task_context = {
            "custom_task_description": self.custom_task_description,
            "lower_is_better": self.lower_is_better,
            "hyperparameter_constraints": self.hyperparameter_constraints,
        }

        sm_mode = self.sm_mode
        top_pct = 0.25 if sm_mode == "generative" else None

        # Initialize LLAMBO with proper task context
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
            A dictionary mapping parameter names to their sampled values.

        Example:
            >>> sampler = LLAMBOSampler()
            >>> # Assuming LLAMBO_instance is initialized
            >>> params = sampler._sample_parameters()
            >>> isinstance(params, dict)
            True
        """
        sampled_configuration = self.LLAMBO_instance.sample_configurations()
        return sampled_configuration

    def _debug_print(self, message: str) -> None:
        """
        Print debug message if debug mode is enabled.

        Args:
            message: The message to print.

        Example:
            >>> sampler = LLAMBOSampler(debug=True)
            >>> sampler._debug_print("Test message")
            Test message
        """
        if self.debug:
            print(message)

    def _calculate_speed(self, n_completed: int) -> None:
        """
        Calculate and print optimization speed every 100 trials.

        Args:
            n_completed: Number of completed trials.

        Example:
            >>> sampler = LLAMBOSampler(debug=True)
            >>> sampler.last_time = time.time() - 10  # 10 seconds ago
            >>> sampler.last_trial_count = 100
            >>> sampler._calculate_speed(200)
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
            >>> sampler = LLAMBOSampler(seed=42)
            >>> sampler.reseed_rng()
        """
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        """
        Unified sampling method for all parameter types.

        This method handles both the initial random sampling phase and the
        LLAMBO-based sampling phase.

        Args:
            study: The study object.
            trial: The trial object.
            search_space: Dictionary mapping parameter names to their distributions.

        Returns:
            A dictionary mapping parameter names to their sampled values.

        Example:
            >>> import optuna
            >>> study = optuna.create_study()
            >>> trial = optuna.trial.create_trial(
            ...     params={},
            ...     distributions={},
            ...     value=0.0
            ... )
            >>> sampler = LLAMBOSampler()
            >>> search_space = {"x": optuna.distributions.FloatDistribution(0, 1)}
            >>> params = sampler.sample_relative(study, trial, search_space)
        """
        if len(search_space) == 0:
            return {}
        # delegate first trial to random sampler
        self.search_space = search_space

        if trial.number <= self.n_initial_samples:
            if trial.number == 1:
                self.lower_is_better = (
                    True if study.direction == optuna.study.StudyDirection.MINIMIZE else False
                )
                self.init_configs = self.generate_random_samples(
                    search_space, self.n_initial_samples
                )
                self._initialize_llambo(search_space)
            return self.init_configs[trial.number - 1]

        if trial.number == self.n_initial_samples + 1:
            # Pass the observed data from initial trials to initialize LLAMBO
            self.LLAMBO_instance._initialize(
                self.init_configs,
                self.LLAMBO_instance.observed_configs,
                self.LLAMBO_instance.observed_fvals,
            )

        parameters = self._sample_parameters()

        return parameters

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Optional[list[float]] = None,
    ) -> None:
        """
        Update the LLAMBO history after a trial is completed.

        Args:
            study: The study object.
            trial: The trial object.
            state: The state of the trial.
            values: The values observed in the trial.

        Example:
            >>> import optuna
            >>> study = optuna.create_study()
            >>> trial = optuna.trial.create_trial(
            ...     params={},
            ...     distributions={},
            ...     value=0.0
            ... )
            >>> sampler = LLAMBOSampler()
            >>> sampler.LLAMBO_instance = LLAMBO({}, "discriminative")
            >>> sampler.after_trial(
            ...     study,
            ...     trial,
            ...     optuna.trial.TrialState.COMPLETE,
            ...     [0.5]
            ... )
        """
        if self.LLAMBO_instance is not None:
            if state == optuna.trial.TrialState.COMPLETE and values is not None:
                self.LLAMBO_instance.update_history(trial.params, values[0])

    def generate_random_samples(
        self, search_space: dict[str, optuna.distributions.BaseDistribution], num_samples: int = 1
    ) -> list[dict[str, Any]]:
        """
        Generate random samples using the RandomSampler's core logic directly.

        Args:
            search_space: Dictionary mapping parameter names to their distributions.
            num_samples: Number of random samples to generate.

        Returns:
            A list of dictionaries mapping parameter names to their sampled values.

        Example:
            >>> sampler = LLAMBOSampler()
            >>> search_space = {"x": optuna.distributions.FloatDistribution(0, 1)}
            >>> samples = sampler.generate_random_samples(search_space, 3)
            >>> len(samples)
            3
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
