from __future__ import annotations

import threading
import time
from typing import Any
from typing import Optional

import nest_asyncio
import optuna
from optuna.samplers import RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
import optunahub
import pandas as pd

from .llambo.llambo import LLAMBO


# Avoid asyncio issues in Notebook environment
nest_asyncio.apply()


class LLAMBOSampler(optunahub.samplers.SimpleBaseSampler):
    """
    LLAMBO (Language Models for Bayesian Optimization) sampler implementation for Optuna.

    This sampler uses large language models to guide the optimization process. It works in a hybrid
    fashion, splitting the search space into numerical and categorical parts, using LLAMBO for the
    numerical parameters and a RandomSampler for categorical parameters.

    Args:
        custom_task_description: Optional description of the optimization task for the LLM.
        n_initial_samples: Number of initial random samples before using LLAMBO.
        sm_mode: Surrogate model mode, either "discriminative" or "generative".
        num_candidates: Number of candidate points to generate.
        n_templates: Number of prompt templates to use.
        n_gens: Number of generations per template.
        alpha: Exploration-exploitation trade-off parameter.
        api_key: API key for the language model service.
        model: Language model identifier to use.
        max_requests_per_minute: Maximum number of requests per minute.
        search_space: Optional search space to sample from.
        debug: Whether to print debug information.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        custom_task_description: Optional[str] = None,
        n_initial_samples: int = 5,
        sm_mode: str = "generative",
        num_candidates: int = 10,
        n_templates: int = 2,
        n_gens: int = 10,
        alpha: float = 0.1,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        max_requests_per_minute: int = 100,
        search_space: Optional[dict[str, optuna.distributions.BaseDistribution]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(search_space)
        self.seed = seed
        self._rng = LazyRandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)
        self.last_time = time.time()
        self.last_trial_count = 0

        self.custom_task_description = custom_task_description
        self.n_initial_samples = n_initial_samples
        self.sm_mode = sm_mode
        self.num_candidates = num_candidates
        self.n_templates = n_templates
        self.n_gens = n_gens
        self.alpha = alpha
        self.api_key = api_key
        self.model = model
        self.max_requests_per_minute = max_requests_per_minute

        # Initialize attributes that will be set later
        self.lower_is_better = True  # Default, will be set properly in sample_relative
        self.init_configs: list[dict[str, Any]] = []  # Will be populated in sample_relative
        self.hyperparameter_constraints: dict[
            str, list[Any]
        ] = {}  # Will be populated in _initialize_llambo

        # Initialize empty DataFrames instead of lists for thread-safety
        self.init_observed_configs = pd.DataFrame()
        # Initialize with the column structure required by LLAMBO
        self.init_observed_fvals = pd.DataFrame(columns=["score"])

        # Fix the type error by properly annotating LLAMBO_instance
        self.LLAMBO_instance: Optional[LLAMBO] = None

        # Add locks for thread safety
        self._llambo_init_lock = threading.Lock()
        self._obs_data_lock = threading.Lock()

        # Add a flag to track if initialization has been completed
        self._llambo_initialized = False
        self._min_trials_processed = 0

        # Add timeout and retry mechanism
        self._initialization_attempts = 0
        self._max_initialization_attempts = 3
        self._last_initialization_attempt = 0.0
        self._initialization_timeout = 30  # seconds

    def _split_search_space(
        self, search_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> tuple[
        dict[str, optuna.distributions.BaseDistribution],
        dict[str, optuna.distributions.BaseDistribution],
    ]:
        """
        Split search space into numerical and categorical parts.

        Numerical parameters are those with FloatDistribution or IntDistribution,
        while categorical parameters are all others.

        Args:
            search_space: Complete search space with all parameter distributions.

        Returns:
            Tuple containing (numerical_space, categorical_space).
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
    ) -> bool:
        """
        Initialize the LLAMBO instance using only the numerical portion of the search space.

        This method converts Optuna distributions to the format required by LLAMBO and
        creates the LLAMBO instance.

        Args:
            numerical_space: Dictionary of numerical parameter distributions.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if not numerical_space:
            return False

        current_time = time.time()
        # Check if we should retry initialization based on timeout
        if (
            self._initialization_attempts >= self._max_initialization_attempts
            and current_time - self._last_initialization_attempt < self._initialization_timeout
        ):
            return False

        self._last_initialization_attempt = current_time
        self._initialization_attempts += 1

        with self._llambo_init_lock:
            # Double-check locking pattern - check again if already initialized
            if self.LLAMBO_instance is not None:
                return True

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

            if not self.hyperparameter_constraints:
                return False

            # Build a task context that only includes numerical hyperparameters.
            task_context = {
                "custom_task_description": self.custom_task_description,
                "lower_is_better": self.lower_is_better,
                "hyperparameter_constraints": self.hyperparameter_constraints,
            }
            top_pct = 0.25 if self.sm_mode == "generative" else None

            try:
                self.LLAMBO_instance = LLAMBO(
                    task_context,
                    self.sm_mode,
                    n_candidates=self.num_candidates,
                    n_templates=self.n_templates,
                    n_gens=self.n_gens,
                    alpha=self.alpha,
                    n_initial_samples=self.n_initial_samples,
                    top_pct=top_pct,
                    key=self.api_key,
                    model=self.model,
                    max_requests_per_minute=self.max_requests_per_minute,
                )
                return True
            except Exception:
                self.LLAMBO_instance = None
                return False

    def _sample_parameters(self) -> dict[str, Any]:
        """
        Sample parameters using the LLAMBO instance.

        Returns:
            A dictionary mapping parameter names to their sampled values.
        """
        # Ensure LLAMBO instance exists before calling its methods
        if self.LLAMBO_instance is None:
            return {}

        try:
            sampled_configuration = self.LLAMBO_instance.sample_configurations()
            # Ensure we return a dictionary even if sample_configurations returns None
            return sampled_configuration if sampled_configuration is not None else {}
        except Exception:
            return {}

    def reseed_rng(self) -> None:
        """Reset the random number generator seeds."""
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

    def _collect_initial_data(self, study: optuna.study.Study) -> None:
        """
        Collect data from completed trials for initializing LLAMBO.
        Thread-safe method to collect data from completed trials.

        Args:
            study: Optuna study object.
        """
        with self._obs_data_lock:
            # Collect all completed trials data
            completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
            if not completed_trials:
                return

            # Process each completed trial and add to our observed data
            trials_added = 0
            for completed_trial in completed_trials:
                # Skip trials we've already processed
                if self.init_observed_configs.shape[0] > 0:
                    # Check if this trial's parameters are already in our data
                    is_duplicate = False
                    try:
                        # More robust duplicate checking
                        params_df = pd.DataFrame([completed_trial.params])
                        merged = pd.merge(self.init_observed_configs, params_df)
                        is_duplicate = not merged.empty
                    except Exception:
                        is_duplicate = False

                    if is_duplicate:
                        continue

                try:
                    # Add new configuration as DataFrame for concatenation
                    new_config = pd.DataFrame([completed_trial.params])
                    self.init_observed_configs = pd.concat(
                        [self.init_observed_configs, new_config], ignore_index=True
                    )
                    # Add value to DataFrame with consistent column structure
                    new_fval = pd.DataFrame({"score": [completed_trial.value]})
                    self.init_observed_fvals = pd.concat(
                        [self.init_observed_fvals, new_fval], ignore_index=True
                    )
                    trials_added += 1
                except Exception:
                    pass

            # Update how many trials we've processed
            self._min_trials_processed = len(self.init_observed_configs)

    def _initialize_llambo_with_data(
        self, numerical_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> bool:
        """
        Thread-safe method to initialize LLAMBO with observed data.

        Args:
            numerical_space: Dictionary of numerical parameter distributions.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self._min_trials_processed < self.n_initial_samples:
            return False

        with self._llambo_init_lock:
            # Double-check locking pattern
            if self._llambo_initialized:
                return True

            # Ensure LLAMBO instance exists
            if self.LLAMBO_instance is None:
                return False

            # Retain only numerical keys from observed configurations
            try:
                observed_configs_df = self.init_observed_configs.copy()

                if observed_configs_df.empty:
                    return False

                # Ensure we only keep columns that exist in numerical_space
                numerical_cols = [
                    col for col in numerical_space.keys() if col in observed_configs_df.columns
                ]

                if not numerical_cols:
                    return False

                observed_configs_df = observed_configs_df[numerical_cols]

                # Ensure observed_fvals_df has the expected structure
                observed_fvals_df = self.init_observed_fvals.copy()

                if observed_fvals_df.empty or "score" not in observed_fvals_df.columns:
                    return False

                # Initialize LLAMBO with observed data
                self.LLAMBO_instance._initialize(None, observed_configs_df, observed_fvals_df)
                self._llambo_initialized = True
                return True
            except Exception:
                return False

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        """
        Hybrid sampling method that combines LLAMBO for numerical parameters and
        RandomSampler for categorical parameters.

        This method:
          - Splits the search space into numerical and categorical parts.
          - Uses LLAMBO (initialized on the numerical space) to sample numerical parameters.
          - Uses RandomSampler to sample categorical parameters.
          - When passing observed configurations to LLAMBO, only numerical columns are retained.
          - Returns the merged configuration.

        Args:
            study: Optuna study object.
            trial: Current trial being sampled for.
            search_space: Search space to sample from.

        Returns:
            Dictionary mapping parameter names to sampled values.
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

        # Thread safety: ensure all required attributes are initialized
        if not hasattr(self, "lower_is_better") or self.lower_is_better is None:
            if study.direction is not None:
                self.lower_is_better = study.direction == optuna.study.StudyDirection.MINIMIZE
            else:
                # If multi-objective, just use first direction
                self.lower_is_better = study.directions[0] == optuna.study.StudyDirection.MINIMIZE

        if not hasattr(self, "init_configs") or not self.init_configs:
            self.init_configs = self.generate_random_samples(
                numerical_space, self.n_initial_samples
            )

        # Always collect completed trials data - do this regardless of trial number
        self._collect_initial_data(study)

        # For initial trials, use the pre-generated random samples
        if trial.number < self.n_initial_samples:
            # Safely get a configuration - handle out-of-bounds index
            config_idx = min(trial.number, len(self.init_configs) - 1)
            config = self.init_configs[config_idx]
            return {**config, **categorical_params}

        # For trials after initial phase, handle LLAMBO initialization and usage

        # Step 1: Initialize LLAMBO if not done already
        if self.LLAMBO_instance is None:
            self._initialize_llambo(numerical_space)

        # Step 2: Initialize LLAMBO with data if we have enough trials and haven't done so
        if (
            not self._llambo_initialized
            and self.LLAMBO_instance is not None
            and self._min_trials_processed >= self.n_initial_samples
        ):
            self._initialize_llambo_with_data(numerical_space)

        # Step 3: Use LLAMBO to sample if it's initialized, otherwise use random sampling
        try:
            if not self._llambo_initialized or self.LLAMBO_instance is None:
                numerical_params = self.generate_random_samples(numerical_space, 1)[0]
            else:
                result = self._sample_parameters()
                if not result:
                    numerical_params = self.generate_random_samples(numerical_space, 1)[0]
                else:
                    numerical_params = result

            # Ensure integer values are actually integers
            for param_name, value in numerical_params.items():
                if param_name in numerical_space and isinstance(
                    numerical_space[param_name], optuna.distributions.IntDistribution
                ):
                    numerical_params[param_name] = int(value)

        except Exception:
            # If LLAMBO sampling fails, fall back to random sampling
            numerical_params = self.generate_random_samples(numerical_space, 1)[0]

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
        Update LLAMBO's history after trial completion.

        Only updates with the numerical portion of the parameters.

        Args:
            study: Optuna study object.
            trial: Completed trial.
            state: Trial state.
            values: Trial values.
        """
        if state == optuna.trial.TrialState.COMPLETE and values is not None:
            # Update data collection first
            self._collect_initial_data(study)

            # Try to initialize LLAMBO if needed and possible
            if not self._llambo_initialized or self.LLAMBO_instance is None:
                if self._min_trials_processed >= self.n_initial_samples:
                    # Re-create numerical_space
                    numerical_space, _ = self._split_search_space(study.sampler._search_space)

                    if not self.LLAMBO_instance:
                        self._initialize_llambo(numerical_space)

                    if self.LLAMBO_instance and not self._llambo_initialized:
                        self._initialize_llambo_with_data(numerical_space)

            # Update LLAMBO with trial result
            if self._llambo_initialized and self.LLAMBO_instance is not None:
                # Filter trial.params to keep only numerical keys
                filtered_params = {
                    k: v for k, v in trial.params.items() if k in self.hyperparameter_constraints
                }
                try:
                    self.LLAMBO_instance.update_history(filtered_params, values[0])
                except Exception:
                    pass

    def generate_random_samples(
        self, search_space: dict[str, optuna.distributions.BaseDistribution], num_samples: int = 1
    ) -> list[dict[str, Any]]:
        """
        Generate random samples for the numerical search space.

        Uses the RandomSampler to generate initial samples for the search space.

        Args:
            search_space: Search space to sample from.
            num_samples: Number of samples to generate.

        Returns:
            List of dictionaries with parameter names and values.
        """
        samples = []
        for _ in range(num_samples):
            params = {}
            for param_name, distribution in search_space.items():
                params[param_name] = self._random_sampler.sample_independent(
                    study=None,  # Not needed for random sampling
                    trial=None,  # Not used in this context
                    param_name=param_name,
                    param_distribution=distribution,
                )
            samples.append(params)
        return samples
