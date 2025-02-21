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
        n_trials: Total number of optimization trials.
        api_key: API key for the language model service.
        model: Language model identifier to use.
        search_space: Optional search space to sample from.
        debug: Whether to print debug information.
        seed: Random seed for reproducibility.
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

        # Initialize empty DataFrames instead of lists for thread-safety
        self.init_observed_configs = pd.DataFrame()
        # Initialize with the column structure required by LLAMBO
        self.init_observed_fvals = pd.DataFrame(columns=["score"])

        self.LLAMBO_instance = None

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
    ) -> None:
        """
        Initialize the LLAMBO instance using only the numerical portion of the search space.

        This method converts Optuna distributions to the format required by LLAMBO and
        creates the LLAMBO instance.

        Args:
            numerical_space: Dictionary of numerical parameter distributions.
        """
        self.hyperparameter_constraints = {}
        for param_name, distribution in numerical_space.items():
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                dtype = "float"
                dist_type = "log" if distribution.log else "linear"
                bounds = [distribution.low, distribution.high]
                print("DEBUG: the bounds for float:", bounds)
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

        Returns:
            A dictionary mapping parameter names to their sampled values.
        """
        sampled_configuration = self.LLAMBO_instance.sample_configurations()
        return sampled_configuration

    def _debug_print(self, message: str) -> None:
        """
        Print a debug message if debug mode is enabled.

        Args:
            message: Message to print.
        """
        if self.debug:
            print(message)

    def _calculate_speed(self, n_completed: int) -> None:
        """
        Calculate and print the optimization speed statistics.

        Args:
            n_completed: Number of completed trials.
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
        """Reset the random number generator seeds."""
        self._rng.rng.seed()
        self._random_sampler.reseed_rng()

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
        if not hasattr(self, "lower_is_better") or not hasattr(self, "init_configs"):
            self.lower_is_better = (
                True if study.direction == optuna.study.StudyDirection.MINIMIZE else False
            )
            self.init_configs = self.generate_random_samples(
                numerical_space, self.n_initial_samples
            )

        # Make sure LLAMBO is initialized
        if self.LLAMBO_instance is None:
            self._initialize_llambo(numerical_space)

        # If DataFrames aren't initialized yet, do it now
        if not hasattr(self, "init_observed_configs") or self.init_observed_configs is None:
            self.init_observed_configs = pd.DataFrame()

        if not hasattr(self, "init_observed_fvals") or self.init_observed_fvals is None:
            self.init_observed_fvals = pd.DataFrame(columns=["score"])

        # For initial trials, use the pre-generated random samples
        if trial.number <= self.n_initial_samples:
            # Safely get a configuration - handle out-of-bounds index
            config_idx = min(trial.number - 1, len(self.init_configs) - 1)
            config = self.init_configs[config_idx]

            # Update history with completed trials
            if trial.number > 1:
                completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
                if completed_trials:
                    # Find the most recent completed trial that we haven't recorded yet
                    # This is more robust than just taking the last one
                    for completed_trial in reversed(completed_trials):
                        # Skip trials we've already processed (check by trial number)
                        if self.init_observed_configs.shape[0] > 0:
                            # Skip if we've already seen this trial number
                            continue

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
                        break  # Process one trial at a time

            return {**config, **categorical_params}

        # For later trials, use the LLAMBO instance for sampling
        if trial.number == self.n_initial_samples + 1:
            # Check if we need to initialize LLAMBO with observed data
            if len(self.init_observed_configs) == 0:
                # Collect all completed trials data if we haven't done so yet
                completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
                for completed_trial in completed_trials:
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

            # Retain only numerical keys from observed configurations
            observed_configs_df = self.init_observed_configs.copy()
            if not observed_configs_df.empty:
                # Ensure we only keep columns that exist in numerical_space
                numerical_cols = [
                    col for col in numerical_space.keys() if col in observed_configs_df.columns
                ]
                observed_configs_df = observed_configs_df[numerical_cols]

            # Ensure observed_fvals_df has the expected structure
            observed_fvals_df = self.init_observed_fvals.copy()

            # Initialize LLAMBO with observed data
            try:
                self.LLAMBO_instance._initialize(None, observed_configs_df, observed_fvals_df)
            except Exception as e:
                # If initialization fails, fall back to random sampling
                self._debug_print(
                    f"LLAMBO initialization failed: {e}. Falling back to random sampling."
                )
                return {
                    **self.generate_random_samples(numerical_space, 1)[0],
                    **categorical_params,
                }

        # Use LLAMBO to sample numerical parameters.
        try:
            numerical_params = self.LLAMBO_instance.sample_configurations()
        except Exception as e:
            # If LLAMBO sampling fails, fall back to random sampling
            self._debug_print(f"LLAMBO sampling failed: {e}. Falling back to random sampling.")
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
