from __future__ import annotations

from collections.abc import Sequence
import random
import threading
from typing import Any
from typing import Literal
from typing import Optional
import warnings

from confopt.selection.acquisition import QuantileConformalSearcher
from confopt.selection.sampling.expected_improvement_samplers import ExpectedImprovementSampler
from confopt.selection.sampling.thompson_samplers import ThompsonSampler
from confopt.utils.configurations.encoding import ConfigurationEncoder
from confopt.utils.configurations.sampling import get_tuning_configurations
from confopt.utils.configurations.utils import create_config_hash
from confopt.wrapping import CategoricalRange
from confopt.wrapping import FloatRange
from confopt.wrapping import IntRange
from confopt.wrapping import ParameterRange
import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub


def fetch_metric_sign(study: Study) -> int:
    """Determine the metric sign based on study direction.

    Args:
        study: The Optuna study object.

    Returns:
        int: 1 for minimization problems, -1 for maximization problems.
    """
    minimize = study.directions[0].value == StudyDirection.MINIMIZE
    return 1 if minimize else -1


def convert_to_confopt_space(
    optuna_search_space: dict[str, BaseDistribution],
) -> dict[str, ParameterRange]:
    """Convert Optuna search space to ConfOpt parameter ranges.

    Args:
        optuna_search_space: Dictionary mapping parameter names to Optuna distributions.

    Returns:
        dict[str, ParameterRange]: Dictionary mapping parameter names to ConfOpt ranges.

    Raises:
        ValueError: If an unsupported distribution type is encountered.
    """
    confopt_search_space = {}

    for name, distribution in optuna_search_space.items():
        if isinstance(distribution, CategoricalDistribution):
            confopt_search_space[name] = CategoricalRange(choices=list(distribution.choices))
        elif isinstance(distribution, IntDistribution):
            confopt_search_space[name] = IntRange(
                min_value=distribution.low,
                max_value=distribution.high,
                log_scale=distribution.log,
            )
        elif isinstance(distribution, FloatDistribution):
            if distribution.step is not None:
                warnings.warn("`step` will be ignored in ConfOpt hyperparameter mapping.")
            confopt_search_space[name] = FloatRange(
                min_value=distribution.low,
                max_value=distribution.high,
                log_scale=distribution.log,
            )
        else:
            raise ValueError(
                f"Unknown distribution type: {type(distribution)}. ConfOpt can only handle float, integer and categorical hyperparameters."
            )

    return confopt_search_space


def fetch_latest_configurations_and_values(
    trials: list[FrozenTrial], param_names: list[str]
) -> tuple[list[dict], list[float], set[str]]:
    """Extract configurations and values from completed trials.

    Args:
        trials: List of trials to process.
        param_names: List of parameter names to extract.

    Returns:
        tuple: A tuple containing:
            - list[dict]: Configurations from completed trials
            - list[float]: Corresponding objective values
            - set[str]: Set of configuration hashes for duplicate checking
    """
    latest_configs = []
    latest_values = []
    searched_configs_hashes = set()
    for trial in trials:
        if trial.value is not None and set(param_names).issubset(set(trial.params.keys())):
            config = {name: trial.params[name] for name in param_names}
            latest_configs.append(config)
            latest_values.append(trial.value)
            searched_configs_hashes.add(create_config_hash(config))

    return latest_configs, latest_values, searched_configs_hashes


def get_filtered_candidates(
    confopt_space: dict[str, ParameterRange], searched_configs_hashes: set[str], n_candidates: int
) -> list[dict]:
    """Generate candidate configurations filtered to avoid duplicates.

    Args:
        confopt_space: ConfOpt parameter space.
        searched_configs_hashes: Set of already evaluated configuration hashes.
        n_candidates: Number of unique candidate configurations to return.

    Returns:
        list[dict]: List of un-evaluated, randomly sampled candidate configurations.
    """
    n_to_generate = n_candidates + len(searched_configs_hashes)
    candidates = get_tuning_configurations(
        parameter_grid=confopt_space, n_configurations=n_to_generate, sampling_method="uniform"
    )

    filtered_candidates = []
    for config in candidates:
        config_hash = create_config_hash(config)
        if config_hash not in searched_configs_hashes:
            filtered_candidates.append(config)
            if len(filtered_candidates) >= n_candidates:
                break

    return filtered_candidates


def select_best_configuration(
    searcher_object: QuantileConformalSearcher,
    candidate_configs: list[dict],
    encoded_candidates: np.ndarray,
) -> dict:
    """Select the best candidate configuration using the trained searcher.

    Args:
        searcher_object: Trained QuantileConformalSearcher instance.
        candidate_configs: List of candidate configurations.
        encoded_candidates: Encoded candidate configurations as numpy array.

    Returns:
        dict: The selected configuration with the best predicted performance.
    """
    predictions = searcher_object.predict(X=encoded_candidates)
    next_idx = np.argmin(predictions)

    return candidate_configs[next_idx]


class ConfOptSampler(optunahub.samplers.SimpleBaseSampler):  # type: ignore[name-defined]
    """ConfOpt wrapper providing conformally calibrated quantile regression HPO.

    ConfOpt leverages quantile regression surrogates, adjusted with conformal prediction
    to allow for robust integration with acquisition functions like Thompson Sampling
    and Expected Improvement.

    Args:
        search_space: Dictionary mapping parameter names to their distributions.
        searcher: Quantile regression architecture ("qgbm" or "qe", representing
        either a Quantile Gradient Boosting Machine or Quantile Ensemble).
        acquisition_function: Acquisition function to use ("thompson_sampling",
            "optimistic_bayesian_sampling", or "expected_improvement").
        n_candidates: Number of candidate configurations to generate for selection.
        n_startup_trials: Number of random trials before using the surrogate model.
        train_on_pruned_trials: Whether to include pruned trials in model training.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution],
        searcher: Literal["qgbm", "qe"] = "qgbm",
        acquisition_function: Literal[
            "thompson_sampling", "optimistic_bayesian_sampling", "expected_improvement"
        ] = "optimistic_bayesian_sampling",
        n_candidates: int = 3000,
        n_startup_trials: int = 10,
        train_on_pruned_trials: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(search_space)

        # Convert Optuna search space to ConfOpt hyperparameter ranges:
        self.confopt_space = convert_to_confopt_space(optuna_search_space=search_space)
        self.param_names = list(search_space.keys())
        # NOTE: Encoder is not mutated after init, so safe to share across threads
        self.encoder = ConfigurationEncoder(self.confopt_space)

        self.searcher = searcher
        self.acquisition_function = acquisition_function
        self.searcher_config = self.get_searcher_config()

        self.n_candidates = n_candidates
        self.n_startup_trials = n_startup_trials
        self.train_on_pruned_trials = train_on_pruned_trials

        self.searched_config_hashes: set[str] = set()

        # Create the main searcher object for centralized state updates
        self.searcher_object = QuantileConformalSearcher(**self.searcher_config)

        # Thread-local storage for searcher copies (for transient fit/predict operations)
        self._local = threading.local()
        # Lock to protect permanent state updates in after_trial
        self._update_lock = threading.Lock()

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def get_searcher_config(self) -> dict:
        """Get the configuration parameters for creating QuantileConformalSearcher instances.

        Returns:
            dict: Configuration parameters for searcher creation.
        """
        adapter = "DtACI"
        if self.searcher == "qgbm":
            quantile_estimator_architecture = "qgbm"
            max_n_quantiles = 10
            calibration_split_strategy = "adaptive"
        elif self.searcher == "qe":
            quantile_estimator_architecture = "qens5"
            max_n_quantiles = 6
            calibration_split_strategy = "train_test_split"
        else:
            raise ValueError(f"Invalid searcher: {self.searcher}")

        if self.acquisition_function == "thompson_sampling":
            acquisition_object = ThompsonSampler(
                n_quantiles=min(6, max_n_quantiles),
                adapter=adapter,
                enable_optimistic_sampling=False,
            )
        elif self.acquisition_function == "optimistic_bayesian_sampling":
            acquisition_object = ThompsonSampler(
                n_quantiles=min(6, max_n_quantiles),
                adapter=adapter,
                enable_optimistic_sampling=True,
            )
        elif self.acquisition_function == "expected_improvement":
            acquisition_object = ExpectedImprovementSampler(
                n_quantiles=min(10, max_n_quantiles),
                adapter=adapter,
                num_ei_samples=1000,
            )
        else:
            raise ValueError(f"Invalid acquisition function: {self.acquisition_function}")

        return {
            "quantile_estimator_architecture": quantile_estimator_architecture,
            "sampler": acquisition_object,
            "calibration_split_strategy": calibration_split_strategy,
            "n_calibration_folds": 5,
            "n_pre_conformal_trials": 32,
        }

    def get_thread_searcher_copy(self) -> QuantileConformalSearcher:
        """Get or create a thread-local copy of the searcher for transient operations.

        This copy is used for fit/predict operations in sample_relative, where mutations
        are transient and should not affect permanent state between trials.

        Returns:
            QuantileConformalSearcher: Thread-local searcher copy.
        """
        if not hasattr(self._local, "searcher_copy"):
            self._local.searcher_copy = QuantileConformalSearcher(**self.searcher_config)

        return self._local.searcher_copy

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        """Sample a new configuration for the given trial.

        Args:
            study: The Optuna study object.
            trial: The current trial to sample parameters for.
            search_space: Dictionary mapping parameter names to their distributions.

        Returns:
            dict[str, Any]: Dictionary of sampled parameter values.

        Raises:
            NotImplementedError: If multi-objective optimization is attempted.
        """
        if len(study.directions) > 1:
            raise NotImplementedError(
                "At present, ConfOptSampler only supports single-objective optimization. Please rephrase your problem as a single-objective optimization problem, or consider using another sampler."
            )

        metric_sign = fetch_metric_sign(study=study)

        if self.train_on_pruned_trials:
            # NOTE: Currently trains on all trials in pooled fashion. In future it would be
            # good to explicitly model the fidelity.
            surrogate_trials = study.get_trials(states=[TrialState.COMPLETE, TrialState.PRUNED])
        else:
            surrogate_trials = study.get_trials(states=[TrialState.COMPLETE])

        completed_configs, completed_values, searched_configs_hashes = (
            fetch_latest_configurations_and_values(
                trials=surrogate_trials, param_names=self.param_names
            )
        )
        candidate_configs = get_filtered_candidates(
            confopt_space=self.confopt_space,
            searched_configs_hashes=searched_configs_hashes,
            n_candidates=self.n_candidates,
        )

        if len(completed_configs) < self.n_startup_trials:
            suggested_config = candidate_configs[0]
        else:
            encoded_configs = self.encoder.transform(configurations=completed_configs).to_numpy()
            sign_adj_values = np.array(completed_values) * metric_sign
            searcher_copy = self.get_thread_searcher_copy()
            searcher_copy.fit(X=encoded_configs, y=sign_adj_values)

            encoded_candidates = self.encoder.transform(
                configurations=candidate_configs
            ).to_numpy()
            suggested_config = select_best_configuration(
                searcher_object=searcher_copy,
                candidate_configs=candidate_configs,
                encoded_candidates=encoded_candidates,
            )

        return suggested_config

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        """Update the searcher with the trial result.

        Args:
            study: The Optuna study object.
            trial: The completed trial.
            state: The final state of the trial.
            values: The objective values from the trial.

        Raises:
            ValueError: If multi-objective optimization is attempted.
        """
        if not state == TrialState.FAIL:
            if (
                self.train_on_pruned_trials and state == TrialState.PRUNED
            ) or state == TrialState.COMPLETE:
                if values is not None:
                    if len(values) > 1:
                        raise ValueError(
                            "ConfOptSampler only supports single-objective optimization. Please rephrase your problem as a single-objective optimization problem, or consider using another sampler."
                        )

                    metric_sign = fetch_metric_sign(study=study)
                    config = {name: trial.params[name] for name in self.param_names}

                    encoded_config = self.encoder.transform(configurations=[config]).to_numpy()
                    signed_performance = values[0] * metric_sign

                    # Lock-protected to avoid conflicts when writing to shared state.
                    # NOTE: This is a quick operation, so lock doesn't impact performance.
                    with self._update_lock:
                        self.searcher_object.update(
                            X=encoded_config.flatten(), y_true=signed_performance
                        )
