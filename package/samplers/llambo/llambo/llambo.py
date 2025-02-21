from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from llambo.acquisition_function import LLM_ACQ
from llambo.discriminative_sm import LLMDiscriminativeSM
from llambo.generative_sm import LLMGenerativeSM
from llambo.rate_limiter import RateLimiter
from llambo.warping import NumericalTransformer
import numpy as np
import pandas as pd


class LLAMBO:
    """
    Language Models for Bayesian Optimization (LLAMBO)

    This class implements the LLAMBO algorithm, which combines language models
    with Bayesian optimization for hyperparameter optimization.
    """

    def __init__(
        self,
        task_context: Dict[str, Any],
        sm_mode: str,
        n_candidates: int = 10,
        n_templates: int = 2,
        n_gens: int = 10,
        alpha: float = 0.1,
        n_initial_samples: int = 5,
        n_trials: int = 100,
        top_pct: Optional[float] = None,
        use_input_warping: bool = False,
        prompt_setting: Optional[str] = None,
        shuffle_features: bool = False,
        key: str = "",
        model: str = "",
    ) -> None:
        """
        Initialize LLAMBO optimizer.

        Args:
            task_context: Dictionary containing task-specific information
            sm_mode: Surrogate model mode ("generative" or "discriminative")
            n_candidates: Number of candidate points to generate
            n_templates: Number of templates to use for prompting
            n_gens: Number of generations/iterations
            alpha: Exploration-exploitation trade-off parameter
            n_initial_samples: Number of initial random samples
            n_trials: Total number of optimization trials
            top_pct: Top percentage for generative mode
            use_input_warping: Whether to use input warping
            prompt_setting: Custom prompt setting
            shuffle_features: Whether to shuffle features
            key: API key for language model
            model: Language model identifier
        """
        # Store initialization parameters
        self.task_context = task_context
        self.lower_is_better = task_context["lower_is_better"]
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = n_gens
        self.alpha = alpha
        self.n_initial_samples = n_initial_samples
        self.n_trials = n_trials
        self.key = key
        self.model = model

        # Initialize state variables
        self.current_trial = 0
        self.start_time = None

        # Initialize observation tracking with proper columns
        self.observed_configs = pd.DataFrame()
        self.observed_fvals = pd.DataFrame(columns=["score", "generalization_score"])

        # Initialize best values based on optimization direction
        if self.lower_is_better:
            self.best_fval = float("inf")
            self.best_gen_fval = float("inf")
        else:
            self.best_fval = -float("inf")
            self.best_gen_fval = -float("inf")

        # Initialize components
        warping_transformer = None
        if use_input_warping:
            warping_transformer = NumericalTransformer(task_context["hyperparameter_constraints"])

        # Setup rate limiting
        rate_limiter = RateLimiter(max_tokens=100000, time_frame=60, max_requests=1)

        # Initialize surrogate model based on mode
        if sm_mode == "generative":
            self.surrogate_model = LLMGenerativeSM(
                task_context=task_context,
                n_gens=n_gens,
                lower_is_better=self.lower_is_better,
                top_pct=top_pct,
                n_templates=n_templates,
                rate_limiter=None,  # Generative mode doesn't use rate limiting
                key=self.key,
                model=self.model,
            )
        else:  # discriminative mode
            self.surrogate_model = LLMDiscriminativeSM(
                task_context=task_context,
                n_gens=n_gens,
                lower_is_better=self.lower_is_better,
                n_templates=n_templates,
                rate_limiter=rate_limiter,
                warping_transformer=warping_transformer,
                prompt_setting=prompt_setting,
                shuffle_features=shuffle_features,
                key=self.key,
                model=self.model,
            )

        # Initialize acquisition function
        self.acq_func = LLM_ACQ(
            task_context=task_context,
            n_candidates=n_candidates,
            n_templates=n_templates,
            lower_is_better=self.lower_is_better,
            rate_limiter=rate_limiter,
            warping_transformer=warping_transformer,
            prompt_setting=prompt_setting,
            shuffle_features=shuffle_features,
            key=self.key,
            model=self.model,
        )

    def _initialize(
        self,
        init_configs: Optional[pd.DataFrame] = None,
        observed_configs: Optional[pd.DataFrame] = None,
        observed_fvals: Optional[pd.DataFrame] = None,
        test_metric: str = "generalization_score",
    ) -> Tuple[int, float]:
        """
        Initialize the optimizer with either provided or generated configurations.

        Args:
            init_configs: Initial configurations to evaluate
            observed_configs: Previously observed configurations
            observed_fvals: Previously observed function values
            test_metric: Metric name for generalization performance

        Returns:
            Tuple of (cost, time) for initialization
        """
        if observed_configs is not None and observed_fvals is not None:
            # Ensure observed_fvals has required columns
            if "score" not in observed_fvals.columns:
                observed_fvals["score"] = observed_fvals[test_metric]
            if test_metric not in observed_fvals.columns:
                observed_fvals[test_metric] = observed_fvals["score"]

            # Store observations
            self.observed_configs = observed_configs
            self.observed_fvals = observed_fvals

            # Initialize best values from provided data
            if not observed_fvals.empty:
                if self.lower_is_better:
                    self.best_fval = observed_fvals["score"].min()
                    self.best_gen_fval = observed_fvals[test_metric].min()
                else:
                    self.best_fval = observed_fvals["score"].max()
                    self.best_gen_fval = observed_fvals[test_metric].max()
        else:
            # Generate initial samples if not provided
            if init_configs is None or isinstance(init_configs, list):
                if isinstance(init_configs, list):
                    configs_to_evaluate = init_configs
                else:
                    # If no init_configs provided, generate random ones
                    configs_to_evaluate = self._generate_random_configs(self.n_initial_samples)

                # Evaluate initial configurations
                for config in configs_to_evaluate:
                    config_df = pd.DataFrame([config])
                    result_df = pd.DataFrame(
                        [
                            {
                                "score": 0.0,
                                "generalization_score": 0.0,
                            }
                        ]
                    )
                    self._update_observations(config_df, result_df)

        return 0, 0.0  # Return dummy cost/time values

    def _generate_random_configs(self, n_samples: int) -> list:
        """
        Generate random configurations based on hyperparameter constraints.

        Args:
            n_samples: Number of random configurations to generate

        Returns:
            List of random configurations
        """
        constraints = self.task_context["hyperparameter_constraints"]
        configs = []

        for _ in range(n_samples):
            config = {}
            for param_name, constraint in constraints.items():
                dtype, dist_type, bounds = constraint

                if dtype == "float":
                    if dist_type == "log":
                        value = np.exp(np.random.uniform(np.log(bounds[0]), np.log(bounds[1])))
                    else:  # linear
                        value = np.random.uniform(bounds[0], bounds[1])

                elif dtype == "int":
                    if dist_type == "log":
                        value = int(
                            np.exp(np.random.uniform(np.log(bounds[0]), np.log(bounds[1])))
                        )
                    else:  # linear
                        value = np.random.randint(bounds[0], bounds[1] + 1)

                elif dtype == "categorical":
                    value = np.random.choice(bounds)

                config[param_name] = value

            configs.append(config)

        return configs

    def _update_observations(self, new_config: pd.DataFrame, new_fval: pd.DataFrame) -> None:
        """
        Update the observation history with new results.

        Args:
            new_config: New configuration that was evaluated
            new_fval: Corresponding function values
        """
        # Ensure new_fval has required columns
        if "score" not in new_fval.columns:
            new_fval["score"] = 0.0
        if "generalization_score" not in new_fval.columns:
            new_fval["generalization_score"] = new_fval["score"]

        # Update observation history
        self.observed_configs = pd.concat([self.observed_configs, new_config], ignore_index=True)
        self.observed_fvals = pd.concat([self.observed_fvals, new_fval], ignore_index=True)

    def sample_configurations(self) -> Dict[str, Any]:
        """
        Sample new configurations using the acquisition function and surrogate model.

        Returns:
            Dictionary containing the selected configuration
        """
        if self.current_trial >= self.n_trials:
            return None

        # Ensure we have at least one observation
        if self.observed_fvals.empty:
            raise ValueError("No observations available for sampling")

        # Get candidate points using acquisition function
        candidate_points, _, _ = self.acq_func.get_candidate_points(
            self.observed_configs, self.observed_fvals[["score"]], alpha=self.alpha
        )

        # Check the surrogate model type by class name
        if self.surrogate_model.__class__.__name__ == "LLMGenerativeSM":
            # For generative SM, we need to use asyncio to run the coroutine
            import asyncio

            sel_candidate_point, _, _ = asyncio.run(
                self.surrogate_model.select_query_point(
                    self.observed_configs, self.observed_fvals[["score"]], candidate_points
                )
            )
        else:
            # For discriminative SM or any other model, call the method directly
            sel_candidate_point, _, _ = self.surrogate_model.select_query_point(
                self.observed_configs, self.observed_fvals[["score"]], candidate_points
            )

        return sel_candidate_point.to_dict(orient="records")[0]

    def update_history(self, eval_config: Dict[str, Any], eval_result: float) -> None:
        """
        Update optimization history with new evaluation results.

        Args:
            eval_config: Configuration that was evaluated
            eval_result: Result of the evaluation
        """
        # Process evaluation results with proper column structure
        config_df = pd.DataFrame([eval_config])
        result_df = pd.DataFrame(
            [
                {
                    "score": eval_result,
                    "generalization_score": eval_result,
                }
            ]
        )

        # Update observations
        self._update_observations(config_df, result_df)

        # Update best values
        current_score = eval_result
        if self.lower_is_better:
            score_improved = current_score < self.best_fval
        else:
            score_improved = current_score > self.best_fval

        if self.best_fval in (float("inf"), -float("inf")):
            self.best_fval = current_score
            self.best_gen_fval = current_score
            score_improved = True
        elif score_improved:
            self.best_fval = current_score
            self.best_gen_fval = current_score

        self.current_trial += 1

    def _evaluate_config_step(self, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate a single configuration step.

        Args:
            config: Configuration to evaluate

        Returns:
            Tuple of (configuration DataFrame, results DataFrame)
        """
        return pd.DataFrame([config]), pd.DataFrame([{"score": 0.0, "generalization_score": 0.0}])
