import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd

from .acquisition_function import LLM_ACQ
from .discriminative_sm import LLMDiscriminativeSM
from .generative_sm import LLMGenerativeSM
from .warping import NumericalTransformer


# Define type variables for surrogate models to solve type annotation issues
T = TypeVar("T")


class LLAMBO:
    def __init__(
        self,
        task_context: Dict[str, Any],
        sm_mode: str,
        n_candidates: int = 10,
        num_prompt_variants: int = 2,
        n_gens: int = 10,
        alpha: float = 0.1,
        n_initial_samples: int = 5,
        top_pct: Optional[float] = None,
        use_input_warping: bool = False,
        prompt_setting: Optional[str] = None,
        shuffle_features: bool = False,
        key: str = "",
        model: str = "",
        max_requests_per_minute: int = 100,
    ) -> None:
        """
        Initialize the LLAMBO optimizer for hyperparameter tuning using large language models.

        Args:
            task_context (Dict[str, Any]): Dictionary containing context information for the task,
                including hyperparameter constraints and an optional task description. It should also
                include a "lower_is_better" key to indicate whether the optimization is a minimization
                (True) or maximization (False) problem.
            sm_mode (str): The surrogate model mode to use; typically "generative" for a generative
                model or another value for a discriminative model.
            n_candidates (int, optional): Total number of candidate configurations to generate.
                These candidates will be evaluated by the acquisition function. Defaults to 10.
            num_prompt_variants (int, optional): Number of distinct prompt variants (i.e., different
                few-shot prompt templates) to be used. Each variant is sent as a separate inquiry to
                the LLM to increase response diversity. Defaults to 2.
            n_gens (int, optional): Number of generations (or suggestions) produced per inquiry.
                This value is often used internally to divide n_candidates evenly across prompt variants.
                Defaults to 10.
            alpha (float, optional): Exploration-exploitation trade-off parameter used in the acquisition
                function to adjust the target performance. Defaults to 0.1.
            n_initial_samples (int, optional): Number of initial random configurations to sample before
                the surrogate model is used for further sampling. Defaults to 5.
            top_pct (Optional[float], optional): The top percentage threshold of configurations considered
                as high-performing; if not provided, a default value of 0.2 is used.
            use_input_warping (bool, optional): Flag to indicate whether numerical hyperparameters should be
                transformed (warped) before optimization. Defaults to False.
            prompt_setting (Optional[str], optional): Optional parameter to customize prompt details for
                the LLM. Defaults to None.
            shuffle_features (bool, optional): If True, randomizes the order of hyperparameter features in prompts
                to reduce potential bias. Defaults to False.
            key (str, optional): API key for accessing the LLM service.
            model (str, optional): Identifier for the LLM model to be used.
            max_requests_per_minute (int, optional): Maximum number of API requests allowed per minute.
                Used to calculate inter-component delays. Defaults to 100.

        This constructor initializes the LLAMBO framework by setting up observation tracking,
        computing an inter-component delay based on the provided rate limit, and configuring the surrogate
        model (either generative or discriminative) along with the acquisition function. It integrates
        diverse prompt variants (via num_prompt_variants) to ensure robust and varied inquiries to the LLM.
        """
        # Copy and update task context with the optimization direction.
        self.task_context = dict(task_context)
        self.task_context.setdefault("lower_is_better", True)  # default to minimization
        self.lower_is_better = self.task_context["lower_is_better"]

        self.n_candidates = n_candidates
        self.num_prompt_variants = num_prompt_variants
        self.n_gens = n_gens
        self.alpha = alpha
        self.n_initial_samples = n_initial_samples
        self.key = key
        self.model = model
        self.max_requests_per_minute = max_requests_per_minute
        self.sm_mode = sm_mode

        # Initialize accumulated cost tracking
        self.accumulated_cost = 0.0
        # Initialize current trial counter
        self.current_trial = 0

        # Delay between API calls based on rate limit.
        self.delay_seconds = 60.0 / max_requests_per_minute if max_requests_per_minute > 0 else 0
        print(f"Inter-component delay set to {self.delay_seconds:.2f} seconds.")

        # Initialize observation tracking.
        self.observed_configs = pd.DataFrame()
        self.observed_fvals = pd.DataFrame(columns=["score", "generalization_score"])

        # Set initial best values.
        if self.lower_is_better:
            self.best_fval = float("inf")
            self.best_gen_fval = float("inf")
        else:
            self.best_fval = -float("inf")
            self.best_gen_fval = -float("inf")

        warping_transformer = None
        if use_input_warping:
            warping_transformer = NumericalTransformer(
                self.task_context["hyperparameter_constraints"]
            )

        effective_top_pct = 0.2 if top_pct is None else top_pct

        # Declare the surrogate_model type once
        self.surrogate_model: Union[LLMGenerativeSM, LLMDiscriminativeSM]

        # Initialize the surrogate model based on the mode.
        if sm_mode == "generative":
            self.surrogate_model = LLMGenerativeSM(
                task_context=self.task_context,
                n_gens=n_gens,
                lower_is_better=self.lower_is_better,
                top_pct=effective_top_pct,
                num_prompt_variants=num_prompt_variants,
                key=self.key,
                model=self.model,
                max_requests_per_minute=self.max_requests_per_minute,
            )
        else:
            self.surrogate_model = LLMDiscriminativeSM(
                task_context=self.task_context,
                n_gens=n_gens,
                lower_is_better=self.lower_is_better,
                num_prompt_variants=num_prompt_variants,
                warping_transformer=warping_transformer,
                prompt_setting=prompt_setting,
                shuffle_features=shuffle_features,
                key=self.key,
                model=self.model,
                max_requests_per_minute=self.max_requests_per_minute,
            )

        # Initialize acquisition function similarly.
        self.acq_func = LLM_ACQ(
            task_context=self.task_context,
            n_candidates=n_candidates,
            num_prompt_variants=num_prompt_variants,
            lower_is_better=self.lower_is_better,
            key=self.key,
            model=self.model,
            max_requests_per_minute=self.max_requests_per_minute,
            shuffle_features=shuffle_features,
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

    def sample_configurations(self) -> Optional[Dict[str, Any]]:
        """
        Sample new configurations using the acquisition function and surrogate model.

        Returns:
            Dictionary containing the selected configuration
        """

        # Ensure we have at least one observation
        if self.observed_fvals.empty:
            raise ValueError("No observations available for sampling")

        # Get candidate points using acquisition function
        # The acquisition function might return 3 or 4 values, so we need to handle this properly
        acquisition_result = self.acq_func.get_candidate_points(
            self.observed_configs,
            self.observed_fvals[["score"]],
            alpha=self.alpha,
            current_trial=self.current_trial,
            n_initial_samples=self.n_initial_samples,
        )

        # Handle the case where get_candidate_points returns 3 or 4 values
        if len(acquisition_result) == 3:
            candidate_points, acq_cost, acq_time = acquisition_result
            # Track acquisition function cost
            self.accumulated_cost += acq_cost
            print(f"Acquisition function cost: ${acq_cost:.4f}")
        else:
            # If there are 4 values, extract just the ones we need
            candidate_points = acquisition_result[0]
            # If cost is provided as second value, track it
            if len(acquisition_result) > 1 and isinstance(acquisition_result[1], (int, float)):
                acq_cost = acquisition_result[1]
                self.accumulated_cost += acq_cost
                print(f"Acquisition function cost: ${acq_cost:.4f}")

        # Add delay between acquisition function and surrogate model
        print(f"Inserting delay of {self.delay_seconds:.2f} seconds between ACQ and SM components")
        time.sleep(self.delay_seconds)

        # Check the surrogate model type by stored mode
        if self.sm_mode == "generative":
            # For generative SM, we need to use asyncio to run the coroutine
            import asyncio
            import inspect

            # Get the select_query_point method
            query_point_method = self.surrogate_model.select_query_point(
                self.observed_configs, self.observed_fvals[["score"]], candidate_points
            )

            # Check if the result is a coroutine or already the result
            if inspect.iscoroutine(query_point_method):
                # If it's a coroutine, run it with asyncio
                result = asyncio.run(query_point_method)
            else:
                # If it's already the result, use it directly
                result = query_point_method

            # The result might have 3 or 4 elements, extract only what we need
            if len(result) >= 3:
                sel_candidate_point = result[0]
                # Extract and track surrogate model cost
                sm_cost = result[1]
                self.accumulated_cost += sm_cost
                print(f"Surrogate model cost: ${sm_cost:.4f}")
            else:
                # Handle unexpected return format
                raise ValueError(f"Unexpected return format from select_query_point: {result}")
        else:
            # For discriminative SM, call the method directly
            result = self.surrogate_model.select_query_point(
                self.observed_configs, self.observed_fvals[["score"]], candidate_points
            )

            # Extract the first 3 elements regardless of how many are returned
            if len(result) >= 3:
                sel_candidate_point = result[0]
                # Extract and track surrogate model cost
                sm_cost = result[1]
                self.accumulated_cost += sm_cost
                print(f"Surrogate model cost: ${sm_cost:.4f}")
            else:
                # Handle unexpected return format
                raise ValueError(f"Unexpected return format from select_query_point: {result}")

        # Print accumulated cost so far
        print(
            f"---------------------Accumulated cost so far: ${self.accumulated_cost:.4f}---------------------"
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
