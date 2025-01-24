from llambo.acquisition_function import LLM_ACQ
from llambo.discriminative_sm import LLM_DIS_SM
from llambo.generative_sm import LLM_GEN_SM
from llambo.rate_limiter import RateLimiter
from llambo.warping import NumericalTransformer
import pandas as pd


class LLAMBO:
    def __init__(
        self,
        task_context: dict,
        sm_mode: str,
        n_candidates: int,
        n_templates: int,
        n_gens: int,
        alpha: float,
        n_initial_samples: int,
        n_trials: int,
        top_pct: float = None,
        use_input_warping: bool = False,
        prompt_setting: str = None,
        shuffle_features: bool = False,
        key: str = "",
        model: str = "",
    ):
        self.task_context = task_context
        self.lower_is_better = task_context["lower_is_better"]
        self.n_candidates = n_candidates
        self.n_template = n_templates
        self.n_gens = n_gens
        self.alpha = alpha
        self.n_initial_samples = n_initial_samples
        self.n_trials = n_trials
        self.key = key
        self.model = model
        self.current_trial = 0
        self.start_time = None

        # Initialize observation tracking
        self.observed_configs = pd.DataFrame()
        self.observed_fvals = pd.DataFrame()

        # Initialize best values with safe defaults
        if self.lower_is_better:
            self.best_fval = float("inf")
            self.best_gen_fval = float("inf")
        else:
            self.best_fval = -float("inf")
            self.best_gen_fval = -float("inf")

        # Initialize components
        warping_transformer = (
            NumericalTransformer(task_context["hyperparameter_constraints"])
            if use_input_warping
            else None
        )

        rate_limiter = RateLimiter(max_tokens=100000, time_frame=60, max_requests=1)

        # Initialize surrogate model
        if sm_mode == "generative":
            self.surrogate_model = LLM_GEN_SM(
                task_context,
                n_gens,
                self.lower_is_better,
                top_pct,
                n_templates=n_templates,
                rate_limiter=None,
                key=self.key,
                model=self.model,
            )
        else:
            self.surrogate_model = LLM_DIS_SM(
                task_context,
                n_gens,
                self.lower_is_better,
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
            task_context,
            n_candidates,
            n_templates,
            self.lower_is_better,
            rate_limiter=rate_limiter,
            warping_transformer=warping_transformer,
            prompt_setting=prompt_setting,
            shuffle_features=shuffle_features,
            key=self.key,
            model=self.model,
        )

    def _initialize(
        self,
        init_configs: pd.DataFrame = None,
        observed_configs: pd.DataFrame = None,
        observed_fvals: pd.DataFrame = None,
        test_metric: str = "generalization_score",
    ) -> tuple:
        # Handle missing generalization_score column
        if observed_fvals is not None and not observed_fvals.empty:
            # Create generalization_score if it doesn't exist
            if test_metric not in observed_fvals.columns:
                observed_fvals[test_metric] = observed_fvals["score"]

        if observed_configs is not None and observed_fvals is not None:
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
            if init_configs is None:
                init_configs = self.init_f(self.n_initial_samples)

            # Evaluate initial configurations
            for config in init_configs:
                config_df = pd.DataFrame([config])
                result_df = pd.DataFrame(
                    [
                        {
                            "score": 0.0,
                            "generalization_score": 0.0,  # Add both columns
                        }
                    ]
                )  # Replace with actual evaluation
                self._update_observations(config_df, result_df)

        return 0, 0.0  # Return dummy cost/time values

    def _update_observations(self, new_config: pd.DataFrame, new_fval: pd.DataFrame) -> None:
        self.observed_configs = pd.concat([self.observed_configs, new_config], ignore_index=True)
        self.observed_fvals = pd.concat([self.observed_fvals, new_fval], ignore_index=True)

    def sample_configurations(self) -> dict:
        if self.current_trial >= self.n_trials:
            return None

        # Get candidate points
        candidate_points, _, _ = self.acq_func.get_candidate_points(
            self.observed_configs, self.observed_fvals[["score"]], alpha=self.alpha
        )

        # Select best candidate
        sel_candidate_point, _, _ = self.surrogate_model.select_query_point(
            self.observed_configs, self.observed_fvals[["score"]], candidate_points
        )

        return sel_candidate_point.to_dict(orient="records")[0]

    def update_history(self, eval_config: dict, eval_result: float) -> None:
        # Process evaluation results
        config_df = pd.DataFrame([eval_config])
        result_df = pd.DataFrame(
            [
                {
                    "score": eval_result,
                    "generalization_score": eval_result,  # Add duplicate column
                }
            ]
        )

        # Update observations
        self._update_observations(config_df, result_df)

        # Update best values (keep existing logic)
        current_score = eval_result
        if self.lower_is_better:
            score_improved = current_score < self.best_fval
        else:
            score_improved = current_score > self.best_fval

        if self.best_fval in (float("inf"), -float("inf")):
            self.best_fval = current_score
            self.best_gen_fval = current_score  # Update both values
            score_improved = True
        elif score_improved:
            self.best_fval = current_score
            self.best_gen_fval = current_score  # Keep them synchronized

        self.current_trial += 1

    def _evaluate_config_step(self, config: dict) -> tuple:
        return pd.DataFrame([config]), pd.DataFrame([{"score": 0.0}])
