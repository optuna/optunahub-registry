from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
import numpy as np
import pandas as pd
from scipy.stats import norm

from .llm import OpenAI_interface
from .utils import apply_rate_limit


T = TypeVar("T")


def _count_decimal_places(n: float) -> int:
    """
    Count the number of decimal places in a number.

    Args:
        n (float): The number to count decimal places for.

    Returns:
        int: The number of decimal places in the input number.

    Example:
        >>> _count_decimal_places(3.14159)
        5
        >>> _count_decimal_places(42.0)
        0
    """
    s = format(n, ".10f")
    if "." not in s:
        return 0
    num_dp = len(s.split(".")[1].rstrip("0"))
    return num_dp


def prepare_configurations(
    hyperparameter_constraints: dict,
    observed_configs: pd.DataFrame,
    observed_fvals: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
    bootstrapping: bool = False,
    use_feature_semantics: bool = True,
    shuffle_features: bool = False,
    apply_warping: bool = False,
) -> list[dict[str, str]]:
    """
    Prepare and possibly shuffle the configurations for prompt templates.

    Args:
        hyperparameter_constraints (dict): Constraints for each hyperparameter.
        observed_configs (pd.DataFrame): Observed hyperparameter configurations.
        observed_fvals (Optional[pd.DataFrame]): Observed performance values.
        seed (Optional[int]): Random seed for shuffling.
        bootstrapping (bool): Whether to use bootstrap resampling.
        use_feature_semantics (bool): Whether to use feature names in output.
        shuffle_features (bool): Whether to shuffle feature order.
        apply_warping (bool): Whether to apply warping to numeric values.

    Returns:
        list[dict[str, str]]: List of prepared configuration examples.
    """
    examples: list[dict[str, str]] = []
    hyperparameter_names = observed_configs.columns
    observed_configs = observed_configs.copy()

    if seed is not None:
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(observed_configs.index)
        observed_configs = observed_configs.loc[shuffled_indices]
        if observed_fvals is not None:
            observed_fvals = observed_fvals.loc[shuffled_indices]

    if shuffle_features:
        np.random.seed(0)
        shuffled_indices = np.random.permutation(len(hyperparameter_names))
        observed_configs = observed_configs[hyperparameter_names[shuffled_indices]]

    if bootstrapping:
        observed_configs = observed_configs.sample(frac=1, replace=True, random_state=seed)
        if observed_fvals is not None:
            observed_fvals = observed_fvals.loc[observed_configs.index]

    observed_configs = observed_configs.reset_index(drop=True)
    if observed_fvals is not None:
        observed_fvals = observed_fvals.reset_index(drop=True)

    for index, row in observed_configs.iterrows():
        row_string = ""
        for i in range(len(row)):
            hyp_type = hyperparameter_constraints[hyperparameter_names[i]][0]
            hyp_trans = hyperparameter_constraints[hyperparameter_names[i]][1]

            if hyp_type in ["int", "float"]:
                lower_bound = hyperparameter_constraints[hyperparameter_names[i]][2][0]
            else:
                lower_bound = hyperparameter_constraints[hyperparameter_names[i]][2][1]

            # Get base precision from constraint
            n_dp = _count_decimal_places(lower_bound)

            # For float types, ensure we use appropriate precision
            if hyp_type == "float":
                # Get actual precision from the value itself
                actual_dp = _count_decimal_places(row[i])
                # Use at least 1 decimal place for floats, or more if value has more precision
                n_dp = max(1, n_dp, actual_dp)

            prefix = f"{hyperparameter_names[i]}" if use_feature_semantics else f"X{i + 1}"
            row_string += f"{prefix} is "

            if apply_warping:
                if hyp_type == "int" and hyp_trans != "log":
                    row_string += str(int(row[i]))
                elif hyp_type == "float" or hyp_trans == "log":
                    row_string += f"{row[i]:.{n_dp}f}"
                elif hyp_type == "ordinal":
                    row_string += f"{row[i]:.{n_dp}f}"
                else:
                    row_string += row[i]
            else:
                if hyp_type == "int":
                    row_string += str(int(row[i]))
                elif hyp_type == "float":
                    row_string += f"{row[i]:.{n_dp}f}"
                elif hyp_type == "ordinal":
                    row_string += f"{row[i]:.{n_dp}f}"
                else:
                    row_string += row[i]

            if i != len(row) - 1:
                row_string += ", "

        example = {"Q": row_string}
        if observed_fvals is not None:
            row_index = observed_fvals.index.get_loc(index)
            perf = f"## {observed_fvals.values[row_index][0]:.6f} ##"
            example["A"] = perf
        examples.append(example)

    return examples


def gen_prompt_templates(
    task_context: dict,
    observed_configs: pd.DataFrame,
    observed_fvals: pd.DataFrame,
    candidate_configs: pd.DataFrame,
    n_prompts: int = 1,
    bootstrapping: bool = False,
    use_feature_semantics: bool = True,
    shuffle_features: bool = False,
    apply_warping: bool = False,
) -> tuple[list[FewShotPromptTemplate], list[dict[str, str]]]:
    """
    Generate prompt templates for the few-shot learning task for the discriminative surrogate model.

    Args:
        task_context (dict): Context information for the task, which may include keys "n_initial_samples" and "current_trial".
        observed_configs (pd.DataFrame): Observed hyperparameter configurations.
        observed_fvals (pd.DataFrame): Observed performance values.
        candidate_configs (pd.DataFrame): Candidate configurations to evaluate.
        n_prompts (int): Number of prompt templates to generate.
        bootstrapping (bool): Whether to use bootstrap resampling.
        use_feature_semantics (bool): Whether to use feature names in output.
        shuffle_features (bool): Whether to shuffle feature order.
        apply_warping (bool): Whether to apply warping to numeric values.

    Returns:
        Tuple of:
          - A list of FewShotPromptTemplate objects.
          - A list of query examples (dicts).
    """
    custom_task_description = task_context.get("custom_task_description")
    all_prompt_templates: list[FewShotPromptTemplate] = []

    for i in range(n_prompts):
        few_shot_examples = prepare_configurations(
            task_context["hyperparameter_constraints"],
            observed_configs,
            observed_fvals,
            seed=i,
            bootstrapping=bootstrapping,
            use_feature_semantics=use_feature_semantics,
            shuffle_features=shuffle_features,
            apply_warping=apply_warping,
        )

        example_template = """
Hyperparameter configuration: {Q}
Performance: {A}"""

        example_prompt = PromptTemplate(
            input_variables=["Q", "A"],
            template=example_template,
        )

        prefix = (
            "The following are examples of hyperparameter configurations for a "
            "black-box optimization task. "
        )
        if custom_task_description is not None:
            prefix += "Below is a description of the task:\n" + custom_task_description + "\n"
        prefix += (
            "Your response should only contain the predicted performance in the "
            "format ## performance ##."
        )

        # Add adaptive random sampling warning based on task_context values.
        n_initial_samples = task_context.get("n_initial_samples", 0)
        if n_initial_samples > 0 and len(observed_configs) > 0:
            fraction_random = n_initial_samples / len(observed_configs)
            if fraction_random == 1.0:
                warning = "\nNote: All configurations above are based on uniform random sampling. Avoid following this random pattern."
            elif fraction_random >= 0.5:
                percent = int(fraction_random * 100)
                warning = f"\nNote: Approximately {percent}% of the configurations above are based on uniform random sampling. Avoid following this pattern."
            else:
                warning = ""
            prefix += warning

        suffix = """
Hyperparameter configuration: {Q}
Performance: """

        few_shot_prompt = FewShotPromptTemplate(
            examples=few_shot_examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["Q"],
            example_separator="",
        )
        all_prompt_templates.append(few_shot_prompt)

    query_examples = prepare_configurations(
        task_context["hyperparameter_constraints"],
        candidate_configs,
        seed=None,
        bootstrapping=False,
        use_feature_semantics=use_feature_semantics,
        shuffle_features=shuffle_features,
        apply_warping=apply_warping,
    )

    return all_prompt_templates, query_examples


class LLMDiscriminativeSM:
    """
    A discriminative surrogate model using Large Language Models.

    This class implements a surrogate model that models p(y|x) similar to Gaussian Process or
    SMAC approaches. It uses LLMs to generate predictions and uncertainty estimates for given
    configurations.

    Attributes:
        task_context (Any): Context information for the task being modeled.
        n_gens (int): Number of generations/predictions to make per query.
        lower_is_better (bool): Whether lower values are better for optimization.
        bootstrapping (bool): Whether to use bootstrapping for uncertainty estimation.
        num_prompt_variants (int): Number of distinct prompt variants.
        use_recalibration (bool): Whether to use recalibration for uncertainty estimates.
        warping_transformer (Optional[Any]): Transformer for feature warping.
        verbose (bool): Whether to print verbose output.
        prompt_setting (Optional[Any]): Settings for prompt generation.
        shuffle_features (bool): Whether to shuffle features in prompts.
        OpenAI_instance (OpenAI_interface): Interface for OpenAI API calls.
        recalibrator (Optional[Any]): Recalibrator for uncertainty estimates.
        apply_warping (bool): Whether to apply feature warping.

    Example:
        >>> model = LLMDiscriminativeSM(
        ...     task_context="optimization",
        ...     n_gens=5,
        ...     lower_is_better=True,
        ...     key="api-key",
        ...     model="gpt-4-mini"
        ... )
        >>> # Use model for predictions
    """

    def __init__(
        self,
        task_context: Any,
        n_gens: int,
        lower_is_better: bool,
        bootstrapping: bool = False,
        num_prompt_variants: int = 2,
        use_recalibration: bool = False,
        warping_transformer: Optional[Any] = None,
        verbose: bool = False,
        prompt_setting: Optional[Any] = None,
        shuffle_features: bool = False,
        key: str = "",
        model: str = "gpt-4o-mini",
        max_requests_per_minute: int = 100,
        azure: bool = False,
        azure_api_base: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLM discriminative surrogate model.

        Args:
            task_context: Context information for the task.
            n_gens: Number of generations per query.
            lower_is_better: Whether lower values are better.
            bootstrapping: Whether to use bootstrapping.
            num_prompt_variants (int): Number of distinct prompt variants.
            use_recalibration: Whether to use recalibration.
            warping_transformer: Transformer for feature warping.
            verbose: Whether to print verbose output.
            prompt_setting: Settings for prompt generation.
            shuffle_features: Whether to shuffle features.
            key: API key for OpenAI.
            model: Model identifier string.
            max_requests_per_minute: Maximum number of requests per minute
            azure: Whether to use Azure API endpoints.
            azure_api_base: Azure API base URL.
            azure_api_version: Azure API version.
            azure_deployment_name: Azure deployment name.

        Raises:
            AssertionError: If both bootstrapping and recalibration are enabled.
        """

        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.bootstrapping = bootstrapping
        self.num_prompt_variants = num_prompt_variants

        # Assert for compatibility
        if self.bootstrapping and use_recalibration:
            raise ValueError("Cannot enable both bootstrapping and recalibration simultaneously.")

        self.use_recalibration = use_recalibration
        self.warping_transformer = warping_transformer
        self.apply_warping = warping_transformer is not None
        self.recalibrator = None  # To be created later if needed
        self.verbose = verbose
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features

        # Azure parameters
        self.azure = azure
        self.azure_api_base = azure_api_base
        self.azure_api_version = azure_api_version
        self.azure_deployment_name = azure_deployment_name

        # Initialize OpenAI interface
        self.OpenAI_instance = OpenAI_interface(
            key,
            model=model,
            debug=False,
            azure=azure,
            azure_api_base=azure_api_base,
            azure_api_version=azure_api_version,
            azure_deployment_name=azure_deployment_name,
        )

        apply_rate_limit(self.OpenAI_instance, max_requests_per_minute=max_requests_per_minute)

        assert isinstance(self.shuffle_features, bool), "shuffle_features must be a boolean"

    async def _async_generate(
        self, few_shot_template: str, query_example: dict[str, Any], query_idx: int
    ) -> Optional[Tuple[int, Any, float]]:
        """
        Generate a response from the LLM asynchronously.

        Args:
            few_shot_template: Template for few-shot learning.
            query_example: Example to query.
            query_idx: Index of the query.

        Returns:
            Optional[Tuple[int, Any, float]]: Query index, response, and total cost
                if successful, None otherwise.

        Example:
            >>> # In an async function
            >>> template = "Example template {Q}"
            >>> query = {"Q": "sample query"}
            >>> result = model._async_generate(template, query, 0)
        """
        print("Sending inquiries to the LLM - discriminative surrogate model")

        message = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            },
            {
                "role": "user",
                "content": few_shot_template.format(Q=query_example["Q"]),
            },
        ]

        resp, tot_cost = self.OpenAI_instance.ask(message)

        return query_idx, resp, tot_cost

    async def _generate_concurrently(
        self, few_shot_templates: list[str], query_examples: list[dict[str, Any]]
    ) -> list[list[Any]]:
        """
        Perform concurrent generation of responses from the LLM asynchronously.

        Args:
            few_shot_templates: List of few-shot learning templates.
            query_examples: List of query examples.

        Returns:
            list[list[Any]]: Results for each query example.

        Example:
            >>> # In an async function
            >>> templates = ["Template 1 {Q}", "Template 2 {Q}"]
            >>> queries = [{"Q": "query 1"}, {"Q": "query 2"}]
            >>> results = model._generate_concurrently(templates, queries)
        """
        coroutines = [
            self._async_generate(template, query_example, query_idx)
            for template in few_shot_templates
            for query_idx, query_example in enumerate(query_examples)
        ]

        tasks = [asyncio.create_task(c) for c in coroutines]
        results: List[List[Any]] = [[] for _ in range(len(query_examples))]
        llm_response = await asyncio.gather(*tasks)

        for response in llm_response:
            if response is not None:
                query_idx, resp, tot_cost = response
                results[query_idx].append([resp, tot_cost])

        return results

    async def _predict(
        self, all_prompt_templates: list[str], query_examples: list[dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        Make predictions for query examples using prompt templates.

        Args:
            all_prompt_templates: List of prompt templates.
            query_examples: List of query examples.

        Returns:
            Tuple containing:
                - Mean predictions
                - Standard deviations (recalibrated if enabled)
                - Success rate
                - Total cost
                - Time taken
        """
        start = time.time()
        all_preds = []
        tot_cost = 0
        bool_pred_returned = []

        for i in range(0, len(query_examples), 5):
            query_chunk = query_examples[i : i + 5]
            chunk_results = await self._generate_concurrently(all_prompt_templates, query_chunk)
            bool_pred_returned.extend([1 if x is not None else 0 for x in chunk_results])

            for sample_response in chunk_results:
                if not sample_response:
                    sample_preds = [np.nan] * self.n_gens
                else:
                    sample_preds = []
                    for item in sample_response:
                        for gen_text in item:
                            gen_text = str(gen_text)
                            gen_pred = re.findall(r"## (-?[\d.]+) ##", gen_text)

                            if len(gen_pred) == 1:
                                sample_preds.append(float(gen_pred[0]))
                            else:
                                sample_preds.append(np.nan)

                    while len(sample_preds) < self.n_gens:
                        sample_preds.append(np.nan)

                    tot_cost += sum(x[1] for x in sample_response)
                all_preds.append(sample_preds)

        time_taken = time.time() - start
        success_rate = sum(bool_pred_returned) / len(bool_pred_returned)

        # Convert predictions to numpy array
        all_preds = np.array(all_preds).astype(float)

        # Calculate y_mean and y_std
        y_mean = np.nanmean(all_preds, axis=1)
        y_std = np.nanstd(all_preds, axis=1)

        # Handling NaNs in predictions
        y_mean[np.isnan(y_mean)] = np.nanmean(y_mean)
        y_std[np.isnan(y_std)] = np.nanmean(y_std)
        y_std[y_std < 1e-5] = 1e-5

        # Apply recalibration if enabled
        if self.use_recalibration and self.recalibrator is not None:
            recalibrated_res = self.recalibrator(y_mean, y_std, 0.68)
            y_std = np.abs(recalibrated_res.upper - recalibrated_res.lower) / 2

        return y_mean, y_std, success_rate, tot_cost, time_taken

    async def _get_recalibrator(
        self, observed_configs: Any, observed_fvals: Any
    ) -> Tuple[Any, float, float]:
        """
        Get a recalibrator for uncertainty estimates.

        Args:
            observed_configs: Previously observed configurations.
            observed_fvals: Previously observed function values.

        Returns:
            Tuple containing:
                - Recalibrator object
                - Total cost
                - Time taken
        """
        import time

        import numpy as np
        from scipy.stats import linregress
        from scipy.stats import norm

        start_time = time.time()
        tot_cost = 0.0  # Cost for building the recalibrator if any LLM call is made

        # Create a recalibrator class
        class SimpleRecalibrator:
            def __init__(self, slope: float = 1.0, intercept: float = 0.0):
                """
                Initialize the Simple Recalibrator with optional slope and intercept.
                This allows for a simple linear transformation of std to match actual error.
                """
                self.slope = slope
                self.intercept = intercept

            def __call__(self, mean: np.ndarray, std: np.ndarray, confidence_level: float) -> Any:
                """
                Adjust the uncertainty estimation based on the calibration slope and intercept.
                Args:
                    mean: Array of predicted means.
                    std: Array of predicted standard deviations.
                    confidence_level: Desired confidence interval (e.g., 0.68 for 1 sigma).
                Returns:
                    An object with `.lower` and `.upper` attributes.
                """

                class RecalibratorResult:
                    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
                        self.lower = lower
                        self.upper = upper

                # Adjust std using the learned slope and intercept
                adjusted_std = self.slope * std + self.intercept
                z_score = norm.ppf((1 + confidence_level) / 2)

                lower = mean - z_score * adjusted_std
                upper = mean + z_score * adjusted_std

                return RecalibratorResult(lower=lower, upper=upper)

        # Estimate slope & intercept via regression if there are enough points
        if len(observed_configs) > 10:  # Arbitrary threshold for having enough data
            predicted_means = np.array(observed_fvals["mean"])
            predicted_stds = np.array(observed_fvals["std"])

            # True observed values (ground-truth evaluations)
            true_values = np.array(observed_fvals["true"])

            # Calculate absolute errors between predictions and true values
            errors = np.abs(predicted_means - true_values)

            # Avoid divide-by-zero by capping small stds
            capped_stds = np.maximum(predicted_stds, 1e-8)

            # Regress the observed errors against the predicted stds
            slope, intercept, _, _, _ = linregress(capped_stds, errors)

            # Create a recalibrator with the learned parameters
            recalibrator = SimpleRecalibrator(slope=slope, intercept=intercept)
            print(
                f"[Recalibration] Using Linear Regression Recalibrator with slope={slope:.4f}, intercept={intercept:.4f}"
            )

        else:
            # Not enough data for regression, fall back to default scaling (no calibration)
            recalibrator = SimpleRecalibrator()
            print("[Recalibration] Not enough data for regression. Using default recalibration.")

        # Measure time taken
        time_taken = time.time() - start_time

        return recalibrator, tot_cost, time_taken

    async def _evaluate_candidate_points(
        self,
        observed_configs: Any,
        observed_fvals: Any,
        candidate_configs: Any,
        use_feature_semantics: bool = True,
        return_ei: bool = False,
    ) -> Tuple[Any, ...]:
        """
        Evaluate candidate points using the LLM model with support for bootstrapping and recalibration.

        Args:
            observed_configs: Previously observed configurations.
            observed_fvals: Previously observed function values.
            candidate_configs: Candidate configurations to evaluate.
            use_feature_semantics: Whether to use feature semantics.
            return_ei: Whether to return expected improvement.

        Returns:
            Tuple containing evaluation results, which may include:
                - Expected improvement (if return_ei is True)
                - Mean predictions
                - Standard deviations
                - Total cost
                - Time taken

        Example:
            >>> # In an async function
            >>> configs = pd.DataFrame({"x": [1, 2, 3]})
            >>> fvals = pd.Series([0.1, 0.2, 0.3])
            >>> candidates = pd.DataFrame({"x": [4, 5]})
            >>> results = await model._evaluate_candidate_points(configs, fvals, candidates)
        """
        all_run_cost: float = 0.0
        all_run_time: float = 0.0

        if self.use_recalibration and self.recalibrator is None:
            recalibrator, tot_cost, time_taken = await self._get_recalibrator(
                observed_configs, observed_fvals
            )
            self.recalibrator = recalibrator
            print("[Recalibration] COMPLETED")
            all_run_cost += tot_cost
            all_run_time += time_taken

        all_prompt_templates, query_examples = gen_prompt_templates(
            self.task_context,
            observed_configs,
            observed_fvals,
            candidate_configs,
            n_prompts=self.num_prompt_variants,
            bootstrapping=self.bootstrapping,
            use_feature_semantics=use_feature_semantics,
            shuffle_features=self.shuffle_features,
            apply_warping=self.apply_warping,
        )

        print("*" * 100)
        print(f"Number of all_prompt_templates: {len(all_prompt_templates)}")
        print(f"Number of query_examples: {len(query_examples)}")
        print(all_prompt_templates[0].format(Q=query_examples[0]["Q"]))

        y_mean_list, y_std_list = [], []
        total_cost, total_time = 0.0, 0.0

        # Bootstrapping process
        for _ in range(self.num_prompt_variants if self.bootstrapping else 1):
            y_mean, y_std, success_rate, tot_cost, time_taken = await self._predict(
                all_prompt_templates, query_examples
            )
            y_mean_list.append(y_mean)
            y_std_list.append(y_std)
            total_cost += tot_cost
            total_time += time_taken

        if self.bootstrapping:
            y_mean = np.mean(y_mean_list, axis=0)
            y_std = np.sqrt(np.mean(np.square(y_std_list), axis=0))
        else:
            y_mean, y_std = y_mean_list[0], y_std_list[0]

        if self.recalibrator is not None:
            recalibrated_res = self.recalibrator(y_mean, y_std, 0.68)
            y_std = np.abs(recalibrated_res.upper - recalibrated_res.lower) / 2

        all_run_cost += total_cost
        all_run_time += total_time

        if not return_ei:
            return y_mean, y_std, all_run_cost, all_run_time

        best_fval = (
            np.min(observed_fvals.to_numpy())
            if self.lower_is_better
            else np.max(observed_fvals.to_numpy())
        )
        delta = -1 * (y_mean - best_fval) if self.lower_is_better else y_mean - best_fval

        with np.errstate(divide="ignore"):
            Z = delta / y_std
        ei = np.where(y_std > 0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)

        return ei, y_mean, y_std, all_run_cost, all_run_time

    def select_query_point(
        self, observed_configs: Any, observed_fvals: Any, candidate_configs: Any
    ) -> Tuple[Any, float, float]:
        """
        Select the next query point using expected improvement.

        Args:
            observed_configs: Previously observed configurations.
            observed_fvals: Previously observed function values.
            candidate_configs: Candidate configurations to evaluate.

        Returns:
            Tuple containing:
                - Best configuration point
                - Total cost
                - Time taken

        Example:
            >>> configs = pd.DataFrame({"x": [1, 2, 3]})
            >>> fvals = pd.Series([0.1, 0.2, 0.3])
            >>> candidates = pd.DataFrame({"x": [4, 5]})
            >>> best_point, cost, time = model.select_query_point(
            ...     configs, fvals, candidates
            ... )
        """
        if self.warping_transformer is not None:
            observed_configs = self.warping_transformer.warp(observed_configs)
            candidate_configs = self.warping_transformer.warp(candidate_configs)

        y_mean, y_std, cost, time_taken = asyncio.run(
            self._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs)
        )

        best_fval = (
            np.min(observed_fvals.to_numpy())
            if self.lower_is_better
            else np.max(observed_fvals.to_numpy())
        )
        delta = -1 * (y_mean - best_fval) if self.lower_is_better else y_mean - best_fval

        with np.errstate(divide="ignore"):
            Z = delta / y_std

        ei = np.where(y_std > 0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)
        best_point_index = np.argmax(ei)

        if self.warping_transformer is not None:
            candidate_configs = self.warping_transformer.unwarp(candidate_configs)

        best_point = candidate_configs.iloc[[best_point_index], :]

        return best_point, cost, time_taken
