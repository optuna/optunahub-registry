from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np
from scipy.stats import norm

from .discriminative_sm_utils import gen_prompt_templates
from .llm.inquiry import OpenAI_interface
from .rate_limiter import apply_rate_limit


T = TypeVar("T")


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
        n_templates (int): Number of prompt templates to use.
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
        model: str = "gpt-4-mini",
        max_requests_per_minute: int = 100,
    ) -> None:
        """
        Initialize the LLM discriminative surrogate model.

        Args:
            task_context: Context information for the task.
            n_gens: Number of generations per query.
            lower_is_better: Whether lower values are better.
            bootstrapping: Whether to use bootstrapping.
            num_prompt_variants: Number of prompt templates to use.
            use_recalibration: Whether to use recalibration.
            warping_transformer: Transformer for feature warping.
            verbose: Whether to print verbose output.
            prompt_setting: Settings for prompt generation.
            shuffle_features: Whether to shuffle features.
            key: API key for OpenAI.
            model: Model identifier string.
            max_requests_per_minute: Maximum number of requests per minute

        Raises:
            AssertionError: If both bootstrapping and recalibration are enabled.
        """
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.bootstrapping = bootstrapping
        self.num_prompt_variants = num_prompt_variants

        assert not (
            bootstrapping and use_recalibration
        ), "Cannot do recalibration and bootstrapping at the same time"

        self.use_recalibration = use_recalibration
        self.warping_transformer = warping_transformer
        self.apply_warping = warping_transformer is not None
        self.recalibrator = None
        self.verbose = verbose
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features
        self.OpenAI_instance = OpenAI_interface(key, model=model, debug=False)
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
                - Standard deviations
                - Success rate
                - Total cost
                - Time taken

        Example:
            >>> # In an async function
            >>> templates = ["Template {Q}"]
            >>> queries = [{"Q": "example query"}]
            >>> mean, std, rate, cost, time = model._predict(templates, queries)
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

        all_preds = np.array(all_preds).astype(float)
        y_mean = np.nanmean(all_preds, axis=1)
        y_std = np.nanstd(all_preds, axis=1)

        y_mean[np.isnan(y_mean)] = np.nanmean(y_mean)
        y_std[np.isnan(y_std)] = np.nanmean(y_std)
        y_std[y_std < 1e-5] = 1e-5

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
        # Placeholder implementation - replace with actual recalibrator logic
        time_taken = 0.0
        tot_cost = 0.0

        # Create a simple pass-through recalibrator
        class SimpleRecalibrator:
            def __call__(self, mean: np.ndarray, std: np.ndarray, confidence_level: float) -> Any:
                class RecalibratorResult:
                    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
                        self.lower = lower
                        self.upper = upper

                z_score = norm.ppf((1 + confidence_level) / 2)
                lower = mean - z_score * std
                upper = mean + z_score * std
                return RecalibratorResult(lower=lower, upper=upper)

        return SimpleRecalibrator(), tot_cost, time_taken

    async def _evaluate_candidate_points(
        self,
        observed_configs: Any,
        observed_fvals: Any,
        candidate_configs: Any,
        use_feature_semantics: bool = True,
        return_ei: bool = False,
    ) -> Tuple[Any, ...]:
        """
        Evaluate candidate points using the LLM model.

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
            >>> results = model._evaluate_candidate_points(configs, fvals, candidates)
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

        y_mean, y_std, success_rate, tot_cost, time_taken = await self._predict(
            all_prompt_templates, query_examples
        )

        if self.recalibrator is not None:
            recalibrated_res = self.recalibrator(y_mean, y_std, 0.68)
            y_std = np.abs(recalibrated_res.upper - recalibrated_res.lower) / 2

        all_run_cost += float(tot_cost)
        all_run_time += float(time_taken)

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
