"""
LLM-based generative surrogate model for hyperparameter optimization.

This module implements a surrogate model using Large Language Models (LLMs) for
predicting optimal hyperparameter configurations in machine learning tasks.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from typing import Sequence

from llambo.generative_sm_utils import gen_prompt_tempates
from llambo.rate_limiter import apply_rate_limit
from LLM_utils.inquiry import OpenAI_interface
import numpy as np
import pandas as pd


class LLMGenerativeSM:
    """
    LLM-based generative surrogate model for hyperparameter optimization.

    This class implements a forward LLM surrogate model for modeling p(y|x) similar to
    Gaussian Processes or SMAC. It uses OpenAI's API to generate predictions for
    hyperparameter configurations.

    Attributes:
        task_context (dict[str, Any]): Context information about the optimization task.
        n_gens (int): Number of generations to perform.
        lower_is_better (bool): Whether lower objective values are better.
        top_pct (float): Top percentage of configurations to consider.
        n_templates (int): Number of prompt templates to use.
        recalibrator (Optional[Any]): Model recalibration component.
        OpenAI_instance (OpenAI_interface): Interface to OpenAI's API.
        verbose (bool): Whether to print detailed information.

    Example:
        >>> context = {"hyperparameter_constraints": {"learning_rate": [0.001, "log"]}}
        >>> model = LLMGenerativeSM(
        ...     task_context=context,
        ...     n_gens=10,
        ...     lower_is_better=True,
        ...     top_pct=0.1,
        ...     key="your-api-key"
        ... )
    """

    def __init__(
        self,
        task_context: dict[str, Any],
        n_gens: int,
        lower_is_better: bool,
        top_pct: float,
        n_templates: int = 1,
        verbose: bool = False,
        key: str = "",
        model: str = "gpt-4o-mini",
        max_requests_per_minute: int = 100,
    ) -> None:
        """
        Initialize the LLM generative surrogate model.

        Args:
            task_context: Context information about the optimization task.
            n_gens: Number of generations to perform.
            lower_is_better: Whether lower objective values are better.
            top_pct: Top percentage of configurations to consider.
            n_templates: Number of prompt templates to use.
            verbose: Whether to print detailed information.
            key: OpenAI API key.
            model: Name of the OpenAI model to use.
            max_requests_per_minute: Maximum number of requests per minute.
        """
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.top_pct = top_pct
        self.n_templates = n_templates
        self.recalibrator = None
        self.OpenAI_instance = OpenAI_interface(key, model=model, debug=False)
        apply_rate_limit(self.OpenAI_instance, max_requests_per_minute=max_requests_per_minute)

        self.verbose = verbose

    async def _async_generate(
        self,
        few_shot_template: str,
        query_example: dict[str, Any],
        query_idx: int,
    ) -> tuple[int, str, float, int]:
        """
        Generate predictions asynchronously using the LLM.

        Args:
            few_shot_template: Template for few-shot learning.
            query_example: Example to generate prediction for.
            query_idx: Index of the query.

        Returns:
            tuple containing:
                - Query index
                - LLM response
                - Total cost
                - Total tokens used

        Example:
            >>> template = "Given {Q}, predict the performance"
            >>> example = {"Q": "learning_rate=0.01"}
            >>> async def example_usage():
            ...     result = await model._async_generate(template, example, 0)
            ...     return isinstance(result, tuple) and len(result) == 4
            >>> asyncio.run(example_usage())
            True
        """
        print("Sending inquiries to the LLM - generative surrogate model")

        prompt = few_shot_template.format(Q=query_example["Q"])
        message = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            },
            {"role": "user", "content": prompt},
        ]

        resp, tot_cost = self.OpenAI_instance.ask(message)
        # The fourth return value (tokens used) is missing in the implementation
        # Adding a placeholder value of 0 for total tokens
        total_tokens = 0  # This should be updated if token count is available

        return query_idx, resp, tot_cost, total_tokens

    async def _generate_concurrently(
        self,
        few_shot_templates: list[str],
        query_examples: list[dict[str, Any]],
    ) -> list[list[list[Any]]]:
        """
        Perform concurrent generation of responses from the LLM.

        Args:
            few_shot_templates: List of templates for few-shot learning.
            query_examples: List of examples to generate predictions for.

        Returns:
            Nested list of results for each query example.

        Example:
            >>> templates = ["Template {Q}"]
            >>> examples = [{"Q": "config1"}, {"Q": "config2"}]
            >>> async def example_usage():
            ...     results = await model._generate_concurrently(templates, examples)
            ...     return isinstance(results, list) and len(results) == len(examples)
            >>> asyncio.run(example_usage())
            True
        """
        coroutines = []
        for template in few_shot_templates:
            for query_idx, query_example in enumerate(query_examples):
                coroutines.append(self._async_generate(template, query_example, query_idx))

        # Add type annotation for results
        results: list[list[list[Any]]] = [[] for _ in range(len(query_examples))]

        for task in asyncio.as_completed(coroutines):
            response = await task
            if response is not None:
                query_idx, resp, tot_cost, _ = response  # Unpack 4 values, ignoring the last one
                results[query_idx].append([resp, tot_cost])
            else:
                print("None response received")

        return results

    def process_response(self, all_raw_response: Sequence[str]) -> list[float]:
        """
        Process raw responses from the LLM to extract prediction probabilities.

        Args:
            all_raw_response: Sequence of raw response strings from the LLM.

        Returns:
            List of extracted probability values or NaN for invalid responses.

        Example:
            >>> responses = ["## 0.75 ##", "invalid", "## 0.85 ##"]
            >>> probs = model.process_response(responses)
            >>> len(probs) == len(responses)
            True
        """
        all_pred_probs = []
        for raw_response in all_raw_response:
            if isinstance(raw_response, str):
                gen_pred = re.findall(r"## (-?[\d.]+) ##", raw_response)
                if len(gen_pred) == 1:
                    all_pred_probs.append(float(gen_pred[0]))
                else:
                    print("No valid numeric value found in raw_response, appending NaN")
                    all_pred_probs.append(np.nan)
            else:
                print("raw_response is not a string, appending NaN")
                all_pred_probs.append(np.nan)

        return all_pred_probs

    async def _predict(
        self,
        all_prompt_templates: list[str],
        query_examples: list[dict[str, Any]],
    ) -> tuple[np.ndarray, float, float, float]:
        """
        Generate predictions for multiple query examples.

        Args:
            all_prompt_templates: List of prompt templates.
            query_examples: List of query examples.

        Returns:
            tuple containing:
                - Array of mean probabilities
                - Success rate
                - Total cost
                - Time taken

        Example:
            >>> templates = ["Template {Q}"]
            >>> examples = [{"Q": "config1"}, {"Q": "config2"}]
            >>> async def example_usage():
            ...     result = await model._predict(templates, examples)
            ...     return isinstance(result, tuple) and len(result) == 4
            >>> asyncio.run(example_usage())
            True
        """
        start = time.time()
        all_preds = []
        tot_cost = 0
        bool_pred_returned = []

        for i in range(0, len(query_examples), 5):
            query_chunk = query_examples[i : i + 5]
            chunk_results = await self._generate_concurrently(
                all_prompt_templates,
                query_chunk,
            )

            bool_pred_returned.extend([1 if len(x) > 0 else 0 for x in chunk_results])

            for sample_response in chunk_results:
                if not sample_response:
                    sample_preds = [np.nan] * len(all_prompt_templates)
                else:
                    all_raw_response = []
                    for template_response in sample_response:
                        if isinstance(template_response, list) and len(template_response) > 0:
                            llm_response = template_response[0]
                            all_raw_response.append(llm_response)
                        else:
                            print(f"Invalid template_response: {template_response}")
                            all_raw_response.append(np.nan)

                    sample_preds = self.process_response(all_raw_response)
                    tot_cost += sum(
                        x[1] for x in sample_response if isinstance(x, list) and len(x) > 1
                    )
                all_preds.append(sample_preds)

        time_taken = time.time() - start
        success_rate = (
            sum(bool_pred_returned) / len(bool_pred_returned) if bool_pred_returned else 0
        )
        pred_probs = np.array(all_preds).astype(float)
        mean_probs = np.nanmean(pred_probs, axis=1)

        # Fixed return tuple to match the expected 4 values, removing the tokens count
        return mean_probs, success_rate, tot_cost, time_taken

    async def _evaluate_candidate_points(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: np.ndarray,
        candidate_configs: pd.DataFrame,
    ) -> tuple[np.ndarray, float, float]:
        """
        Evaluate candidate points using the LLM model.

        Args:
            observed_configs: DataFrame of observed configurations.
            observed_fvals: Array of observed objective values.
            candidate_configs: DataFrame of candidate configurations.

        Returns:
            tuple containing:
                - Array of predicted probabilities
                - Total cost
                - Total time taken

        Example:
            >>> obs_configs = pd.DataFrame({"param": [0.1, 0.2]})
            >>> obs_fvals = np.array([0.5, 0.6])
            >>> cand_configs = pd.DataFrame({"param": [0.3, 0.4]})
            >>> async def example_usage():
            ...     result = await model._evaluate_candidate_points(
            ...         obs_configs, obs_fvals, cand_configs
            ...     )
            ...     return isinstance(result, tuple) and len(result) == 3
            >>> asyncio.run(example_usage())
            True
        """
        all_run_cost: float = 0.0
        all_run_time: float = 0.0

        if not isinstance(observed_configs, pd.DataFrame):
            observed_configs = pd.DataFrame(observed_configs)
        if not isinstance(candidate_configs, pd.DataFrame):
            candidate_configs = pd.DataFrame(candidate_configs)

        all_prompt_templates, query_examples = gen_prompt_tempates(
            self.task_context,
            observed_configs,
            observed_fvals,
            candidate_configs,
            self.lower_is_better,
            self.top_pct,
            n_prompts=self.n_templates,
        )

        print("*" * 100)
        print(f"Number of all_prompt_templates: {len(all_prompt_templates)}")
        print(f"Number of query_examples: {len(query_examples)}")

        response = await self._predict(all_prompt_templates, query_examples)
        # Unpacking 4 values as returned by the updated _predict method
        pred_probs, success_rate, tot_cost, time_taken = response

        all_run_cost += tot_cost
        all_run_time += time_taken

        return pred_probs, all_run_cost, all_run_time

    def _warp_candidate_points(
        self,
        configurations: pd.DataFrame | dict[str, Any],
    ) -> pd.DataFrame:
        """
        Warp candidate points to log scale if necessary.

        Args:
            configurations: DataFrame or dict of configurations.

        Returns:
            DataFrame of warped configurations.

        Example:
            >>> configs = pd.DataFrame({"param": [0.1, 0.01]})
            >>> model.task_context = {"hyperparameter_constraints":
            ...     {"param": [0.001, "log"]}}
            >>> warped = model._warp_candidate_points(configs)
            >>> isinstance(warped, pd.DataFrame)
            True
        """
        if not isinstance(configurations, pd.DataFrame):
            configurations = pd.DataFrame(configurations)

        warped_configs = configurations.copy().to_dict(orient="records")
        hyperparameter_constraints = self.task_context["hyperparameter_constraints"]

        for config in warped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == "log":
                    config[hyperparameter] = np.log10(config[hyperparameter])

        return pd.DataFrame(warped_configs)

    def _unwarp_candidate_points(
        self,
        configurations: pd.DataFrame | dict[str, Any],
    ) -> pd.DataFrame:
        """
        Unwarp candidate points from log scale if necessary.

        Args:
            configurations: DataFrame or dict of configurations.

        Returns:
            DataFrame of unwarped configurations.

        Example:
            >>> configs = pd.DataFrame({"param": [-1, -2]})  # log10 values
            >>> model.task_context = {"hyperparameter_constraints":
            ...     {"param": [0.001, "log"]}}
            >>> unwarped = model._unwarp_candidate_points(configs)
            >>> isinstance(unwarped, pd.DataFrame)
            True
        """
        if not isinstance(configurations, pd.DataFrame):
            configurations = pd.DataFrame(configurations)

        unwarped_configs = configurations.copy().to_dict(orient="records")
        hyperparameter_constraints = self.task_context["hyperparameter_constraints"]

        for config in unwarped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == "log":
                    config[hyperparameter] = 10 ** config[hyperparameter]

        return pd.DataFrame(unwarped_configs)

    async def select_query_point(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: np.ndarray,
        candidate_configs: pd.DataFrame,
        return_raw_preds: bool = False,
    ) -> tuple[pd.DataFrame, float, float] | tuple[pd.DataFrame, np.ndarray, float, float]:
        """
        Select the next query point using expected improvement.

        This method evaluates candidate configurations and selects the most promising
        point for the next evaluation based on predicted probabilities.

        Args:
            observed_configs: DataFrame of previously observed configurations.
            observed_fvals: Array of observed objective values.
            candidate_configs: DataFrame of candidate configurations to evaluate.
            return_raw_preds: Whether to return raw prediction probabilities.

        Returns:
            If return_raw_preds is False:
                tuple containing:
                    - DataFrame with the selected configuration
                    - Total cost
                    - Time taken
            If return_raw_preds is True:
                tuple containing:
                    - DataFrame with the selected configuration
                    - Array of prediction probabilities
                    - Total cost
                    - Time taken

        Example:
            >>> obs_configs = pd.DataFrame({"param": [0.1, 0.2]})
            >>> obs_fvals = np.array([0.5, 0.6])
            >>> cand_configs = pd.DataFrame({"param": [0.3, 0.4]})
            >>> async def example_usage():
            ...     result = await model.select_query_point(
            ...         obs_configs, obs_fvals, cand_configs
            ...     )
            ...     return isinstance(result[0], pd.DataFrame)
            >>> asyncio.run(example_usage())
            True
        """
        if not isinstance(observed_configs, pd.DataFrame):
            observed_configs = pd.DataFrame(observed_configs)
        if not isinstance(candidate_configs, pd.DataFrame):
            candidate_configs = pd.DataFrame(candidate_configs)

        observed_configs = self._warp_candidate_points(observed_configs)
        candidate_configs = self._warp_candidate_points(candidate_configs)

        pred_probs, cost, time_taken = await self._evaluate_candidate_points(
            observed_configs,
            observed_fvals,
            candidate_configs,
        )

        best_point_index = np.argmax(pred_probs)
        candidate_configs = self._unwarp_candidate_points(candidate_configs)
        best_point = candidate_configs.iloc[[best_point_index], :]

        if return_raw_preds:
            return best_point, pred_probs, cost, time_taken
        return best_point, cost, time_taken
