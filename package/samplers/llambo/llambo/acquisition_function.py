from __future__ import annotations

import asyncio
import math
import time
from typing import Any
from typing import Optional

from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
from llambo.rate_limiter import RateLimiter
from LLM_utils.inquiry import OpenAI_interface
import numpy as np
import pandas as pd


class LLM_ACQ:
    """A class to implement the acquisition function for Bayesian Optimization using LLMs.

    This class generates candidate hyperparameter configurations for optimization tasks by
    leveraging Large Language Models (LLMs).

    Args:
        task_context (dict[str, Any]): Contextual information about the optimization task.
        n_candidates (int): Number of candidate configurations to generate.
        n_templates (int): Number of prompt templates to use for generating candidates.
        lower_is_better (bool): Whether lower values of the objective function are better.
        jitter (bool, optional): Whether to add jitter to the desired performance target.
            Defaults to False.
        rate_limiter (Optional[RateLimiter], optional): Manages API rate limits for LLM requests.
            Defaults to None.
        warping_transformer (Optional[Any], optional): Applies transformations (e.g., log scaling)
            to hyperparameters. Defaults to None.
        prompt_setting (Optional[str], optional): Controls the level of context provided in the prompts.
            Defaults to None.
        shuffle_features (bool, optional): Whether to shuffle the order of hyperparameters in the prompts.
            Defaults to False.
        key (str, optional): API key for LLM service. Defaults to "".
        model (str, optional): Name of the LLM model to use. Defaults to "gpt-4o-mini".

    Attributes:
        task_context (dict[str, Any]): Contextual information about the optimization task.
        n_candidates (int): Number of candidate configurations to generate.
        n_templates (int): Number of prompt templates to use.
        n_gens (int): Number of generations per template.
        lower_is_better (bool): Whether lower values are better.
        apply_jitter (bool): Whether jitter is applied.
        OpenAI_instance (OpenAI_interface): Interface for OpenAI API calls.
        rate_limiter (RateLimiter): Rate limiter for API calls.
        warping_transformer (Optional[Any]): Transformer for hyperparameter scaling.
        apply_warping (bool): Whether warping is applied.
        prompt_setting (Optional[str]): Prompt context setting.
        shuffle_features (bool): Whether features are shuffled.

    Example:
        >>> task_context = {
        ...     "hyperparameter_constraints": {
        ...         "learning_rate": ["float", "log", [1e-4, 1e-1]],
        ...         "batch_size": ["int", None, [16, 128]]
        ...     }
        ... }
        >>> acq = LLM_ACQ(
        ...     task_context=task_context,
        ...     n_candidates=10,
        ...     n_templates=2,
        ...     lower_is_better=True
        ... )
    """

    def __init__(
        self,
        task_context: dict[str, Any],
        n_candidates: int,
        n_templates: int,
        lower_is_better: bool,
        jitter: bool = False,
        rate_limiter: Optional[RateLimiter] = None,
        warping_transformer: Optional[Any] = None,
        prompt_setting: Optional[str] = None,
        shuffle_features: bool = False,
        key: str = "",
        model: str = "gpt-4o-mini",
    ) -> None:
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = int(n_candidates / n_templates)
        self.lower_is_better = lower_is_better
        self.apply_jitter = jitter
        self.OpenAI_instance = OpenAI_interface(api_key=key, model=model, debug=True)

        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=40000, time_frame=60)
        else:
            self.rate_limiter = rate_limiter

        if warping_transformer is None:
            self.warping_transformer = None
            self.apply_warping = False
        else:
            self.warping_transformer = warping_transformer
            self.apply_warping = True

        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features

        assert isinstance(self.shuffle_features, bool), "shuffle_features must be a boolean"

    def _jitter(self, desired_fval: float) -> float:
        """Add jitter to observed fvals to prevent duplicates.

        Args:
            desired_fval (float): The desired performance target.

        Returns:
            float: The jittered performance target.

        Example:
            >>> acq = LLM_ACQ(task_context={}, n_candidates=10, n_templates=2, lower_is_better=True)
            >>> acq.observed_best = 0.8
            >>> acq.observed_worst = 0.9
            >>> acq.alpha = 0.1
            >>> acq.apply_jitter = True
            >>> jittered_val = acq._jitter(0.85)
            >>> 0.8 <= jittered_val <= 0.85
            True
        """
        if not self.apply_jitter:
            return desired_fval

        assert hasattr(self, "observed_best"), "observed_best must be set before calling _jitter"
        assert hasattr(self, "observed_worst"), "observed_worst must be set before calling _jitter"
        assert hasattr(self, "alpha"), "alpha must be set before calling _jitter"

        jittered = np.random.uniform(
            low=min(desired_fval, self.observed_best),
            high=max(desired_fval, self.observed_best),
            size=1,
        ).item()

        return jittered

    def _count_decimal_places(self, n: float) -> int:
        """Count the number of decimal places in a number.

        Args:
            n (float): The number to count decimal places for.

        Returns:
            int: The number of decimal places.

        Example:
            >>> acq = LLM_ACQ(task_context={}, n_candidates=10, n_templates=2, lower_is_better=True)
            >>> acq._count_decimal_places(123.456)
            3
            >>> acq._count_decimal_places(123.0)
            0
        """
        s = format(n, ".10f")
        if "." not in s:
            return 0
        n_dp = len(s.split(".")[1].rstrip("0"))
        return n_dp

    def _prepare_configurations_acquisition(
        self,
        observed_configs: Optional[pd.DataFrame] = None,
        observed_fvals: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
        use_feature_semantics: bool = True,
        shuffle_features: bool = False,
    ) -> list[dict[str, str]]:
        """Prepare and (possibly shuffle) few-shot examples for prompt templates.

        Args:
            observed_configs (Optional[pd.DataFrame], optional): Observed hyperparameter configurations.
                Defaults to None.
            observed_fvals (Optional[pd.DataFrame], optional): Observed performance values.
                Defaults to None.
            seed (Optional[int], optional): Random seed for shuffling. Defaults to None.
            use_feature_semantics (bool, optional): Whether to use feature names in prompts.
                Defaults to True.
            shuffle_features (bool, optional): Whether to shuffle hyperparameter order.
                Defaults to False.

        Returns:
            list[dict[str, str]]: A list of few-shot examples for the prompts.

        Example:
            >>> configs = pd.DataFrame({'x1': [1.0, 2.0], 'x2': [0.1, 0.2]})
            >>> fvals = pd.DataFrame({'value': [0.5, 0.6]})
            >>> acq = LLM_ACQ(
            ...     task_context={"hyperparameter_constraints": {
            ...         "x1": ["float", None, [0, 10]],
            ...         "x2": ["float", None, [0, 1]]
            ...     }},
            ...     n_candidates=10,
            ...     n_templates=2,
            ...     lower_is_better=True
            ... )
            >>> examples = acq._prepare_configurations_acquisition(configs, fvals)
            >>> len(examples) == len(configs)
            True
        """
        examples = []

        if seed is not None:
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(observed_configs.index)
            observed_configs = observed_configs.loc[shuffled_indices]
            if observed_fvals is not None:
                observed_fvals = observed_fvals.loc[shuffled_indices]
        else:
            if isinstance(observed_fvals, pd.DataFrame):
                if self.lower_is_better:
                    observed_fvals = observed_fvals.sort_values(
                        by=observed_fvals.columns[0], ascending=False
                    )
                else:
                    observed_fvals = observed_fvals.sort_values(
                        by=observed_fvals.columns[0], ascending=True
                    )
                observed_configs = observed_configs.loc[observed_fvals.index]

        if shuffle_features:
            np.random.seed(0)
            shuffled_columns = np.random.permutation(observed_configs.columns)
            observed_configs = observed_configs[shuffled_columns]

        if observed_configs is not None:
            hyperparameter_names = observed_configs.columns
            for index, row in observed_configs.iterrows():
                row_string = "## "
                for i in range(len(row)):
                    hyp_type = self.task_context["hyperparameter_constraints"][
                        hyperparameter_names[i]
                    ][0]
                    hyp_transform = self.task_context["hyperparameter_constraints"][
                        hyperparameter_names[i]
                    ][1]

                    if use_feature_semantics:
                        row_string += f"{hyperparameter_names[i]}: "
                    else:
                        row_string += f"X{i+1}: "

                    if hyp_type in ["int", "float"]:
                        lower_bound = self.task_context["hyperparameter_constraints"][
                            hyperparameter_names[i]
                        ][2][0]
                    else:
                        lower_bound = self.task_context["hyperparameter_constraints"][
                            hyperparameter_names[i]
                        ][2][1]
                    n_dp = self._count_decimal_places(lower_bound)
                    value = row[i]

                    if self.apply_warping:
                        if hyp_type == "int" and hyp_transform != "log":
                            row_string += str(int(value))
                        elif hyp_type == "float" or hyp_transform == "log":
                            row_string += f"{value:.{n_dp}f}"
                        elif hyp_type == "ordinal":
                            row_string += f"{value:.{n_dp}f}"
                        else:
                            row_string += value
                    else:
                        if hyp_type == "int":
                            row_string += str(int(value))
                        elif hyp_type in ["float", "ordinal"]:
                            row_string += f"{value:.{n_dp}f}"
                        else:
                            row_string += value

                    if i != len(row) - 1:
                        row_string += ", "
                row_string += " ##"
                example = {"Q": row_string}
                if observed_fvals is not None:
                    row_index = observed_fvals.index.get_loc(index)
                    perf = f"{observed_fvals.values[row_index][0]:.6f}"
                    example["A"] = perf
                examples.append(example)
        elif observed_fvals is not None:
            examples = [{"A": f"{observed_fvals:.6f}"}]
        else:
            raise ValueError("No observed configurations or performance values provided.")

        return examples

    def _gen_prompt_templates_acquisitions(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        desired_fval: float,
        n_prompts: int = 1,
        use_context: str = "full_context",
        use_feature_semantics: bool = True,
        shuffle_features: bool = False,
    ) -> tuple[list[FewShotPromptTemplate], list[list[dict[str, str]]]]:
        """Generate prompt templates for the acquisition function.

        Args:
            observed_configs (pd.DataFrame): Observed hyperparameter configurations.
            observed_fvals (pd.DataFrame): Observed performance values.
            desired_fval (float): The target performance value to achieve.
            n_prompts (int, optional): Number of prompt templates to generate. Defaults to 1.
            use_context (str, optional): Level of context to include in prompts.
                Defaults to "full_context".
            use_feature_semantics (bool, optional): Whether to use feature names in prompts.
                Defaults to True.
            shuffle_features (bool, optional): Whether to shuffle the order of hyperparameters.
                Defaults to False.

        Returns:
            tuple[list[FewShotPromptTemplate], list[list[dict[str, str]]]]: A tuple containing
                the generated prompt templates and query templates.

        Example:
            >>> configs = pd.DataFrame({'x1': [1.0, 2.0], 'x2': [0.1, 0.2]})
            >>> fvals = pd.DataFrame({'value': [0.5, 0.6]})
            >>> acq = LLM_ACQ(
            ...     task_context={"hyperparameter_constraints": {
            ...         "x1": ["float", None, [0, 10]],
            ...         "x2": ["float", None, [0, 1]]
            ...     }},
            ...     n_candidates=10,
            ...     n_templates=2,
            ...     lower_is_better=True
            ... )
            >>> templates, queries = acq._gen_prompt_templates_acquisitions(
            ...     configs, fvals, 0.4
            ... )
            >>> len(templates) == 1 and len(queries) == 1
            True
        """
        all_prompt_templates = []
        all_query_templates = []

        for i in range(n_prompts):
            few_shot_examples = self._prepare_configurations_acquisition(
                observed_configs,
                observed_fvals,
                seed=i,
                use_feature_semantics=use_feature_semantics,
            )
            jittered_desired_fval = self._jitter(desired_fval)

            task_context = self.task_context
            hyperparameter_constraints = task_context.get("hyperparameter_constraints", {})
            custom_task_description = task_context.get("custom_task_description", None)

            example_template = """
    Performance: {A}
    Hyperparameter configuration: {Q}"""

            example_prompt = PromptTemplate(input_variables=["Q", "A"], template=example_template)

            prefix = "There is a black-box optimization task. "
            if custom_task_description is not None:
                prefix += "Below is a description of the task:\n"
                prefix += custom_task_description
                prefix += "\n"
            prefix += "The allowable ranges for the hyperparameters are:\n"

            for i, (hyperparameter, constraint) in enumerate(hyperparameter_constraints.items()):
                # Safely handle potentially None constraints
                if constraint is None:
                    print(f"Warning: Constraint for {hyperparameter} is None, skipping")
                    continue

                if len(constraint) < 3:
                    print(f"Warning: Constraint for {hyperparameter} is incomplete, skipping")
                    continue

                try:
                    if constraint[0] == "float":
                        n_dp = self._count_decimal_places(constraint[2][0])
                        if constraint[1] == "log" and self.apply_warping:
                            lower_bound = np.log10(constraint[2][0])
                            upper_bound = np.log10(constraint[2][1])
                        else:
                            lower_bound = constraint[2][0]
                            upper_bound = constraint[2][1]

                        if use_feature_semantics:
                            prefix += (
                                f"- {hyperparameter}: [{lower_bound:.{n_dp}f}, "
                                f"{upper_bound:.{n_dp}f}]"
                            )
                        else:
                            prefix += (
                                f"- X{i + 1}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                            )

                        if constraint[1] == "log" and self.apply_warping:
                            prefix += f" (log scale, precise to {n_dp} decimals)"
                        else:
                            prefix += f" (float, precise to {n_dp} decimals)"
                    elif constraint[0] == "int":
                        if constraint[1] == "log" and self.apply_warping:
                            lower_bound = np.log10(constraint[2][0])
                            upper_bound = np.log10(constraint[2][1])
                            n_dp = self._count_decimal_places(lower_bound)
                        else:
                            lower_bound = constraint[2][0]
                            upper_bound = constraint[2][1]
                            n_dp = 0

                        if use_feature_semantics:
                            prefix += (
                                f"- {hyperparameter}: [{lower_bound:.{n_dp}f}, "
                                f"{upper_bound:.{n_dp}f}]"
                            )
                        else:
                            prefix += (
                                f"- X{i + 1}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                            )

                        if constraint[1] == "log" and self.apply_warping:
                            prefix += f" (log scale, precise to {n_dp} decimals)"
                        else:
                            prefix += " (int)"
                    elif constraint[0] == "ordinal":
                        if use_feature_semantics:
                            prefix += f"- {hyperparameter}: "
                        else:
                            prefix += f"- X{i + 1}: "
                        prefix += f" (ordinal, must take value in {constraint[2]})"
                    elif constraint[0] == "categorical":
                        if use_feature_semantics:
                            prefix += f"- {hyperparameter}: "
                        else:
                            prefix += f"- X{i + 1}: "
                        prefix += f" (categorical, must take value in {constraint[2]})"
                    else:
                        print(f"Warning: Unknown hyperparameter value type: {constraint[0]}")
                except Exception as e:
                    print(f"Error processing constraint for {hyperparameter}: {e}")
                    print(f"Constraint value: {constraint}")
                    continue

                prefix += "\n"

            prefix += "Recommend a configuration that can achieve the target performance of "
            prefix += f"{jittered_desired_fval:.6f}. "
            if use_context in ["partial_context", "full_context"]:
                prefix += (
                    "Do not recommend values at the minimum or maximum of allowable range, "
                    "do not recommend rounded values. Recommend values with highest possible "
                    "precision, as requested by the allowed ranges. "
                )
            prefix += (
                "Your response must only contain the predicted configuration, "
                "in the format ## configuration ##.\n"
            )

            suffix = """
    Performance: {A}
    Hyperparameter configuration:"""

            few_shot_prompt = FewShotPromptTemplate(
                examples=few_shot_examples,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=["A"],
                example_separator="",
            )
            all_prompt_templates.append(few_shot_prompt)

            query_examples = self._prepare_configurations_acquisition(
                observed_fvals=jittered_desired_fval,
                seed=None,
                shuffle_features=shuffle_features,
            )
            all_query_templates.append(query_examples)

        return all_prompt_templates, all_query_templates

    async def _async_generate(self, user_message: str) -> Optional[tuple[Any, float, int]]:
        """Generate a response from the LLM asynchronously.

        Args:
            user_message (str): The prompt message to send to the LLM.

        Returns:
            Optional[tuple[Any, float, int]]: A tuple containing the LLM response,
                total cost, and total tokens used.

        Example:
            >>> # This is an async method that returns Optional[tuple[Any, float, int]]
            >>> acq = LLM_ACQ(task_context={}, n_candidates=10, n_templates=2, lower_is_better=True)
            >>> # response = asyncio.run(acq._async_generate("Test message"))
            >>> # response will be either None or (response_content, cost, tokens)
        """
        print("Sending inquiries to the LLM - acquisition function")

        message = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            },
            {"role": "user", "content": user_message},
        ]

        resp, tot_cost = self.OpenAI_instance.ask(message)

        return resp, tot_cost

    async def _async_generate_concurrently(
        self,
        prompt_templates: list[FewShotPromptTemplate],
        query_templates: list[list[dict[str, str]]],
    ) -> list[Optional[tuple[Any, float, int]]]:
        """Perform concurrent generation of responses from the LLM asynchronously.

        Args:
            prompt_templates (list[FewShotPromptTemplate]): List of prompt templates.
            query_templates (list[list[dict[str, str]]]): List of query templates.

        Returns:
            list[Optional[tuple[Any, float, int]]]: A list of results from the LLM requests.

        Example:
            >>> # This is an async method that returns list[Optional[tuple[Any, float, int]]]
            >>> acq = LLM_ACQ(task_context={}, n_candidates=10, n_templates=2, lower_is_better=True)
            >>> templates = [PromptTemplate(input_variables=["x"], template="test {x}")]
            >>> query_temps = [[{"A": "0.5"}]]
            >>> # responses = asyncio.run(acq._async_generate_concurrently(templates, query_temps))
            >>> # responses will be a list of (response_content, cost, tokens) tuples or None
        """
        coroutines = []
        for prompt_template, query_template in zip(prompt_templates, query_templates):
            coroutines.append(
                self._async_generate(prompt_template.format(A=query_template[0]["A"]))
            )

        tasks = [asyncio.create_task(c) for c in coroutines]

        assert len(tasks) == int(self.n_templates)

        results = [None] * len(coroutines)
        llm_response = await asyncio.gather(*tasks)

        for idx, response in enumerate(llm_response):
            if response is not None:
                resp, tot_cost = response
                results[idx] = (resp, tot_cost)

        return results

    def _convert_to_json(self, response_str: str) -> dict[str, float]:
        """Parse LLM response string into JSON.

        Args:
            response_str (str): The LLM response string.

        Returns:
            dict[str, float]: A dictionary containing the parsed hyperparameter configuration.

        Example:
            >>> acq = LLM_ACQ(task_context={}, n_candidates=10, n_templates=2, lower_is_better=True)
            >>> acq._convert_to_json("x1: 1.0, x2: 2.0")
            {'x1': 1.0, 'x2': 2.0}
        """
        pairs = response_str.split(",")
        response_json = {}
        for pair in pairs:
            key, value = [x.strip() for x in pair.split(":")]
            response_json[key] = float(value)

        return response_json

    def _filter_candidate_points(
        self,
        observed_points: list[dict[str, float]],
        candidate_points: list[dict[str, float]],
        precision: int = 8,
    ) -> pd.DataFrame:
        """Filter candidate points that already exist in observed points.

        Also removes duplicates and validates against hyperparameter constraints.

        Args:
            observed_points (list[dict[str, float]]): List of observed hyperparameter configurations.
            candidate_points (list[dict[str, float]]): List of candidate hyperparameter configurations.
            precision (int, optional): Number of decimal places for comparison. Defaults to 8.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered candidate configurations.

        Example:
            >>> acq = LLM_ACQ(
            ...     task_context={"hyperparameter_constraints": {
            ...         "x1": ["float", None, [0, 10]],
            ...         "x2": ["float", None, [0, 1]]
            ...     }},
            ...     n_candidates=10,
            ...     n_templates=2,
            ...     lower_is_better=True
            ... )
            >>> observed = [{"x1": 1.0, "x2": 0.1}]
            >>> candidates = [{"x1": 1.0, "x2": 0.1}, {"x1": 2.0, "x2": 0.2}]
            >>> filtered = acq._filter_candidate_points(observed, candidates)
            >>> len(filtered) == 1
            True
        """
        rounded_observed = [
            {key: round(value, precision) for key, value in d.items()} for d in observed_points
        ]
        rounded_candidate = [
            {key: round(value, precision) for key, value in d.items()} for d in candidate_points
        ]
        filtered_candidates = [
            x
            for i, x in enumerate(candidate_points)
            if rounded_candidate[i] not in rounded_observed
        ]

        def is_within_range(value: float, allowed_range: tuple[str, str, list[float]]) -> bool:
            """Check if a value is within an allowed range."""
            value_type, transform, search_range = allowed_range
            if value_type == "int":
                min_val, max_val = search_range
                if transform == "log" and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                    return min_val <= value <= max_val
                return min_val <= value <= max_val and int(value) == value
            elif value_type == "float":
                min_val, max_val = search_range
                if transform == "log" and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                return min_val <= value <= max_val
            elif value_type == "ordinal":
                return any(math.isclose(value, x, abs_tol=1e-2) for x in allowed_range[2])
            else:
                raise ValueError("Unknown hyperparameter value type")

        def is_dict_within_ranges(
            d: dict[str, float],
            ranges_dict: dict[str, tuple[str, str, list[float]]],
        ) -> bool:
            """Check if all values in a dictionary are within their respective allowable ranges."""
            return all(
                key in ranges_dict and is_within_range(value, ranges_dict[key])
                for key, value in d.items()
            )

        def filter_dicts_by_ranges(
            dict_list: list[dict[str, float]],
            ranges_dict: dict[str, tuple[str, str, list[float]]],
        ) -> list[dict[str, float]]:
            """Return only those dictionaries where all values are within their respective ranges."""
            return [d for d in dict_list if is_dict_within_ranges(d, ranges_dict)]

        hyperparameter_constraints = self.task_context["hyperparameter_constraints"]
        filtered_candidates = filter_dicts_by_ranges(
            filtered_candidates, hyperparameter_constraints
        )

        filtered_candidates = pd.DataFrame(filtered_candidates)
        filtered_candidates = filtered_candidates.drop_duplicates()
        filtered_candidates = filtered_candidates.reset_index(drop=True)
        return filtered_candidates

    def get_candidate_points(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        use_feature_semantics: bool = True,
        use_context: str = "full_context",
        alpha: float = -0.2,
    ) -> tuple[pd.DataFrame, float, float]:
        """Generate candidate points for the acquisition function.

        Args:
            observed_configs (pd.DataFrame): Observed hyperparameter configurations.
            observed_fvals (pd.DataFrame): Observed performance values.
            use_feature_semantics (bool, optional): Whether to use feature names. Defaults to True.
            use_context (str, optional): Level of context to include. Defaults to "full_context".
            alpha (float, optional): Controls target performance relative to best observed.
                Defaults to -0.2.

        Returns:
            tuple[pd.DataFrame, float, float]: A tuple containing the filtered candidate points,
                total cost, and time taken.

        Example:
            >>> configs = pd.DataFrame({'x1': [1.0, 2.0], 'x2': [0.1, 0.2]})
            >>> fvals = pd.DataFrame({'value': [0.5, 0.6]})
            >>> acq = LLM_ACQ(
            ...     task_context={"hyperparameter_constraints": {
            ...         "x1": ["float", None, [0, 10]],
            ...         "x2": ["float", None, [0, 1]]
            ...     }},
            ...     n_candidates=10,
            ...     n_templates=2,
            ...     lower_is_better=True
            ... )
            >>> candidates, cost, time = acq.get_candidate_points(configs, fvals)
            >>> isinstance(candidates, pd.DataFrame)
            True
        """
        assert -1 <= alpha <= 1, "alpha must be between -1 and 1"
        if alpha == 0:
            alpha = -1e-3
        self.alpha = alpha

        if self.prompt_setting is not None:
            use_context = self.prompt_setting

        start_time = time.time()

        range_val = np.abs(np.max(observed_fvals.values) - np.min(observed_fvals.values))
        if range_val == 0:
            range_val = 0.1 * np.abs(np.max(observed_fvals.values))
        alpha_range = [0.1, 1e-2, 1e-3, -1e-3, -1e-2, -0.1]

        if self.lower_is_better:
            self.observed_best = np.min(observed_fvals.values)
            self.observed_worst = np.max(observed_fvals.values)
            range_val = self.observed_worst - self.observed_best
            desired_fval = self.observed_best - alpha * range_val

            iteration = 0
            max_iterations = 10

            while desired_fval <= 0.00001 and iteration < max_iterations:
                alpha_updated = False

                for alpha_ in alpha_range:
                    if alpha_ < alpha:
                        alpha = alpha_
                        desired_fval = self.observed_best - alpha * range_val
                        alpha_updated = True
                        break

                if not alpha_updated:
                    break

                iteration += 1

            print(
                f"Adjusted alpha: {alpha} | [original alpha: {self.alpha}], "
                f"desired fval: {desired_fval:.6f}"
            )

        else:
            self.observed_best = np.max(observed_fvals.values)
            self.observed_worst = np.min(observed_fvals.values)
            range_val = self.observed_best - self.observed_worst
            desired_fval = self.observed_best + alpha * range_val

            iteration = 0
            max_iterations = 10

            while desired_fval >= 0.9999 and iteration < max_iterations:
                alpha_updated = False

                for alpha_ in alpha_range:
                    if alpha_ < alpha:
                        alpha = alpha_
                        desired_fval = self.observed_best + alpha * range_val
                        alpha_updated = True
                        break

                if not alpha_updated:
                    print("[WARNING] No smaller alpha found in alpha_range! Breaking loop")
                    break

                iteration += 1

            print(
                f"Adjusted alpha: {alpha} | [original alpha: {self.alpha}], "
                f"desired fval: {desired_fval:.6f}"
            )

        self.desired_fval = desired_fval

        if self.warping_transformer is not None:
            observed_configs = self.warping_transformer.warp(observed_configs)

        prompt_templates, query_templates = self._gen_prompt_templates_acquisitions(
            observed_configs,
            observed_fvals,
            desired_fval,
            n_prompts=self.n_templates,
            use_context=use_context,
            use_feature_semantics=use_feature_semantics,
            shuffle_features=self.shuffle_features,
        )

        print("=" * 100)
        print("EXAMPLE ACQUISITION PROMPT")
        print(f"Length of prompt templates: {len(prompt_templates)}")
        print(f"Length of query templates: {len(query_templates)}")
        print(prompt_templates[0].format(A=query_templates[0][0]["A"]))
        print("=" * 100)

        number_candidate_points = 0
        filtered_candidate_points = pd.DataFrame()

        retry = 0
        while number_candidate_points < 5:
            llm_responses = asyncio.run(
                self._async_generate_concurrently(prompt_templates, query_templates)
            )

            candidate_points = []
            tot_cost = 0

            for response in llm_responses:
                if response is None:
                    continue
                for response_content in response:
                    try:
                        response_content = response_content.split("##")[1].strip()
                        candidate_points.append(self._convert_to_json(response_content))
                    except Exception:
                        print(response_content)
                        continue
                tot_cost += response[1]

            proposed_points = self._filter_candidate_points(
                observed_configs.to_dict(orient="records"), candidate_points
            )
            filtered_candidate_points = pd.concat(
                [filtered_candidate_points, proposed_points], ignore_index=True
            )
            number_candidate_points = filtered_candidate_points.shape[0]

            print(
                f"Attempt: {retry}, number of proposed candidate points: {len(candidate_points)}, ",
                f"number of accepted candidate points: {filtered_candidate_points.shape[0]}",
            )

            retry += 1
            if retry > 10:
                print(f"Desired fval: {desired_fval:.6f}")
                print(f"Number of proposed candidate points: {len(candidate_points)}")
                print(f"Number of accepted candidate points: {filtered_candidate_points.shape[0]}")
                if len(candidate_points) > 5:
                    filtered_candidate_points = pd.DataFrame(candidate_points)
                    break
                else:
                    raise RuntimeError("LLM failed to generate candidate points after 10 retries")

        if self.warping_transformer is not None:
            filtered_candidate_points = self.warping_transformer.unwarp(filtered_candidate_points)

        end_time = time.time()
        time_taken = end_time - start_time

        return filtered_candidate_points, tot_cost, time_taken
