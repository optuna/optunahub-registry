from __future__ import annotations

import asyncio
import re
import time
from typing import Any
from typing import Optional
from typing import Sequence

from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
import numpy as np
import pandas as pd

from .llm import OpenAI_interface
from .utils import apply_rate_limit


def _count_decimal_places(n: float) -> int:
    """
    Count the number of decimal places in a number.

    Args:
        n (float): The number to analyze.

    Returns:
        int: The number of decimal places, with a minimum of 2.

    Example:
        >>> _count_decimal_places(3.14159)
        5
        >>> _count_decimal_places(42.0)
        2
        >>> _count_decimal_places(42)
        0
    """
    s = format(n, ".10f")
    if "." not in s:
        return 0
    num_dp = len(s.split(".")[1].rstrip("0"))
    return 2 if num_dp == 0 else num_dp


def prepare_configurations(
    hyperparameter_constraints: dict,
    lower_is_better: bool,
    top_pct: float,
    observed_configs: pd.DataFrame,
    observed_fvals: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
) -> list[dict[str, str]]:
    """
    Prepare and possibly shuffle the configurations for prompt templates.

    Args:
        hyperparameter_constraints (dict): Constraints for each hyperparameter.
        lower_is_better (bool): Whether lower values indicate better performance.
        top_pct (float): Percentage threshold for top performers.
        observed_configs (pd.DataFrame): DataFrame containing observed configurations.
        observed_fvals (Optional[pd.DataFrame]): DataFrame containing observed function values.
        seed (Optional[int]): Random seed for shuffling.

    Returns:
        list[dict[str, str]]: List of examples formatted for prompt templates.
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

    observed_configs = observed_configs.reset_index(drop=True)
    if observed_fvals is not None:
        observed_fvals = observed_fvals.reset_index(drop=True)
        if lower_is_better:
            labels = (observed_fvals < np.percentile(observed_fvals, int(top_pct * 100))).astype(
                int
            )
        else:
            labels = (
                observed_fvals > np.percentile(observed_fvals, int(100 - top_pct * 100))
            ).astype(int)

    for index, row in observed_configs.iterrows():
        row_string = ""
        for i in range(len(row)):
            # Get the hyperparameter constraints
            constraint = hyperparameter_constraints[hyperparameter_names[i]]
            hyp_type = constraint[0]  # Type (int, float, etc.)

            # Get the lower bound of the constraint
            lower_bound = constraint[2][0]

            # Determine base precision from constraint
            n_dp = _count_decimal_places(lower_bound)

            # For float types, ensure we use appropriate precision
            if hyp_type == "float":
                # Get actual precision from the value itself
                actual_dp = _count_decimal_places(row[i])
                # Use at least 1 decimal place for floats, or more if value has more precision
                n_dp = max(1, n_dp, actual_dp)

            # Format the value based on its type
            if hyp_type == "int":
                value = str(int(row[i]))
            elif hyp_type == "float":
                value = f"{row[i]:.{n_dp}f}"
            else:
                value = str(row[i])

            row_string += f"{hyperparameter_names[i]}: {value}"

            if i != len(row) - 1:
                row_string += ", "

        example = {"Q": row_string}
        if observed_fvals is not None:
            row_index = observed_fvals.index.get_loc(index)
            label = f"## {labels.values[row_index][0]} ##"
            example["A"] = label
        examples.append(example)

    return examples


def gen_prompt_tempates(
    task_context: dict,
    observed_configs: pd.DataFrame,
    observed_fvals: pd.DataFrame,
    candidate_configs: pd.DataFrame,
    lower_is_better: bool,
    top_pct: float,
    n_prompts: int = 1,
) -> tuple[list[FewShotPromptTemplate], list[dict[str, str]]]:
    """
    Generate prompt templates for the few-shot learning task for the generative surrogate model.

    Args:
        task_context (dict): Context information about the optimization task, possibly including "n_initial_samples" and "current_trial".
        observed_configs (pd.DataFrame): Previously observed configurations.
        observed_fvals (pd.DataFrame): Observed performance values.
        candidate_configs (pd.DataFrame): Candidate configurations to evaluate.
        lower_is_better (bool): Whether lower values indicate better performance.
        top_pct (float): Percentage threshold for top performers.
        n_prompts (int): Number of prompt templates to generate.

    Returns:
        Tuple of:
          - A list of FewShotPromptTemplate objects.
          - A list of query examples (dicts).
    """
    custom_task_description = task_context.get("custom_task_description")
    all_prompt_templates: list[FewShotPromptTemplate] = []

    for i in range(n_prompts):
        # Prepare few-shot examples using observed configurations and values.
        few_shot_examples = prepare_configurations(
            task_context["hyperparameter_constraints"],
            lower_is_better,
            top_pct,
            observed_configs,
            observed_fvals,
            seed=i,
        )

        example_template = """
Hyperparameter configuration: {Q}
Classification: {A}"""

        example_prompt = PromptTemplate(
            input_variables=["Q", "A"],
            template=example_template,
        )

        prefix = (
            "The following are examples of hyperparameter configurations "
            "for a black-box optimization task. "
        )
        if custom_task_description is not None:
            prefix += "Below is a description of the task:\n" + custom_task_description + "\n"
        prefix += (
            f"The performance classification is 1 if the configuration is in the "
            f"best-performing {top_pct * 100}% of all configurations and 0 otherwise. "
        )
        prefix += (
            "Your response should only contain the predicted performance "
            "classification in the format ## performance classification ##."
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
Classification: """

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
        lower_is_better,
        top_pct,
        candidate_configs,
        seed=None,
    )

    return all_prompt_templates, query_examples


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
        num_prompt_variants (int): Number of distinct prompt variants.
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
        num_prompt_variants: int = 2,
        verbose: bool = False,
        key: str = "",
        model: str = "gpt-4o-mini",
        max_requests_per_minute: int = 100,
        azure: bool = False,
        azure_api_base: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLM generative surrogate model.

        Args:
            task_context: Context information about the optimization task.
            n_gens: Number of generations to perform.
            lower_is_better: Whether lower objective values are better.
            top_pct: Top percentage of configurations to consider.
            num_prompt_variants (int): Number of distinct prompt variants.
            verbose: Whether to print detailed information.
            key: OpenAI API key.
            model: Name of the OpenAI model to use.
            max_requests_per_minute: Maximum number of requests per minute.
            azure: Whether to use Azure API endpoints.
            azure_api_base: Azure API base URL.
            azure_api_version: Azure API version.
            azure_deployment_name: Azure deployment name.
        """
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.top_pct = top_pct
        self.num_prompt_variants = num_prompt_variants
        self.recalibrator = None

        # Azure parameters
        self.azure = azure
        self.azure_api_base = azure_api_base
        self.azure_api_version = azure_api_version
        self.azure_deployment_name = azure_deployment_name

        # Initialize OpenAI interface with Azure support if needed
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

        self.verbose = verbose

    async def _async_generate(
        self,
        few_shot_template: str,
        query_example: dict[str, Any],
        query_idx: int,
    ) -> tuple[int, str | None, float, int]:
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
        if resp is None:
            resp = ""
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
            n_prompts=self.num_prompt_variants,
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

        This method evaluates candidate configurations and selects the best one based on the predicted
        probabilities from the generative surrogate model.
        """
        # Ensure inputs are in DataFrame format
        if not isinstance(observed_configs, pd.DataFrame):
            observed_configs = pd.DataFrame(observed_configs)
        if not isinstance(candidate_configs, pd.DataFrame):
            candidate_configs = pd.DataFrame(candidate_configs)

        # Apply warping transformations if applicable
        observed_configs = self._warp_candidate_points(observed_configs)
        candidate_configs = self._warp_candidate_points(candidate_configs)

        # Evaluate candidate points using the surrogate model
        try:
            pred_probs, cost, time_taken = await self._evaluate_candidate_points(
                observed_configs,
                observed_fvals,
                candidate_configs,
            )
        except Exception as e:
            print(f"Error in _evaluate_candidate_points: {e}")
            pred_probs = np.full(len(candidate_configs), np.nan)  # Return NaNs if failure occurs
            cost = 0.0
            time_taken = 0.0

        if np.isnan(pred_probs).all():
            print("All predictions are NaN. Falling back to random selection.")
            best_point_index = np.random.choice(len(candidate_configs))
        else:
            # A higher probability always indicates better configurations
            # regardless of optimization direction
            best_point_index = np.nanargmax(pred_probs)
            print(
                f"Selected configuration with highest probability ({pred_probs[best_point_index]:.4f}) of being in top-performing set"
            )

        # Unwarp the candidate points before returning
        candidate_configs = self._unwarp_candidate_points(candidate_configs)
        best_point = candidate_configs.iloc[[best_point_index], :]

        if return_raw_preds:
            return best_point, pred_probs, cost, time_taken
        return best_point, cost, time_taken
