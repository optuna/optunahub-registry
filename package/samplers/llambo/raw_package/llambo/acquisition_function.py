from __future__ import annotations

import asyncio
import math
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
from llambo.rate_limiter import RateLimiter
from LLM_utils.inquiry import OpenAI_interface
import numpy as np
import pandas as pd


class LLM_ACQ:
    """
    A class to implement the acquisition function for Bayesian Optimization using Large Language Models (LLMs).

    This class generates candidate hyperparameter configurations for optimization tasks by leveraging LLMs.

    Attributes:
        task_context (dict): Contextual information about the optimization task.
        n_candidates (int): Number of candidate configurations to generate.
        n_templates (int): Number of prompt templates_mixed to use for generating candidates.
        lower_is_better (bool): Whether lower values of the objective function are better.
        apply_jitter (bool): Whether to add jitter to the desired performance target.
        rate_limiter (RateLimiter): Manages API rate limits for LLM requests.
        warping_transformer: Applies transformations (e.g., log scaling) to hyperparameters.
        prompt_setting (str): Controls the level of context provided in the prompts.
        shuffle_features (bool): Whether to shuffle the order of hyperparameters in the prompts.
    """

    def __init__(
        self,
        task_context: Dict[str, Any],
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
        """
        Initialize the LLM Acquisition function.

        Args:
            task_context (dict): Contextual information about the optimization task.
            n_candidates (int): Number of candidate configurations to generate.
            n_templates (int): Number of prompt templates_mixed to use for generating candidates.
            lower_is_better (bool): Whether lower values of the objective function are better.
            jitter (bool): Whether to add jitter to the desired performance target.
            rate_limiter (RateLimiter): Manages API rate limits for LLM requests.
            warping_transformer: Applies transformations (e.g., log scaling) to hyperparameters.
            prompt_setting (str): Controls the level of context provided in the prompts.
            shuffle_features (bool): Whether to shuffle the order of hyperparameters in the prompts.
        """
        print("DEBUG, init acquisition function")
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = int(n_candidates / n_templates)
        self.lower_is_better = lower_is_better
        self.apply_jitter = jitter
        self.OpenAI_instance = OpenAI_interface(api_key=key, model=model, debug=True)

        # Initialize rate limiter if not provided
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=40000, time_frame=60)
        else:
            self.rate_limiter = rate_limiter

        # Initialize warping transformer if provided
        if warping_transformer is None:
            self.warping_transformer = None
            self.apply_warping = False
        else:
            self.warping_transformer = warping_transformer
            self.apply_warping = True

        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features

        # Validate shuffle_features input
        assert isinstance(self.shuffle_features, bool), "shuffle_features must be a boolean"

    def _jitter(self, desired_fval: float) -> float:
        """
        Add jitter to observed fvals to prevent duplicates.

        Args:
            desired_fval (float): The desired performance target.

        Returns:
            float: The jittered performance target.
        """
        if not self.apply_jitter:
            return desired_fval

        # Validate required attributes
        assert hasattr(self, "observed_best"), "observed_best must be set before calling _jitter"
        assert hasattr(self, "observed_worst"), "observed_worst must be set before calling _jitter"
        assert hasattr(self, "alpha"), "alpha must be set before calling _jitter"

        # Add uniform random noise to the desired performance target
        jittered = np.random.uniform(
            low=min(desired_fval, self.observed_best),
            high=max(desired_fval, self.observed_best),
            size=1,
        ).item()

        return jittered

    def _count_decimal_places(self, n: float) -> int:
        """
        Count the number of decimal places in a number.

        Args:
            n (float): The number to count decimal places for.

        Returns:
            int: The number of decimal places.
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
    ) -> List[Dict[str, str]]:
        """
        Prepare and (possibly shuffle) few-shot examples for prompt templates_mixed.

        Args:
            observed_configs (pd.DataFrame): Observed hyperparameter configurations.
            observed_fvals (pd.DataFrame): Observed performance values.
            seed (int): Random seed for shuffling.
            use_feature_semantics (bool): Whether to use feature names in the prompts.
            shuffle_features (bool): Whether to shuffle the order of hyperparameters.

        Returns:
            list[dict[str, str]]: A list of few-shot examples for the prompts.
        """
        examples = []

        if seed is not None:
            # Shuffle observed configurations if a seed is provided
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(observed_configs.index)
            observed_configs = observed_configs.loc[shuffled_indices]
            if observed_fvals is not None:
                observed_fvals = observed_fvals.loc[shuffled_indices]
        else:
            # Sort observed configurations by performance values if no seed is provided
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
            # Shuffle the order of hyperparameters
            np.random.seed(0)
            shuffled_columns = np.random.permutation(observed_configs.columns)
            observed_configs = observed_configs[shuffled_columns]

        print("DEBUG-task_context", self.task_context)
        print("DEBUG-task_context type", type(self.task_context))

        # Serialize the observed configurations into few-shot examples
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
            raise Exception("No observed configurations or performance values provided.")

        return examples

    def _gen_prompt_tempates_acquisitions(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        desired_fval: float,
        n_prompts: int = 1,
        use_context: str = "full_context",
        use_feature_semantics: bool = True,
        shuffle_features: bool = False,
    ) -> Tuple[List[FewShotPromptTemplate], List[List[Dict[str, str]]]]:
        """
        Generate prompt templates_mixed for the acquisition function.

        Args:
            observed_configs (pd.DataFrame): Observed hyperparameter configurations.
            observed_fvals (pd.DataFrame): Observed performance values.
            desired_fval (float): The desired performance target.
            n_prompts (int): Number of prompt templates_mixed to generate.
            use_context (str): Level of context to include in the prompts.
            use_feature_semantics (bool): Whether to use feature names in the prompts.
            shuffle_features (bool): Whether to shuffle the order of hyperparameters.

        Returns:
            tuple[list[FewShotPromptTemplate], list[list[dict[str, str]]]]: A tuple containing the prompt templates_mixed and query templates_mixed.
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

            # Extract task context information
            task_context = self.task_context
            model = task_context["model"]
            task = task_context["task"]
            tot_feats = task_context["tot_feats"]
            cat_feats = task_context["cat_feats"]
            num_feats = task_context["num_feats"]
            n_classes = task_context["n_classes"]
            metric = (
                "mean squared error"
                if task_context["metric"] == "neg_mean_squared_error"
                else task_context["metric"]
            )
            num_samples = task_context["num_samples"]
            hyperparameter_constraints = task_context["hyperparameter_constraints"]

            # Define the example template for the prompt
            example_template = """
Performance: {A}
Hyperparameter configuration: {Q}"""

            example_prompt = PromptTemplate(input_variables=["Q", "A"], template=example_template)

            # Build the prefix for the prompt
            prefix = f"The following are examples of performance of a {model} measured in {metric} and the corresponding model hyperparameter configurations."
            if use_context == "full_context":
                if task == "classification":
                    prefix += f" The model is evaluated on a tabular {task} task containing {n_classes} classes."
                elif task == "regression":
                    prefix += f" The model is evaluated on a tabular {task} task."
                else:
                    raise Exception("Unknown task type.")
                prefix += f" The tabular dataset contains {num_samples} samples and {tot_feats} features ({cat_feats} categorical, {num_feats} numerical)."
            prefix += " The allowable ranges for the hyperparameters are:\n"

            # Add hyperparameter constraints to the prefix
            for i, (hyperparameter, constraint) in enumerate(hyperparameter_constraints.items()):
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
                            f"- {hyperparameter}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                        )
                    else:
                        prefix += f"- X{i+1}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"

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
                            f"- {hyperparameter}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"
                        )
                    else:
                        prefix += f"- X{i+1}: [{lower_bound:.{n_dp}f}, {upper_bound:.{n_dp}f}]"

                    if constraint[1] == "log" and self.apply_warping:
                        prefix += f" (log scale, precise to {n_dp} decimals)"
                    else:
                        prefix += " (int)"
                elif constraint[0] == "ordinal":
                    if use_feature_semantics:
                        prefix += f"- {hyperparameter}: "
                    else:
                        prefix += f"- X{i+1}: "
                    prefix += f" (ordinal, must take value in {constraint[2]})"
                else:
                    raise Exception("Unknown hyperparameter value type.")

                prefix += "\n"
            prefix += f"Recommend a configuration that can achieve the target performance of {jittered_desired_fval:.6f}. "
            if use_context in ["partial_context", "full_context"]:
                prefix += "Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. Recommend values with highest possible precision, as requested by the allowed ranges. "
            prefix += "Your response must only contain the predicted configuration, in the format ## configuration ##.\n"

            # Define the suffix for the prompt
            suffix = """
Performance: {A}
Hyperparameter configuration:"""

            # Create the few-shot prompt template
            few_shot_prompt = FewShotPromptTemplate(
                examples=few_shot_examples,
                example_prompt=example_prompt,
                prefix=prefix,
                suffix=suffix,
                input_variables=["A"],
                example_separator="",
            )
            all_prompt_templates.append(few_shot_prompt)

            # Prepare query templates_mixed for the prompts
            query_examples = self._prepare_configurations_acquisition(
                observed_fvals=jittered_desired_fval, seed=None, shuffle_features=shuffle_features
            )
            all_query_templates.append(query_examples)

        return all_prompt_templates, all_query_templates

    async def _async_generate(self, user_message: str) -> Optional[Tuple[Any, float, int]]:
        """
        Generate a response from the LLM asynchronously.

        Args:
            user_message (str): The prompt message to send to the LLM.

        Returns:
            Optional[tuple[Any, float, int]]: A tuple containing the LLM response, total cost, and total tokens used.
        """
        print("Sending inquiries to the LLM - acquisition function")

        message = []
        message.append(
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            }
        )
        message.append({"role": "user", "content": user_message})

        resp, tot_cost = self.OpenAI_instance.ask(message)
        tot_tokens = 1000

        # # Prepare the message for the LLM
        # message = []
        # message.append({"role": "system", "content": "You are an AI assistant that helps people find information."})
        # message.append({"role": "user", "content": user_message})
        #
        # MAX_RETRIES = 3
        #
        # async with ClientSession(trust_env=True) as session:
        #     openai.aiosession.set(session)
        #
        #     resp = None
        #     for retry in range(MAX_RETRIES):
        #         try:
        #             print("Sending inquiry to the LLM")
        #             print("-----prompt message beginning marker-----")
        #             for idx, msg in enumerate(message):
        #                 role = msg["role"].capitalize()
        #                 content = msg["content"]
        #                 print(f"{idx + 1}. {role}:")
        #                 print(f"   {content}\n")
        #             print("-----prompt message ending marker-----")
        #             print(f"Requesting {self.n_gens} responses")
        #
        #             start_time = time.time()
        #             self.rate_limiter.add_request(request_text=user_message, current_time=start_time)
        #             resp = await openai.ChatCompletion.acreate(
        #                 engine=self.chat_engine,
        #                 messages=message,
        #                 temperature=0.8,
        #                 max_tokens=500,
        #                 top_p=0.95,
        #                 n=self.n_gens,
        #                 request_timeout=300,
        #             )
        #
        #             print("The response from the LLM is below:")
        #
        #             print("-----response beginning marker-----")
        #             for choice in resp.get("choices", []):
        #                 index = choice.get("index", "N/A")
        #                 content = choice.get("message", {}).get("content", "No content available")
        #                 print(f"---Response {index + 1}:\n{content}\n---")
        #             print("-----response ending marker-----")
        #
        #             self.rate_limiter.add_request(request_token_count=resp["usage"]["total_tokens"], current_time=start_time)
        #             break
        #         except Exception as e:
        #             print(f"[AF] RETRYING LLM REQUEST {retry + 1}/{MAX_RETRIES}...")
        #             error_details = traceback.format_exc()
        #             print("Detailed Traceback:")
        #             print(error_details)
        #             print("The response is:", resp)
        #
        # await openai.aiosession.get().close()
        #
        # if resp is None:
        #     return None
        #
        # # Calculate the total cost and tokens used
        # tot_tokens = resp["usage"]["total_tokens"]
        # tot_cost = 0.0015 * (resp["usage"]["prompt_tokens"] / 1000) + 0.002 * (resp["usage"]["completion_tokens"] / 1000)

        return resp, tot_cost, tot_tokens

    async def _async_generate_concurrently(
        self,
        prompt_templates: List[FewShotPromptTemplate],
        query_templates: List[List[Dict[str, str]]],
    ) -> List[Optional[Tuple[Any, float, int]]]:
        """
        Perform concurrent generation of responses from the LLM asynchronously.

        Args:
            prompt_templates (list[FewShotPromptTemplate]): List of prompt templates_mixed.
            query_templates (list[list[dict[str, str]]]): List of query templates_mixed.

        Returns:
            list[Optional[tuple[Any, float, int]]]: A list of results from the LLM requests.
        """
        coroutines = []
        for prompt_template, query_template in zip(prompt_templates, query_templates):
            coroutines.append(
                self._async_generate(prompt_template.format(A=query_template[0]["A"]))
            )

        tasks = [asyncio.create_task(c) for c in coroutines]

        # Validate the number of tasks
        assert len(tasks) == int(self.n_templates)

        results = [None] * len(coroutines)

        llm_response = await asyncio.gather(*tasks)

        for idx, response in enumerate(llm_response):
            if response is not None:
                resp, tot_cost, tot_tokens = response
                results[idx] = (resp, tot_cost, tot_tokens)

        return results  # format [(resp, tot_cost, tot_tokens), None, (resp, tot_cost, tot_tokens)]

    def _convert_to_json(self, response_str: str) -> Dict[str, float]:
        """
        Parse LLM response string into JSON.

        Args:
            response_str (str): The LLM response string.

        Returns:
            dict[str, float]: A dictionary containing the parsed hyperparameter configuration.
        """
        pairs = response_str.split(",")
        response_json = {}
        for pair in pairs:
            key, value = [x.strip() for x in pair.split(":")]
            response_json[key] = float(value)

        return response_json

    def _filter_candidate_points(
        self,
        observed_points: List[Dict[str, float]],
        candidate_points: List[Dict[str, float]],
        precision: int = 8,
    ) -> pd.DataFrame:
        """
        Filter candidate points that already exist in observed points. Also remove duplicates.

        Args:
            observed_points (list[dict[str, float]]): List of observed hyperparameter configurations.
            candidate_points (list[dict[str, float]]): List of candidate hyperparameter configurations.
            precision (int): Number of decimal places to round values for comparison.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered candidate configurations.
        """
        # Drop points that already exist in observed points
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

        def is_within_range(value: float, allowed_range: Tuple[str, str, List[float]]) -> bool:
            """Check if a value is within an allowed range."""
            value_type, transform, search_range = allowed_range
            if value_type == "int":
                [min_val, max_val] = search_range
                if transform == "log" and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                    return min_val <= value <= max_val
                else:
                    return min_val <= value <= max_val and int(value) == value
            elif value_type == "float":
                [min_val, max_val] = search_range
                if transform == "log" and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                return min_val <= value <= max_val
            elif value_type == "ordinal":
                # Check that value is in allowed range up to 2 decimal places
                return any(math.isclose(value, x, abs_tol=1e-2) for x in allowed_range[2])
            else:
                raise Exception("Unknown hyperparameter value type")

        def is_dict_within_ranges(
            d: Dict[str, float], ranges_dict: Dict[str, Tuple[str, str, List[float]]]
        ) -> bool:
            """Check if all values in a dictionary are within their respective allowable ranges."""
            return all(
                key in ranges_dict and is_within_range(value, ranges_dict[key])
                for key, value in d.items()
            )

        def filter_dicts_by_ranges(
            dict_list: List[Dict[str, float]], ranges_dict: Dict[str, Tuple[str, str, List[float]]]
        ) -> List[Dict[str, float]]:
            """Return only those dictionaries where all values are within their respective allowable ranges."""
            return [d for d in dict_list if is_dict_within_ranges(d, ranges_dict)]

        # Check that constraints are satisfied
        print("DEBUG, before filtering", self.task_context)
        hyperparameter_constraints = self.task_context["hyperparameter_constraints"]
        filtered_candidates = filter_dicts_by_ranges(
            filtered_candidates, hyperparameter_constraints
        )
        print("DEBUG, after filtering", self.task_context)

        filtered_candidates = pd.DataFrame(filtered_candidates)
        # Drop duplicates
        filtered_candidates = filtered_candidates.drop_duplicates()
        # Reset index
        filtered_candidates = filtered_candidates.reset_index(drop=True)
        return filtered_candidates

    def get_candidate_points(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        use_feature_semantics: bool = True,
        use_context: str = "full_context",
        alpha: float = -0.2,
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Generate candidate points for the acquisition function.

        Args:
            observed_configs (pd.DataFrame): Observed hyperparameter configurations.
            observed_fvals (pd.DataFrame): Observed performance values.
            use_feature_semantics (bool): Whether to use feature names in the prompts.
            use_context (str): Level of context to include in the prompts.
            alpha (float): Controls how much better (or worse) the target performance is compared to the best observed performance.

        Returns:
            tuple[pd.DataFrame, float, float]: A tuple containing the filtered candidate points, total cost, and time taken.
        """
        print("DEBUG: Initial task_context:", self.task_context)  # Debug message

        assert -1 <= alpha <= 1, "alpha must be between -1 and 1"
        if alpha == 0:
            alpha = -1e-3  # A little bit of randomness never hurt anyone
        self.alpha = alpha

        if self.prompt_setting is not None:
            use_context = self.prompt_setting

        # Generate prompt templates_mixed
        start_time = time.time()

        # Get desired f_val for candidate points
        range = np.abs(np.max(observed_fvals.values) - np.min(observed_fvals.values))
        if range == 0:
            # Sometimes there is no variability in y :')
            range = 0.1 * np.abs(np.max(observed_fvals.values))
        alpha_range = [0.1, 1e-2, 1e-3, -1e-3, -1e-2, 1e-1]

        print("DEBUG", "acq 1")
        if self.lower_is_better:
            self.observed_best = np.min(observed_fvals.values)
            self.observed_worst = np.max(observed_fvals.values)
            range = self.observed_worst - self.observed_best
            desired_fval = self.observed_best - alpha * range

            print(
                f"\n[DEBUG] Initial values - Best: {self.observed_best:.6f}, Worst: {self.observed_worst:.6f}, Range: {range:.6f}"
            )
            print(f"[DEBUG] Starting alpha: {alpha}, Initial desired_fval: {desired_fval:.6f}")

            iteration = 0
            max_iterations = 10  # Safety net

            while desired_fval <= 0.00001 and iteration < max_iterations:
                print(
                    f"\n[DEBUG] Iteration {iteration} - Current alpha: {alpha:.4f}, desired_fval: {desired_fval:.6f}"
                )
                alpha_updated = False

                for i, alpha_ in enumerate(alpha_range):
                    print(f"[DEBUG] Trying alpha_{i}: {alpha_:.4f}")
                    if alpha_ < alpha:
                        alpha = alpha_
                        desired_fval = self.observed_best - alpha * range
                        alpha_updated = True
                        print(
                            f"[DEBUG] Found new alpha: {alpha:.4f}, new desired_fval: {desired_fval:.6f}"
                        )
                        break

                if not alpha_updated:
                    print("[WARNING] No smaller alpha found in alpha_range! Breaking loop")
                    break

                iteration += 1

            print(f"\n[DEBUG] Final alpha: {alpha:.4f}, Final desired_fval: {desired_fval:.6f}")
            print(
                f"Adjusted alpha: {alpha} | [original alpha: {self.alpha}], desired fval: {desired_fval:.6f}"
            )

        else:
            # Similar debug prints for the else case
            self.observed_best = np.max(observed_fvals.values)
            self.observed_worst = np.min(observed_fvals.values)
            range = self.observed_best - self.observed_worst
            desired_fval = self.observed_best + alpha * range

            print(
                f"\n[DEBUG] Initial values - Best: {self.observed_best:.6f}, Worst: {self.observed_worst:.6f}, Range: {range:.6f}"
            )
            print(f"[DEBUG] Starting alpha: {alpha}, Initial desired_fval: {desired_fval:.6f}")

            iteration = 0
            max_iterations = 10  # Safety net

            while desired_fval >= 0.9999 and iteration < max_iterations:
                print(
                    f"\n[DEBUG] Iteration {iteration} - Current alpha: {alpha:.4f}, desired_fval: {desired_fval:.6f}"
                )
                alpha_updated = False

                for i, alpha_ in enumerate(alpha_range):
                    print(f"[DEBUG] Trying alpha_{i}: {alpha_:.4f}")
                    if alpha_ < alpha:
                        alpha = alpha_
                        desired_fval = self.observed_best + alpha * range
                        alpha_updated = True
                        print(
                            f"[DEBUG] Found new alpha: {alpha:.4f}, new desired_fval: {desired_fval:.6f}"
                        )
                        break

                if not alpha_updated:
                    print("[WARNING] No smaller alpha found in alpha_range! Breaking loop")
                    break

                iteration += 1

            print(f"\n[DEBUG] Final alpha: {alpha:.4f}, Final desired_fval: {desired_fval:.6f}")
            print(
                f"Adjusted alpha: {alpha} | [original alpha: {self.alpha}], desired fval: {desired_fval:.6f}"
            )

        self.desired_fval = desired_fval

        print(
            "DEBUG: task_context after setting desired_fval:", self.task_context
        )  # Debug message

        if self.warping_transformer is not None:
            observed_configs = self.warping_transformer.warp(observed_configs)

        print(
            "DEBUG: task_context after warping observed_configs:", self.task_context
        )  # Debug message

        prompt_templates, query_templates = self._gen_prompt_tempates_acquisitions(
            observed_configs,
            observed_fvals,
            desired_fval,
            n_prompts=self.n_templates,
            use_context=use_context,
            use_feature_semantics=use_feature_semantics,
            shuffle_features=self.shuffle_features,
        )

        print(
            "DEBUG: task_context after generating prompt templates_mixed:", self.task_context
        )  # Debug message

        print("=" * 100)
        print("EXAMPLE ACQUISITION PROMPT")
        print(f"Length of prompt templates_mixed: {len(prompt_templates)}")
        print(f"Length of query templates_mixed: {len(query_templates)}")
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
            tot_tokens = 0
            # Loop through n_coroutine async calls
            for response in llm_responses:
                if response is None:
                    continue
                # Loop through n_gen responses
                for response_content in response:
                    try:
                        response_content = response_content.split("##")[1].strip()
                        candidate_points.append(self._convert_to_json(response_content))
                    except:
                        print(response_content)
                        continue
                tot_cost += response[1]
                tot_tokens += response[2]

            print("DEBUG: task_context after LLM responses:", self.task_context)  # Debug message

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
            if retry > 3:
                print(f"Desired fval: {desired_fval:.6f}")
                print(f"Number of proposed candidate points: {len(candidate_points)}")
                print(f"Number of accepted candidate points: {filtered_candidate_points.shape[0]}")
                if len(candidate_points) > 5:
                    filtered_candidate_points = pd.DataFrame(candidate_points)
                    break
                else:
                    raise Exception("LLM failed to generate candidate points")

        if self.warping_transformer is not None:
            filtered_candidate_points = self.warping_transformer.unwarp(filtered_candidate_points)

        print(
            "DEBUG: task_context after unwarping candidate points:", self.task_context
        )  # Debug message

        end_time = time.time()
        time_taken = end_time - start_time

        return filtered_candidate_points, tot_cost, time_taken
