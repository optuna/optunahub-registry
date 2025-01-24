import asyncio
import re
import time

from llambo.generative_sm_utils import gen_prompt_tempates
from llambo.rate_limiter import RateLimiter
from LLM_utils.inquiry import OpenAI_interface
import numpy as np
import pandas as pd


class LLM_GEN_SM:
    def __init__(
        self,
        task_context,
        n_gens,
        lower_is_better,
        top_pct,
        n_templates=1,
        rate_limiter=None,
        verbose=False,
        key: str = "",
        model: str = "gpt-4o-mini",
    ):
        """Initialize the forward LLM surrogate model. This is modelling p(y|x) as in GP/SMAC etc."""
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.top_pct = top_pct
        self.n_templates = n_templates
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=240000, time_frame=60, max_requests=2900)
        else:
            self.rate_limiter = rate_limiter
        self.recalibrator = None
        self.OpenAI_instance = OpenAI_interface(key, model=model, debug=False)
        self.verbose = verbose

    async def _async_generate(self, few_shot_template, query_example, query_idx):
        print("Sending inquiries to the LLM - generative surrogate model")

        # Generate the prompt
        prompt = few_shot_template.format(Q=query_example["Q"])

        message = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            },
            {"role": "user", "content": prompt},
        ]

        # Send the request to the LLM
        resp, tot_cost = self.OpenAI_instance.ask(message)
        tot_tokens = 1000  # Placeholder, replace with actual token count if available

        return query_idx, resp, tot_cost, tot_tokens

    async def _generate_concurrently(self, few_shot_templates, query_examples):
        """Perform concurrent generation of responses from the LLM async."""
        coroutines = []
        for template in few_shot_templates:
            for query_idx, query_example in enumerate(query_examples):
                coroutines.append(self._async_generate(template, query_example, query_idx))

        tasks = [asyncio.create_task(c) for c in coroutines]

        results = [[] for _ in range(len(query_examples))]  # Nested list

        llm_response = await asyncio.gather(*tasks)

        for response in llm_response:
            if response is not None:
                query_idx, resp, tot_cost, tot_tokens = response
                results[query_idx].append([resp, tot_cost, tot_tokens])
            else:
                print(f"None response received for query_idx: {query_idx}")

        return results

    def process_response(self, all_raw_response):
        all_pred_probs = []  # p(s<\tau | h)
        for raw_response in all_raw_response:
            # Extract numeric value from the string using regex
            if isinstance(raw_response, str):
                gen_pred = re.findall(r"## (-?[\d.]+) ##", raw_response)
                if len(gen_pred) == 1:
                    all_pred_probs.append(float(gen_pred[0]))  # Append the extracted value
                else:
                    print("No valid numeric value found in raw_response, appending NaN")
                    all_pred_probs.append(np.nan)  # Append NaN if no match or multiple matches
            else:
                print("raw_response is not a string, appending NaN")
                all_pred_probs.append(np.nan)  # Append NaN if raw_response is not a string

        return all_pred_probs

    async def _predict(self, all_prompt_templates, query_examples):
        start = time.time()
        all_preds = []
        tot_tokens = 0
        tot_cost = 0

        bool_pred_returned = []

        # Make predictions in chunks of 5, for each chunk make concurrent calls
        for i in range(0, len(query_examples), 5):
            query_chunk = query_examples[i : i + 5]

            chunk_results = await self._generate_concurrently(all_prompt_templates, query_chunk)

            bool_pred_returned.extend(
                [1 if x is not None else 0 for x in chunk_results]
            )  # Track effective number of predictions returned

            for _, sample_response in enumerate(chunk_results):
                if not sample_response:  # If sample prediction is an empty list
                    sample_preds = [np.nan] * self.n_gens
                else:
                    all_raw_response = []
                    for template_response in sample_response:
                        if isinstance(template_response, list) and len(template_response) > 0:
                            # Extract the LLM response (first element of template_response)
                            llm_response = template_response[0]
                            if isinstance(llm_response, str):
                                all_raw_response.append(llm_response)
                            else:
                                print(f"LLM response is not a string: {llm_response}")
                                all_raw_response.append(np.nan)
                        else:
                            print(f"Invalid template_response: {template_response}")
                            all_raw_response.append(np.nan)

                    sample_preds = self.process_response(all_raw_response)
                    tot_cost += sum(
                        [x[1] for x in sample_response if isinstance(x, list) and len(x) > 1]
                    )
                    tot_tokens += sum(
                        [x[2] for x in sample_response if isinstance(x, list) and len(x) > 2]
                    )
                all_preds.append(sample_preds)

        end = time.time()
        time_taken = end - start

        success_rate = sum(bool_pred_returned) / len(bool_pred_returned)

        pred_probs = np.array(all_preds).astype(float)
        mean_probs = np.nanmean(pred_probs, axis=1)

        return mean_probs, success_rate, tot_cost, tot_tokens, time_taken

    async def _evaluate_candidate_points(
        self, observed_configs, observed_fvals, candidate_configs
    ):
        """Evaluate candidate points using the LLM model."""
        all_run_cost = 0
        all_run_time = 0

        # Ensure observed_configs and candidate_configs are DataFrames
        if not isinstance(observed_configs, pd.DataFrame):
            observed_configs = pd.DataFrame(observed_configs)
        if not isinstance(candidate_configs, pd.DataFrame):
            candidate_configs = pd.DataFrame(candidate_configs)

        # Use the original hyperparameter_constraints without modification
        hyperparameter_constraints = self.task_context["hyperparameter_constraints"]

        # Generate prompt templates_mixed and query examples
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

        # Make predictions using the LLM
        response = await self._predict(all_prompt_templates, query_examples)

        pred_probs, success_rate, tot_cost, tot_tokens, time_taken = response

        all_run_cost += tot_cost
        all_run_time += time_taken

        return pred_probs, all_run_cost, all_run_time

    def _warp_candidate_points(self, configurations):
        """Warp candidate points to log scale if necessary."""
        if not isinstance(configurations, pd.DataFrame):
            configurations = pd.DataFrame(configurations)

        warped_configs = configurations.copy().to_dict(orient="records")
        hyperparameter_constraints = self.task_context["hyperparameter_constraints"]
        for config in warped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == "log":
                    config[hyperparameter] = np.log10(config[hyperparameter])

        warped_configs = pd.DataFrame(warped_configs)
        return warped_configs

    def _unwarp_candidate_points(self, configurations):
        """Unwarp candidate points from log scale if necessary."""
        if not isinstance(configurations, pd.DataFrame):
            configurations = pd.DataFrame(configurations)

        unwarped_configs = configurations.copy().to_dict(orient="records")
        hyperparameter_constraints = self.task_context["hyperparameter_constraints"]
        for config in unwarped_configs:
            for hyperparameter, constraint in hyperparameter_constraints.items():
                if constraint[1] == "log":
                    config[hyperparameter] = 10 ** config[hyperparameter]

        unwarped_configs = pd.DataFrame(unwarped_configs)
        return unwarped_configs

    def select_query_point(
        self, observed_configs, observed_fvals, candidate_configs, return_raw_preds=False
    ):
        """Select the next query point using expected improvement."""
        # Ensure observed_configs and candidate_configs are DataFrames
        if not isinstance(observed_configs, pd.DataFrame):
            observed_configs = pd.DataFrame(observed_configs)
        if not isinstance(candidate_configs, pd.DataFrame):
            candidate_configs = pd.DataFrame(candidate_configs)

        # Warp candidate points
        observed_configs = self._warp_candidate_points(observed_configs)
        candidate_configs = self._warp_candidate_points(candidate_configs)

        pred_probs, cost, time_taken = asyncio.run(
            self._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs)
        )

        best_point_index = np.argmax(pred_probs)

        # Unwarp candidate points
        candidate_configs = self._unwarp_candidate_points(candidate_configs)

        best_point = candidate_configs.iloc[
            [best_point_index], :
        ]  # Return selected point as dataframe not series

        if return_raw_preds:
            return best_point, pred_probs, cost, time_taken
        else:
            return best_point, cost, time_taken
