from __future__ import annotations

from typing import Optional

from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
import numpy as np
import pandas as pd


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

    Example:
        >>> constraints = {'learning_rate': [0.001, 0.1, [0.001]]}
        >>> configs = pd.DataFrame({'learning_rate': [0.01, 0.05]})
        >>> prepare_configurations(constraints, True, 0.5, configs)
        [{'Q': 'learning_rate: 0.010'}, {'Q': 'learning_rate: 0.050'}]
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
            lower_bound = hyperparameter_constraints[hyperparameter_names[i]][2]
            n_dp = _count_decimal_places(lower_bound[0]) + 2

            value = (
                f"{row[i]:.{n_dp}f}"
                if isinstance(row[i], float) and not row[i] % 1 == 0
                else str(row[i])
            )
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
    Generate prompt templates for the few-shot learning task.

    Args:
        task_context (dict): Context information about the optimization task.
        observed_configs (pd.DataFrame): Previously observed configurations.
        observed_fvals (pd.DataFrame): Observed function values.
        candidate_configs (pd.DataFrame): Candidate configurations to evaluate.
        lower_is_better (bool): Whether lower values indicate better performance.
        top_pct (float): Percentage threshold for top performers.
        n_prompts (int): Number of prompt templates to generate.

    Returns:
        tuple[list[FewShotPromptTemplate], list[dict[str, str]]]: Tuple containing:
            - List of prompt templates
            - List of query examples

    Example:
        >>> context = {
        ...     "hyperparameter_constraints": {"lr": ["float", "linear", [0.001, 0.1]]}
        ... }
        >>> observed = pd.DataFrame({"lr": [0.01, 0.05]})
        >>> fvals = pd.DataFrame({"value": [0.8, 0.9]})
        >>> candidates = pd.DataFrame({"lr": [0.03]})
        >>> templates, queries = gen_prompt_tempates(
        ...     context, observed, fvals, candidates, False, 0.5
        ... )
    """
    # Get custom task description if available
    custom_task_description = task_context.get("custom_task_description")
    all_prompt_templates: list[FewShotPromptTemplate] = []

    for i in range(n_prompts):
        # Prepare few-shot examples using observed configurations and values
        few_shot_examples = prepare_configurations(
            task_context["hyperparameter_constraints"],
            lower_is_better,
            top_pct,
            observed_configs,
            observed_fvals,
            seed=i,
        )

        # Define the example template format for each configuration-performance pair
        example_template = """
Hyperparameter configuration: {Q}
Classification: {A}"""

        example_prompt = PromptTemplate(
            input_variables=["Q", "A"],
            template=example_template,
        )

        # Build the prefix for the prompt template
        prefix = (
            "The following are examples of hyperparameter configurations "
            "for a black-box optimization task. "
        )
        if custom_task_description is not None:
            prefix += "Below is a description of the task:\n"
            prefix += custom_task_description
            prefix += "\n"

        # Add information about the classification scheme
        prefix += (
            f"The performance classification is 1 if the configuration is in the "
            f"best-performing {top_pct * 100}% of all configurations and 0 otherwise. "
        )
        prefix += (
            "Your response should only contain the predicted performance "
            "classification in the format ## performance classification ##."
        )

        # Define the suffix for querying new configurations
        suffix = """
Hyperparameter configuration: {Q}
Classification: """

        # Create the few-shot prompt template
        few_shot_prompt = FewShotPromptTemplate(
            examples=few_shot_examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["Q"],
            example_separator="",
        )
        all_prompt_templates.append(few_shot_prompt)

    # Prepare query examples using candidate configurations
    query_examples = prepare_configurations(
        task_context["hyperparameter_constraints"],
        lower_is_better,
        top_pct,
        candidate_configs,
        seed=None,
    )

    return all_prompt_templates, query_examples
