from __future__ import annotations

from typing import Optional

from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
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
