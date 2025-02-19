from __future__ import annotations

from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
import numpy as np


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

    Example:
        >>> constraints = {"learning_rate": ["float", "linear", (0.001, 0.1)]}
        >>> configs = pd.DataFrame({"learning_rate": [0.01, 0.05]})
        >>> prepare_configurations(constraints, configs)
        [{'Q': 'learning_rate is 0.010'}, {'Q': 'learning_rate is 0.050'}]
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

            n_dp = _count_decimal_places(lower_bound)
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
    Generate prompt templates for the few-shot learning task.

    Args:
        task_context (dict): Context information for the task.
        observed_configs (pd.DataFrame): Observed hyperparameter configurations.
        observed_fvals (pd.DataFrame): Observed performance values.
        candidate_configs (pd.DataFrame): Candidate configurations to evaluate.
        n_prompts (int): Number of prompt templates to generate.
        bootstrapping (bool): Whether to use bootstrap resampling.
        use_feature_semantics (bool): Whether to use feature names in output.
        shuffle_features (bool): Whether to shuffle feature order.
        apply_warping (bool): Whether to apply warping to numeric values.

    Returns:
        tuple[list[FewShotPromptTemplate], list[dict[str, str]]]: Generated prompt
            templates and query examples.

    Example:
        >>> context = {"hyperparameter_constraints": {...}}
        >>> configs = pd.DataFrame(...)
        >>> fvals = pd.DataFrame(...)
        >>> candidates = pd.DataFrame(...)
        >>> templates, examples = gen_prompt_templates(
        ...     context, configs, fvals, candidates
        ... )
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
            prefix += "Below is a description of the task:\n" f"{custom_task_description}\n"
        prefix += (
            "Your response should only contain the predicted performance in the "
            "format ## performance ##."
        )

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
