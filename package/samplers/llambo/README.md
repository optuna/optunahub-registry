---
author: [Jinglue Xu, Tennison Liu, Nicolás Astorga, Nabeel Seedat, and Mihaela van der Schaar]
title: LLAMBO (Large Language Models to Enhance Bayesian Optimization)
description: This repository integrates Large Language Models (LLMs) with Bayesian Optimization, enabling enhanced surrogate modeling, zero-shot warmstarting, and more efficient candidate sampling in hyperparameter optimization and other black-box optimization tasks.
tags: [sampler, LLM, Bayesian Optimization, generative, discriminative, dynamic search space]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

### Large Language Models to Enhance Bayesian Optimization (LLAMBO)

**LLAMBO**, by [Liu et al.](https://arxiv.org/pdf/2402.03921), is a novel approach that integrates Large Language Models (LLMs) into the Bayesian Optimization (BO) framework to improve the optimization of complex, expensive-to-evaluate black-box functions. By leveraging the contextual understanding and few-shot learning capabilities of LLMs, LLAMBO enhances multiple facets of the BO pipeline:

1. **Zero-Shot Warmstarting**\
   LLAMBO frames the optimization problem in natural language, allowing the LLM to propose promising initial solutions. This jump-starts the search by exploiting the LLM’s pre-trained knowledge base.

1. **Enhanced Surrogate Modeling**\
   Traditional BO uses surrogate models (e.g., Gaussian Processes) trained solely on observed data. LLAMBO augments this with the LLM’s few-shot learning capacity, particularly beneficial in sparse data regimes.

1. **Efficient Candidate Sampling**\
   LLAMBO orchestrates iterative sampling by conditioning the LLM on both historical evaluations and high-level problem context. This results in candidate points that effectively balance exploration and exploitation.

1. **Cost Information**\
   As of March 2025, using GPT-4o-mini costs $0.150 per 1M input tokens and $0.600 per 1M output tokens. With such pricing, the estimated cost per run with `n_trials = 30` ranges between $0.05 and $0.10. This implementation also displays the accumulated cost at the end of each trial."

### Implementation

This implementation of LLAMBO differs from the [original implementation](https://github.com/tennisonliu/LLAMBO/) in several key ways:

1. **Categorical Variable Handling**: It delegates categorical variables to [random search](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html), improving flexibility in optimization.

1. **Adaptation beyond Tabular Machine Learning Tasks**:

   - The prompting style is adapted to work beyond tabular machine learning tasks to general black-box optimization problems.
   - Users can specify a custom task description using the `custom_task_description` parameter of `LLAMBOSampler`.

1. **Improved Prompt Templates to Prevent Repetition**:

   - This implementation includes an explicit prompt that ensures previously observed values are **not recommended again**, reducing redundancy:
     ```
     The following values have already been observed and **must not be recommended again**:  
     f"{observed_values_str}"  
     ```
   - It also enforces integer constraints by explicitly stating:
     ```
     "Do not recommend float values; you can only recommend integer values."  
     ```

1. **Adaptive Few-Shot Templates for Type-Aware Formatting**:

   - The system generates hyperparameter values with precision appropriate to their type in the few-shot templates.
   - For float-valued hyperparameters, it maintains proper decimal precision (ensuring at least one decimal place).
   - For integer-valued hyperparameters, it formats values as whole numbers without decimal places.
   - This type-aware formatting helps the LLM better understand the precision requirements for different hyperparameter types.

1. **Alternative Rate Limiting Mechanism**:

   - A different rate limiter is implemented (see `llambo/rate_limiter.py`) to regulate LLM call rate limits for general users.

______________________________________________________________________

## APIs

______________________________________________________________________

### 1. `LLAMBOSampler`

**File Location**

- `sampler_base.py`

**Description**\
A custom Optuna sampler that integrates LLAMBO-based surrogate modeling into the search process. The sampler splits parameters into *numerical* (handled by LLAMBO) and *categorical* (handled by a `RandomSampler`). For numerical parameters, it uses either a *discriminative* or *generative* LLM-based surrogate model, depending on `sm_mode`.

```python
class LLAMBOSampler(SimpleBaseSampler):
    def __init__(
        self,
        custom_task_description: Optional[str] = None,
        n_initial_samples: int = 5,
        sm_mode: str = "generative",
        num_candidates: int = 10,
        n_templates: int = 2,
        n_gens: int = 10,
        alpha: float = 0.1,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        max_requests_per_minute: int = 100,
        search_space: Optional[dict[str, optuna.distributions.BaseDistribution]] = None,
        seed: Optional[int] = None,
    ):
        ...
```

#### **Parameters**

- **`custom_task_description`** *(str, optional)*\
  A user-defined description of the optimization task, used for prompt generation. This can be a concise one-sentence summary or a detailed, multi-paragraph explanation.

  For example, the prompt can be:

  ```
    Optimize RandomForest hyperparameters for digit classification.
  ```

  or:

  ```
    Optimize a RandomForest classifier for handwritten digit classification using the digits dataset.
    
    Dataset information:
    - The digits dataset contains 1797 8x8 images of handwritten digits (0-9)
    - Each image is represented as a 64-dimensional feature vector
    - The task is a 10-class classification problem

    Hyperparameters to tune:
    - n_estimators: Number of trees in the forest (range: 10-1000)
    - max_depth: Maximum depth of each tree (range: 2-100)
    - min_samples_split: Minimum samples required to split a node (range: 2-50)
    - min_samples_leaf: Minimum samples required at a leaf node (range: 1-30)
    - max_features: Number of features to consider when looking for the best split (options: "sqrt", "log2", None)
    - bootstrap: Whether to use bootstrap samples (options: True, False)
    - criterion: Function to measure split quality (options: "gini", "entropy", "log_loss")
    - ccp_alpha: Complexity parameter for minimal cost-complexity pruning (range: 0.0-0.1)
    - max_leaf_nodes: Maximum number of leaf nodes (range: 2-1000)
    - min_impurity_decrease: Minimum impurity decrease required for split (range: 0.0-0.5)

    Background knowledge:
    - RandomForest is an ensemble method that combines multiple decision trees
    - It's generally robust to overfitting but may require tuning to achieve optimal performance
    - For small datasets like digits, a moderate number of trees with controlled depth often works well
    - The performance metric is classification accuracy
  ```

- **`n_initial_samples`** *(int, default=5)*\
  Number of initial random samples before LLAMBO-based sampling starts.

- **`sm_mode`** *({"discriminative", "generative"}, default="generative")*\
  Defines which LLM-based surrogate model to use for numerical parameters.

- **`num_candidates`** *(int, default=10)*\
  Number of candidate points generated in each iteration for the surrogate model.

- **`n_templates`** *(int, default=2)*\
  Number of prompt templates to be used when querying the LLM-based surrogate.

- **`n_gens`** *(int, default=10)*\
  Number of generations/predictions made per prompt template.

- **`alpha`** *(float, default=0.1)*\
  Exploration-exploitation parameter for the acquisition function.

- **`api_key`** *(str, optional)*\
  API key for the language model service (e.g., OpenAI).

- **`model`** *(str, default="gpt-4o-mini")*\
  Identifier for the chosen LLM model. Currently supports supports gpt-4o-mini, gpt-4o, deepseek-chat, and deepseek-reasoner.

- **`max_requests_per_minute`** *(int, default=100)*\
  Maximum rate of LLM queries per minute.  This is especially useful when your API has a rate limit, such as the restrictions in OpenAI's free-tier plan.

- **`search_space`** *(dict, optional)*\
  If specified, defines the parameter distributions for the sampler. Leave it as `None` in general.

- **`seed`** *(int, optional)*\
  Seed for random number generation to ensure reproducible sampling.

#### **Key Methods**

1. **`sample_relative`**\
   Generates a parameter configuration for an incoming Optuna trial. Splits the search space, sampling numerical parameters using LLAMBO and categorical parameters with a random sampler.

1. **`after_trial`**\
   Updates the LLAMBO surrogate model after each completed trial, so that future sampling can leverage newly observed data.

1. **`generate_random_samples`**\
   Helper method to provide random initial samples for numeric distributions before the LLM-based surrogate is employed.

______________________________________________________________________

### 2. `LLAMBO`

**File Location**

- `llambo/llambo.py`

**Description**\
Serves as the central orchestrator of LLAMBO. Manages the overall optimization loop, including the surrogate model (generative or discriminative) and the acquisition function.

```python
class LLAMBO:
    def __init__(
        self,
        task_context: Dict[str, Any],
        sm_mode: str,
        n_candidates: int = 10,
        ...
    ):
        ...
```

#### **Key Parameters**

- **`task_context`** *(Dict\[str, Any\])*\
  Includes metadata describing the optimization goal, such as whether *lower is better*, a dictionary of hyperparameter constraints, etc.

- **`sm_mode`** *(str)*\
  `"discriminative"` or `"generative"`, selecting which type of LLM-based surrogate model to use.

- **`n_candidates`** *(int, default=10)*\
  Number of candidate points sampled at each iteration.

- **`n_templates`** *(int, default=2)*, **`n_gens`** *(int, default=10)*, **`alpha`** *(float, default=0.1)*\
  Various tuning parameters that affect how prompts are created and how exploration is balanced against exploitation.

- **`use_input_warping`** *(bool, default=False)*\
  Whether to apply transformations (e.g., log-scaling) to numeric parameters before passing them to the LLM.

- **`key`** *(str, optional)*, **`model`** *(str, optional)*, **`max_requests_per_minute`** *(int, default=100)*\
  Relevant LLM API credentials and rate limit settings.

#### **Key Methods**

1. **`_initialize`**\
   Prepares the optimizer, optionally using user-provided initial observations or generating them randomly.

1. **`sample_configurations`**\
   Returns new candidate configurations to evaluate, using the underlying acquisition function to propose points.

1. **`update_history`**\
   Adds newly observed configurations and their objective values to the model’s dataset, improving future sampling.

______________________________________________________________________

### 3. Surrogate Models

LLAMBO provides two main types of LLM-based surrogate models, each providing a unique approach to predicting objective function behavior.

#### **3.1 `LLMDiscriminativeSM`**

**File Location**

- `llambo/discriminative_sm.py`

**Description**\
A discriminative approach that prompts the LLM to output direct performance predictions (numeric values) for candidate configurations. Useful for tasks where approximate numeric predictions are feasible.

```python
class LLMDiscriminativeSM:
    def __init__(
        self,
        task_context: Any,
        n_gens: int,
        lower_is_better: bool,
        ...
    ):
        ...
```

#### **Key Parameters**

- **`lower_is_better`** *(bool)*\
  Informs the LLM about the direction of optimization.
- **`n_gens`** *(int)*\
  Number of generations per query.
- **`use_recalibration`** *(bool)*\
  Whether to apply a recalibration step on predicted outputs.

**Key Methods**

- **`select_query_point(...)`**: Main entry point for selecting the next candidate based on predicted mean and variance (via an Expected Improvement–style approach).
- **`_evaluate_candidate_points(...)`**: Evaluates a set of candidate points by prompting the LLM for numeric predictions (and optionally uncertainties).

______________________________________________________________________

#### **3.2 `LLMGenerativeSM`**

**File Location**

- `llambo/generative_sm.py`

**Description**\
A generative approach that classifies configurations into *top-tier* vs. *not top-tier* using user-defined thresholds. Rather than producing a numeric performance estimate, it returns a probability or classification label.

```python
class LLMGenerativeSM:
    def __init__(
        self,
        task_context: dict[str, Any],
        n_gens: int,
        lower_is_better: bool,
        top_pct: float,
        ...
    ):
        ...
```

#### **Key Parameters**

- **`top_pct`** *(float)*\
  Percentage threshold to determine “top performing” vs. not.
- **`n_templates`**, **`n_gens`**\
  Similar usage as in the discriminative model for generating queries and responses.

**Key Methods**

- **`select_query_point(...)`**: Evaluates candidate configurations by prompting the LLM for a classification (e.g., “is this in the top 10%?”) and chooses the best candidate accordingly.
- **`_evaluate_candidate_points(...)`**: Generates classification-based predictions for each candidate in parallel.

______________________________________________________________________

### 4. `LLM_ACQ` (Acquisition Function)

**File Location**

- `llambo/acquisition_function.py`

**Description**\
Implements the acquisition function that suggests candidate points. Uses textual prompts to the LLM, describing a *desired performance target* based on previously observed data. The LLM proposes new points that can (hopefully) meet or exceed that performance target.

```python
class LLM_ACQ:
    def __init__(
        self,
        task_context: dict[str, Any],
        n_candidates: int,
        n_templates: int,
        lower_is_better: bool,
        ...
    ):
        ...
```

#### **Key Parameters**

- **`n_candidates`** *(int)*\
  Number of configurations to generate at once.
- **`n_templates`** *(int)*\
  Number of different prompt templates to help diversify LLM outputs.
- **`lower_is_better`** *(bool)*\
  Direction of optimization (influences how performance thresholds are derived).

#### **Key Methods**

1. **`get_candidate_points(...)`**\
   Creates textual prompts describing the “desired_fval” for the objective. Retrieves LLM suggestions, then filters out duplicates or out-of-bounds suggestions.

1. **`_filter_candidate_points(...)`**\
   Ensures that the proposed candidate points do not replicate existing solutions, remain within the specified bounds, and respect integer constraints.

______________________________________________________________________

### 5. Utility Modules

Beyond the sampler, surrogate models, and acquisition function, LLAMBO includes several utility components.

#### **5.1 `rate_limiter.py`**

**Classes**

- **`RateLimiter`**
- **`OpenAIRateLimiter`**

These classes enforce request-per-minute constraints to avoid exceeding LLM API quotas. They can be applied as decorators or used directly to wrap LLM calls.

```python
limiter = RateLimiter(max_requests_per_minute=60)
@limiter.rate_limited
def some_llm_call(...):
    ...
```

#### **5.2 `warping.py`**

Implements numeric warping/unwarping for parameters with log-scale or other transformations:

```python
class NumericalTransformer:
    def warp(...):
        ...
    def unwarp(...):
        ...
```

#### **5.3 `discriminative_sm_utils.py` / `generative_sm_utils.py`**

Helper functions for constructing prompts, enumerating data frames, and shuffling or formatting feature columns before sending them to the LLM.

______________________________________________________________________

## Installation

The recommended Python version is `3.10` and above.

1. `optuna` and `optunahub` are required.
1. `pip install -r https://hub.optuna.org/samplers/llambo/requirements.txt`

(You also need credentials or an API key for your chosen LLM.)

______________________________________________________________________

## Example

Below is a minimal example using **LLAMBOSampler** with Optuna:

```python
import optuna
from optuna import Trial
import optunahub
import os

# 1. Define a sample objective function (e.g., a simple 2D function)
def objective(trial: Trial):
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return (x**2 + y**2)  # Minimization

# 2. Instantiate LLAMBOSampler

api_key = os.environ.get(
    "API_KEY",
    "", # Replace with your actual key or load via env variable
)

module = optunahub.load_module("samplers/llambo")
LLAMBOSampler = module.LLAMBOSampler
sampler = LLAMBOSampler(
    custom_task_description="Minimize x^2 + y^2 over the range [-5, 5].",
    sm_mode="generative",   # or "discriminative"
    api_key=api_key, 
    model="gpt-4o-mini", # supports gpt-4o-mini, gpt-4o, deepseek-chat, and deepseek-reasoner
)

# 3. Create an Optuna study and optimize
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=30)

# 4. Print results
print("Best value:", study.best_value)
print("Best params:", study.best_params)
```

1. **Objective**: Illustrates a basic 2D paraboloid to be minimized.
1. **LLAMBOSampler**: Uses the generative LLM-based model to suggest numeric parameters.
1. **Study**: Runs 30 trials, of which the initial few use random sampling, with subsequent trials guided by LLAMBO’s surrogate.
1. **Results**: Print the minimal value found and the associated `x, y`.
