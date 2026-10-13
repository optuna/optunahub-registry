---
author: Sanjana Reddy Katham
title: SVM Hyperparameter Optimization Benchmark
description: A cheap real-world benchmark that tunes the C, gamma, kernel and conditional degree of a scikit-learn SVM classifier by minimizing its cross-validated error on built-in datasets.
tags: [benchmark, HPO, hyperparameter optimization, SVM, conditional search space, real world problem]
optuna_versions: [5.0.0]
license: MIT License
---

## Abstract

This package provides a hyperparameter optimization (HPO) benchmark based on a Support Vector Machine classifier (`sklearn.svm.SVC`).
Each evaluation trains the SVM with stratified k-fold cross-validation on a small dataset that ships with scikit-learn, and returns the mean classification error (`1 - accuracy`), which is minimized.

The benchmark is useful for testing samplers on a real machine learning task that runs in seconds on a laptop, without downloading any data or precomputed tables.
Its search space mixes three kinds of parameters:

- log-scale continuous parameters: `C` (regularization strength) and `gamma` (kernel coefficient),
- a categorical parameter: `kernel`,
- a conditional integer parameter: `degree`, which is only sampled when `kernel == "poly"`.

## APIs

### class `Problem(dataset_name: str = "breast_cancer", n_splits: int = 5, seed: int | None = None)`

- `dataset_name`: Name of the built-in scikit-learn dataset. One of `"breast_cancer"`, `"wine"` and `"digits"`. See `Problem.available_dataset_names`.
- `n_splits`: Number of folds for stratified k-fold cross-validation. Must be at least 2.
- `seed`: Seed that controls how the cross-validation folds are shuffled. The same seed always gives the same result for the same parameters.

#### Search Space

| Parameter | Distribution                                                    | Notes                                                                    |
| --------- | --------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `C`       | `FloatDistribution(1e-3, 1e3, log=True)`                        | Regularization strength.                                                 |
| `gamma`   | `FloatDistribution(1e-4, 1e1, log=True)`                        | Used by the `rbf`, `poly` and `sigmoid` kernels and ignored by `linear`. |
| `kernel`  | `CategoricalDistribution(["linear", "rbf", "poly", "sigmoid"])` |                                                                          |
| `degree`  | `IntDistribution(2, 5)`                                         | Conditional: only sampled when `kernel == "poly"`.                       |

#### Methods and Properties

- `search_space`: Return the full search space (including the conditional `degree`).
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization direction, which is `[StudyDirection.MINIMIZE]`.
  - Returns: `list[optuna.study.StudyDirection]`
- `available_dataset_names`: The names of the available datasets.
  - Returns: `list[str]`
- `__call__(trial: optuna.Trial)`: Sample a configuration from the trial and return its cross-validated error. `degree` is suggested only for the polynomial kernel.
  - Returns: `float`
- `evaluate(params: dict[str, int | float | str])`: Return the cross-validated error for a dictionary of parameters. `degree` is optional and defaults to 3.
  - Returns: `float`
- `reseed(seed: int | None = None)`: Change the seed that controls how the folds are shuffled.

Features are standardized with `StandardScaler` inside the cross-validation pipeline, so no statistics leak from the validation folds.

## Installation

```shell
$ pip install scikit-learn
```

## Example

```python
import optuna
import optunahub


svm_hpo = optunahub.load_module("benchmarks/svm_hpo")
problem = svm_hpo.Problem(dataset_name="wine", seed=42)

study = optuna.create_study(directions=problem.directions)
study.optimize(problem, n_trials=30)
print(study.best_params, study.best_value)
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/benchmarks/svm_hpo/example.py) for a comparison of `TPESampler` and `RandomSampler` over several seeds.
On the `wine` dataset with 30 trials and 5 seeds, it produced the following results:

| Sampler         | Mean best error | Mean median error after the 10 warm-up trials |
| --------------- | --------------- | --------------------------------------------- |
| `TPESampler`    | 0.0170          | 0.0618                                        |
| `RandomSampler` | 0.0191          | 0.1492                                        |

TPE finds slightly better configurations, and after its warm-up phase its typical trial is much better than a random one, because it concentrates on the promising region of the search space.

## Others

### Running the Tests

From the root of the repository:

```shell
$ pip install pytest scikit-learn
$ pytest package/benchmarks/svm_hpo/tests
```
