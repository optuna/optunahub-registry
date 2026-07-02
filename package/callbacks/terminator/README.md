---
author: Alnusjaponica, HideakiImamura, Sip4818, c-bata, contramundum53, cross32768, eukaryo, g-votte, gen740, hvy, jot-s-bindra, kAIto47802, nabenabe0928, not522, nzw0301, sawa3030, smygw72, torotoki, toshihikoyanase, virendrapatil24, and y0z
title: Terminator Callback
description: A callback to automatically stop the optimization when further improvement is unlikely.
tags: [callback, terminator, early-stopping, built-in]
optuna_versions: [4.0.0]
license: MIT License
---

## Abstract

This callback implements an automatic stopping mechanism for Optuna studies, aiming to avoid unnecessary computation. The optimization is terminated when the statistical error of the objective function (e.g., cross-validation error) exceeds the room left for optimization (i.e., the estimated potential for improvement).

The mechanism is described in the following papers:

- `A. Makarova et al. Automatic termination for hyperparameter optimization. <https://proceedings.mlr.press/v188/makarova22a.html>`\_\_
- `H. Ishibashi et al. A stopping criterion for Bayesian optimization by the gap of expected minimum simple regrets. <https://proceedings.mlr.press/v206/ishibashi23a.html>`\_\_

## APIs

### Callback

- `TerminatorCallback(terminator: BaseTerminator | None = None)`
  - A callback that wraps a `Terminator` so that it can be passed to `study.optimize` via the `callbacks` argument. When the terminator judges that the study should be stopped, `study.stop()` is called.
  - `terminator`: A terminator object that determines whether to terminate the optimization. Defaults to a `Terminator` object with default `improvement_evaluator` and `error_evaluator`.

### Terminator

- `Terminator(improvement_evaluator: BaseImprovementEvaluator | None = None, error_evaluator: BaseErrorEvaluator | None = None, min_n_trials: int = 20)`
  - `improvement_evaluator`: An evaluator for the room left for optimization. Defaults to `RegretBoundEvaluator`.
  - `error_evaluator`: An evaluator for the statistical error, e.g. cross-validation error. Defaults to `CrossValidationErrorEvaluator` (or `StaticErrorEvaluator(constant=0)` when `improvement_evaluator` is a `BestValueStagnationEvaluator`).
  - `min_n_trials`: The minimum number of trials before termination is considered. Must be a positive integer. Defaults to `20`.
  - `Terminator.should_terminate(study: Study) -> bool`: Judges whether the study should be terminated based on the reported values.

### Improvement evaluators

- `RegretBoundEvaluator(top_trials_ratio: float = 0.5, min_n_trials: int = 20, seed: int | None = None)`
  - Estimates an upper bound on the regret of the current best solution under a Gaussian process model assumption.
- `BestValueStagnationEvaluator(max_stagnation_trials: int = 30)`
  - Evaluates the number of remaining trials before the best value has stagnated for `max_stagnation_trials` trials.
- `EMMREvaluator(deterministic_objective: bool = False, delta: float = 0.1, min_n_trials: int = 2, seed: int | None = None)`
  - Evaluates the Expected Minimum Model Regret (EMMR), an upper bound of the expected minimum simple regret. Intended to be paired with `MedianErrorEvaluator`.

### Error evaluators

- `CrossValidationErrorEvaluator()`
  - Uses the scaled variance of the cross-validation scores of the best trial as the statistical error. Requires `report_cross_validation_scores` to be called inside the objective function.
- `StaticErrorEvaluator(constant: float)`
  - Always returns a constant value as the error estimate.
- `MedianErrorEvaluator(paired_improvement_evaluator: BaseImprovementEvaluator, warm_up_trials: int = 10, n_initial_trials: int = 20, threshold_ratio: float = 0.01)`
  - Returns a threshold computed as a ratio to the median of the initial improvement values. Intended to be paired with `EMMREvaluator`.
- `report_cross_validation_scores(trial: Trial, scores: list[float]) -> None`
  - Reports the cross-validation scores of a trial. Must be called inside the objective function when using `CrossValidationErrorEvaluator`. The length of `scores` must be greater than one.

### Base classes

- `BaseTerminator`, `BaseImprovementEvaluator`, and `BaseErrorEvaluator` are provided for implementing custom terminators and evaluators.

## Example

### Add the Terminator callback to Optuna optimization

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import optuna
import optunahub


module = optunahub.load_module("callbacks/terminator")
TerminatorCallback = module.TerminatorCallback
report_cross_validation_scores = module.report_cross_validation_scores


def objective(trial):
    X, y = load_wine(return_X_y=True)

    clf = RandomForestClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 32),
        min_samples_split=trial.suggest_float("min_samples_split", 0, 1),
        criterion=trial.suggest_categorical("criterion", ("gini", "entropy")),
    )

    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
    report_cross_validation_scores(trial, scores)
    return scores.mean()


study = optuna.create_study(direction="maximize")
terminator = TerminatorCallback()
study.optimize(objective, n_trials=50, callbacks=[terminator])
```

### Use the Terminator directly within an ask-and-tell loop

```python
import logging

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import optuna
import optunahub


module = optunahub.load_module("callbacks/terminator")
Terminator = module.Terminator
report_cross_validation_scores = module.report_cross_validation_scores


study = optuna.create_study(direction="maximize")
terminator = Terminator()
min_n_trials = 20

while True:
    trial = study.ask()

    X, y = load_wine(return_X_y=True)

    clf = RandomForestClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 32),
        min_samples_split=trial.suggest_float("min_samples_split", 0, 1),
        criterion=trial.suggest_categorical("criterion", ("gini", "entropy")),
    )

    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
    report_cross_validation_scores(trial, scores)

    value = scores.mean()
    logging.info(f"Trial #{trial.number} finished with value {value}.")
    study.tell(trial, value)

    if trial.number > min_n_trials and terminator.should_terminate(study):
        logging.info("Terminated by Optuna Terminator!")
        break
```

### Use EMMR-based termination

```python
import optuna
import optunahub


module = optunahub.load_module("callbacks/terminator")
Terminator = module.Terminator
EMMREvaluator = module.EMMREvaluator
MedianErrorEvaluator = module.MedianErrorEvaluator


sampler = optuna.samplers.TPESampler(seed=0)
study = optuna.create_study(sampler=sampler, direction="minimize")

emmr_improvement_evaluator = EMMREvaluator()
median_error_evaluator = MedianErrorEvaluator(emmr_improvement_evaluator)
terminator = Terminator(
    improvement_evaluator=emmr_improvement_evaluator,
    error_evaluator=median_error_evaluator,
)

for i in range(1000):
    trial = study.ask()

    ys = [trial.suggest_float(f"x{i}", -10.0, 10.0) for i in range(5)]
    value = sum(ys[i] ** 2 for i in range(5))

    study.tell(trial, value)

    if terminator.should_terminate(study):
        # Terminated by Optuna Terminator!
        break
```

## Others

This package is ported from the [`optuna.terminator`](https://optuna.readthedocs.io/en/stable/reference/terminator.html) module. Please refer to the Optuna documentation for further details on the terminator mechanism.
