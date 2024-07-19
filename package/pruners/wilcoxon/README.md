---
author: Optuna team
title: Wilcoxon Pruner
description: Pruner based on the Wilcoxon signed-rank test.
tags: [pruner, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- WilcoxonPruner

## Example

```python
import optuna
import numpy as np


# We minimize the mean evaluation loss over all the problem instances.
def evaluate(param, instance):
    # A toy loss function for demonstrative purpose.
    return (param - instance) ** 2


problem_instances = np.linspace(-1, 1, 100)


def objective(trial):
    # Sample a parameter.
    param = trial.suggest_float("param", -1, 1)

    # Evaluate performance of the parameter.
    results = []

    # For best results, shuffle the evaluation order in each trial.
    instance_ids = np.random.permutation(len(problem_instances))
    for instance_id in instance_ids:
        loss = evaluate(param, problem_instances[instance_id])
        results.append(loss)

        # Report loss together with the instance id.
        # CAVEAT: You need to pass the same id for the same instance,
        # otherwise WilcoxonPruner cannot correctly pair the losses across trials and
        # the pruning performance will degrade.
        trial.report(loss, instance_id)

        if trial.should_prune():
            # Return the current predicted value instead of raising `TrialPruned`.
            # This is a workaround to tell the Optuna about the evaluation
            # results in pruned trials. (See the note below.)
            return sum(results) / len(results)

    return sum(results) / len(results)


study = optuna.create_study(pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1))
study.optimize(objective, n_trials=100)
```

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html) for more details.
