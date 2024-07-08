from __future__ import annotations

import numpy as np
import optuna
import optunahub
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


module = optunahub.load_module("samplers/dehb")
DEHBSampler = module.DEHBSampler
DEHBPruner = module.DEHBPruner

module = optunahub.load_module("visualization/plot_step_distribution")
plot_step_distribution = module.plot_step_distribution

X, y = load_iris(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
classes = np.unique(y)
n_train_iter = 100


def objective(trial: optuna.Trial) -> float:
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    clf = SGDClassifier(alpha=alpha)

    for step in range(n_train_iter):
        clf.partial_fit(X_train, y_train, classes=classes)

        intermediate_value = -clf.score(X_valid, y_valid)
        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return -clf.score(X_valid, y_valid)


if __name__ == "__main__":
    sampler = DEHBSampler()
    pruner = DEHBPruner(min_resource=1, max_resource=n_train_iter, reduction_factor=3)
    study = optuna.create_study(sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=1000)
    print(study.best_params)
    print(study.best_value)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"history-{sampler.__class__.__name__.lower()}.png")
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(f"slice-{sampler.__class__.__name__.lower()}.png")
    fig = plot_step_distribution(study)
    fig.write_image(f"step-{sampler.__class__.__name__.lower()}.png")
