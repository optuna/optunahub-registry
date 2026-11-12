"""DisCoMBO on a mixed search space, with an expert intervention."""

from __future__ import annotations

import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.0, 1.0)
    layers = trial.suggest_int("layers", 1, 8)
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])
    # A synthetic validation accuracy, maximal for lr = 1e-2, momentum = 0.9, 3 layers and Adam.
    return float(
        1.0
        - (np.log10(lr) + 2) ** 2 / 10
        - (momentum - 0.9) ** 2
        - 0.05 * abs(layers - 3)
        - 0.2 * (optimizer != "adam")
    )


sampler = optunahub.load_module(package="samplers/discombo").DisCoMBOSampler(seed=0)

# From trial 40 on, an expert believes that Adam with a learning rate around 1e-2 works well.
# Point beliefs are given as parameter values, distributions over log-scaled parameters in
# natural-log space.
sampler.intervene(
    {"optimizer": "adam", "lr": {"dist": "gauss", "parameters": [np.log(1e-2), 0.5]}},
    trial_number=40,
)

study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
for record in sampler.get_disco_scores():
    # A low DisCo score indicates that the beliefs contradict the observations.
    print(
        f"DisCo score of the intervention at trial {record['trial_number']}: "
        f"{record['disco_score']:.3f}"
    )
