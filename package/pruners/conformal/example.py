from __future__ import annotations

import numpy as np
import optuna
import optunahub


T = 30


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    rng = np.random.default_rng(trial.number)
    final = 2.15 + 0.05 * abs(np.log10(lr) + 4) + rng.normal(0, 0.005)
    for step in range(T):
        value = final + 0.2 * np.exp(-step / 4) + rng.normal(0, 0.004) * np.exp(-step / 8)
        if step == T - 1:
            value = final
        trial.report(float(value), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return final


module = optunahub.load_module(package="pruners/conformal")

# The first n_calibration completed trials run unpruned and calibrate the rule. After that the
# rule is frozen and prunes with a bound of alpha on the fraction of trials that were good and
# got killed. The bound needs a random or quasi-random sampler.
pruner = module.ConformalPruner(n_calibration=40, alpha=0.05)
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0), pruner=pruner)
study.optimize(objective, n_trials=160)

states = [t.state for t in study.trials]
print("pruned:", states.count(optuna.trial.TrialState.PRUNED), "of", len(states))
print(
    "calibrated on trials", pruner.calibration_trials_[:3], "...", pruner.calibration_trials_[-1]
)
print(
    "tau",
    pruner.rule_.tau_,
    "lambda",
    pruner.rule_.lambda_,
    "first peek",
    pruner.rule_.first_peek_,
)
