from collections.abc import Callable

import optuna
import optunahub


BisectSampler = optunahub.load_module("samplers/bisect").BisectSampler


def objective(trial: optuna.Trial, score_func: Callable[[optuna.Trial], float]) -> float:
    x = trial.suggest_float("x", -1, 1)
    # For each param, e.g., `ZZZ`, please set `ZZZ_is_too_high`.
    trial.set_user_attr("x_is_too_high", x > 0.5)
    y = trial.suggest_float("y", -1, 1, step=0.2)
    trial.set_user_attr("y_is_too_high", y > 0.2)
    # Please use `BisectSampler.score_func`.
    return BisectSampler.score_func(trial)


sampler = BisectSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(lambda t: objective(t, BisectSampler.score_func), n_trials=20)
