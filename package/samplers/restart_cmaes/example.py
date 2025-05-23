import optuna
import optunahub


module = optunahub.load_module("samplers/restart_cmaes")
RestartCmaEsSampler = module.RestartCmaEsSampler


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


sampler = RestartCmaEsSampler()  # CMA-ES without restart (default)
# sampler = RestartCmaEsSampler(restart_strategy="ipop")  # IPOP-CMA-ES
# sampler = RestartCmaEsSampler(restart_strategy="bipop")  # BIPOP-CMA-ES
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=20)
