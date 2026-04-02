"""Example comparing GP-based samplers with different acquisition functions."""

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return -(x**2 + y**2)  # Maximize (minimum at origin)


if __name__ == "__main__":
    mod = optunahub.load_module("samplers/gp_acqf_samplers")

    # GPPISampler: Probability of Improvement
    study_pi = optuna.create_study(direction="maximize")
    study_pi.optimize(objective, n_trials=50, sampler=mod.GPPISampler(seed=42))
    print(f"PI  best value: {study_pi.best_value:.4f}, params: {study_pi.best_params}")

    # GPUCBSampler: Upper Confidence Bound
    study_ucb = optuna.create_study(direction="maximize")
    study_ucb.optimize(objective, n_trials=50, sampler=mod.GPUCBSampler(beta=2.0, seed=42))
    print(f"UCB best value: {study_ucb.best_value:.4f}, params: {study_ucb.best_params}")

    # GPTSSampler: Thompson Sampling
    study_ts = optuna.create_study(direction="maximize")
    study_ts.optimize(objective, n_trials=50, sampler=mod.GPTSSampler(seed=42))
    print(f"TS  best value: {study_ts.best_value:.4f}, params: {study_ts.best_params}")

    # GPEISampler: Alias for GPSampler (Expected Improvement, baseline)
    study_ei = optuna.create_study(direction="maximize")
    study_ei.optimize(objective, n_trials=50, sampler=mod.GPEISampler(seed=42))
    print(f"EI  best value: {study_ei.best_value:.4f}, params: {study_ei.best_params}")
