from __future__ import annotations

import optuna
import optunahub


optuna.logging.set_verbosity(optuna.logging.WARNING)


def hartmann6(trial: optuna.Trial) -> float:
    """Hartmann-6 benchmark. Global minimum: -3.32237 at ~(0.20, 0.15, 0.48, 0.27, 0.31, 0.66)."""
    from botorch.test_functions import Hartmann
    import torch

    f = Hartmann(dim=6)
    x = torch.tensor([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(6)])
    return f(x.unsqueeze(0)).item()


if __name__ == "__main__":
    OrthoBoSampler = optunahub.load_module("samplers/orthobo").OrthoBoSampler

    sampler = OrthoBoSampler(n_startup_trials=10, mc_budget=64)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(hartmann6, n_trials=30)

    print(f"Best value : {study.best_value:.5f}  (global min: -3.32237)")
    print(f"Best params: {study.best_params}")
