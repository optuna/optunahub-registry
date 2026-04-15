"""Example: LevyFlightSampler on a 2-D benchmark function.

This script compares LevyFlightSampler against Optuna's default TPE sampler
on a shifted Rosenbrock function — a classic optimisation benchmark with a
narrow curved valley that trips up many hill-climbing approaches.

Run:
    python example.py
"""

import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Objective — shifted Rosenbrock (minimum at x=1.5, y=-0.5 → value=0)
# ---------------------------------------------------------------------------


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    # Rosenbrock: sum 100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2
    # Shifted so minimum is at (1.5, -0.5)
    a, b = x - 1.5, y + 0.5
    return 100.0 * (b - a**2) ** 2 + (1.0 - a) ** 2


N_TRIALS = 100
SEED = 42

# ---------------------------------------------------------------------------
# Run with LevyFlightSampler
# ---------------------------------------------------------------------------
import optunahub  # noqa: E402


mod = optunahub.load_module("samplers/levy_flight_sampler")
levy_sampler = mod.LevyFlightSampler(beta=1.5, step_scale=0.1, seed=SEED)

levy_study = optuna.create_study(sampler=levy_sampler, direction="minimize")
levy_study.optimize(objective, n_trials=N_TRIALS)

print("=== LevyFlightSampler ===")
print(f"  Best value : {levy_study.best_value:.6f}")
print(f"  Best params: {levy_study.best_params}")

# ---------------------------------------------------------------------------
# Run with default TPE for comparison
# ---------------------------------------------------------------------------
tpe_study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=SEED), direction="minimize"
)
tpe_study.optimize(objective, n_trials=N_TRIALS)

print("\n=== TPESampler (default) ===")
print(f"  Best value : {tpe_study.best_value:.6f}")
print(f"  Best params: {tpe_study.best_params}")