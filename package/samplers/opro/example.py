"""Run OPROSampler with a local Ollama server.

Before running this example, install Ollama (https://ollama.com), pull a model, and install the
dependencies:

    ollama pull qwen2.5:7b
    pip install optuna optunahub openai

To use the OpenAI API instead, set ``OPENAI_API_KEY`` and use, for example,
``OPROSampler(model="gpt-4o-mini")`` without ``base_url``.
"""

from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    penalty = {"linear": 2.0, "rbf": 0.0, "poly": 1.0}[kernel]
    return (x - 1.5) ** 2 + (y + 2) ** 2 + penalty


OPROSampler = optunahub.load_module("samplers/opro").OPROSampler
sampler = OPROSampler(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama accepts any non-empty key.
    task_description="A toy function of a float x, an integer y, and a categorical kernel.",
    seed=42,
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=30)
print(study.best_trial.params, study.best_value)
