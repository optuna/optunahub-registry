---
author: SayokoAkiike
title: OPRO Sampler (Large Language Models as Optimizers)
description: A sampler that uses a large language model as the optimizer. It shows the model the best trials so far and asks it for better parameters, following OPRO. It works with any OpenAI-compatible API, including local models served by Ollama.
tags: [sampler, LLM, large language model, OPRO, black-box optimization]
optuna_versions: [5.1.0]
license: MIT License
---

## Abstract

OPRO (Optimization by PROmpting), proposed by [Yang et al.](https://arxiv.org/abs/2309.03409), uses a large language model (LLM) as the optimizer itself. In each step, the LLM receives a meta-prompt that describes the problem and lists past solutions with their scores, sorted so that the best solution comes last. The LLM is then asked for a new solution that is better than all of them. The new solutions are evaluated and added to the list, and the next step starts.

This package brings that loop to Optuna as `OPROSampler`:

- Unlike LLM-assisted Bayesian optimization such as [LLAMBO](https://hub.optuna.org/samplers/llambo/), OPRO has no surrogate model or acquisition function. The whole optimizer is one prompt per candidate, which makes it simple and easy to inspect.
- It works with any OpenAI-compatible API through `base_url`, for example the OpenAI API or a local model served by [Ollama](https://ollama.com), which needs no API key or payment.
- Float, integer (including `step` and `log`), and categorical parameters are handled by the LLM.

This is an independent implementation based on the paper. It does not contain code from the [original repository](https://github.com/google-deepmind/opro).

## APIs

- `OPROSampler(*, model: str, base_url: str | None = None, api_key: str | None = None, task_description: str | None = None, n_startup_trials: int = 5, n_candidates_per_step: int = 8, max_trajectory_size: int = 20, temperature: float = 1.0, search_space: dict[str, BaseDistribution] | None = None, seed: int | None = None, client: Any | None = None)`
  - `model`: The name of the LLM, for example `"gpt-4o-mini"` for OpenAI or `"qwen2.5:7b"` for Ollama.
  - `base_url`: The base URL of an OpenAI-compatible API, for example `"http://localhost:11434/v1"` for Ollama. If `None`, the OpenAI API is used.
  - `api_key`: The API key. If `None`, the `OPENAI_API_KEY` environment variable is used. Ollama accepts any non-empty string.
  - `task_description`: An optional description of the objective, added to the prompt.
  - `n_startup_trials`: The number of trials sampled at random before the LLM is used.
  - `n_candidates_per_step`: The number of candidates requested from the LLM in each step. They are queued and used by the following trials.
  - `max_trajectory_size`: The maximum number of past trials shown in the prompt. The best trials are shown.
  - `temperature`: The sampling temperature of the LLM.
  - `search_space`: The search space. If `None`, it is inferred from the completed trials.
  - `seed`: The seed for the random sampling of the startup trials and of the fallback.
  - `client`: An OpenAI-compatible client object with `chat.completions.create`. If given, `base_url` and `api_key` are ignored.

### How it works

1. The first `n_startup_trials` trials are sampled at random.
1. In each step, the sampler builds a prompt with the parameter definitions and the best `max_trajectory_size` completed trials, sorted from the worst to the best. It asks the LLM `n_candidates_per_step` times for new parameters, given as a JSON object.
1. Each answer is checked against the search space. Numbers are clipped to the range and rounded to the step, and answers with missing parameters, unknown categories, or no JSON object are discarded. Duplicates of past trials are discarded.
1. The valid candidates are queued and used by the following trials. When the queue is empty, the next step starts with the latest trials. If no candidate is valid, the trial is sampled at random.

### Differences from the paper

- The paper asks for a solution in the form `[w, b]`. This implementation asks for a JSON object, so that parameters can be referred to by name.
- The paper evaluates all candidates of a step together. Optuna runs one trial at a time, so the candidates are queued instead.
- Only single-objective optimization is supported.

## Installation

```shell
pip install openai
```

To use a local model, install [Ollama](https://ollama.com) and pull a model:

```shell
ollama pull qwen2.5:7b
```

## Example

```python
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
    api_key="ollama",
    seed=42,
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=30)
print(study.best_trial.params, study.best_value)
```

## Others

### Reference

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large Language Models as Optimizers. arXiv:2309.03409.
