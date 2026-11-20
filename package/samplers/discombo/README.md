---
author: Jonas Seng, Bennet Wittelsbach, Kristian Kersting
title: 'DisCoMBO: Expert-in-the-Loop Black-Box Optimization via Distributional Conformance'
description: DisCoMBO samples candidates from a probabilistic circuit conditioned on good objective values, selects them by their distributional conformance score, and lets experts inject beliefs whose conformance with the observations it reports.
tags: [sampler, probabilistic circuits, expert-in-the-loop, user prior, hyperparameter optimization, mixed search space]
optuna_versions: [4.9.0, 5.0.0]
license: MIT License
---

## Abstract

DisCoMBO is the sampler of the NeurIPS 2026 paper [DisCoMBO: Steering Expert-in-the-Loop Black-Box Optimization via Distributional Conformance](https://arxiv.org/abs/2609.36472).
Instead of a Gaussian process or an ensemble, it uses a probabilistic circuit (a sum-product network) as a generative surrogate of the hyperparameters and the objective value.
Candidates are sampled from the circuit conditioned on good objective values and selected by their **Dis**tributional **Co**nformance score (DisCo): a bounded, [0, 1]-normalized measure of how much a candidate surprises the circuit.
DisCo recovers properties comparable to kernel-based uncertainty known from Gaussian processes while maintaining linear-time inference, and DisCoMBO cycles between exploring surprising and exploiting conforming candidates.
The paper proves that DisCoMBO is a zero-regret algorithm.

Expert knowledge is injected directly as interventions, i.e. beliefs about good parameter values, via conditional sampling from the circuit.
For each intervention, DisCoMBO reports its conformance with the observations, so that misleading beliefs can be detected; the influence of an intervention decays over the following iterations.

The sampler handles mixed search spaces: continuous, integer (also with log scale or steps) and categorical parameters.

## APIs

- `DisCoMBOSampler(search_space: dict[str, BaseDistribution] | None = None, *, seed: int | None = None, n_startup_trials: int = 20, batch_size: int = 10, **optimizer_kwargs)`
  - `search_space`: A dictionary of [Optuna distributions](https://optuna.readthedocs.io/en/stable/reference/distributions.html). If not provided, the search space is inferred from the completed trials; parameters outside of it (e.g. conditional parameters) are sampled at random.
  - `seed`: Seed for the random number generators.
  - `n_startup_trials`: Number of random trials before DisCoMBO takes over.
  - `batch_size`: Number of configurations DisCoMBO proposes per iteration, i.e. per learned circuit. The proposals are assigned to the following trials, also to trials running in parallel.
  - `optimizer_kwargs`: Further arguments of [`discombo.DisCoOptimizer`](https://github.com/bewit/DisCoMBO/blob/main/discombo/optimizers/disco_optimizer.py), e.g. `stratified_pool_size` (number of candidates sampled per iteration) or `stratified_cycle_length` (iterations from exploration to exploitation).
- `DisCoMBOSampler.intervene(beliefs: dict[str, Any], trial_number: int)`: Injects expert beliefs into the batch DisCoMBO proposes at or after trial `trial_number`. `beliefs` maps parameter names to
  - a value (a point intervention), given like the values of the parameter, e.g. `"adam"` or `1e-3`, or
  - a distribution `{"dist": ..., "parameters": [...]}`: `"gauss"` (mean, std) or `"uniform"` (low, high) for continuous parameters, in natural-log space for parameters with `log=True`; `"cat"` (one weight per choice) or `"int_uniform"` (low and exclusive high choice index) for categorical parameters and integer parameters with fewer than 10 values.
- `DisCoMBOSampler.get_disco_scores() -> list[dict[str, Any]]`: The distributional conformance (DisCo) score of each applied intervention: the trial number, the beliefs (in DisCoMBO's representation) and the mean DisCo score `disco_score` of the beliefs w.r.t. the circuit conditioned on the best observed value. Low scores indicate that the beliefs contradict the observations.

DisCoMBOSampler supports single-objective optimization.
Failed and pruned trials are ignored.

## Installation

DisCoMBO is available on [PyPI](https://pypi.org/project/discombo/). Install the dependencies of this sampler with

```shell
pip install -r https://hub.optuna.org/samplers/discombo/requirements.txt
```

DisCoMBO runs on the CPU. On Linux, `pip` installs the CUDA build of PyTorch by default, which takes several GB; to get the much smaller CPU build, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    n = trial.suggest_int("n", 1, 8)
    kind = trial.suggest_categorical("kind", ["a", "b", "c"])
    return x**2 + abs(n - 3) + (kind != "b")


sampler = optunahub.load_module(package="samplers/discombo").DisCoMBOSampler(seed=0)
sampler.intervene({"kind": "b"}, trial_number=30)  # an expert believes that "b" is a good choice
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
print(study.best_params, sampler.get_disco_scores()[0]["disco_score"])
```

See [example.py](example.py) for an example with a distribution as belief.

## Others

DisCoMBO itself is developed at [github.com/bewit/DisCoMBO](https://github.com/bewit/DisCoMBO) and licensed under the Apache License 2.0; this sampler wraps it under the MIT License.

### Reference

Jonas Seng, Bennet Wittelsbach, and Kristian Kersting. DisCoMBO: Steering Expert-in-the-Loop Black-Box Optimization via Distributional Conformance. In Advances in Neural Information Processing Systems (NeurIPS), 2026.

### Bibtex

```bibtex
@inproceedings{seng2026discombo,
  title         = {{DisCoMBO}: Steering Expert-in-the-Loop Black-Box Optimization via Distributional Conformance},
  author        = {Seng, Jonas and Wittelsbach, Bennet and Kersting, Kristian},
  booktitle     = {Advances in Neural Information Processing Systems},
  volume        = {40},
  year          = {2026},
  eprint        = {2609.36472},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.36472}
}
```
