---
author: Optuna team
title: CMA-ES Sampler that support IPOP-CMA-ES and BIPOP-CMA-ES
description: A CMA-ES-based sampler that supports advanced restart strategies, IPOP-CMA-ES and BIPOP-CMA-ES.
tags: [sampler, CMA-ES, IPOP-CMA-ES, BIPOP-CMA-ES]
optuna_versions: [4.2.1]
license: MIT License
---

## Abstract

This package offers a CMA-ES-based sampler with support for advanced restart strategies, specifically IPOP-CMA-ES and BIPOP-CMA-ES. Originally implemented in Optuna prior to v4.4.0, this functionality was removed to enhance the maintainability of Optuna's core algorithms.

Please note that this sampler does not support CategoricalDistribution. However, `optuna.distributions.FloatDistribution` with `step`, (`optuna.trial.Trial.suggest_float`) and `optuna.distributions.IntDistribution` (`optuna.trial.Trial.suggest_int`) are supported.

If your search space contains categorical parameters, I recommend you to use `optuna.samplers.TPESampler` instead. Furthermore, there is room for performance improvements in parallel optimization settings. This sampler cannot use some trials for updating the parameters of multivariate normal distribution.

This sampler uses [`cmaes`](https://github.com/CyberAgentAILab/cmaes) as the backend.

## APIs

- `RestartCmaEsSampler(x0: dict[str, Any] | None = None, sigma0: float | None = None, n_startup_trials: int = 1, independent_sampler: BaseSampler | None = None, warn_independent_sampling: bool = True, seed: int | None = None, *, restart_strategy: str | None = None, popsize: int | None = None, inc_popsize: int = 2, store_optimizer_state_in_storage: bool = False,)`
  - `x0`: A dictionary of an initial parameter values for CMA-ES. By default, the mean of `low` and `high` for each distribution is used. Note that `x0` is sampled uniformly within the search space domain for each restart if you specify `restart_strategy` argument.

  - `sigma0`: Initial standard deviation of CMA-ES. By default, `sigma0` is set to `min_range / 6`, where `min_range` denotes the minimum range of the distributions in the search space.

  - `seed`: A random seed for CMA-ES.

  - `n_startup_trials`: The independent sampling is used instead of the CMA-ES algorithm until the given number of trials finish in the same study.

  - `independent_sampler`: A `optuna.samplers.BaseSampler` instance that is used for independent sampling. The parameters not contained in the relative search space are sampled by this sampler. The search space for `optuna.samplers.CmaEsSampler` is determined by `optuna.search_space.intersection_search_space()`. If `None` is specified, `optuna.samplers.RandomSampler` is used as the default.

  - `warn_independent_sampling`: If this is `True`, a warning message is emitted when the value of a parameter is sampled by using an independent sampler. Note that the parameters of the first trial in a study are always sampled via an independent sampler, so no warning messages are emitted in this case.

  - `restart_strategy`: Strategy for restarting CMA-ES optimization when converges to a local minimum. If `None` is given, CMA-ES will not restart (default). If 'ipop' is given, CMA-ES will restart with increasing population size. if 'bipop' is given, CMA-ES will restart with the population size increased or decreased. Please see also `inc_popsize` parameter.

  - `popsize`: A population size of CMA-ES. When `restart_strategy = 'ipop'` or `restart_strategy = 'bipop'` is specified, this is used as the initial population size.

  - `inc_popsize`: Multiplier for increasing population size before each restart. This argument will be used when `restart_strategy = 'ipop'` or `restart_strategy = 'bipop'` is specified.

  - `store_optimizer_state_in_storage`: If `True`, the internal state of the CMA-ES optimizer is saved in the `system_attrs` of each trial, enabling persistent experiments and distributed optimization. If `False`, the state is stored in-memory for faster execution but cannot be shared across processes or recovered after interruptions. The default is `False` for better performance in single-process optimizations.

## Example

```python
import optuna
import optunahub


module = optunahub.load_module("samplers/restart_cmaes")
RestartCmaEsSampler = module.RestartCmaEsSampler


def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


sampler = RestartCmaEsSampler()  # CMA-ES without restart (default)
# sampler = RestartCmaEsSampler(restart_strategy="ipop")  # IPOP-CMA-ES
# sampler = RestartCmaEsSampler(restart_strategy="bipop")  # BIPOP-CMA-ES
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=20)

```

## Installation

```sh
pip install -r https://hub.optuna.org/samplers/restart_cmaes/requirements.txt
```

### Reference

1. Hansen, N. (2016). [The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772).
1. Auger, A., & Hansen, N. (2005). [A restart CMA evolution strategy with increasing population size](https://doi.org/10.1109/CEC.2005.1554902).
1. Hansen, N. (2009). [Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed](https://doi.org/10.1145/1570256.1570333).
1. Ros, R., & Hansen, N. (2008). [A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity](https://doi.org/10.1007/978-3-540-87700-4_30).
1. Nomura, M., Watanabe, S., Akimoto, Y., Ozaki, Y., & Onishi, M. (2021). [Warm Starting CMA-ES for Hyperparameter Optimization](https://doi.org/10.1609/aaai.v35i10.17109).
1. Hamano, R., Saito, S., Nomura, M., & Shirakawa, S. (2022). [CMA-ES with Margin: Lower-Bounding Marginal Probability for Mixed-Integer Black-Box Optimization](https://doi.org/10.1145/3512290.3528827).
1. Nomura, M., Akimoto, Y., & Ono, I. (2023). [CMA-ES with Learning Rate Adaptation: Can CMA-ES with Default Population Size Solve Multimodal and Noisy Problems?](https://doi.org/10.1145/3583131.3590358).
