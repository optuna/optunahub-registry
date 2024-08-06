---
author: Please fill in the author name here. (e.g., John Smith)
title: Please fill in the title of the feature here. (e.g., Gaussian-Process Expected Improvement Sampler)
description: Please fill in the description of the feature here. (e.g., This sampler searches for each trial based on expected improvement using Gaussian process.)
tags: [Please fill in the list of tags here. (e.g., sampler, visualization, pruner)]
optuna_versions: ['Please fill in the list of versions of Optuna in which you have confirmed the feature works, e.g., 3.6.1.']
license: MIT License
---

<!--
This is an example of the frontmatters.
All columns must be string.
You can omit quotes when value types are not ambiguous.
For tags, a package placed in
- package/samplers/ must include the tag "sampler"
- package/visualilzation/ must include the tag "visualization"
- package/pruners/ must include the tag "pruner"
respectively.

---
author: Optuna team
title: My Sampler
description: A description for My Sampler.
tags: [sampler, 2nd tag for My Sampler, 3rd tag for My Sampler]
optuna_versions: [3.6.1]
license: "MIT License"
---
-->

Please read the [tutorial guide](https://optuna.github.io/optunahub-registry/recipes/001_first.html) to register your feature in OptunaHub.
You can find more detailed explanation of the following contents in the tutorial.
Looking at [other packages' implementations](https://github.com/optuna/optunahub-registry/tree/main/package) will also help you.

## Abstract

You can provide an abstract for your package here.
This section is helpful to advertise your package to potential users.

**Example**

This package provides a sampler based on Gaussian process-based Bayesian optimization. The sampler is highly sample-efficient, so it is suitable for computationally expensive optimization problems with a limited evaluation budget, such as hyperparameter optimization of machine learning algorithms.

## Installation

If you have additional dependencies, please fill in the installation guide here.
If no additional dependencies is required, **this section can be removed**.

**Example**

```shell
$ pip install scipy torch
```

## APIs

Please provide documentation for the classes/functions in your package.
We highly recommend you provide enough information for users to use your package.

**Example**

### GPSampler(\*, seed=None, independent_sampler=None, n_startup_trials=10, deterministic_objective=False)

A sampler class of a Gaussian process-based surrogate Bayesian optimization algorithm.

#### Parameters

- `seed (int | None)` – Random seed to initialize internal random number generator. Defaults to None (a seed is picked randomly).
- `independent_sampler (BaseSampler | None)` – Sampler used for initial sampling (for the first n_startup_trials trials) and for conditional parameters. Defaults to None (a random sampler with the same seed is used).
- `n_startup_trials (int)` – Number of initial trials. Defaults to 10.
- `deterministic_objective (bool)` – Whether the objective function is deterministic or not. If True, the sampler will fix the noise variance of the surrogate model to the minimum value (slightly above 0 to ensure numerical stability). Defaults to False.

### load_study(\*, study_name, storage, sampler=None, pruner=None)

#### Parameters

- `study_name (str | None)` – Study’s name. Each study has a unique name as an identifier. If None, checks whether the storage contains a single study, and if so loads that study. study_name is required if there are multiple studies in the storage.
- `storage (str | storages.BaseStorage)` – Database URL such as sqlite:///example.db.
- `sampler` ('samplers.BaseSampler' | None) – A sampler object that implements background algorithm for value suggestion.
- `pruner (pruners.BasePruner | None)` – A pruner object that decides early stopping of unpromising trials.

#### Return type

Study

## Example

Please fill in the code snippet to use the implemented feature here.

**Example**

```python
import optuna
import optunahub


def objective(trial):
  x = trial.suggest_float("x", -5, 5)
  return x**2


sampler = optunahub.load_module(package="samplers/gp").GPSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
```

## Others

Please fill in any other information if you have here by adding child sections (###).
If there is no additional information, **this section can be removed**.

<!--
For example, you can add sections to introduce a corresponding paper.

### Reference
Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.

### Bibtex
```
@inproceedings{optuna_2019,
    title={Optuna: A Next-generation Hyperparameter Optimization Framework},
    author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
    booktitle={Proceedings of the 25th {ACM} {SIGKDD} International Conference on Knowledge Discovery and Data Mining},
    year={2019}
}
```
-->
