---
author: Bryon Tjanaka
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
This section will help attract potential users to your package.

**Example**

This package provides a sampler based on Gaussian process-based Bayesian optimization. The sampler is highly sample-efficient, so it is suitable for computationally expensive optimization problems with a limited evaluation budget, such as hyperparameter optimization of machine learning algorithms.

## Class or Function Names

Please fill in the class/function names which you implement here.

**Example**

- GPSampler

## Installation

If you have additional dependencies, please fill in the installation guide here.
If no additional dependencies is required, **this section can be removed**.

**Example**

```shell
$ pip install scipy torch
```

If your package has `requirements.txt`, it will be automatically uploaded to the OptunaHub, and the package dependencies will be available to install as follows.

```shell
 pip install -r https://hub.optuna.org/{category}/{your_package_name}/requirements.txt
```

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
