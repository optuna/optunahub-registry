---
author: Yasunori Morishima
title: 'Meta-Learn TPE: Task Similarity-Based Warm-Starting for TPE'
description: A TPE sampler that accelerates optimization by transferring knowledge from previously completed studies on related tasks.
tags: [sampler, tpe, meta-learning, warm-starting, transfer-learning]
optuna_versions: [v4.1.0]
license: MIT License
---

## Abstract

This package provides a meta-learning extension of the Tree-structured Parzen Estimator (TPE) that leverages previously completed Optuna studies on related tasks to accelerate optimization.

The algorithm is based on the approach described in:

- [Speeding Up Multi-Objective Hyperparameter Optimization by Task Similarity-Based Meta-Learning for the Tree-Structured Parzen Estimator](https://arxiv.org/abs/2212.06751) (IJCAI 2023)

The key idea is to compute task similarity between the target task and source tasks based on the overlap of their promising regions, and then use a weighted mixture of TPE models to guide the search. When source tasks are similar to the target, the sampler converges faster by exploiting shared structure.

## Class or Function Names

- `MetaLearnTPESampler`

### Arguments

| Name               | Type              | Default    | Description                                            |
| ------------------ | ----------------- | ---------- | ------------------------------------------------------ |
| `source_studies`   | `Sequence[Study]` | (required) | Completed Optuna studies on related tasks              |
| `n_startup_trials` | `int`             | `10`       | Number of random trials before meta-learning activates |
| `seed`             | `int \| None`     | `None`     | Random seed for reproducibility                        |
| `n_ei_candidates`  | `int`             | `24`       | Number of EI candidates per task                       |

## Installation

```shell
$ pip install optuna
```

No additional dependencies are required beyond Optuna.

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 10.0)
    y = trial.suggest_float("y", 0.0, 15.0)
    return (x - 2) ** 2 + (y - 3) ** 2


# 1. Run source studies on related tasks.
source_study = optuna.create_study()
source_study.optimize(objective, n_trials=30)

# 2. Create the meta-learning sampler.
sampler = optunahub.load_module(
    package="samplers/meta_learn_tpe"
).MetaLearnTPESampler(
    source_studies=[source_study],
    seed=42,
)

# 3. Optimize a new (related) target task.
target_study = optuna.create_study(sampler=sampler)
target_study.optimize(objective, n_trials=50)
print(target_study.best_params)
```

## How It Works

1. **Build TPE models**: For each source study and the target study, a TPE model is fitted, splitting trials into "below" (promising) and "above" (non-promising) groups.
1. **Compute task similarity**: The overlap between the target's promising region and each source's promising region is measured using Total Variation distance.
1. **Weight tasks**: Source tasks with higher similarity receive larger weights. The target task weight ensures that as optimization progresses, the sampler increasingly relies on the target data.
1. **Weighted acquisition**: Candidates are sampled from all tasks' below distributions, and scored using a weighted mixture of all tasks' TPE likelihoods.

## Bibtex

```bibtex
@inproceedings{watanabe_meta_learn_tpe_2023,
    title={Speeding Up Multi-Objective Hyperparameter Optimization by Task Similarity-Based Meta-Learning for the Tree-Structured Parzen Estimator},
    author={Watanabe, Shuhei and Awad, Noor and Onishi, Masaki and Hutter, Frank},
    booktitle={International Joint Conference on Artificial Intelligence},
    year={2023}
}
```
