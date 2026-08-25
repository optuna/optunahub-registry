---
author: Ahmed Eldeeb
title: Distributionally Robust Level-Set Estimation Sampler
description: Active learning that identifies the design values whose worst-case probability of exceeding a threshold is above a given level, rather than optimising an objective.
tags: [sampler, active learning, level set estimation, Gaussian process, distributionally robust, robust optimization]
optuna_versions: [4.9.0]
license: MIT License
---

## Class or Function Names

- `DRLevelSetSampler`

## Installation

```shell
pip install numpy scipy
```

## Overview

Some black-box functions depend on two kinds of variable: a design variable `x` you control,
and an environmental variable `w` you do not, which varies randomly in use. Manufacturing
tolerances, ambient conditions and market states are all of this kind. The useful question is
often not "which `x` is best" but "which `x` are *reliable*" — for which does `f(x, w)` stay
above a threshold often enough, whatever `w` does.

Writing `P_w(f(x, w) > h)` for that probability, this package identifies

```
H = { x : F(x) > alpha },    F(x) = inf over the ambiguity set of  sum_w 1[f(x,w) > h] p(w)
```

The infimum is what makes it *distributionally robust*: the reference distribution of `w` is
usually not known exactly, so `F` takes the worst case over every distribution within an L1
ball of a user-supplied reference. `F` is the DRPTR measure, and `H` is the reliable set.

The sampler chooses which `(x, w)` pair to evaluate next so as to settle the membership of as
many design points as possible per evaluation. It maintains credible bounds on `F` and
classifies each `x` as reliable, unreliable, or still undecided.

**The result of a study is not `study.best_trial`.** Nothing is being optimised, so the study
direction is ignored and the deliverable is the classification, read back with
`sampler.classify(study)` or `sampler.reliable_set(study)`.

## Example

```python
import numpy as np
import optuna
import optunahub

x_points = np.linspace(-2.0, 2.0, 24)
w_points = np.linspace(-1.0, 1.0, 10)


def objective(trial: optuna.Trial) -> float:
    x = x_points[trial.suggest_int("x_index", 0, len(x_points) - 1)]
    w = w_points[trial.suggest_int("w_index", 0, len(w_points) - 1)]
    return float(np.sin(2.0 * x) + 0.5 * np.cos(3.0 * w) - 0.3 * (x * w) ** 2)


sampler = optunahub.load_module("samplers/drlse").DRLevelSetSampler(
    x_points, w_points, h=0.0, alpha=0.4, eps=0.3, seed=0
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=40)

reliable, unreliable, undecided = sampler.classify(study)
print(x_points[reliable])
```

See `example.py`.

## API Reference

### `DRLevelSetSampler`

| Argument           | Type          | Default  | Description                                                                       |
| ------------------ | ------------- | -------- | --------------------------------------------------------------------------------- |
| `x_points`         | `np.ndarray`  | required | Candidate design values.                                                          |
| `w_points`         | `np.ndarray`  | required | Environmental values.                                                             |
| `h`                | `float`       | required | Threshold of the robustness event `f(x, w) > h`.                                  |
| `alpha`            | `float`       | required | Level defining the reliable set.                                                  |
| `eps`              | `float`       | required | Radius of the L1 ambiguity set around `p_ref`.                                    |
| `p_ref`            | `np.ndarray`  | uniform  | Reference p.m.f. over `w_points`.                                                 |
| `eta`              | `float`       | `0.0`    | Accuracy parameter. See below.                                                    |
| `beta`             | `float`       | `4.0`    | Squared credible-interval width; bounds are `mean ± sqrt(beta)·sd`.               |
| `gamma`            | `float`       | `1e-3`   | Weight of the variance tie-break in the acquisition.                              |
| `sigma_f`          | `float`       | `1.0`    | Prior standard deviation.                                                         |
| `scale`            | `float`       | `1.0`    | Divisor of the squared distance in the kernel, so the *square* of a length scale. |
| `noise`            | `float`       | `1e-4`   | Observation noise variance.                                                       |
| `n_startup_trials` | `int`         | `5`      | Random evaluations before the acquisition is used.                                |
| `seed`             | `int \| None` | `None`   | Random seed.                                                                      |

Methods: `classify(study)` returns boolean masks `(H, L, U)` over `x_points`;
`reliable_set(study)` returns the design values currently classified as reliable.

### The accuracy parameter `eta`

With `eta = 0` the credible bounds are sound, so no design point is ever misclassified — but
a point whose true `F(x)` sits exactly at `alpha` may stay undecided forever. A positive
`eta` relaxes the lower bound to bracket `1[f > h - eta]` instead of `1[f > h]`, which
guarantees termination at the cost of a bounded misclassification loss (Theorem 4.1). Note
that with `eta > 0`, `H` may contain points whose true DRPTR is at or below `alpha`.

## Benchmark

Recall of the reliable set on a 24×10 grid, 20 randomly drawn problems per family, paired on
instance. Precision is 1.000 for every method at every budget, because `eta = 0` makes false
positives impossible, so recall is the whole story.

Three objective families, differing in how well the sampler's Gaussian kernel can represent
them. The first is a draw from that kernel itself, so the model is correct by construction;
the other two are not, and are the ones worth reading.

| queries                                        |                      |        12 |        24 |        36 |        48 |        60 |
| ---------------------------------------------- | -------------------- | --------: | --------: | --------: | --------: | --------: |
| **A.** GP draw from the sampler's own kernel   | this sampler         | **0.713** | **0.957** | **0.982** | **0.991** | **0.991** |
|                                                | uncertainty sampling |     0.348 |     0.813 |     0.947 |     0.964 |     0.968 |
|                                                | random               |     0.264 |     0.698 |     0.824 |     0.908 |     0.946 |
| **B.** exponential-kernel draw, nowhere smooth | this sampler         | **0.528** | **0.713** | **0.747** | **0.763** | **0.763** |
|                                                | uncertainty sampling |     0.225 |     0.557 |     0.629 |     0.711 |     0.705 |
|                                                | random               |     0.271 |     0.455 |     0.507 |     0.552 |     0.555 |
| **C.** analytic function with a kink           | this sampler         | **0.824** | **0.982** | **0.996** | **1.000** | **1.000** |
|                                                | uncertainty sampling |     0.332 |     0.759 |     0.900 |     0.913 |     0.941 |
|                                                | random               |     0.262 |     0.707 |     0.796 |     0.908 |     0.879 |

Paired difference against uncertainty sampling, marked where the 95% bootstrap interval
excludes zero:

| family |      T=12 |      T=24 |      T=36 |      T=48 |   T=60 |
| ------ | --------: | --------: | --------: | --------: | -----: |
| A      | +0.365 \* | +0.143 \* | +0.034 \* | +0.027 \* | +0.023 |
| B      | +0.303 \* | +0.156 \* | +0.118 \* |    +0.052 | +0.058 |
| C      | +0.492 \* | +0.223 \* | +0.096 \* | +0.087 \* | +0.059 |

### Reading the benchmark

The advantage over uncertainty sampling holds whether or not the kernel matches the
objective, and is largest where it matters: at a twelfth of the grid evaluated it is worth
0.30 to 0.49 recall. It stops being significant by 60 queries, a quarter of the grid, because
by then simply covering the space is enough and the acquisition has little left to add.

**The absolute numbers are a different story, and are governed by the kernel rather than the
acquisition.** On family B the sampler plateaus at 0.763 recall and stays there — extra
queries buy nothing, because a Gaussian kernel cannot represent a sample path that is
continuous but nowhere differentiable, so the credible bounds never tighten enough to settle
the remaining design points. If your objective is rough, expect this, and change the kernel
rather than the budget.

Family C, a deterministic function with a kink, is the *easiest* of the three despite also
being misspecified. Sharp structure concentrates the reliable set and gives the acquisition
something to find.

## Others

### Implementation notes

The infimum of eq. (3.1) is solved in closed form rather than as the linear program the paper
describes. An L1 ball intersected with the simplex permits relocating at most `eps / 2` of
probability mass, so the worst case is reached by emptying the costliest atoms onto the
cheapest — `O(|Omega| log |Omega|)`, with no solver dependency. The closed form agrees with an
explicit LP to `2e-16`, which the test suite checks.

All three computational lemmas of section 3.3 are implemented. Lemma 3.1 turns the
expectation over `y*` into an exact finite sum over regions; Lemma 3.2 skips the transport
problem whenever the reference PTR already fails the level test; Lemma 3.3 drops regions of
negligible predictive probability, controlled by `zeta`.

Lemma 3.3 is worth its parameter. On an 800-point grid one acquisition evaluation takes
352 ms with `zeta = 0` and 84 ms at the `1e-6` default, a 4.2x saving, for a maximum error of
`8e-08` and no change to the selected candidate. The saving saturates quickly — `zeta = 1e-2`
is no faster than `1e-9` — because almost all of it comes from discarding regions whose
probability is essentially zero.

### At the paper's scale

Cost grows as `n_x * n_w * |U_t| * n_w`, so it is worth knowing what the paper's 50×50 grid
costs. On four problems with a non-degenerate reliable set, 2500 joint points:

| queries | this sampler | uncertainty sampling | random |
| ------: | -----------: | -------------------: | -----: |
|      25 |    **0.922** |                0.797 |  0.452 |
|      50 |    **1.000** |                0.982 |  0.825 |
|     100 |    **1.000** |                0.982 |  0.982 |
|     200 |    **1.000** |                0.982 |  0.982 |

Full recall arrives at 50 queries, 2% of the grid, and the two baselines plateau at 0.982
without ever closing the gap. Four problems is too few for a confidence interval, so read
this as a demonstration that the method runs at that size and behaves sensibly, not as a
measured effect.

A 200-query run takes about two minutes (median 118 s), against 3-8 s for the baselines. The
acquisition is therefore roughly twenty to forty times the cost of uncertainty sampling per
run, which is the trade being made: it is worth it when an evaluation of `f` is expensive
relative to a second of compute, and not otherwise.

### Reference

Yu Inatsu, Shogo Iwazaki and Ichiro Takeuchi. Active Learning for Distributionally Robust
Level-Set Estimation. In *Proceedings of the 38th International Conference on Machine
Learning*, PMLR 139:4574-4584, 2021.

### Bibtex

```
@InProceedings{pmlr-v139-inatsu21a,
  title = {Active Learning for Distributionally Robust Level-Set Estimation},
  author = {Inatsu, Yu and Iwazaki, Shogo and Takeuchi, Ichiro},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages = {4574--4584},
  year = {2021},
  volume = {139},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```
