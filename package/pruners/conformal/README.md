---
author: Adam Noonan
title: Conformal Pruner
description: Prunes trials with a conformal bound on how often it prunes a trial that would have finished well. Calibrates on the first trials, then freezes.
tags: [pruner, conformal, early stopping, risk control]
optuna_versions: [4.6.0]
license: MIT License
---

## Abstract

Every built-in pruner is a heuristic, and none states how often it prunes a trial that would have
ended well. This pruner does. You pick `alpha`, and the expected fraction of trials that were
*good and pruned* stays at or below `alpha`, under one condition: the trials it calibrates on and
the trials it judges must come from the same random process.

It runs the first `n_calibration` completed trials unpruned, calibrates a threshold on their
learning curves by conformal risk control, then freezes and prunes. The construction was measured
on 400 real LoRA fine-tuning runs under a pre-registered protocol; the curves and the reproduction
script are in the [surestop](https://github.com/ACNoonan/surestop) repository.

## Class or Function Names

- `ConformalPruner(n_calibration=60, alpha=0.05, *, good_threshold=None, good_quantile=0.20, min_peek=0, hold_until_resolved=None, predictor=last_value)`
  - `n_calibration`: completed trials that run unpruned and calibrate the rule, by trial number.
  - `alpha`: bound on the expected fraction of trials that are good and pruned.
  - `good_threshold`: a trial is good if its final reported value is at or below this (at or above, for a maximize study). Overrides `good_quantile`.
  - `good_quantile`: when no threshold is given, the good threshold is this quantile of the calibration finals. The bound is then measured rather than proven (0.88 to 0.97 times `alpha` in simulation).
  - `min_peek`: index of the earliest step at which a prune may fire.
  - `hold_until_resolved`: if set (for example `0.90`), also hold prunes until the calibration trials' early ranking agrees with their final ranking at this Spearman correlation.
  - `predictor`: a causal forecaster of the final value from the curve so far. Default: the latest reported value.
- `ConformalPruner.from_curves(curves, steps, *, maximize=False, **kw)`: a pruner calibrated offline on a previous sweep's completed curves, so it prunes from the first trial.
- `KillRule`: the underlying numpy rule, usable without Optuna.

Every calibration trial must report on the same step grid. The final outcome is the last reported intermediate value rather than `trial.value`.

## Installation

```shell
$ pip install numpy scipy
```

## Example

```python
import optuna
import optunahub

module = optunahub.load_module(package="pruners/conformal")
pruner = module.ConformalPruner(n_calibration=60, alpha=0.05)
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)
study.optimize(objective, n_trials=300)
```

See [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/pruners/conformal/example.py) for a runnable example.

## Others

### What the bound does not cover

- **It needs exchangeable trials.** `RandomSampler`, `QMCSampler` and a shuffled `GridSampler` give that. `TPESampler`, `GPSampler` and `CmaEsSampler` adapt to earlier trials, so later trials are not exchangeable with the calibration trials. The pruner warns once and the bound should not be relied on.
- **It bounds the joint rate rather than the share of good trials pruned.** When good trials are rare, a large share of them can still be pruned: about a third in the measurement, mostly borderline ones. Lower `alpha` if that matters.
- **`alpha = 0.05` needs at least 19 calibration trials.** The default of 60 is a floor for a useful threshold.
- **The rule never recalibrates.** Later trials are survivors of its own prunes, and fitting on them would bias it.

### Measurement

400 LoRA fine-tuning runs of Qwen2.5-0.5B, 50 evaluations each, two pre-registered splits at `alpha = 0.05`. The shipped rule (latest value as predictor) recorded 18 false prunes of 200 on the random split and 7 of 203 on the past-to-future split, with a mean joint rate of 5.0% over 2000 resplits. A rule that pruned everything at once failed the same gate on both splits. Details, curves and the reproduction script: [surestop](https://github.com/ACNoonan/surestop).

### Reference

The mechanism, a conformal threshold on the maximum of a running score calibrated on one class, follows Xie et al., *Statistical Early Stopping for Reasoning Models* (arXiv:2602.13935), who apply it to LLM reasoning traces. This package applies it to training runs.
