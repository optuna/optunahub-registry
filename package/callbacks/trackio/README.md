---
author: ParagEkbote
title: Trackio Callback
description: A callback to track Optuna trials with Trackio.
tags: [callback, trackio, logging, built-in]
optuna_versions: [4.9.0]
license: MIT License
---

## Installation

```shell
pip install trackio
```

## Abstract

This callback enables tracking of Optuna studies in Trackio. By default, the study is tracked as a single experiment run, where all suggested hyperparameters and optimized metrics are logged and visualized as a function of optimizer steps.

Trackio is offline-first and does not require authentication for local experiment tracking. Optionally, tracked experiments can be synchronized to Hugging Face Spaces for remote visualization and sharing.

The callback also supports multi-run mode, where each Optuna trial is tracked as an independent Trackio run. This is useful for sweep-style dashboards, parameter importance analysis, and per-trial experiment inspection.

## APIs

- `TrackioCallback(project: str, metric_name: str | Sequence[str] = "value", as_multirun: bool = False, space_id: str | None = None, dataset_id: str | None = None, private: bool | None = None, resume: str = "allow", sync_on_finish: bool = False, sync_frequency: str = "study", sync_run_in_background: bool = False, trackio_kwargs: dict[str, Any] | None = None)`

  - `project`:
    Name of the Trackio project used for local storage and optional synchronization.

  - `metric_name`:
    Name assigned to the optimized metric. In case of multi-objective optimization, a list of names can be passed. These names will be assigned to objective values in the order returned by the objective function.

    If a single name is provided, it will be broadcast to multiple objectives using numerical suffixes such as `value_0`, `value_1`.

  - `as_multirun`:
    Creates a new Trackio run for each Optuna trial. Useful for generating sweep-style dashboards and trial-level visualizations.

  - `space_id`:
    Optional Hugging Face Space ID (`"username/space-name"`) used for synchronization and remote visualization.

  - `dataset_id`:
    Optional Hugging Face Dataset ID used for exporting experiment metadata and metrics.

  - `private`:
    Whether synchronized Hugging Face artifacts should be private.

  - `resume`:
    Resume policy for Trackio runs. Accepted values are `"allow"`, `"must"`, and `"never"`.

  - `sync_on_finish`:
    Whether to synchronize the project to Hugging Face after study completion.

  - `sync_frequency`:
    Synchronization frequency strategy.

    - `"study"` synchronizes once after the study completes.
    - `"trial"` synchronizes after each completed trial in multirun mode.

  - `sync_run_in_background`:
    Whether synchronization should run asynchronously in a background thread.

  - `trackio_kwargs`:
    Additional keyword arguments passed directly to `trackio.init()`.

- `TrackioCallback.track_in_trackio() -> Callable`

  - Decorator for enabling Trackio logging inside the objective function.

    The decorator initializes and finalizes Trackio runs automatically. Additional metrics logged inside the objective function using `trackio.log()` are associated with the corresponding Optuna trial run.

    Use as:

    ```python
    @trackioc.track_in_trackio()
    ```

- `TrackioCallback.finish() -> None`

  - Explicitly finalizes synchronization and cleanup after `study.optimize()` completes.

## Installation

```shell
pip install trackio
```

## Example

### Add Trackio callback to Optuna optimization

```python
import optuna
import optunahub


module = optunahub.load_module("callbacks/trackio")
TrackioCallback = module.TrackioCallback


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study = optuna.create_study(
    study_name="trackio-demo",
)

trackioc = TrackioCallback(
    project="my-optuna-study",
)

study.optimize(
    objective,
    n_trials=10,
    callbacks=[trackioc],
)

trackioc.finish()
```

### Trackio logging in multirun mode

```python
import optuna
import optunahub
import trackio


module = optunahub.load_module("callbacks/trackio")
TrackioCallback = module.TrackioCallback


trackioc = TrackioCallback(
    project="my-optuna-study",
    as_multirun=True,
)


@trackioc.track_in_trackio()
def objective(trial):
    x = trial.suggest_float("x", -10, 10)

    # Additional logging inside Trackio
    trackio.log(
        {
            "power": 2,
            "base_of_metric": x - 2,
        }
    )

    return (x - 2) ** 2


study = optuna.create_study(
    study_name="trackio-multirun",
)

study.optimize(
    objective,
    n_trials=10,
    callbacks=[trackioc],
)

trackioc.finish()
```

## Notes

- Trackio synchronization to Hugging Face Spaces is eventually consistent and may take time to become remotely visible.

- For large studies or multirun experiments, it is strongly recommended to:

  1. complete the Optuna study locally first,
  1. verify local experiment tracking,
  1. then synchronize results to Hugging Face.

- In most cases, the recommended configuration is:

```python
TrackioCallback(
    ...,
    sync_on_finish=True,
    sync_frequency="study",
)
```

instead of per-trial synchronization.

- Per-trial synchronization may significantly increase runtime due to repeated remote uploads and Hugging Face Space propagation delays.

- To ensure proper Trackio lifecycle management in multi-run mode, the objective function should always be wrapped with:

```python
@trackioc.track_in_trackio()
```
