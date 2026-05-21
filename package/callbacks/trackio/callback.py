from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import functools
from typing import Any
from typing import cast
from typing import TYPE_CHECKING
import uuid

import optuna
from optuna._imports import try_import


with try_import() as _imports:
    import trackio


if TYPE_CHECKING:
    from optuna.study.study import ObjectiveFuncType


class TrackioCallback:
    """Callback to track Optuna trials with Trackio.

    This callback enables tracking of an Optuna study using Trackio.
    By default, the entire study is recorded as a single experiment run,
    where all suggested hyperparameters and optimized metrics are logged
    and visualized as a function of optimizer steps.

    Trackio is offline-first and does not require authentication for local
    usage. Optionally, results can be synchronized to Hugging Face Spaces
    or exported as Hugging Face Datasets for sharing, visualization, and
    reproducibility.

    .. note::
        Trackio does not require users to be logged in for local experiment
        tracking. Authentication is only required when synchronizing results
        to Hugging Face Hub (e.g., Spaces or Datasets).

    .. note::
        Unlike Weights & Biases, Trackio does not rely on global mutable state.
        Each run is explicitly initialized and finalized, which makes this
        callback safe to use in long-running processes and research pipelines.

    .. note::
        To ensure deterministic trial ordering in logged metrics, this
        callback should only be used with ``study.optimize(n_jobs=1)``.
        Parallel optimization may result in out-of-order steps.
    """

    def __init__(
        self,
        project: str,
        metric_name: str | Sequence[str] = "value",
        *,
        as_multirun: bool = False,
        space_id: str | None = None,
        dataset_id: str | None = None,
        private: bool | None = None,
        resume: str = "allow",
        sync_on_finish: bool = False,
        sync_frequency: str = "study",
        sync_run_in_background: bool = True,
        trackio_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _imports.check()

        if not isinstance(metric_name, (str, Sequence)):
            raise TypeError(
                f"metric_name must be str or sequence[str], got {type(metric_name)}"
            )

        if sync_frequency not in {"study", "trial"}:
            raise ValueError(
                "sync_frequency must be either 'study' or 'trial'"
            )

        self._project: str = project
        self._metric_name: str | Sequence[str] = metric_name
        self._as_multirun: bool = as_multirun
        self._space_id: str | None = space_id
        self._dataset_id: str | None = dataset_id
        self._private: bool | None = private
        self._resume: str = resume
        self._sync_on_finish: bool = sync_on_finish
        self._sync_frequency: str = sync_frequency
        self._sync_run_in_background: bool = sync_run_in_background
        self._trackio_kwargs: dict[str, Any] = trackio_kwargs or {}

        # Explicit internal state
        self._objective_wrapped: bool = False
        self._resolved_run_name: str | None = None
        self._study_instance_id: str | None = None
        self._active_trial_number: int | None = None
        self._study_run_initialized: bool = False

    def __call__(
        self,
        _study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        if trial.values is None:
            return

        # Multirun logging is handled entirely
        # inside the wrapped objective lifecycle.
        if self._as_multirun:
            if not self._objective_wrapped:
                print(
                    "TrackioCallback(as_multirun=True) requires the objective "
                    "to be wrapped with @trackioc.track_in_trackio(). "
                )
            return

        metrics = self._build_metrics(trial)

        trackio.log(
            {
                **trial.params,
                **metrics,
                "trial_number": trial.number,
            },
            step=trial.number,
        )

    def track_in_trackio(self) -> Callable:
        """Decorator for enabling Trackio logging inside the objective function.

        This decorator wraps an Optuna objective function so that a Trackio run
        is initialized before the objective executes and finalized afterward.
        Any calls to :func:`trackio.log` inside the objective will be associated
        with the correct run.

        The decorator is required when logging from inside the objective
        function, since Optuna callbacks are invoked *after* a trial finishes
        and therefore cannot manage per-trial runtime state.

        When ``as_multirun=True``, a separate Trackio run is created for each
        Optuna trial. When ``as_multirun=False``, all trials are logged into a
        single run.

        Returns:
            A wrapped objective function with Trackio logging enabled.
        """

        def decorator(func: ObjectiveFuncType) -> ObjectiveFuncType:
            self._objective_wrapped = True
            wrapped = self._wrap_objective(func)

            @functools.wraps(func)
            def wrapper(trial: optuna.trial.Trial) -> Any:
                return wrapped(trial)

            return wrapper

        return decorator

    def finish(self) -> None:
        """Finalize Trackio synchronization and cleanup."""

        if self._sync_on_finish:
            self._safe_sync()

        if (
            not self._as_multirun
            and self._study_run_initialized
        ):
            cast(Any, trackio).finish()

    def _safe_sync(self) -> None:
        try:
            trackio.sync(
                project=self._project,
                run_in_background=self._sync_run_in_background,
            )

        except TimeoutError as exc:
            print(
                "Trackio sync timed out while waiting for "
                "remote visibility. Local experiment data "
                "was successfully recorded and uploaded. "
                f"Original error: {exc}"
            )

    def _initialize_study_identity(
        self,
        study_name: str | None,
    ) -> None:
        if self._resolved_run_name is not None:
            return

        resolved_study_name = study_name or "optuna-study"

        if self._study_instance_id is None:
            self._study_instance_id = uuid.uuid4().hex[:8]

        self._resolved_run_name = (
            f"{resolved_study_name}-{self._study_instance_id}"
        )

    def _wrap_objective(self, func: ObjectiveFuncType) -> ObjectiveFuncType:
        @functools.wraps(func)
        def wrapped(trial: optuna.trial.Trial) -> Any:
            study = trial.study

            self._initialize_study_identity(
                study.study_name,
            )

            base_name = cast(str, self._resolved_run_name)

            if self._as_multirun:
                run_name = f"{base_name}/trial-{trial.number}"
                self._active_trial_number = trial.number

                trackio.init(
                    project=self._project,
                    name=run_name,
                    space_id=self._space_id,
                    dataset_id=self._dataset_id,
                    private=self._private,
                    resume=self._resume,
                    **self._trackio_kwargs,
                )

            else:
                run_name = base_name

                if not self._study_run_initialized:
                    trackio.init(
                        project=self._project,
                        name=run_name,
                        space_id=self._space_id,
                        dataset_id=self._dataset_id,
                        private=self._private,
                        resume=self._resume,
                        **self._trackio_kwargs,
                    )

                    self._study_run_initialized = True

                    trackio.log(
                        {
                            "study_name": study.study_name,
                        }
                    )

            trial_completed = False

            try:
                result = func(trial)

                if self._as_multirun:
                    values = (
                        [result]
                        if not isinstance(result, Sequence)
                        else result
                    )

                    metrics = self._build_result_metrics(values)

                    trackio.log(
                        {
                            **trial.params,
                            **metrics,
                            "trial_number": trial.number,
                        },
                        step=trial.number,
                    )

                trial_completed = True

                return result

            except optuna.exceptions.TrialPruned:
                trackio.log({"trial_state": "pruned"})
                raise

            except Exception as exc:
                trackio.log(
                    {
                        "trial_state": "failed",
                        "error": str(exc),
                    }
                )
                raise

            finally:
                if self._as_multirun:
                    cast(Any, trackio).finish()
                    self._active_trial_number = None

                    if (
                        trial_completed
                        and self._sync_on_finish
                        and self._sync_frequency == "trial"
                    ):
                        self._safe_sync()

        return wrapped

    def _build_result_metrics(
        self,
        values: Sequence[float],
    ) -> dict[str, float]:
        if isinstance(self._metric_name, str):
            if len(values) == 1:
                names = [self._metric_name]
            else:
                names = [
                    f"{self._metric_name}_{i}"
                    for i in range(len(values))
                ]
        else:
            if len(self._metric_name) != len(values):
                raise ValueError(
                    "Metric names must match number of objectives "
                    f"({len(self._metric_name)} vs {len(values)})"
                )

            names = list(self._metric_name)

        return dict(zip(names, values))

    def _build_metrics(
        self,
        trial: optuna.trial.FrozenTrial,
    ) -> dict[str, float]:
        values = trial.values
        assert values is not None

        return self._build_result_metrics(values)