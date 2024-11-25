from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler


class CmaMaeSampler(optunahub.samplers.SimpleBaseSampler):
    """A sampler using CMA-MAE as implemented in pyribs.

    `CMA-MAE <https://dl.acm.org/doi/abs/10.1145/3583131.3590389>`_ is a quality
    diversity algorithm that has demonstrated state-of-the-art performance in a
    variety of domains. `Pyribs <https://pyribs.org>`_ is a bare-bones Python
    library for quality diversity optimization algorithms. For a primer on
    CMA-MAE, quality diversity, and pyribs, we recommend referring to the series
    of `pyribs tutorials <https://docs.pyribs.org/en/stable/tutorials.html>`_.

    For simplicity, this implementation provides a default instantiation of
    CMA-MAE with a `GridArchive
    <https://docs.pyribs.org/en/stable/api/ribs.archives.GridArchive.html>`_ and
    `EvolutionStrategyEmitter
    <https://docs.pyribs.org/en/stable/api/ribs.emitters.EvolutionStrategyEmitter.html>`_
    with improvement ranking, all wrapped up in a `Scheduler
    <https://docs.pyribs.org/en/stable/api/ribs.schedulers.Scheduler.html>`_.
    However, it is possible to implement many variations of CMA-MAE and other
    quality diversity algorithms using pyribs.

    Note that this sampler assumes the measures are set to user_attrs of each trial.
    To do so, please call ``trial.set_user_attr("YOUR MEASURE NAME", measure_value)`` for each
    measure.

    Args:
        param_names: List of names of parameters to optimize.
        measure_names: List of names of measures.
        archive_dims: Number of archive cells in each dimension of the measure
            space, e.g. ``[20, 30, 40]`` indicates there should be 3 dimensions
            with 20, 30, and 40 cells. (The number of dimensions is implicitly
            defined in the length of this argument).
        archive_ranges: Upper and lower bound of each dimension of the measure
            space for the archive, e.g. ``[(-1, 1), (-2, 2)]`` indicates the
            first dimension should have bounds :math:`[-1,1]` (inclusive), and
            the second dimension should have bounds :math:`[-2,2]` (inclusive).
            ``ranges`` should be the same length as ``dims``.
        archive_learning_rate: The learning rate for threshold updates in the
            archive.
        archive_threshold_min: The initial threshold value for all the cells in
            the archive.
        n_emitters: Number of emitters to use in CMA-MAE.
        emitter_x0: Mapping from parameter names to their initial values.
        emitter_sigma0: Initial step size / standard deviation of the
            distribution from which solutions are sampled in the emitter.
        emitter_batch_size: Number of solutions for each emitter to generate on
            each iteration.
    """

    def __init__(
        self,
        *,
        param_names: list[str],
        measure_names: list[str],
        archive_dims: list[int],
        archive_ranges: list[tuple[float, float]],
        archive_learning_rate: float,
        archive_threshold_min: float,
        n_emitters: int,
        emitter_x0: dict[str, float],
        emitter_sigma0: float,
        emitter_batch_size: int,
    ) -> None:
        self._validate_params(param_names, emitter_x0)
        self._param_names = param_names.copy()
        self._measure_names = measure_names.copy()

        # NOTE: SimpleBaseSampler must know Optuna search_space information.
        search_space = {name: FloatDistribution(-1e9, 1e9) for name in self._param_names}
        super().__init__(search_space=search_space)

        emitter_x0_np = self._convert_to_pyribs_params(emitter_x0)

        archive = GridArchive(
            solution_dim=len(param_names),
            dims=archive_dims,
            ranges=archive_ranges,
            learning_rate=archive_learning_rate,
            threshold_min=archive_threshold_min,
        )
        result_archive = GridArchive(
            solution_dim=len(param_names),
            dims=archive_dims,
            ranges=archive_ranges,
        )
        emitters = [
            EvolutionStrategyEmitter(
                archive,
                x0=emitter_x0_np,
                sigma0=emitter_sigma0,
                ranker="imp",
                selection_rule="mu",
                restart_rule="basic",
                batch_size=emitter_batch_size,
            )
            for _ in range(n_emitters)
        ]

        # Number of solutions generated in each batch from pyribs.
        self._batch_size = n_emitters * emitter_batch_size

        # Public to allow access for, e.g., visualization.
        self.scheduler = Scheduler(
            archive,
            emitters,
            result_archive=result_archive,
        )

        self._values_to_tell: list[list[float]] = []
        self._stored_trial_numbers: list[int] = []

    def _validate_params(self, param_names: list[str], emitter_x0: dict[str, float]) -> None:
        dim = len(param_names)
        param_set = set(param_names)
        if dim != len(param_set):
            raise ValueError(
                "Some elements in param_names are duplicated. Please make it a unique list."
            )

        if set(param_names) != emitter_x0.keys():
            raise ValueError(
                "emitter_x0 does not contain the parameters listed in param_names. "
                "Please provide an initial value for each parameter."
            )

    def _validate_param_names(self, given_param_names: Iterable[str]) -> None:
        if set(self._param_names) != set(given_param_names):
            raise ValueError(
                "The given param names must match the param names "
                "initially passed to this sampler."
            )

    def _convert_to_pyribs_params(self, params: dict[str, float]) -> np.ndarray:
        np_params = np.empty(len(self._param_names), dtype=float)
        for i, p in enumerate(self._param_names):
            np_params[i] = params[p]
        return np_params

    def _convert_to_optuna_params(self, params: np.ndarray) -> dict[str, float]:
        dict_params = {}
        for i, p in enumerate(self._param_names):
            dict_params[p] = params[i]
        return dict_params

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, float]:
        self._validate_param_names(search_space.keys())

        # Note: Batch optimization means we need to enqueue trials.
        solutions = self.scheduler.ask()
        next_params = self._convert_to_optuna_params(solutions[0])
        for solution in solutions[1:]:
            params = self._convert_to_optuna_params(solution)
            study.enqueue_trial(params)

        return next_params

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._validate_param_names(trial.params.keys())

        # Store the trial result.
        direction0 = study.directions[0]
        minimize_in_optuna = direction0 == StudyDirection.MINIMIZE
        if values is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not support Failed trials, "
                f"but trial#{trial.number} failed."
            )
        user_attrs = trial.user_attrs
        if any(measure_name not in user_attrs for measure_name in self._measure_names):
            raise KeyError(
                f"All of measure in measure_names={self._measure_names} must be set to "
                "trial.user_attrs. Please call trial.set_user_attr(<measure_name>, <value>) "
                "for each measure."
            )

        self._raise_error_if_multi_objective(study)
        modified_values = [
            float(values[0]),
            float(user_attrs[self._measure_names[0]]),
            float(user_attrs[self._measure_names[1]]),
        ]
        if minimize_in_optuna:
            # The direction of the first objective (pyribs maximizes).
            modified_values[0] = -values[0]
        self._values_to_tell.append(modified_values)
        self._stored_trial_numbers.append(trial.number)

        # If we have not retrieved the whole batch of solutions, then we should
        # not tell() the results to the scheduler yet.
        if len(self._values_to_tell) != self._batch_size:
            return

        # Tell the batch results to external sampler once the batch is ready.
        values_to_tell = np.asarray(self._values_to_tell)[np.argsort(self._stored_trial_numbers)]
        self.scheduler.tell(objective=values_to_tell[:, 0], measures=values_to_tell[:, 1:])

        # Empty the results.
        self._values_to_tell = []
        self._stored_trial_numbers = []
