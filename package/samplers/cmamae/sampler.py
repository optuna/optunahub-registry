from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import numpy as np
import optunahub
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

SimpleBaseSampler = optunahub.load_module("samplers/simple").SimpleBaseSampler


class CmaMaeSampler(SimpleBaseSampler):
    """A sampler using CMA-MAE as implemented in pyribs.

    `CMA-MAE <https://arxiv.org/abs/2205.10752>`_ is a quality diversity
    algorithm that has demonstrated state-of-the-art performance in a variety of
    domains. `pyribs <https://pyribs.org>`_ is a bare-bones Python library for
    quality diversity optimization algorithms. For a primer on CMA-MAE and
    pyribs, we recommend referring to the series of `pyribs tutorials
    <https://docs.pyribs.org/en/stable/tutorials.html>`_.

    For simplicity, this implementation provides a default instantiation of
    CMA-MAE with a `GridArchive
    <https://docs.pyribs.org/en/stable/api/ribs.archives.GridArchive.html>`_ and
    `EvolutionStrategyEmitter
    <https://docs.pyribs.org/en/stable/api/ribs.emitters.EvolutionStrategyEmitter.html>`_
    with improvement ranking, all wrapped up in a `Scheduler
    <https://docs.pyribs.org/en/stable/api/ribs.schedulers.Scheduler.html>`_.

    Args:
        param_names: List of names of parameters to optimize.
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
        self._param_names = param_names[:]

        # NOTE: SimpleBaseSampler must know Optuna search_space information.
        search_space = {
            name: FloatDistribution(-1e9, 1e9) for name in self._param_names
        }
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
            ) for _ in range(n_emitters)
        ]

        # Number of solutions generated in each batch from pyribs.
        self._batch_size = n_emitters * emitter_batch_size

        self._scheduler = Scheduler(
            archive,
            emitters,
            result_archive=result_archive,
        )

        self._values_to_tell: list[list[float]] = []

    def _validate_params(self, param_names: list[str],
                         emitter_x0: dict[str, float]) -> None:
        dim = len(param_names)
        param_set = set(param_names)
        if dim != len(param_set):
            raise ValueError(
                "Some elements in param_names are duplicated. Please make it a unique list."
            )

        if set(param_names) != emitter_x0.keys():
            raise ValueError(
                "emitter_x0 does not contain the parameters listed in param_names. "
                "Please provide an initial value for each parameter.")

    def _validate_param_names(self, given_param_names: Iterable[str]) -> None:
        if set(self._param_names) != set(given_param_names):
            raise ValueError("The given param names must match the param names "
                             "initially passed to this sampler.")

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
            self, study: Study, trial: FrozenTrial,
            search_space: dict[str, BaseDistribution]) -> dict[str, float]:
        self._validate_param_names(search_space.keys())

        # Note: Batch optimization means we need to enqueue trials.
        solutions = self._scheduler.ask()
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
        # TODO: Is it safe to assume the parameters will always come back in the
        # order that they were sent out by the scheduler? Pyribs makes that
        # assumption and stores the solutions internally. If not, maybe we can
        # retrieve solutions based on their trial ID?

        self._validate_param_names(trial.params.keys())

        # Store the trial result.
        self._values_to_tell.append(values)

        # If we have not retrieved the whole batch of solutions, then we should
        # not tell() the results to the scheduler yet.
        if len(self._values_to_tell) != self._batch_size:
            return

        # Tell the batch results to external sampler once the batch is ready.
        values = np.asarray(self._values_to_tell)
        self._scheduler.tell(objective=values[:, 0], measures=values[:, 1:])

        # Empty the results.
        self._values_to_tell = []
