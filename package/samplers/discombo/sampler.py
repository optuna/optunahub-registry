from __future__ import annotations

import inspect
import math
import threading
from typing import Any
import warnings

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub

from discombo import DisCoOptimizer
from discombo import SimpleSearchSpace


# Integer parameters with at least this many values are modeled as continuous (and rounded back to
# integers), integer parameters with fewer values as categorical. DisCoMBO itself models integers
# with at least 10 values by a Gaussian leaf.
_MIN_INT_VALUES_CONTINUOUS = 10

# DisCoOptimizer arguments that are set by the sampler itself.
_RESERVED_ARGUMENTS = (
    "search_space",
    "objective",
    "iterations",
    "num_samples",
    "initial_samples",
    "seed",
    "abort_after_intervention",
)
# Deprecated DisCoOptimizer arguments.
_DEPRECATED_ARGUMENTS = ("exploration_method", "mc_type")


class DisCoMBOSampler(optunahub.samplers.SimpleBaseSampler):
    """Sampler based on DisCoMBO (Distributional Conformance Model-Based Optimization).

    DisCoMBO learns a probabilistic circuit over the hyperparameters and the objective value of the
    completed trials. It samples candidates from the circuit conditioned on good objective values
    and selects them by their distributional conformance score (DisCo), a p-value of how much a
    candidate surprises the circuit, cycling between exploration and exploitation. Expert beliefs
    can be injected as interventions; their conformance with the observations is reported.

    The first ``n_startup_trials`` trials are sampled at random. Afterwards, DisCoMBO proposes
    batches of ``batch_size`` configurations, which are assigned to the following trials.

    Args:
        search_space:
            A dictionary of Optuna distributions. If None, the search space is inferred from the
            completed trials (the intersection search space); parameters outside of it are sampled
            at random.
        seed:
            Seed for the random number generators.
        n_startup_trials:
            Number of random trials before DisCoMBO takes over.
        batch_size:
            Number of configurations DisCoMBO proposes per iteration, i.e. per learned circuit.
        **optimizer_kwargs:
            Further arguments of ``discombo.DisCoOptimizer`` (see the DisCoMBO documentation), e.g.
            ``stratified_pool_size`` or ``stratified_cycle_length``.
    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        *,
        seed: int | None = None,
        n_startup_trials: int = 20,
        batch_size: int = 10,
        **optimizer_kwargs: Any,
    ) -> None:
        super().__init__(search_space, seed)
        if n_startup_trials < 1 or batch_size < 1:
            raise ValueError("n_startup_trials and batch_size must be positive.")
        reserved = sorted(set(optimizer_kwargs) & set(_RESERVED_ARGUMENTS))
        if reserved:
            raise ValueError(f"These DisCoOptimizer arguments are set by the sampler: {reserved}.")
        known = set(inspect.signature(DisCoOptimizer).parameters) - set(_DEPRECATED_ARGUMENTS)
        unknown = sorted(set(optimizer_kwargs) - known)
        if unknown:
            raise TypeError(f"Unknown DisCoOptimizer arguments: {unknown}.")
        self._n_startup_trials = n_startup_trials
        self._batch_size = batch_size
        self._optimizer_kwargs = optimizer_kwargs
        self._rng = np.random.RandomState(seed)
        self._lock = threading.Lock()
        self._scheduled_interventions: list[tuple[int, dict[str, Any]]] = []
        self._intervention_records: list[dict[str, Any]] = []
        self._reset(None)

    def intervene(self, beliefs: dict[str, Any], trial_number: int) -> None:
        """Inject expert beliefs into the batch DisCoMBO proposes at or after ``trial_number``.

        Args:
            beliefs:
                Maps parameter names either to a value (a point intervention, given like the values
                of the parameter, e.g. ``"adam"`` or ``1e-3``) or to a distribution over values
                ``{"dist": ..., "parameters": [...]}`` in DisCoMBO's representation:
                ``"gauss"`` (mean, std) and ``"uniform"`` (low, high) for continuous parameters,
                in natural-log space for parameters with ``log=True``; ``"cat"`` (one weight per
                choice) and ``"int_uniform"`` (low, high exclusive, as choice indices) for
                categorical parameters and integer parameters with fewer than 10 values.
            trial_number:
                The intervention is applied to the first batch DisCoMBO proposes for a trial with
                this or a higher number.
        """
        with self._lock:
            self._scheduled_interventions.append((trial_number, dict(beliefs)))
            self._scheduled_interventions.sort(key=lambda item: item[0])

    def get_disco_scores(self) -> list[dict[str, Any]]:
        """Return the distributional conformance (DisCo) score of each applied intervention.

        Returns:
            One record per applied intervention with the keys ``"trial_number"`` (the trial the
            intervened batch started at), ``"beliefs"``, ``"disco_score"`` (the mean DisCo score of
            the beliefs w.r.t. the circuit conditioned on the best observed value; low scores
            indicate that the beliefs contradict the observations) and ``"sample_disco_scores"``
            (the DisCo scores of the individual samples of the beliefs).
        """
        with self._lock:
            return [dict(record) for record in self._intervention_records]

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if not search_space:
            return {}
        if len(study.directions) > 1:
            raise ValueError("DisCoMBOSampler does not support multi-objective optimization.")
        with self._lock:
            if search_space != self._search_space:
                self._reset(search_space)
            if not self._queue:
                observations = self._observations(study, search_space)
                if len(observations) < self._n_startup_trials:
                    return {}
                self._propose(search_space, observations, trial.number)
            config = self._queue.pop(0)
        return {name: _to_optuna(config[name], dist) for name, dist in search_space.items()}

    def reseed_rng(self) -> None:
        # The proposals of an existing DisCoMBO optimizer are shared by all threads via the queue,
        # so reseeding only affects the random startup trials and DisCoMBO optimizers created later.
        super().reseed_rng()
        self._rng = np.random.RandomState()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]  # locks cannot be pickled
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _reset(self, search_space: dict[str, BaseDistribution] | None) -> None:
        self._search_space = search_space
        self._optimizer: Any = None
        self._told: set[int] = set()
        self._queue: list[dict[str, Any]] = []
        self._applied_beliefs: list[dict[str, Any]] = []
        self._applied_iterations: list[int] = []

    def _observations(
        self, study: Study, search_space: dict[str, BaseDistribution]
    ) -> list[tuple[int, dict[str, Any], float]]:
        """Completed trials of the search space as (number, configuration, maximized value)."""
        sign = 1.0 if study.directions[0] == StudyDirection.MAXIMIZE else -1.0
        observations = []
        for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            if t.value is None or not math.isfinite(t.value):
                continue
            if all(
                name in t.params and t.distributions[name] == d for name, d in search_space.items()
            ):
                config = {name: t.params[name] for name in search_space}
                observations.append((t.number, config, sign * t.value))
        return observations

    def _propose(
        self,
        search_space: dict[str, BaseDistribution],
        observations: list[tuple[int, dict[str, Any], float]],
        trial_number: int,
    ) -> None:
        """Tell DisCoMBO the new observations and queue its next batch."""
        if self._optimizer is None:
            definition = {name: _to_definition(dist) for name, dist in search_space.items()}
            self._optimizer = DisCoOptimizer(
                SimpleSearchSpace(definition),
                iterations=2**31 - 1,
                num_samples=self._batch_size,
                initial_samples=self._n_startup_trials,
                seed=int(self._rng.randint(2**31 - 1)),
                **self._optimizer_kwargs,
            )
        new = [(n, c, v) for n, c, v in observations if n not in self._told]
        self._optimizer.tell([c for _, c, _ in new], [v for _, _, v in new])
        self._told.update(n for n, _, _ in new)

        beliefs = self._next_intervention(search_space, trial_number)
        if beliefs is not None:
            self._applied_beliefs.append(beliefs)
            self._applied_iterations.append(self._optimizer.iteration)
            self._optimizer.intervene(self._applied_beliefs, self._applied_iterations)
        n_records = len(self._optimizer.get_disco_scores())
        self._queue = list(self._optimizer.ask())
        for record in self._optimizer.get_disco_scores()[n_records:]:
            self._intervention_records.append(
                {
                    "trial_number": trial_number,
                    "beliefs": self._applied_beliefs[record["intervention_idx"]],
                    "disco_score": record["disco_score"],
                    "sample_disco_scores": record["sample_disco_scores"],
                }
            )

    def _next_intervention(
        self, search_space: dict[str, BaseDistribution], trial_number: int
    ) -> dict[str, Any] | None:
        """Pop the earliest due intervention with valid beliefs, in DisCoMBO's representation."""
        while (
            self._scheduled_interventions and self._scheduled_interventions[0][0] <= trial_number
        ):
            _, beliefs = self._scheduled_interventions.pop(0)
            converted = {}
            for name, belief in beliefs.items():
                if name not in search_space:
                    warnings.warn(
                        f"Ignoring the belief about {name!r}, which is not in the search space."
                    )
                    continue
                try:
                    converted[name] = _to_discombo_belief(belief, search_space[name])
                except (ValueError, TypeError):
                    warnings.warn(
                        f"Ignoring the belief {belief!r} about {name!r}, an invalid value."
                    )
            if converted:
                return converted
        return None


def _n_int_values(dist: IntDistribution) -> int:
    return (dist.high - dist.low) // dist.step + 1


def _int_as_categorical(dist: IntDistribution) -> bool:
    return _n_int_values(dist) < _MIN_INT_VALUES_CONTINUOUS


def _to_definition(dist: BaseDistribution) -> dict[str, Any]:
    """DisCoMBO's search space definition of an Optuna distribution."""
    if isinstance(dist, CategoricalDistribution):
        return {"dtype": "cat", "allowed": list(dist.choices)}
    if isinstance(dist, IntDistribution):
        if _int_as_categorical(dist):
            return {"dtype": "int", "allowed": list(range(dist.low, dist.high + 1, dist.step))}
        return {
            "dtype": "float",
            "min": float(dist.low),
            "max": float(dist.high),
            "is_log": dist.log,
        }
    if isinstance(dist, FloatDistribution):
        return {"dtype": "float", "min": dist.low, "max": dist.high, "is_log": dist.log}
    raise NotImplementedError(f"Unsupported distribution: {dist}.")


def _to_optuna(value: Any, dist: BaseDistribution) -> Any:
    """Optuna's value of a value proposed by DisCoMBO."""
    if isinstance(dist, CategoricalDistribution):
        return value
    if isinstance(dist, IntDistribution):
        k = round((float(value) - dist.low) / dist.step)
        return int(min(max(dist.low + k * dist.step, dist.low), dist.high))
    assert isinstance(dist, FloatDistribution)
    v = min(max(float(value), dist.low), dist.high)
    if dist.step is not None:
        k = round((v - dist.low) / dist.step)
        v = min(dist.low + k * dist.step, dist.high)
    return v


def _to_discombo_belief(belief: Any, dist: BaseDistribution) -> Any:
    """DisCoMBO's representation of a belief about one parameter."""
    if isinstance(belief, dict):
        return belief  # a distribution, already given in DisCoMBO's representation
    if isinstance(dist, CategoricalDistribution):
        return dist.choices.index(belief)
    if isinstance(dist, IntDistribution) and _int_as_categorical(dist):
        return list(range(dist.low, dist.high + 1, dist.step)).index(int(belief))
    assert isinstance(dist, (FloatDistribution, IntDistribution))
    return math.log(belief) if dist.log else float(belief)
