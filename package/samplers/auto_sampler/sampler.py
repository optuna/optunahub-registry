from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import GPSampler
from optuna.samplers import NSGAIISampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class AutoSampler(BaseSampler):
    """Sampler automatically choosing an appropriate sampler based on search space.

    This sampler is convenient when you are unsure what sampler to use.

    Example:

        .. testcode::

            import optuna
            import optunahub


            def objective(trial):
                x = trial.suggest_float("x", -5, 5)
                return x**2

            module = optunahub.load_module("samplers/auto_sampler")
            study = optuna.create_study(sampler=module.AutoSampler())
            study.optimize(objective, n_trials=300)

    .. note::
        This sampler requires optional dependencies of Optuna.
        You can install them with ``pip install "optuna[optional]"``.
        Alternatively, you can install them with ``pip install -r https://hub.optuna.org/samplers/auto_sampler/requirements.txt``.

    Args:
        seed: Seed for random number generator.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or they are pruned, but this behavior is
            subject to change in the future releases.

            .. note::
                If you enable this feature, Optuna's default sampler will be selected automatically.

    """

    def __init__(
        self,
        seed: int | None = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ) -> None:
        self._rng = LazyRandomState(seed)
        seed_for_random_sampler = self._rng.rng.randint(1 << 32)
        self._sampler: BaseSampler = RandomSampler(seed=seed_for_random_sampler)
        self._constraints_func = constraints_func

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._sampler.reseed_rng()

    def _include_conditional_param(self, study: Study) -> bool:
        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED))
        if len(trials) == 0:
            return False

        param_key = set(trials[0].params)
        for t in trials:
            if param_key != set(t.params):
                return True

        return False

    def _determine_sampler(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> None:
        if len(study.directions) > 1:
            # Fallback to the default sampler if the study has multiple objectives.
            if isinstance(self._sampler, NSGAIISampler):
                return
            # TODO(toshihikoyanase): add warning message about fallback.
            self._sampler = NSGAIISampler(constraints_func=self._constraints_func)
            return

        if isinstance(self._sampler, TPESampler):
            return

        seed = self._rng.rng.randint(1 << 32)
        if self._constraints_func is not None:
            # Fallback to the default sampler if the study has constraints.
            # TODO(toshihikoyanase): add warning message about fallback.
            self._sampler = TPESampler(seed=seed, constraines_func=self._constraints_func)
            return

        if any(
            isinstance(d, CategoricalDistribution) for d in search_space.values()
        ) or self._include_conditional_param(study):
            # NOTE(nabenabe): The statement above is always true for Trial#1.
            # Use ``TPESampler`` if search space includes conditional or categorical parameters.
            # TBD: group=True?
            self._sampler = TPESampler(
                seed=seed, multivariate=True, warn_independent_sampling=False
            )
            return

        complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        complete_trials.sort(key=lambda trial: trial.datetime_complete)
        if len(complete_trials) < 250:
            # Use ``GPSampler`` if search space is numerical and n_trials <= 250.
            if not isinstance(self._sampler, GPSampler):
                self._sampler = GPSampler(seed=seed)
            return

        if not isinstance(self._sampler, CmaEsSampler):
            # Use ``CmaEsSampler`` if search space is numerical and n_trials > 250.
            # Warm start CMA-ES with trials up to trial.number of 249.
            warm_start_trials = complete_trials[:250]
            # NOTE(nabenabe): ``CmaEsSampler`` internally falls back to ``RandomSampler`` for
            # 1D problems.
            self._sampler = CmaEsSampler(
                seed=seed, source_trials=warm_start_trials, warn_independent_sampling=False
            )

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space = IntersectionSearchSpace().calculate(study)
        sampler_key = "auto:sampler"
        if len(search_space) == 0:
            study._storage.set_trial_system_attr(
                trial._trial_id, sampler_key, self._sampler.__class__.__name__
            )
            return {}

        self._determine_sampler(study, trial, search_space)
        study._storage.set_trial_system_attr(
            trial._trial_id, sampler_key, self._sampler.__class__.__name__
        )
        return self._sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._sampler.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._sampler.after_trial(study, trial, state, values)
