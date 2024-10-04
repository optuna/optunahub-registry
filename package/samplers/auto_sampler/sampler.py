from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING

from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import GPSampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
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
            from optuna.samplers import AutoSampler


            def objective(trial):
                x = trial.suggest_float("x", -5, 5)
                return x**2


            study = optuna.create_study(sampler=AutoSampler())
            study.optimize(objective, n_trials=10)

    .. note::
        This sampler might require ``scipy``, ``torch``, and ``cmaes``.
        You can install these dependencies with ``pip install scipy torch cmaes``.

    Args:
        seed: Seed for random number generator.

    """

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._sampler: BaseSampler = RandomSampler(seed=seed)

    def reseed_rng(self) -> None:
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
        if isinstance(self._sampler, TPESampler):
            return

        if any(
            isinstance(d, CategoricalDistribution) for d in search_space.values()
        ) or self._include_conditional_param(study):
            # NOTE(nabenabe): The statement above is always true for Trial#1.
            # Use ``TPESampler`` if search space includes conditional or categorical parameters.
            # TBD: group=True?
            self._sampler = TPESampler(
                seed=self._seed, multivariate=True, warn_independent_sampling=False
            )
            return

        if trial.number < 250:
            # Use ``GPSampler`` if search space is numerical and n_trials <= 250.
            if not isinstance(self._sampler, GPSampler):
                self._sampler = GPSampler(seed=self._seed)
            return

        if not isinstance(self._sampler, CmaEsSampler):
            # Use ``CmaEsSampler`` if search space is numerical and n_trials > 250.
            # Warm start CMA-ES with trials up to trial.number of 249.
            warm_start_trials = study.get_trials(
                deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)
            )
            # NOTE(nabenabe): ``CmaEsSampler`` internally falls back to ``RandomSampler`` for
            # 1D problems.
            self._sampler = CmaEsSampler(
                seed=self._seed, source_trials=warm_start_trials, warn_independent_sampling=False
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
