from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING

from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import GPSampler
from optuna.samplers import NSGAIIISampler
from optuna.samplers import NSGAIISampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._nsgaiii._sampler import _GENERATION_KEY as NSGA3_GENERATION_KEY
from optuna.samplers.nsgaii._sampler import _GENERATION_KEY as NSGA2_GENERATION_KEY
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


MAXINT32 = (1 << 31) - 1
THRESHOLD_OF_MANY_OBJECTIVES = 4


class AutoSampler(BaseSampler):
    _N_COMPLETE_TRIALS_FOR_CMAES = 250
    _N_COMPLETE_TRIALS_FOR_NSGA = 1000

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
        seed_for_random_sampler = self._rng.rng.randint(MAXINT32)
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

    def _determine_multi_objective_sampler(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> None:
        if isinstance(self._sampler, (NSGAIISampler, NSGAIIISampler)):
            return

        seed = self._rng.rng.randint(MAXINT32)
        if not isinstance(self._sampler, TPESampler):
            self._sampler = TPESampler(
                seed=seed,
                multivariate=True,
                warn_independent_sampling=False,
                constraints_func=self._constraints_func,
                constant_liar=True,
            )
            return

        complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        complete_trials.sort(key=lambda trial: trial.datetime_complete)
        if len(complete_trials) >= self._N_COMPLETE_TRIALS_FOR_NSGA:
            nsga_sampler_cls = (
                NSGAIISampler
                if len(study.directions) < THRESHOLD_OF_MANY_OBJECTIVES
                else NSGAIIISampler
            )
            # Use ``NSGAIISampler`` if search space is numerical and
            # len(complete_trials) <= _N_COMPLETE_TRIALS_FOR_NSGA.
            self._sampler = nsga_sampler_cls(constraints_func=self._constraints_func, seed=seed)

    def _determine_single_objective_sampler(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> None:
        if isinstance(self._sampler, TPESampler):
            return

        seed = self._rng.rng.randint(MAXINT32)
        if (
            self._constraints_func is not None
            or any(isinstance(d, CategoricalDistribution) for d in search_space.values())
            or self._include_conditional_param(study)
        ):
            # NOTE(nabenabe): The statement above is always true for Trial#1.
            # Use ``TPESampler`` if search space includes conditional or categorical parameters.
            # TBD: group=True?
            self._sampler = TPESampler(
                seed=seed,
                multivariate=True,
                warn_independent_sampling=False,
                constraines_func=self._constraints_func,
                constant_liar=True,
            )
            return

        complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        complete_trials.sort(key=lambda trial: trial.datetime_complete)
        if len(complete_trials) <= self._N_COMPLETE_TRIALS_FOR_CMAES:
            # Use ``GPSampler`` if search space is numerical and
            # len(complete_trials) <= _N_COMPLETE_TRIALS_FOR_CMAES.
            if not isinstance(self._sampler, GPSampler):
                self._sampler = GPSampler(seed=seed)
            return

        if not isinstance(self._sampler, CmaEsSampler):
            # Use ``CmaEsSampler`` if search space is numerical and
            # len(complete_trials) > _N_COMPLETE_TRIALS_FOR_CMAES.
            # Warm start CMA-ES with the first _N_COMPLETE_TRIALS_FOR_CMAES complete trials.
            warm_start_trials = complete_trials[: self._N_COMPLETE_TRIALS_FOR_CMAES]
            # NOTE(nabenabe): ``CmaEsSampler`` internally falls back to ``RandomSampler`` for
            # 1D problems.
            self._sampler = CmaEsSampler(
                seed=seed, source_trials=warm_start_trials, warn_independent_sampling=False
            )

    def _determine_sampler(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> None:
        if len(study.directions) == 1:
            self._determine_single_objective_sampler(study, trial, search_space)
        else:
            self._determine_multi_objective_sampler(study, trial, search_space)

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
        n_objectives = len(study.directions)
        if n_objectives > 1 and isinstance(self._sampler, TPESampler):
            # NOTE(nabenabe): Set generation 0 so that NSGAIISampler can use the trial information
            # obtained during the optimization using TPESampler.
            # NOTE(nabenabe): Use NSGA-III for many objective problems.
            _GENERATION_KEY = (
                NSGA2_GENERATION_KEY
                if n_objectives < THRESHOLD_OF_MANY_OBJECTIVES
                else NSGA3_GENERATION_KEY
            )
            study._storage.set_trial_system_attr(trial._trial_id, _GENERATION_KEY, 0)

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
