from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
import pandas as pd

import hebo as hebo_module
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.hebo import HEBO


_logger = get_logger(f"optuna.{__name__}")


class HEBOSampler(optunahub.samplers.SimpleBaseSampler):
    """A sampler using `HEBO <https://github.com/huawei-noah/HEBO/tree/master/HEBO>__` as the backend.

    For further information about HEBO algorithm, please refer to the following paper:
    - `HEBO Pushing The Limits of Sample-Efficient Hyperparameter Optimisation <https://arxiv.org/abs/2012.03826>__`

    Args:
        search_space:
            By specifying search_space, the sampling speed at each iteration becomes slightly quicker, but this argument is not necessary to run this sampler. Default is :obj:`None`.

        seed:
            A seed for the initialization of ``HEBOSampler``. Default is :obj:`None`.
            Please note that the Bayesian optimization part is not deterministic even if seed is
            fixed due to the backend implementation.

        constant_liar:
            If :obj:`True`, penalize running trials to avoid suggesting parameter configurations
            nearby. Default is :obj:`False`.

            .. note::
                Abnormally terminated trials often leave behind a record with a state of
                ``RUNNING`` in the storage.
                Such "zombie" trial parameters will be avoided by the constant liar algorithm
                during subsequent sampling.
                When using an :class:`~optuna.storages.RDBStorage`, it is possible to enable the
                ``heartbeat_interval`` to change the records for abnormally terminated trials to
                ``FAIL``.
                (This note is quoted from `TPESampler <https://github.com/optuna/optuna/blob/v4.1.0/optuna/samplers/_tpe/sampler.py#L215-L222>__`.)

            .. note::
                It is recommended to set this value to :obj:`True` during distributed
                optimization to avoid having multiple workers evaluating similar parameter
                configurations. In particular, if each objective function evaluation is costly
                and the durations of the running states are significant, and/or the number of
                workers is high.
                (This note is quoted from `TPESampler <https://github.com/optuna/optuna/blob/v4.1.0/optuna/samplers/_tpe/sampler.py#L224-L229>__`.)

            .. note::
                HEBO algorithm involves multi-objective optimization of multiple acquisition functions.
                While `constant_liar` is a simple way to get diverse params for parallel optimization,
                it may not be the best approach for HEBO.

        num_obj:
            If :obj:`>1`, use the multi-objective version of HEBO (GeneralBO) as the backend.
            Default is :obj:`1`.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler. If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler`
            is used as the default.
    """  # NOQA

    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        *,
        seed: int | None = None,
        constant_liar: bool = False,
        independent_sampler: BaseSampler | None = None,
        num_obj: int = 1,
    ) -> None:
        super().__init__(search_space, seed)
        self._hebo = None
        self._multi_objective = num_obj > 1
        if search_space is not None and not constant_liar:
            design_space = self._convert_to_hebo_design_space(search_space)
            if self._multi_objective:
                self._hebo = GeneralBO(design_space, num_obj=num_obj)
            else:
                self._hebo = HEBO(design_space, scramble_seed=seed)
        self._intersection_search_space = IntersectionSearchSpace()
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._constant_liar = constant_liar
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _suggest_and_transform_to_dict(
        hebo: HEBO | GeneralBO, search_space: dict[str, BaseDistribution]
    ) -> dict[str, float]:
        params = {}
        for name, row in hebo.suggest().items():
            if name not in search_space:
                continue

            dist = search_space[name]
            if (
                isinstance(dist, (IntDistribution, FloatDistribution))
                and not dist.log
                and dist.step is not None
            ):
                step_index = row.iloc[0]
                params[name] = dist.low + step_index * dist.step
            else:
                params[name] = row.iloc[0]

        return params

    @staticmethod
    def _transform_to_dict_and_observe(
        hebo: HEBO | GeneralBO,
        search_space: dict[str, BaseDistribution],
        study: Study,
        trials: list[FrozenTrial],
    ) -> None:
        directions = study.directions if study._is_multi_objective() else [study.direction]
        sign = np.array([1 if d == StudyDirection.MINIMIZE else -1 for d in directions])

        values = np.array(
            [
                trial.values
                if trial.state == TrialState.COMPLETE and trial.values is not None
                else [np.nan] * len(directions)
                for trial in trials
            ],
            dtype=float,
        )

        worst_values = []
        for idx, direction in enumerate(directions):
            column = values[:, idx]
            worst_value = (
                np.nanmax(column) if direction == StudyDirection.MINIMIZE else np.nanmin(column)
            )
            worst_value = 0.0 if np.isnan(worst_value) else worst_value
            worst_values.append(worst_value)
        worst_values = np.array(worst_values, dtype=float)

        for idx in range(len(directions)):
            nan_mask = np.isnan(values[:, idx])
            if nan_mask.any():
                values[nan_mask, idx] = worst_values[idx]

        # Assume that the back-end HEBO implementation aims to minimize.
        nan_padded_values = sign * values
        params = pd.DataFrame([t.params for t in trials])
        for name, dist in search_space.items():
            if (
                isinstance(dist, (IntDistribution, FloatDistribution))
                and not dist.log
                and dist.step is not None
            ):
                # NOTE(nabenabe): We do not round here because HEBO treats params as float even if
                # the domain is defined on integer. By not rounding, HEBO can handle any changes in
                # the domain of these parameters such as changes in low, high, and step.
                params[name] = (params[name] - dist.low) / dist.step

        # GeneralBO API changed in 0.3.6
        if isinstance(hebo, GeneralBO) and hebo_module.__version__ > "0.3.5":
            hebo.observe_new_data(params, nan_padded_values)
        else:
            hebo.observe(params, nan_padded_values)

    def _sample_relative_define_and_run(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._suggest_and_transform_to_dict(self._hebo, search_space)

    def _sample_relative_stateless(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if self._constant_liar:
            target_states = [TrialState.COMPLETE, TrialState.RUNNING]
        else:
            target_states = [TrialState.COMPLETE]

        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=target_states, use_cache=use_cache)
        is_complete = np.array([t.state == TrialState.COMPLETE for t in trials])
        if not np.any(is_complete):
            # note: The backend HEBO implementation uses Sobol sampling here.
            # This sampler does not call `hebo.suggest()` here because
            # Optuna needs to know search space by running the first trial in Define-by-Run.
            return {}

        trials = [t for t in trials if set(search_space.keys()) <= set(t.params.keys())]
        seed = int(self._rng.integers(low=1, high=(1 << 31)))
        design_space = self._convert_to_hebo_design_space(search_space)
        if self._multi_objective:
            hebo = GeneralBO(design_space, num_obj=len(study.directions))
        else:
            hebo = HEBO(design_space, scramble_seed=seed)
        self._transform_to_dict_and_observe(hebo, search_space, study, trials)
        return self._suggest_and_transform_to_dict(hebo, search_space)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if study._is_multi_objective() and not self._multi_objective:
            raise ValueError(
                f"To use {self.__class__.__name__} for multi-objective optimization, please specify the 'num_obj' parameter."
            )
        if self._hebo is None or self._constant_liar is True:
            return self._sample_relative_stateless(study, trial, search_space)
        else:
            return self._sample_relative_define_and_run(study, trial, search_space)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._hebo is not None and values is not None:
            # Note that trial.values is None and trial.state is RUNNNING here.
            trial_ = optuna.trial.create_trial(
                state=state,  # updated
                params=trial.params,
                distributions=trial.distributions,
                values=values,  # updated
            )
            self._transform_to_dict_and_observe(
                hebo=self._hebo, search_space=trial.distributions, study=study, trials=[trial_]
            )

    def _convert_to_hebo_design_space(
        self, search_space: dict[str, BaseDistribution]
    ) -> DesignSpace:
        design_space = []
        for name, distribution in search_space.items():
            config: dict[str, Any] = {"name": name}
            if isinstance(distribution, (FloatDistribution, IntDistribution)):
                if not distribution.log and distribution.step is not None:
                    config["type"] = "int"
                    # NOTE(nabenabe): high is adjusted in Optuna so that below is divisable.
                    n_steps = int(
                        np.round((distribution.high - distribution.low) / distribution.step + 1)
                    )
                    config["lb"] = 0
                    config["ub"] = n_steps - 1
                else:
                    config["lb"] = distribution.low
                    config["ub"] = distribution.high
                    if distribution.log:
                        config["type"] = (
                            "pow_int" if isinstance(distribution, IntDistribution) else "pow"
                        )
                    else:
                        assert not isinstance(distribution, IntDistribution)
                        config["type"] = "num"
            elif isinstance(distribution, CategoricalDistribution):
                config["type"] = "cat"
                config["categories"] = distribution.choices
            else:
                raise NotImplementedError(f"Unsupported distribution: {distribution}")

            design_space.append(config)

        return DesignSpace().parse(design_space)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return optuna.search_space.intersection_search_space(
            study._get_trials(deepcopy=False, use_cache=True)
        )

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        if any(param_name in trial.params for trial in trials):
            _logger.warning(f"Use `RandomSampler` for {param_name} due to dynamic search space.")

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
