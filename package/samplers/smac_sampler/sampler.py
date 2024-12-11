from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import warnings

from ConfigSpace import Categorical
from ConfigSpace import Configuration
from ConfigSpace import ConfigurationSpace
from ConfigSpace import Float
from ConfigSpace import Integer
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
from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.function import EI
from smac.acquisition.function import LCB
from smac.acquisition.function import PI
from smac.facade import BlackBoxFacade
from smac.facade import HyperparameterOptimizationFacade
from smac.initial_design import AbstractInitialDesign
from smac.initial_design import LatinHypercubeInitialDesign
from smac.initial_design import RandomInitialDesign
from smac.initial_design import SobolInitialDesign
from smac.model.abstract_model import AbstractModel
from smac.runhistory.dataclasses import TrialInfo
from smac.runhistory.dataclasses import TrialValue
from smac.runhistory.enumerations import StatusType
from smac.scenario import Scenario


_SMAC_INSTANCE_KEY = "smac:instance"
_SMAC_SEED_KEY = "smac:seed"


class SMACSampler(optunahub.samplers.SimpleBaseSampler):
    """
    A sampler that uses SMAC3 v2.2.0.

    Please check the API reference for more details:
        https://automl.github.io/SMAC3/main/5_api.html

    Args:
        search_space:
            A dictionary of Optuna distributions.
        n_trials:
            Number of trials to be evaluated in a study.
            This argument is used to determine the number of initial configurations by SMAC3.
            Use at most ``n_trials * init_design_max_ratio`` number of configurations in the
            initial design.
            This argument does not have to be precise, but it is better to be exact for better
            performance.
        seed:
            Seed for random number generator.
            If ``None`` is given, seed is generated randomly.
        surrogate_model_type:
            What model to use for the probabilistic model.
            Either "gp" (Gaussian process), "gp_mcmc" (Gaussian process with MCMC), or "rf"
            (random forest). Default to "rf" (random forest).
        acq_func_type:
            What acquisition function to use.
            Either "ei" (expected improvement), "ei_log" (expected improvement with log-scaled
            function), "pi" (probability of improvement), or "lcb" (lower confidence bound).
            Default to "ei_log".
        init_design_type:
            What initialization sampler to use.
            Either "sobol" (Sobol sequence), "lhd" (Latin hypercube), or "random".
            Default to "sobol".
        surrogate_model_rf_num_trees:
            The number of trees used for random forest.
            Equivalent to ``n_estimators`` in ``RandomForestRegressor`` in sklearn.
        surrogate_model_rf_ratio_features:
            The ratio of features to use for each tree training in random forest.
            Equivalent to ``max_features`` in ``RandomForestRegressor`` in sklearn.
        surrogate_model_rf_min_samples_split:
            The minimum number of samples required to split an internal node:
            Equivalent to ``min_samples_split`` in ``RandomForestRegressor`` in sklearn.
        surrogate_model_rf_min_samples_leaf:
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at least
            ``min_samples_leaf`` training samples in each of the left and right branches.
            This may have the effect of smoothing the model, especially in regression.
            Equivalent to ``min_samples_leaf`` in ``RandomForestRegressor`` in sklearn.
        init_design_n_configs:
            Number of initial configurations.
        init_design_n_configs_per_hyperparameter:
            Number of initial configurations per hyperparameter.
            For example, if my configuration space covers five hyperparameters and
            n_configs_per_hyperparameter is set to 10, then 50 initial configurations will be
            sampled.
        init_design_max_ratio:
            Use at most ``n_trials * init_design_max_ratio`` number of configurations in the
            initial design. Additional configurations are not affected by this parameter.
        output_directory:
            Output directory path, defaults to "smac3_output".
            The directory in which to save the output.
            The files are saved in `./output_directory/name/seed`.
    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution],
        n_trials: int = 100,
        seed: int | None = None,
        *,
        surrogate_model_type: str = "rf",
        acq_func_type: str = "ei_log",
        init_design_type: str = "sobol",
        surrogate_model_rf_num_trees: int = 10,
        surrogate_model_rf_ratio_features: float = 1.0,
        surrogate_model_rf_min_samples_split: int = 2,
        surrogate_model_rf_min_samples_leaf: int = 1,
        init_design_n_configs: int | None = None,
        init_design_n_configs_per_hyperparameter: int = 10,
        init_design_max_ratio: float = 0.25,
        output_directory: str = "smac3_output",
    ) -> None:
        super().__init__(search_space)
        self._cs, self._hp_scale_value = self._convert_to_config_space_design_space(search_space)
        scenario = Scenario(
            configspace=self._cs,
            deterministic=True,
            n_trials=n_trials,
            seed=seed or -1,
            output_directory=Path(output_directory),
        )
        surrogate_model = self._get_surrogate_model(
            scenario,
            surrogate_model_type,
            rf_num_trees=surrogate_model_rf_num_trees,
            rf_ratio_features=surrogate_model_rf_ratio_features,
            rf_min_samples_split=surrogate_model_rf_min_samples_split,
            rf_min_samples_leaf=surrogate_model_rf_min_samples_leaf,
        )
        acq_func = self._get_acq_func(acq_func_type)
        init_design = self._get_init_design(
            scenario=scenario,
            init_design_type=init_design_type,
            n_configs=init_design_n_configs,
            n_configs_per_hyperparameter=init_design_n_configs_per_hyperparameter,
            max_ratio=init_design_max_ratio,
        )
        config_selector = HyperparameterOptimizationFacade.get_config_selector(
            scenario=scenario, retrain_after=1
        )

        def _dummmy_target_func(config: Configuration, seed: int = 0) -> float:
            # A placeholder function that allows us to initialize a new SMAC facade.
            return 0

        smac = HyperparameterOptimizationFacade(
            scenario,
            target_function=_dummmy_target_func,
            model=surrogate_model,
            acquisition_function=acq_func,
            config_selector=config_selector,
            initial_design=init_design,
            overwrite=True,
        )
        self.smac = smac

    def _get_surrogate_model(
        self,
        scenario: Scenario,
        model_type: str = "rf",
        rf_num_trees: int = 10,
        rf_ratio_features: float = 1.0,
        rf_min_samples_split: int = 2,
        rf_min_samples_leaf: int = 1,
    ) -> AbstractModel:
        if model_type == "gp":
            return BlackBoxFacade.get_model(scenario=scenario)
        elif model_type == "gp_mcmc":
            return BlackBoxFacade.get_model(scenario=scenario, model_type="mcmc")
        elif model_type == "rf":
            return HyperparameterOptimizationFacade.get_model(
                scenario=scenario,
                n_trees=rf_num_trees,
                ratio_features=rf_ratio_features,
                min_samples_split=rf_min_samples_split,
                min_samples_leaf=rf_min_samples_leaf,
            )
        else:
            raise ValueError(f"Unknown Surrogate Model Type {model_type}")

    def _get_acq_func(self, acq_func_type: str) -> AbstractAcquisitionFunction:
        all_acq_func = {"ei": EI(), "ei_log": EI(log=True), "pi": PI(), "lcb": LCB()}
        return all_acq_func[acq_func_type]

    def _get_init_design(
        self,
        scenario: Scenario,
        init_design_type: str,
        n_configs: int | None = None,
        n_configs_per_hyperparameter: int | None = 10,
        max_ratio: float = 0.25,
    ) -> AbstractInitialDesign:
        if init_design_type == "sobol":
            init_design = SobolInitialDesign(
                scenario=scenario,
                n_configs=n_configs,
                n_configs_per_hyperparameter=n_configs_per_hyperparameter,
                max_ratio=max_ratio,
            )
        elif init_design_type == "lhd":
            init_design = LatinHypercubeInitialDesign(
                scenario=scenario,
                n_configs=n_configs,
                n_configs_per_hyperparameter=n_configs_per_hyperparameter,
                max_ratio=max_ratio,
            )
        elif init_design_type == "random":
            init_design = RandomInitialDesign(
                scenario=scenario,
                n_configs=n_configs,
                n_configs_per_hyperparameter=n_configs_per_hyperparameter,
                max_ratio=max_ratio,
            )
        else:
            raise NotImplementedError(f"Unknown Initial Design Type: {init_design_type}")
        return init_design

    def reseed_rng(self) -> None:
        warnings.warn(
            "SMACSampler does not support reseeding the random number generator. Please instantiate a new SMACSampler with a different random seed instead."
        )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, float]:
        trial_info: TrialInfo = self.smac.ask()
        cfg = trial_info.config
        study._storage.set_trial_system_attr(
            trial._trial_id, _SMAC_INSTANCE_KEY, trial_info.instance
        )
        study._storage.set_trial_system_attr(trial._trial_id, _SMAC_SEED_KEY, trial_info.seed)
        params = {}
        for name, hp_value in cfg.items():
            # SMAC uses np.int64 for integer parameters
            if isinstance(hp_value, np.int64):
                hp_value = hp_value.item()  # Convert to Python int.
            if name in self._hp_scale_value:
                hp_value = self._integer_to_step_hp(
                    integer_value=hp_value, scale_info=self._hp_scale_value[name]
                )
            params[name] = hp_value

        return params

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        # Transform the trial info to smac.
        params = trial.params
        cfg_params = {}
        for name, hp_value in params.items():
            if name in self._hp_scale_value:
                hp_value = self._step_hp_to_integer(
                    hp_value, scale_info=self._hp_scale_value[name]
                )
            cfg_params[name] = hp_value

        # params to smac HP, in SMAC, we always perform the minimization.
        assert values is not None
        values_to_minimize = [
            v if d == StudyDirection.MINIMIZE else -v for d, v in zip(study.directions, values)
        ]
        y = np.asarray(values_to_minimize)
        if state == TrialState.COMPLETE:
            status = StatusType.SUCCESS
        elif state == TrialState.RUNNING:
            status = StatusType.RUNNING
        else:
            status = StatusType.CRASHED
        trial_value = TrialValue(y, status=status)

        cfg = Configuration(configuration_space=self._cs, values=cfg_params)
        instance = study._storage.get_trial_system_attrs(trial._trial_id).get(_SMAC_INSTANCE_KEY)
        seed = study._storage.get_trial_system_attrs(trial._trial_id).get(_SMAC_SEED_KEY)
        info = TrialInfo(cfg, seed=seed, instance=instance)
        self.smac.tell(info=info, value=trial_value, save=False)

    def _transform_step_hp_to_integer(
        self, distribution: FloatDistribution | IntDistribution
    ) -> tuple[int, tuple[int | float, int | float]]:
        """
        ConfigSpace does not support Float distribution with step, so we need to manually transform
        this type of HP into an integer value, i.e. the corresponding index of the grid.
        To construct a new integer value, we need to know the possible values contained in the
        hyperparameter and the information required to transform the integral values back to the
        target function.
        """
        assert distribution.step is not None
        n_discrete_values = int(
            np.round((distribution.high - distribution.low) / distribution.step)
        )
        return n_discrete_values, (distribution.low, distribution.step)

    def _step_hp_to_integer(
        self, hp_value: float | int, scale_info: tuple[int | float, int | float]
    ) -> int:
        return int(np.round((hp_value - scale_info[0]) / scale_info[1]))

    def _integer_to_step_hp(
        self, integer_value: int, scale_info: tuple[int | float, int | float]
    ) -> float | int:
        """
        This method is the inverse of _transform_step_hp_to_integer, we will transform the
        integer_value back to the target hyperparameter values.
        """
        return integer_value * scale_info[1] + scale_info[0]

    def _convert_to_config_space_design_space(
        self, search_space: dict[str, BaseDistribution]
    ) -> tuple[ConfigurationSpace, dict]:
        config_space = ConfigurationSpace()
        scale_values: dict[str, tuple] = {}
        for name, distribution in search_space.items():
            if isinstance(distribution, (FloatDistribution, IntDistribution)):
                if distribution.step is not None:
                    # See the doc-string of _transform_step_hp_to_integer.
                    n_discrete_values, scale_values_hp = self._transform_step_hp_to_integer(
                        distribution
                    )
                    scale_values[name] = scale_values_hp
                    hp = Integer(name, bounds=(0, n_discrete_values))
                else:
                    if isinstance(distribution, FloatDistribution):
                        hp = Float(
                            name,
                            bounds=(distribution.low, distribution.high),
                            log=distribution.log,
                        )
                    else:
                        hp = Integer(
                            name,
                            bounds=(distribution.low, distribution.high),
                            log=distribution.log,
                        )
            elif isinstance(distribution, CategoricalDistribution):
                hp = Categorical(name, items=distribution.choices)
            else:
                raise NotImplementedError(f"Unknown Hyperparameter Type: {type(distribution)}")
            if hp is not None:
                config_space.add(hp)
        return config_space, scale_values
