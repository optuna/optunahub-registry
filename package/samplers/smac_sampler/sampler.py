from __future__ import annotations

from collections.abc import Sequence

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
from smac.utils.configspace import get_config_hash


SimpleBaseSampler = optunahub.load_module("samplers/simple").SimpleBaseSampler


def dummmy_target_func(config: Configuration, seed: int = 0) -> float:
    # This is only a placed holder function that allows us to initialize a new SMAC facade
    return 0


class SMACSampler(SimpleBaseSampler):  # type: ignore
    def __init__(
        self,
        search_space: dict[str, BaseDistribution],
        n_trials: int = 100,  # This is required for initial design
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
    ) -> None:
        super().__init__(search_space)
        self._cs, self._hp_scale_value = self._convert_to_config_space_design_space(search_space)
        # TODO init SMAC facade according to the given arguments
        scenario = Scenario(configspace=self._cs, deterministic=True, n_trials=n_trials)
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
        smac = HyperparameterOptimizationFacade(
            scenario,
            target_function=dummmy_target_func,
            model=surrogate_model,
            acquisition_function=acq_func,
            config_selector=config_selector,
            initial_design=init_design,
            overwrite=True,
        )
        self.smac = smac

        # this value is used to store the instance-seed paris of each evaluated configuraitons
        self._runs_instance_seed_keys: dict[str, tuple[str | None, int]] = {}

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

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, float]:
        trial_info: TrialInfo = self.smac.ask()
        cfg = trial_info.config
        self._runs_instance_seed_keys[get_config_hash(cfg)] = (
            trial_info.instance,
            trial_info.seed,
        )
        params = {}
        for name, hp_value in cfg.items():
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
        # Transform the trail info to smac
        params = trial.params
        cfg_params = {}
        for name, hp_value in params.items():
            if name in self._hp_scale_value:
                hp_value = self._step_hp_to_intger(hp_value, scale_info=self._hp_scale_value[name])
            cfg_params[name] = hp_value

        # params to smac HP
        y = np.asarray(values)
        if state == TrialState.COMPLETE:
            status = StatusType.SUCCESS
        elif state == TrialState.RUNNING:
            status = StatusType.RUNNING
        else:
            status = StatusType.CRASHED
        trial_value = TrialValue(y, status=status)

        cfg = Configuration(configuration_space=self._cs, values=cfg_params)
        # Since Optuna does not provide us the
        instance, seed = self._runs_instance_seed_keys[get_config_hash(cfg)]
        info = TrialInfo(cfg, seed=seed, instance=instance)
        self.smac.tell(info=info, value=trial_value, save=False)

    def _transform_step_hp_to_integer(
        self, distribution: FloatDistribution | IntDistribution
    ) -> tuple[int, tuple[int | float]]:
        """
        Given that step discretises the target Float distribution and this is not supported by ConfigSpace, we need to
        manually transform this type of HP into integral values. To construct a new integer value, we need to know the
        amount of possible values contained in the hyperparameter and the information required to transform the integral
        values back to the target function
        """
        assert distribution.step is not None
        n_discrete_values = int(
            np.round((distribution.high - distribution.low) / distribution.step)
        )
        return n_discrete_values, (distribution.low, distribution.step)  # type: ignore

    def _step_hp_to_intger(
        self, hp_value: float | int, scale_info: tuple[int | float, int | float]
    ) -> int:
        return int(np.round((hp_value - scale_info[0]) / scale_info[1]))

    def _integer_to_step_hp(
        self, integer_value: int, scale_info: tuple[int | float, int | float]
    ) -> float | int:
        """
        This function is the inverse of _transform_step_hp_to_intger, we will transform the integer_value back to the
        target hyperparameter values
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
                    # Given that step discretises the target Float distribution and this is not supported by
                    # ConfigSpace, we need to manually transform this type of HP into integral values to sampler and
                    # transform them back to the raw HP values during evaluation phases. Hence,
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
                config_space.add_hyperparameter(hp)
        return config_space, scale_values
