from typing import Any

import numpy as np
import optuna
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import FrozenTrial
import optunahub


class DESampler(optunahub.load_module("samplers/simple").SimpleSampler):  # type: ignore
    def __init__(self, search_space: dict[str, BaseDistribution]) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        #  parent_generation, parent_population = self._collect_parent_population(study)

        # generation = parent_generation + 1
        # study._storage.set_trial_system_attr(trial._trial_id, _GENERATION_KEY, generation)

        # if parent_generation < 0:
        #     return {}

        # return self._child_generation_strategy(study, search_space, parent_population)
        # return self._random_sampler.sample_independent(
        #     study, trial, param_name, param_distribution
        # )
        trials = study._get_trials(deepcopy=False, use_cache=True)
        if len(trials) < 3:
            params = {}
            for n, d in search_space.items():
                if isinstance(d, FloatDistribution):
                    params[n] = self._rng.uniform(d.low, d.high)
                elif isinstance(d, IntDistribution):
                    params[n] = self._rng.randint(d.low, d.high)
                else:
                    raise ValueError("Unsupported distribution")
            return params
        
        arr = np.array(trials)
        self._rng.choice(arr, 3, replace=False)
        x = self._rng.choice(arr, 3, replace=False)
        print(x)
        
        
        
        params = {}
        for n, d in search_space.items():
            if isinstance(d, FloatDistribution):
                params[n] = self._rng.uniform(d.low, d.high)
            elif isinstance(d, IntDistribution):
                params[n] = self._rng.randint(d.low, d.high)
            else:
                raise ValueError("Unsupported distribution")
        return params
    
    def __init__(self,
            agent_max,           # エージェント数
            crossover_rate=0.5,  # 交叉率
            scaling=0.5,         # 差分の適用率
        ):
            self.agent_max = agent_max
            self.crossover_rate = crossover_rate
            self.scaling = scaling    

    def init(self, problem):
        self.problem = problem

        # 初期位置の生成
        self.agents = []
        for _ in range(self.agent_max):
            self.agents.append(problem.create())


    def step(self):

        for i, agent in enumerate(self.agents):

            # iを含まない3個体をランダムに選択
            x1, x2, x3 = self.rng.sample([ j for j in range(len(self.agents)) if j != i ], 3)
            pos1 = self.agents[x1].getArray()
            pos2 = self.agents[x2].getArray()
            pos3 = self.agents[x3].getArray()

            # 3個体から変異ベクトルをだす
            pos1 = np.asarray(pos1)
            pos2 = np.asarray(pos2)
            pos3 = np.asarray(pos3)
            m_pos = pos1 + self.scaling * (pos2 - pos3)

            # 変異ベクトルで交叉させる(一様交叉)
            pos = agent.getArray()
            ri = self.rng.randint(0, len(pos))  # 1成分は必ず変異ベクトル
            for j in range(len(pos)):
                if  ri == j or self.rng.rng() < self.crossover_rate:
                    pos[j] = m_pos[j]
                else:
                    pass  # 更新しない

            # 優れている個体なら置き換える
            new_agent = self.problem.create(pos)
            self.count += 1
            if agent.getScore() < new_agent.getScore():
                self.agents[i] = new_agent

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # Following parameters are randomly sampled here.
        # 1. A parameter in the initial population/first generation.
        # 2. A parameter to mutate.
        # 3. A parameter excluded from the intersection search space.

        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )


if __name__ == "__main__":

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0, 1)

        return x

    sampler = DESampler({"x": FloatDistribution(0, 1)})
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print(study.best_trial.value, study.best_trial.params)
