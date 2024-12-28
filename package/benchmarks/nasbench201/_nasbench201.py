from __future__ import annotations

from hpo_benchmarks import NASBench201
import optuna
import optunahub


_INDEX_SUFFIX = "_index"
_DIRECTIONS = {
    "minimize": optuna.study.StudyDirection.MINIMIZE,
    "maximize": optuna.study.StudyDirection.MAXIMIZE,
}


def _extract_search_space(bench: NASBench201) -> dict[str, optuna.distributions.BaseDistribution]:
    param_types = bench.param_types
    search_space = {}
    for param_name, choices in bench.search_space.items():
        n_choices = len(choices)
        key = f"{param_name}{_INDEX_SUFFIX}"
        if param_types[param_name] == str:
            dist = optuna.distributions.CategoricalDistribution(list(range(n_choices)))
        else:
            dist = optuna.distributions.IntDistribution(low=0, high=n_choices - 1)
        search_space[key] = dist
    return search_space


class Problem(optunahub.benchmarks.BaseProblem):
    available_metric_names: list[str] = NASBench201.available_metric_names
    available_dataset_names: list[int] = NASBench201.available_dataset_names

    def __init__(
        self, dataset_id: int, seed: int | None = None, metric_names: list[str] | None = None
    ):
        if dataset_id < 0 or dataset_id >= len(self.available_dataset_names):
            n_datasets = len(self.available_dataset_names)
            raise ValueError(
                f"dataset_id must be between 0 and {n_datasets - 1}, but got {dataset_id}."
            )

        self.dataset_name = self.available_dataset_names[dataset_id]
        self._problem = NASBench201(
            dataset_name=self.dataset_name, seed=seed, metric_names=metric_names
        )
        self._search_space = _extract_search_space(self._problem)

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [_DIRECTIONS[self._problem.directions[name]] for name in self.metric_names]

    def evaluate(self, params: dict[str, int | float | str]) -> list[float]:
        problem_search_space = self._problem.search_space
        len_suffix = len(_INDEX_SUFFIX)
        modified_params = {}
        for index_name, choice_index in params.items():
            param_name = index_name[:-len_suffix]
            modified_params[param_name] = problem_search_space[param_name][choice_index]

        results = self._problem(modified_params)
        return [results[name] for name in self.metric_names]

    def reseed(self, seed: int | None = None) -> None:
        self._problem.reseed(seed)

    @property
    def metric_names(self) -> list[str]:
        return self._problem.metric_names
