from __future__ import annotations

from _scalar_aggregation_func import pbi
from _scalar_aggregation_func import tchebycheff
from _scalar_aggregation_func import weighted_sum
import numpy as np
from optuna import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from scipy.spatial import cKDTree
from scipy.stats import qmc


class MOEAdElitePopulationSelectionStrategy:
    def __init__(
        self,
        seed: int | None,
        population_size: int,
        n_neighbors: int,
        scalar_aggregation_func: str,
    ) -> None:
        self._seed = seed
        self._population_size = population_size
        self._n_neighbors = n_neighbors
        self._weight_vectors = None

        if scalar_aggregation_func == "tchebycheff":
            self._scalar_aggregation_func = tchebycheff
        elif scalar_aggregation_func == "PBI":
            self._scalar_aggregation_func = pbi
        elif scalar_aggregation_func == "weighted_sum":
            self._scalar_aggregation_func = weighted_sum
        else:
            raise ValueError(
                "`scalar_aggregation_function` must be one of 'weighted_sum', 'tchebycheff', 'PBI'."
            )

    def __call__(self, study: Study, population: list[FrozenTrial]) -> list[FrozenTrial]:
        if self._weight_vectors is None:
            weight_vectors = self._generate_weight_vectors(
                self._population_size, len(study.directions)
            )
            self._compute_neighborhoods(weight_vectors)

        if len(population) == self._population_size:
            return population

        self._update_reference_point(study.directions, population)

        if self._weight_vectors is not None:
            return self._update_neighboring_solutions(population, self._weight_vectors)
        else:
            raise ValueError("Weight vectors are not generated.")

    def _update_neighboring_solutions(
        self, population: list[FrozenTrial], weight_vectors: np.ndarray
    ) -> list[FrozenTrial]:
        elite_population: list[FrozenTrial] = []
        offset = len(population) // 2

        for id, neighbor_ids in self._neighbors.items():
            id_old = offset + id
            elite = population[id_old]
            lambda_ = weight_vectors[id]

            g_old = self._scalar_aggregation_func(
                lambda_, population[id_old], self._reference_point, self._nadir_point
            )
            for n_id in neighbor_ids:
                # check new population
                target_population = population[n_id]
                g_new = self._scalar_aggregation_func(
                    lambda_, target_population, self._reference_point, self._nadir_point
                )
                if g_new < g_old:
                    elite = target_population
                    g_old = g_new

                # check old population
                target_population = population[n_id + offset]
                g_new = self._scalar_aggregation_func(
                    lambda_, target_population, self._reference_point, self._nadir_point
                )
                if g_new < g_old:
                    elite = target_population
                    g_old = g_new

            elite_population.append(elite)

        return elite_population

    def _update_reference_point(
        self, directions: list[StudyDirection], population: list[FrozenTrial]
    ) -> None:
        self._reference_point = []
        self._nadir_point = []  # using for normalize of subproblem objective values

        for i, direction in enumerate(directions):
            target_values = np.array([trial.values[i] for trial in population])
            if direction == StudyDirection.MINIMIZE:
                self._reference_point.append(np.min(target_values, axis=0))
                self._nadir_point.append(np.max(target_values, axis=0))
            else:
                self._reference_point.append(np.max(target_values, axis=0))
                self._nadir_point.append(np.min(target_values, axis=0))

    # More uniform sequences generation method is better.
    def _generate_weight_vectors(self, n_vector: int, n_objective: int) -> np.ndarray:
        if n_objective == 2:
            x = np.linspace(0, 1, n_vector)
            y = 1 - x
            self._weight_vectors = np.column_stack((x, y))
        else:
            sampler = qmc.Halton(d=n_objective, seed=self._seed, scramble=True)
            vectors = sampler.random(n_vector)
            sum = np.sum(vectors, axis=1, keepdims=True)
            self._weight_vectors = vectors / sum

        return self._weight_vectors

    def _compute_neighborhoods(self, weight_vectors: np.ndarray) -> None:
        self._neighbors: dict[int, list[int]] = {}

        tree = cKDTree(weight_vectors)
        for i, weight_vector in enumerate(weight_vectors):
            _, idx = tree.query(weight_vector, k=self._n_neighbors + 1)

            # include itself, first element is itself
            self._neighbors[i] = idx
