from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


class SPEAIIElitePopulationSelectionStrategy:
    def __init__(
        self,
        *,
        population_size: int,
        archive_size: int,
    ) -> None:
        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")
        self._population_size = population_size
        self._archive_size = archive_size

    def __call__(
        self,
        study: Study,
        population: list[FrozenTrial],
    ) -> list[FrozenTrial]:
        archive: list[FrozenTrial] = []
        fitness_values = self._calculate_fitness(study, population)
        archive = self._environmental_selection(population, fitness_values)
        return archive

    def _calculate_fitness(
        self,
        study: Study,
        trials: list[FrozenTrial],
    ) -> np.ndarray:
        """Calculate SPEA2 fitness values for all trials.

        SPEA2 fitness = Raw fitness + Density

        Strength S(i) = number of individuals dominated by i
        Raw fitness R(i) = sum of strength values of all dominators
        Density D(i) = 1 / (sigma_k + 2), where sigma_k is distance to k-th nearest neighbor

        Args:
            study: Optuna study object.
            trials: List of trials to evaluate.

        Returns:
            Array of fitness values (lower is better).
        """
        n_trials = len(trials)
        if n_trials == 0:
            return np.array([])

        strength = self._calculate_strength(study, trials)
        raw_fitness = self._calculate_raw_fitness(study, trials, strength)
        density = self._calculate_density(trials)

        fitness = raw_fitness + density
        return fitness

    def _calculate_strength(self, study: Study, trials: list[FrozenTrial]) -> np.ndarray:
        """
        Strength S(i) = number of individuals dominated by i
        """
        n_trials = len(trials)
        strength = np.zeros(n_trials)
        for i in range(n_trials):
            for j in range(n_trials):
                if i != j and _dominates(trials[i], trials[j], study.directions):
                    strength[i] += 1
        return strength

    def _calculate_raw_fitness(
        self, study: Study, trials: list[FrozenTrial], strength: np.ndarray
    ) -> np.ndarray:
        """
        Raw fitness R(i) = sum of strength values of all dominators
        """
        n_trials = len(trials)
        raw_fitness = np.zeros(n_trials)
        for i in range(n_trials):
            for j in range(n_trials):
                if i != j and _dominates(trials[j], trials[i], study.directions):
                    raw_fitness[i] += strength[j]
        return raw_fitness

    def _calculate_distance_matrix(self, values_matrix: np.ndarray) -> np.ndarray:
        """Calculate pairwise Euclidean distance matrix efficiently using broadcasting.

        Args:
            values_matrix: Matrix of shape (n_trials, n_objectives) containing objective values.

        Returns:
            Distance matrix of shape (n_trials, n_trials).
        """
        diff = values_matrix[:, np.newaxis, :] - values_matrix[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances

    def _calculate_density(self, trials: list[FrozenTrial]) -> np.ndarray:
        """
        Density D(i) = 1 / (sigma_k^i + 2)
        where sigma_k^i is the distance to the k-th nearest neighbor.
        """
        k = int(math.sqrt(self._population_size + self._archive_size))

        n_trials = len(trials)
        if n_trials == 0:
            return np.array([])

        values_matrix = np.array([trial.values for trial in trials])
        distance_matrix = self._calculate_distance_matrix(values_matrix)

        # For each trial, find k-th nearest neighbor distance
        density = np.zeros(n_trials)
        for i in range(n_trials):
            sorted_distances = np.sort(distance_matrix[i])
            sigma_k = sorted_distances[k] if k < len(sorted_distances) else sorted_distances[-1]
            density[i] = 1.0 / (sigma_k + 2.0)

        return density

    def _environmental_selection(
        self,
        trials: list[FrozenTrial],
        fitness: np.ndarray,
    ) -> list[FrozenTrial]:
        """Perform environmental selection to create the archive.

        SPEA2 environmental selection:
        1. Copy all nondominated individuals (F < 1) to archive
        2. If archive individuals < archive_size: fill with best dominated individuals
        3. If archive individuals > archive_size: truncate using archive truncation operator

        Args:
            study: Optuna study object.
            trials: All trials.
            fitness: Fitness values for all trials.

        Returns:
            Archive of selected trials.
        """
        # Copy all nondominated individuals (F < 1) to archive
        nondominated_indices = np.where(fitness < 1.0)[0]
        archive_indices = list(nondominated_indices)

        # If archive individuals < archive_size: fill with best dominated individuals
        if len(archive_indices) < self._archive_size:
            dominated_indices = np.where(fitness >= 1.0)[0]
            if len(dominated_indices) > 0:
                sorted_dominated = sorted(dominated_indices, key=lambda i: fitness[i])
                n_addition = min(self._archive_size - len(archive_indices), len(sorted_dominated))
                archive_indices.extend(sorted_dominated[:n_addition])

        # If archive individuals > archive_size: truncate using archive truncation operator
        elif len(archive_indices) > self._archive_size:
            archive_trials = [trials[i] for i in archive_indices]
            selected_indices = self._truncate_archive(archive_trials)
            archive_indices = [archive_indices[i] for i in selected_indices]

        return [trials[i] for i in archive_indices]

    def _truncate_archive(
        self,
        archive_trials: list[FrozenTrial],
    ) -> list[int]:
        """Truncate archive using SPEA2 archive truncation method.

        Iteratively remove individuals with minimum distance to other individuals,
        using lexicographic ordering of sorted distance lists to break ties.
        This preserves boundary solutions.

        Args:
            archive_trials: Trials in the archive.

        Returns:
            Indices of selected trials.
        """
        n_trials = len(archive_trials)
        remaining = list(range(n_trials))

        values_matrix = np.array([trial.values for trial in archive_trials])
        distance_matrix = self._calculate_distance_matrix(values_matrix)

        while len(remaining) > self._archive_size:
            n_remaining = len(remaining)

            remaining_indices = np.array(remaining)
            distance_submatrix = distance_matrix[np.ix_(remaining_indices, remaining_indices)]

            distance_lists = []
            for i in range(n_remaining):
                distances = np.concatenate(
                    [distance_submatrix[i, :i], distance_submatrix[i, i + 1 :]]
                )
                distances.sort()
                distance_lists.append(distances)

            remove_idx = 0
            min_dist_list = distance_lists[0]

            for i in range(1, len(distance_lists)):
                for k in range(min(len(min_dist_list), len(distance_lists[i]))):
                    if distance_lists[i][k] < min_dist_list[k]:
                        remove_idx = i
                        min_dist_list = distance_lists[i]
                        break
                    elif distance_lists[i][k] > min_dist_list[k]:
                        break

            remaining.pop(remove_idx)

        return remaining

    def _get_archive_from_study(
        self,
        study: Study,
        generation: int,
    ) -> list[FrozenTrial]:
        """Retrieve the archive from a specific generation stored in study attributes.

        Args:
            study: Optuna study object.
            generation: Generation number.

        Returns:
            List of trials in the archive for the specified generation.
        """
        archive_key = f"spea2:archive:generation_{generation}"
        trial_numbers = study.user_attrs.get(archive_key, [])

        if not trial_numbers:
            return []

        # Get trials by their trial numbers
        all_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        trial_dict = {t.number: t for t in all_trials}

        archive = []
        for number in trial_numbers:
            if number in trial_dict:
                archive.append(trial_dict[number])

        return archive
