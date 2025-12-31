from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

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

        # Environmental selection: create new archive P̄_{t+1}
        archive = self._environmental_selection(study, population, fitness_values)

        return archive

    def _calculate_fitness(
        self,
        study: Study,
        trials: list[FrozenTrial],
    ) -> np.ndarray:
        """Calculate SPEA2 fitness values for all trials.

        SPEA2 fitness = raw fitness + density

        Raw fitness R(i) = sum of strength values of all dominators
        Strength S(i) = number of individuals dominated by i
        Density D(i) = 1 / (sigma_k + 2), where sigma_k is distance to k-th nearest neighbor

        Args:
            study: Optuna study object.
            trials: List of trials to evaluate.

        Returns:
            Array of fitness values (lower is better).
        """
        n = len(trials)
        if n == 0:
            return np.array([])

        # Get objective values
        objectives = self._get_objective_values(study, trials)

        # Calculate strength values S(i) = number of individuals dominated by i
        strength = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(
                    objectives[i], objectives[j], study.directions
                ):
                    strength[i] += 1

        # Calculate raw fitness R(i) = sum of strength of dominators
        raw_fitness = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(
                    objectives[j], objectives[i], study.directions
                ):
                    raw_fitness[i] += strength[j]

        # Calculate density using k-nearest neighbor
        k = int(math.sqrt(n))
        density = self._calculate_density(objectives, k)

        # Final fitness = raw fitness + density
        fitness = raw_fitness + density

        return fitness

    def _get_objective_values(
        self,
        study: Study,
        trials: list[FrozenTrial],
    ) -> np.ndarray:
        """Extract objective values from trials.

        Args:
            study: Optuna study object.
            trials: List of trials.

        Returns:
            Array of shape (n_trials, n_objectives) containing objective values.
        """
        n_objectives = len(study.directions)
        objectives = np.zeros((len(trials), n_objectives))

        for i, trial in enumerate(trials):
            if n_objectives == 1:
                objectives[i, 0] = trial.value
            else:
                objectives[i] = trial.values

        return objectives

    def _dominates(
        self,
        values1: np.ndarray,
        values2: np.ndarray,
        directions: list[StudyDirection],
    ) -> bool:
        """Check if values1 Pareto-dominates values2.

        Args:
            values1: First objective values.
            values2: Second objective values.
            directions: Optimization directions.

        Returns:
            True if values1 dominates values2.
        """
        better_in_all = True
        better_in_at_least_one = False

        for v1, v2, direction in zip(values1, values2, directions):
            if direction == StudyDirection.MINIMIZE:
                if v1 > v2:
                    better_in_all = False
                if v1 < v2:
                    better_in_at_least_one = True
            else:  # MAXIMIZE
                if v1 < v2:
                    better_in_all = False
                if v1 > v2:
                    better_in_at_least_one = True

        return better_in_all and better_in_at_least_one

    def _calculate_density(
        self,
        objectives: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Calculate density estimation using k-nearest neighbor.

        Density D(i) = 1 / (sigma_k^i + 2)
        where sigma_k^i is the distance to the k-th nearest neighbor.

        Args:
            objectives: Array of objective values (n_trials, n_objectives).
            k: Number of nearest neighbors to consider.

        Returns:
            Array of density values.
        """
        n = len(objectives)
        if n == 0:
            return np.array([])

        # Ensure k is valid
        k = min(k, n - 1)
        if k <= 0:
            return np.zeros(n)

        density = np.zeros(n)

        for i in range(n):
            # Calculate distances to all other individuals
            distances = []
            for j in range(n):
                if i != j:
                    # Euclidean distance in objective space
                    dist = np.linalg.norm(objectives[i] - objectives[j])
                    distances.append(dist)

            # Sort distances and get k-th nearest neighbor distance
            distances.sort()
            sigma_k = distances[k - 1] if k - 1 < len(distances) else distances[-1]

            # Calculate density
            density[i] = 1.0 / (sigma_k + 2.0)

        return density

    def _environmental_selection(
        self,
        study: Study,
        trials: list[FrozenTrial],
        fitness: np.ndarray,
    ) -> list[FrozenTrial]:
        """Perform environmental selection to create the archive.

        SPEA2 environmental selection:
        1. Copy all nondominated individuals (F < 1) to archive
        2. If archive size < N-bar: fill with best dominated individuals
        3. If archive size > N-bar: truncate using archive truncation operator

        Args:
            study: Optuna study object.
            trials: All trials.
            fitness: Fitness values for all trials.

        Returns:
            Archive of selected trials.
        """
        # Step 1: Select all nondominated individuals (fitness < 1)
        nondominated_indices = np.where(fitness < 1.0)[0]
        archive_indices = list(nondominated_indices)

        # Step 2: Fill archive if needed
        if len(archive_indices) < self._archive_size:
            # Add best dominated individuals
            dominated_indices = np.where(fitness >= 1.0)[0]
            if len(dominated_indices) > 0:
                # Sort dominated individuals by fitness
                sorted_dominated = sorted(dominated_indices, key=lambda i: fitness[i])
                n_to_add = min(
                    self._archive_size - len(archive_indices), len(sorted_dominated)
                )
                archive_indices.extend(sorted_dominated[:n_to_add])

        # Step 3: Truncate archive if needed
        elif len(archive_indices) > self._archive_size:
            archive_trials = [trials[i] for i in archive_indices]
            selected_indices = self._truncate_archive(archive_trials, study)
            archive_indices = [archive_indices[i] for i in selected_indices]

        return [trials[i] for i in archive_indices]

    def _truncate_archive(
        self,
        archive_trials: list[FrozenTrial],
        study: Study,
    ) -> list[int]:
        """Truncate archive using SPEA2 archive truncation method.

        Iteratively remove individuals with minimum distance to other individuals,
        preserving boundary solutions.

        Args:
            archive_trials: Trials in the archive.
            study: Optuna study object.

        Returns:
            Indices of selected trials.
        """
        objectives = self._get_objective_values(study, archive_trials)
        n = len(archive_trials)
        remaining = list(range(n))

        # Remove individuals until archive size is reached
        while len(remaining) > self._archive_size:
            # Calculate distance matrix for remaining individuals
            min_distances = np.full(len(remaining), np.inf)

            for i, idx_i in enumerate(remaining):
                distances = []
                for j, idx_j in enumerate(remaining):
                    if i != j:
                        dist = np.linalg.norm(objectives[idx_i] - objectives[idx_j])
                        distances.append(dist)

                if distances:
                    distances.sort()
                    # Find individual with minimum distance to others
                    for k, dist in enumerate(distances):
                        if k == 0 or dist != distances[k - 1]:
                            min_distances[i] = dist
                            break

            # Remove individual with smallest distance
            remove_idx = np.argmin(min_distances)
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
