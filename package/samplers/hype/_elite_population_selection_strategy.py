from __future__ import annotations

from typing import Sequence
from typing import TYPE_CHECKING
import warnings

import numpy as np
from optuna.samplers.nsgaii._constraints_evaluation import _evaluate_penalty
from optuna.study import StudyDirection
from optuna.study._multi_objective import _fast_non_domination_rank
from optuna.trial import FrozenTrial
from scipy.stats import qmc


if TYPE_CHECKING:
    from optuna.study import Study


class HypEElitePopulationSelectionStrategy:
    """HypE (Hypervolume Estimation Algorithm) Elite Population Selection Strategy.

    This implements the environmental selection strategy described in:
    Bader, J. and Zitzler, E. (2011). HypE: An Algorithm for Fast Hypervolume-Based
    Many-Objective Optimization. Evolutionary Computation, 19(1):45-76.
    """

    def __init__(
        self,
        *,
        population_size: int,
        n_samples: int,
    ) -> None:
        """Initialize HypE elite selection strategy.

        Args:
            population_size: Size of the population.
            n_samples: Number of samples for hypervolume estimation.
                      Default is 10000 as recommended in the paper.
        """
        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        # Check if n_samples is a power of 2 for optimal Sobol sequence performance
        if n_samples > 0 and (n_samples & (n_samples - 1)) != 0:
            warnings.warn(
                f"`n_samples` ({n_samples}) is not a power of 2. "
                "Sobol sequence works most effectively with powers of 2 (e.g., 1024, 2048, 4096). "
                "Consider using a power of 2 for better QMC performance.",
                UserWarning,
                stacklevel=2,
            )

        self._population_size = population_size
        self._n_samples = n_samples
        # Cache for Sobol sampler reuse
        self._sobol_sampler: qmc.Sobol | None = None
        self._sobol_dimensions: int = 0

    def __call__(
        self,
        study: Study,
        population: list[FrozenTrial],
    ) -> list[FrozenTrial]:
        """Select elite population using HypE environmental selection.

        Args:
            study: Optuna study object.
            population: Current population of trials.

        Returns:
            Selected elite population.
        """
        if len(population) <= self._population_size:
            return population

        # Perform nondominated sorting
        fronts = self._rank_population(population, study.directions)

        # Build new population from fronts
        new_population: list[FrozenTrial] = []
        last_front_idx = -1

        for i, front in enumerate(fronts):
            if len(new_population) + len(front) <= self._population_size:
                new_population.extend(front)
                last_front_idx = i
            else:
                # This front doesn't fit completely
                last_front_idx = i
                break

        # If we haven't filled the population, process the last front
        if len(new_population) < self._population_size and last_front_idx < len(fronts):
            last_front = fronts[last_front_idx]
            k = len(new_population) + len(last_front) - self._population_size

            # Truncate the last front using hypervolume-based selection
            selected_from_last = self._truncate_population(study, last_front, k)
            new_population.extend(selected_from_last)

        return new_population[: self._population_size]

    def _rank_population(
        self,
        population: list[FrozenTrial],
        directions: Sequence[StudyDirection],
        *,
        is_constrained: bool = False,
    ) -> list[list[FrozenTrial]]:
        """
        Same as optuna.samplers.nsgaii._elite_population_selection_strategy._rank_population in optuna v4.6.
        """
        if len(population) == 0:
            return []

        objective_values = np.array([trial.values for trial in population], dtype=np.float64)
        objective_values *= np.array(
            [-1.0 if d == StudyDirection.MAXIMIZE else 1.0 for d in directions]
        )
        penalty = _evaluate_penalty(population) if is_constrained else None

        domination_ranks = _fast_non_domination_rank(objective_values, penalty=penalty)
        population_per_rank: list[list[FrozenTrial]] = [
            [] for _ in range(max(domination_ranks) + 1)
        ]
        for trial, rank in zip(population, domination_ranks):
            if rank == -1:
                continue
            population_per_rank[rank].append(trial)

        return population_per_rank

    def _truncate_population(
        self,
        study: Study,
        population: list[FrozenTrial],
        k: int,
    ) -> list[FrozenTrial]:
        """Truncate population by removing k individuals using hypervolume estimation.

        This implements the iterative greedy strategy from Algorithm 6 in the paper,
        where individuals are removed one by one based on their I_h^k values.

        Args:
            study: Optuna study object.
            population: Population to truncate.
            k: Number of individuals to remove.

        Returns:
            Truncated population.
        """
        remaining = list(population)
        n_to_select = len(population) - k  # Number of individuals to keep

        # Iteratively remove worst individuals
        for _ in range(k):
            if len(remaining) <= 1:
                break

            # Calculate reference point
            reference_point = self._calculate_reference_point(study, remaining)

            # Estimate I_h^k values for remaining individuals
            # k_fitness represents the number of individuals that will be selected from
            # the remaining population. This is used in the HypE fitness calculation
            # (Equation 12 in the paper) to weight hypervolume contributions.
            k_fitness = len(remaining) - n_to_select + 1
            fitness_values = self._estimate_hypervolume_contributions(
                study, remaining, reference_point, k_fitness
            )

            # Remove individual with minimum fitness (minimum expected hypervolume loss)
            min_idx = np.argmin(fitness_values)
            remaining.pop(min_idx)

        return remaining

    def _calculate_reference_point(
        self, study: Study, population: list[FrozenTrial]
    ) -> np.ndarray:
        """Calculate reference point for hypervolume calculation.

        The reference point is set to be slightly worse than the worst objective
        values in the population (by 10% of the range).

        Args:
            study: Optuna study object.
            population: Current population.

        Returns:
            Reference point as numpy array.
        """
        if not population:
            return np.array([])

        values = np.array([trial.values for trial in population])
        n_objectives = len(study.directions)
        reference_point = np.zeros(n_objectives)

        for i in range(n_objectives):
            max_val = np.max(values[:, i])
            min_val = np.min(values[:, i])
            if study.directions[i] == StudyDirection.MINIMIZE:
                reference_point[i] = max_val + 0.1 * (max_val - min_val)
            else:  # MAXIMIZE
                reference_point[i] = min_val - 0.1 * (max_val - min_val)

        return reference_point

    def _estimate_hypervolume_contributions(
        self,
        study: Study,
        population: list[FrozenTrial],
        reference_point: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Estimate hypervolume contributions using Monte Carlo or Quasi-Monte Carlo sampling.

        This implements Algorithm 3 from the paper, estimating I_h^k(a, P, R)
        for each individual a in the population.

        The sampling method can be either:
        - Monte Carlo (MC): Uses pseudo-random sampling with uniform distribution.
          Convergence rate: O(1/√N)
        - Quasi-Monte Carlo (QMC): Uses Sobol sequence for better space coverage
          and faster convergence. Convergence rate: O(1/N)

        QMC Performance:
            Due to the superior convergence rate, QMC can achieve equivalent accuracy
            with fewer samples than MC. This leads to faster execution time while
            maintaining solution quality. The advantage is more pronounced in
            higher-dimensional objective spaces.

        Args:
            study: Optuna study object.
            population: Population to evaluate.
            reference_point: Reference point for hypervolume calculation.
            k: Number of solutions to be removed (fitness parameter).

        Returns:
            Array of estimated hypervolume contributions.
        """
        n_trials = len(population)
        if n_trials == 0:
            return np.array([])

        values = np.array([trial.values for trial in population])

        # Normalize to minimization problem
        normalized_values = values.copy()
        for i, direction in enumerate(study.directions):
            if direction == StudyDirection.MAXIMIZE:
                normalized_values[:, i] = -normalized_values[:, i]

        normalized_ref = reference_point.copy()
        for i, direction in enumerate(study.directions):
            if direction == StudyDirection.MAXIMIZE:
                normalized_ref[i] = -normalized_ref[i]

        # Calculate sampling box bounds
        lower_bounds = np.min(normalized_values, axis=0)
        upper_bounds = normalized_ref

        # Calculate box volume
        box_volume = np.prod(upper_bounds - lower_bounds)
        if box_volume <= 0 or k <= 0:
            return np.ones(n_trials)

        # Initialize fitness values
        fitness = np.zeros(n_trials)

        # Generate samples using cached Sobol sampler
        n_dimensions = len(lower_bounds)
        if self._sobol_sampler is None or self._sobol_dimensions != n_dimensions:
            self._sobol_sampler = qmc.Sobol(d=n_dimensions, scramble=True)
            self._sobol_dimensions = n_dimensions
        else:
            self._sobol_sampler.reset()
        samples_unit = self._sobol_sampler.random(n=self._n_samples)
        samples = qmc.scale(samples_unit, lower_bounds, upper_bounds)

        # Pre-compute alpha coefficients for all possible dominator counts (1 to k)
        # α_i = Π_{j=1}^{i-1} (k-j)/(|P|-j) where i = len(dominators)
        alpha_cache = np.ones(k + 1)
        for i in range(2, k + 1):
            alpha = 1.0
            for j in range(1, i):
                if n_trials - j != 0:
                    alpha *= (k - j) / (n_trials - j)
            alpha_cache[i] = alpha

        # Process each sample
        for sample in samples:
            # Find which solutions dominate the sample
            dominators = []
            for i in range(n_trials):
                if np.all(normalized_values[i] <= sample):
                    dominators.append(i)

            # Update fitness if hit is in relevant partition (|dominators| <= k)
            n_dominators = len(dominators)
            if 0 < n_dominators <= k:
                # Use pre-computed alpha coefficient
                alpha = alpha_cache[n_dominators]

                # Update fitness for each dominator
                contribution = (alpha / n_dominators) * (box_volume / self._n_samples)
                for idx in dominators:
                    fitness[idx] += contribution

        return fitness

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
        archive_key = f"hype:archive:generation_{generation}"
        trial_numbers = study.user_attrs.get(archive_key, [])

        if not trial_numbers:
            return []

        # Get trials by their trial numbers
        all_trials = study.trials
        trial_dict = {t.number: t for t in all_trials}

        archive = []
        for number in trial_numbers:
            if number in trial_dict:
                archive.append(trial_dict[number])

        return archive
