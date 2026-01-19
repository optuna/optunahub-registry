from __future__ import annotations

from typing import Literal
from typing import Sequence
from typing import TYPE_CHECKING
import warnings

import numpy as np
import numpy.typing as npt
from optuna._hypervolume import compute_hypervolume
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
        seed: int | None = None,
        hypervolume_method: Literal["auto", "exact", "estimation"] = "auto",
    ) -> None:
        """Initialize HypE elite selection strategy.

        Args:
            population_size: Size of the population.
            n_samples: Number of samples for hypervolume estimation.
            seed: Random seed for Sobol sequence generator.
            hypervolume_method: Method for hypervolume contribution calculation.
                If "auto", "exact" is used when the number of objectives
                is 3 or less, and "estimation" is used otherwise, following the
                original HypE paper.
                If "exact", exact hypervolume calculation is always used.
                If "estimation", Monte Carlo estimation is always used.
        """
        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        if n_samples > 0 and (n_samples & (n_samples - 1)) != 0:
            warnings.warn(
                f"`n_samples` ({n_samples}) is not a power of 2. "
                "Sobol sequence works most effectively with powers of 2 (e.g., 1024, 2048, 4096). "
                "Consider using a power of 2 for better QMC performance.",
            )
        if n_samples < 1:
            raise ValueError("`n_samples` must be greater than or equal to 1.")

        self._population_size = population_size
        self._n_samples = n_samples
        self._sobol_sampler: qmc.Sobol | None = None
        self._sobol_dimensions: int = 0
        self._seed = seed
        self._hypervolume_method = hypervolume_method

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
        population_par_rank = self._rank_population(population, study.directions)

        elite_population: list[FrozenTrial] = []

        for individuals in population_par_rank:
            if len(elite_population) + len(individuals) <= self._population_size:
                elite_population.extend(individuals)
            else:
                n_truncate = len(elite_population) + len(individuals) - self._population_size
                remaining_individuals = self._truncate_by_hypervolume_contributions(
                    study, individuals, n_truncate
                )
                elite_population.extend(remaining_individuals)
                break

        return elite_population

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

    def _truncate_by_hypervolume_contributions(
        self,
        study: Study,
        population: list[FrozenTrial],
        n_truncate: int,
    ) -> list[FrozenTrial]:
        """Truncate last non-fitting nondominated population by removing k individuals using hypervolume."""
        remaining_individuals = list(population)
        n_select = len(population) - n_truncate
        reference_point = self._calculate_reference_point(study, remaining_individuals)

        for _ in range(n_truncate):
            if len(remaining_individuals) <= 1:
                break

            use_exact = self._should_use_exact_hypervolume(len(study.directions))
            if use_exact:
                fitness_values = self._compute_exact_hypervolume_contributions(
                    study, remaining_individuals, reference_point, n_select
                )
            else:
                fitness_values = self._estimate_hypervolume_contributions(
                    study, remaining_individuals, reference_point, n_select
                )

            # Remove individual with minimum fitness (minimum expected hypervolume loss)
            min_idx = np.argmin(fitness_values)
            remaining_individuals.pop(min_idx)

        return remaining_individuals

    def _should_use_exact_hypervolume(self, n_objectives: int) -> bool:
        if self._hypervolume_method == "auto":
            # Follow the original HypE paper: use exact for 3 or fewer objectives
            return n_objectives <= 3
        return self._hypervolume_method == "exact"

    def _calculate_reference_point(
        self, study: Study, population: list[FrozenTrial]
    ) -> npt.NDArray[np.float64]:
        """Calculate reference point for hypervolume calculation.

        The reference point is set to be slightly worse than the worst objective
        values in the population (by 10% of the range).
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
            else:
                reference_point[i] = min_val - 0.1 * (max_val - min_val)

        return reference_point

    def _compute_exact_hypervolume_contributions(
        self,
        study: Study,
        population: list[FrozenTrial],
        reference_point: npt.NDArray[np.float64],
        k: int,
    ) -> npt.NDArray[np.float64]:
        """Compute exact hypervolume contributions for each individual."""
        n_population = len(population)
        values = np.array([trial.values for trial in population])

        # Normalize to minimization problem
        normalized_values = values.copy()
        normalized_ref = reference_point.copy()
        for i, direction in enumerate(study.directions):
            if direction == StudyDirection.MAXIMIZE:
                normalized_values[:, i] = -normalized_values[:, i]
                normalized_ref[i] = -normalized_ref[i]

        total_hypervolume = compute_hypervolume(normalized_values, normalized_ref)

        # Compute hypervolume contribution for each individual
        contributions = np.zeros(n_population)
        for i in range(n_population):
            remaining_values = np.concatenate(
                [normalized_values[:i], normalized_values[i + 1 :]], axis=0
            )
            if len(remaining_values) > 0:
                hv_without_i = compute_hypervolume(remaining_values, normalized_ref)
            else:
                hv_without_i = 0.0

            contributions[i] = total_hypervolume - hv_without_i

        return contributions

    def _estimate_hypervolume_contributions(
        self,
        study: Study,
        population: list[FrozenTrial],
        reference_point: npt.NDArray[np.float64],
        fitness_param: int,
    ) -> npt.NDArray[np.float64]:
        """Estimate hypervolume contributions using Monte Carlo or Quasi-Monte Carlo sampling.

        This implements Algorithm 3 from the paper, estimating I_h^k(a, P, R)
        for each individual a in the population.
        """
        n_population = len(population)
        if n_population == 0:
            raise ValueError("Population must contain at least one trial.")

        values = np.array([trial.values for trial in population])

        # Normalize to minimization problem
        normalized_values = values.copy()
        normalized_ref = reference_point.copy()
        for i, direction in enumerate(study.directions):
            if direction == StudyDirection.MAXIMIZE:
                normalized_values[:, i] = -normalized_values[:, i]
                normalized_ref[i] = -normalized_ref[i]

        # determine sampling box S
        lower_bounds = np.min(normalized_values, axis=0)
        upper_bounds = normalized_ref
        box_volume = np.prod(upper_bounds - lower_bounds)
        if box_volume <= 0 or fitness_param <= 0:
            return np.ones(n_population)

        # Initialize fitness values
        hv_estimates = np.zeros(n_population)

        # perform sampling
        sampling_points = self._generate_sampling_points(lower_bounds, upper_bounds)
        for sample in sampling_points:
            dominators = []
            for i in range(n_population):
                if np.all(normalized_values[i] <= sample):
                    dominators.append(i)

            # hit in a relevant partition (|UP| <= k)
            n_dominators = len(dominators)
            if 0 < n_dominators <= fitness_param:
                alpha = 1.0
                for j in range(1, n_dominators):
                    if n_population - j != 0:
                        alpha *= (fitness_param - j) / (n_population - j)

                # Update hypervolume estimates for each dominator
                contribution = (alpha / n_dominators) * (box_volume / self._n_samples)
                for idx in dominators:
                    hv_estimates[idx] += contribution

        return hv_estimates

    def _generate_sampling_points(
        self,
        lower_bounds: npt.NDArray[np.float64],
        upper_bounds: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate sampling points using Sobol sequence for hypervolume estimation."""
        n_dimensions = len(lower_bounds)
        if self._sobol_sampler is None or self._sobol_dimensions != n_dimensions:
            self._sobol_sampler = qmc.Sobol(d=n_dimensions, scramble=True, seed=self._seed)
            self._sobol_dimensions = n_dimensions
        else:
            self._sobol_sampler.reset()
        samples_unit = self._sobol_sampler.random(n=self._n_samples)
        sampling_points = qmc.scale(samples_unit, lower_bounds, upper_bounds)

        return sampling_points
