from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState
import optunahub


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Dict
    from typing import List
    from typing import Optional
    from typing import Union

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class Particle:
    """Encapsulates a single particle and its PSO update logic."""

    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        # Store position/velocity as floats and copy to avoid external aliasing.
        self.position: np.ndarray = position.astype(float).copy()
        self.velocity: np.ndarray = velocity.astype(float).copy()

        # Personal best state.
        self.pbest_position: np.ndarray = self.position.copy()
        self.pbest_score: float = np.inf

    def set_position(
        self,
        position: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> None:
        """Set the current position, clipped to bounds."""
        self.position = np.clip(position.astype(float), lower_bound, upper_bound)

    def update_personal_best(self, fitness: float) -> None:
        """Update personal best if the new fitness is better (smaller)."""
        if fitness < self.pbest_score:
            self.pbest_score = float(fitness)
            self.pbest_position = self.position.copy()

    def step(
        self,
        gbest_position: np.ndarray,
        inertia: float,
        cognitive: float,
        social: float,
        rng: np.random.RandomState,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        v_max: np.ndarray,
    ) -> None:
        """
        Perform one PSO step:
          v = inertia * v + cognitive * r1 * (pbest - x) + social * r2 * (gbest - x)
          x = x + v
        Velocity is clamped to [-v_max, v_max] and positions are clipped to [lower, upper].
        Velocity components that hit the boundary are zeroed to reduce bouncing.
        """
        dim = self.position.size
        r1: np.ndarray = rng.rand(dim)
        r2: np.ndarray = rng.rand(dim)

        cognitive_term: np.ndarray = cognitive * r1 * (self.pbest_position - self.position)
        social_term: np.ndarray = social * r2 * (gbest_position - self.position)

        # Update velocity and clamp.
        self.velocity = inertia * self.velocity + cognitive_term + social_term
        self.velocity = np.clip(self.velocity, -v_max, v_max)

        # Update position and clip to bounds.
        new_pos: np.ndarray = self.position + self.velocity
        new_pos_clipped: np.ndarray = np.clip(new_pos, lower_bound, upper_bound)

        # Zero velocities where we hit bounds to avoid oscillation at edges.
        hit_mask: np.ndarray = (new_pos_clipped <= lower_bound) | (new_pos_clipped >= upper_bound)
        self.velocity[hit_mask] = 0.0

        self.position = new_pos_clipped


class PSOSampler(optunahub.samplers.SimpleBaseSampler):
    """
    Particle Swarm Optimization (PSO) sampler using Optuna's SimpleBaseSampler.

    Key behavior:
    - First generation: returns {} from sample_relative so Optuna's RandomSampler initializes the population.
    - After each trial (in after_trial), we accumulate COMPLETE results. When n_particles results
      are collected, update the swarm and compute the next generation candidates.
    - Supports numeric (Float/Int) distributions. Categorical distributions are suggested through RandomSampler.
    """

    def __init__(
        self,
        *,
        search_space: Optional[Dict[str, BaseDistribution]] = None,
        n_particles: int = 10,
        inertia: float = 0.5,
        cognitive: float = 1.5,
        social: float = 1.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(search_space, seed)
        if n_particles <= 0:
            raise ValueError("n_particles must be > 0.")
        if inertia < 0.0:
            raise ValueError("inertia must be >= 0.0.")
        if cognitive < 0.0 or social < 0.0:
            raise ValueError("cognitive and social must be >= 0.0.")

        self.search_space = search_space
        self.n_particles: int = n_particles
        self.inertia: float = inertia
        self.cognitive: float = cognitive
        self.social: float = social

        self._rng: np.random.RandomState = np.random.RandomState(seed)

        # Search space metadata (initialized lazily).
        self._initialized: bool = False
        self.dim: int = 0
        # Numeric-only names used for PSO vectorization.
        self.param_names: List[str] = []  # numeric param names
        self._numeric_dists: Dict[str, BaseDistribution] = {}
        self.lower_bound: np.ndarray = np.array([], dtype=float)
        self.upper_bound: np.ndarray = np.array([], dtype=float)
        self.v_max: np.ndarray = np.array([], dtype=float)

        # Swarm state.
        self.particles: List[Particle] = []
        self.gbest_position: Optional[np.ndarray] = None
        self.gbest_score: float = np.inf

        # Accumulators for the current generation (filled in after_trial).
        self._acc_positions: List[np.ndarray] = []
        self._acc_fitness: List[float] = []

        # Precomputed candidates for the next generation and the serving pointer.
        self._next_candidates: List[Dict[str, Union[int, float]]] = []
        self._next_index: int = 0

    def _lazy_init(self, search_space: Dict[str, BaseDistribution]) -> None:
        """Initialize internal state based on the current search space (numeric-only for PSO)."""
        # Split numeric vs. categorical distributions.
        self.param_names = []

        self._numeric_dists = {
            name: dist
            for name, dist in search_space.items()
            if isinstance(
                dist,
                (optuna.distributions.FloatDistribution, optuna.distributions.IntDistribution),
            )
            and not dist.single()
        }

        self.param_names = sorted(self._numeric_dists.keys())
        self.dim = len(self.param_names)

        if self.dim > 0:
            self.lower_bound = np.array(
                [self._numeric_dists[n].low for n in self.param_names], dtype=float
            )
            self.upper_bound = np.array(
                [self._numeric_dists[n].high for n in self.param_names], dtype=float
            )
            self.v_max = (self.upper_bound - self.lower_bound).astype(float)
        else:
            # No numeric params -> PSO operates on 0-D; RandomSampler will handle all params.
            self.lower_bound = np.array([], dtype=float)
            self.upper_bound = np.array([], dtype=float)
            self.v_max = np.array([], dtype=float)

        # Reset dynamic state.
        self.particles = []
        self.gbest_position = None
        self.gbest_score = np.inf
        self._acc_positions.clear()
        self._acc_fitness.clear()
        self._next_candidates.clear()
        self._next_index = 0

        self._initialized = True

    def infer_relative_search_space(
        self, study: Study, _: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        if self.search_space is not None:
            return self.search_space

        inferred = self._intersection_search_space.calculate(study)

        numeric = {
            n: d
            for n, d in inferred.items()
            if not d.single()
            and isinstance(
                d, (optuna.distributions.FloatDistribution, optuna.distributions.IntDistribution)
            )
        }

        if numeric:
            self.search_space = numeric

        return numeric

    def sample_relative(
        self,
        study: Study,
        _: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Union[int, float]]:
        """
        Provide parameters for the next trial.
        - If we have precomputed candidates, serve them in order (one per call).
        - Otherwise, return {} to delegate sampling to Optuna's RandomSampler.
        """
        self._ensure_single_objective(study)

        if len(search_space) == 0:
            return {}

        if not self._initialized:
            self._lazy_init(search_space)

        # Serve next precomputed numeric candidate if available.
        if self._next_index < len(self._next_candidates):
            params = self._next_candidates[self._next_index]
            self._next_index += 1
            # Note: Only numeric params are returned; categorical params will be sampled independently.
            return params

        # No precomputed numeric candidates -> delegate to RandomSampler for all params.
        return {}

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        """
        Observe the trial outcome and update PSO state at generation boundaries.
        This method:
        - Accumulates COMPLETE trials into the current generation buffer.
        - When n_particles results are collected, updates/initializes the swarm,
          advances one PSO step, and precomputes the next generation candidates.
        """
        # Prevent multi-objective usage
        self._ensure_single_objective(study, values)

        # Use only COMPLETE trials to build generations.
        if state is not TrialState.COMPLETE:
            return

        # Ensure we were initialized by sample_relative at least once.
        if not self._initialized:
            return

        # Extract numeric fitness (lower is better; flip sign if maximizing).
        if values is None or len(values) == 0:
            return  # Defensive: nothing to do if no value present.
        raw_value = float(values[0])
        fitness = raw_value if study.direction == StudyDirection.MINIMIZE else -raw_value

        # Vectorize the trial's parameters and clip to bounds.
        x = self._encode_trial_params(trial)

        # Accumulate this result.
        self._acc_positions.append(x)
        self._acc_fitness.append(fitness)

        # When we have a full generation of COMPLETE results, update the swarm.
        if len(self._acc_positions) == self.n_particles:
            X_gen = np.vstack(self._acc_positions).astype(float)
            f_gen = np.array(self._acc_fitness, dtype=float)

            if not self.particles:
                # Initialize particles from the first completed generation.
                self._initialize_swarm(X_gen, f_gen)
            else:
                # Update particle positions and personal bests with the latest results.
                self._update_personal_bests(X_gen, f_gen)

            # Update global best across particles.
            self._update_global_best()

            # Advance swarm by one PSO step.
            assert self.gbest_position is not None
            for p in self.particles:
                p.step(
                    gbest_position=self.gbest_position,
                    inertia=self.inertia,
                    cognitive=self.cognitive,
                    social=self.social,
                    rng=self._rng,
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                    v_max=self.v_max,
                )

            # Precompute next generation candidates for sample_relative to serve.
            self._prepare_next_candidates()

            # Reset accumulators for the next generation.
            self._acc_positions.clear()
            self._acc_fitness.clear()

    def reseed_rng(self) -> None:
        """Reseed both Optuna's RandomSampler and this sampler's RNG."""
        super().reseed_rng()
        self._rng = np.random.RandomState()

    # -------------------- Internal helpers --------------------

    def _ensure_single_objective(
        self,
        study: Study,
        values: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Guard against multi-objective usage.
        - Uses the public Study.directions API: length > 1 means multi-objective.
        - Also checks the provided `values` (if any) from after_trial; length > 1 means multi-objective.
        """
        directions = getattr(study, "directions", None)
        if directions is not None and len(directions) != 1:
            raise NotImplementedError("PSOSampler does not support multi-objective studies.")
        if values is not None and len(values) != 1:
            raise NotImplementedError("PSOSampler does not support multi-objective studies.")

    def _encode_trial_params(self, trial: FrozenTrial) -> np.ndarray:
        """Vectorize numeric params from a trial in the fixed numeric order and clip to bounds."""
        vec = np.empty(self.dim, dtype=float)
        for i, name in enumerate(self.param_names):
            vec[i] = float(trial.params.get(name, self.lower_bound[i]))
        return np.clip(vec, self.lower_bound, self.upper_bound)

    def _initialize_swarm(self, X_gen: np.ndarray, f_gen: np.ndarray) -> None:
        """Initialize the swarm from the first completed generation."""
        self.particles = []
        for i in range(self.n_particles):
            pos = X_gen[i].copy()
            # Initialize velocity uniformly within [-v_max, v_max] for each dimension.
            vel = self._rng.uniform(low=-self.v_max, high=self.v_max, size=self.dim)
            particle = Particle(position=pos, velocity=vel)
            # Initialize pbest with current fitness.
            particle.update_personal_best(f_gen[i])
            self.particles.append(particle)

    def _update_personal_bests(self, X_gen: np.ndarray, f_gen: np.ndarray) -> None:
        """Set current positions and update each particle's personal best with new fitness."""
        for i in range(self.n_particles):
            p = self.particles[i]
            p.set_position(X_gen[i], self.lower_bound, self.upper_bound)
            p.update_personal_best(f_gen[i])

    def _update_global_best(self) -> None:
        """Find the best personal best among all particles."""
        best_idx = int(np.argmin([p.pbest_score for p in self.particles]))
        best = self.particles[best_idx]
        self.gbest_position = best.pbest_position.copy()
        self.gbest_score = float(best.pbest_score)

    def _prepare_next_candidates(self) -> None:
        """Convert current numeric particle positions to parameter dicts and reset the serve pointer."""
        self._next_candidates = [
            self._decode_position_to_params(p.position) for p in self.particles
        ]
        self._next_index = 0

    def _decode_position_to_params(self, x: np.ndarray) -> Dict[str, Union[int, float]]:
        """
        Convert a numeric position vector into a parameter dict containing only numeric params.
        Categorical params are intentionally omitted so Optuna's RandomSampler will sample them.
        """
        params: Dict[str, Union[int, float]] = {}
        for i, name in enumerate(self.param_names):
            dist = self._numeric_dists[name]
            lo = float(self.lower_bound[i])
            hi = float(self.upper_bound[i])
            xi = float(np.clip(x[i], lo, hi))

            val: Union[int, float]
            if isinstance(dist, optuna.distributions.IntDistribution):
                step_attr = getattr(dist, "step", None)
                step = int(step_attr) if step_attr is not None else 1
                snapped = lo + round((xi - lo) / step) * step
                val = int(np.clip(snapped, lo, hi))
            else:
                val = float(xi)

            params[name] = val

        # Note: No categorical params here. They will be sampled independently by RandomSampler.
        return params
