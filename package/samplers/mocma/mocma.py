from collections.abc import Sequence
import math
from typing import Any

import numpy as np
import optuna
from optuna._hypervolume import compute_hypervolume
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.study._multi_objective import _dominates
from optuna.study._multi_objective import _fast_non_domination_rank
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState


_EPS = 1e-8


class MoCmaSampler(BaseSampler):
    """A sampler based on the Multi-Objective Covariance Matrix Adaptation Evolution Strategy (MO-CMA-ES).

    This implementation provides a strong variant of the MO-CMA-ES algorithm called s-MO-CMA,
    which employs a selection strategy based on the contributing hypervolume (aka S-metric) of each individual.
    For detailed information about MO-CMA-ES algorithm, please refer to the following papers:

    - `Christian Igel, Nikolaus Hansen, Stefan Roth. Covariance Matrix Adaptation for Multi-objective Optimization.
      Evolutionary Computation (2007) 15 (1): 1-28. <https://doi.org/10.1162/evco.2007.15.1.1>`__

    Args:
        search_space:
            A dictionary containing the search space that defines the parameter space.
            The keys are the parameter names and the values are the parameter's distribution.
            If the search space is not provided, the sampler will infer the search space dynamically.
        popsize:
            Population size of the CMA-ES algorithm.
            If not provided, the population size will be set based on the search space dimensionality.
        seed:
            Seed for random number generator.
    """

    def __init__(
        self,
        *,
        search_space: dict[str, BaseDistribution] | None = None,
        popsize: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._rng = LazyRandomState(seed)
        self._independent_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._seed = seed
        self._popsize = popsize
        self._search_space = search_space
        self._intersection_search_space = IntersectionSearchSpace()

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: optuna.study, trial: optuna.trial.FrozenTrial
    ) -> dict[str, BaseDistribution]:
        # If search space information is available (define-and-run)
        if self._search_space is not None:
            return self._search_space

        # Calculate search space dynamically
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if not isinstance(distribution, (FloatDistribution, IntDistribution)):
                # Categorical parameters are handled by the _independend_sampler.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: optuna.study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        # If search space information is not avalable (i.e., first trial & define-by-run)
        if len(search_space) == 0:
            study._storage.set_trial_system_attr(trial._trial_id, "mocma:g", 0)
            if self._popsize is None:
                # If population size information is not available, we set instance number k to 0
                # because we cannot know how many instances exist per generation.
                # This may cause inefficiency in parallelization.
                study._storage.set_trial_system_attr(trial._trial_id, "mocma:k", 0)
            else:
                study._storage.set_trial_system_attr(
                    trial._trial_id,
                    "mocma:k",
                    int(self._rng.rng.choice(len(range(self._popsize)))),
                )
            return {}

        trans = _SearchSpaceTransform(search_space, transform_0_1=True)
        n = len(trans.bounds)  # Search space dimensionality
        # Set population size based on the search space demensionality if not given.
        if self._popsize is None:
            self._popsize = 4 + math.floor(3 * math.log(n))

        # Compute generation g and instance k.
        complete_trials = [
            t for t in study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        ]

        # Classify trials by generation and instance number.
        classified_trials: dict[int, dict] = {0: {}}
        g = 0  # current generation
        for t in complete_trials:
            g_ = t.system_attrs["mocma:g"]
            k_ = t.system_attrs["mocma:k"]
            g = max(g, g_)
            if g_ not in classified_trials:
                classified_trials[g_] = {}
            if k_ not in classified_trials[g_]:
                classified_trials[g_][k_] = []
            classified_trials[g_][k_].append(t)

        generation_finished = True
        ks = []
        for k in range(self._popsize):  # k indicates the (1+1)-CMA-ES instance number.
            if k not in classified_trials[g]:
                generation_finished = False
                ks.append(k)
        if generation_finished:
            g += 1  # Move to the next generation
            ks = list(range(self._popsize))

        # Randomly select an instance number for the current generation.
        # This will enhance the performance when n_jobs > 1.
        k = self._rng.rng.choice(ks)

        study._storage.set_trial_system_attr(trial._trial_id, "mocma:g", g)
        study._storage.set_trial_system_attr(trial._trial_id, "mocma:k", int(k))

        # CMA-ES parameters
        sigma = 1 / 6
        d = 1 + math.floor(n / 2)  # damping parameter
        p_targetsucc = 1 / (5 + 1 / 2)
        c_p = p_targetsucc / (2 + p_targetsucc)
        c_c = 2 / (n + 2)
        c_cov = 2 / (n**2 + 6)
        p_thresh = 0.44

        if g == 0:
            # Generate initial parents randomly.

            return {}  # Fall back to random sampling.
        elif g == 1 and generation_finished:
            # Set parameters for the first generation (g = 0).
            elites = [
                list(sorted(instance, key=lambda x: x.datetime_complete))[0]
                for instance in classified_trials[g - 1].values()
            ]
            for a in elites:
                study._storage.set_study_system_attr(
                    study._study_id, f"mocma:trial:{a._trial_id}:sigma", sigma
                )
                study._storage.set_study_system_attr(
                    study._study_id,
                    f"mocma:trial:{a._trial_id}:cov",
                    np.eye(len(a.params)).tolist(),
                )
                study._storage.set_study_system_attr(
                    study._study_id, f"mocma:trial:{a._trial_id}:p_succ", p_targetsucc
                )
                study._storage.set_study_system_attr(
                    study._study_id,
                    f"mocma:trial:{a._trial_id}:p_c",
                    np.zeros(len(a.params)).tolist(),
                )
            elite_ids = [e._trial_id for e in elites]
            study._storage.set_study_system_attr(
                study._study_id, f"mocma:generation:{g-1}:elite_ids", elite_ids
            )
        elif g >= 2 and generation_finished:
            # This section conducts the parameter updates for g-1 with individuals for g-1 and g-2
            # before generating individuals for generation g.
            study_system_attrs = study._storage.get_study_system_attrs(study._study_id)

            parents = [
                [t for t in complete_trials if t._trial_id == eid][0]
                for eid in study_system_attrs[f"mocma:generation:{g-2}:elite_ids"]
            ]
            # Handling conditional parameters for parents
            # (Discard cma parmeter values for paramaters not in the intersection search space)
            for a in parents:
                indices = [i for i, n in enumerate(a.params) if n in search_space]
                a.params = {n: a.params[n] for n in search_space}
                p_c_a = np.asarray(study_system_attrs[f"mocma:trial:{a._trial_id}:p_c"])
                p_c_a = p_c_a[indices]
                study._storage.set_study_system_attr(
                    study._study_id, f"mocma:trial:{a._trial_id}:p_c", p_c_a.tolist()
                )

            offsprings = [
                list(sorted(instance, key=lambda x: x.datetime_complete))[0]
                for instance in classified_trials[g - 1].values()
            ]
            # Handling conditional parameters for offsprings
            for a_ in offsprings:
                a_.params = {n: a_.params[n] for n in search_space}

            # Reload study_system_attrs
            study_system_attrs = study._storage.get_study_system_attrs(study._study_id)

            for a_ in offsprings:
                # Find parent a for a_
                a = [p for p in parents if p._trial_id == a_.system_attrs["mocma:parent_id"]][0]
                lambda_succ = int(_dominates(a_, a, study.directions))

                # Update parent step size
                p_succ_a = study_system_attrs[f"mocma:trial:{a._trial_id}:p_succ"]
                p_succ_a = (1 - c_p) * p_succ_a + c_p * lambda_succ
                sigma_a = study_system_attrs[f"mocma:trial:{a._trial_id}:sigma"]
                sigma_a = sigma_a * math.exp(
                    (1 / d) * ((p_succ_a - p_targetsucc) / (1 - p_targetsucc))
                )
                sigma_a = max(sigma_a, _EPS)
                study._storage.set_study_system_attr(
                    study._study_id, f"mocma:trial:{a._trial_id}:p_succ", p_succ_a
                )
                study._storage.set_study_system_attr(
                    study._study_id, f"mocma:trial:{a._trial_id}:sigma", sigma_a
                )

                # Update offspring step size and covariance matrix
                p_succ_a_ = np.asarray(study_system_attrs[f"mocma:trial:{a_._trial_id}:p_succ"])
                p_succ_a_ = (1 - c_p) * p_succ_a_ + c_p * lambda_succ
                sigma_a_ = study_system_attrs[f"mocma:trial:{a_._trial_id}:sigma"]
                sigma_a_ = sigma_a_ * math.exp(
                    (1 / d) * ((p_succ_a_ - p_targetsucc) / (1 - p_targetsucc))
                )
                sigma_a_ = max(sigma_a_, _EPS)
                cov_a_ = np.asarray(study_system_attrs[f"mocma:trial:{a_._trial_id}:cov"])
                p_c = np.asarray(study_system_attrs[f"mocma:trial:{a._trial_id}:p_c"])
                if p_succ_a_ < p_thresh:
                    values_a_ = np.asarray(list(a_.params.values()))
                    values_a = np.asarray(list(a.params.values()))
                    x_step = (values_a_ - values_a) / sigma_a
                    p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c)) * x_step
                    cov_a_ = (1 - c_cov) * cov_a_ + c_cov * p_c @ p_c.T
                else:
                    p_c = (1 - c_c) * p_c
                    cov_a_ = (1 - c_cov) * cov_a_ + c_cov * (
                        p_c @ p_c.T + c_c * (2 - c_c) * cov_a_
                    )

                study._storage.set_study_system_attr(
                    study._study_id, f"mocma:trial:{a_._trial_id}:sigma", float(sigma_a_)
                )
                study._storage.set_study_system_attr(
                    study._study_id, f"mocma:trial:{a_._trial_id}:cov", cov_a_.tolist()
                )
                study._storage.set_study_system_attr(
                    study._study_id, f"mocma:trial:{a_._trial_id}:p_succ", float(p_succ_a_)
                )
                study._storage.set_study_system_attr(
                    study._study_id, f"mocma:trial:{a_._trial_id}:p_c", p_c.tolist()
                )

            # Selecting elites
            population = np.asarray(parents + offsprings)
            objective_values = np.asarray([i.values for i in population])
            non_domination_ranks = _fast_non_domination_rank(
                objective_values, n_below=self._popsize
            )
            elites = []
            for i in range(len(population)):
                # Selection based on non-dmination ranks
                front_i = population[non_domination_ranks == i].tolist()
                if len(elites) + len(front_i) <= self._popsize:
                    elites += front_i
                    continue

                # Optuna's hypervolume module assumes minimization
                rank_i_vals = np.asarray(
                    [
                        [
                            v if d == StudyDirection.MINIMIZE else -v
                            for v, d in zip(vs, study.directions)
                        ]
                        for vs in objective_values[non_domination_ranks == i]
                    ]
                )
                worst_point = np.max(rank_i_vals, axis=0)
                reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
                reference_point[reference_point == 0] = _EPS

                # Selection based on hypervolume contributions
                while len(elites) < self._popsize:
                    hypervolume = compute_hypervolume(rank_i_vals, reference_point)
                    contribution_scores = [
                        hypervolume
                        - compute_hypervolume(
                            np.concatenate([rank_i_vals[:j], rank_i_vals[j + 1 :]], axis=0),
                            reference_point,
                            assume_pareto=True,
                        )
                        for j in range(len(rank_i_vals))
                    ]  # Smaller is better

                    candidate = np.argmin(contribution_scores)
                    elites.append(front_i[candidate])

                    # Remove selected candidate
                    rank_i_vals = np.delete(rank_i_vals, candidate, axis=0)
                    del front_i[candidate]
            elite_ids = [e._trial_id for e in elites]
            study._storage.set_study_system_attr(
                study._study_id, f"mocma:generation:{g-1}:elite_ids", elite_ids
            )

        # Load/reload study_system_attrs after updates
        study_system_attrs = study._storage.get_study_system_attrs(study._study_id)

        # Generate individual for generation g and instance k
        a = [
            t
            for t in complete_trials
            if t._trial_id == study_system_attrs[f"mocma:generation:{g-1}:elite_ids"][k]
        ][0]
        mean = trans.transform(a.params)
        sigma = study_system_attrs[f"mocma:trial:{a._trial_id}:sigma"]
        cov = np.asarray(study_system_attrs[f"mocma:trial:{a._trial_id}:cov"])
        p_c = np.asarray(study_system_attrs[f"mocma:trial:{a._trial_id}:p_c"])
        p_succ = study_system_attrs[f"mocma:trial:{a._trial_id}:p_succ"]

        # Handling conditional parameters
        # (Discard cma parmeter values for paramaters not in the intersection search space)
        indices = np.asarray([i for i, n in enumerate(a.params) if n in search_space])
        cov = cov[np.ix_(indices, indices)]
        p_c = p_c[indices]

        study._storage.set_trial_system_attr(trial._trial_id, "mocma:parent_id", a._trial_id)
        study._storage.set_study_system_attr(
            study._study_id, f"mocma:trial:{trial._trial_id}:sigma", float(sigma)
        )
        study._storage.set_study_system_attr(
            study._study_id, f"mocma:trial:{trial._trial_id}:cov", cov.tolist()
        )
        study._storage.set_study_system_attr(
            study._study_id, f"mocma:trial:{trial._trial_id}:p_succ", float(p_succ)
        )
        study._storage.set_study_system_attr(
            study._study_id, f"mocma:trial:{trial._trial_id}:p_c", p_c.tolist()
        )

        x = np.clip(
            self._rng.rng.multivariate_normal(mean, sigma**2 * cov),
            a_min=trans.bounds[:, 0],
            a_max=trans.bounds[:, 1],
        )
        external_values = trans.untransform(x)
        for pn, pv in search_space.items():
            external_values[pn] = np.clip(
                external_values[pn], search_space[pn].low, search_space[pn].high
            )
            if isinstance(pv, IntDistribution):
                external_values[pn] = int(external_values[pn])
            elif isinstance(pv, FloatDistribution):
                external_values[pn] = float(external_values[pn])

        return external_values

    def sample_independent(
        self,
        study: optuna.study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: optuna.study,
        trial: optuna.trial.FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
