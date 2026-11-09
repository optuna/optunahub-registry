from __future__ import annotations

from copy import copy
import inspect
import json
from typing import Any
from typing import Callable
from typing import Sequence

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.samplers._tpe.sampler import default_gamma
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


# Optuna v5 removed the ``constraints_enabled`` argument of ``_split_trials``.
_SPLIT_TRIALS_TAKES_CONSTRAINTS_ENABLED = (
    "constraints_enabled" in inspect.signature(_split_trials).parameters
)
# Use TPE's format so the target sampler can also read pending relative parameters.
_RELATIVE_PARAMS_KEY = "tpe:relative_params"
_SYSTEM_ATTR_MAX_LENGTH = 2045


class MetaLearnTPESampler(BaseSampler):
    """A TPE sampler with meta-learning from previous studies.

    This sampler accelerates optimization by leveraging knowledge from
    previously completed Optuna studies on related tasks. It computes
    task similarity based on the overlap of promising regions and uses
    weighted mixture of TPE models across tasks.

    Based on the algorithm in:
    Watanabe et al., "Speeding Up Multi-Objective Hyperparameter Optimization
    by Task Similarity-Based Meta-Learning for the Tree-Structured Parzen
    Estimator", IJCAI 2023.

    Args:
        source_studies:
            A sequence of completed Optuna studies on related tasks.
            These studies should share the same or similar search spaces.
        n_startup_trials:
            Number of trials before meta-learning kicks in. During startup,
            the sampler falls back to standard TPE.
            With group=True, the threshold applies to matching completed target
            observations in each subspace.
        seed:
            Random seed for reproducibility.
        n_ei_candidates:
            Number of candidates sampled from each task's below distribution.
        constant_liar:
            If True, penalize parameters of running target trials to avoid
            proposing similar parameters to concurrent workers. Defaults to False.
            Source studies continue to use completed trials only.
        group:
            If True, decompose a conditional search space and perform meta-learning
            separately for each subspace, using trials containing its parameters.
            Defaults to False.
    """

    def __init__(
        self,
        *,
        source_studies: Sequence[Study],
        n_startup_trials: int = 10,
        seed: int | None = None,
        n_ei_candidates: int = 24,
        constant_liar: bool = False,
        group: bool = False,
    ) -> None:
        """Initialize MetaLearnTPESampler with source studies for knowledge transfer."""
        if len(source_studies) == 0:
            raise ValueError("source_studies must contain at least one study.")

        self._source_studies = list(source_studies)
        self._n_startup_trials = n_startup_trials
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._n_ei_candidates = n_ei_candidates
        self._constant_liar = constant_liar
        self._group = group

        # Build internal TPE samplers for source studies.
        self._source_samplers: list[TPESampler] = []
        for i in range(len(source_studies)):
            self._source_samplers.append(
                TPESampler(
                    seed=seed + i + 1 if seed is not None else None,
                    multivariate=True,
                    n_startup_trials=0,
                    n_ei_candidates=n_ei_candidates,
                )
            )

        # Target task sampler.
        self._target_sampler = TPESampler(
            seed=seed,
            multivariate=True,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            constant_liar=constant_liar,
            group=group,
        )

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        """Infer the search space for relative sampling, delegated to the target TPE sampler."""
        return self._target_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        """Sample parameters using weighted mixture of target and source TPE models."""
        if self._group:
            search_space_group = self._target_sampler._search_space_group
            assert search_space_group is not None
            params = {}
            for subspace in search_space_group.search_spaces:
                # Match TPE's ordering and respect restricted relative search spaces.
                subspace = {
                    name: distribution
                    for name, distribution in sorted(subspace.items())
                    if name in search_space and not distribution.single()
                }
                params.update(self._sample_relative(study, trial, subspace))
        else:
            params = self._sample_relative(study, trial, search_space)
        if params and self._constant_liar:
            # Other workers may need these values before suggest_* stores them.
            params_str = json.dumps(params)
            for i in range(0, len(params_str), _SYSTEM_ATTR_MAX_LENGTH):
                study._storage.set_trial_system_attr(
                    trial._trial_id,
                    f"{_RELATIVE_PARAMS_KEY}:{i // _SYSTEM_ATTR_MAX_LENGTH}",
                    params_str[i : i + _SYSTEM_ATTR_MAX_LENGTH],
                )
        return params

    def _sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if len(search_space) == 0:
            return {}

        states = (
            (TrialState.COMPLETE, TrialState.RUNNING)
            if self._constant_liar
            else (TrialState.COMPLETE,)
        )
        target_trials = study._get_trials(
            deepcopy=False, states=states, use_cache=not self._constant_liar
        )
        if self._constant_liar:
            target_trials = [
                _with_running_params(t, search_space) if t.state == TrialState.RUNNING else t
                for t in target_trials
                if t.number != trial.number
            ]
            # A running trial cannot penalize the joint model without all its parameters.
            target_trials = [
                t
                for t in target_trials
                if t.state != TrialState.RUNNING or search_space.keys() <= t.params.keys()
            ]
        if self._group:
            target_trials = [t for t in target_trials if search_space.keys() <= t.params.keys()]
        n_complete = sum(t.state == TrialState.COMPLETE for t in target_trials)

        # Before enough trials, fall back to the target TPE sampler.
        if n_complete < self._n_startup_trials or (self._group and n_complete == 0):
            return self._target_sampler._sample_relative(study, trial, search_space)

        # Build Parzen estimators for target task.
        target_below, target_above = self._build_parzen_estimators(
            study, search_space, target_trials
        )

        # Build Parzen estimators for each source task.
        source_estimators: list[tuple[Any, Any, tuple[int, int]]] = []
        for source_study, source_sampler in zip(self._source_studies, self._source_samplers):
            source_trials = source_study._get_trials(
                deepcopy=False,
                states=(TrialState.COMPLETE,),
                use_cache=True,
            )
            if self._group:
                source_trials = [
                    t for t in source_trials if search_space.keys() <= t.params.keys()
                ]
            if len(source_trials) == 0:
                continue
            s_below, s_above = self._build_parzen_estimators(
                source_study, search_space, source_trials, sampler=source_sampler
            )
            n_below = max(1, int(_get_gamma(source_sampler, source_study)(len(source_trials))))
            source_estimators.append((s_below, s_above, (n_below, len(source_trials) - n_below)))

        if len(source_estimators) == 0:
            return self._target_sampler._sample_relative(study, trial, search_space)

        # Compute task similarities and weights.
        similarities = self._compute_task_similarities(target_below, source_estimators)
        task_weights = self._compute_task_weights(similarities)

        # Sample candidates from all below estimators.
        all_samples: dict[str, list[np.ndarray]] = {p: [] for p in search_space}

        target_candidates = target_below.sample(self._rng, self._n_ei_candidates)
        for p in search_space:
            all_samples[p].append(target_candidates[p])

        for s_below, _, _ in source_estimators:
            s_candidates = s_below.sample(self._rng, self._n_ei_candidates)
            for p in search_space:
                all_samples[p].append(s_candidates[p])

        samples = {p: np.hstack(arrs) for p, arrs in all_samples.items()}

        # Compute weighted acquisition function values.
        n_tasks = 1 + len(source_estimators)
        n_cands = samples[list(search_space.keys())[0]].shape[0]

        ll_below = np.zeros((n_tasks, n_cands))
        ll_above = np.zeros((n_tasks, n_cands))
        n_samples_below = np.zeros(n_tasks)
        n_samples_above = np.zeros(n_tasks)

        # Target task.
        ll_below[0] = target_below.log_pdf(samples)
        ll_above[0] = target_above.log_pdf(samples)
        gamma_t = max(1, int(_get_gamma(self._target_sampler, study)(n_complete)))
        n_samples_below[0] = gamma_t
        n_samples_above[0] = len(target_trials) - gamma_t

        # Source tasks.
        for i, (s_below, s_above, (n_below, n_above)) in enumerate(source_estimators):
            ll_below[i + 1] = s_below.log_pdf(samples)
            ll_above[i + 1] = s_above.log_pdf(samples)
            n_samples_below[i + 1] = n_below
            n_samples_above[i + 1] = n_above

        # Weighted mixture of likelihoods (log-sum-exp for numerical stability).
        w_below = task_weights * n_samples_below
        w_above = task_weights * n_samples_above
        w_below_norm = w_below / (w_below.sum() + 1e-12)
        w_above_norm = w_above / (w_above.sum() + 1e-12)

        # log(sum_i w_i * exp(ll_i)) using log-sum-exp trick.
        log_l = _log_weighted_sum_exp(w_below_norm, ll_below)
        log_g = _log_weighted_sum_exp(w_above_norm, ll_above)
        acq_values = log_l - log_g

        ret = TPESampler._compare(samples, acq_values)
        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])
        return ret

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample a parameter independently, delegated to the target TPE sampler."""
        return self._target_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _build_parzen_estimators(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        trials: list[FrozenTrial],
        sampler: TPESampler | None = None,
    ) -> tuple[Any, Any]:
        """Build below/above Parzen estimators for the given trials."""
        if sampler is None:
            sampler = self._target_sampler

        n_finished = sum(t.state != TrialState.RUNNING for t in trials)
        n_below = max(1, int(_get_gamma(sampler, study)(n_finished)))
        if _SPLIT_TRIALS_TAKES_CONSTRAINTS_ENABLED:
            below_trials, above_trials = _split_trials(study, trials, n_below, False)
        else:
            below_trials, above_trials = _split_trials(study, trials, n_below)
        mpe_below = sampler._build_parzen_estimator(
            study, search_space, below_trials, handle_below=True
        )
        mpe_above = sampler._build_parzen_estimator(
            study, search_space, above_trials, handle_below=False
        )
        return mpe_below, mpe_above

    def _compute_task_similarities(
        self,
        target_below: Any,
        source_estimators: list[tuple[Any, Any, tuple[int, int]]],
    ) -> np.ndarray:
        """Compute similarity between target and each source task.

        Uses Monte-Carlo estimation of the overlap between the promising
        regions (below distributions) via Total Variation distance.
        """
        n_mc_samples = 1000
        uniform_samples = target_below.sample(self._rng, n_mc_samples)

        target_log_pdf = target_below.log_pdf(uniform_samples)
        target_pdf = np.exp(target_log_pdf - target_log_pdf.max())
        target_pdf /= target_pdf.sum() + 1e-12

        similarities = np.zeros(len(source_estimators))
        for i, (s_below, _, _) in enumerate(source_estimators):
            source_log_pdf = s_below.log_pdf(uniform_samples)
            source_pdf = np.exp(source_log_pdf - source_log_pdf.max())
            source_pdf /= source_pdf.sum() + 1e-12

            tv = 0.5 * np.sum(np.abs(target_pdf - source_pdf))
            similarities[i] = np.clip((1 - tv) / (1 + tv), 0.0, 1.0)

        return similarities

    def _compute_task_weights(self, similarities: np.ndarray) -> np.ndarray:
        """Compute task weights from similarities.

        Returns array of shape (1 + n_source,) where index 0 is the
        target task weight.
        """
        n_tasks = 1 + len(similarities)
        weights = np.zeros(n_tasks)

        weights[0] = 1.0 - np.sum(similarities) / n_tasks
        for i, sim in enumerate(similarities):
            weights[i + 1] = sim / n_tasks

        weights = np.maximum(weights, 0.0)
        weights /= weights.sum() + 1e-12
        return weights


def _with_running_params(
    trial: FrozenTrial, search_space: dict[str, BaseDistribution]
) -> FrozenTrial:
    """Read shared relative parameters without modifying the storage's trial object."""
    chunks = []
    i = 0
    while chunk := trial.system_attrs.get(f"{_RELATIVE_PARAMS_KEY}:{i}"):
        chunks.append(chunk)
        i += 1
    if not chunks:
        return trial
    try:
        params = json.loads("".join(chunks))
    except json.JSONDecodeError:
        # Another worker may still be writing the remaining chunks.
        return trial
    updated_trial = copy(trial)
    updated_trial.params = {**params, **trial.params}
    # Optuna v4.1 also reads distributions for parameters not yet suggested.
    updated_trial.distributions = {**search_space, **trial.distributions}
    return updated_trial


def _get_gamma(sampler: TPESampler, study: Study) -> Callable[[int], int]:
    """Return the ``gamma`` function that ``sampler`` uses for ``study``.

    In Optuna v5, ``TPESampler._gamma`` stays ``None`` until ``TPESampler._sample``
    is called, so resolve the default in the same way as ``TPESampler._sample``.
    """
    if sampler._gamma is not None:
        return sampler._gamma
    if len(study.directions) <= 1:
        return default_gamma
    # Only Optuna v5 reaches here, and it provides ``default_gamma_multiobjective``.
    from optuna.samplers._tpe.sampler import default_gamma_multiobjective

    return default_gamma_multiobjective


def _log_weighted_sum_exp(weights: np.ndarray, log_values: np.ndarray) -> np.ndarray:
    """Compute log(sum_i w_i * exp(v_i)) in a numerically stable way.

    Args:
        weights: Shape (n_tasks,).
        log_values: Shape (n_tasks, n_candidates).

    Returns:
        Shape (n_candidates,).
    """
    max_log = np.max(log_values, axis=0, keepdims=True)
    weighted_sum = np.sum(weights[:, np.newaxis] * np.exp(log_values - max_log), axis=0)
    return max_log[0] + np.log(weighted_sum + 1e-12)
