"""A hierarchical (mixture-of-experts) variant of Optuna's multivariate TPESampler.

On a conditional (dynamic) search space, Optuna's multivariate TPESampler is inadequate. With
``group=False`` the conditional parameters fall back to independent (random) sampling, because a
dynamic search space is not supported for ``multivariate=True``. With ``group=True`` the space
is decomposed into parameter groups that are each sampled *independently*, so correlation
*between* groups is lost.

This sampler is built on the ``group=True`` decomposition but samples the groups *hierarchically*
instead of independently: parameters that are always present are sampled first, and parameters
that only appear under certain conditions are sampled afterwards, conditioned on the values
already chosen. An always-present parameter and a parameter that only appears in one branch can
therefore be modeled jointly. The conditional structure — which parameters are requested next
given the values sampled so far — is either learned from observed trials with a
``DecisionTreeClassifier`` or provided exactly by the user through ``conditional_fn``.

See ``README.md`` and ``docs/adr/0001-*`` for the relationship to the union-search-space approach
in optuna#6697.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.logging import get_logger
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.samplers._tpe.sampler import default_gamma
from optuna.samplers._tpe.sampler import default_weights
from optuna.search_space import intersection_search_space
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier


_logger = get_logger(f"optuna.{__name__}")

# A user-provided exact map: given the parameter values chosen so far (external
# representation, like ``trial.params``), return the names of the parameters that the
# objective will request next. Mirrors the objective's ``suggest_*`` control flow.
ConditionalFn = Callable[[dict[str, Any]], Iterable[str]]


class _TreeBranchClassifier:
    """Learned default that predicts which child group activates given parent values.

    A ``DecisionTreeClassifier`` is fit per parent group on the observed trials, mapping
    the (encoded) parent-parameter values to the index of the child group the trial
    activated (or a "no child" sentinel). The fit is cached and only redone when new
    trials make the cached tree mispredict (Approach B self-correction).
    """

    def __init__(self) -> None:
        """Initialize empty per-group tree, fit-size, and encoder caches."""
        self._trees: dict[int, DecisionTreeClassifier] = {}
        self._n_trials_at_fit: dict[int, int] = {}
        self._encoders: dict[str, Callable[[NDArray[Any]], NDArray[Any]]] = {}

    def reset(self) -> None:
        """Drop all cached trees, e.g. after the hierarchy structure changed."""
        self._trees = {}
        self._n_trials_at_fit = {}

    def predict(
        self,
        path_params: dict[str, NDArray[np.float64]],
        parent_trials: list[FrozenTrial],
        groups: list[dict[str, BaseDistribution]],
        child_group_indices: list[int],
        parent_group_index: int,
        sampler: HierarchicalTPESampler,
    ) -> NDArray[np.int64]:
        """Predict which child group activates for each candidate row.

        Args:
            path_params: The parameter values sampled so far, keyed by name, each an array
                with one entry per candidate row (internal representation).
            parent_trials: Observed trials that contain the parent group's parameters.
            groups: All parameter groups of the search space.
            child_group_indices: Indices into ``groups`` of the parent's child groups.
            parent_group_index: Index into ``groups`` of the parent group.
            sampler: The owning sampler, used to read trial parameters consistently.

        Returns:
            An array with one entry per candidate row, holding the local index into
            ``child_group_indices`` of the activated child, or ``len(child_group_indices)``
            if no child group is activated.
        """
        n_child = len(child_group_indices)
        n_rows = len(next(iter(path_params.values())))
        if len(parent_trials) == 0:
            return np.full(n_rows, n_child, dtype=np.int64)

        parent_group = groups[parent_group_index]
        self._ensure_encoders(parent_group)
        features = self._trial_features(parent_trials, parent_group, sampler)
        targets = self._trial_targets(parent_trials, groups, child_group_indices, sampler)
        self._fit_if_needed(features, targets, parent_group_index, len(parent_trials))

        sample_features = self._sample_features(path_params, parent_group)
        predictions = self._trees[parent_group_index].predict(sample_features)
        return np.asarray(predictions, dtype=np.int64)

    def _ensure_encoders(self, parent_group: dict[str, BaseDistribution]) -> None:
        """Create and cache a feature encoder for each parameter in the parent group.

        Args:
            parent_group: The parent group's parameters and their distributions."""
        for name, distribution in parent_group.items():
            if name in self._encoders:
                continue
            if isinstance(distribution, CategoricalDistribution):
                classes = [
                    distribution.to_internal_repr(choice) for choice in distribution.choices
                ]
                self._encoders[name] = self._make_categorical_encoder(classes)
            else:
                self._encoders[name] = self._make_numeric_encoder()

    @staticmethod
    def _make_categorical_encoder(
        classes: list[Any],
    ) -> Callable[[NDArray[Any]], NDArray[Any]]:
        """Build a one-hot encoder for a categorical parameter.

        Args:
            classes: The internal representations of the categorical choices, in order.

        Returns:
            A function that one-hot encodes an array of internal categorical values.
        """

        def encode(x: NDArray[Any]) -> NDArray[Any]:
            """One-hot encode internal categorical values against ``classes``.

            Args:
                x: An array of internal-representation categorical values.

            Returns:
                The label-binarized (one-hot) encoding of ``x``.
            """
            return np.asarray(label_binarize(np.asarray(x).astype(int), classes=classes))

        return encode

    @staticmethod
    def _make_numeric_encoder() -> Callable[[NDArray[Any]], NDArray[Any]]:
        """Build an encoder for a numerical parameter.

        Returns:
            A function that reshapes an array of numerical values into a column feature.
        """

        def encode(x: NDArray[Any]) -> NDArray[Any]:
            """Reshape numerical values into a single feature column.

            Args:
                x: An array of numerical values.

            Returns:
                ``x`` as a float column of shape ``(len(x), 1)``.
            """
            return np.asarray(x, dtype=float).reshape(-1, 1)

        return encode

    def _trial_features(
        self,
        trials: list[FrozenTrial],
        parent_group: dict[str, BaseDistribution],
        sampler: HierarchicalTPESampler,
    ) -> NDArray[np.float64]:
        """Encode the parent-group parameters of each trial into a feature matrix.

        Args:
            trials: The trials to encode (each must contain the parent group's parameters).
            parent_group: The parent group's parameters and their distributions.
            sampler: The owning sampler, used to read trial parameters consistently.

        Returns:
            A feature matrix of shape ``(len(trials), n_encoded_features)``.
        """
        features = []
        for name, distribution in parent_group.items():
            internal = np.asarray(
                [
                    distribution.to_internal_repr(sampler._get_params(trial)[name])
                    for trial in trials
                ]
            )
            features.append(self._encoders[name](internal))
        return np.concatenate(features, axis=1)

    def _trial_targets(
        self,
        trials: list[FrozenTrial],
        groups: list[dict[str, BaseDistribution]],
        child_group_indices: list[int],
        sampler: HierarchicalTPESampler,
    ) -> NDArray[np.int64]:
        """Label each trial with the child group it activated.

        Args:
            trials: The trials to label.
            groups: All parameter groups of the search space.
            child_group_indices: Indices into ``groups`` of the parent's child groups.
            sampler: The owning sampler, used to read trial parameters consistently.

        Returns:
            An array with one entry per trial, holding the local index into
            ``child_group_indices`` of the activated child, or ``len(child_group_indices)``
            if the trial activated no child group.
        """
        n_child = len(child_group_indices)
        targets = []
        for trial in trials:
            params = set(sampler._get_params(trial).keys())
            for local_index, group_index in enumerate(child_group_indices):
                if params.issuperset(groups[group_index]):
                    targets.append(local_index)
                    break
            else:
                targets.append(n_child)  # No child group is active for this trial.
        return np.asarray(targets, dtype=np.int64)

    def _sample_features(
        self,
        path_params: dict[str, NDArray[np.float64]],
        parent_group: dict[str, BaseDistribution],
    ) -> NDArray[np.float64]:
        """Encode the parent-group values of each candidate row into a feature matrix.

        Args:
            path_params: The parameter values sampled so far, keyed by name, each an array
                with one entry per candidate row (internal representation).
            parent_group: The parent group's parameters and their distributions.

        Returns:
            A feature matrix of shape ``(n_rows, n_encoded_features)``.
        """
        features = [self._encoders[name](path_params[name]) for name in parent_group]
        return np.concatenate(features, axis=1)

    def _fit_if_needed(
        self,
        features: NDArray[np.float64],
        targets: NDArray[np.int64],
        index: int,
        n_trials: int,
    ) -> None:
        """Fit (or reuse) the cached tree for a parent group, refitting on mismatch.

        The cached tree is reused when no new trials have arrived. Otherwise it is validated
        against the observed trials and only refit (with a warning) when it mispredicts. The
        tree is grown until it classifies every observed trial correctly, or, if the branches
        are not separable by the parent values, the best unconstrained tree is kept.

        Args:
            features: The encoded parent-group features of the observed trials.
            targets: The activated-child label of each observed trial.
            index: The parent group index used as the cache key.
            n_trials: The number of observed trials, used to skip revalidation when unchanged."""
        if index in self._trees:
            if self._n_trials_at_fit.get(index) == n_trials:
                # No new trials since the last fit; the cached tree is still valid.
                return
            if bool((self._trees[index].predict(features) == targets).all()):
                self._n_trials_at_fit[index] = n_trials
                return
            _logger.warning(
                "Refitting the branch classifier for parameter group %d because newly "
                "observed trials are misclassified. If this keeps happening, the conditional "
                "structure is likely not a deterministic function of the parent parameters.",
                index,
            )

        n_samples = features.shape[0]
        # Grow the tree until it classifies every observed trial correctly.
        for depth in range(1, max(2, n_samples + 1)):
            classifier = DecisionTreeClassifier(max_depth=depth)
            classifier.fit(features, targets)
            if bool((classifier.predict(features) == targets).all()):
                self._trees[index] = classifier
                self._n_trials_at_fit[index] = n_trials
                return

        # The branches are not perfectly separable by the parent values (e.g. stochastic
        # activation). Keep the best unconstrained tree; mispredictions self-correct via
        # the independent-sampling fallback.
        classifier = DecisionTreeClassifier()
        classifier.fit(features, targets)
        self._trees[index] = classifier
        self._n_trials_at_fit[index] = n_trials


class HierarchicalTPESampler(TPESampler):
    """A hierarchical (mixture-of-experts) multivariate TPESampler for conditional spaces.

    The sampler decomposes the search space into Optuna parameter groups (sets of
    parameters that always co-occur), infers a parent/child hierarchy between them, and
    samples each group with its own Parzen estimators ("experts"). For every EI candidate
    it routes from parents to children using either a learned ``DecisionTreeClassifier`` or
    a user-supplied ``conditional_fn``.

    The hierarchy is always inferred automatically and is self-correcting: if a branch is
    mispredicted, the objective requests a parameter that was not sampled hierarchically, and
    that parameter is drawn by independent (univariate) TPE instead.
    """

    def __init__(
        self,
        *,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] = default_gamma,
        weights: Callable[[int], np.ndarray] = default_weights,
        seed: int | None = None,
        multivariate: bool = True,
        group: bool = True,
        warn_independent_sampling: bool = True,
        constant_liar: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        categorical_distance_func: (dict[str, Callable[[Any, Any], float]] | None) = None,
        conditional_fn: ConditionalFn | None = None,
    ) -> None:
        """Construct the sampler.

        All arguments except ``conditional_fn`` are forwarded to
        :class:`~optuna.samplers.TPESampler` and behave identically. ``multivariate`` and
        ``group`` default to ``True`` because the hierarchical algorithm requires them; if
        either is set to ``False`` the sampler logs an INFO message and falls back to the
        standard :class:`~optuna.samplers.TPESampler` behavior.

        Args:
            consider_prior: Forwarded to :class:`~optuna.samplers.TPESampler`.
            prior_weight: Forwarded to :class:`~optuna.samplers.TPESampler`.
            consider_magic_clip: Forwarded to :class:`~optuna.samplers.TPESampler`.
            consider_endpoints: Forwarded to :class:`~optuna.samplers.TPESampler`.
            n_startup_trials: Number of random trials before hierarchical sampling starts.
            n_ei_candidates: Number of expected-improvement candidates drawn per trial. These
                are partitioned across the hierarchy, so deeper structures benefit from larger
                values.
            gamma: Forwarded to :class:`~optuna.samplers.TPESampler`.
            weights: Forwarded to :class:`~optuna.samplers.TPESampler`.
            seed: Random seed.
            multivariate: Must be ``True`` for hierarchical sampling; otherwise the sampler
                falls back to standard TPE.
            group: Must be ``True`` for hierarchical sampling; otherwise the sampler falls
                back to standard TPE. Forced to ``False`` when ``multivariate`` is ``False``.
            warn_independent_sampling: Forwarded to :class:`~optuna.samplers.TPESampler`.
            constant_liar: Forwarded to :class:`~optuna.samplers.TPESampler`.
            constraints_func: Forwarded to :class:`~optuna.samplers.TPESampler`.
            categorical_distance_func: Forwarded to :class:`~optuna.samplers.TPESampler`.
            conditional_fn: Optional exact map of the objective's conditional structure,
                ``Callable[[dict[str, Any]], Iterable[str]]``. It is called once per hierarchy
                level while building each candidate. Its **input** is a dict of every parameter
                sampled so far on the current branch path, keyed by name, with values in the
                external representation (exactly as in ``trial.params`` — the chosen category,
                the float/int value); it holds the always-present parameters plus any
                conditional parameters already chosen higher up, and not parameters that have
                not been sampled yet. It must **return** the names of the parameters the
                objective requests next given that input (the parameters the very next
                ``suggest_*`` calls would use), or an empty iterable at a leaf. A conditional
                parameter group activates for a candidate when all of its parameter names are in
                the returned set (extra names are harmless). If ``None`` (default), this mapping
                is learned from observed trials with a ``DecisionTreeClassifier``. See
                ``README.md`` for worked single- and multi-level examples."""
        # ``group`` requires ``multivariate`` in Optuna. If the user disables
        # ``multivariate`` we disable ``group`` too and fall back to standard TPE.
        if not multivariate:
            group = False

        super().__init__(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            weights=weights,
            seed=seed,
            multivariate=multivariate,
            group=group,
            warn_independent_sampling=warn_independent_sampling,
            constant_liar=constant_liar,
            constraints_func=constraints_func,
            categorical_distance_func=categorical_distance_func,
        )

        self._conditional_fn = conditional_fn
        self._tree_classifier = _TreeBranchClassifier()
        self._cached_hierarchy: list[int | None] | None = None
        self._hierarchy_cache_key: Any = None
        self._fallback_logged = False
        self._name_to_distribution: dict[str, BaseDistribution] = {}

        # Univariate TPE used to sample parameters that were not produced hierarchically
        # (e.g. when a branch prediction was wrong). Warnings are disabled because this
        # fallback is expected behavior, not a misconfiguration.
        self._independent_sampler = TPESampler(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            weights=weights,
            seed=seed,
            multivariate=False,
            group=False,
            warn_independent_sampling=False,
            constant_liar=constant_liar,
            constraints_func=constraints_func,
            categorical_distance_func=categorical_distance_func,
        )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        """Sample the relative parameters for a trial using hierarchical TPE.

        Falls back to the standard TPESampler behavior when ``multivariate``/``group`` are not
        both enabled, before startup is complete, or when no conditional structure exists.

        Args:
            study: The study being optimized.
            trial: The trial for which to sample parameters.
            search_space: The relative search space inferred for this trial.

        Returns:
            A mapping from parameter name to sampled value (external representation).
        """
        if not (self._multivariate and self._group):
            if not self._fallback_logged:
                _logger.info(
                    "%s requires multivariate=True and group=True to sample hierarchically; "
                    "falling back to the standard TPESampler behavior.",
                    self.__class__.__name__,
                )
                self._fallback_logged = True
            return super().sample_relative(study, trial, search_space)

        assert self._search_space_group is not None
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE, TrialState.PRUNED)
        if len(study._get_trials(deepcopy=False, states=states, use_cache=True)) < (
            self._n_startup_trials
        ):
            return {}

        trials = self._get_trials(study)
        hierarchy = self._determine_hierarchy(trials)
        if all(parent is None for parent in hierarchy):
            # No conditional structure was observed, so the groups are independent and the
            # hierarchical routing is a no-op. Delegate to the standard group TPESampler,
            # which makes the behavior identical to TPESampler(multivariate=True, group=True).
            return super().sample_relative(study, trial, search_space)

        params = self._sample_hierarchy(study, search_space, trials, hierarchy)

        # Parameters that were not sampled hierarchically are unlikely to be requested by
        # the objective, but may be if a branch prediction was wrong. Provide a value for
        # them through independent sampling so they are available if needed.
        for param_name, param_distribution in search_space.items():
            if param_name not in params:
                params[param_name] = self._independent_sampler.sample_independent(
                    study, trial, param_name, param_distribution
                )
        return params

    def _get_trials(self, study: Study) -> list[FrozenTrial]:
        """Get the trials used for sampling, including running trials under constant liar.

        Args:
            study: The study being optimized.

        Returns:
            The complete and pruned trials, plus running trials when ``constant_liar`` is set.
        """
        if self._constant_liar:
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not self._constant_liar
        return study._get_trials(deepcopy=False, states=states, use_cache=use_cache)

    def _sample_hierarchy(
        self,
        study: Study,
        search_space: dict[str, Any],
        trials: list[FrozenTrial],
        hierarchy: list[int | None],
    ) -> dict[str, Any]:
        """Generate hierarchical candidates and return the best configuration's parameters.

        Args:
            study: The study being optimized.
            search_space: The relative search space inferred for this trial.
            trials: The observed trials to estimate the experts from.
            hierarchy: The inferred parent index of each group (``None`` for a root group).

        Returns:
            A mapping from parameter name to the value of the best candidate (external
            representation), restricted to the parameters active on that candidate's path.
        """
        assert self._search_space_group is not None
        groups = self._search_space_group.search_spaces
        self._name_to_distribution = {
            name: distribution for group in groups for name, distribution in group.items()
        }

        mask, samples, delta_ll = self._get_hierarchical_samples(
            study, hierarchy, groups, trials, self._n_ei_candidates, ancestor_params={}
        )

        best = int(np.argmax(delta_ll))
        ret: dict[str, Any] = {}
        for param_name, distribution in search_space.items():
            if param_name not in mask or not bool(mask[param_name][best]):
                continue
            # ``samples[param_name]`` is compressed to the rows where the parameter was
            # active; map the global best row to its compressed index.
            compressed_index = np.empty(self._n_ei_candidates, dtype=np.int64)
            compressed_index[mask[param_name]] = np.arange(int(mask[param_name].sum()))
            value = samples[param_name][compressed_index[best]]
            ret[param_name] = distribution.to_external_repr(value)
        return ret

    def _get_hierarchical_samples(
        self,
        study: Study,
        hierarchy: list[int | None],
        groups: list[dict[str, BaseDistribution]],
        trials: list[FrozenTrial],
        n_samples: int,
        ancestor_params: dict[str, NDArray[np.float64]],
    ) -> tuple[dict[str, NDArray[np.bool_]], dict[str, NDArray[np.float64]], NDArray[np.float64]]:
        """Recursively sample the root groups of a (sub)hierarchy and their descendants.

        Args:
            study: The study being optimized.
            hierarchy: The parent index of each group (``None`` marks a root of this subtree).
            groups: All parameter groups of the search space.
            trials: The observed trials available to this subtree.
            n_samples: The number of candidate rows to sample.
            ancestor_params: Parameters already sampled on the path into this subtree, keyed
                by name and aligned to the ``n_samples`` rows (internal representation).

        Returns:
            A tuple ``(mask, samples, delta_ll)`` where ``mask[name]`` marks the rows for which
            ``name`` was sampled, ``samples[name]`` holds the sampled values compressed to
            those rows, and ``delta_ll`` is the accumulated below/above log-likelihood ratio
            per row.
        """
        samples: dict[str, NDArray[np.float64]] = {}
        mask: dict[str, NDArray[np.bool_]] = {}
        for group in groups:
            for param_name in group:
                mask[param_name] = np.zeros(n_samples, dtype=bool)
        delta_ll = np.zeros(n_samples)

        for i in range(len(hierarchy)):
            if hierarchy[i] is not None:
                continue
            group_samples, d_ll = self._get_samples(
                study, trials, groups[i], n_samples, ancestor_params
            )
            for param_name in group_samples:
                mask[param_name][:] = True
                samples[param_name] = group_samples[param_name]
            delta_ll += d_ll

            self._get_hierarchical_child_samples(
                study, hierarchy, groups, trials, mask, samples, delta_ll, i, ancestor_params
            )
        return mask, samples, delta_ll

    def _get_hierarchical_child_samples(
        self,
        study: Study,
        hierarchy: list[int | None],
        groups: list[dict[str, BaseDistribution]],
        trials: list[FrozenTrial],
        mask: dict[str, NDArray[np.bool_]],
        samples: dict[str, NDArray[np.float64]],
        delta_ll: NDArray[np.float64],
        parent_group_index: int,
        ancestor_params: dict[str, NDArray[np.float64]],
    ) -> None:
        """Predict and sample the child groups of a parent group, in place.

        For each candidate row, the child group to activate is predicted, the matching rows
        are recursively sampled, and ``mask``, ``samples`` and ``delta_ll`` are updated in
        place.

        Args:
            study: The study being optimized.
            hierarchy: The parent index of each group for the current subtree.
            groups: All parameter groups of the search space.
            trials: The observed trials available to this subtree.
            mask: Per-parameter row masks, updated in place.
            samples: Per-parameter sampled values, updated in place.
            delta_ll: Accumulated per-row log-likelihood ratio, updated in place.
            parent_group_index: Index into ``groups`` of the parent group.
            ancestor_params: Parameters sampled on the path into this subtree, aligned to the
                rows (internal representation)."""
        child_group_indices = [
            index for index in range(len(hierarchy)) if hierarchy[index] == parent_group_index
        ]
        if len(child_group_indices) == 0:
            return

        parent_group = groups[parent_group_index]
        parent_trials = [
            trial for trial in trials if set(parent_group).issubset(self._get_params(trial).keys())
        ]
        # All parameters chosen so far on the path to this branch, aligned to the rows.
        path_params = {**ancestor_params, **samples}
        predictions = self._predict_children(
            path_params, parent_trials, groups, child_group_indices, parent_group_index
        )

        for local_index, index in enumerate(child_group_indices):
            prediction_mask = predictions == local_index
            n_child = int(prediction_mask.sum())
            if n_child == 0:
                continue
            child_hierarchy: list[int | None] = [
                -1 if item is None else None if j == index else item
                for j, item in enumerate(hierarchy)
            ]
            child_ancestor = {name: arr[prediction_mask] for name, arr in path_params.items()}
            child_mask, child_samples, child_delta_ll = self._get_hierarchical_samples(
                study, child_hierarchy, groups, parent_trials, n_child, child_ancestor
            )
            delta_ll[prediction_mask] += child_delta_ll
            for param_name in child_samples:
                mask[param_name][prediction_mask] = child_mask[param_name]
                samples[param_name] = child_samples[param_name]

    def _predict_children(
        self,
        path_params: dict[str, NDArray[np.float64]],
        parent_trials: list[FrozenTrial],
        groups: list[dict[str, BaseDistribution]],
        child_group_indices: list[int],
        parent_group_index: int,
    ) -> NDArray[np.int64]:
        """Predict the activated child group per candidate row, via map or learned tree.

        Args:
            path_params: The parameter values sampled so far, keyed by name, aligned to the
                candidate rows (internal representation).
            parent_trials: Observed trials that contain the parent group's parameters.
            groups: All parameter groups of the search space.
            child_group_indices: Indices into ``groups`` of the parent's child groups.
            parent_group_index: Index into ``groups`` of the parent group.

        Returns:
            An array with one entry per row: the local index into ``child_group_indices`` of
            the activated child, or ``len(child_group_indices)`` for no child.
        """
        if self._conditional_fn is not None:
            return self._predict_children_from_fn(path_params, groups, child_group_indices)
        return self._tree_classifier.predict(
            path_params, parent_trials, groups, child_group_indices, parent_group_index, self
        )

    def _predict_children_from_fn(
        self,
        path_params: dict[str, NDArray[np.float64]],
        groups: list[dict[str, BaseDistribution]],
        child_group_indices: list[int],
    ) -> NDArray[np.int64]:
        """Predict the activated child group per row using the user-supplied exact map.

        Args:
            path_params: The parameter values sampled so far, keyed by name, aligned to the
                candidate rows (internal representation).
            groups: All parameter groups of the search space.
            child_group_indices: Indices into ``groups`` of the parent's child groups.

        Returns:
            An array with one entry per row: the local index into ``child_group_indices`` of
            the activated child, or ``len(child_group_indices)`` for no child.
        """
        assert self._conditional_fn is not None
        n_child = len(child_group_indices)
        child_param_sets = [set(groups[index]) for index in child_group_indices]
        names = list(path_params)
        n_rows = len(path_params[names[0]])
        predictions = np.full(n_rows, n_child, dtype=np.int64)
        for row in range(n_rows):
            external = {
                name: self._name_to_distribution[name].to_external_repr(path_params[name][row])
                for name in names
            }
            requested = set(self._conditional_fn(external))
            for local_index, child_params in enumerate(child_param_sets):
                if child_params <= requested:
                    predictions[row] = local_index
                    break
        return predictions

    def _get_samples(
        self,
        study: Study,
        trials: list[FrozenTrial],
        group: dict[str, BaseDistribution],
        n_samples: int,
        ancestor_params: dict[str, NDArray[np.float64]],
    ) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64]]:
        """Sample a group and score it by its below/above ratio, conditioned on its ancestors.

        When the group has path-ancestors that vary on its branch, the score is the
        *conditional* ratio ``log l_below(group | ancestors) - log l_above(group | ancestors)``,
        computed as ``joint(ancestors, group) - marginal(ancestors)`` on the branch trials.
        This is the exact chain-rule factor: it captures the correlation between the group and
        its always-present ancestors without double-counting the ancestors (already scored by
        the root group). A root group, or one whose ancestors are all constant on the branch,
        falls back to the plain marginal ratio.

        Args:
            study: The study being optimized.
            trials: The observed trials available; filtered to those containing ``group``.
            group: The parameter group to sample.
            n_samples: The number of candidate rows to sample.
            ancestor_params: Parameters already sampled on the path, aligned to the rows.

        Returns:
            A tuple ``(samples, delta_ll)`` of the sampled group values and the (conditional)
            below/above log-likelihood ratio per row.
        """
        trial_subset = [
            trial for trial in trials if set(group).issubset(self._get_params(trial).keys())
        ]
        n = sum(trial.state != TrialState.RUNNING for trial in trial_subset)
        below_trials, above_trials = _split_trials(
            study, trial_subset, self._gamma(n), self._constraints_func is not None
        )

        ancestors = self._conditioning_ancestors(ancestor_params, group)
        if not ancestors:
            mpe_below = self._build_parzen_estimator(study, group, below_trials, handle_below=True)
            mpe_above = self._build_parzen_estimator(
                study, group, above_trials, handle_below=False
            )
            samples = mpe_below.sample(self._rng.rng, n_samples)
            return samples, self._compute_acquisition_func(samples, mpe_below, mpe_above)

        joint_space = {**ancestors, **group}
        joint_below = self._build_parzen_estimator(
            study, joint_space, below_trials, handle_below=True
        )
        joint_above = self._build_parzen_estimator(
            study, joint_space, above_trials, handle_below=False
        )
        anc_below = self._build_parzen_estimator(study, ancestors, below_trials, handle_below=True)
        anc_above = self._build_parzen_estimator(
            study, ancestors, above_trials, handle_below=False
        )

        proposal = joint_below.sample(self._rng.rng, n_samples)
        samples = {name: proposal[name] for name in group}
        ancestor_values = {name: ancestor_params[name] for name in ancestors}
        joint_ratio = self._compute_acquisition_func(
            {**ancestor_values, **samples}, joint_below, joint_above
        )
        ancestor_ratio = self._compute_acquisition_func(ancestor_values, anc_below, anc_above)
        return samples, joint_ratio - ancestor_ratio

    def _conditioning_ancestors(
        self,
        ancestor_params: dict[str, NDArray[np.float64]],
        group: dict[str, BaseDistribution],
    ) -> dict[str, BaseDistribution]:
        """Pick the path ancestors to condition a group on: all non-single path parameters.

        Every parameter sampled earlier on the path is conditioned on, except the group's own
        parameters and any single-valued ones. Gates that are *constant on the branch* are kept
        intentionally: a Parzen mixture has a uniform prior component, so a constant gate kernel
        does **not** cancel in ``joint - marginal`` (it down-weights the prior), and including
        these gates measurably improves robustness across ``n_ei_candidates``.

        Args:
            ancestor_params: Parameters sampled on the path into this subtree.
            group: The group being sampled (its own parameters are never ancestors).

        Returns:
            A search space of the ancestor parameters to condition on.
        """
        return {
            name: self._name_to_distribution[name]
            for name in ancestor_params
            if name not in group and not self._name_to_distribution[name].single()
        }

    def _determine_hierarchy(self, trials: list[FrozenTrial]) -> list[int | None]:
        """Infer the parent index of each parameter group (or ``None`` for a root group).

        The result is cached and only recomputed when a new distinct parameter combination
        appears, because the hierarchy depends solely on which groups co-occur (not on
        parameter values or trial counts). Recomputing it cascades into resetting the learned
        branch classifier.

        Args:
            trials: The observed trials to infer the hierarchy from.

        Returns:
            A list with one entry per group: the index of its direct parent group, or ``None``
            if the group is a root.
        """
        assert self._search_space_group is not None
        groups = self._search_space_group.search_spaces
        finished_trials = [trial for trial in trials if trial.state != TrialState.RUNNING]
        combinations = frozenset(
            frozenset(self._get_params(trial).keys()) for trial in finished_trials
        )
        groups_key = tuple(frozenset(group) for group in groups)
        cache_key = (combinations, groups_key)
        if self._hierarchy_cache_key == cache_key and self._cached_hierarchy is not None:
            return self._cached_hierarchy

        hierarchy = self._compute_hierarchy(finished_trials, groups)
        if self._cached_hierarchy is not None and self._cached_hierarchy != hierarchy:
            self._tree_classifier.reset()
        self._hierarchy_cache_key = cache_key
        self._cached_hierarchy = hierarchy
        return hierarchy

    def _compute_hierarchy(
        self, trials: list[FrozenTrial], groups: list[dict[str, BaseDistribution]]
    ) -> list[int | None]:
        """Compute the strict parent of each group from the trials' intersection spaces.

        A group ``i`` is a candidate parent of group ``j`` when ``i``'s parameters are always
        present whenever ``j``'s are (i.e. contained in the intersection search space of the
        trials containing ``j``). Candidate parents that are themselves ancestors of another
        candidate are dropped, leaving each group with its single direct parent.

        Args:
            trials: The (non-running) observed trials.
            groups: All parameter groups of the search space.

        Returns:
            A list with one entry per group: the index of its direct parent, or ``None`` for a
            root group.

        Raises:
            RuntimeError: If a strict hierarchy cannot be determined.
        """
        n_groups = len(groups)
        candidates: list[list[int]] = [[] for _ in range(n_groups)]
        for j, child_group in enumerate(groups):
            child_intersection = intersection_search_space(
                [
                    trial
                    for trial in trials
                    if set(child_group).issubset(self._get_params(trial).keys())
                ]
            )
            for i, parent_group in enumerate(groups):
                if i == j:
                    continue
                if set(parent_group).issubset(child_intersection):
                    candidates[j].append(i)

        # Each group keeps only its direct parent: drop any candidate parent that is itself
        # an ancestor (candidate) of another candidate parent of the same group.
        for _ in range(1000):
            changed = False
            for item in candidates:
                if len(item) <= 1:
                    continue
                for index1 in list(item):
                    if any(index1 in candidates[index2] for index2 in item if index2 != index1):
                        item.remove(index1)
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break
        else:
            raise RuntimeError("Unable to determine a strict hierarchy between parameter groups.")

        return [item[0] if len(item) == 1 else None for item in candidates]
