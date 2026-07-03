from __future__ import annotations

import decimal
from typing import Any

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import TrialState
import plotly.graph_objects as go


_STATE_COLORS = {
    TrialState.COMPLETE: "#2E7D32",
    TrialState.RUNNING: "#1565C0",
    TrialState.PRUNED: "#F9A825",
    TrialState.FAIL: "#C62828",
    TrialState.WAITING: "#6A1B9A",
}
_INTERNAL_COLOR = "#B0BEC5"
_UNEXPANDED_COLOR = "#ECEFF1"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _enumerate_candidates(distribution: BaseDistribution) -> list[Any]:
    # Enumerate all candidate values of a (finite) distribution using its public attributes.
    # Returns an empty list if the distribution is not finite (e.g. a step-less float range),
    # in which case the "unexplored" count for that branch is simply not shown.
    if isinstance(distribution, CategoricalDistribution):
        return list(distribution.choices)
    if isinstance(distribution, IntDistribution):
        return list(range(distribution.low, distribution.high + 1, distribution.step))
    if isinstance(distribution, FloatDistribution):
        if distribution.step is None:
            return []
        low = decimal.Decimal(str(distribution.low))
        high = decimal.Decimal(str(distribution.high))
        step = decimal.Decimal(str(distribution.step))
        candidates = []
        while low <= high:
            candidates.append(float(low))
            low += step
        return candidates
    return []


def plot_brute_force_tree(study: optuna.Study) -> go.Figure:
    """Plot the search tree explored by :class:`~optuna.samplers.BruteForceSampler`.

    Each path from the root to a leaf represents the sequence of parameters suggested in one
    trial. Leaves are colored by trial state (complete/pruned/failed/running), and a synthetic
    ``... unexplored`` leaf is added under every branching point that still has unvisited
    candidates, so the remaining search space is visible at a glance. Hovering over a complete
    trial's leaf shows its objective value(s).

    Args:
        study:
            A study, ideally optimized with :class:`~optuna.samplers.BruteForceSampler`. Any
            study works, but the "unexplored" counts are only meaningful for finite,
            grid-like search spaces.

    Returns:
        A :class:`plotly.graph_objects.Figure` with an interactive icicle chart.
    """
    trials = study.get_trials(deepcopy=False)

    ids: list[str] = ["root"]
    labels: list[str] = ["root"]
    parents: list[str] = [""]
    colors: list[str] = [_INTERNAL_COLOR]
    hovertexts: list[str] = [f"{len(trials)} trial(s)"]
    sizes: list[float] = [0]

    node_index: dict[str, int] = {"root": 0}
    node_distribution: dict[str, BaseDistribution] = {}
    node_observed_values: dict[str, set[Any]] = {}

    for trial in trials:
        node_id = "root"
        for param_name, value in trial.params.items():
            node_distribution.setdefault(node_id, trial.distributions[param_name])
            node_observed_values.setdefault(node_id, set()).add(value)

            child_id = f"{node_id}/{param_name}={_format_value(value)}"
            if child_id not in node_index:
                node_index[child_id] = len(ids)
                ids.append(child_id)
                labels.append(f"{param_name}={_format_value(value)}")
                parents.append(node_id)
                colors.append(_INTERNAL_COLOR)
                hovertexts.append(f"{param_name}={value}")
                sizes.append(0)
            node_id = child_id

        # Give every trial its own leaf, so duplicate parameter combinations remain visible.
        leaf_id = f"{node_id}#trial{trial.number}"
        node_index[leaf_id] = len(ids)
        ids.append(leaf_id)
        labels.append(f"trial {trial.number}")
        parents.append(node_id)
        sizes.append(1)

        state = trial.state
        colors.append(_STATE_COLORS.get(state, _INTERNAL_COLOR))
        if state == TrialState.COMPLETE:
            hovertexts.append(f"trial {trial.number} ({state.name}): values={trial.values}")
        else:
            hovertexts.append(f"trial {trial.number} ({state.name})")

    for node_id, distribution in node_distribution.items():
        candidates = _enumerate_candidates(distribution)
        n_unexplored = len(candidates) - len(node_observed_values[node_id])
        if n_unexplored <= 0:
            continue
        unexplored_id = f"{node_id}/...unexplored"
        node_index[unexplored_id] = len(ids)
        ids.append(unexplored_id)
        labels.append(f"{n_unexplored} unexplored")
        parents.append(node_id)
        colors.append(_UNEXPANDED_COLOR)
        hovertexts.append(f"{n_unexplored} candidate value(s) not sampled yet")
        sizes.append(n_unexplored)

    # Aggregate leaf sizes into their ancestors. Children are always appended after their
    # parent, so a single reverse pass is enough regardless of tree depth.
    for i in range(len(ids) - 1, 0, -1):
        sizes[node_index[parents[i]]] += sizes[i]

    fig = go.Figure(
        go.Icicle(
            ids=ids,
            labels=labels,
            parents=parents,
            values=sizes,
            branchvalues="total",
            hovertext=hovertexts,
            hoverinfo="text",
            marker={"colors": colors},
        )
    )
    fig.update_layout(
        title="Brute Force Search Tree",
        margin={"t": 50, "l": 25, "r": 25, "b": 25},
    )
    return fig


__all__ = ["plot_brute_force_tree"]
