from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntDistribution
from optuna.trial import TrialState
import plotly.graph_objects as go

from ._tree import _TreeNode
from ._tree import _UnexpandedTreeNode
from ._tree import build_full_tree


if TYPE_CHECKING:
    from optuna.study import Study


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


def plot_brute_force_tree(study: Study) -> go.Figure:
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
    states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING, TrialState.FAIL)
    trials = study._storage.get_all_trials(study._study_id, deepcopy=False, states=states)
    tree = build_full_tree(trials)

    trials_by_number = {t.number: t for t in trials}
    param_dists: dict[str, Any] = {}
    for t in trials:
        for param_name, dist in t.distributions.items():
            param_dists.setdefault(param_name, dist)

    ids: list[str] = ["root"]
    labels: list[str] = ["root"]
    parents: list[str] = [""]
    colors: list[str] = [_INTERNAL_COLOR]
    hovertexts: list[str] = [f"{len(trials)} trial(s)"]
    sizes: list[float] = [0]

    def _internal_to_external(param_name: str, internal_value: float) -> Any:
        dist = param_dists.get(param_name)
        if isinstance(dist, CategoricalDistribution):
            return dist.choices[int(internal_value)]
        if isinstance(dist, IntDistribution):
            return int(internal_value)
        return internal_value

    def _walk(node: _TreeNode, icicle_id: str) -> float:
        total_size: float = 0

        if node.trial_number >= 0:
            trial = trials_by_number[node.trial_number]
            leaf_id = f"{icicle_id}#trial{trial.number}"
            ids.append(leaf_id)
            labels.append(f"trial {trial.number}")
            parents.append(icicle_id)
            state = trial.state
            colors.append(_STATE_COLORS.get(state, _INTERNAL_COLOR))
            sizes.append(1)
            if state == TrialState.COMPLETE:
                hovertexts.append(f"trial {trial.number} ({state.name}): values={trial.values}")
            else:
                hovertexts.append(f"trial {trial.number} ({state.name})")
            total_size += 1

        if not node.children:
            return total_size

        assert node.param_name is not None
        n_unexpanded = 0

        for internal_value, child in node.children.items():
            if isinstance(child, _UnexpandedTreeNode):
                n_unexpanded += 1
                continue

            ext_value = _internal_to_external(node.param_name, internal_value)
            child_icicle_id = f"{icicle_id}/{node.param_name}={_format_value(ext_value)}"
            child_idx = len(ids)

            ids.append(child_icicle_id)
            labels.append(f"{node.param_name}={_format_value(ext_value)}")
            parents.append(icicle_id)
            colors.append(_INTERNAL_COLOR)
            hovertexts.append(f"{node.param_name}={ext_value}")
            sizes.append(0)

            child_size = _walk(child, child_icicle_id)
            sizes[child_idx] = child_size
            total_size += child_size

            labels[child_idx] += (
                f"<br>(Done: {child.count_completed()}, Total: {child.count_tree_size()})"
            )

        if n_unexpanded > 0:
            unexplored_id = f"{icicle_id}/...unexplored"
            ids.append(unexplored_id)
            labels.append(f"{n_unexpanded} unexplored")
            parents.append(icicle_id)
            colors.append(_UNEXPANDED_COLOR)
            hovertexts.append(f"{n_unexpanded} candidate value(s) not sampled yet")
            sizes.append(n_unexpanded)
            total_size += n_unexpanded

        return total_size

    root_size = _walk(tree, "root")
    sizes[0] = root_size
    labels[0] += f"<br>(Done: {tree.count_completed()}, Total: {tree.count_tree_size()})"

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
