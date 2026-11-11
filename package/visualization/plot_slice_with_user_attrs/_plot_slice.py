from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import math
import numbers
from typing import Any
from typing import cast
from typing import NamedTuple
import warnings

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_NUMERICAL = "numerical"
_CATEGORICAL = "categorical"
_SKIP = "skip"
_CONSTRAINTS_KEY = "constraints"

_MAX_TRIAL_NUMBERS_IN_WARNING = 5

COLOR_SCALE = plotly.colors.sequential.Blues

_logger = get_logger(f"optuna.{__name__}")


def _classify_value(value: Any) -> tuple[str, str | None]:
    if value is None:
        return _SKIP, None
    if isinstance(value, bool):
        return _CATEGORICAL, None
    if isinstance(value, numbers.Real):
        if math.isfinite(value):
            return _NUMERICAL, None
        return _SKIP, "non-finite"
    if isinstance(value, str):
        return _CATEGORICAL, None
    return _SKIP, "unsupported type"


def _is_feasible(trial: FrozenTrial) -> bool:
    if hasattr(trial, "constraints"):
        # Optuna v5.0.0 or later.
        constraints = trial.constraints.values()
    else:
        # Optuna v4.
        constraints = trial.system_attrs.get(_CONSTRAINTS_KEY) or []
    return all(c <= 0.0 for c in constraints)


class _CollectedValues(NamedTuple):
    x: list[Any]
    kinds: list[str]
    y: list[float]
    trial_numbers: list[int]
    constraints: list[bool]
    skipped: dict[str, list[int]]


def _collect_user_attr_values(
    trials: list[FrozenTrial],
    key: str,
    target: Callable[[FrozenTrial], float] | None,
) -> _CollectedValues:
    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast("float", t.value)

        target = _target

    collected = _CollectedValues(
        x=[], kinds=[], y=[], trial_numbers=[], constraints=[], skipped=defaultdict(list)
    )

    for t in trials:
        if key not in t.user_attrs:
            continue
        value = t.user_attrs[key]
        kind, reason = _classify_value(value)
        if kind == _SKIP:
            if reason is not None:
                collected.skipped[reason].append(t.number)
            continue
        collected.x.append(value)
        collected.kinds.append(kind)
        collected.y.append(target(t))
        collected.trial_numbers.append(t.number)
        collected.constraints.append(_is_feasible(t))

    return collected


# Copied from optuna/visualization/_slice.py (MIT License).
# `x_labels` uses `Any` instead of Optuna's private `CategoricalChoiceType`.
class _SliceSubplotInfo(NamedTuple):
    param_name: str
    x: list[Any]
    y: list[float]
    trial_numbers: list[int]
    is_log: bool
    is_numerical: bool
    constraints: list[bool]
    x_labels: tuple[Any, ...] | None


def _get_x_labels(values: list[Any]) -> tuple[Any, ...]:
    # Dedupe by repr() because True == 1 == 1.0 share the same hash.
    labels: dict[str, Any] = {}
    for value in values:
        labels.setdefault(repr(value), value)
    return tuple(labels.values())


def _get_user_attr_subplot_info(key: str, collected: _CollectedValues) -> _SliceSubplotInfo:
    is_numerical = all(kind == _NUMERICAL for kind in collected.kinds)
    return _SliceSubplotInfo(
        param_name=f"user_attrs.{key}",
        x=collected.x,
        y=collected.y,
        trial_numbers=collected.trial_numbers,
        is_log=False,
        is_numerical=is_numerical,
        constraints=collected.constraints,
        x_labels=None if is_numerical else _get_x_labels(collected.x),
    )


class _SlicePlotInfo(NamedTuple):
    target_name: str
    subplots: list[_SliceSubplotInfo]


def _check_plot_args(
    study: Study,
    target: Callable[[FrozenTrial], float] | None,
    target_name: str,
) -> None:
    if target is None and len(study.directions) > 1:
        raise ValueError(
            "If the `study` is being used for multi-objective optimization, "
            "please specify the `target`."
        )
    if target is not None and target_name == "Objective Value":
        warnings.warn(
            "`target` is specified, but `target_name` is the default value, 'Objective Value'."
        )


# Copied from optuna/visualization/_utils.py (MIT License) and simplified.
def _filter_nonfinite(
    trials: list[FrozenTrial],
    target: Callable[[FrozenTrial], float] | None,
) -> list[FrozenTrial]:
    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast("float", t.value)

        target = _target

    filtered_trials = []
    for t in trials:
        if math.isfinite(float(target(t))):
            filtered_trials.append(t)
        else:
            _logger.warning(
                f"Trial {t.number} is omitted in visualization "
                "because its objective value is inf or nan."
            )
    return filtered_trials


# Copied from optuna/visualization/_utils.py (MIT License).
def _is_log_scale(trials: list[FrozenTrial], param: str) -> bool:
    for trial in trials:
        if param not in trial.params:
            continue
        dist = trial.distributions[param]
        return isinstance(dist, (FloatDistribution, IntDistribution)) and dist.log
    return False


# Copied from optuna/visualization/_slice.py (MIT License).
def _get_param_subplot_info(
    trials: list[FrozenTrial],
    param: str,
    target: Callable[[FrozenTrial], float] | None,
    log_scale: bool,
    numerical: bool,
    x_labels: tuple[Any, ...] | None,
) -> _SliceSubplotInfo:
    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast("float", t.value)

        target = _target

    plot_info = _SliceSubplotInfo(
        param_name=param,
        x=[],
        y=[],
        trial_numbers=[],
        is_log=log_scale,
        is_numerical=numerical,
        constraints=[],
        x_labels=x_labels,
    )

    for t in trials:
        if param not in t.params:
            continue
        plot_info.x.append(t.params[param])
        plot_info.y.append(target(t))
        plot_info.trial_numbers.append(t.number)
        plot_info.constraints.append(_is_feasible(t))

    return plot_info


def _warn_skipped_user_attr_values(key: str, skipped: dict[str, list[int]]) -> None:
    # Warn once per reason instead of once per trial to avoid flooding the log.
    for reason, trial_numbers in skipped.items():
        shown = ", ".join(str(n) for n in trial_numbers[:_MAX_TRIAL_NUMBERS_IN_WARNING])
        if len(trial_numbers) > _MAX_TRIAL_NUMBERS_IN_WARNING:
            shown += ", ..."
        _logger.warning(
            f"{len(trial_numbers)} trial(s) are omitted for user_attrs.{key} "
            f"because of {reason} values (trial numbers: {shown})."
        )


def _get_slice_plot_info(
    study: Study,
    params: list[str] | None,
    user_attrs: list[str] | None,
    target: Callable[[FrozenTrial], float] | None,
    target_name: str,
) -> _SlicePlotInfo:
    if isinstance(user_attrs, str):
        raise TypeError("`user_attrs` must be a list of strings, not a string.")
    _check_plot_args(study, target, target_name)

    trials = _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    )

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return _SlicePlotInfo(target_name, [])

    # From here to `sorted_params` is copied from optuna/visualization/_slice.py (MIT License).
    all_params = {p_name for t in trials for p_name in t.params}

    distributions = {}
    for trial in trials:
        for param_name, distribution in trial.distributions.items():
            if param_name not in distributions:
                distributions[param_name] = distribution

    x_labels = {}
    for param_name, distribution in distributions.items():
        if isinstance(distribution, CategoricalDistribution):
            x_labels[param_name] = distribution.choices

    if params is None:
        sorted_params = sorted(all_params)
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError(f"Parameter {input_p_name} does not exist in your study.")
        sorted_params = sorted(set(params))

    sorted_user_attrs: list[str] = []
    if user_attrs is not None:
        all_user_attrs = {key for t in trials for key in t.user_attrs}
        for input_key in user_attrs:
            if input_key not in all_user_attrs:
                raise ValueError(f"User attribute {input_key} does not exist in your study.")
        sorted_user_attrs = sorted(set(user_attrs))

    subplots = [
        _get_param_subplot_info(
            trials=trials,
            param=param,
            target=target,
            log_scale=_is_log_scale(trials, param),
            numerical=not isinstance(distributions[param], CategoricalDistribution),
            x_labels=x_labels.get(param),
        )
        for param in sorted_params
    ]
    for key in sorted_user_attrs:
        collected = _collect_user_attr_values(trials, key, target)
        _warn_skipped_user_attr_values(key, collected.skipped)
        subplots.append(_get_user_attr_subplot_info(key, collected))

    return _SlicePlotInfo(target_name=target_name, subplots=subplots)


# The drawing code below is copied from optuna/visualization/_slice.py (MIT License).
# Changes: plotly is imported directly, infeasible trial numbers are kept, and categorical
# values are grouped by repr() so that True, 1, and 1.0 are treated as different values.
class _PlotValues(NamedTuple):
    x: list[Any]
    y: list[float]
    trial_numbers: list[int]


def _get_slice_plot(info: _SlicePlotInfo) -> go.Figure:
    layout = go.Layout(title="Slice Plot")

    if len(info.subplots) == 0:
        return go.Figure(data=[], layout=layout)
    elif len(info.subplots) == 1:
        figure = go.Figure(data=_generate_slice_subplot(info.subplots[0]), layout=layout)
        figure.update_xaxes(title_text=info.subplots[0].param_name)
        figure.update_yaxes(title_text=info.target_name)
        if not info.subplots[0].is_numerical:
            figure.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=_get_categorical_labels(info.subplots[0].x_labels),
            )
        elif info.subplots[0].is_log:
            figure.update_xaxes(type="log")
    else:
        figure = make_subplots(rows=1, cols=len(info.subplots), shared_yaxes=True)
        figure.update_layout(layout)
        showscale = True  # showscale option only needs to be specified once.
        for column_index, subplot_info in enumerate(info.subplots, start=1):
            trace = _generate_slice_subplot(subplot_info)
            trace[0].update(marker={"showscale": showscale})  # showscale's default is True.
            if showscale:
                showscale = False
            for t in trace:
                figure.add_trace(t, row=1, col=column_index)
            figure.update_xaxes(title_text=subplot_info.param_name, row=1, col=column_index)
            if column_index == 1:
                figure.update_yaxes(title_text=info.target_name, row=1, col=column_index)
            if not subplot_info.is_numerical:
                figure.update_xaxes(
                    type="category",
                    categoryorder="array",
                    categoryarray=_get_categorical_labels(subplot_info.x_labels),
                    row=1,
                    col=column_index,
                )
            elif subplot_info.is_log:
                figure.update_xaxes(type="log", row=1, col=column_index)
        if len(info.subplots) > 3:
            # Ensure that each subplot has a minimum width without relying on autusizing.
            figure.update_layout(width=300 * len(info.subplots))

    return figure


def _generate_slice_subplot(subplot_info: _SliceSubplotInfo) -> list[go.Scatter]:
    trace = []

    feasible = _PlotValues([], [], [])
    infeasible = _PlotValues([], [], [])

    for x, y, num, c in zip(
        subplot_info.x, subplot_info.y, subplot_info.trial_numbers, subplot_info.constraints
    ):
        if subplot_info.is_numerical and x is None:
            continue
        if c:
            feasible.x.append(x)
            feasible.y.append(y)
            feasible.trial_numbers.append(num)
        else:
            infeasible.x.append(x)
            infeasible.y.append(y)
            infeasible.trial_numbers.append(num)

    if subplot_info.is_numerical:
        feasible_x = feasible.x
        feasible_y = feasible.y
        feasible_c = feasible.trial_numbers
        infeasible_x = infeasible.x
        infeasible_y = infeasible.y
    else:
        feasible_x, feasible_y, feasible_c = _get_categorical_plot_values(subplot_info, feasible)
        infeasible_x, infeasible_y, _ = _get_categorical_plot_values(subplot_info, infeasible)

    trace.append(
        go.Scatter(
            x=feasible_x,
            y=feasible_y,
            mode="markers",
            name="Feasible Trial",
            marker={
                "line": {"width": 0.5, "color": "Grey"},
                "color": feasible_c,
                "colorscale": COLOR_SCALE,
                "colorbar": {
                    "title": "Trial",
                    "x": 1.0,  # Offset the colorbar position with a fixed width `xpad`.
                    "xpad": 40,
                },
            },
            showlegend=False,
        )
    )
    if len(infeasible_x) > 0:
        trace.append(
            go.Scatter(
                x=infeasible_x,
                y=infeasible_y,
                mode="markers",
                name="Infeasible Trial",
                marker={
                    "color": "#cccccc",
                },
                showlegend=False,
            )
        )

    return trace


def _get_categorical_plot_values(
    subplot_info: _SliceSubplotInfo, values: _PlotValues
) -> tuple[list[Any], list[float], list[int]]:
    assert subplot_info.x_labels is not None
    value_x = []
    value_y = []
    value_c = []
    points_dict = defaultdict(list)
    for x, y, number in zip(values.x, values.y, values.trial_numbers):
        points_dict[repr(x)].append((y, number))
    for x_label in subplot_info.x_labels:
        for y, number in points_dict[repr(x_label)]:
            value_x.append(repr(x_label))
            value_y.append(y)
            value_c.append(number)
    return value_x, value_y, value_c


def _get_categorical_labels(
    x_labels: tuple[Any, ...] | None,
) -> list[str] | None:
    if x_labels is None:
        return None
    return [repr(x_label) for x_label in x_labels]


def plot_slice(
    study: Study,
    params: list[str] | None = None,
    *,
    user_attrs: list[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> go.Figure:
    """Plot the parameter and user attribute relationship as slice plot in a study.

    This is an extension of :func:`optuna.visualization.plot_slice` that can also plot
    user attributes set by :meth:`optuna.trial.Trial.set_user_attr`.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        params:
            Parameter list to visualize. The default is all parameters.
            Pass an empty list to plot only user attributes.
        user_attrs:
            User attribute keys to visualize. The default is :obj:`None`, which plots no user
            attributes. Numbers are plotted on a linear axis. Booleans and strings are plotted as
            categories. Trials without the key or with :obj:`None` are not plotted. Other values,
            such as ``nan``, ``inf``, lists, and dicts, are not plotted with a warning.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """
    return _get_slice_plot(_get_slice_plot_info(study, params, user_attrs, target, target_name))
