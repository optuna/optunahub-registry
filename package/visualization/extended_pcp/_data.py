from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any
from typing import NamedTuple
from typing import Optional
from typing import Sequence

import numpy as np
from optuna.distributions import CategoricalDistribution
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.study._multi_objective import _fast_non_domination_rank
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


DEFAULT_MISSING_LABEL = "None"
_CONSTRAINTS_KEY = "constraints"
_VALID_LOW_WITH_MISSING = 0.12
_MISSING_VALUE = object()


class _DimensionInfo(NamedTuple):
    label: str
    values: tuple[float, ...]
    tickvals: tuple[float, ...]
    ticktext: tuple[str, ...]


class _PlotInfo(NamedTuple):
    dimensions: list[_DimensionInfo]
    color_values: tuple[float, ...]
    constraints_satisfied: tuple[bool, ...]
    color_label: str
    reverse_scale: bool
    is_rank_color: bool
    draw_order: tuple[int, ...]


def _get_plot_info(
    study: Study,
    params: Optional[list[str]],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    objective_names: Optional[list[str]],
    missing_label: str,
) -> _PlotInfo:
    trials = list(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    is_multi_objective = target is None and len(study.directions) > 1

    if is_multi_objective:
        multi_objective_values, trials = _get_multi_objective_values(study, trials)
        objective_labels = _get_objective_labels(study, objective_names)
        objective_columns = [list(column) for column in zip(*multi_objective_values)]
        objective_dims = [
            _build_numeric_dimension(label, values, missing_label=None, is_log=False)
            for label, values in zip(objective_labels, objective_columns)
        ]
        pareto_ranks = _calculate_pareto_ranks(multi_objective_values, study.directions)
        draw_order = tuple(np.argsort(np.asarray(pareto_ranks))[::-1].tolist())
        color_values = tuple(float(rank) for rank in pareto_ranks)
        color_label = "Pareto Rank"
        reverse_scale = True
        is_rank_color = True
    else:
        target_values, trials = _get_target_values(study, trials, target)
        if len(target_values) == 0:
            objective_dims = []
        else:
            objective_dims = [
                _build_numeric_dimension(
                    target_name, target_values, missing_label=None, is_log=False
                )
            ]
        draw_order = tuple(range(len(trials)))
        color_values = tuple(float(v) for v in target_values)
        color_label = target_name
        reverse_scale = target is not None or (
            len(study.directions) == 1 and study.direction == StudyDirection.MINIMIZE
        )
        is_rank_color = False

    if len(trials) == 0:
        return _PlotInfo(
            dimensions=[],
            color_values=(),
            constraints_satisfied=(),
            color_label=color_label,
            reverse_scale=reverse_scale,
            is_rank_color=is_rank_color,
            draw_order=(),
        )

    param_names = _get_param_names(trials, params)
    param_dims = [
        _build_param_dimension(trials, param_name, missing_label) for param_name in param_names
    ]

    return _PlotInfo(
        dimensions=objective_dims + param_dims,
        color_values=color_values,
        constraints_satisfied=tuple(_satisfies_constraints(trial) for trial in trials),
        color_label=color_label,
        reverse_scale=reverse_scale,
        is_rank_color=is_rank_color,
        draw_order=draw_order,
    )


def _get_multi_objective_values(
    study: Study, trials: list[FrozenTrial]
) -> tuple[list[tuple[float, ...]], list[FrozenTrial]]:
    n_objectives = len(study.directions)
    values = []
    filtered_trials = []
    for trial in trials:
        if trial.values is None or len(trial.values) != n_objectives:
            continue
        trial_values = tuple(float(v) for v in trial.values)
        if not all(math.isfinite(v) for v in trial_values):
            continue
        values.append(trial_values)
        filtered_trials.append(trial)
    return values, filtered_trials


def _get_target_values(
    study: Study,
    trials: list[FrozenTrial],
    target: Optional[Callable[[FrozenTrial], float]],
) -> tuple[list[float], list[FrozenTrial]]:
    values = []
    filtered_trials = []
    for trial in trials:
        value = target(trial) if target is not None else trial.value
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        values.append(value)
        filtered_trials.append(trial)
    return values, filtered_trials


def _satisfies_constraints(trial: FrozenTrial) -> bool:
    constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
    return constraints is None or all(float(value) <= 0.0 for value in constraints)


def _get_objective_labels(study: Study, objective_names: Optional[list[str]]) -> list[str]:
    n_objectives = len(study.directions)
    if objective_names is None:
        return [f"Objective {i}" for i in range(n_objectives)]
    if len(objective_names) != n_objectives:
        raise ValueError(
            "The length of objective_names must match the number of study objectives."
        )
    return objective_names


def _get_param_names(trials: list[FrozenTrial], params: Optional[list[str]]) -> list[str]:
    all_params = {param_name for trial in trials for param_name in trial.params}
    if params is None:
        return sorted(all_params)

    missing_params = [param_name for param_name in params if param_name not in all_params]
    if missing_params:
        raise ValueError(
            "Parameter {} does not exist in completed trials.".format(", ".join(missing_params))
        )
    return list(params)


def _build_param_dimension(
    trials: list[FrozenTrial], param_name: str, missing_label: str
) -> _DimensionInfo:
    is_categorical = any(
        isinstance(trial.distributions.get(param_name), CategoricalDistribution)
        for trial in trials
        if param_name in trial.distributions
    )
    values: list[Any] = [
        trial.params[param_name] if param_name in trial.params else _MISSING_VALUE
        for trial in trials
    ]

    if is_categorical or not all(_is_numeric_value(v) for v in values if v is not _MISSING_VALUE):
        return _build_categorical_dimension(_truncate_label(param_name), values, missing_label)

    numeric_values: list[Optional[float]] = []
    for value in values:
        if value is _MISSING_VALUE:
            numeric_values.append(None)
        else:
            numeric_values.append(float(value))
    return _build_numeric_dimension(
        _truncate_label(param_name),
        numeric_values,
        missing_label=missing_label,
        is_log=_is_log_scale(trials, param_name),
    )


def _build_numeric_dimension(
    label: str,
    values: Sequence[Optional[float]],
    *,
    missing_label: Optional[str],
    is_log: bool,
) -> _DimensionInfo:
    has_missing = missing_label is not None and any(value is None for value in values)
    valid_values = [float(value) for value in values if value is not None]

    if len(valid_values) == 0:
        return _DimensionInfo(
            label=label,
            values=tuple(0.0 for _ in values),
            tickvals=(0.0,),
            ticktext=(missing_label or DEFAULT_MISSING_LABEL,),
        )

    transformed_values = [_transform_numeric_value(value, is_log) for value in valid_values]
    min_value = min(transformed_values)
    max_value = max(transformed_values)

    scaled_values = []
    for value in values:
        if value is None:
            scaled_values.append(0.0)
            continue
        transformed = _transform_numeric_value(float(value), is_log)
        scaled_values.append(_scale_numeric_value(transformed, min_value, max_value, has_missing))

    ticks: list[float] = []
    ticktext: list[str] = []
    if has_missing:
        assert missing_label is not None
        ticks.append(0.0)
        ticktext.append(missing_label)

    for tick in _get_numeric_ticks(min_value, max_value, is_log):
        ticks.append(_scale_numeric_value(tick, min_value, max_value, has_missing))
        ticktext.append(_format_numeric_tick(tick, is_log))

    ticks, ticktext = _dedupe_ticks(ticks, ticktext)
    return _DimensionInfo(
        label=label,
        values=tuple(scaled_values),
        tickvals=tuple(ticks),
        ticktext=tuple(ticktext),
    )


def _build_categorical_dimension(
    label: str, values: Sequence[Any], missing_label: str
) -> _DimensionInfo:
    has_missing = any(value is _MISSING_VALUE for value in values)
    categories = _get_categories(values)
    if len(categories) == 0:
        return _DimensionInfo(
            label=label,
            values=tuple(0.0 for _ in values),
            tickvals=(0.0,),
            ticktext=(missing_label,),
        )

    category_positions = _get_category_positions(len(categories), has_missing)
    category_to_position = {
        _category_key(category): position
        for category, position in zip(categories, category_positions)
    }

    scaled_values = []
    for value in values:
        if value is _MISSING_VALUE:
            scaled_values.append(0.0)
        else:
            scaled_values.append(category_to_position[_category_key(value)])

    ticks = []
    ticktext = []
    if has_missing:
        ticks.append(0.0)
        ticktext.append(missing_label)
    ticks.extend(category_positions)
    ticktext.extend(
        _format_category(category, missing_label, has_missing) for category in categories
    )

    return _DimensionInfo(
        label=label,
        values=tuple(scaled_values),
        tickvals=tuple(ticks),
        ticktext=tuple(ticktext),
    )


def _get_categories(values: Sequence[Any]) -> list[Any]:
    observed = [value for value in values if value is not _MISSING_VALUE]
    if all(_is_numeric_value(value) for value in observed):
        return sorted(observed)

    categories: list[Any] = []
    seen_keys = set()
    for value in observed:
        key = _category_key(value)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        categories.append(value)
    return sorted(categories, key=lambda value: _category_key(value))


def _is_log_scale(trials: list[FrozenTrial], param_name: str) -> bool:
    return any(
        bool(getattr(trial.distributions.get(param_name), "log", False))
        for trial in trials
        if param_name in trial.distributions
    )


def _is_numeric_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float, np.number)):
        return False
    return math.isfinite(float(value))


def _transform_numeric_value(value: float, is_log: bool) -> float:
    if not is_log:
        return value
    if value <= 0:
        raise ValueError("Log-scale parameters must have positive values.")
    return math.log10(value)


def _scale_numeric_value(
    value: float, min_value: float, max_value: float, has_missing: bool
) -> float:
    valid_low = _VALID_LOW_WITH_MISSING if has_missing else 0.0
    if min_value == max_value:
        return (valid_low + 1.0) / 2.0
    return valid_low + (value - min_value) / (max_value - min_value) * (1.0 - valid_low)


def _get_numeric_ticks(min_value: float, max_value: float, is_log: bool) -> list[float]:
    if min_value == max_value:
        return [min_value]

    if not is_log:
        return [float(v) for v in np.linspace(min_value, max_value, num=5)]

    tick_values = [float(v) for v in range(math.ceil(min_value), math.floor(max_value) + 1)]
    if not any(math.isclose(min_value, tick, rel_tol=0.0, abs_tol=1e-12) for tick in tick_values):
        tick_values.insert(0, min_value)
    if not any(math.isclose(max_value, tick, rel_tol=0.0, abs_tol=1e-12) for tick in tick_values):
        tick_values.append(max_value)
    return tick_values


def _format_numeric_tick(value: float, is_log: bool) -> str:
    if is_log:
        value = math.pow(10, value)
    return f"{value:.3g}"


def _get_category_positions(n_categories: int, has_missing: bool) -> list[float]:
    if n_categories == 1:
        return [(1.0 + (_VALID_LOW_WITH_MISSING if has_missing else 0.0)) / 2.0]

    low = _VALID_LOW_WITH_MISSING if has_missing else 0.0
    return [float(v) for v in np.linspace(low, 1.0, num=n_categories)]


def _format_category(value: Any, missing_label: str, has_missing: bool) -> str:
    label = str(value)
    if has_missing and label == missing_label:
        return f"{label} (value)"
    return label


def _category_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _dedupe_ticks(ticks: list[float], ticktext: list[str]) -> tuple[list[float], list[str]]:
    deduped_ticks = []
    deduped_text = []
    seen = set()
    for tick, text in zip(ticks, ticktext):
        key = round(tick, 12)
        if key in seen:
            continue
        seen.add(key)
        deduped_ticks.append(tick)
        deduped_text.append(text)
    return deduped_ticks, deduped_text


def _calculate_pareto_ranks(
    values: Sequence[Sequence[float]], directions: Sequence[StudyDirection]
) -> list[int]:
    if len(values) == 0:
        return []

    loss_values = np.asarray(values, dtype=np.float64)
    for objective_id, direction in enumerate(directions):
        if direction == StudyDirection.MAXIMIZE:
            loss_values[:, objective_id] *= -1

    return _fast_non_domination_rank(loss_values).tolist()


def _truncate_label(label: str) -> str:
    return label if len(label) <= 20 else f"{label[:17]}..."
