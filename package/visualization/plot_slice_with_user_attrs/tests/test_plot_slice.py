from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Any

import numpy as np
import optuna
import pytest

from .. import plot_slice
from .._plot_slice import _CATEGORICAL
from .._plot_slice import _classify_value
from .._plot_slice import _NUMERICAL
from .._plot_slice import _SKIP


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna.logging.enable_propagation()
    yield
    optuna.logging.disable_propagation()
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def _create_study(user_attr_values: list[Any], key: str = "attr") -> optuna.Study:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for value in user_attr_values:
        trial = study.ask()
        trial.suggest_float("x", 0.0, 1.0)
        trial.suggest_categorical("k", ["a", "b"])
        if value is not ...:
            trial.set_user_attr(key, value)
        study.tell(trial, float(trial.number))
    return study


def _x_titles(figure: Any) -> list[str]:
    n_axes = len([k for k in figure.layout if k.startswith("xaxis")])
    names = ["xaxis"] + [f"xaxis{i}" for i in range(2, n_axes + 1)]
    return [figure.layout[name].title.text for name in names]


def _plot_slice_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.endswith("_plot_slice")]


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, (_SKIP, None)),
        (True, (_CATEGORICAL, None)),
        (False, (_CATEGORICAL, None)),
        (1, (_NUMERICAL, None)),
        (1.5, (_NUMERICAL, None)),
        (np.int64(3), (_NUMERICAL, None)),
        (np.float32(0.5), (_NUMERICAL, None)),
        (math.nan, (_SKIP, "non-finite")),
        (math.inf, (_SKIP, "non-finite")),
        (np.float32("nan"), (_SKIP, "non-finite")),
        ("adam", (_CATEGORICAL, None)),
        ([1], (_SKIP, "unsupported type")),
        ({"a": 1}, (_SKIP, "unsupported type")),
        (np.bool_(True), (_SKIP, "unsupported type")),
    ],
)
def test_classify_value(value: Any, expected: tuple[str, str | None]) -> None:
    assert _classify_value(value) == expected


@pytest.mark.parametrize("params", [None, [], ["x"], ["k"], ["k", "x"]])
def test_same_as_builtin_without_user_attrs(params: list[str] | None) -> None:
    study = _create_study([1.0, 2.0, 3.0])
    figure = plot_slice(study, params=params)
    expected = optuna.visualization.plot_slice(study, params=params)
    assert figure.to_dict() == expected.to_dict()


def test_numerical_user_attr() -> None:
    study = _create_study([0.1, 0.2, 0.3, np.int64(4)])
    figure = plot_slice(study, params=[], user_attrs=["attr"])
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [0.1, 0.2, 0.3, 4]
    assert figure.layout.xaxis.title.text == "user_attrs.attr"
    assert figure.layout.xaxis.type is None


def test_categorical_user_attr_keeps_true_1_and_1_0_separate() -> None:
    study = _create_study([1, True, 1.0, "adam", 1, True])
    figure = plot_slice(study, params=[], user_attrs=["attr"])
    assert figure.layout.xaxis.type == "category"
    assert list(figure.layout.xaxis.categoryarray) == ["1", "True", "1.0", "'adam'"]
    assert list(figure.data[0].x) == ["1", "1", "True", "True", "1.0", "'adam'"]
    assert list(figure.data[0].y) == [0.0, 4.0, 1.0, 5.0, 2.0, 3.0]


def test_subplot_order_is_params_then_user_attrs() -> None:
    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("b_attr", 0.5)
        trial.set_user_attr("attr", 1.0)
        trial.suggest_float("x", 0.0, 1.0)
        trial.suggest_categorical("k", ["a", "b"])
        return 0.0

    study = optuna.create_study()
    study.optimize(objective, n_trials=2)
    figure = plot_slice(study, user_attrs=["b_attr", "attr"])
    assert _x_titles(figure) == ["k", "x", "user_attrs.attr", "user_attrs.b_attr"]


def test_missing_key_and_none_are_skipped_silently(caplog: pytest.LogCaptureFixture) -> None:
    study = _create_study([1.0, None, ..., 2.0])
    figure = plot_slice(study, params=[], user_attrs=["attr"])
    assert list(figure.data[0].x) == [1.0, 2.0]
    assert _plot_slice_warnings(caplog) == []


def test_invalid_values_warn_once_per_reason(caplog: pytest.LogCaptureFixture) -> None:
    values: list[Any] = [math.nan] * 7 + [[1], 1.0]
    study = _create_study(values)
    figure = plot_slice(study, params=[], user_attrs=["attr"])
    assert list(figure.data[0].x) == [1.0]
    assert _plot_slice_warnings(caplog) == [
        (
            "7 trial(s) are omitted for user_attrs.attr because of non-finite values "
            "(trial numbers: 0, 1, 2, 3, 4, ...)."
        ),
        (
            "1 trial(s) are omitted for user_attrs.attr because of unsupported type values "
            "(trial numbers: 7)."
        ),
    ]


def test_warnings_follow_optuna_verbosity(caplog: pytest.LogCaptureFixture) -> None:
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = _create_study([math.nan, 1.0])
    plot_slice(study, params=[], user_attrs=["attr"])
    assert _plot_slice_warnings(caplog) == []


def test_all_invalid_key_gives_empty_subplot() -> None:
    study = _create_study([[1], {"a": 1}])
    figure = plot_slice(study, params=[], user_attrs=["attr"])
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == []


@pytest.mark.filterwarnings("ignore")
def test_infeasible_trials_are_plotted_for_categorical_values() -> None:
    def constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
        return [1.0 if trial.number % 2 else -1.0]

    sampler = optuna.samplers.TPESampler(constraints_func=constraints, seed=0)
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("attr", "adam" if trial.number % 3 else "sgd")
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=6)
    figure = plot_slice(study, params=[], user_attrs=["attr"])
    assert [(d.name, len(d.x)) for d in figure.data] == [
        ("Feasible Trial", 3),
        ("Infeasible Trial", 3),
    ]


def test_nonfinite_objective_values_are_omitted() -> None:
    study = _create_study([1.0, 2.0])
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": 0.5, "k": "a"},
            distributions=study.trials[0].distributions,
            value=math.inf,
            user_attrs={"attr": 3.0},
        )
    )
    figure = plot_slice(study, params=[], user_attrs=["attr"])
    assert list(figure.data[0].x) == [1.0, 2.0]


def test_empty_study() -> None:
    figure = plot_slice(optuna.create_study(), user_attrs=["attr"])
    assert len(figure.data) == 0


def test_multi_objective_with_target() -> None:
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(lambda t: (t.set_user_attr("attr", 1.0), (0.0, 1.0))[1], n_trials=2)
    figure = plot_slice(
        study, user_attrs=["attr"], target=lambda t: t.values[1], target_name="Second"
    )
    assert figure.layout.yaxis.title.text == "Second"
    assert list(figure.data[-1].y) == [1.0, 1.0]


def test_error_if_user_attrs_is_str() -> None:
    study = _create_study([1.0])
    with pytest.raises(TypeError):
        plot_slice(study, user_attrs="attr")  # type: ignore[arg-type]


def test_error_if_user_attr_does_not_exist() -> None:
    study = _create_study([1.0])
    with pytest.raises(ValueError):
        plot_slice(study, user_attrs=["unknown"])


def test_error_if_multi_objective_without_target() -> None:
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(lambda t: (t.set_user_attr("attr", 1.0), (0.0, 1.0))[1], n_trials=1)
    with pytest.raises(ValueError):
        plot_slice(study, user_attrs=["attr"])


def test_warning_if_target_name_is_default() -> None:
    study = _create_study([1.0])
    with pytest.warns(UserWarning):
        plot_slice(study, user_attrs=["attr"], target=lambda t: t.values[0])
