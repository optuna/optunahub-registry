"""Unit tests for MultiMetricPruner and MultiMetricPrunerTrial."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from optuna.trial import TrialState
import optunahub
import pytest


_MODULE_DIR = os.path.join(os.path.dirname(__file__), "..")

module = optunahub.load_local_module("pruners/multi_metric_pruner", registry_root="package/")
MultiMetricPruner = module.MultiMetricPruner
MultiMetricPrunerTrial = module.MultiMetricPrunerTrial
_USER_ATTR_KEY = "multi_pruner:values"

_spec = importlib.util.spec_from_file_location(
    "_nondomination", os.path.join(_MODULE_DIR, "_hypervolume", "_nondomination.py")
)
_nondom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_nondom)
_fast_non_domination_rank = _nondom._fast_non_domination_rank


# ── helpers ───────────────────────────────────────────────────────────────────


class AlwaysPrunePruner(BasePruner):
    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return True


class NeverPrunePruner(BasePruner):
    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return False


def _make_pruner(
    base: BasePruner | None = None,
    *,
    directions: dict | None = None,
    joint: bool = True,
) -> MultiMetricPruner:
    if base is None:
        base = optuna.pruners.NopPruner()
    if directions is None:
        directions = {"loss": "minimize", "acc": "minimize"}
    return MultiMetricPruner(base, metric_directions=directions, joint=joint)


def _make_study(joint: bool = True, directions: dict | None = None) -> optuna.Study:
    return optuna.create_study(
        directions=["minimize", "minimize"],
        pruner=_make_pruner(directions=directions, joint=joint),
    )


def _ask_wrapped(study: optuna.Study) -> MultiMetricPrunerTrial:
    return MultiMetricPrunerTrial(study.ask())


def _add_complete_trial(
    study: optuna.Study,
    values_per_step: dict[int, dict[str, float]],
    final_values: list[float],
) -> None:
    """Add a finished trial with given intermediate values."""
    trial = study.ask()
    wrapped = MultiMetricPrunerTrial(trial)
    for step, vals in values_per_step.items():
        wrapped.report(vals, step)
    study.tell(trial, final_values)


# ── MultiMetricPruner.__init__ ────────────────────────────────────────────────


class TestMultiMetricPrunerInit:
    def test_valid_joint_true(self) -> None:
        pruner = _make_pruner(joint=True)
        assert pruner._joint is True
        assert set(pruner._metric_directions) == {"loss", "acc"}

    def test_valid_joint_false(self) -> None:
        pruner = _make_pruner(joint=False)
        assert pruner._joint is False

    def test_accepts_study_direction_enum(self) -> None:
        pruner = MultiMetricPruner(
            optuna.pruners.NopPruner(),
            metric_directions={
                "loss": StudyDirection.MINIMIZE,
                "acc": StudyDirection.MAXIMIZE,
            },
            joint=True,
        )
        assert pruner._metric_directions["loss"] == StudyDirection.MINIMIZE
        assert pruner._metric_directions["acc"] == StudyDirection.MAXIMIZE

    def test_empty_metric_directions_raises(self) -> None:
        with pytest.raises(ValueError, match="must have at least one entry"):
            MultiMetricPruner(optuna.pruners.NopPruner(), metric_directions={}, joint=True)

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'minimize' or 'maximize'"):
            MultiMetricPruner(
                optuna.pruners.NopPruner(),
                metric_directions={"loss": "neither"},
                joint=True,
            )

    def test_metric_directions_deepcopied(self) -> None:
        original = {"loss": "minimize"}
        pruner = MultiMetricPruner(
            optuna.pruners.NopPruner(), metric_directions=original, joint=True
        )
        original["loss"] = "maximize"
        assert pruner._metric_directions["loss"] == "minimize"


# ── MultiMetricPrunerTrial.report ─────────────────────────────────────────────


class TestMultiMetricPrunerTrialReport:
    def test_stores_values_in_user_attrs(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5, "acc": 0.8}, step=0)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert attrs["0"]["loss"] == pytest.approx(0.5)
        assert attrs["0"]["acc"] == pytest.approx(0.8)

    def test_int_value_cast_to_float(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 1}, step=0)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert isinstance(attrs["0"]["loss"], float)

    def test_non_castable_value_raises(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        with pytest.raises(TypeError, match="must be a dict of float"):
            wrapped.report({"loss": "not_a_number"}, step=0)

    def test_non_dict_raises(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        with pytest.raises(TypeError, match="must be a dict"):
            wrapped.report([0.5, 0.8], step=0)  # type: ignore[arg-type]

    def test_empty_dict_raises(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        with pytest.raises(ValueError, match="must have at least one entry"):
            wrapped.report({}, step=0)

    def test_negative_step_raises(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        with pytest.raises(ValueError, match="cannot be negative"):
            wrapped.report({"loss": 0.5}, step=-1)

    def test_unknown_metric_raises(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        with pytest.raises(ValueError, match="unknown metric"):
            wrapped.report({"unknown": 0.5}, step=0)

    def test_duplicate_warns_and_does_not_overwrite(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5}, step=0)
        with pytest.warns(Warning, match="already reported"):
            wrapped.report({"loss": 0.9}, step=0)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert attrs["0"]["loss"] == pytest.approx(0.5)

    def test_per_metric_splits_multi_dict(self) -> None:
        """joint=False should still store all metrics, but split per-metric calls."""
        study = _make_study(joint=False)
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5, "acc": 0.8}, step=0)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert attrs["0"]["loss"] == pytest.approx(0.5)
        assert attrs["0"]["acc"] == pytest.approx(0.8)

    def test_partial_report_allowed(self) -> None:
        """Reporting a subset of metric_directions keys must work (per-metric mode design intent)."""
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=MultiMetricPruner(
                optuna.pruners.NopPruner(),
                metric_directions={"train_loss": "minimize", "val_loss": "minimize"},
                joint=False,
            ),
        )
        wrapped = _ask_wrapped(study)
        # Only train_loss reported — val_loss is omitted intentionally.
        wrapped.report({"train_loss": 0.5}, step=0)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert "train_loss" in attrs["0"]
        assert "val_loss" not in attrs["0"]

    def test_step_stored_as_string_key(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.3, "acc": 0.7}, step=5)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert "5" in attrs

    def test_accumulates_values_across_steps(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        for step in range(5):
            wrapped.report({"loss": float(step), "acc": float(step) * 2}, step=step)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert len(attrs) == 5

    def test_getattr_delegates_to_inner_trial(self) -> None:
        study = _make_study()
        wrapped = _ask_wrapped(study)
        x = wrapped.suggest_float("x", 0.0, 1.0)
        assert 0.0 <= x <= 1.0
        assert "x" in wrapped.params


# ── MultiMetricPrunerTrial.should_prune ──────────────────────────────────────


class TestMultiMetricPrunerTrialShouldPrune:
    def test_wrong_pruner_raises(self) -> None:
        study = optuna.create_study(pruner=optuna.pruners.NopPruner())
        wrapped = MultiMetricPrunerTrial(study.ask())
        with pytest.raises(ValueError, match="MultiMetricPruner"):
            wrapped.should_prune()

    def test_no_reports_returns_false(self) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=_make_pruner(AlwaysPrunePruner()),
        )
        wrapped = _ask_wrapped(study)
        assert wrapped.should_prune() is False

    def test_joint_true_prunes_when_base_prunes(self) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=_make_pruner(AlwaysPrunePruner(), joint=True),
        )
        _add_complete_trial(study, {0: {"loss": 1.0, "acc": 1.0}}, [1.0, 1.0])
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 2.0, "acc": 2.0}, step=0)
        assert wrapped.should_prune() is True

    def test_joint_false_prunes_when_any_metric_prunes(self) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=_make_pruner(AlwaysPrunePruner(), joint=False),
        )
        _add_complete_trial(study, {0: {"loss": 1.0, "acc": 1.0}}, [1.0, 1.0])
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        assert wrapped.should_prune() is True

    def test_joint_false_does_not_prune_when_base_never_prunes(self) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=_make_pruner(NeverPrunePruner(), joint=False),
        )
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        assert wrapped.should_prune() is False

    def test_joint_false_with_metric_name_checks_only_that_metric(self) -> None:
        """should_prune(metric_name=...) routes to per-metric prune."""
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=_make_pruner(AlwaysPrunePruner(), joint=False),
        )
        _add_complete_trial(study, {0: {"loss": 1.0}}, [1.0, 1.0])
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5}, step=0)
        assert wrapped.should_prune(metric_name="loss") is True

    def test_joint_true_ignores_metric_name(self) -> None:
        """joint=True should use Pareto ranking regardless of metric_name arg."""
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=_make_pruner(NeverPrunePruner(), joint=True),
        )
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        # NeverPrunePruner means no pruning even with metric_name specified.
        assert wrapped.should_prune(metric_name="loss") is False


# ── MultiMetricPruner.prune routing ──────────────────────────────────────────


class TestMultiMetricPrunerPrune:
    def test_no_reports_returns_false(self) -> None:
        pruner = _make_pruner(AlwaysPrunePruner())
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        frozen = trial._get_latest_trial()
        assert pruner.prune(study, frozen) is False

    def test_joint_false_iterates_all_metrics_when_no_name(self) -> None:
        call_count: list[int] = [0]

        class CountingPruner(BasePruner):
            def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
                call_count[0] += 1
                return False

        pruner = _make_pruner(CountingPruner(), joint=False)
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        frozen = trial._get_latest_trial()
        pruner.prune(study, frozen)
        assert call_count[0] == 2  # once per metric

    def test_joint_false_short_circuits_on_first_prune(self) -> None:
        call_count: list[int] = [0]

        class FirstAlwaysPruner(BasePruner):
            def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
                call_count[0] += 1
                return True  # always prune

        pruner = _make_pruner(FirstAlwaysPruner(), joint=False)
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        frozen = trial._get_latest_trial()
        result = pruner.prune(study, frozen)
        assert result is True
        assert call_count[0] == 1  # stopped after first True

    def test_joint_false_with_metric_name_calls_base_once(self) -> None:
        call_count: list[int] = [0]

        class CountingPruner(BasePruner):
            def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
                call_count[0] += 1
                return False

        pruner = _make_pruner(CountingPruner(), joint=False)
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        frozen = trial._get_latest_trial()
        pruner.prune(study, frozen, metric_name="loss")
        assert call_count[0] == 1


# ── Pareto bonus (joint=True) ─────────────────────────────────────────────────


class TestParetoBonus:
    def test_pareto_front_trial_not_pruned(self) -> None:
        """A Pareto-dominant trial must not be pruned even when MedianPruner would prune it."""
        # MedianPruner with n_startup_trials=0 prunes aggressively.
        pruner = MultiMetricPruner(
            optuna.pruners.MedianPruner(n_startup_trials=0),
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=True,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        # Add completed trials with bad values (high Pareto rank).
        for val in [5.0, 6.0, 7.0, 8.0]:
            _add_complete_trial(study, {0: {"loss": val, "acc": val}}, [val, val])
        # Current trial is clearly on the Pareto front (dominates all prior trials).
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 0.1, "acc": 0.1}, step=0)
        assert not wrapped.should_prune()

    def test_dominated_trial_can_be_pruned(self) -> None:
        """A trial dominated by others should be prunable."""
        pruner = MultiMetricPruner(
            optuna.pruners.MedianPruner(n_startup_trials=0),
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=True,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        # Add completed trials with good values.
        for val in [0.1, 0.2, 0.3, 0.4]:
            _add_complete_trial(study, {0: {"loss": val, "acc": val}}, [val, val])
        # Current trial is dominated by all prior trials.
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 10.0, "acc": 10.0}, step=0)
        assert wrapped.should_prune()


# ── End-to-end tests from example.py ─────────────────────────────────────────


class TestEndToEnd:
    def test_multi_metric_mode(self) -> None:
        """Mode 1: joint=True with Pareto-ranked intermediate values."""

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            trial = MultiMetricPrunerTrial(trial)
            x = trial.suggest_float("x", -5.0, 5.0)
            for step in range(10):
                trial.report({"loss": (x - step * 0.1) ** 2, "acc": (x + step * 0.1) ** 2}, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return x**2, (x - 2.0) ** 2

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=MultiMetricPruner(
                optuna.pruners.MedianPruner(n_startup_trials=3),
                metric_directions={"loss": "minimize", "acc": "minimize"},
                joint=True,
            ),
        )
        study.optimize(objective, n_trials=30)
        assert len(study.trials) == 30
        assert any(t.state == TrialState.PRUNED for t in study.trials)

    def test_per_metric_mode(self) -> None:
        """Mode 2: joint=False, each metric evaluated independently."""

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            trial = MultiMetricPrunerTrial(trial)
            x = trial.suggest_float("x", -5.0, 5.0)
            for step in range(10):
                loss = (x - step * 0.1) ** 2
                acc = 1.0 / (1.0 + (x + step * 0.1) ** 2)
                trial.report({"loss": loss, "acc": acc}, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return x**2, 1.0 / (1.0 + (x - 2.0) ** 2)

        study = optuna.create_study(
            directions=["minimize", "maximize"],
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=MultiMetricPruner(
                optuna.pruners.MedianPruner(n_startup_trials=3),
                metric_directions={"loss": "minimize", "acc": "maximize"},
                joint=False,
            ),
        )
        study.optimize(objective, n_trials=30)
        assert len(study.trials) == 30
        assert any(t.state == TrialState.PRUNED for t in study.trials)

    def test_mixed_frequency_mode(self) -> None:
        """Mode 3: metrics reported at different step intervals."""

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            trial = MultiMetricPrunerTrial(trial)
            x = trial.suggest_float("x", -5.0, 5.0)
            for step in range(10):
                trial.report({"train_loss": (x - step * 0.1) ** 2}, step)
                if trial.should_prune(metric_name="train_loss"):
                    raise optuna.TrialPruned()
                if step % 5 == 0:
                    trial.report({"val_loss": (x + step * 0.05) ** 2}, step)
                    if trial.should_prune(metric_name="val_loss"):
                        raise optuna.TrialPruned()
            return x**2, (x - 2.0) ** 2

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=MultiMetricPruner(
                optuna.pruners.MedianPruner(n_startup_trials=3),
                metric_directions={"train_loss": "minimize", "val_loss": "minimize"},
                joint=False,
            ),
        )
        study.optimize(objective, n_trials=30)
        assert len(study.trials) == 30
        assert any(t.state == TrialState.PRUNED for t in study.trials)

    def test_per_metric_coexists_with_partial_reports(self) -> None:
        """Per-metric mode: reporting a subset of metric_directions must not error."""

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            trial = MultiMetricPrunerTrial(trial)
            x = trial.suggest_float("x", -5.0, 5.0)
            for step in range(5):
                # Only train_loss is reported; val_loss is never reported.
                trial.report({"train_loss": (x - step * 0.1) ** 2}, step)
                if trial.should_prune(metric_name="train_loss"):
                    raise optuna.TrialPruned()
            return x**2, (x - 2.0) ** 2

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=MultiMetricPruner(
                optuna.pruners.MedianPruner(n_startup_trials=3),
                metric_directions={"train_loss": "minimize", "val_loss": "minimize"},
                joint=False,
            ),
        )
        study.optimize(objective, n_trials=15)
        assert len(study.trials) == 15


# ── _fast_non_domination_rank ─────────────────────────────────────────────────


class TestFastNonDominationRank:
    def test_empty_returns_empty(self) -> None:
        result = _fast_non_domination_rank(np.empty((0, 2), dtype=float))
        assert len(result) == 0

    def test_single_point_rank_zero(self) -> None:
        ranks = _fast_non_domination_rank(np.array([[1.0, 2.0]]))
        assert ranks[0] == 0

    def test_dominated_point_has_higher_rank(self) -> None:
        # [1, 1] dominates [2, 2]
        ranks = _fast_non_domination_rank(np.array([[1.0, 1.0], [2.0, 2.0]]))
        assert ranks[0] < ranks[1]

    def test_pareto_points_same_rank(self) -> None:
        # Neither [1, 2] nor [2, 1] dominates the other.
        ranks = _fast_non_domination_rank(np.array([[1.0, 2.0], [2.0, 1.0]]))
        assert ranks[0] == ranks[1] == 0

    def test_three_points_two_fronts(self) -> None:
        # [1, 3] and [3, 1] are on rank-0 front; [4, 4] is rank 1.
        ranks = _fast_non_domination_rank(np.array([[1.0, 3.0], [3.0, 1.0], [4.0, 4.0]]))
        assert ranks[0] == 0
        assert ranks[1] == 0
        assert ranks[2] == 1

    def test_single_objective_ranks_by_value(self) -> None:
        values = np.array([[3.0], [1.0], [2.0]])
        ranks = _fast_non_domination_rank(values)
        # 1.0 -> rank 0, 2.0 -> rank 1, 3.0 -> rank 2
        assert ranks[1] == 0
        assert ranks[2] == 1
        assert ranks[0] == 2

    def test_three_objectives(self) -> None:
        # [1,1,1] dominates all others.
        # [1,2,2] and [2,1,2] dominate [2,2,2] but not each other → rank 1.
        # [2,2,2] is dominated by all → rank 2.
        values = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 2.0, 2.0], [2.0, 1.0, 2.0]])
        ranks = _fast_non_domination_rank(values)
        assert ranks[0] == 0  # [1,1,1]: sole Pareto front
        assert ranks[2] == 1  # [1,2,2]: rank 1
        assert ranks[3] == 1  # [2,1,2]: rank 1
        assert ranks[1] == 2  # [2,2,2]: dominated by all others
