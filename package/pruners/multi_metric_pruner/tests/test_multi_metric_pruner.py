"""Unit tests for MultiMetricPruner and MultiMetricPrunerTrial."""

from __future__ import annotations

import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from optuna.trial import TrialState
import pytest

from multi_metric_pruner import MultiMetricPruner
from multi_metric_pruner import MultiMetricPrunerTrial
from multi_metric_pruner._hypervolume._nondomination import _fast_non_domination_rank
from multi_metric_pruner._hypervolume._ordering import _argsort_by_hv_contribution
from multi_metric_pruner._hypervolume.hssp import _solve_hssp
from multi_metric_pruner._pruner import _tie_break
from multi_metric_pruner._pruner import _USER_ATTR_KEY


# ── helpers ───────────────────────────────────────────────────────────────────


class AlwaysPrunePruner(BasePruner):
    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return True


class NeverPrunePruner(BasePruner):
    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return False


class CountingPruner(BasePruner):
    def __init__(self) -> None:
        self.call_count = 0

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        self.call_count += 1
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
        original: dict[str, str] = {"loss": "minimize"}
        pruner = MultiMetricPruner(
            optuna.pruners.NopPruner(),
            metric_directions=original,  # type: ignore[arg-type]
            joint=True,
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
            wrapped.report({"loss": "not_a_number"}, step=0)  # type: ignore[arg-type,dict-item]

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

    def test_is_instance_of_optuna_trial(self) -> None:
        """Integrations that check isinstance(trial, optuna.Trial) must accept the wrapper."""
        study = _make_study()
        wrapped = _ask_wrapped(study)
        assert isinstance(wrapped, optuna.Trial)


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

    def test_joint_true_metric_name_ignored_with_real_pruner(self) -> None:
        """With joint=True and a real base pruner, metric_name must not change the outcome."""
        pruner = MultiMetricPruner(
            optuna.pruners.MedianPruner(n_startup_trials=0),
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=True,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        for val in [0.1, 0.2, 0.3, 0.4]:
            _add_complete_trial(study, {0: {"loss": val, "acc": val}}, [val, val])
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 10.0, "acc": 10.0}, step=0)
        # Dominated trial — all three call forms must agree and return True.
        assert wrapped.should_prune()
        assert wrapped.should_prune(metric_name="loss")
        assert wrapped.should_prune(metric_name="acc")


# ── MultiMetricPruner.prune routing ──────────────────────────────────────────


class TestMultiMetricPrunerPrune:
    def test_no_reports_returns_false(self) -> None:
        pruner = _make_pruner(AlwaysPrunePruner())
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        frozen = trial._get_latest_trial()
        assert pruner.prune(study, frozen) is False

    def test_joint_false_iterates_all_metrics_when_no_name(self) -> None:
        base = CountingPruner()
        pruner = _make_pruner(base, joint=False)
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        frozen = trial._get_latest_trial()
        pruner.prune(study, frozen)
        assert base.call_count == 2  # once per metric

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
        base = CountingPruner()
        pruner = _make_pruner(base, joint=False)
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        frozen = trial._get_latest_trial()
        pruner.prune(study, frozen, metric_name="loss")
        assert base.call_count == 1


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


# ── Pareto with maximize direction ───────────────────────────────────────────


class TestParetoWithMaximizeDirection:
    def test_pareto_front_protected_with_maximize_metric(self) -> None:
        """A trial that excels on the maximize metric is on the Pareto front → not pruned."""
        pruner = MultiMetricPruner(
            optuna.pruners.MedianPruner(n_startup_trials=0),
            metric_directions={"loss": "minimize", "acc": "maximize"},
            joint=True,
        )
        study = optuna.create_study(directions=["minimize", "maximize"], pruner=pruner)
        # Completed trials: good loss, bad acc.
        for val in [2.0, 3.0, 4.0, 5.0]:
            _add_complete_trial(study, {0: {"loss": val, "acc": 0.1}}, [val, 0.1])
        # Current: bad loss, good acc — neither side dominates the other → Pareto front.
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 10.0, "acc": 0.9}, step=0)
        assert not wrapped.should_prune()

    def test_dominated_trial_pruned_with_maximize_metric(self) -> None:
        """A trial worse on both minimize and maximize metrics is prunable."""
        pruner = MultiMetricPruner(
            optuna.pruners.MedianPruner(n_startup_trials=0),
            metric_directions={"loss": "minimize", "acc": "maximize"},
            joint=True,
        )
        study = optuna.create_study(directions=["minimize", "maximize"], pruner=pruner)
        # Completed trials: good loss AND good acc.
        for val in [0.1, 0.2, 0.3, 0.4]:
            _add_complete_trial(study, {0: {"loss": val, "acc": 0.9}}, [val, 0.9])
        # Current: bad loss AND bad acc → dominated by all completed trials.
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 5.0, "acc": 0.1}, step=0)
        assert wrapped.should_prune()


# ── Pareto rank changes across steps ─────────────────────────────────────────


class TestParetoRankAcrossSteps:
    def test_dominated_at_early_step_but_pareto_front_at_latest_not_pruned(self) -> None:
        """A trial dominated at step 0 but Pareto-dominant at step 1 must not be pruned."""
        pruner = MultiMetricPruner(
            optuna.pruners.MedianPruner(n_startup_trials=0),
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=True,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        # Completed: strong at step 0, weak at step 1.
        for _ in range(4):
            _add_complete_trial(
                study,
                {0: {"loss": 0.1, "acc": 0.1}, 1: {"loss": 5.0, "acc": 5.0}},
                [0.1, 0.1],
            )
        # Current: weak at step 0 (dominated), dominant at step 1.
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 5.0, "acc": 5.0}, step=0)
        wrapped.report({"loss": 0.01, "acc": 0.01}, step=1)
        assert not wrapped.should_prune()

    def test_pareto_dominance_at_any_step_protects_from_pruning(self) -> None:
        """A trial Pareto-dominant at step 0 stays protected even after degrading at step 1.

        MedianPruner uses the trial's best intermediate value across all steps, so a trial
        that achieves a very negative synthetic rank at step 0 (-1.0) keeps that protection
        regardless of what happens at the latest step.
        """
        pruner = MultiMetricPruner(
            optuna.pruners.MedianPruner(n_startup_trials=0),
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=True,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        # Completed: weak at step 0, strong at step 1.
        for _ in range(4):
            _add_complete_trial(
                study,
                {0: {"loss": 5.0, "acc": 5.0}, 1: {"loss": 0.1, "acc": 0.1}},
                [0.1, 0.1],
            )
        # Current: dominant at step 0 (synthetic rank -1.0), then dominated at step 1.
        # The best-over-steps value is -1.0, which is below the median → not pruned.
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 0.01, "acc": 0.01}, step=0)
        wrapped.report({"loss": 10.0, "acc": 10.0}, step=1)
        assert not wrapped.should_prune()


# ── Tie-break bonus integration ───────────────────────────────────────────────


class TestTieBreakBonusIntegration:
    """Verify that HV-contribution tie-breaking within a shared Pareto rank affects pruning.

    Setup: one completed trial at rank 0 ([1,1]) anchors the non-dominated front.
    Five identical rank-1 completed trials create a majority at a known synthetic rank range.
    The current trial is also at rank 1 but differs in HV contribution, so the tie-break
    bonus is either -0.5 (high contribution → lower synthetic rank → not pruned) or -0.1
    (low contribution → higher synthetic rank → pruned) relative to the median.
    """

    def test_high_hv_contribution_not_pruned(self) -> None:
        # Completed rank-1 trials: 5×[3,3] — lower HV contribution than [2,4].
        # Their synthetic ranks fill [0.58, …, 0.90]; median ≈ 0.70.
        # Current [2,4]: bonus -0.5 → synthetic rank 0.50 < median → not pruned.
        pruner = MultiMetricPruner(
            optuna.pruners.MedianPruner(n_startup_trials=0),
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=True,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        _add_complete_trial(study, {0: {"loss": 1.0, "acc": 1.0}}, [1.0, 1.0])
        for _ in range(5):
            _add_complete_trial(study, {0: {"loss": 3.0, "acc": 3.0}}, [3.0, 3.0])
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 2.0, "acc": 4.0}, step=0)
        assert not wrapped.should_prune()

    def test_low_hv_contribution_pruned(self) -> None:
        # Completed rank-1 trials: 5×[2,4] — higher HV contribution than [3,3].
        # Their synthetic ranks fill [0.50, …, 0.82]; median ≈ 0.62.
        # Current [3,3]: bonus -0.1 → synthetic rank 0.90 > median → pruned.
        pruner = MultiMetricPruner(
            optuna.pruners.MedianPruner(n_startup_trials=0),
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=True,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        _add_complete_trial(study, {0: {"loss": 1.0, "acc": 1.0}}, [1.0, 1.0])
        for _ in range(5):
            _add_complete_trial(study, {0: {"loss": 2.0, "acc": 4.0}}, [2.0, 4.0])
        trial = study.ask()
        wrapped = MultiMetricPrunerTrial(trial)
        wrapped.report({"loss": 3.0, "acc": 3.0}, step=0)
        assert wrapped.should_prune()


# ── End-to-end tests from example.py ─────────────────────────────────────────


class TestEndToEnd:
    def test_multi_metric_mode(self) -> None:
        """Mode 1: joint=True with Pareto-ranked intermediate values."""

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            mmt = MultiMetricPrunerTrial(trial)
            x = mmt.suggest_float("x", -5.0, 5.0)
            for step in range(10):
                mmt.report({"loss": (x - step * 0.1) ** 2, "acc": (x + step * 0.1) ** 2}, step)
                if mmt.should_prune():
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
            mmt = MultiMetricPrunerTrial(trial)
            x = mmt.suggest_float("x", -5.0, 5.0)
            for step in range(10):
                loss = (x - step * 0.1) ** 2
                acc = 1.0 / (1.0 + (x + step * 0.1) ** 2)
                mmt.report({"loss": loss, "acc": acc}, step)
                if mmt.should_prune():
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
            mmt = MultiMetricPrunerTrial(trial)
            x = mmt.suggest_float("x", -5.0, 5.0)
            for step in range(10):
                mmt.report({"train_loss": (x - step * 0.1) ** 2}, step)
                if mmt.should_prune(metric_name="train_loss"):
                    raise optuna.TrialPruned()
                if step % 5 == 0:
                    mmt.report({"val_loss": (x + step * 0.05) ** 2}, step)
                    if mmt.should_prune(metric_name="val_loss"):
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
            mmt = MultiMetricPrunerTrial(trial)
            x = mmt.suggest_float("x", -5.0, 5.0)
            for step in range(5):
                # Only train_loss is reported; val_loss is never reported.
                mmt.report({"train_loss": (x - step * 0.1) ** 2}, step)
                if mmt.should_prune(metric_name="train_loss"):
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


# ── _argsort_by_hv_contribution ───────────────────────────────────────────────

_HV_LOSS_VALS = [
    [[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [2.0, 2.0], [4.0, 4.0]],  # 2D
    [[0.0, 3.0, 1.0], [1.0, 1.0, 1.0], [3.0, 0.0, 2.0], [2.0, 2.0, 0.0], [4.0, 4.0, 4.0]],  # 3D
]


class TestArgsortByHvContribution:
    @pytest.mark.parametrize("loss_vals", _HV_LOSS_VALS)
    def test_is_a_permutation(self, loss_vals: list) -> None:
        lvals = np.array(loss_vals)
        ref = np.full(lvals.shape[1], 5.0)
        order = _argsort_by_hv_contribution(lvals, ref)
        assert sorted(order.tolist()) == list(range(len(lvals)))

    @pytest.mark.parametrize("loss_vals", _HV_LOSS_VALS)
    def test_prefix_matches_greedy_selection(self, loss_vals: list) -> None:
        # The first k of the order must be exactly the greedy best-k subset that the selector
        # returns. This pins the order to genuine HV-contribution order for every prefix.
        lvals = np.array(loss_vals)
        n = len(lvals)
        ref = np.full(lvals.shape[1], 5.0)
        all_indices = np.arange(n)
        order = _argsort_by_hv_contribution(lvals, ref)
        for k in range(1, n):
            greedy_best_k = set(_solve_hssp(lvals, all_indices, k, ref).tolist())
            assert set(order[:k].tolist()) == greedy_best_k

    def test_full_set_is_greedy_unlike_solve_hssp(self) -> None:
        # `_solve_hssp` short-circuits the full subset to original order; the helper must not.
        lvals = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
        ref = np.array([5.0, 5.0])
        order = _argsort_by_hv_contribution(lvals, ref)
        # [1,1] is the largest single contributor, [4,4] the smallest.
        assert order[0] == 1
        assert order[-1] == 4
        assert _solve_hssp(lvals, np.arange(5), 5, ref).tolist() == [0, 1, 2, 3, 4]  # original

    def test_duplicates_are_tied_adjacent_group(self) -> None:
        # Rows 1 and 3 are identical, so they share a contribution and must be adjacent, and the
        # unique groups must keep greedy order.
        lvals = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [1.0, 1.0], [4.0, 4.0]])
        ref = np.array([5.0, 5.0])
        order = _argsort_by_hv_contribution(lvals, ref).tolist()
        assert abs(order.index(1) - order.index(3)) == 1  # tied pair adjacent
        # Group order matches the order of the unique representatives.
        uniq = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [4.0, 4.0]])
        uniq_order = _argsort_by_hv_contribution(uniq, ref).tolist()
        assert uniq_order[0] == 1  # [1,1] group first

    @pytest.mark.parametrize(
        "loss_vals, expected",
        [
            ([[1.0, 2.0]], [0]),  # single point
            ([[2.0, 2.0], [1.0, 1.0]], [1, 0]),  # dominated point goes last
            ([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [0, 1, 2]),  # all tied
        ],
    )
    def test_edge_cases(self, loss_vals: list, expected: list) -> None:
        lvals = np.array(loss_vals)
        ref = np.array([5.0, 5.0])
        assert _argsort_by_hv_contribution(lvals, ref).tolist() == expected


# ── _tie_break ────────────────────────────────────────────────────────────────


class TestTieBreak:
    def test_current_trial_best_in_rank_gets_largest_bonus(self) -> None:
        # Regression: the current trial (last index) used to be pinned to a 0.0 bonus. When it is
        # the strongest HV contributor in its rank it must now get the most negative bonus.
        ranks = np.array([1, 1, 1])
        lvals = np.array([[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]])  # current [2,2] dominates the box
        indices, bonuses = _tie_break(lvals, ranks)
        bonus_of = dict(zip(indices.tolist(), bonuses.tolist()))
        assert bonus_of[2] == pytest.approx(-0.5)
        assert bonus_of[2] == min(bonuses)

    def test_current_trial_weakest_in_rank_gets_zero_bonus(self) -> None:
        ranks = np.array([1, 1, 1])
        lvals = np.array([[1.0, 3.0], [3.0, 1.0], [0.0, 4.0]])  # current [0,4] is the thin sliver
        indices, bonuses = _tie_break(lvals, ranks)
        bonus_of = dict(zip(indices.tolist(), bonuses.tolist()))
        assert bonus_of[2] == pytest.approx(-0.1)

    def test_bonuses_bounded(self) -> None:
        ranks = np.array([1, 1, 1])
        lvals = np.array([[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]])
        _, bonuses = _tie_break(lvals, ranks)
        assert np.all(bonuses >= -0.5) and np.all(bonuses <= 0.0)

    def test_single_in_rank_gets_zero_bonus(self) -> None:
        lvals = np.array([[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]])
        indices, bonuses = _tie_break(lvals, np.array([0, 2, 1]))
        assert bonuses.tolist() == [0.0]
        assert indices.tolist() == [2]
