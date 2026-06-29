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
    def __init__(self, *, result: bool = False) -> None:
        self.call_count = 0
        self._result = result

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        self.call_count += 1
        return self._result


class TrackingPruner(BasePruner):
    def __init__(self) -> None:
        self.called = False

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        self.called = True
        return False


def _make_pruner(
    base: BasePruner | dict[str, BasePruner] | None = None,
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
    trial = study.ask()
    wrapped = MultiMetricPrunerTrial(trial)
    for step, vals in values_per_step.items():
        wrapped.report(vals, step)
    study.tell(trial, final_values)


def _make_pareto_study(
    *,
    metric_directions: dict[str, str],
    study_directions: list[str] | None = None,
) -> optuna.Study:
    if study_directions is None:
        study_directions = list(metric_directions.values())
    pruner = MultiMetricPruner(
        optuna.pruners.MedianPruner(n_startup_trials=0),
        metric_directions=metric_directions,
        joint=True,
    )
    return optuna.create_study(directions=study_directions, pruner=pruner)


# ── MultiMetricPruner.__init__ ────────────────────────────────────────────────


class TestMultiMetricPrunerInit:
    @pytest.mark.parametrize("joint", [True, False])
    def test_valid_construction(self, joint: bool) -> None:
        pruner = _make_pruner(joint=joint)
        assert pruner._joint is joint
        assert set(pruner._metric_directions) == {"loss", "acc"}

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
        wrapped = _ask_wrapped(_make_study())
        with pytest.raises(TypeError, match="must be a dict of float"):
            wrapped.report({"loss": "not_a_number"}, step=0)  # type: ignore[arg-type,dict-item]

    def test_non_dict_raises(self) -> None:
        wrapped = _ask_wrapped(_make_study())
        with pytest.raises(TypeError, match="must be a dict"):
            wrapped.report([0.5, 0.8], step=0)  # type: ignore[arg-type]

    def test_empty_dict_raises(self) -> None:
        wrapped = _ask_wrapped(_make_study())
        with pytest.raises(ValueError, match="must have at least one entry"):
            wrapped.report({}, step=0)

    def test_negative_step_raises(self) -> None:
        wrapped = _ask_wrapped(_make_study())
        with pytest.raises(ValueError, match="cannot be negative"):
            wrapped.report({"loss": 0.5}, step=-1)

    def test_unknown_metric_raises(self) -> None:
        wrapped = _ask_wrapped(_make_study())
        with pytest.raises(ValueError, match="unknown metric"):
            wrapped.report({"unknown": 0.5}, step=0)

    def test_duplicate_warns_and_does_not_overwrite(self) -> None:
        wrapped = _ask_wrapped(_make_study())
        wrapped.report({"loss": 0.5}, step=0)
        with pytest.warns(Warning, match="already reported"):
            wrapped.report({"loss": 0.9}, step=0)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert attrs["0"]["loss"] == pytest.approx(0.5)

    def test_per_metric_splits_multi_dict(self) -> None:
        wrapped = _ask_wrapped(_make_study(joint=False))
        wrapped.report({"loss": 0.5, "acc": 0.8}, step=0)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert attrs["0"]["loss"] == pytest.approx(0.5)
        assert attrs["0"]["acc"] == pytest.approx(0.8)

    def test_partial_report_allowed(self) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=MultiMetricPruner(
                optuna.pruners.NopPruner(),
                metric_directions={"train_loss": "minimize", "val_loss": "minimize"},
                joint=False,
            ),
        )
        wrapped = _ask_wrapped(study)
        wrapped.report({"train_loss": 0.5}, step=0)
        attrs = wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})
        assert "train_loss" in attrs["0"]
        assert "val_loss" not in attrs["0"]

    def test_step_stored_as_string_key(self) -> None:
        wrapped = _ask_wrapped(_make_study())
        wrapped.report({"loss": 0.3, "acc": 0.7}, step=5)
        assert "5" in wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})

    def test_accumulates_values_across_steps(self) -> None:
        wrapped = _ask_wrapped(_make_study())
        for step in range(5):
            wrapped.report({"loss": float(step), "acc": float(step) * 2}, step=step)
        assert len(wrapped._trial.user_attrs.get(_USER_ATTR_KEY, {})) == 5

    def test_getattr_delegates_to_inner_trial(self) -> None:
        wrapped = _ask_wrapped(_make_study())
        x = wrapped.suggest_float("x", 0.0, 1.0)
        assert 0.0 <= x <= 1.0
        assert "x" in wrapped.params

    def test_is_instance_of_optuna_trial(self) -> None:
        assert isinstance(_ask_wrapped(_make_study()), optuna.Trial)


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
        assert _ask_wrapped(study).should_prune() is False

    @pytest.mark.parametrize("joint", [True, False])
    def test_prunes_when_base_prunes(self, joint: bool) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=_make_pruner(AlwaysPrunePruner(), joint=joint),
        )
        _add_complete_trial(study, {0: {"loss": 1.0, "acc": 1.0}}, [1.0, 1.0])
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 2.0, "acc": 2.0}, step=0)
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
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=_make_pruner(AlwaysPrunePruner(), joint=False),
        )
        _add_complete_trial(study, {0: {"loss": 1.0}}, [1.0, 1.0])
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5}, step=0)
        assert wrapped.should_prune(metric_name="loss") is True

    def test_joint_true_ignores_metric_name(self) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            pruner=_make_pruner(NeverPrunePruner(), joint=True),
        )
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        assert wrapped.should_prune(metric_name="loss") is False

    def test_joint_true_metric_name_ignored_with_real_pruner(self) -> None:
        study = _make_pareto_study(metric_directions={"loss": "minimize", "acc": "minimize"})
        for val in [0.1, 0.2, 0.3, 0.4]:
            _add_complete_trial(study, {0: {"loss": val, "acc": val}}, [val, val])
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 10.0, "acc": 10.0}, step=0)
        assert wrapped.should_prune()
        assert wrapped.should_prune(metric_name="loss")
        assert wrapped.should_prune(metric_name="acc")


# ── MultiMetricPruner.prune routing ──────────────────────────────────────────


class TestMultiMetricPrunerPrune:
    def test_no_reports_returns_false(self) -> None:
        pruner = _make_pruner(AlwaysPrunePruner())
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        frozen = study.ask()._get_latest_trial()
        assert pruner.prune(study, frozen) is False

    def test_joint_false_iterates_all_metrics_when_no_name(self) -> None:
        base = CountingPruner()
        pruner = _make_pruner(base, joint=False)
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        MultiMetricPrunerTrial(trial).report({"loss": 0.5, "acc": 0.5}, step=0)
        pruner.prune(study, trial._get_latest_trial())
        assert base.call_count == 2

    def test_joint_false_short_circuits_on_first_prune(self) -> None:
        base = CountingPruner(result=True)
        pruner = _make_pruner(base, joint=False)
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        MultiMetricPrunerTrial(trial).report({"loss": 0.5, "acc": 0.5}, step=0)
        assert pruner.prune(study, trial._get_latest_trial()) is True
        assert base.call_count == 1

    def test_joint_false_with_metric_name_calls_base_once(self) -> None:
        base = CountingPruner()
        pruner = _make_pruner(base, joint=False)
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        MultiMetricPrunerTrial(trial).report({"loss": 0.5, "acc": 0.5}, step=0)
        pruner.prune(study, trial._get_latest_trial(), metric_name="loss")
        assert base.call_count == 1


# ── Pareto pruning (joint=True) ─────────────────────────────────────────────


class TestParetoPruning:
    @pytest.mark.parametrize(
        "metric_dirs, completed_vals, completed_acc, current_vals, should_prune",
        [
            # Pareto-dominant trial (all minimize) → not pruned
            (
                {"loss": "minimize", "acc": "minimize"},
                [5.0, 6.0, 7.0, 8.0],
                None,
                {"loss": 0.1, "acc": 0.1},
                False,
            ),
            # Dominated trial (all minimize) → pruned
            (
                {"loss": "minimize", "acc": "minimize"},
                [0.1, 0.2, 0.3, 0.4],
                None,
                {"loss": 10.0, "acc": 10.0},
                True,
            ),
            # Mixed directions: bad loss, good acc → Pareto front → not pruned
            (
                {"loss": "minimize", "acc": "maximize"},
                [2.0, 3.0, 4.0, 5.0],
                0.1,
                {"loss": 10.0, "acc": 0.9},
                False,
            ),
            # Mixed directions: bad on both → dominated → pruned
            (
                {"loss": "minimize", "acc": "maximize"},
                [0.1, 0.2, 0.3, 0.4],
                0.9,
                {"loss": 5.0, "acc": 0.1},
                True,
            ),
        ],
    )
    def test_pareto_pruning_decision(
        self,
        metric_dirs: dict,
        completed_vals: list,
        completed_acc: float | None,
        current_vals: dict,
        should_prune: bool,
    ) -> None:
        study = _make_pareto_study(
            metric_directions=metric_dirs,
            study_directions=list(metric_dirs.values()),
        )
        for val in completed_vals:
            acc = completed_acc if completed_acc is not None else val
            step_vals = {k: val if k == "loss" else acc for k in metric_dirs}
            _add_complete_trial(study, {0: step_vals}, [val, acc])
        wrapped = _ask_wrapped(study)
        wrapped.report(current_vals, step=0)
        assert wrapped.should_prune() == should_prune


# ── Pareto rank changes across steps ─────────────────────────────────────────


class TestParetoRankAcrossSteps:
    @pytest.mark.parametrize(
        "completed_steps, current_steps, expected_not_pruned",
        [
            # Dominated at step 0, dominant at step 1 → not pruned
            (
                {0: {"loss": 0.1, "acc": 0.1}, 1: {"loss": 5.0, "acc": 5.0}},
                [(0, 5.0, 5.0), (1, 0.01, 0.01)],
                True,
            ),
            # Dominant at step 0, dominated at step 1 → still protected
            (
                {0: {"loss": 5.0, "acc": 5.0}, 1: {"loss": 0.1, "acc": 0.1}},
                [(0, 0.01, 0.01), (1, 10.0, 10.0)],
                True,
            ),
        ],
    )
    def test_rank_across_steps(
        self,
        completed_steps: dict,
        current_steps: list,
        expected_not_pruned: bool,
    ) -> None:
        study = _make_pareto_study(metric_directions={"loss": "minimize", "acc": "minimize"})
        for _ in range(4):
            _add_complete_trial(study, completed_steps, [0.1, 0.1])
        wrapped = _ask_wrapped(study)
        for step, loss, acc in current_steps:
            wrapped.report({"loss": loss, "acc": acc}, step=step)
        assert (not wrapped.should_prune()) is expected_not_pruned


# ── Tie-break bonus integration ───────────────────────────────────────────────


class TestTieBreakBonusIntegration:
    @pytest.mark.parametrize(
        "rank1_vals, current_vals, expected_not_pruned",
        [
            # High HV contribution → not pruned
            ({"loss": 3.0, "acc": 3.0}, {"loss": 2.0, "acc": 4.0}, True),
            # Low HV contribution → pruned
            ({"loss": 2.0, "acc": 4.0}, {"loss": 3.0, "acc": 3.0}, False),
        ],
    )
    def test_hv_contribution_affects_pruning(
        self, rank1_vals: dict, current_vals: dict, expected_not_pruned: bool
    ) -> None:
        study = _make_pareto_study(metric_directions={"loss": "minimize", "acc": "minimize"})
        _add_complete_trial(study, {0: {"loss": 1.0, "acc": 1.0}}, [1.0, 1.0])
        for _ in range(5):
            _add_complete_trial(study, {0: rank1_vals}, [rank1_vals["loss"], rank1_vals["acc"]])
        wrapped = _ask_wrapped(study)
        wrapped.report(current_vals, step=0)
        assert (not wrapped.should_prune()) is expected_not_pruned


# ── End-to-end tests ─────────────────────────────────────────────────────────


_E2E_CONFIGS = [
    pytest.param(
        True,
        {"loss": "minimize", "acc": "minimize"},
        ["minimize", "minimize"],
        optuna.pruners.MedianPruner(n_startup_trials=3),
        id="joint",
    ),
    pytest.param(
        False,
        {"loss": "minimize", "acc": "maximize"},
        ["minimize", "maximize"],
        optuna.pruners.MedianPruner(n_startup_trials=3),
        id="per-metric",
    ),
    pytest.param(
        False,
        {"loss": "minimize", "acc": "maximize"},
        ["minimize", "maximize"],
        {
            "loss": optuna.pruners.MedianPruner(n_startup_trials=3),
            "acc": optuna.pruners.MedianPruner(n_startup_trials=5),
        },
        id="per-metric-dict",
    ),
]


class TestEndToEnd:
    @pytest.mark.parametrize("joint, metric_dirs, study_dirs, base_pruner", _E2E_CONFIGS)
    def test_optimize_with_pruning(
        self,
        joint: bool,
        metric_dirs: dict,
        study_dirs: list,
        base_pruner: BasePruner | dict[str, BasePruner],
    ) -> None:
        def objective(trial: optuna.Trial) -> tuple[float, float]:
            mmt = MultiMetricPrunerTrial(trial)
            x = mmt.suggest_float("x", -5.0, 5.0)
            for step in range(10):
                mmt.report({"loss": (x - step * 0.1) ** 2, "acc": (x + step * 0.1) ** 2}, step)
                if mmt.should_prune():
                    raise optuna.TrialPruned()
            return x**2, (x - 2.0) ** 2

        study = optuna.create_study(
            directions=study_dirs,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=MultiMetricPruner(base_pruner, metric_directions=metric_dirs, joint=joint),
        )
        study.optimize(objective, n_trials=15)
        assert len(study.trials) == 15
        assert any(t.state == TrialState.PRUNED for t in study.trials)

    def test_mixed_frequency_mode(self) -> None:
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
        study.optimize(objective, n_trials=15)
        assert len(study.trials) == 15
        assert any(t.state == TrialState.PRUNED for t in study.trials)

    def test_per_metric_coexists_with_partial_reports(self) -> None:
        def objective(trial: optuna.Trial) -> tuple[float, float]:
            mmt = MultiMetricPrunerTrial(trial)
            x = mmt.suggest_float("x", -5.0, 5.0)
            for step in range(5):
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
        assert len(_fast_non_domination_rank(np.empty((0, 2), dtype=float))) == 0

    def test_single_point_rank_zero(self) -> None:
        assert _fast_non_domination_rank(np.array([[1.0, 2.0]]))[0] == 0

    def test_dominated_point_has_higher_rank(self) -> None:
        ranks = _fast_non_domination_rank(np.array([[1.0, 1.0], [2.0, 2.0]]))
        assert ranks[0] < ranks[1]

    def test_pareto_points_same_rank(self) -> None:
        ranks = _fast_non_domination_rank(np.array([[1.0, 2.0], [2.0, 1.0]]))
        assert ranks[0] == ranks[1] == 0

    def test_three_points_two_fronts(self) -> None:
        ranks = _fast_non_domination_rank(np.array([[1.0, 3.0], [3.0, 1.0], [4.0, 4.0]]))
        assert ranks[0] == 0 and ranks[1] == 0 and ranks[2] == 1

    def test_single_objective_ranks_by_value(self) -> None:
        ranks = _fast_non_domination_rank(np.array([[3.0], [1.0], [2.0]]))
        assert ranks[1] == 0 and ranks[2] == 1 and ranks[0] == 2

    def test_three_objectives(self) -> None:
        values = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 2.0, 2.0], [2.0, 1.0, 2.0]])
        ranks = _fast_non_domination_rank(values)
        assert ranks[0] == 0 and ranks[2] == 1 and ranks[3] == 1 and ranks[1] == 2


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
        assert sorted(_argsort_by_hv_contribution(lvals, ref).tolist()) == list(range(len(lvals)))

    @pytest.mark.parametrize("loss_vals", _HV_LOSS_VALS)
    def test_prefix_matches_greedy_selection(self, loss_vals: list) -> None:
        lvals = np.array(loss_vals)
        ref = np.full(lvals.shape[1], 5.0)
        order = _argsort_by_hv_contribution(lvals, ref)
        for k in range(1, len(lvals)):
            greedy_best_k = set(_solve_hssp(lvals, np.arange(len(lvals)), k, ref).tolist())
            assert set(order[:k].tolist()) == greedy_best_k

    def test_full_set_is_greedy_unlike_solve_hssp(self) -> None:
        lvals = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
        ref = np.array([5.0, 5.0])
        order = _argsort_by_hv_contribution(lvals, ref)
        assert order[0] == 1 and order[-1] == 4
        assert _solve_hssp(lvals, np.arange(5), 5, ref).tolist() == [0, 1, 2, 3, 4]

    def test_duplicates_are_tied_adjacent_group(self) -> None:
        lvals = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [1.0, 1.0], [4.0, 4.0]])
        ref = np.array([5.0, 5.0])
        order = _argsort_by_hv_contribution(lvals, ref).tolist()
        assert abs(order.index(1) - order.index(3)) == 1
        uniq_order = _argsort_by_hv_contribution(
            np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [4.0, 4.0]]), ref
        ).tolist()
        assert uniq_order[0] == 1

    @pytest.mark.parametrize(
        "loss_vals, expected",
        [
            ([[1.0, 2.0]], [0]),
            ([[2.0, 2.0], [1.0, 1.0]], [1, 0]),
            ([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [0, 1, 2]),
        ],
    )
    def test_edge_cases(self, loss_vals: list, expected: list) -> None:
        assert (
            _argsort_by_hv_contribution(np.array(loss_vals), np.array([5.0, 5.0])).tolist()
            == expected
        )


# ── _tie_break ────────────────────────────────────────────────────────────────


class TestTieBreak:
    def test_current_trial_best_in_rank_gets_largest_bonus(self) -> None:
        ranks = np.array([1, 1, 1])
        lvals = np.array([[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]])
        indices, bonuses = _tie_break(lvals, ranks)
        bonus_of = dict(zip(indices.tolist(), bonuses.tolist()))
        assert bonus_of[2] == pytest.approx(-0.5)
        assert bonus_of[2] == min(bonuses)

    def test_current_trial_weakest_in_rank_gets_smallest_bonus(self) -> None:
        ranks = np.array([1, 1, 1])
        lvals = np.array([[1.0, 3.0], [3.0, 1.0], [0.0, 4.0]])
        indices, bonuses = _tie_break(lvals, ranks)
        assert dict(zip(indices.tolist(), bonuses.tolist()))[2] == pytest.approx(-0.1)

    def test_bonuses_bounded(self) -> None:
        _, bonuses = _tie_break(
            np.array([[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]]), np.array([1, 1, 1])
        )
        assert np.all(bonuses >= -0.5) and np.all(bonuses <= 0.0)

    def test_single_in_rank_gets_zero_bonus(self) -> None:
        indices, bonuses = _tie_break(
            np.array([[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]]), np.array([0, 2, 1])
        )
        assert bonuses.tolist() == [0.0] and indices.tolist() == [2]


# ── Per-metric dict base_pruner ──────────────────────────────────────────────


class TestDictBasePruner:
    def test_dict_base_pruner_joint_true_raises(self) -> None:
        with pytest.raises(ValueError, match="single BasePruner when `joint=True`"):
            MultiMetricPruner(
                {"loss": optuna.pruners.NopPruner(), "acc": optuna.pruners.NopPruner()},
                metric_directions={"loss": "minimize", "acc": "minimize"},
                joint=True,
            )

    @pytest.mark.parametrize(
        "pruner_dict",
        [
            {"loss": optuna.pruners.NopPruner()},
            {
                "loss": optuna.pruners.NopPruner(),
                "acc": optuna.pruners.NopPruner(),
                "extra": optuna.pruners.NopPruner(),
            },
        ],
        ids=["missing-key", "extra-key"],
    )
    def test_dict_base_pruner_keys_mismatch_raises(self, pruner_dict: dict) -> None:
        with pytest.raises(ValueError, match="keys must match"):
            MultiMetricPruner(
                pruner_dict,
                metric_directions={"loss": "minimize", "acc": "minimize"},
                joint=False,
            )

    def test_dict_base_pruner_valid(self) -> None:
        pruner = MultiMetricPruner(
            {"loss": optuna.pruners.NopPruner(), "acc": optuna.pruners.NopPruner()},
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=False,
        )
        assert isinstance(pruner._base_pruner, dict)

    def test_dict_base_pruner_routes_per_metric(self) -> None:
        loss_pruner, acc_pruner = TrackingPruner(), TrackingPruner()
        pruner = MultiMetricPruner(
            {"loss": loss_pruner, "acc": acc_pruner},
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=False,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        MultiMetricPrunerTrial(trial).report({"loss": 0.5, "acc": 0.8}, step=0)
        pruner.prune(study, trial._get_latest_trial())
        assert loss_pruner.called and acc_pruner.called

    def test_dict_base_pruner_with_metric_name(self) -> None:
        loss_pruner, acc_pruner = TrackingPruner(), TrackingPruner()
        pruner = MultiMetricPruner(
            {"loss": loss_pruner, "acc": acc_pruner},
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=False,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        trial = study.ask()
        MultiMetricPrunerTrial(trial).report({"loss": 0.5}, step=0)
        pruner.prune(study, trial._get_latest_trial(), metric_name="loss")
        assert loss_pruner.called and not acc_pruner.called

    def test_dict_base_pruner_prunes_when_one_metric_triggers(self) -> None:
        pruner = MultiMetricPruner(
            {"loss": AlwaysPrunePruner(), "acc": NeverPrunePruner()},
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=False,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        _add_complete_trial(study, {0: {"loss": 1.0, "acc": 1.0}}, [1.0, 1.0])
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        assert wrapped.should_prune() is True

    def test_dict_base_pruner_does_not_prune_when_none_triggers(self) -> None:
        pruner = MultiMetricPruner(
            {"loss": NeverPrunePruner(), "acc": NeverPrunePruner()},
            metric_directions={"loss": "minimize", "acc": "minimize"},
            joint=False,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], pruner=pruner)
        wrapped = _ask_wrapped(study)
        wrapped.report({"loss": 0.5, "acc": 0.5}, step=0)
        assert wrapped.should_prune() is False
