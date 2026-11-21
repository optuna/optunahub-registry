from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
import optunahub
import pytest


module = optunahub.load_local_module(package="samplers/opro", registry_root="package/")
OPROSampler = module.OPROSampler
parse_params = module.sampler.parse_params


class FakeClient:
    """An OpenAI-compatible client that answers with ``respond(prompt)``."""

    def __init__(self, respond: Callable[[str], str]) -> None:
        self.prompts: list[str] = []
        self._respond = respond
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        message = SimpleNamespace(content=self._respond(prompt))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    return (x - 3) ** 2 + (y + 1) ** 2


def test_startup_trials_do_not_call_the_model() -> None:
    client = FakeClient(lambda prompt: '{"x": 3.0, "y": -1}')
    sampler = OPROSampler(model="fake", client=client, n_startup_trials=5, seed=0)
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, n_trials=5)

    assert client.prompts == []


def test_proposals_of_the_model_are_used() -> None:
    client = FakeClient(lambda prompt: 'I suggest {"x": 3.0, "y": -1}')
    sampler = OPROSampler(model="fake", client=client, n_startup_trials=3, seed=0)
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, n_trials=4)

    assert study.trials[3].params == {"x": 3.0, "y": -1}
    assert study.best_value == 0.0


def test_one_step_fills_the_queue_for_the_following_trials() -> None:
    proposals = iter(f'{{"x": {i}.5, "y": 0}}' for i in range(100))
    client = FakeClient(lambda prompt: next(proposals))
    sampler = OPROSampler(
        model="fake", client=client, n_startup_trials=2, n_candidates_per_step=4, seed=0
    )
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, n_trials=6)

    # One step after the startup trials gives the candidates for trials 2 to 5.
    assert len(client.prompts) == 4
    assert [t.params["x"] for t in study.trials[2:]] == [0.5, 1.5, 2.5, 3.5]


def test_invalid_answers_fall_back_to_random_sampling() -> None:
    client = FakeClient(lambda prompt: "I cannot answer that.")
    sampler = OPROSampler(model="fake", client=client, n_startup_trials=1, seed=0)
    study = optuna.create_study(sampler=sampler)

    with pytest.warns(UserWarning, match="no valid new parameters"):
        study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_prompt_lists_the_best_trials_from_worst_to_best() -> None:
    client = FakeClient(lambda prompt: '{"x": 3.0, "y": -1}')
    sampler = OPROSampler(
        model="fake", client=client, n_startup_trials=4, max_trajectory_size=2, seed=0
    )
    study = optuna.create_study(sampler=sampler)
    for x in [0.0, 1.0, 2.0, 9.0]:
        study.enqueue_trial({"x": x, "y": -1})
    study.optimize(objective, n_trials=4)

    study.optimize(objective, n_trials=1)

    prompt = client.prompts[0]
    # Only the two best trials are shown, the best (x = 2) last.
    assert '"x": 9.0' not in prompt
    assert '"x": 0.0' not in prompt
    assert prompt.index('"x": 1.0') < prompt.index('"x": 2.0')
    assert "Lower values are better." in prompt


def test_prompt_for_maximization_shows_the_highest_value_last() -> None:
    client = FakeClient(lambda prompt: '{"x": 5.0}')
    sampler = OPROSampler(model="fake", client=client, n_startup_trials=3, seed=0)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    for x in [3.0, 1.0, 2.0]:
        study.enqueue_trial({"x": x})
    study.optimize(lambda t: t.suggest_float("x", 0, 10), n_trials=3)

    study.optimize(lambda t: t.suggest_float("x", 0, 10), n_trials=1)

    prompt = client.prompts[0]
    assert prompt.index('"x": 1.0') < prompt.index('"x": 2.0') < prompt.index('"x": 3.0')
    assert "Higher values are better." in prompt


def test_multi_objective_is_rejected() -> None:
    client = FakeClient(lambda prompt: '{"x": 1.0}')
    sampler = OPROSampler(model="fake", client=client, n_startup_trials=0)
    study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"])

    with pytest.raises(ValueError, match="single-objective"):
        study.optimize(lambda t: (t.suggest_float("x", 0, 1), 0.0), n_trials=1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"x": 0.5, "n": 3, "c": "b"}', {"x": 0.5, "n": 3, "c": "b"}),
        # The last JSON object is used.
        (
            '{"x": 0.1, "n": 1, "c": "a"} then {"x": 0.2, "n": 3, "c": "b"}',
            {"x": 0.2, "n": 3, "c": "b"},
        ),
        # Values are clipped to the range and integers are rounded to the step.
        ('{"x": 7.0, "n": 4.6, "c": "a"}', {"x": 1.0, "n": 5, "c": "a"}),
        ('{"x": -1, "n": 100, "c": "c"}', {"x": 0.0, "n": 9, "c": "c"}),
    ],
)
def test_parse_params_fits_values_to_the_search_space(text: str, expected: dict[str, Any]) -> None:
    search_space = {
        "x": FloatDistribution(0.0, 1.0),
        "n": IntDistribution(1, 9, step=2),
        "c": CategoricalDistribution(["a", "b", "c"]),
    }

    assert parse_params(text, search_space) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"x": 0.5}',  # a parameter is missing
        '{"x": "high", "n": 3, "c": "a"}',  # not a number
        '{"x": 0.5, "n": 3, "c": "d"}',  # not one of the choices
        '{"x": NaN, "n": 3, "c": "a"}',
    ],
)
def test_parse_params_rejects_invalid_answers(text: str) -> None:
    search_space = {
        "x": FloatDistribution(0.0, 1.0),
        "n": IntDistribution(1, 9, step=2),
        "c": CategoricalDistribution(["a", "b", "c"]),
    }

    assert parse_params(text, search_space) is None
