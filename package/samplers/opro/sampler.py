from __future__ import annotations

from collections import deque
import json
import math
import re
from typing import Any
import warnings

from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub


_JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


class OPROSampler(optunahub.samplers.SimpleBaseSampler):
    """A sampler that uses a large language model as the optimizer, following OPRO.

    In each step, the sampler shows the language model the best trials so far, sorted from the
    worst to the best, and asks it for new parameters that are better than all of them. The
    proposals of one step are queued and used by the following trials. When the queue is empty,
    a new step is run with the latest trials.

    Args:
        model:
            The name of the language model, for example ``"gpt-4o-mini"`` for OpenAI or
            ``"qwen2.5:7b"`` for a local Ollama server.
        base_url:
            The base URL of an OpenAI-compatible API, for example
            ``"http://localhost:11434/v1"`` for Ollama. If :obj:`None`, the OpenAI API is used.
        api_key:
            The API key. If :obj:`None`, the ``OPENAI_API_KEY`` environment variable is used.
            Local servers such as Ollama accept any non-empty string.
        task_description:
            An optional description of the objective, added to the prompt.
        n_startup_trials:
            The number of trials sampled at random before the language model is used.
        n_candidates_per_step:
            The number of candidates requested from the language model in each step.
        max_trajectory_size:
            The maximum number of past trials shown in the prompt. The best trials are shown.
        temperature:
            The sampling temperature of the language model.
        search_space:
            The search space. If :obj:`None`, it is inferred from the completed trials.
        seed:
            The seed for the random sampling of the startup trials and of the fallback.
        client:
            An OpenAI-compatible client object that has ``chat.completions.create``. If given,
            ``base_url`` and ``api_key`` are ignored. This is mainly for tests.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        task_description: str | None = None,
        n_startup_trials: int = 5,
        n_candidates_per_step: int = 8,
        max_trajectory_size: int = 20,
        temperature: float = 1.0,
        search_space: dict[str, BaseDistribution] | None = None,
        seed: int | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(search_space=search_space, seed=seed)
        if n_startup_trials < 0:
            raise ValueError("n_startup_trials must be non-negative.")
        if n_candidates_per_step < 1:
            raise ValueError("n_candidates_per_step must be at least 1.")
        if max_trajectory_size < 1:
            raise ValueError("max_trajectory_size must be at least 1.")
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._task_description = task_description
        self._n_startup_trials = n_startup_trials
        self._n_candidates_per_step = n_candidates_per_step
        self._max_trajectory_size = max_trajectory_size
        self._temperature = temperature
        self._client = client
        self._queue: deque[dict[str, Any]] = deque()

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if len(study.directions) > 1:
            raise ValueError("OPROSampler supports only single-objective optimization.")
        if len(search_space) == 0:
            return {}

        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        trials = [t for t in trials if search_space.keys() <= t.params.keys()]
        if len(trials) < self._n_startup_trials or len(trials) == 0:
            # An empty dict lets every parameter fall back to random sampling.
            return {}

        while self._queue:
            params = self._queue.popleft()
            if params.keys() == search_space.keys():
                return params

        prompt = build_meta_prompt(
            trials,
            search_space,
            study.direction,
            self._max_trajectory_size,
            self._task_description,
        )
        seen = {_params_key({name: t.params[name] for name in search_space}) for t in trials}
        for _ in range(self._n_candidates_per_step):
            proposal = self._propose(prompt, search_space)
            if proposal is not None and _params_key(proposal) not in seen:
                seen.add(_params_key(proposal))
                self._queue.append(proposal)

        if not self._queue:
            warnings.warn(
                "The language model gave no valid new parameters. Falling back to random "
                "sampling for this trial."
            )
            return {}
        return self._queue.popleft()

    def _propose(
        self, prompt: str, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any] | None:
        response = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        content = response.choices[0].message.content or ""
        return parse_params(content, search_space)

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client


def build_meta_prompt(
    trials: list[FrozenTrial],
    search_space: dict[str, BaseDistribution],
    direction: StudyDirection,
    max_trajectory_size: int,
    task_description: str | None = None,
) -> str:
    """Build the prompt that shows the past trials and asks for better parameters."""
    minimize = direction == StudyDirection.MINIMIZE
    # Keep the best trials, and list them from the worst to the best, so that the best trial is
    # right before the instruction.
    best_first = sorted(trials, key=lambda t: _value(t) if minimize else -_value(t))
    shown = list(reversed(best_first[:max_trajectory_size]))

    goal = "minimize" if minimize else "maximize"
    better = "lower" if minimize else "higher"
    lines = [f"You are helping me {goal} an objective function."]
    if task_description:
        lines.append(f"Task: {task_description}")
    lines.append("")
    lines.append("The parameters are:")
    for name, dist in search_space.items():
        lines.append(f"- {name}: {_describe_distribution(dist)}")
    lines.append("")
    lines.append(
        f"Below are parameters I have tried and their objective values. They are sorted from "
        f"the worst to the best, so the best one is the last. {better.capitalize()} values are "
        f"better."
    )
    lines.append("")
    for t in shown:
        params = {name: t.params[name] for name in search_space}
        lines.append(f"parameters: {json.dumps(params)}")
        lines.append(f"value: {_value(t):.6g}")
        lines.append("")
    lines.append(
        "Propose new parameters that are different from all parameters above and that you "
        f"expect to give a {better} value than all of them. Do not write code. End your answer "
        "with the parameters as one JSON object whose keys are the parameter names."
    )
    return "\n".join(lines)


def parse_params(text: str, search_space: dict[str, BaseDistribution]) -> dict[str, Any] | None:
    """Read the last JSON object in ``text`` and fit it to the search space.

    Returns :obj:`None` if no JSON object has a valid value for every parameter.
    """
    for match in reversed(_JSON_OBJECT_PATTERN.findall(text)):
        try:
            raw = json.loads(match)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        params = {}
        for name, dist in search_space.items():
            if name not in raw:
                break
            value = _fit_to_distribution(raw[name], dist)
            if value is None:
                break
            params[name] = value
        else:
            return params
    return None


def _fit_to_distribution(value: Any, dist: BaseDistribution) -> Any | None:
    if isinstance(dist, CategoricalDistribution):
        for choice in dist.choices:
            if value == choice or str(value) == str(choice):
                return choice
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    if isinstance(dist, IntDistribution):
        clipped = min(max(round(value), dist.low), dist.high)
        steps = round((clipped - dist.low) / dist.step)
        return int(min(dist.low + steps * dist.step, dist.high))
    if isinstance(dist, FloatDistribution):
        clipped_float = min(max(float(value), dist.low), dist.high)
        if dist.step is not None:
            steps = round((clipped_float - dist.low) / dist.step)
            clipped_float = min(dist.low + steps * dist.step, dist.high)
        return clipped_float
    return None


def _describe_distribution(dist: BaseDistribution) -> str:
    if isinstance(dist, CategoricalDistribution):
        return f"one of {json.dumps(list(dist.choices))}"
    if isinstance(dist, IntDistribution):
        text = f"integer from {dist.low} to {dist.high}"
    elif isinstance(dist, FloatDistribution):
        text = f"float from {dist.low:g} to {dist.high:g}"
    else:
        raise TypeError(f"Unsupported distribution: {dist}")
    if dist.log:
        text += ", log scale"
    if isinstance(dist, IntDistribution) and dist.step != 1:
        text += f", in steps of {dist.step}"
    if isinstance(dist, FloatDistribution) and dist.step is not None:
        text += f", in steps of {dist.step:g}"
    return text


def _value(trial: FrozenTrial) -> float:
    assert trial.value is not None
    return trial.value


def _params_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, default=str)


__all__ = ["OPROSampler", "build_meta_prompt", "parse_params"]
