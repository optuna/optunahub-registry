from __future__ import annotations

from dataclasses import dataclass
import decimal
from functools import lru_cache
from typing import cast
from typing import TYPE_CHECKING

from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

    ChoicesArgsType = tuple[int | float, int | float, int | float | None]  # low, high, step


@lru_cache
def _enumerate_candidates(
    low: int | float, high: int | float, step: int | float | None
) -> tuple[float, ...]:
    if step is None:
        raise ValueError(
            "FloatDistribution.step must be given for BruteForceSampler"
            " (otherwise, the search space will be infinite)."
        )
    if isinstance(low, int) and isinstance(high, int) and isinstance(step, int):
        return tuple(range(low, high + 1, step))
    else:
        low_ = decimal.Decimal(str(low))
        high_ = decimal.Decimal(str(high))
        step_ = decimal.Decimal(str(step))
        ret = []
        while low_ <= high_:
            ret.append(float(low_))
            low_ += step_
        return tuple(ret)


class _UnexpandedTreeNode:
    def count_tree_size(self) -> int:
        return 1

    def count_completed(self) -> int:
        return 0


_UNEXPANDED_NODE = _UnexpandedTreeNode()


@dataclass
class _TreeNode:
    param_name: str | None = None
    children: dict[float, _TreeNode | _UnexpandedTreeNode] | None = None
    choices_args: ChoicesArgsType | None = None
    n_completed: int = 0
    trial_number: int = -1

    def _validate_search_space_consistency(
        self, param_name: str | None, choices_args: ChoicesArgsType | None
    ) -> None:
        if self.param_name != param_name:
            raise ValueError(f"param_name mismatch: {self.param_name} != {param_name}")
        if choices_args != self.choices_args:
            assert self.children is not None and choices_args is not None
            choices_old = list(self.children)
            choices_new = _enumerate_candidates(*choices_args)
            raise ValueError(
                f"search_space mismatch in {param_name}: {choices_old} != {choices_new}"
            )

    def expand(self, param_name: str | None, choices_args: ChoicesArgsType) -> None:
        if self.children is None:
            self.param_name = param_name
            choices = _enumerate_candidates(*choices_args)
            self.children = {value: _UNEXPANDED_NODE for value in choices}
            self.choices_args = choices_args
        else:
            self._validate_search_space_consistency(param_name, choices_args)

    def add_path(self, trial_path: list[tuple[str, ChoicesArgsType, float]]) -> _TreeNode | None:
        current_node = self
        for param_name, choices_args, value in trial_path:
            current_node.expand(param_name, choices_args)
            if not (children := current_node.children):
                return None
            elif (next_node := children.get(value)) is None:
                return None
            elif next_node is _UNEXPANDED_NODE:
                next_node = _TreeNode()
                children[value] = next_node
            current_node = cast(_TreeNode, next_node)
        return current_node

    def count_tree_size(self) -> int:
        if not (children := self.children):
            return 1
        return sum(child.count_tree_size() for child in children.values())

    def count_completed(self) -> int:
        if not (children := self.children):
            return self.n_completed
        return self.n_completed + sum(child.count_completed() for child in children.values())


def build_full_tree(trials: list[FrozenTrial]) -> _TreeNode:
    tree = _TreeNode()
    cat_internal_repr_cache: dict[str, dict[CategoricalChoiceType, float]] = {}

    def _get_trial_path(trial: FrozenTrial) -> list[tuple[str, ChoicesArgsType, float]]:
        trial_path: list[tuple[str, ChoicesArgsType, float]] = []
        trial_params = trial.params
        for name, dist in trial.distributions.items():
            if name not in cat_internal_repr_cache:
                cat_internal_repr_cache[name] = {}
                if isinstance(dist, CategoricalDistribution):
                    cat_internal_repr_cache[name] = {c: i for i, c in enumerate(dist.choices)}
            if cat_repr := cat_internal_repr_cache[name]:
                if (value := cat_repr.get(param_val := trial_params[name])) is None:
                    value = dist.to_internal_repr(param_val)
                dist = cast(CategoricalDistribution, dist)
                trial_path.append((name, (0, len(dist.choices) - 1, 1), value))
            else:
                dist = cast("IntDistribution | FloatDistribution", dist)
                trial_path.append((name, (dist.low, dist.high, dist.step), trial_params[name]))
        return trial_path

    for trial in trials:
        if (leaf := tree.add_path(_get_trial_path(trial))) is not None:
            leaf.trial_number = trial.number
            if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]:
                leaf.n_completed += 1
    return tree
