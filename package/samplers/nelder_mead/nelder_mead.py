from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import optuna.study.study
import optunahub

from .generate_initial_simplex import generate_initial_simplex


class NelderMeadSampler(optunahub.samplers.SimpleBaseSampler):
    """A sampler based on the Nelder-Mead simplex algorithm.
       This algorithm does not support conditional parameters that make a tree-structured search space.
       Sampling for such parameters is delegated to independent_sampler (default: RandomSampler).

       Several important matters:

       1. NelderMeadSampler does not support parallel execution, i.e., `n_jobs > 1`.

       2. If it is run "define-by-run" (no search space defined), the first trial is ignored because this trial is not involved in the Nelder-Mead algorithm.

       3. When a sampled solution is out of search space, our implementation employs the extreme barrier method, which is the box constraint handling method.
       This method provides a large constraint value (= "inf", representing infeasible) as the objective function value.

    Args:
        search_space:
            A dictionary of :class:`~optuna.distributions.BaseDistribution` that defines the search space.
            If this argument is :obj:`None`, the search space is estimated during the first trial.
            In this case, ``independent_sampler`` is used instead of the Nelder-Mead algorithm during the first trial.

        centroid:
            A centroid of the initial simplex.

        edge:
            A distance from the centroid of the initial simplex to each vertex.

        seed:
            A seed number.

    """

    def __init__(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution] | None = None,
        centroid: float = 0.5,
        edge: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(search_space)

        assert 0.0 <= centroid <= 1.0, "The centroid must be exists in the unit hypercube. "

        assert (
            0.0 < edge <= max(centroid, 1 - centroid)
        ), f"Maximum edge length is {max(centroid, 1 - centroid)}"

        if search_space is None:
            self._NM_state = "estimate_search_space"
            self._ignore_trial_number = 1
        else:
            self._NM_state = "generate_initial_simplex"
            self._ignore_trial_number = 0

        self._edge = edge
        self._centroid = centroid
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)

        self._coef = {"r": 1.0, "ic": -0.5, "oc": 0.5, "e": 2.0, "s": 0.5}
        self._f: list[float] = []
        self._shrink_num = 0

        self._independent_sampler = optuna.samplers.RandomSampler(seed=self._seed)

    def order_by(self) -> None:
        """
        Sorting the vertices of the simplex based on their objective function values.
        Computing the centroid excluding the best vertex.
        """
        order = np.argsort(self._f)
        self._y: np.ndarray = self._y[order]
        self._f = np.asarray(self._f)[order].tolist()
        self._yc = self._y[:-1].mean(axis=0)

    def out_of_boundary(self, y: np.ndarray) -> bool:
        """
        Check whether the input vertex is in the search space.
        """
        for yi, b in zip(y, self._bdrys):
            if float(b[0]) <= float(yi) <= float(b[1]):
                pass
            else:
                return True
        return False

    def suggest_eval_param(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution],
        eval_solution: np.ndarray,
    ) -> tuple[dict, bool, np.ndarray]:
        params = {}
        for i, (name, distribution) in enumerate(search_space.items()):
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                params[name] = float(
                    distribution.low + eval_solution[i] * (distribution.high - distribution.low)
                )
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                params[name] = int(
                    distribution.low + eval_solution[i] * (distribution.high - distribution.low)
                )
            else:
                raise NotImplementedError

        if self.out_of_boundary(eval_solution):
            out_of_boundary = True
        else:
            out_of_boundary = False

        return params, out_of_boundary, eval_solution

    def search(
        self,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
        study: optuna.study.Study,
        f_val: float | None = None,
    ) -> tuple[dict, bool]:
        # Initialization
        if self._NM_state == "Initialization":
            if trial.number > self._ignore_trial_number:
                self._f.append(study.trials_dataframe()["value"].values[-2])

            if f_val == float("inf"):
                self._f.append(float("inf"))

            if len(self._y) != len(self._f):
                params, out_of_boundary, normalized_param = self.suggest_eval_param(
                    search_space, self._y[len(self._f)]
                )
            else:
                self._NM_state = "Reflection_org"
        # Reflection, Expansion, Outside contraction, Inside Contraction, Shrinkage.
        else:
            if f_val is None:
                objective_value = study.trials_dataframe()["value"].values[-2]
            else:
                objective_value = f_val

            if self._NM_state == "Reflection":
                self._fr = objective_value
                if self._f[0] <= self._fr < self._f[-2]:
                    self._y[-1] = self._yr
                    self._f[-1] = self._fr
                    self._NM_state = "Reflection_org"
                elif self._fr < self._f[0]:
                    self._NM_state = "Expansion"
                elif self._f[-2] <= self._fr < self._f[-1]:
                    self._NM_state = "Outside_constraction"
                elif self._f[-1] <= self._fr:
                    self._NM_state = "Inside_constraction"

            elif self._NM_state == "Expansion":
                self._fe = objective_value
                if self._fe < self._fr:
                    self._y[-1] = self._ye
                    self._f[-1] = self._fe
                else:
                    self._y[-1] = self._yr
                    self._f[-1] = self._fr
                self._NM_state = "Reflection_org"

            elif self._NM_state == "Outside_constraction":
                self._foc = objective_value
                if self._foc <= self._fr:
                    self._y[-1] = self._yoc
                    self._f[-1] = self._foc
                    self._NM_state = "Reflection_org"
                else:
                    self._NM_state = "Shrinkage"

            elif self._NM_state == "Inside_constraction":
                self._fic = objective_value
                if self._fic < self._f[-1]:
                    self._y[-1] = self._yic
                    self._f[-1] = self._fic
                    self._NM_state = "Reflection_org"
                else:
                    self._NM_state = "Shrinkage"

            elif self._NM_state == "Shrinkage":
                self._fs = objective_value
                self._y[self._shrink_num + 1] = self._ys
                self._f[self._shrink_num + 1] = self._fs
                self._shrink_num += 1
                if self._shrink_num == self._dim:
                    self._NM_state = "Reflection_org"
                    self._shrink_num = 0

        if self._NM_state == "Reflection_org":
            self.order_by()
            self._yr: np.ndarray = self._yc + self._coef["r"] * (self._yc - self._y[-1])
            params, out_of_boundary, normalized_param = self.suggest_eval_param(
                search_space, self._yr
            )
            self._NM_state = "Reflection"

        elif self._NM_state == "Expansion":
            self._ye: np.ndarray = self._yc + self._coef["e"] * (self._yc - self._y[-1])
            params, out_of_boundary, normalized_param = self.suggest_eval_param(
                search_space, self._ye
            )

        elif self._NM_state == "Outside_constraction":
            self._yoc: np.ndarray = self._yc + self._coef["oc"] * (self._yc - self._y[-1])
            params, out_of_boundary, normalized_param = self.suggest_eval_param(
                search_space, self._yoc
            )

        elif self._NM_state == "Inside_constraction":
            self._yic: np.ndarray = self._yc + self._coef["ic"] * (self._yc - self._y[-1])
            params, out_of_boundary, normalized_param = self.suggest_eval_param(
                search_space, self._yic
            )

        elif self._NM_state == "Shrinkage":
            self._ys: np.ndarray = self._y[0] + self._coef["s"] * (
                self._y[self._shrink_num + 1] - self._y[0]
            )
            params, out_of_boundary, normalized_param = self.suggest_eval_param(
                search_space, self._ys
            )

        self._current_y = self._y.copy()
        if self._NM_state != "Initialization":
            if self._NM_state == "Shrinkage":
                self._current_y[self._shrink_num + 1] = normalized_param
            else:
                self._current_y[-1] = normalized_param

        return params, out_of_boundary

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if search_space == {}:
            self._NM_state = "generate_initial_simplex"
            return {}

        if self._NM_state == "generate_initial_simplex":
            self._dim = len(search_space)

            # Our implementation normalizes the search space to unit hypercube [0, 1]^n.
            self._bdrys = np.array([[0, 1] for _ in range(self._dim)])
            self._y = generate_initial_simplex(
                dim=self._dim, edge=self._edge, centroid=self._edge, rng=self._rng
            )
            self._NM_state = "Initialization"

        params, out_of_boundary = self.search(trial, search_space, study)

        if out_of_boundary:
            while True:
                params, out_of_boundary = self.search(
                    trial, search_space, study, f_val=float("inf")
                )
                if not out_of_boundary:
                    break
        trial.set_user_attr("simplex", self._current_y)

        return params

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
