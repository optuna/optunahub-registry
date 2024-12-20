from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import optunahub

from . import _shape_functions as shape_functions
from . import _transformation_functions as transformation_functions


class Problem(optunahub.benchmarks.BaseProblem):
    """A _WFG (Walking Fish Group) problem."""

    def __init__(self, problem_id: int, dimension: int, n_objectives: int, k: int) -> None:
        """Initialize the problem.
        Args:
            problem_id:
                The problem ID.
            dimension:
                The dimension of the problem.
            n_objectives:
                The number of objectives.
            k:
                The degree of the Pareto front.

        Please refer to the following paper for the details of the problems:
        S. Huband, P. Hingston, L. Barone, and L. While, A review of multiobjective test problems and a scalable
        test problem toolkit, IEEE Transactions on Evolutionary Computation, 2006, 10(5): 477-506.
        """

        assert 1 <= problem_id <= 9, f"problem_id must be in [1, 9], but got {problem_id}."
        assert dimension > 0
        assert n_objectives > 0
        assert k > 0

        self._dimension = dimension
        self._n_objectives = n_objectives

        self._problem: _WFG1 | _WFG2 | _WFG3 | _WFG4 | _WFG5 | _WFG6 | _WFG7 | _WFG8 | _WFG9
        if problem_id == 1:
            self._problem = _WFG1(dimension, n_objectives, k)
        elif problem_id == 2:
            self._problem = _WFG2(dimension, n_objectives, k)
        elif problem_id == 3:
            self._problem = _WFG3(dimension, n_objectives, k)
        elif problem_id == 4:
            self._problem = _WFG4(dimension, n_objectives, k)
        elif problem_id == 5:
            self._problem = _WFG5(dimension, n_objectives, k)
        elif problem_id == 6:
            self._problem = _WFG6(dimension, n_objectives, k)
        elif problem_id == 7:
            self._problem = _WFG7(dimension, n_objectives, k)
        elif problem_id == 8:
            self._problem = _WFG8(dimension, n_objectives, k)
        elif problem_id == 9:
            self._problem = _WFG9(dimension, n_objectives, k)
        else:
            assert False, "do not reach here"

        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(low=0.0, high=2.0 * (i + 1))
            for i in range(self._dimension)
        }

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._n_objectives

    def evaluate(self, params: dict[str, float]) -> list[float]:
        """Evaluate the objective function.
        Args:
            params:
                Decision variable, e.g., evaluate({"x0": 1.0, "x1": 2.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The objective value.

        """
        param_list = [params[name] for name in self._search_space]
        return self._problem(np.asarray(param_list)).tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)


class _BaseWFG:
    def __init__(
        self,
        S: np.ndarray,
        A: np.ndarray,
        upper_bounds: np.ndarray,
        shapes: list[shape_functions.BaseShapeFunction],
        transformations: list[list[transformation_functions.BaseTransformations]],
    ) -> None:
        assert all(S > 0)
        assert all((A == 0) + (A == 1))
        assert all(upper_bounds > 0)

        self._S = S
        self._A = A
        self._upper_bounds = upper_bounds
        self._shapes = shapes
        self._transformations = transformations

    def __call__(self, z: np.ndarray) -> np.ndarray:
        S = self._S
        A = self._A
        unit_z = z / self._upper_bounds
        shapes = self._shapes
        transformations = self._transformations

        y = unit_z
        for t_p in transformations:
            _y = np.empty((len(t_p),))
            for i in range(len(t_p)):
                if isinstance(t_p[i], transformation_functions.BaseReductionTransformation):
                    _y[i] = t_p[i](y)
                else:
                    _y[i] = t_p[i](y[i])
            y = _y

        x = np.empty(y.shape)
        x[:-1] = np.maximum(y[-1], A) * (y[:-1] - 0.5) + 0.5
        x[-1] = y[-1]

        f = x[-1] + S * np.asarray([h(m + 1, x[:-1]) for m, h in enumerate(shapes)])
        return f


class _WFG1:
    """_WFG1

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):
        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConvexShapeFunction(M) for _ in range(M - 1)]
        shapes.append(shape_functions.MixedConvexOrConcaveShapeFunction(M, 1, 5))

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(4)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        # transformations[0] = [lambda y: y for _ in range(k)]
        for _ in range(n - k):
            transformations[0].append(transformation_functions.LinearShiftTransformation(0.35))

        # transformations[1] = [lambda y: y for _ in range(k)]

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[1].append(
                transformation_functions.FlatRegionBiasTransformation(0.8, 0.75, 0.85)
            )

        transformations[2] = [
            transformation_functions.PolynomialBiasTransformation(0.02) for _ in range(n)
        ]

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[3] = [
            transformation_functions.WeightedSumReductionTransformation(
                2 * np.arange(i * k // (M - 1) + 1, (i + 1) * k // (M - 1) + 1),
                lambda y: _input_converter(i, y),
            )
            for i in range(M - 1)
        ]
        transformations[3].append(
            transformation_functions.WeightedSumReductionTransformation(
                2 * np.arange(k, n) + 1,
                lambda y: y[k:n],
            )
        )

        self.wfg = _BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class _WFG2:
    """_WFG2

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):
        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments // 2
        assert (n_arguments - k) % 2 == 0

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConvexShapeFunction(M) for _ in range(M - 1)]
        shapes.append(shape_functions.DisconnectedShapeFunction(M, 1, 1, 5))

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[0].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            indices = [k + 2 * (i + 1 - k) - 2, k + 2 * (i - k + 1) - 1]
            return y[indices]

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for i in range(k, n // 2):
            transformations[1].append(
                transformation_functions.NonSeparableReductionTransformation(
                    2, lambda y: _input_converter0(i, y)
                )
            )

        def _input_converter1(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = [
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(k // (M - 1)),
                lambda y: _input_converter1(i, y),
            )
            for i in range(M - 1)
        ]
        transformations[2].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n // 2 - k),
                lambda y: y[k : n // 2],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = _BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class _WFG3:
    """_WFG3

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):
        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments // 2
        assert (n_arguments - k) % 2 == 0

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.zeros(M - 1)
        A[0] = 1
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.LinearShapeFunction(M) for _ in range(M)]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[0].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            indices = [k + 2 * (i + 1 - k) - 2, k + 2 * (i - k + 1) - 1]
            return y[indices]

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for i in range(k, n // 2):
            transformations[1].append(
                transformation_functions.NonSeparableReductionTransformation(
                    2, lambda y: _input_converter0(i, y)
                )
            )

        def _input_converter1(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = [
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(k // (M - 1)),
                lambda y: _input_converter1(i, y),
            )
            for i in range(M - 1)
        ]
        transformations[2].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n // 2 - k),
                lambda y: y[k : n // 2],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = _BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class _WFG4:
    """_WFG4

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):
        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(2)]

        transformations[0] = [
            transformation_functions.MultiModalShiftTransformation(30, 10, 0.35) for _ in range(n)
        ]

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        # transformations[1] = []
        for i in range(M - 1):
            transformations[1].append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter(i, y)
                )
            )
        transformations[1].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1]]

        self.wfg = _BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class _WFG5:
    """_WFG5

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):
        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(2)]

        transformations[0] = [
            transformation_functions.DeceptiveShiftTransformation(0.35, 0.001, 0.05)
            for _ in range(n)
        ]

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[1] = []
        for i in range(M - 1):
            transformations[1].append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter(i, y)
                )
            )
        transformations[1].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1]]

        self.wfg = _BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class _WFG6:
    """_WFG6

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):
        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(2)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[0].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        # transformations[1] = []
        for i in range(M - 1):
            transformations[1].append(
                transformation_functions.NonSeparableReductionTransformation(
                    k // (M - 1), lambda y: _input_converter(i, y)
                )
            )
        transformations[1].append(
            transformation_functions.NonSeparableReductionTransformation(
                n - k,
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1]]

        self.wfg = _BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class _WFG7:
    """_WFG7

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):
        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        class _InputConverter:
            def __init__(self, i: int, n: int) -> None:
                self._i = i
                self._n = n

            def __call__(self, y: np.ndarray) -> np.ndarray:
                return y[self._i : self._n]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [
            transformation_functions.ParameterDependentBiasTransformation(
                np.ones(n - i),
                _InputConverter(i, n),
                0.98 / 49.98,
                0.02,
                50,
                i,
            )
            for i in range(k)
        ]
        for _ in range(n - k):
            transformations[0].append(transformation_functions.IdenticalTransformation())

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[1].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter1(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = []
        for i in range(M - 1):
            transformations[2].append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter1(i, y)
                )
            )
        transformations[2].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = _BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class _WFG8:
    """_WFG8

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):
        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            return y[: i - 1]

        class _InputConverter:
            def __init__(self, i: int) -> None:
                self._i = i

            def __call__(self, y: np.ndarray) -> np.ndarray:
                return y[: self._i - 1]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for i in range(k, n):
            transformations[0].append(
                transformation_functions.ParameterDependentBiasTransformation(
                    np.ones(i - 1),
                    _InputConverter(i),
                    0.98 / 49.98,
                    0.02,
                    50,
                    i,
                )
            )

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[1].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = []
        for i in range(M - 1):
            transformations[2].append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter(i, y)
                )
            )
        transformations[2].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = _BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class _WFG9:
    """_WFG9

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):
        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        class _InputConverter:
            def __init__(self, i: int, n: int) -> None:
                self._i = i
                self._n = n

            def __call__(self, y: np.ndarray) -> np.ndarray:
                return y[self._i : self._n]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [
            transformation_functions.ParameterDependentBiasTransformation(
                np.ones(n - i),
                _InputConverter(i, n),
                0.98 / 49.98,
                0.02,
                50,
                i,
            )
            for i in range(n - 1)
        ]
        transformations[0].append(transformation_functions.IdenticalTransformation())

        transformations[1] = [
            transformation_functions.DeceptiveShiftTransformation(0.35, 0.001, 0.05)
            for _ in range(k)
        ]
        for _ in range(n - k):
            transformations[1].append(
                transformation_functions.MultiModalShiftTransformation(30, 95, 0.35)
            )

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = []
        for i in range(M - 1):
            transformations[2].append(
                transformation_functions.NonSeparableReductionTransformation(
                    k // (M - 1), lambda y: _input_converter(i, y)
                )
            )
        transformations[2].append(
            transformation_functions.NonSeparableReductionTransformation(
                n - k,
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = _BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)
