from __future__ import annotations

import math

import optuna
from optunahub.benchmarks import BaseProblem


class F1_ExpGaussian(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=-5, high=5),
            "x2": optuna.distributions.FloatDistribution(low=-5, high=5),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2 = params["x1"], params["x2"]
        return math.exp(-((x1 - 1) ** 2)) / (1.2 + (x2 - 2.5) ** 2)


class F2_ComplexTrigonometric(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=-5, high=5),
            "x2": optuna.distributions.FloatDistribution(low=0, high=10),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2 = params["x1"], params["x2"]
        cos_x1 = math.cos(x1)
        sin_x1 = math.sin(x1)
        return math.exp(-(x1**2)) * cos_x1 * sin_x1 * (cos_x1 * sin_x1**2 - 1) * (x2 - 5)


class F3_HighDimensional(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            f"x{i+1}": optuna.distributions.FloatDistribution(low=-5, high=10) for i in range(5)
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        sum_squares = sum((params[f"x{i+1}"] - 3) ** 2 for i in range(5))
        return 10 / (5 + sum_squares)


class F4_Rational(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=2, high=15),  # Avoid x1=10
            "x2": optuna.distributions.FloatDistribution(low=0.1, high=5),  # Avoid x2=0
            "x3": optuna.distributions.FloatDistribution(low=-5, high=5),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2, x3 = params["x1"], params["x2"], params["x3"]
        # Add small epsilon to avoid division by zero
        denominator = x2**2 * (x1 - 10)
        if abs(denominator) < 1e-10:
            return -1e6  # Large negative penalty
        return 30 * (x1 - 1) * (x3 - 1) / denominator


class F5_SimpleTrigonometric(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=0, high=2 * math.pi),
            "x2": optuna.distributions.FloatDistribution(low=0, high=2 * math.pi),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2 = params["x1"], params["x2"]
        return 6 * math.sin(x1) * math.cos(x2)


class F6_NonlinearCombination(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=0, high=8),
            "x2": optuna.distributions.FloatDistribution(low=0, high=8),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2 = params["x1"], params["x2"]
        return (x1 - 3) * (x2 - 3) + 2 * math.sin((x1 - 4) * (x2 - 4))


class F7_PolynomialRational(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=0, high=6),
            "x2": optuna.distributions.FloatDistribution(low=0, high=6),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2 = params["x1"], params["x2"]
        return (x1 - 3) ** 4 + (x2 - 3) ** 3 - (x2 - 3) / ((x2 - 2) ** 4 + 10)


class F8_PowerSum(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=0.1, high=5),
            "x2": optuna.distributions.FloatDistribution(low=0.1, high=5),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2 = params["x1"], params["x2"]
        return 1 / (1 + x1 ** (-4)) + 1 / (1 + x2 ** (-4))


class F9_QuarticQuadratic(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=-2, high=2),
            "x2": optuna.distributions.FloatDistribution(low=-2, high=2),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MINIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2 = params["x1"], params["x2"]
        return x1**4 - x1**3 + x2**2 / 2 - x2


class F10_InverseQuadratic(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=-5, high=5),
            "x2": optuna.distributions.FloatDistribution(low=-5, high=5),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2 = params["x1"], params["x2"]
        return 8 / (2 + x1**2 + x2**2)


class F11_FractionalPower(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(
                low=0, high=5
            ),  # x1 >= 0 for fractional power
            "x2": optuna.distributions.FloatDistribution(
                low=0, high=5
            ),  # x2 >= 0 for fractional power
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2 = params["x1"], params["x2"]
        return x1 ** (3 / 5) + x2 ** (3 / 2) - x2 - x1


class F12_ProductSum(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            f"x{i+1}": optuna.distributions.FloatDistribution(low=-5, high=5) for i in range(10)
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        return (
            params["x1"] * params["x2"]
            + params["x3"] * params["x4"]
            + params["x5"] * params["x6"]
            + params["x7"] * params["x8"]
            + params["x9"] * params["x10"]
        )


class F13_LinearRational(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=0.1, high=5),  # Avoid zero
            "x2": optuna.distributions.FloatDistribution(low=-5, high=5),
            "x3": optuna.distributions.FloatDistribution(low=0.1, high=5),  # Avoid zero
            "x4": optuna.distributions.FloatDistribution(low=-5, high=5),
            "x5": optuna.distributions.FloatDistribution(low=0.1, high=5),  # Avoid zero
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2, x3, x4, x5 = (params[f"x{i+1}"] for i in range(5))
        return -5.41 + 4.9 * x4 - x1 + x2 / x5 - (3 * x4) / (x1 * x3)


class F14_ProductRatio(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            f"x{i+1}": optuna.distributions.FloatDistribution(low=0.1, high=5)
            for i in range(6)  # All positive to avoid division issues
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2, x3, x4, x5, x6 = (params[f"x{i+1}"] for i in range(6))
        return (x5 * x6) / (x2 * x4) * (x1 * x3)


class F15_ComplexRational(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=-5, high=5),
            "x2": optuna.distributions.FloatDistribution(low=-5, high=5),
            "x3": optuna.distributions.FloatDistribution(low=-5, high=5),
            "x4": optuna.distributions.FloatDistribution(low=0.1, high=5),  # Positive for cube
            "x5": optuna.distributions.FloatDistribution(low=0.1, high=3),  # Limited range for x^4
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2, x3, x4, x5 = (params[f"x{i+1}"] for i in range(5))
        numerator = 2 * x2 + 3 * x3**2
        denominator = 4 * x4**3 + 5 * x5**4
        return 0.81 + 24.3 * numerator / denominator


class F16_TrigonometricRatio(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            f"x{i+1}": optuna.distributions.FloatDistribution(
                low=0.1, high=1.4
            )  # Avoid tan singularities
            for i in range(5)
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2, x3, x4, x5 = (params[f"x{i+1}"] for i in range(5))
        try:
            term1 = math.tan(x1) / math.tan(x3)
            term2 = math.tan(x5) / math.tan(x2)
            term3 = math.tan(x4)
            return 32 - 3 * term1 * term2 * term3
        except (ZeroDivisionError, ValueError):
            return -1e6  # Penalty for invalid values


class F17_MixedTrigonometric(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(low=0, high=2 * math.pi),
            "x2": optuna.distributions.FloatDistribution(
                low=0.1, high=1.4
            ),  # Avoid tan singularities
            "x3": optuna.distributions.FloatDistribution(low=-3, high=3),
            "x4": optuna.distributions.FloatDistribution(
                low=0.1, high=math.pi - 0.1
            ),  # Avoid sin=0
            "x5": optuna.distributions.FloatDistribution(low=-5, high=5),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2, x3, x4, x5 = (params[f"x{i+1}"] for i in range(5))
        try:
            term1 = math.cos(x1) - math.tan(x2)
            term2 = math.tanh(x3) / math.sin(x4)
            return 22 - 4.2 * term1 * term2
        except (ZeroDivisionError, ValueError):
            return -1e6  # Penalty for invalid values


class F18_Product(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            f"x{i+1}": optuna.distributions.FloatDistribution(low=-3, high=3) for i in range(5)
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        return math.prod(params[f"x{i+1}"] for i in range(5))


class F19_ComplexTrigExp(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x1": optuna.distributions.FloatDistribution(
                low=0.1, high=1.4
            ),  # Avoid tan singularities
            "x2": optuna.distributions.FloatDistribution(low=-2, high=2),
            "x3": optuna.distributions.FloatDistribution(low=0, high=2 * math.pi),
            "x4": optuna.distributions.FloatDistribution(low=0, high=2 * math.pi),
            "x5": optuna.distributions.FloatDistribution(
                low=0.1, high=1.4
            ),  # Avoid tan singularities
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x2, x3, x4, x5 = (params[f"x{i+1}"] for i in range(5))
        try:
            term1 = math.tan(x1) / math.exp(x2)
            term2 = math.cos(x3) * math.sin(x4) - math.tan(x5)
            return 12 - 6 * term1 * term2
        except (ZeroDivisionError, ValueError):
            return -1e6  # Penalty for invalid values


class F20_HarmonicSum(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            f"x{i+1}": optuna.distributions.FloatDistribution(low=0.01, high=10)  # Positive only
            for i in range(10)
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MINIMIZE]  # Minimize sum of reciprocals

    def evaluate(self, params: dict[str, float]) -> float:
        return sum(1 / params[f"x{i+1}"] for i in range(10))


class F21_ScaledTrigonometric(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            f"x{i+1}": optuna.distributions.FloatDistribution(low=-2 * math.pi, high=2 * math.pi)
            for i in range(5)
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MAXIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        x1, x5 = params["x1"], params["x5"]
        return 2 - 2.1 * math.cos(9.8 * x1) * math.sin(1.3 * x5)
