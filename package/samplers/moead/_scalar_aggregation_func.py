import numpy as np
from optuna.trial import FrozenTrial


def weighted_sum(
    weight_vector: list[float], trial: FrozenTrial, reference_point: list[float]
) -> float:
    value = np.array(trial.values)
    ref = np.array(reference_point)
    return np.sum(weight_vector * (value - ref))


def tchebycheff(
    weight_vector: list[float], trial: FrozenTrial, reference_point: list[float]
) -> float:
    value = np.array(trial.values)
    ref = np.array(reference_point)
    return np.max(weight_vector * np.abs(value - ref))


# TODO: Is this method correct?
def pbi(
    weight_vector: list[float],
    trial: FrozenTrial,
    reference_point: list[float],
    theta: float = 5.0,
) -> float:
    diff = trial.values - reference_point
    d1 = np.dot(diff, weight_vector)
    d2 = np.linalg.norm(diff - d1 * weight_vector)

    return d1 + theta * d2
