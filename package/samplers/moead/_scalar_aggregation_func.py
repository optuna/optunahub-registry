import numpy as np
from optuna.trial import FrozenTrial


def weighted_sum(
    weight_vector: list[float],
    trial: FrozenTrial,
    reference_point: list[float],
    nadir_point: list[float],
) -> float:
    lambda_ = np.array(weight_vector)
    value = np.array(trial.values)
    ref = np.array(reference_point)
    nadir = np.array(nadir_point)
    return float(np.sum(lambda_ * (value - ref) / (nadir - ref)))


def tchebycheff(
    weight_vector: list[float],
    trial: FrozenTrial,
    reference_point: list[float],
    nadir_point: list[float],
) -> float:
    lambda_ = np.array(weight_vector)
    value = np.array(trial.values)
    ref = np.array(reference_point)
    nadir = np.array(nadir_point)
    return float(np.max(lambda_ * np.abs((value - ref) / (nadir - ref))))


# TODO: Is this method correct?
def pbi(
    weight_vector: list[float],
    trial: FrozenTrial,
    reference_point: list[float],
    nadir_point: list[float],
    theta: float = 5.0,
) -> float:
    diff = trial.values - reference_point
    d1 = np.dot(diff, weight_vector)
    d2 = np.linalg.norm(diff - d1 * weight_vector)

    return d1 + theta * d2
