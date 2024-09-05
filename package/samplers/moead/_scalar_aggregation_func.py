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
