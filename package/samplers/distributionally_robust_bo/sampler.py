from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import numpy.typing as npt
import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.trial import TrialState
from scipy.optimize import linprog


def _compute_worst_case_probability(
    indicator_above_h: npt.NDArray[np.float64],
    ref_p: npt.NDArray[np.float64],
    epsilon_t: float,
) -> float:
    """Computes the worst-case probability for the chance constraint.

    Identifies the worst-case probability distribution within an ambiguity set
    defined by epsilon_t using a linear program.
    """
    grid_num = len(ref_p)

    # Variables structured as duals: [p_1^+, p_1^-, p_2^+, p_2^-, ..., p_n^+, p_n^-]
    c = np.zeros(2 * grid_num)
    c[0::2] = indicator_above_h
    c[1::2] = -indicator_above_h

    a_eq = np.zeros((1, 2 * grid_num))
    a_eq[0, 0::2] = 1.0
    a_eq[0, 1::2] = -1.0
    b_eq = np.array([0.0])

    d_mat = -np.eye(2 * grid_num)
    b_d = np.zeros(2 * grid_num)

    f2_mat = np.ones((1, 2 * grid_num))
    b_f2 = np.array([epsilon_t])

    f3_base = np.zeros((grid_num, 2 * grid_num))
    for i in range(grid_num):
        f3_base[i, 2 * i] = 1.0
        f3_base[i, 2 * i + 1] = -1.0

    # Flipping >= constraints to <= for scipy.optimize.linprog compatibility
    a_f3_1 = -f3_base
    b_f3_1 = -(np.ones(grid_num) - ref_p)

    a_f3_2 = f3_base
    b_f3_2 = ref_p

    a_ub = np.vstack([d_mat, f2_mat, a_f3_1, a_f3_2])
    b_ub = np.concatenate([b_d, b_f2, b_f3_1, b_f3_2])

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, method="highs")

    if res.success:
        ref_p_above = float(np.dot(indicator_above_h, ref_p))
        return float(res.fun + ref_p_above)

    return 0.0


class DistributionallyRobustSampler(BaseSampler):
    """A sampler based on Distributionally Robust Bayesian Optimization.

    Implements the algorithm proposed in "Bayesian Optimization for
    Distributionally Robust Chance-constrained Problem" (ICML 2022).
    """

    def __init__(
        self,
        epsilon_t: float = 0.15,
        h: float = 5.0,
        alpha: float = 0.53,
        seed: Optional[int] = None,
    ) -> None:
        self._epsilon_t = epsilon_t
        self._h = h
        self._alpha = alpha
        self._rng = np.random.RandomState(seed)
        self._random_sampler = optuna.samplers.RandomSampler(seed=seed)

    def infer_relative_search_space(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        x_train = []
        y_train = []
        for t in completed_trials:
            if param_name in t.params and t.value is not None:
                x_train.append(t.params[param_name])
                y_train.append(t.value)

        if len(x_train) < 5:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        x_train_np = np.array(x_train).reshape(-1, 1)
        y_train_np = np.array(y_train)

        # Standardize targets to prevent matrix singularity
        y_mean = np.mean(y_train_np)
        y_std = np.std(y_train_np) + 1e-8
        y_scaled = (y_train_np - y_mean) / y_std

        # RBF Kernel with dynamic length scale
        length_scale = np.std(x_train_np) + 1e-8
        sigma_f = 1.0
        sigma_n = 1e-4

        def rbf_kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
            return sigma_f * np.exp(-0.5 * sqdist / (length_scale**2))

        k_mat = rbf_kernel(x_train_np, x_train_np) + sigma_n * np.eye(len(x_train_np))
        try:
            k_inv = np.linalg.inv(k_mat)
        except np.linalg.LinAlgError:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        if isinstance(param_distribution, optuna.distributions.FloatDistribution):
            candidates = np.linspace(param_distribution.low, param_distribution.high, 50).reshape(
                -1, 1
            )
        elif isinstance(param_distribution, optuna.distributions.IntDistribution):
            candidates = np.arange(param_distribution.low, param_distribution.high + 1).reshape(
                -1, 1
            )
        else:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        k_star = rbf_kernel(candidates, x_train_np)
        k_star_star = rbf_kernel(candidates, candidates)

        post_mean_scaled = k_star @ k_inv @ y_scaled
        post_mean = post_mean_scaled * y_std + y_mean

        post_var_scaled = np.diag(k_star_star - k_star @ k_inv @ k_star.T)
        post_var = np.clip(post_var_scaled, 1e-8, None) * (y_std**2)
        post_std = np.sqrt(post_var)

        beta = 2.0
        lcb = post_mean - beta * post_std
        ucb = post_mean + beta * post_std

        grid_num = len(candidates)
        ref_p = np.ones(grid_num) / grid_num

        lcb_above_h = (lcb > self._h).astype(np.float64)
        ucb_above_h = (ucb > self._h).astype(np.float64)

        dist_ptr_lcb = _compute_worst_case_probability(lcb_above_h, ref_p, self._epsilon_t)
        dist_ptr_ucb = _compute_worst_case_probability(ucb_above_h, ref_p, self._epsilon_t)

        # AF_num == 6 from ICML 2022 Proposed Algorithm
        af_score = np.zeros(grid_num)
        for i in range(grid_num):
            if dist_ptr_lcb >= self._alpha:
                af_score[i] = lcb[i]
            else:
                af_score[i] = ucb[i] - 1e5 * max(0, self._alpha - dist_ptr_ucb)

        # Inject micro-jitter to prevent AF collapse on identical scores
        jitter = self._rng.normal(0, 1e-4, size=grid_num)
        best_idx = int(np.argmax(af_score + jitter))
        best_candidate = candidates[best_idx][0]

        # Force exploration if GP gets stuck on a recent point
        recent_values = x_train[-5:] if len(x_train) >= 5 else x_train
        if best_candidate in recent_values:
            top_k_indices = np.argsort(af_score)[-max(1, grid_num // 10) :]
            best_idx = self._rng.choice(top_k_indices)
            best_candidate = candidates[best_idx][0]

        if isinstance(param_distribution, optuna.distributions.IntDistribution):
            return int(best_candidate)
        return float(best_candidate)
