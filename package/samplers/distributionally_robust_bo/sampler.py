import numpy as np
import numpy.typing as npt
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
