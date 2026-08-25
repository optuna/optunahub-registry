"""Distributionally robust probability threshold robustness (DRPTR).

For a finite environmental space ``Omega`` with reference p.m.f. ``p_ref`` and ambiguity set
``A = {p : ||p - p_ref||_1 <= eps}`` intersected with the simplex, the DRPTR of a cost
vector ``c`` is the infimum ``F = inf_{p in A} sum_w c[w] p[w]``.

Follows eq. (3.1) of Inatsu, Iwazaki & Takeuchi, "Active Learning for Distributionally
Robust Level-Set Estimation", ICML 2021. The paper solves this as a linear program; an L1
ball intersected with the simplex admits a closed form, used here instead.
"""

from __future__ import annotations

import numpy as np


def drptr(c: np.ndarray, p_ref: np.ndarray, eps: float) -> float:
    """Infimum of ``c @ p`` over the ambiguity set, for a general cost vector.

    An L1 ball of radius ``eps`` permits relocating at most ``eps / 2`` of probability mass,
    because shifting mass ``d`` between two atoms changes the L1 distance by ``2 d``. The
    infimum is attained by emptying the costliest atoms in order and placing all the freed
    mass on an atom of minimum cost.
    """
    c = np.asarray(c, dtype=float)
    p = np.asarray(p_ref, dtype=float)
    if c.size == 0:
        raise ValueError("Cost vector must be non-empty.")
    budget = min(eps / 2.0, 1.0)
    if budget <= 0.0:
        return float(c @ p)

    c_min = c.min()
    order = np.argsort(-c, kind="stable")
    c_ord, p_ord = c[order], p[order]
    available = np.where(c_ord > c_min, p_ord, 0.0)
    cum_before = np.concatenate(([0.0], np.cumsum(available)[:-1]))
    take = np.clip(budget - cum_before, 0.0, available)
    return float(c_ord @ (p_ord - take) + c_min * take.sum())


def drptr_binary(indicator: np.ndarray, p_ref: np.ndarray, eps: float, ptr: float) -> float:
    """Infimum for a 0/1 cost vector, in closed form.

    Specialisation of :func:`drptr` for the indicator vectors that arise throughout the
    algorithm. When some atom has cost zero there is somewhere to move mass, so the whole
    budget of ``eps / 2`` is applied; when every atom costs one there is nowhere to move it
    and the value is the reference PTR exactly.

    Args:
        indicator: 0/1 cost vector over the environmental points.
        p_ref: Reference p.m.f. Only used via ``ptr``; accepted for signature symmetry.
        eps: Radius of the L1 ambiguity set.
        ptr: The precomputed reference PTR ``indicator @ p_ref``, which callers in the
            inner loop already have to hand.
    """
    del p_ref
    if indicator.all():
        return float(ptr)
    return max(0.0, float(ptr) - eps / 2.0)


def drptr_linprog(c: np.ndarray, p_ref: np.ndarray, eps: float) -> float:
    """Reference implementation solving the infimum as an explicit linear program.

    Retained for tests. :func:`drptr` is the implementation used by the package.
    """
    from scipy.optimize import linprog

    n = len(c)
    obj = np.concatenate([c, np.zeros(n)])
    a_ub = np.block([[np.eye(n), -np.eye(n)], [-np.eye(n), -np.eye(n)]])
    a_ub = np.vstack([a_ub, np.concatenate([np.zeros(n), np.ones(n)])])
    b_ub = np.concatenate([p_ref, -p_ref, [eps]])
    a_eq = np.concatenate([np.ones(n), np.zeros(n)])[None, :]
    res = linprog(obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=[(0, None)] * (2 * n))
    if not res.success:
        raise RuntimeError(f"Linear program failed: {res.message}")
    return float(res.fun)
