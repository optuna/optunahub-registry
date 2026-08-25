"""MILE-style acquisition for DRPTR level-set estimation, section 3.2 and Lemmas 3.1-3.2.

The acquisition of eq. (3.2) is

    a_t(x*, w*) = sum over unclassified x of  E_{y*}[ 1[ l_t^F(x; 0 | x*, w*, y*) > alpha ] ]

that is, the expected number of currently unclassified design points that a new observation
at ``(x*, w*)`` would move into the reliable set. Note the ``; 0``: the paper evaluates the
inner bound at accuracy ``eta = 0`` regardless of the ``eta`` used for classification, and
this module follows it.

Adding one observation shifts every lower bound affinely in ``y*``, so the indicator vector
``1[l_t(x, w) > h]`` is piecewise constant in ``y*`` with at most ``|Omega|`` breakpoints per
design point. Lemma 3.1 turns the expectation into a finite sum over those regions, computed
exactly rather than by sampling. Lemma 3.2 skips the transport problem whenever the
reference PTR already fails the level test, and Lemma 3.3 drops regions carrying negligible
predictive probability, at a bounded cost in accuracy.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.special import ndtri

from ._drptr import drptr_binary
from ._lse import DRLevelSetEstimator


def acquisition_values(
    estimator: DRLevelSetEstimator,
    beta: float,
    gamma: float = 1e-3,
    use_variance_term: bool = True,
    zeta: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Acquisition ``a_t^(1)`` of Definition 3.1 over every joint grid point.

    Args:
        estimator: The estimator holding the current observations.
        beta: Squared credible-interval width, as in
            :meth:`DRLevelSetEstimator.indicator_bounds`.
        gamma: Weight of the variance tie-break term. Definition 3.1 takes
            ``max{a_t, gamma * sigma_t}`` so that a candidate is still chosen when ``a_t``
            vanishes everywhere, which it does whenever no single observation can classify
            anything.
        use_variance_term: Set to :obj:`False` to return the bare ``a_t``, dropping the
            tie-break the theory relies on. Intended for testing.
        zeta: Accuracy of the Lemma 3.3 approximation. Regions carrying predictive
            probability below ``zeta / (n_w + 1)`` are skipped, which bounds the error in
            each design point's contribution by ``zeta`` and so the total by
            ``|U_t| * zeta``. Set to zero to evaluate every region exactly.

    Returns:
        ``(values, unclassified)``. ``values`` is indexed over the ``n_x * n_w`` joint grid;
        ``unclassified`` holds the indices of unclassified *design* points, over ``n_x``. An
        empty ``unclassified`` means the run has terminated and ``values`` is all zeros, so
        callers must check it rather than taking an argmax.
    """
    n_w, noise = estimator.n_w, estimator.noise
    mean, cov = estimator.posterior_joint()
    var = np.diag(cov)
    sd = np.sqrt(var)
    _, _, unclassified_mask = estimator.classify(beta, posterior=(mean, var))
    unclassified = np.flatnonzero(unclassified_mask)
    values = np.zeros(estimator.kernel.shape[0])
    if unclassified.size == 0:
        return values, unclassified

    if zeta < 0.0:
        raise ValueError(f"zeta must be non-negative, got {zeta}.")
    region_tol = zeta / (n_w + 1)  # Lemma 3.3
    sqrt_beta = np.sqrt(beta)
    rows = [slice(ix * n_w, (ix + 1) * n_w) for ix in unclassified]
    p_ref = estimator.p_ref

    for star in range(estimator.kernel.shape[0]):
        denom = var[star] + noise
        slope = cov[:, star] / denom  # d mean_new / d y*
        intercept = mean - slope * mean[star]
        sd_new = np.sqrt(np.clip(var - cov[:, star] ** 2 / denom, 1e-12, None))
        pred_sd = np.sqrt(denom)
        total = 0.0

        for row in rows:
            a_row, b_row, s_row = intercept[row], slope[row], sd_new[row]
            # l_new(w) = a + b*y* - sqrt(beta)*s crosses h at y* = (h + sqrt(beta)*s - a)/b.
            live = np.abs(b_row) > 1e-15
            breakpoints = (estimator.h + sqrt_beta * s_row[live] - a_row[live]) / b_row[live]
            edges = np.concatenate(([-np.inf], np.sort(breakpoints), [np.inf]))
            cdf = ndtr((edges - mean[star]) / pred_sd)
            for region in range(len(edges) - 1):
                p_region = cdf[region + 1] - cdf[region]
                if p_region < region_tol:
                    continue
                # Representative y* at the region's probability midpoint. Always finite and
                # strictly interior, which an offset from an infinite edge would not be.
                y_rep = mean[star] + pred_sd * ndtri(0.5 * (cdf[region] + cdf[region + 1]))
                indicator = (a_row + b_row * y_rep - sqrt_beta * s_row > estimator.h).astype(float)
                ptr = float(indicator @ p_ref)
                # Lemma 3.2: the DRPTR is an infimum over the ambiguity set and so cannot
                # exceed the reference PTR. If that is already at or below alpha, the
                # indicator is zero and the transport problem can be skipped.
                if ptr <= estimator.alpha:
                    continue
                if drptr_binary(indicator, p_ref, estimator.eps, ptr) > estimator.alpha:
                    total += p_region

        values[star] = max(total, gamma * sd[star]) if use_variance_term else total
    return values, unclassified
