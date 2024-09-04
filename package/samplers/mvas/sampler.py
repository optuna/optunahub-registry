# mypy; ignore-errors
# flake8: noqa
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import optunahub

from gp import GP
from kern import Rbf


def get_input_candidate(x_n_grids):
    x_n_grid_list = [int(g) for g in x_n_grids.split(",")]
    x_n_dim = len(x_n_grid_list)
    xs = np.linspace(0, 1, x_n_grid_list[0])[:, np.newaxis]
    for i in range(x_n_dim - 1):
        new_axis_x = np.tile(
            np.linspace(0, 1, x_n_grid_list[i + 1])[np.newaxis, :, np.newaxis],
            (len(xs), 1, 1),
        )
        xs = np.tile(xs[:, np.newaxis, :], (1, x_n_grid_list[i + 1], 1))
        xs = np.concatenate([xs, new_axis_x], axis=-1).reshape([-1, i + 2])

    return xs


class MeanVarianceAnalysisScalarizationSimulatorSampler(
    optunahub.samplers.SimpleBaseSampler
):
    # By default, search space will be estimated automatically like Optuna's built-in samplers.
    # You can fix the search spacd by `search_space` argument of `SimpleSampler` class.
    def __init__(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution],
        beta=3.0,
        alpha=0.5,
        lengthscale=0.25,
        outputscale=1.0,
        noise_var=1e-4,
        wdim=1,
    ) -> None:
        assert all(
            [
                isinstance(d, optuna.distributions.FloatDistribution)
                and d.low == 0.0
                and d.high == 1.0
                for d in search_space.values()
            ]
        )
        assert len(search_space) > wdim
        assert 0.0 <= alpha <= 1.0
        super().__init__(search_space)
        self._rng = np.random.RandomState()
        self._beta = beta
        self._kern = Rbf(
            len(search_space), lengthscale=lengthscale, outputscale=outputscale
        )
        self._noise_var = noise_var
        self._wdim = wdim
        self._xdim = len(search_space) - wdim
        self._alpha = alpha

    # You need to implement sample_relative method.
    # This method returns a dictionary of hyperparameters.
    # The keys of the dictionary are the names of the hyperparameters, which must be the same as the keys of the search_space argument.
    # The values of the dictionary are the values of the hyperparameters.
    # In this example, sample_relative method returns a dictionary of randomly sampled hyperparameters.
    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        # If search space is empty, all parameter values are sampled randomly by SimpleBaseSampler.
        if search_space == {}:
            return {}

        states = (optuna.trial.TrialState.COMPLETE,)
        trials = study._get_trials(
            deepcopy=False, states=states, use_cache=True
        )

        if len(trials) < 1:
            return {}

        X = np.zeros((len(trials), len(search_space)))  # train_size x n_dim
        for i, trial in enumerate(trials):
            X[i, :] = np.asarray(list(trial.params.values()))

        _sign = (
            -1.0
            if study.direction == optuna.study.StudyDirection.MINIMIZE
            else 1.0
        )
        Y = np.zeros((len(trials), 1))
        for i, trial in enumerate(trials):
            Y[i, 0] = _sign * trial.value

        model = GP(X, Y[:, 0], kern=self._kern, noise_var=self._noise_var)

        xs = get_input_candidate(",".join(["100"] * self._xdim))
        ws = get_input_candidate(",".join(["20"] * self._wdim))
        xws = get_input_candidate(
            ",".join(["100"] * self._xdim) + "," + ",".join(["20"] * self._wdim)
        )
        pws = np.ones(len(ws)) / len(ws)
        nx = len(xs)
        nw = len(ws)

        pos_mu, pos_var = model.predict_f(xws)
        fucb = (pos_mu + self._beta * np.sqrt(pos_var)).reshape([nx, nw])
        flcb = (pos_mu - self._beta * np.sqrt(pos_var)).reshape([nx, nw])
        fmean_ucb = np.sum(fucb * pws, axis=1)
        fmean_lcb = np.sum(flcb * pws, axis=1)
        fdev_ucb = fucb - fmean_lcb[:, np.newaxis]
        fdev_lcb = flcb - fmean_ucb[:, np.newaxis]
        fsqdev_lcb = ((fdev_ucb * fdev_lcb) > 0) * np.minimum(
            fdev_ucb**2, fdev_lcb**2
        )
        fsqdev_ucb = np.maximum(fdev_ucb**2, fdev_lcb**2)
        fvar_lcb = np.sum(fsqdev_lcb * pws, axis=1)
        fvar_ucb = np.sum(fsqdev_ucb * pws, axis=1)
        fmv_ucb = self._alpha * fmean_ucb - (1 - self._alpha) * np.sqrt(
            fvar_lcb
        )
        next_xidx = fmv_ucb.argmax()
        xt = xs[next_xidx].flatten()
        next_widx = pos_var.reshape([nx, nw])[next_xidx, :].argmax()
        wt = ws[next_widx].flatten()

        params = {}  # type: dict[str, Any]
        for i, n in enumerate(search_space.keys()):
            params[n] = xt[i] if i < self._xdim else wt[i - self._xdim]
        return params
