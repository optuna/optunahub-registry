from typing import Any

import GPy
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np
import numpyro
from numpyro.infer import init_to_value
import optuna
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.samplers._base import BaseSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
from scipy import optimize


class PLMBOSampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        *,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
    ) -> None:
        super().__init__(search_space)
        self.X: np.ndarray | None = None
        self.Y: np.ndarray | None = None
        self._rng = np.random.RandomState()
        self.obj_dim: int | None = None
        self.input_dim: int | None = None
        self.bounds: list | None = None
        self.pc: np.ndarray | None = None
        self.ir: list | None = None
        self.w: np.ndarray | None = None
        self.gp_models: list | None = None
        self.sample_w = None
        self._n_startup_trials = n_startup_trials
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if self.obj_dim is None:
            self.obj_dim = len(study.directions)
        if self.pc is None:
            self.pc = np.zeros((0, 2, self.obj_dim))
        if self.ir is None:
            self.ir = []
        if self.w is None:
            self.w = np.full(self.obj_dim, 1 / self.obj_dim)

        if search_space == {}:
            return {}

        # Bounds initialization requires search space information
        if self.bounds is None:
            self.bounds = []
            for n, d in search_space.items():
                if isinstance(d, FloatDistribution):
                    self.bounds.append((d.low, d.high))
                else:
                    raise ValueError("Unsupported distribution")
        if self.input_dim is None:
            self.input_dim = len(self.bounds)

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < self._n_startup_trials:
            return {}

        self.X = np.array([list(t.params.values()) for t in trials])
        self.Y = np.array([t.values for t in trials])

        self.__fit_gp()

        self.__add_comparison()
        self.__update_w()

        x_candidates = []
        values = []
        for i in range(10):
            x0 = [
                np.random.uniform(self.bounds[i][0], self.bounds[i][1])
                for i in range(self.input_dim)
            ]
            opt = optimize.minimize(self.__acq, x0, bounds=self.bounds)
            x_candidates.append(opt.x)
            values.append(opt.fun)
        x_candidate = x_candidates[np.argmin(values)]

        params = {}
        for i, n in enumerate(search_space):
            if isinstance(search_space[n], FloatDistribution):
                params[n] = x_candidate[i]
            else:
                raise ValueError("Unsupported distribution")

        print(params)
        return params

    def __add_comparison(self) -> None:
        y_rnd_1 = np.random.rand(self.obj_dim)
        y_rnd_2 = np.random.rand(self.obj_dim)

        print("1:", y_rnd_1, "2:", y_rnd_2)

        while True:
            try:
                winner = int(input("Which is better? 1 or 2: "))
                break
            except ValueError:
                print("Invalid input!")
        if winner == 1:
            self.pc = np.r_[self.pc, [[y_rnd_1, y_rnd_2]]]
        elif winner == 2:
            self.pc = np.r_[self.pc, [[y_rnd_2, y_rnd_1]]]

        y_rnd = np.random.rand(self.obj_dim)

        print(y_rnd)

        while True:
            try:
                winner = (
                    int(
                        input(
                            f"Which objective function should be improved the most? 1 ~ {self.obj_dim}: "
                        )
                    )
                    - 1
                )
                break
            except ValueError:
                print("Invalid input!")

        assert self.obj_dim is not None
        if winner >= 0 and winner < self.obj_dim:
            for i in range(self.obj_dim):
                if i != winner:
                    assert self.ir is not None
                    self.ir.append([y_rnd, winner, i])

    def __fit_gp(self) -> None:
        self.gp_models = []
        assert self.obj_dim is not None
        for i in range(self.obj_dim):
            kernel = GPy.kern.RBF(self.input_dim)
            assert self.Y is not None
            model = GPy.models.GPRegression(
                self.X, self.Y[:, i].reshape(self.Y.shape[0], 1), kernel
            )
            model[".*Gaussian_noise.variance"].constrain_bounded(0.000001, 0.001, warning=False)
            model[".*rbf.variance"].constrain_bounded(0.01, 3, warning=False)
            model[".*rbf.lengthscale"].constrain_bounded(0.2, 50, warning=False)
            model.optimize(messages=False, max_iters=1e5)
            assert self.gp_models is not None
            self.gp_models.append(model)

    def __update_w(self) -> None:
        u_sigma = 0.01
        # preference information
        assert self.pc is not None
        assert self.ir is not None
        y_pc = np.ones((len(self.pc)))
        y_ir = np.ones((len(self.ir)))

        def mcmc_model() -> None:
            # prior
            w = numpyro.sample("w", numpyro.distributions.Dirichlet(np.full(self.obj_dim, 2)))

            assert self.pc is not None
            u_w = self.__u_est(self.pc[:, 0], w)
            u_l = self.__u_est(self.pc[:, 1], w)

            assert self.ir is not None
            l_f = [l_[0] for l_ in self.ir[:]]
            para = (u_w - u_l) / (np.sqrt(2) * u_sigma)
            para = jnp.where(para < -30, -30, para)
            para = norm.cdf(para, 0, 1)
            para = jnp.maximum(para, 1e-14)

            du_w = self.__dudf(l_f, [li[1] for li in self.ir[:]], w)
            du_l = self.__dudf(l_f, [li[2] for li in self.ir[:]], w)
            para_d = (du_w - du_l) / (np.sqrt(2) * u_sigma)
            para_d = norm.cdf(para_d, 0, 1)
            para_d = jnp.maximum(para_d, 1e-14)

            # observations
            with numpyro.plate("data", len(y_pc)):
                numpyro.sample("obs", numpyro.distributions.Bernoulli(para), obs=y_pc)
            with numpyro.plate("data2", len(y_ir)):
                numpyro.sample("obs2", numpyro.distributions.Bernoulli(para_d), obs=y_ir)

        if len(y_pc) == 0:

            def mcmc_model() -> None:
                w = numpyro.sample("w", numpyro.distributions.Dirichlet(np.full(self.obj_dim, 2)))  # noqa: F841

        # sampling
        kern = numpyro.infer.NUTS(
            mcmc_model, init_strategy=init_to_value(values={"w": jnp.array(self.w)})
        )
        mcmc = numpyro.infer.MCMC(kern, num_warmup=2000, num_samples=1000)
        mcmc.run(jax.random.PRNGKey(1))
        sample_n_1 = mcmc.get_samples()["w"]

        mean1 = np.mean(sample_n_1, axis=0)

        kern = numpyro.infer.NUTS(mcmc_model)
        mcmc = numpyro.infer.MCMC(kern, num_warmup=2000, num_samples=1000)
        mcmc.run(jax.random.PRNGKey(1))
        sample_n_2 = mcmc.get_samples()["w"]

        mean2 = np.mean(sample_n_2, axis=0)

        def ll(w: np.ndarray) -> np.ndarray:
            assert self.pc is not None
            assert self.ir is not None
            u_w = self.__u_est(self.pc[:, 0], w)
            u_l = self.__u_est(self.pc[:, 1], w)
            l_f = [l_[0] for l_ in self.ir[:]]
            para = (u_w - u_l) / (np.sqrt(2) * u_sigma)
            para = np.where(para < -20, -20, para)
            para = norm.cdf(para, 0, 1)
            para = np.maximum(para, 1e-14)

            du_w = self.__dudf(l_f, [li[1] for li in self.ir[:]], w)
            du_l = self.__dudf(l_f, [li[2] for li in self.ir[:]], w)
            para_d = (du_w - du_l) / (np.sqrt(2) * u_sigma)
            para_d = norm.cdf(para_d, 0, 1)
            para_d = np.maximum(para_d, 1e-14)

            return np.sum(np.log(para)) + np.sum(np.log(para_d))

        if ll(mean1) > ll(mean2):
            self.sample_w = sample_n_1
        else:
            self.sample_w = sample_n_2

        self.w = np.mean(self.sample_w, axis=0)

    def __acq(self, x: np.ndarray) -> np.ndarray:
        # initialize acquisition
        alpha = 0

        # current best
        assert self.sample_w is not None
        n = len(self.sample_w)
        ubest = np.zeros(len(self.sample_w))
        assert self.Y is not None
        ubest_tmp = np.zeros((self.Y.shape[0], n))
        for i in range(self.Y.shape[0]):
            ubest_tmp[i, :] = np.min(np.tile(self.Y[i], (n, 1)) / self.sample_w, axis=1)
        ubest_tmp = np.where(ubest_tmp < -500, -500, ubest_tmp)
        ubest_tmp = np.where(ubest_tmp > 500, 500, ubest_tmp)
        ubest = np.max(ubest_tmp, axis=0)

        normal = []
        assert self.obj_dim is not None
        for i in range(self.obj_dim):
            normal.append(np.random.normal(0, 1, n))

        # calculate acquisition
        yy = []
        for i in range(self.obj_dim):
            gp_pred = self.gp_models[i].predict(x.reshape(1, self.input_dim))
            yy.append(
                (
                    np.tile(gp_pred[0][0], (1, n))
                    + np.sqrt(gp_pred[1][0]) * normal[i]
                    + np.random.normal(0, 0.001, (1, n))
                )[0]
            )
        w_sample = []
        for i in range(self.obj_dim):
            w_sample.append(self.sample_w[:, i])
        uu = self.__u_est(np.array(yy).T, np.array(w_sample).T).T
        uaftermax = np.maximum(uu - ubest, 0)
        alpha = np.mean(uaftermax)
        return -alpha

    def __u_est(self, x: np.ndarray, w: np.ndarray) -> jnp.ndarray:
        x = jnp.array(x)
        w = jnp.array(w)
        x = (x / w).T
        x = jnp.where(x < -500, -500, x)
        x = jnp.where(x > 500, 500, x)
        re = x[0]
        assert self.obj_dim is not None
        for i in range(1, self.obj_dim):
            re = jnp.minimum(x[i], re)
        return re

    # differential of U
    def __dudf(self, x: np.ndarray, f: list, weight: np.ndarray) -> jnp.ndarray:
        x = jnp.array(x)
        weight = jnp.array(weight)
        f = jnp.array(f)
        x = x / weight
        min_idx = jnp.argmin(x, axis=1)
        min_idx = min_idx.astype(jnp.int32)
        re = jnp.zeros((len(x), self.obj_dim))
        vidx = jnp.array(range(len(x)), dtype=jnp.int32)
        sliceidx = jnp.s_[vidx, min_idx]
        re = re.at[sliceidx].set(1 / weight[min_idx])
        return re[[vidx], [f]]
