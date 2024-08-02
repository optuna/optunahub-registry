from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod

#########################################################################################
import time
from typing import Any

import GPy
import numpy as np
import optuna
from optuna.distributions import FloatDistribution
import optunahub
from scipy import optimize
from scipy.stats import qmc


#########################################################################################


class RFM_RBF:
    """
    rbf(gaussian) kernel of GPy k(x, y) = variance * exp(- 0.5 * ||x - y||_2^2 / lengthscale**2)
    """

    def __init__(self, lengthscales, input_dim, variance=1, basis_dim=1000):
        self.basis_dim = basis_dim
        self.std = np.sqrt(variance)
        self.random_weights = (1 / np.atleast_2d(lengthscales)) * np.random.normal(
            size=(basis_dim, input_dim)
        )
        self.random_offset = np.random.uniform(0, 2 * np.pi, size=basis_dim)

    def transform(self, X):
        X = np.atleast_2d(X)
        X_transform = X.dot(self.random_weights.T) + self.random_offset
        X_transform = self.std * np.sqrt(2 / self.basis_dim) * np.cos(X_transform)
        return X_transform

    """
    Xは一点のみ
    """

    def transform_grad(self, X):
        X = np.atleast_2d(X)
        X_transform_grad = X.dot(self.random_weights.T) + self.random_offset
        X_transform_grad = (
            -self.std
            * np.sqrt(2 / self.basis_dim)
            * np.sin(X_transform_grad)
            * self.random_weights.T
        )
        return X_transform_grad


def minimize(func, start_points, bounds, jac=None, first_ftol=1e-1, second_ftol=1e-2):
    x = np.copy(start_points)
    func_values = list()
    for i in range(np.shape(x)[0]):
        res = optimize.minimize(
            func, x0=x[i], bounds=bounds, method="L-BFGS-B", options={"ftol": first_ftol}, jac=jac
        )
        func_values.append(res["fun"])
        x[i] = res["x"]

    if second_ftol < first_ftol:
        f_min = np.min(func_values)
        f_max = np.max(func_values)
        index = np.where(func_values <= f_min + (f_max - f_min) * 1e-1)[0]

        for i in index:
            res = optimize.minimize(
                func,
                x0=x[i],
                bounds=bounds,
                method="L-BFGS-B",
                options={"ftol": second_ftol},
                jac=jac,
            )
            func_values[i] = res["fun"]
            x[i] = res["x"]

    min_index = np.argmin(func_values)
    return x[min_index], func_values[min_index]


class GPy_model(GPy.models.GPRegression):
    def __init__(self, X, Y, kernel, noise_var=1e-6, normalizer=True):
        super().__init__(X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        self[".*Gaussian_noise.variance"].constrain_fixed(noise_var)

        if normalizer:
            self.std = self.normalizer.std.copy()
            self.mean = self.normalizer.mean.copy()
        else:
            self.std = 1.0
            self.mean = 0.0

    def minus_predict(self, x):
        x = np.atleast_2d(x)
        return -1 * super().predict_noiseless(x)[0]

    def minus_predict_gradients(self, x):
        x = np.atleast_2d(x)
        # タプルで[0]に平均の勾配, [1]に分散の勾配が返ってくる
        # mu_jac = np.array(list(super().predictive_gradients(x))[0]).ravel()
        mu_jac = super().predictive_gradients(x)[0].ravel()
        return -1 * mu_jac

    def posterior_covariance_between_points(self, X1, X2):
        Kx1 = self.kern.K(X1, self.X)
        Kx2 = self.kern.K(self.X, X2)
        K12 = self.kern.K(X1, X2)

        cov = K12 - Kx1 @ self.posterior.woodbury_inv @ Kx2
        return (self.std**2) * cov

    # X1とX2の共分散行列を求めるものは元からあるが, X1とX2(fidelityの特徴量以外は共通)の共分散行列の対角項だけ求めるものが欲しいため作った
    def diag_covariance_between_points(self, X1, X2):
        assert np.shape(X1) == np.shape(X2), "cannot compute diag (not square matrix)"
        Kx1 = self.kern.K(X1, self.X)
        Kx2 = self.kern.K(self.X, X2)
        K12 = (
            self.kern.K(np.atleast_2d(X1[0, :]), np.atleast_2d(X2[0, :]))
            * np.c_[np.ones(np.shape(X1)[0])]
        )

        diag_var = K12 - np.c_[np.einsum("ij,jk,ki->i", Kx1, self.posterior.woodbury_inv, Kx2)]
        return (self.std**2) * diag_var

    def my_optimize(self, num_restarts=10):
        super().optimize()
        super().optimize_restarts(num_restarts=num_restarts)

    def add_XY(self, X, Y):
        new_X = np.r_[self.X, X]
        new_Y = np.r_[self.Y, Y]

        fidelity_sort_index = np.argsort(new_X[:, -1])
        new_X = new_X[fidelity_sort_index]
        new_Y = new_Y[fidelity_sort_index]
        self.set_XY(new_X, new_Y)

        if self.normalizer is not None:
            self.std = self.normalizer.std.copy()
            self.mean = self.normalizer.mean.copy()
        else:
            self.std = 1.0
            self.mean = 0.0


def set_gpy_regressor(
    GPmodel, X, Y, kernel_bounds, noise_var=1e-6, optimize_num=10, optimize=True, normalizer=True
):
    data_num, input_dim = np.shape(X)
    if GPmodel is None:
        # カーネルの設定
        kernel = GPy.kern.RBF(input_dim=input_dim, variance=1, ARD=True)
        gp_regressor = GPy_model(
            X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer
        )
        gp_regressor[".*Gaussian_noise.variance"].constrain_fixed(noise_var)
        gp_regressor[".*rbf.variance"].constrain_fixed(1)
        if gp_regressor.kern.ARD:
            for i in range(input_dim):
                gp_regressor[".*rbf.lengthscale"][[i]].constrain_bounded(
                    kernel_bounds[0, i], kernel_bounds[1, i]
                )
        else:
            gp_regressor[".*rbf.lengthscale"].constrain_bounded(
                kernel_bounds[0, 0], kernel_bounds[1, 0]
            )
        if optimize:
            gp_regressor.optimize()
            gp_regressor.optimize_restarts(num_restarts=optimize_num)
    else:
        gp_regressor = GPy_model(
            X=X,
            Y=Y,
            kernel=GPmodel.kern,
            noise_var=GPmodel[".*Gaussian_noise.variance"].values,
            normalizer=GPmodel.normalizer,
        )
    return gp_regressor


class BO_core(object):
    __metaclass__ = ABCMeta

    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, optimize=True):
        if GPmodel is None:
            self.GPmodel = set_gpy_regressor(GPmodel, X, Y, kernel_bounds, optimize=optimize)
        else:
            self.GPmodel = GPmodel
        self.y_max = np.max(Y)
        self.unique_X = np.unique(X, axis=0)
        self.input_dim = np.shape(X)[1]
        self.bounds = bounds
        self.bounds_list = bounds.T.tolist()
        self.sampling_num = 10
        self.inference_point = None
        self.top_number = 50
        self.preprocessing_time = 0
        self.max_inputs = None

    def update(self, X, Y, optimize=False):
        self.GPmodel.add_XY(X, Y)
        if optimize:
            self.GPmodel.my_optimize()

        self.y_max = np.max(self.GPmodel.Y)
        self.unique_X = np.unique(self.GPmodel.X, axis=0)

    @abstractmethod
    def acq(self, x):
        pass

    @abstractmethod
    def next_input_pool(self, X):
        pass

    @abstractmethod
    def next_input(self):
        pass

    def _upper_bound(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        return mean + 5.0 * np.sqrt(var)

    def _lower_bound(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        return mean - 5.0 * np.sqrt(var)

    def posteriori_maximum(self):
        num_start = 100

        sampler = qmc.Halton(d=self.input_dim, scramble=False)
        sample = sampler.random(n=num_start)
        x0s = qmc.scale(sample, self.bounds[0], self.bounds[1])

        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean, _ = self.GPmodel.predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number :]
            x0s = np.r_[x0s, self.unique_X[top_idx]]

        if self.inference_point is not None:
            x0s = np.r_[x0s, self.inference_point]

        x_min, f_min = minimize(
            self.GPmodel.minus_predict,
            x0s,
            self.bounds_list,
            jac=self.GPmodel.minus_predict_gradients,
        )

        self.inference_point = np.atleast_2d(x_min)
        return x_min, -1 * f_min

    def sampling_RFM(self, pool_X=None, MES_correction=True):
        # 基底をサンプリング, n_compenontsは基底数, random_stateは基底サンプリング時のseed的なの
        basis_dim = 500 + np.shape(self.GPmodel.X)[0]
        self.rbf_features = RFM_RBF(
            lengthscales=self.GPmodel[".*rbf.lengthscale"].values,
            input_dim=self.input_dim,
            basis_dim=basis_dim,
        )
        X_train_features = self.rbf_features.transform(self.GPmodel.X)

        max_sample = np.zeros(self.sampling_num)
        max_inputs = list()

        A_inv = np.linalg.inv(
            (X_train_features.T).dot(X_train_features)
            + np.eye(self.rbf_features.basis_dim)
            * self.GPmodel[".*Gaussian_noise.variance"].values
        )
        weights_mean = A_inv.dot(X_train_features.T).dot(
            (self.GPmodel.Y - self.GPmodel.mean) / self.GPmodel.std
        )
        weights_var = A_inv * self.GPmodel[".*Gaussian_noise.variance"].values

        try:
            L = np.linalg.cholesky(weights_var)
        except np.linalg.LinAlgError as e:
            print("In RFM-based sampling,", e)
            L = np.linalg.cholesky(weights_var + 1e-5 * np.eye(np.shape(weights_var)[0]))

        # 自分で多次元正規乱数のサンプリング
        standard_normal_rvs = np.random.normal(
            0, 1, size=(np.size(weights_mean), self.sampling_num)
        )
        self.weights_sample = np.c_[weights_mean] + L.dot(standard_normal_rvs)

        if pool_X is None:
            num_start = 100 * self.input_dim

            sampler = qmc.Halton(d=self.input_dim, scramble=False)
            sample = sampler.random(n=num_start)
            x0s = qmc.scale(sample, self.bounds[0], self.bounds[1])

            if np.shape(self.unique_X)[0] <= self.top_number:
                x0s = np.r_[x0s, self.unique_X]
            else:
                mean, _ = self.GPmodel.predict(self.unique_X)
                mean = mean.ravel()
                top_idx = np.argpartition(mean, -self.top_number)[-self.top_number :]
                x0s = np.r_[x0s, self.unique_X[top_idx]]
        else:
            if np.size(pool_X[(self._upper_bound(pool_X) >= self.y_max).ravel()]) > 0:
                pool_X = pool_X[(self._upper_bound(pool_X) >= self.y_max).ravel()]

        for j in range(self.sampling_num):

            def BLR(x):
                X_features = self.rbf_features.transform(x)
                sampled_value = X_features.dot(np.c_[self.weights_sample[:, j]])
                return -(sampled_value * self.GPmodel.std + self.GPmodel.mean).ravel()

            def BLR_gradients(x):
                X_features = self.rbf_features.transform_grad(x)
                sampled_value = X_features.dot(np.c_[self.weights_sample[:, j]])
                return -(sampled_value * self.GPmodel.std).ravel()

            if pool_X is None:
                f_min = np.inf
                x_min = x0s[0]

                x_min, f_min = minimize(BLR, x0s, self.bounds_list, jac=BLR_gradients)
                max_sample[j] = -1 * f_min
                max_inputs.append(x_min)
            else:
                pool_Y = BLR(pool_X)
                min_index = np.argmin(pool_Y)
                max_sample[j] = -1 * pool_Y[min_index]
                max_inputs.append(pool_X[min_index])

        # 観測最大値 + 3×観測ノイズより小さい値は補正
        if MES_correction:
            correction_value = self.y_max + 5 * np.sqrt(
                self.GPmodel[".*Gaussian_noise.variance"].values
            )
            max_sample[max_sample < correction_value] = correction_value
        return max_sample, np.array(max_inputs)

    def sample_path(self, X):
        """
        入力集合 Xに対して, sampling_num個のRFMを用いたsample_pathの対応する値を返す

        Parameter
        -----------------------
        X: numpy array
            inputs (N \times input_dim)

        Return
        -----------------------
        sampled_outputs: numpy array
            sample_path f_s(X) (N \times sampling_num)
        """
        X_features = self.rbf_features.transform(X)
        sampled_outputs = (
            X_features.dot(np.c_[self.weights_sample]) * self.GPmodel.std + self.GPmodel.mean
        )
        return sampled_outputs


class BO(BO_core):
    __metaclass__ = ABCMeta

    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, optimize=True):
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel, optimize=optimize)

    def next_input_pool(self, X):
        self.acquisition_values = self.acq(X)
        min_idx = np.argmin(self.acquisition_values)
        next_input = np.atleast_2d(X[min_idx])
        idx = np.where(np.all(X == next_input, axis=1) == True)[0][0]
        idx_list = np.arange(np.shape(X)[0]).tolist()
        idx_list.remove(idx)
        X = X[idx_list, :]
        return next_input, X

    def next_input(self):
        num_start = 100 * self.input_dim

        sampler = qmc.Halton(d=self.input_dim, scramble=False)
        sample = sampler.random(n=num_start)
        x0s = qmc.scale(sample, self.bounds[0], self.bounds[1])

        x0s = x0s[(self._upper_bound(x0s) >= self.y_max).ravel()]
        if np.shape(self.unique_X)[0] <= self.top_number:
            x0s = np.r_[x0s, self.unique_X]
        else:
            mean, _ = self.GPmodel.predict(self.unique_X)
            mean = mean.ravel()
            top_idx = np.argpartition(mean, -self.top_number)[-self.top_number :]
            x0s = np.r_[x0s, self.unique_X[top_idx]]

        f_min = np.inf
        x_min = x0s[0]
        if self.max_inputs is not None:
            x0s = np.r_[x0s, self.max_inputs]

        x_min, f_min = minimize(self.acq, x0s, self.bounds_list, first_ftol=1e-2, second_ftol=1e-3)
        print("optimized acquisition function value:", -1 * f_min)
        return np.atleast_2d(x_min)


class PI_from_MaxSample(BO):
    def __init__(self, X, Y, bounds, kernel_bounds, GPmodel=None, pool_X=None, optimize=True):
        super().__init__(X, Y, bounds, kernel_bounds, GPmodel=GPmodel, optimize=optimize)

        self.input_dim = np.shape(X)[1]
        self.pool_X = pool_X
        self.sampling_num = 1

        start = time.time()
        self.maximums, self.max_inputs = self.sampling_RFM(pool_X, MES_correction=False)
        self.preprocessing_time = time.time() - start
        print("sampled maximums:", self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def update(self, X, Y, optimize=False):
        super().update(X, Y, optimize=optimize)
        start = time.time()
        self.maximums, self.max_inputs = self.sampling_RFM(self.pool_X, MES_correction=False)
        self.preprocessing_time = time.time() - start
        print("sampled maximums:", self.maximums)
        # print('sampled max inputs:', self.max_inputs)

    def acq(self, x):
        x = np.atleast_2d(x)
        mean, var = self.GPmodel.predict_noiseless(x)
        std = np.sqrt(var)

        return ((self.maximums - mean) / std).ravel()

    def next_input_pool(self, X):
        self.acquisition_values = self.acq(X)
        min_idx = np.argmin(self.acquisition_values)
        next_input = np.atleast_2d(X[min_idx])
        idx = np.where(np.all(X == next_input, axis=1) == True)[0][0]
        idx_list = np.arange(np.shape(X)[0]).tolist()
        idx_list.remove(idx)
        X = X[idx_list, :]
        return next_input, X, self.acquisition_values[min_idx]


#########################################################################################

SimpleBaseSampler = optunahub.load_module("samplers/simple").SimpleBaseSampler


class PIMSSampler(SimpleBaseSampler):  # type: ignore
    # By default, search space will be estimated automatically like Optuna's built-in samplers.
    # You can fix the search spacd by `search_space` argument of `SimpleSampler` class.
    def __init__(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution],
        kernel_bounds: np.ndarray,
    ) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()

        self.kernel_bounds = kernel_bounds
        self.bounds = np.zeros_like(kernel_bounds)

        for i, distribution in enumerate(search_space.values()):
            d = distribution
            assert isinstance(d, FloatDistribution)
            self.bounds[0, i] = d.low
            self.bounds[1, i] = d.high

        self.optimizer = None

    # You need to implement `sample_relative` method.
    # This method returns a dictionary of hyperparameters.
    # The keys of the dictionary are the names of the hyperparameters, which must be the same as the keys of the `search_space` argument.
    # The values of the dictionary are the values of the hyperparameters.
    # In this example, `sample_relative` method returns a dictionary of randomly sampled hyperparameters.
    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        # `search_space` argument must be identical to `search_space` argument input to `__init__` method.
        # This method is automatically invoked by Optuna and `SimpleBaseSampler`.

        # If search space is empty, all parameter values are sampled randomly by SimpleBaseSampler.
        if search_space == {}:
            return {}

        states = (optuna.trial.TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < 1:
            return {}
        elif self.optimizer is None:
            X = np.zeros((len(trials), len(search_space)))
            for i, trial in enumerate(trials):
                X[i, :] = np.asarray(list(trial.params.values()))

            _sign = -1.0 if study.direction == optuna.study.StudyDirection.MINIMIZE else 1.0
            Y = np.zeros((len(trials), 1))
            for i, trial in enumerate(trials):
                Y[i, 0] = _sign * trial.value

            self.optimizer = PI_from_MaxSample(
                X=X,
                Y=Y,
                bounds=self.bounds,
                kernel_bounds=self.kernel_bounds,
            )
        else:
            X_new = np.asarray(list(trials[-1].params.values()))
            _sign = -1.0 if study.direction == optuna.study.StudyDirection.MINIMIZE else 1.0
            Y_new = _sign * trials[-1].value

            if len(trials) % 5 == 4:
                self.optimizer.update(np.atleast_2d(X_new), np.atleast_2d(Y_new), optimize=True)
            else:
                self.optimizer.update(np.atleast_2d(X_new), np.atleast_2d(Y_new), optimize=False)

        new_inputs = self.optimizer.next_input()

        params = {}
        for name, value in zip(search_space.keys(), new_inputs[0]):
            params[name] = value
        return params


########################################################
def f(x):
    return -np.sin(3 * np.sum(x**2)) - np.sum(x**2) ** 2 + 0.7 * np.sum(x**2)


if __name__ == "__main__":

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return f(np.asarray([x, y]))

    search_space = {
        "x": optuna.distributions.FloatDistribution(0, 1),
        "y": optuna.distributions.FloatDistribution(0, 1),
    }

    kernel_bounds = np.array([[1e-3, 1e-3], [1e3, 1e3]])
    sampler = PIMSSampler(search_space=search_space, kernel_bounds=kernel_bounds)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    # study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)
    # optuna.visualization.plot_optimization_history(study)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("optuna_history.png")
