# mypy; ignore-errors
import numpy as np
from scipy.linalg import cho_solve


class GP:
    def __init__(self, x, y, kern, noise_var=1e-2):
        """Gaussian Process Regression

        Parameters
        ----------
        x : 2d-ndarray
            Input data X
        y : 1d-ndarray
            Output data y
        kern : kernel class object
            kernel class
        noise_var : float, optional
            Observation noise's variance, by default is 1e-2
        """

        self.noise_var = noise_var

        self.x = x
        self.y = y
        self.n_data = x.shape[0]
        self.n_dim = x.shape[1]
        self.kern = kern

        self.K = self.kern.K(x, x)
        self.K_varI = self.K + noise_var * np.eye(self.n_data)

        self.K_varI_L = np.linalg.cholesky(
            self.K + np.eye(self.n_data) * self.noise_var
        ).T

    @staticmethod
    def prior_sampling(xs, rng, kern):
        cov = kern.K(xs, xs)
        return rng.multivariate_normal(np.zeros(len(xs)), cov)

    def posterior_sampling(self, xs, rng):
        mu = self.predict_mean(xs)
        cov = self.predict_cov(xs, xs)
        return rng.multivariate_normal(mu, cov)

    def predict_cov(self, x1, x2):
        """Return predict covariance

        Parameters
        ----------
        x1 : 2d-ndarray
            Input data 1
        x2 : 2d-ndarray
            Input data 2

        Returns
        -------
        2d-ndarray
            predict covariance
        """
        k1 = self.kern.K(x1, self.x)
        k2 = self.kern.K(self.x, x2)
        return self.kern.K(x1, x2) - np.matmul(
            k1, cho_solve((self.K_varI_L, False), k2)
        )

    def predict_f(self, x, full_var=False):
        """Return predict mean and variance of unobserved f

        Parameters
        ----------
        x : 2d-ndarray
            Input data
        full_var : bool, optional
            If true, return full covariance matrix, by default is False

        Returns
        -------
        (mean, variance)
            Predict mean and variance of f
        """
        k = self.kern.K(x, self.x)
        mean = np.matmul(k, cho_solve((self.K_varI_L, False), self.y))
        if full_var:
            var = self.kern.K(x, x) - np.matmul(
                k, cho_solve((self.K_varI_L, False), k.T)
            )
        else:
            var = self.kern.K(x, x, diag=True) - np.sum(
                k * cho_solve((self.K_varI_L, False), k.T).T, axis=1
            )
        return mean, var

    def predict_fvar(self, x, full_var=False):
        """Return predict variance of unobserved f

        Parameters
        ----------
        x : 2d-ndarray
            Input data
        full_var : bool, optional
            If true, return full covariance matrix, by default is False

        Returns
        -------
        1d or 2d-ndarary
            Predict mean and variance of f
        """
        k = self.kern.K(x, self.x)
        if full_var:
            return self.kern.K(x, x) - np.matmul(
                k, cho_solve((self.K_varI_L, False), k.T)
            )
        else:
            return self.kern.K(x, x, diag=True) - np.sum(
                k * cho_solve((self.K_varI_L, False), k.T).T, axis=1
            )

    def predict_mean(self, x):
        """Return predict mean

        Parameters
        ----------
        x : 2d-ndarray
            Input data

        Returns
        -------
        1d-ndarray
            Predict mean
        """
        k = self.kern.K(x, self.x)
        return np.matmul(k, cho_solve((self.K_varI_L, False), self.y))
