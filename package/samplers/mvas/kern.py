# mypy; ignore-errors
# flake8: noqa
import numpy as np


class Rbf:
    def __init__(self, xdim, lengthscale=1.0, outputscale=1.0):
        self.xdim = xdim
        self.lengthscale = lengthscale
        self.outputscale = outputscale

    def K(self, x1, x2, diag=False):
        """calculate kernel matrix

        Parameters
        ----------
        x1 : 2d-ndarray
            Input data 1
        x2 : 2d-ndarray
            Input data 2
        diag : bool, optional
            Caluculate only diagonal elements of kernel matrix

        Returns
        -------
        1d or 2d-ndarray
            Kernel matrix(or its diagonal elements)
        """
        if diag:
            return self.outputscale * np.exp(
                -np.sum((x1 - x2) ** 2, axis=1) / (2 * self.lengthscale**2)
            )
        else:
            return self.outputscale * np.exp(
                -np.sum(
                    (x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=2
                )
                / (2 * self.lengthscale**2)
            )


class Matern32:
    def __init__(self, xdim, lengthscale=1.0, outputscale=1.0):
        self.xdim = xdim
        self.lengthscale = lengthscale
        self.outputscale = outputscale

    def K(self, x1, x2, diag=False):
        """calculate kernel matrix

        Parameters
        ----------
        x1 : 2d-ndarray
            Input data 1
        x2 : 2d-ndarray
            Input data 2
        diag : bool, optional
            Caluculate only diagonal elements of kernel matrix

        Returns
        -------
        1d or 2d-ndarray
            Kernel matrix(or its diagonal elements)
        """
        if diag:
            dist = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
            return (
                self.outputscale
                * (1 + np.sqrt(3) * dist / self.lengthscale)
                * np.exp(-np.sqrt(3) * dist / self.lengthscale)
            )
        else:
            dist = np.sqrt(
                np.sum(
                    (x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=2
                )
            )
            return (
                self.outputscale
                * (1 + np.sqrt(3) * dist / self.lengthscale)
                * np.exp(-np.sqrt(3) * dist / self.lengthscale)
            )


class Matern52:
    def __init__(self, xdim, lengthscale=1.0, outputscale=1.0):
        self.xdim = xdim
        self.lengthscale = lengthscale
        self.outputscale = outputscale

    def K(self, x1, x2, diag=False):
        """calculate kernel matrix

        Parameters
        ----------
        x1 : 2d-ndarray
            Input data 1
        x2 : 2d-ndarray
            Input data 2
        diag : bool, optional
            Caluculate only diagonal elements of kernel matrix

        Returns
        -------
        1d or 2d-ndarray
            Kernel matrix(or its diagonal elements)
        """
        if diag:
            dist = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
            return (
                self.outputscale
                * (
                    1
                    + np.sqrt(5) * dist / self.lengthscale
                    + 5 * dist**2 / (3 * self.lengthscale**2)
                )
                * np.exp(-np.sqrt(5) * dist / self.lengthscale)
            )
        else:
            dist = np.sqrt(
                np.sum(
                    (x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=2
                )
            )
            return (
                self.outputscale
                * (
                    1
                    + np.sqrt(5) * dist / self.lengthscale
                    + 5 * dist**2 / (3 * self.lengthscale**2)
                )
                * np.exp(-np.sqrt(5) * dist / self.lengthscale)
            )


class Ard_se:
    def __init__(self, xdim, outputscale=1.0):
        self.xdim = xdim
        self.lengthscale = np.empty([xdim])
        self.outputscale = outputscale

    def K(self, x1, x2, diag=False):
        """calculate kernel matrix

        Parameters
        ----------
        x1 : 2d-ndarray
            Input data 1
        x2 : 2d-ndarray
            Input data 2
        diag : bool, optional
            Caluculate only diagonal elements of kernel matrix

        Returns
        -------
        1d or 2d-ndarray
            Kernel matrix(or its diagonal elements)
        """
        if diag:
            return self.outputscale * np.exp(
                -np.sum((x1 - x2) ** 2 / (2 * self.lengthscale**2), axis=1)
            )
        else:
            return self.outputscale * np.exp(
                -np.sum(
                    (x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2
                    / (2 * self.lengthscale[np.newaxis, np.newaxis, :] ** 2),
                    axis=2,
                )
            )
