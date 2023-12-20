from __future__ import annotations
import warnings
from typing import Tuple

from copy import deepcopy
import numpy as np
from scipy.optimize import curve_fit

from ..utils import check_2d_input


def simple_2d_gaussian(
    x: np.ndarray,
    mu0: float,
    mu1: float,
    sigma: float,
) -> np.ndarray:
    """
    Probability density function of a 2D Gaussian with
    mean = (mu0, mu1) and covariance matrix = diag(sigma**2, sigma**2)

    Params
    ------
    x
        Points in the 2D space
    mu0
        Mean of the first coordinate
    mu1
        Mean of the second coordinate
    sigma
        Standard deviation of the two coordinates

    Returns
    -------
    z
        Values of the probability density function
    """
    check_2d_input(x)
    x0, x1 = x[..., 0], x[..., 1]
    x0_, x1_ = (x0 - mu0) / sigma, (x1 - mu1) / sigma
    z = 1 / (2 * np.pi * sigma**2) * np.exp(-0.5 * (x0_**2 + x1_**2))
    return z


def simple_2d_gaussian_triple_mixture(
    x: np.ndarray,
    mu0_1: float,
    mu1_1: float,
    mu0_2: float,
    mu1_2: float,
    mu0_3: float,
    mu1_3: float,
    sigma: float,
    angle1: float,
    angle2: float,
) -> np.ndarray:
    """
    Probability density function corresponding to the sum of three
    `simple_2d_gaussian`s

    Parameters
    ----------
    x
        Points in the 2D space
    mu0_i
        Mean of the first coordinate for the i^th Gaussian
    mu1_i
        Mean of the second coordinate for the i^th Gaussian
    sigma
        Standard deviation of the two coordinates for the i^th Gaussian
    angle1, angle2
        Weight of the 1st Gaussian is sin(angle2)**2 * cos(angle1)**2
        and of the 2nd Gaussian is sin(angle2)**2 * sin(angle1)**2
        and of the 3rd Gaussian is cos(angle2)**2
        to ensure that the PDF is normalized
    """
    check_2d_input(x)
    a1, a2, a3 = (
        np.sin(angle1) ** 2 * np.cos(angle2) ** 2,
        np.sin(angle1) ** 2 * np.sin(angle2) ** 2,
        np.cos(angle1) ** 2,
    )

    z = (
        a1 * simple_2d_gaussian(x, mu0=mu0_1, mu1=mu1_1, sigma=sigma)
        + a2 * simple_2d_gaussian(x, mu0=mu0_2, mu1=mu1_2, sigma=sigma)
        + a3 * simple_2d_gaussian(x, mu0=mu0_3, mu1=mu1_3, sigma=sigma)
    )

    return z


def histogram_2d(
    x: np.ndarray,
    n_bins: Tuple[int, int] = [100, 100],
    density: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs a 2d histogram and returns the
    counts, x0, and x1 data flattened.

    Parameters
    ----------
    x
        Points in the 2D space
    n_bins
        List of two elements corresponding to the
        number of bins for the first and second coordinate
    density
        If True, returns the probability density function values

    Returns
    -------
    counts
        Counts or PDF with shape=n_bins
    xx0_centers
        Centers of the bins for the first coordinate, with shape=n_bins
    xx1_centers
        Centers of the bins for the second coordinate, with shape=n_bins
    """
    check_2d_input(x, axis=1)
    x0, x1 = x[:, 0], x[:, 1]
    counts, x0_edges, x1_edges = np.histogram2d(x0, x1, bins=n_bins, density=density)
    x0_centers = 0.5 * (x0_edges[:-1] + x0_edges[1:])
    x1_centers = 0.5 * (x1_edges[:-1] + x1_edges[1:])
    # using indexing="ij" so that xx0_centers and xx1_centers have
    # the same shape as counts, which follows (nx0_bins, nx1_bins)
    xx0_centers, xx1_centers = np.meshgrid(x0_centers, x1_centers, indexing="ij")
    return counts, xx0_centers, xx1_centers


class ThreeStateClassifier2D:
    """
    Read `gmda.md`
    """

    def __init__(self):
        self._pdf_function = simple_2d_gaussian_triple_mixture
        self._param_names = [
            "mu0_1",
            "mu1_1",
            "mu0_2",
            "mu1_2",
            "mu0_3",
            "mu1_3",
            "sigma",
            "angle1",
            "angle2",
        ]
        self._params_0 = np.zeros(len(self._param_names))
        self._params_1 = np.zeros(len(self._param_names))
        self._params_2 = np.zeros(len(self._param_names))

        return

    def fit(
        self,
        shots_0: np.ndarray,
        shots_1: np.ndarray,
        shots_2: np.ndarray,
        n_bins: list = [100, 100],
        **kargs,
    ) -> ThreeStateClassifier2D:
        """
        Fits the given data to extract the best parameters for classification.

        Parameters
        ----------
        shots_0: np.ndarray(N, 2)
            N points corresponding to class 0
        shots_1: np.ndarray(M, 2)
            M points corresponding to class 1
        shots_2: np.ndarray(P, 2)
            P points corresponding to class 2
        n_bins:
            List of two elements corresponding to the
            number of bins for the first and second coordinate
            used in the 2d histograms
        kargs
            Extra arguments for scipy.optimize.curve_fit

        Returns
        -------
        `ThreeStateClassifier2D` containing the fitted parameters
        """
        check_2d_input(shots_0, axis=1)
        check_2d_input(shots_1, axis=1)
        check_2d_input(shots_2, axis=1)

        # loss="soft_l1" leads to more stable fits
        fit_kargs = {"loss": "soft_l1"}
        fit_kargs.update(kargs)

        all_shots = np.concatenate([shots_0, shots_1, shots_2])
        counts, xx = self._flatten_hist(*histogram_2d(all_shots, n_bins=n_bins))

        # in the first fit the shots_i are concatenated
        # to extract the means and covariance matrices,
        # thus the Gaussian weights are approx. 1/3.
        guess = [
            *np.average(shots_0, axis=0),
            *np.average(shots_1, axis=0),
            *np.average(shots_2, axis=0),
            np.average(np.std(shots_0, axis=0)),
            0.7854,
            0.9553,
        ]

        bounds = (
            (
                *np.min(shots_0, axis=0),
                *np.min(shots_1, axis=0),
                *np.min(shots_2, axis=0),
                1e-10,
                0,
                0,
            ),
            (
                *np.max(shots_0, axis=0),
                *np.max(shots_1, axis=0),
                *np.max(shots_2, axis=0),
                np.max(all_shots),
                np.pi / 2,
                np.pi / 2,
            ),
        )

        popt, pcov = curve_fit(
            self._pdf_function, xx, counts, p0=guess, bounds=bounds, **fit_kargs
        )
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted means and covariances may not be accurate")

        self._params_0 = deepcopy(popt)
        self._params_1 = deepcopy(popt)
        self._params_2 = deepcopy(popt)

        bounds = ((0, 0), (np.pi / 2, np.pi / 2))

        # PDF state 0
        pdf = lambda x, angle1, angle2: self._pdf_function(
            x, *self._params_0[:-2], angle1, angle2
        )
        guess = [0.1, np.pi / 2 - 0.1]  # avoid getting stuck in max bound
        counts, xx = self._flatten_hist(*histogram_2d(shots_0, n_bins=n_bins))
        popt, pcov = curve_fit(pdf, xx, counts, p0=guess, bounds=bounds, **fit_kargs)
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted means and covariances may not be accurate")
        self._params_0[-2:] = popt

        # PDF state 1
        pdf = lambda x, angle1, angle2: self._pdf_function(
            x, *self._params_1[:-2], angle1, angle2
        )
        guess = [1.4706, np.pi / 2 - 0.1]  # avoid getting stuck in max bound
        counts, xx = self._flatten_hist(*histogram_2d(shots_1, n_bins=n_bins))
        popt, pcov = curve_fit(pdf, xx, counts, p0=guess, bounds=bounds, **fit_kargs)
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted means and covariances may not be accurate")
        self._params_1[-2:] = popt

        # PDF state 2
        pdf = lambda x, angle1, angle2: self._pdf_function(
            x, *self._params_2[:-2], angle1, angle2
        )
        guess = [np.pi / 4, 0.2255]
        counts, xx = self._flatten_hist(*histogram_2d(shots_2, n_bins=n_bins))
        popt, pcov = curve_fit(pdf, xx, counts, p0=guess, bounds=bounds, **fit_kargs)
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted means and covariances may not be accurate")
        self._params_2[-2:] = popt

        return self

    def params(self) -> dict:
        """
        Returns the fitted params. The output can be used to load
        a new class using the '.load' function.

        Returns
        -------
        Dictionary with the following structure:
        {
            0: {"mu0_1": float, "mu1_1": float, "mu0_2": float, "mu1_2": float,
                "mu0_3": float, "mu1_3": float, "sigma": float, "angle1": float,
                "angle2": float},
            1: {"mu0_1": float, "mu1_1": float, "mu0_2": float, "mu1_2": float,
                "mu0_3": float, "mu1_3": float, "sigma": float, "angle1": float,
                "angle2": float},
            2: {"mu0_1": float, "mu1_1": float, "mu0_2": float, "mu1_2": float,
                "mu0_3": float, "mu1_3": float, "sigma": float, "angle1": float,
                "angle2": float},
        }
        """
        self._check_params()

        params = {
            0: {k: v for k, v in zip(self._param_names, self._params_0)},
            1: {k: v for k, v in zip(self._param_names, self._params_1)},
            2: {k: v for k, v in zip(self._param_names, self._params_2)},
        }

        return params

    def load(self, params: dict) -> ThreeStateClassifier2D:
        """
        Load the parameters for the PDFs.

        Returns
        -------
        `ThreeStateClassifier2D` class with the loaded params
        """
        if set(params) != set([0, 1, 2]):
            raise ValueError("params must have keys: [0, 1, 2]")

        self._params_0 = np.array([params[0][k] for k in self._param_names])
        self._params_1 = np.array([params[1][k] for k in self._param_names])
        self._params_2 = np.array([params[2][k] for k in self._param_names])

        self._check_params()

        return self

    def pdf_0(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the probability density function of state 0
        for the given 2D values.

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        self._check_params()
        check_2d_input(x)
        return self._pdf_function(x, *self._params_0)

    def pdf_1(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the probability density function of state 1
        for the given 2D values.

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        self._check_params()
        check_2d_input(x)
        return self._pdf_function(x, *self._params_1)

    def pdf_2(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the probability density function of state 2
        for the given 2D values.

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        self._check_params()
        check_2d_input(x)
        return self._pdf_function(x, *self._params_2)

    def predict(self, x) -> np.ndarray:
        """
        Returns the classes (0, 1 or 2) for the specified 2D values.

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        probs = [self.pdf_0(x), self.pdf_1(x), self.pdf_2(x)]
        return np.argmax(probs, axis=0)

    def _flatten_hist(
        self, counts: np.ndarray, xx0: np.ndarray, xx1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flattens the output of the `histogram_2d` to
        a single input with shape=(Nx*Ny, 2) and
        counts with shape=(Nx*Ny).

        Parameters
        ----------
        counts: np.ndarray(nx_bins, ny_bins)
        xx0: np.ndarray(nx_bins, ny_bins)
        xx1: np.ndarray(nx_bins, ny_bins)

        Returns
        -------
        counts: np.ndarray(nx_bins * ny_bin)
        xx: np.ndarray(nx_bins * ny_bin, 2)
        """
        counts = counts.reshape(-1)
        xx0, xx1 = xx0.reshape(-1, 1), xx1.reshape(-1, 1)
        xx = np.concatenate([xx0, xx1], axis=1)
        return counts, xx

    def _check_params(self):
        if (
            (self._params_0 is None)
            or (self._params_1 is None)
            or (self._params_2 is None)
        ):
            raise ValueError(
                "Model does not have the fitted params, "
                "please run the fit ('.fit') or load the params ('.load')"
            )

        if len(self._params_0) != len(self._param_names):
            raise ValueError(
                f"0-state parameters must correspond to {self._param_names}, "
                f"but {self._params_0} were given"
            )
        if len(self._params_1) != len(self._param_names):
            raise ValueError(
                f"1-state parameters must correspond to {self._param_names}, "
                f"but {self._params_1} were given"
            )
        if len(self._params_2) != len(self._param_names):
            raise ValueError(
                f"1-state parameters must correspond to {self._param_names}, "
                f"but {self._params_1} were given"
            )
        return
