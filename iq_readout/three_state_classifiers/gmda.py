from __future__ import annotations
import warnings
from typing import Tuple

from copy import deepcopy
import numpy as np
from scipy.optimize import curve_fit

from ..classifiers import ThreeStateClassifier
from ..utils import check_2d_input, histogram_2d, reshape_histogram_2d, FIT_KARGS


def simple_2d_gaussian(
    z: np.ndarray,
    mu_x: float,
    mu_y: float,
    sigma: float,
) -> np.ndarray:
    """
    Probability density function of a 2D Gaussian with
    mean = (mu_x, mu_y) and covariance matrix = diag(sigma**2, sigma**2)

    Params
    ------
    z: np.array(..., 2)
        Points in the 2D space
    mu_x
        Mean of the first coordinate
    mu_y
        Mean of the second coordinate
    sigma
        Standard deviation of the two coordinates

    Returns
    -------
    probs: np.array(...)
        Values of the probability density function
    """
    check_2d_input(z)
    x, y = z[..., 0], z[..., 1]
    x_norm, y_norm = (x - mu_x) / sigma, (y - mu_y) / sigma
    prob = 1 / (2 * np.pi * sigma**2) * np.exp(-0.5 * (x_norm**2 + y_norm**2))
    return prob


def simple_2d_gaussian_triple_mixture(
    z: np.ndarray,
    mu_0_x: float,
    mu_0_y: float,
    mu_1_x: float,
    mu_1_y: float,
    mu_2_x: float,
    mu_2_y: float,
    sigma: float,
    angle1: float,
    angle2: float,
) -> np.ndarray:
    """
    Probability density function corresponding to the sum of three
    `simple_2d_gaussian`s

    Parameters
    ----------
    z: np.array(..., 2)
        Points in the 2D space
    mu_i_x
        Mean of the first coordinate for the i^th Gaussian
    mu_i_y
        Mean of the second coordinate for the i^th Gaussian
    sigma
        Standard deviation of the two coordinates for the i^th Gaussian
    angle1, angle2
        Weight of the 1st Gaussian is sin(angle2)**2 * cos(angle1)**2
        and of the 2nd Gaussian is sin(angle2)**2 * sin(angle1)**2
        and of the 3rd Gaussian is cos(angle2)**2
        to ensure that the PDF is normalized
    """
    check_2d_input(z)
    a_0, a_1, a_2 = (
        np.sin(angle1) ** 2 * np.cos(angle2) ** 2,
        np.sin(angle1) ** 2 * np.sin(angle2) ** 2,
        np.cos(angle1) ** 2,
    )

    prob_0 = simple_2d_gaussian(z, mu_x=mu_0_x, mu_y=mu_0_y, sigma=sigma)
    prob_1 = simple_2d_gaussian(z, mu_x=mu_1_x, mu_y=mu_1_y, sigma=sigma)
    prob_2 = simple_2d_gaussian(z, mu_x=mu_2_x, mu_y=mu_2_y, sigma=sigma)

    prob = a_0 * prob_0 + a_1 * prob_1 + a_2 * prob_2

    return prob


class GaussMixClassifier(ThreeStateClassifier):
    """
    Read `gmda.md` and `ThreeStateClassifier` documentation
    """

    _pdf_func_0 = simple_2d_gaussian_triple_mixture
    _pdf_func_1 = simple_2d_gaussian_triple_mixture
    _pdf_func_2 = simple_2d_gaussian_triple_mixture
    # parameter name ordering must match the ordering in the pdf functions
    names = [
        "mu_0_x",
        "mu_0_y",
        "mu_1_x",
        "mu_1_y",
        "mu_2_x",
        "mu_2_y",
        "sigma",
        "angle1",
        "angle2",
    ]
    _param_names = {
        0: names,
        1: names,
        2: names,
    }

    @property
    def statistics(self) -> Dict[str, np.ndarray]:
        """
        Returns dictionary with general statistical data:
        - mu_0: np.array([float, float])
        - mu_1: np.array([float, float])
        - mu_2: np.array([float, float])
        - cov_0: np.array([[float, float], [float, float]])
        - cov_1: np.array([[float, float], [float, float]])
        - cov_2: np.array([[float, float], [float, float]])
        It can also include other information.

        NB: this property is used for plotting and for storing useful
            information in the YAML file
        """
        return {}

    @classmethod
    def fit(
        cls: GaussMixClassifier,
        shots_0: np.ndarray,
        shots_1: np.ndarray,
        shots_2: np.ndarray,
        n_bins: list = [100, 100],
    ) -> GaussMixClassifier:
        """
        Fits the given data to extract the best parameters for classification.

        Parameters
        ----------
        shots_0: np.ndarray(N, 2)
            IQ data when preparing state 0
        shots_1: np.ndarray(M, 2)
            IQ data when preparing state 1
        shots_2: np.ndarray(P, 2)
            IQ data when preparing state 2
        n_bins: (nx_bins, ny_bins)
            Number of bins for the first and second coordinate
            used in the 2d histograms

        Returns
        -------
        `GaussMixClassifier` containing the fitted parameters
        """
        check_2d_input(shots_0, axis=1)
        check_2d_input(shots_1, axis=1)
        check_2d_input(shots_2, axis=1)

        # populate `params` during fitting
        params = {state: {} for state in range(3)}

        all_shots = np.concatenate([shots_0, shots_1, shots_2])
        counts, zz = reshape_histogram_2d(*histogram_2d(all_shots, n_bins=n_bins))

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

        popt_comb, pcov = curve_fit(
            cls._pdf_func_0,  # it is the same for all states
            zz,
            counts,
            p0=guess,
            bounds=bounds,
            **FIT_KARGS,
        )
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt_comb > 0.1).any():
            warnings.warn("Fitted means and covariances may not be accurate")

        mu_0, mu_1, mu_2 = popt_comb[:2], popt_comb[2:4], popt_comb[4:6]
        sigma = popt_comb[6]
        for s in range(3):
            params[s]["mu_0_x"], params[s]["mu_0_y"] = mu_0
            params[s]["mu_1_x"], params[s]["mu_1_y"] = mu_1
            params[s]["mu_2_x"], params[s]["mu_2_y"] = mu_2
            params[s]["sigma"] = sigma

        # get amplitudes of Gaussians for each state
        # Note: fitting in log scale improves the results, however there is the
        # problem of having counts=0 (np.log(0) = inf) due to undersampling
        bounds = ((0, 0), (np.pi / 2, np.pi / 2))

        # PDF state 0
        log_pdf = lambda z, angle1, angle2: np.log10(
            cls._pdf_func_0(z, *popt_comb[:-2], angle1, angle2)
        )
        guess = [0.1, np.pi / 2 - 0.25]  # avoid getting stuck in max bound
        counts, zz = reshape_histogram_2d(*histogram_2d(shots_0, n_bins=n_bins))
        zz, counts = zz[counts != 0], counts[counts != 0]
        popt, pcov = curve_fit(
            log_pdf, zz, np.log10(counts), p0=guess, bounds=bounds, **FIT_KARGS
        )
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted means and covariances may not be accurate")
        params[0]["angle1"], params[0]["angle2"] = popt

        # PDF state 1
        log_pdf = lambda z, angle1, angle2: np.log10(
            cls._pdf_func_1(z, *popt_comb[:-2], angle1, angle2)
        )
        guess = [1.4706, np.pi / 2 - 0.25]  # avoid getting stuck in max bound
        counts, zz = reshape_histogram_2d(*histogram_2d(shots_1, n_bins=n_bins))
        zz, counts = zz[counts != 0], counts[counts != 0]
        popt, pcov = curve_fit(
            log_pdf, zz, np.log10(counts), p0=guess, bounds=bounds, **FIT_KARGS
        )
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted means and covariances may not be accurate")
        params[1]["angle1"], params[1]["angle2"] = popt

        # PDF state 2
        log_pdf = lambda z, angle1, angle2: np.log10(
            cls._pdf_func_2(z, *popt_comb[:-2], angle1, angle2)
        )
        guess = [np.pi / 4, 0.2255]
        counts, zz = reshape_histogram_2d(*histogram_2d(shots_2, n_bins=n_bins))
        zz, counts = zz[counts != 0], counts[counts != 0]
        popt, pcov = curve_fit(
            log_pdf, zz, np.log10(counts), p0=guess, bounds=bounds, **FIT_KARGS
        )
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted means and covariances may not be accurate")
        params[2]["angle1"], params[2]["angle2"] = popt

        return cls(params)
