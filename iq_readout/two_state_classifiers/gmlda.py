from __future__ import annotations
import warnings

from copy import deepcopy
import numpy as np
from scipy.optimize import curve_fit

from ..utils import check_2d_input, rotate_data, get_angle


def simple_1d_gaussian(
    x: np.ndarray,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """
    Probability density function of a 1D Gaussian with
    mean = mu and standard deviation = sigma

    Params
    ------
    x
        Points in the 1D space
    mu
        Mean of the first coordinate
    sigma
        Standard deviation of the two coordinates

    Returns
    -------
    z
        Values of the probability density function
    """
    z = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * (x - mu) ** 2 / sigma**2)
    return z


def simple_1d_gaussian_double_mixture(
    x: np.ndarray,
    mu_1: float,
    mu_2: float,
    sigma: float,
    angle: float,
) -> np.ndarray:
    """
    Probability density function corresponding to the sum of two
    `simple_1d_gaussian`s

    Parameters
    ----------
    x
        Points in the 1D space
    mu_i
        Mean for the i^th Gaussian
    sigma
        Standard deviation of the two coordinates for both Gaussians
    angle
        Weight of the 1st Gaussian is sin(angle)**2 and
        of the 2nd Gaussian is cos(angle)**2 to ensure that
        the PDF is normalized
    """
    a1, a2 = np.sin(angle) ** 2, np.cos(angle) ** 2

    z = a1 * simple_1d_gaussian(x, mu=mu_1, sigma=sigma) + a2 * simple_1d_gaussian(
        x, mu=mu_2, sigma=sigma
    )

    return z


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


def simple_2d_gaussian_double_mixture(
    x: np.ndarray,
    mu0_1: float,
    mu1_1: float,
    mu0_2: float,
    mu1_2: float,
    sigma: float,
    angle: float,
) -> np.ndarray:
    """
    Probability density function corresponding to the sum of two
    `simple_2d_gaussian`s

    Parameters
    ----------
    x
        Points in the 2D space
    mu0_i
        Mean of the first coordinate for the i^th Gaussian
    mu1_i
        Mean of the second coordinate for the i^th Gaussian
    sigma_i
        Standard deviation of the two coordinates for the i^th Gaussian
    angle
        Weight of the 1st Gaussian is sin(angle)**2 and
        of the 2nd Gaussian is cos(angle)**2 to ensure that
        the PDF is normalized
    """
    check_2d_input(x)
    a1, a2 = np.sin(angle) ** 2, np.cos(angle) ** 2

    z = a1 * simple_2d_gaussian(
        x, mu0=mu0_1, mu1=mu1_1, sigma=sigma
    ) + a2 * simple_2d_gaussian(x, mu0=mu0_2, mu1=mu1_2, sigma=sigma)

    return z


class TwoStateLinearClassifierFit:
    """
    Read `gmlda.md`
    """

    def __init__(self):
        self._pdf_function_proj = simple_1d_gaussian_double_mixture
        self._pdf_function = simple_2d_gaussian_double_mixture
        self._param_names = [
            "mu_0",
            "mu_1",
            "sigma",
            "angle",  # this is not the rotation angle!
        ]
        self._params_0 = None
        self._params_1 = None
        self.rot_angle = None  # this is the rotation angle
        self.threshold = None

        return

    def fit(
        self,
        shots_0: np.ndarray,
        shots_1: np.ndarray,
        n_bins: int = 100,
    ) -> TwoStateLinearClassifierFit:
        """
        Fits the given data to extract the best parameters for classification.

        Parameters
        ----------
        shots_0: np.ndarray(N, 2)
            N points corresponding to class 0
        shots_1: np.ndarray(M, 2)
            M points corresponding to class 1
        n_bins:
            Number of bins for the 1d histograms

        Returns
        -------
        `TwoStateLinearClassifierFit` containing the fitted parameters
        """
        check_2d_input(shots_0, axis=1)
        check_2d_input(shots_1, axis=1)
        if not isinstance(n_bins, int):
            raise ValueError(f"n_bins must be int, but {type(n_bins)} given")

        # the mixture of 2 Gaussians does not affect the direction
        # of \vec{mu0} - \vec{mu1}
        # Using \vec{mu1} - \vec{mu0} to have the projected 0 blob
        # on the left of the 1 blob
        mu_0, mu_1 = np.average(shots_0, axis=0), np.average(shots_1, axis=0)
        self.rot_angle = get_angle(mu_1 - mu_0)

        # rotate and project data
        shots_0_1d, shots_1_1d = self.project(shots_0), self.project(shots_1)

        # get means and standard deviations
        all_shots = np.concatenate([shots_0_1d, shots_1_1d])
        counts, x = np.histogram(all_shots, bins=n_bins, density=True)
        x = 0.5 * (x[1:] + x[:-1])

        bounds = (
            (
                np.min(shots_0_1d),
                np.min(shots_1_1d),
                1e-10,
                0,
            ),
            (
                np.max(shots_0_1d),
                np.max(shots_1_1d),
                np.max(all_shots),
                np.pi / 2,
            ),
        )
        guess = (
            np.average(shots_0_1d),
            np.average(shots_1_1d),
            np.std(shots_0_1d),
            np.pi / 4,
        )

        popt, pcov = curve_fit(
            self._pdf_function_proj, x, counts, p0=guess, bounds=bounds, loss="soft_l1"
        )  # loss="soft_l1" leads to more stable fits
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted means and covariances may not be accurate")
        self._params_0, self._params_1 = deepcopy(popt), deepcopy(popt)
        self.threshold = 0.5 * (popt[0] + popt[1])

        # get amplitudes of Gaussians for each state
        # PDF state 0
        pdf = lambda x, angle: self._pdf_function_proj(x, *self._params_0[:-1], angle)
        guess = [np.pi / 2 - 0.01]  # avoid getting stuck in max bound
        counts, x = np.histogram(shots_0_1d, bins=n_bins, density=True)
        x = 0.5 * (x[1:] + x[:-1])
        popt, pcov = curve_fit(
            pdf, x, counts, p0=guess, bounds=(0, np.pi / 2), loss="soft_l1"
        )  # loss="soft_l1" leads to more stable fits
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fit for state=0 may not be accurate")
        self._params_0[-1] = popt

        # PDF state 1
        pdf = lambda x, angle: self._pdf_function_proj(x, *self._params_1[:-1], angle)
        guess = [0.2255]
        counts, x = np.histogram(shots_1_1d, bins=n_bins, density=True)
        x = 0.5 * (x[1:] + x[:-1])
        popt, pcov = curve_fit(
            pdf, x, counts, p0=guess, bounds=(0, np.pi / 2), loss="soft_l1"
        )  # loss="soft_l1" leads to more stable fits
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fit for state=1 may not be accurate")
        self._params_1[-1] = popt

        return self

    def params(self) -> dict:
        """
        Returns the fitted params. The output can be used to load
        a new class using the '.load' function.

        Returns
        -------
        Dictionary with the following structure:
        {
            0: {"mu_0": float, "mu_1": float, "sigma": float, "angle": float},
            1: {"mu_0": float, "mu_1": float, "sigma": float, "angle": float},
            "rot_angle": float,
            "threshold": float,
        }
        """
        self._check_params()

        params = {
            0: {k: v for k, v in zip(self._param_names, self._params_0)},
            1: {k: v for k, v in zip(self._param_names, self._params_1)},
            "rot_angle": self.rot_angle,
            "threshold": self.threshold,
        }

        return params

    def load(self, params: dict) -> TwoStateLinearClassifierFit:
        """
        Load the parameters for the PDFs.

        Parameters
        ----------
        params
            Dictionary with the following structure:
            {
                0: {"mu_0": float, "mu_1": float, "sigma": float, "angle": float},
                1: {"mu_0": float, "mu_1": float, "sigma": float, "angle": float},
                "rot_angle": float,
                "threshold": float,
            }

        Returns
        -------
        TwoStateLinearClassifierFit
        """
        if set(params) != set([0, 1, "rot_angle", "threshold"]):
            raise ValueError("params must have keys: [0, 1, 'rot_angle', 'threshold']")

        self._params_0 = np.array([params[0][k] for k in self._param_names])
        self._params_1 = np.array([params[1][k] for k in self._param_names])
        self.rot_angle = params["rot_angle"]
        self.threshold = params["threshold"]

        self._check_params()

        return self

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project the data in the 01 axis.

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        check_2d_input(x)
        if self.rot_angle is None:
            self._check_params()
        return rotate_data(x, -self.rot_angle)[:, 0]

    def pdf_0_projected(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the probability density function of state 0
        for the projection of the given 2D values.
        Note that p(x1,x2|0) != p(x_projected|0).

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        self._check_params()
        z = self.project(x)
        return self._pdf_function_proj(z, *self._params_0)

    def pdf_1_projected(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the probability density function of state 1
        for the projection of the given 2D values.
        Note that p(x1,x2|1) != p(x_projected|1).

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        self._check_params()
        z = self.project(x)
        return self._pdf_function_proj(z, *self._params_1)

    def pdf_0(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the probability density function of state 0
        for the given 2D values.
        Note that p(x1,x2|0) != p(x_projected|0).

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        self._check_params()
        check_2d_input(x)
        mu_0 = rotate_data([[self._params_0[0], 0]], self.rot_angle)[0]
        mu_1 = rotate_data([[self._params_0[1], 0]], self.rot_angle)[0]
        params = [*mu_0, *mu_1, *self._params_0[-2:]]
        return self._pdf_function(x, *params)

    def pdf_1(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the probability density function of state 1
        for the given 2D values.
        Note that p(x1,x2|1) != p(x_projected|1).

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        self._check_params()
        check_2d_input(x)
        mu_0 = rotate_data([[self._params_1[0], 0]], self.rot_angle)[0]
        mu_1 = rotate_data([[self._params_1[1], 0]], self.rot_angle)[0]
        params = [*mu_0, *mu_1, *self._params_1[-2:]]
        return self._pdf_function(x, *params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the classes (0 or 1) for the specified 2D values.

        Parameters
        ----------
        x: np.ndarray(..., 2)

        Returns
        -------
        np.ndarray(...)
        """
        probs = [self.pdf_0(x), self.pdf_1(x)]
        return np.argmax(probs, axis=0)

    def _check_params(self):
        if (
            (self._params_0 is None)
            or (self._params_1 is None)
            or (self.rot_angle is None)
            or (self.threshold is None)
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
        if not isinstance(self.rot_angle, float):
            raise ValueError(
                "rotation angle must be a float, "
                f"but {type(self.rot_angle)} was given"
            )
        if not isinstance(self.threshold, float):
            raise ValueError(
                "rotation angle must be a float, "
                f"but {type(self.threshold)} was given"
            )
        return
