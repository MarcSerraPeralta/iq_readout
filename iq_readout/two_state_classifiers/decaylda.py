from __future__ import annotations
import warnings

from copy import deepcopy
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

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


def decay_amplitude_1d_pdf(
    x: np.ndarray,
    mu_0: float,
    mu_1: float,
    sigma: float,
    angle: float,
    t1_norm: float,
):
    """
    See `decaylda.md`

    Params
    ------
    x
        Points in the 1D space
    mu_i
        Mean for the Gaussian noise for state `i`
    sigma
        Standard deviation of the Gaussian noise for state `i`
    t1_norm
        T1 normalized with respect to the measurement duration

    Returns
    -------
    z
        Values of the probability density function
    """
    a1, a2 = np.sin(angle) ** 2, np.cos(angle) ** 2
    C = np.sign(mu_1 - mu_0) * (mu_0 - x) / (np.sqrt(2) * sigma) + sigma / (
        np.sqrt(2) * np.abs(mu_1 - mu_0) * t1_norm
    )
    P = 0.5 * (mu_1 - mu_0) ** 2 / sigma**2
    z_0 = np.exp(-0.5 * (x - mu_1) ** 2 / sigma**2) / np.sqrt(2 * np.pi * sigma**2)
    z_1 = (
        np.exp(-0.5 * (x - mu_0) ** 2 / sigma**2 + C**2)
        / (np.sqrt(2 * np.pi * sigma**2) * t1_norm)
        * np.sqrt(np.pi / (4 * P))
        * (erf(C + np.sqrt(P)) - erf(C))
        / (1 - np.exp(-1 / t1_norm))
    )

    z = a1 * z_0 + a2 * z_1

    return z


class DecayLinearClassifierFit:
    """
    Read `gmlda.md`
    """

    def __init__(self):
        self._pdf_function_proj_0 = simple_1d_gaussian_double_mixture
        self._pdf_function_proj_1 = decay_amplitude_1d_pdf
        self._param_names_0 = [
            "mu_0",
            "mu_1",
            "sigma",
            "angle",  # this is not the rotation angle!
        ]
        self._param_names_1 = [
            "mu_0",
            "mu_1",
            "sigma",
            "angle",  # this is not the rotation angle!
            "t1_norm",
        ]
        self._params_0 = None
        self._params_1 = None
        self.rot_angle = None  # this is the rotation angle
        self.threshold = None
        self.rot_shift = None

        return

    def fit(
        self,
        shots_0: np.ndarray,
        shots_1: np.ndarray,
        n_bins: int = 100,
    ) -> DecayLinearClassifierFit:
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
        `DecayLinearClassifierFit` containing the fitted parameters
        """
        check_2d_input(shots_0, axis=1)
        check_2d_input(shots_1, axis=1)
        if not isinstance(n_bins, int):
            raise ValueError(f"n_bins must be int, but {type(n_bins)} given")

        # the decay does not affect the direction
        # of \vec{mu0} - \vec{mu1}
        # Using \vec{mu1} - \vec{mu0} to have the projected 0 blob
        # on the left of the 1 blob
        mu_0, mu_1 = np.average(shots_0, axis=0), np.average(shots_1, axis=0)
        self.rot_angle = get_angle(mu_1 - mu_0)
        self.rot_shift = rotate_data([mu_0], -self.rot_angle)[0, 1]

        # rotate and project data
        shots_0_1d, shots_1_1d = self.project(shots_0), self.project(shots_1)

        # get fit for state=0
        # Note: fitting in log scale improves the results, however there is the
        # problem of having counts=0 (np.log(0) = inf) due to undersampling
        counts, x = np.histogram(shots_0_1d, bins=n_bins, density=True)
        x = 0.5 * (x[1:] + x[:-1])
        x, counts = x[counts != 0], counts[counts != 0]

        bounds = (
            (np.min(shots_0_1d), np.min(shots_0_1d), 1e-10, 0),
            (np.max(shots_0_1d), np.max(shots_0_1d), np.max(shots_0_1d), np.pi / 2),
        )
        guess = (
            np.average(shots_0_1d),
            np.average(shots_1_1d),
            np.std(shots_0_1d),
            np.pi / 2 - 0.25,
        )

        popt, pcov = curve_fit(
            lambda x, *p: np.log10(self._pdf_function_proj_0(x, *p)),
            x,
            np.log10(counts),
            p0=guess,
            bounds=bounds,
            loss="soft_l1",
        )  # loss="soft_l1" leads to more stable fits
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted mean and covariance of state=0 may not be accurate")
        self._params_0 = deepcopy(popt)

        # get fit for state=1
        # Note: fitting in log scale improves the results, however there is the
        # problem of having counts=0 (np.log(0) = inf) due to undersampling
        counts, x = np.histogram(shots_1_1d, bins=n_bins, density=True)
        x = 0.5 * (x[1:] + x[:-1])
        x, counts = x[counts != 0], counts[counts != 0]

        bounds = (
            (np.min(shots_1_1d), 1e-10, 0, 1e-4),
            (
                np.max(shots_1_1d),
                np.max(shots_1_1d),
                np.pi / 2,
                100,
            ),
        )
        guess = (
            np.average(shots_1_1d),
            np.std(shots_0_1d),
            0.2255,
            1 / 20,
        )

        popt, pcov = curve_fit(
            lambda x, *p: np.log10(self._pdf_function_proj_1(x, self._params_0[0], *p)),
            x,
            np.log10(counts),
            p0=guess,
            bounds=bounds,
            loss="soft_l1",
        )  # loss="soft_l1" leads to more stable fits
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted mean and covariance of state=1 may not be accurate")
        self._params_1 = np.concatenate([[self._params_0[0]], deepcopy(popt)])

        self.threshold = np.inf

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
            "rot_shift": float,
        }
        """
        self._check_params()

        params = {
            0: {k: v for k, v in zip(self._param_names_0, self._params_0)},
            1: {k: v for k, v in zip(self._param_names_1, self._params_1)},
            "rot_angle": self.rot_angle,
            "threshold": self.threshold,
            "rot_shift": self.rot_shift,
        }

        return params

    def load(self, params: dict) -> DecayLinearClassifierFit:
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
                "rot_shift": float,
            }

        Returns
        -------
        DecayLinearClassifierFit
        """
        if set(params) != set([0, 1, "rot_angle", "threshold", "rot_shift"]):
            raise ValueError(
                "params must have keys: [0, 1, 'rot_angle', 'threshold', 'rot_shift']"
            )

        self._params_0 = np.array([params[0][k] for k in self._param_names_0])
        self._params_1 = np.array([params[1][k] for k in self._param_names_1])
        self.rot_angle = params["rot_angle"]
        self.threshold = params["threshold"]
        self.rot_shift = params["rot_shift"]

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
        return rotate_data(x, -self.rot_angle)[..., 0]

    def pdf_0_projected(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the probability density function of state 0
        for the projection of the given 1D values.
        Note that p(x1,x2|0) != p(x_projected|0).

        Parameters
        ----------
        x: np.ndarray(...)

        Returns
        -------
        np.ndarray(...)
        """
        self._check_params()
        return self._pdf_function_proj_0(x, *self._params_0)

    def pdf_1_projected(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the probability density function of state 1
        for the projection of the given 1D values.
        Note that p(x1,x2|1) != p(x_projected|1).

        Parameters
        ----------
        x: np.ndarray(...)

        Returns
        -------
        np.ndarray(...)
        """
        self._check_params()
        return self._pdf_function_proj_1(x, *self._params_1)

    def predict(self, x: np.ndarray, p0: float = 1 / 2) -> np.ndarray:
        """
        Returns the classes (0 or 1) for the specified 2D values.

        Parameters
        ----------
        x: np.ndarray(..., 2)
        p0
            Probability of the qubit's state being 0 just before the measurement

        Returns
        -------
        np.ndarray(...)
        """
        if (p0 > 1) or (p0 < 0):
            raise ValueError(
                "The speficied 'p0' must be a physical probability, "
                f"but p0={p0} (and p2={1-p0}) were given"
            )
        z = self.project(x)
        probs = [self.pdf_0_projected(z) * p0, self.pdf_1_projected(z) * (1 - p0)]
        return np.argmax(probs, axis=0)

    def _check_params(self):
        if (
            (self._params_0 is None)
            or (self._params_1 is None)
            or (self.rot_angle is None)
            or (self.threshold is None)
            or (self.rot_shift is None)
        ):
            raise ValueError(
                "Model does not have the fitted params, "
                "please run the fit ('.fit') or load the params ('.load')"
            )

        if len(self._params_0) != len(self._param_names_0):
            raise ValueError(
                f"0-state parameters must correspond to {self._param_names_0}, "
                f"but {self._params_0} were given"
            )
        if len(self._params_1) != len(self._param_names_1):
            raise ValueError(
                f"1-state parameters must correspond to {self._param_names_1}, "
                f"but {self._params_1} were given"
            )
        if not isinstance(self.rot_angle, float):
            raise ValueError(
                "rotation angle must be a float, "
                f"but {type(self.rot_angle)} was given"
            )
        if not isinstance(self.threshold, float):
            raise ValueError(
                "threshold must be a float, " f"but {type(self.threshold)} was given"
            )
        if not isinstance(self.rot_shift, float):
            raise ValueError(
                "rotated height of the means must be a float, "
                f"but {type(self.rot_shift)} was given"
            )
        return