from __future__ import annotations
from typing import Dict
import warnings

from copy import deepcopy
import numpy as np
from scipy.optimize import curve_fit

from ..classifiers import TwoStateLinearClassifier
from ..utils import check_2d_input, rotate_data, get_angle, FIT_KARGS

from ..pdfs import (
    simple_2d_gaussian_double_mixture,
    decay_amplitude_2d_pdf,
    simple_1d_gaussian_double_mixture,
    decay_amplitude_1d_pdf,
)


class DecayLinearClassifier(TwoStateLinearClassifier):
    """
    Read `gmlda.md` and `TwoStateLinearClassifier` documentation
    """

    _pdf_func_0 = simple_2d_gaussian_double_mixture
    _pdf_func_1 = decay_amplitude_2d_pdf
    # parameter name ordering must match the ordering in the pdf functions
    _param_names = {
        0: ["mu_0_x", "mu_0_y", "mu_1_x", "mu_1_y", "sigma", "angle"],
        1: ["mu_0_x", "mu_0_y", "mu_1_x", "mu_1_y", "sigma", "angle", "t1_norm"],
    }
    _pdf_func_0_proj = simple_1d_gaussian_double_mixture
    _pdf_func_1_proj = decay_amplitude_1d_pdf
    # parameter name ordering must match the ordering in the pdf functions
    _param_names_proj = {
        0: ["mu_0", "mu_1", "sigma", "angle"],
        1: ["mu_0", "mu_1", "sigma", "angle", "t1_norm"],
    }

    @property
    def params_proj(self) -> Dict[int, Dict[str, float]]:
        """Returns the parameters for the projected PDFs, computed
        from ``params``.

        The structure of the output dictionary is:

        .. code-block:: python
        
           {
               0: {"param1": float, ...},
               1: {"param1": float, ...},
           }

        """
        params_proj = {state: {} for state in range(2)}

        for state in range(2):
            params_proj[state]["sigma"] = self.params[state]["sigma"]
            params_proj[state]["angle"] = self.params[state]["angle"]

            mu_0, mu_1 = self.statistics["mu_0"], self.statistics["mu_1"]
            rot_angle = get_angle(mu_1 - mu_0)
            params_proj[state]["mu_0"] = rotate_data(mu_0, -rot_angle)[..., 0]
            params_proj[state]["mu_1"] = rotate_data(mu_1, -rot_angle)[..., 0]

        params_proj[1]["t1_norm"] = self.params[1]["t1_norm"]

        return params_proj

    @property
    def statistics(self) -> Dict[str, np.ndarray]:
        """
        Returns dictionary with general statistical data:
        - mu_0: np.array([float, float])
        - mu_1: np.array([float, float])
        - cov_0: np.array([[float, float], [float, float]])
        - cov_1: np.array([[float, float], [float, float]])
        It can also include other information such as rot_angle, rot_shift, ...

        NB: this property is used for plotting and for storing useful
            information in the YAML file
        """
        statistics = {}

        statistics["mu_0"] = np.array(
            [self.params[0]["mu_0_x"], self.params[0]["mu_0_y"]]
        )
        statistics["mu_1"] = np.array(
            [self.params[1]["mu_1_x"], self.params[1]["mu_1_y"]]
        )
        statistics["cov_0"] = self.params[0]["sigma"] * np.eye(2)
        statistics["cov_1"] = self.params[1]["sigma"] * np.eye(2)

        p0 = np.sin(self.params[1]["angle"]) ** 2
        statistics["t1_norm_from_p0"] = -1 / np.log(1 - p0)
        statistics["t1_norm_from_fit"] = self.params[1]["t1_norm"]

        return statistics

    @classmethod
    def fit(
        cls: DecayLinearClassifier,
        shots_0: np.ndarray,
        shots_1: np.ndarray,
        n_bins: int = 100,
    ) -> DecayLinearClassifier:
        """
        Fits the given data to extract the best parameters for classification.

        Parameters
        ----------
        shots_0: np.array(N, 2)
            IQ data when preparing state 0
        shots_1: np.array(M, 2)
            IQ data when preparing state 1
        n_bins:
            Number of bins for the 1d histograms

        Returns
        -------
        `DecayLinearClassifier` containing the fitted parameters
        """
        check_2d_input(shots_0, axis=1)
        check_2d_input(shots_1, axis=1)

        # populate `params` during fitting
        params = {state: {} for state in range(2)}

        # The decay does not affect the direction of \vec{mu0} - \vec{mu1}.
        # Using \vec{mu1} - \vec{mu0} to have the projected 0 blob
        # on the left of the 1 blob
        mu_0, mu_1 = np.average(shots_0, axis=0), np.average(shots_1, axis=0)
        rot_angle = get_angle(mu_1 - mu_0)
        rot_shift = rotate_data(mu_0, -rot_angle)[1]

        # rotate and project data
        shots_0_1d = rotate_data(shots_0, -rot_angle)[..., 0]
        shots_1_1d = rotate_data(shots_1, -rot_angle)[..., 0]

        # get means and standard deviations together because they are shared
        # between the distributions
        all_shots = np.concatenate([shots_0_1d, shots_1_1d])
        counts, x = np.histogram(all_shots, bins=n_bins, density=True)
        x = 0.5 * (x[1:] + x[:-1])

        c_0, c_1 = len(shots_0) / len(all_shots), len(shots_1) / len(all_shots)

        def combined_pdf(x, mu_0, mu_1, sigma, angle0, angle1, t1_norm):
            prob_a = cls._pdf_func_0_proj(x, mu_0, mu_1, sigma, angle0)
            prob_b = cls._pdf_func_1_proj(x, mu_0, mu_1, sigma, angle1, t1_norm)
            return c_0 * prob_a + c_1 * prob_b

        bounds = (
            (np.min(shots_0_1d), np.min(shots_1_1d), 1e-10, 0, 0, 1e-4),
            (
                np.max(shots_0_1d),
                np.max(shots_1_1d),
                np.max(shots_0_1d),
                np.pi / 2,
                np.pi / 2,
                np.inf,
            ),
        )
        guess = (
            np.average(shots_0_1d),
            np.average(shots_1_1d),
            np.std(shots_0_1d),
            np.pi / 2 - 0.25,
            0.2255,
            25,
        )

        popt_comb, pcov = curve_fit(
            combined_pdf, x, counts, p0=guess, bounds=bounds, **FIT_KARGS
        )
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt_comb > 0.1).any():
            warnings.warn("Fitted mean and covariance of state=0 may not be accurate")

        mu_0 = rotate_data([popt_comb[0], rot_shift], rot_angle)
        mu_1 = rotate_data([popt_comb[1], rot_shift], rot_angle)
        for s in range(2):
            params[s]["mu_0_x"], params[s]["mu_0_y"] = mu_0
            params[s]["mu_1_x"], params[s]["mu_1_y"] = mu_1
            params[s]["sigma"] = popt_comb[2]

        # get fit for state=0
        # Note: fitting in log scale improves the results, however there is the
        # problem of having counts=0 (np.log(0) = inf) due to undersampling
        counts, x = np.histogram(shots_0_1d, bins=n_bins, density=True)
        x = 0.5 * (x[1:] + x[:-1])
        x, counts = x[counts != 0], counts[counts != 0]

        bounds = ((0,), (np.pi / 2,))
        guess = (np.pi / 2 - 0.25,)

        popt, pcov = curve_fit(
            lambda x, *p: np.log10(cls._pdf_func_0_proj(x, *popt_comb[:3], *p)),
            x,
            np.log10(counts),
            p0=guess,
            bounds=bounds,
            **FIT_KARGS,
        )
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted mean and covariance of state=0 may not be accurate")

        params[0]["angle"] = popt[0]

        # get fit for state=1
        # Note: fitting in log scale improves the results, however there is the
        # problem of having counts=0 (np.log(0) = inf) due to undersampling
        counts, x = np.histogram(shots_1_1d, bins=n_bins, density=True)
        x = 0.5 * (x[1:] + x[:-1])
        x, counts = x[counts != 0], counts[counts != 0]

        bounds = ((0, 1e-4), (np.pi / 2, np.inf))
        guess = (0.2255, 5)

        popt, pcov = curve_fit(
            lambda x, *p: np.log10(cls._pdf_func_1_proj(x, *popt_comb[:3], *p)),
            x,
            np.log10(counts),
            p0=guess,
            bounds=bounds,
            **FIT_KARGS,
        )
        perr = np.sqrt(np.diag(pcov))
        if (perr / popt > 0.1).any():
            warnings.warn("Fitted mean and covariance of state=1 may not be accurate")

        params[1]["angle"] = popt[0]
        params[1]["t1_norm"] = popt[1]

        return cls(params)
