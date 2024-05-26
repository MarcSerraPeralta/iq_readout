from __future__ import annotations
import warnings
from typing import Dict

import numpy as np
from scipy.optimize import curve_fit

from ..classifiers import TwoStateLinearClassifier
from ..utils import check_2d_input, rotate_data, get_angle
from ..pdfs import pdf_from_hist2d, pdf_from_hist1d


def _get_mean_from_hist2d(bin_centers_x: np.ndarray, bin_centers_y: np.ndarray, density: np.ndarray):
    """
    Returns the mean from the output of a 2D histogram.

    Parameters
    ----------
    bin_centers_x: np.ndarray(Nx)
        X-axis centers of the bins from the 2D histogram.
        Note that the output of `numpy.histogram2d` are
        the edegs of the bins.
    bin_centers_y: np.ndarray(Ny)
        Y-axis centers of the bins from the 2D histogram.
        Note that the output of `numpy.histogram2d` are
        the edegs of the bins.
    density: np.ndarray(Nx, Ny)
        Normalized counts of the 2D histogram.

    Returns
    -------
    mean: np.ndarray(2)
        Mean of the 2D histogram data.
    """
    mu_x = (bin_centers_x * density.sum(axis=1)).sum()
    mu_y = (bin_centers_y * density.sum(axis=0)).sum()
    return np.array([mu_x, mu_y])


def _get_hist1d_from_hist2d(bin_centers_x: np.ndarray, bin_centers_y: np.ndarray, density: np.ndarray, project_funct: callable, bins1d=None):
    """
    Returns the 1D histogram of the projected data from a 2D histogram.

    Parameters
    ----------
    bin_centers_x: np.ndarray(Nx)
        X-axis centers of the bins from the 2D histogram.
        Note that the output of `numpy.histogram2d` are
        the edegs of the bins.
    bin_centers_y: np.ndarray(Ny)
        Y-axis centers of the bins from the 2D histogram.
        Note that the output of `numpy.histogram2d` are
        the edegs of the bins.
    density: np.ndarray(Nx, Ny)
        Normalized counts of the 2D histogram.
    project_funct:
        Function that projects the 2D data into 1D.
    bins1d:
        `bins` argument for `numpy.histogram`.
        By default, uses the average of the bins for 
        the two axis of the 2D histogram.

    Returns
    -------
    bin_centers_1d: np.ndarray(Nbins)
        Centers of the 1D bins.
    density_1d: np.ndarray(Nbins)
        Normalized counts of the 1D histogram.
    """
    if bins1d is None:
        bins1d = int(0.5 * (len(bin_centers_x) + len(bin_centers_y)))

    xx, yy = np.meshgrid(bin_centers_x, bin_centers_y)
    zz = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=-1)
    zz_proj = project_funct(zz)
    xedges, density_1d = np.histogram(zz_proj.flatten(), weights=density.flatten(), bins=bins1d, density=True)
    bin_centers_1d = 0.5*(xedges[:-1] + xedges[1:])
    return bin_centers_1d, density_1d


class MaxFidLinearClassifier(TwoStateLinearClassifier):
    """
    Read `gmlda.md` and `TwoStateLinearClassifier` documentation
    """

    _pdf_func_0 = pdf_from_hist2d
    _pdf_func_1 = pdf_from_hist2d
    # parameter name ordering must match the ordering in the pdf functions
    _param_names = {
        0: ["bins_x", "bins_y", "pdf_values"],
        1: ["bins_x", "bins_y", "pdf_values"],
    }
    _pdf_func_0_proj = pdf_from_hist1d
    _pdf_func_1_proj = pdf_from_hist1d
    # parameter name ordering must match the ordering in the pdf functions
    _param_names_proj = {
        0: ["bins", "pdf_values"],
        1: ["bins", "pdf_values"],
    }

    @property
    def params_proj(self) -> Dict[int, Dict[str, float]]:
        """
        Returns the parameters for the projected pdfs, computed
        from `params`.
        The structure of the output dictionary is:
        {
            0: {"param1": float, ...},
            1: {"param1": float, ...}
        }
        """
        params_proj = {state: {} for state in range(2)}

        for state in range(2):
            bins, pdf_values = _get_hist1d_from_hist2d(self.params[state]["bins_x"], self.params[state]["bins_y"], self.params[state]["pdf_values"], self.project)
            params_proj[state]["bins"] = bins
            params_proj[state]["pdf_values"] = pdf_values

        return params_proj

    @property
    def statistics(self) -> Dict[str, np.ndarray]:
        """
        Returns dictionary with general statistical data.

        NB: this property is used for plotting and for storing useful
            information in the YAML file
        """
        statistics = {}

        mu_0 = _get_mean_from_hist2d(self.params[0]["bins_x"], self.params[0]["bins_y"], self.params[0]["pdf_values"])
        mu_1 = _get_mean_from_hist2d(self.params[1]["bins_x"], self.params[1]["bins_y"], self.params[1]["pdf_values"])

        statistics["mu_0"] = mu_0
        statistics["mu_1"] = mu_1
        statistics["rot_angle"] = get_angle(mu_1 - mu_0)

        return statistics

    def _get_threshold(self, bin_centers: np.ndarray, pdf_values_0: np.ndarray, pdf_values_1: np.ndarray, p_0: float):
        """
        Returns the threshold that maximizes the assigment fidelity. 

        Parameters
        ----------
        bin_centers: np.ndarray(N)
            Bin centers of the two 1D histogram.
        pdf_values_i: np.ndarray(N)
            Normalized counts of the 1D histogram when preparing in state=i.
        p_0:
            Prior probability of state 0.

        Returns
        -------
        threshold: float
            Threshold that maximizes the assigment fidelity. 
        """
        cdf_0 = np.cumsum(pdf_values_0)
        cdf_1 = np.cumsum(pdf_values_1)
        diff = cdf_0 - cdf_1
        threshold = bin_centers[np.argmax(diff)]
        return threshold

    @classmethod
    def fit(
        cls: MaxFidLinearClassifier,
        shots_0: np.ndarray,
        shots_1: np.ndarray,
        n_bins: int = 100,
    ) -> MaxFidLinearClassifier:
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
        `MaxFidLinearClassifier` containing the fitted parameters
        """
        check_2d_input(shots_0, axis=1)
        check_2d_input(shots_1, axis=1)

        # populate `params` during fitting
        params = {state: {} for state in range(2)}

        all_shots = np.concatenate([shots_0, shots_1], axis=0)

        bins_x = np.linspace(np.min(all_shots, axis=0), np.max(all_shots, axis=0), n_bins)
        bins_y = np.linspace(np.min(all_shots, axis=1), np.max(all_shots, axis=1), n_bins)

        _, _, pdf_values_0 = np.histogram2d(shots_0, bins=[bins_x, bins_y], density=True)
        _, _, pdf_values_1 = np.histogram2d(shots_1, bins=[bins_x, bins_y], density=True)

        for state in range(2):
            params[state]["bins_x"] = bins_x
            params[state]["bins_y"] = bins_y
        params[0]["pdf_values"] = pdf_values_0
        params[1]["pdf_values"] = pdf_values_1

        return cls(params)
