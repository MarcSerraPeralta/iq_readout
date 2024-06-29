from __future__ import annotations
from typing import Dict, Callable, Union, Tuple

import numpy as np

from ..classifiers import TwoStateLinearClassifier
from ..utils import check_2d_input, get_angle, rotate_data
from ..pdfs import pdf_from_hist2d, pdf_from_hist1d


def _get_mean_from_hist2d(
    bin_centers_x: np.ndarray, bin_centers_y: np.ndarray, density: np.ndarray
):
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
    dx = bin_centers_x[1] - bin_centers_x[0]
    dy = bin_centers_y[1] - bin_centers_y[0]
    mu_x = (bin_centers_x * density.sum(axis=1)).sum() * dx * dy
    mu_y = (bin_centers_y * density.sum(axis=0)).sum() * dx * dy
    return np.array([mu_x, mu_y])


def _get_hist1d_from_hist2d(
    bin_centers_x: np.ndarray,
    bin_centers_y: np.ndarray,
    density: np.ndarray,
    project_funct: Callable,
    bins1d=None,
):
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
    density_1d, xedges = np.histogram(
        zz_proj.flatten(), weights=density.T.flatten(), bins=bins1d, density=True
    )
    bin_centers_1d = 0.5 * (xedges[:-1] + xedges[1:])
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
            bins, pdf_values = _get_hist1d_from_hist2d(
                self.params[state]["bins_x"],
                self.params[state]["bins_y"],
                self.params[state]["pdf_values"],
                self.project,
            )
            params_proj[state]["bins"] = bins
            params_proj[state]["pdf_values"] = pdf_values

        return params_proj

    @property
    def statistics(self) -> Dict[str, np.ndarray]:
        """
        Returns dictionary with general statistical data:

        * ``mu_0``: ``np.array([float, float])``
        * ``mu_1``: ``np.array([float, float])``
        * ``rot_angle``: ``float``
        * ``threshold``: ``float``

        NB: this property is used for plotting and for storing useful
        information in the YAML file.
        """
        statistics = {}

        mu_0 = _get_mean_from_hist2d(
            self.params[0]["bins_x"],
            self.params[0]["bins_y"],
            self.params[0]["pdf_values"],
        )
        mu_1 = _get_mean_from_hist2d(
            self.params[1]["bins_x"],
            self.params[1]["bins_y"],
            self.params[1]["pdf_values"],
        )

        # mu_0 and mu_1 are required for the "project" function
        statistics["mu_0"] = mu_0
        statistics["mu_1"] = mu_1
        statistics["rot_angle"] = get_angle(mu_1 - mu_0)
        statistics["threshold"] = self._get_threshold()

        return statistics

    def _get_threshold(self, p_0: float = 1 / 2):
        """
        Returns the threshold that maximizes the assigment fidelity.
        It uses the 1D histogram data from `params_proj`.

        Parameters
        ----------
        p_0:
            Prior probability of state 0.
            By default, 1/2.

        Returns
        -------
        threshold: float
            Threshold that maximizes the assigment fidelity.
        """
        bin_centers = self.params_proj[0]["bins"]
        cdf_0 = np.cumsum(self.params_proj[0]["pdf_values"])
        cdf_1 = np.cumsum(self.params_proj[1]["pdf_values"])
        diff = cdf_0 - cdf_1
        idx_max = np.argmax(diff)
        threshold = 0.5 * (bin_centers[idx_max] + bin_centers[idx_max + 1])
        return threshold

    @classmethod
    def fit(
        cls: MaxFidLinearClassifier,
        shots_0: np.ndarray,
        shots_1: np.ndarray,
        n_bins: Union[Tuple[int, int], int] = 100,
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
        if isinstance(n_bins, int):
            n_bins = (n_bins, n_bins)
        if (n_bins[0] <= 1) or (n_bins[1] <= 1):
            raise ValueError("Each element of `n_bins` must be strictly larger"
                             f" than 1, but {n_bins} was given.")

        # populate `params` during fitting
        params = {state: {} for state in range(2)}

        all_shots = np.concatenate([shots_0, shots_1], axis=0)

        bins_x = np.linspace(
            np.min(all_shots[:, 0]), np.max(all_shots[:, 0]), n_bins[0]
        )
        bins_y = np.linspace(
            np.min(all_shots[:, 1]), np.max(all_shots[:, 1]), n_bins[1]
        )
        bin_centers_x = 0.5 * (bins_x[:-1] + bins_x[1:])
        bin_centers_y = 0.5 * (bins_y[:-1] + bins_y[1:])

        pdf_values_0, _, _ = np.histogram2d(
            shots_0[:, 0], shots_0[:, 1], bins=[bins_x, bins_y], density=True
        )
        pdf_values_1, _, _ = np.histogram2d(
            shots_1[:, 0], shots_1[:, 1], bins=[bins_x, bins_y], density=True
        )

        for state in range(2):
            params[state]["bins_x"] = bin_centers_x
            params[state]["bins_y"] = bin_centers_y
        params[0]["pdf_values"] = pdf_values_0
        params[1]["pdf_values"] = pdf_values_1

        return cls(params)

    def predict(self, z: np.ndarray, p_0: float = 1 / 2) -> np.ndarray:
        """
        Classifies the given data to 0 or 1 using a threshold:
        - 0 if z_proj <= threshold
        - 1 otherwise

        Parameters
        ----------
        z: np.array(..., 2)
            Points to classify
        p_0
            Probability to measure outcome 0.
            By default 1/2.

        Returns
        -------
        prediction: np.array(...)
            Classification of the given data. It only contains 0s and 1s.
        """
        if (p_0 > 1) or (p_0 < 0):
            raise ValueError(
                "The speficied 'p_0' must be a physical probability, "
                f"but p_0={p_0} (and p1={1-p_0}) were given"
            )

        threshold = self._get_threshold(p_0)
        z_proj = self.project(z)
        return z_proj > threshold

    def project(self, z: np.ndarray) -> np.ndarray:
        """
        Returns the projection of the given IQ data to
        the mu_0 - mu_1 axis

        Parameters
        ----------
        z: np.array(..., 2)
            IQ points

        Returns
        -------
        z_proj: np.array(...)
            Projection of IQ points to mu_0 - mu_1 axis
        """
        check_2d_input(z)
        mu_0 = _get_mean_from_hist2d(
            self.params[0]["bins_x"],
            self.params[0]["bins_y"],
            self.params[0]["pdf_values"],
        )
        mu_1 = _get_mean_from_hist2d(
            self.params[1]["bins_x"],
            self.params[1]["bins_y"],
            self.params[1]["pdf_values"],
        )
        rot_angle = get_angle(mu_1 - mu_0)
        return rotate_data(z, -rot_angle)[..., 0]
