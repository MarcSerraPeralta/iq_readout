"""Common functions used in IQ readout.
"""

from typing import Tuple

import numpy as np

FIT_KARGS = dict(
    loss="soft_l1",  # loss="soft_l1" leads to more stable fits
    ftol=1e-10,  # higher accuracy in params
    xtol=1e-10,  # higher accuracy in params
    gtol=1e-10,  # higher accuracy in params
)


def check_2d_input(x: np.ndarray, axis=-1):
    """
    Checks that `x` has length 2 at the last axis.
    If `axis != -1`, it also checks that `x` has `axis+1` axis.

    Parameters
    ----------
    x
        Input to check.
    axis
        Axis of the last dimension fo the input.
        By default `-1`.
    """
    if x.shape[axis] != 2:
        raise ValueError(
            "input must be specified with shape (..., 2), "
            f"but the following shape was given: {x.shape}"
        )
    if (axis != -1) and (len(x.shape) != axis + 1):
        raise ValueError(
            f"input must have {axis+1} axis, "
            f"but the following shape was given: {x.shape}"
        )

    return


def get_angle(vector: np.ndarray) -> float:
    """
    The counterclockwise angle from the x-axis in the range (-pi, pi].

    Parameters
    ----------
    vector: np.ndarray(2)
        Input vector.

    Returns
    -------
    angle
        Counterclockwise angle of `vector`.
    """
    assert vector.shape == (2,)

    angle = np.arctan2(*vector[::-1])  # arctan(y/x)
    return angle


def rotate_data(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Counterclock-wise rotation of `x` by an angle theta.

    Parameters
    ----------
    x: np.ndarray(..., 2)
        Input data to rotate
    theta: float
        Angle used to rotate the data.
        Must be following counterclockwise sign.

    Returns
    -------
    output: np.ndarray(..., 2)
        Rotated data.
    """
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.einsum("ij,...j->...i", rot, x)


def histogram_2d(
    z: np.ndarray,
    n_bins: Tuple[int, int] = [100, 100],
    density: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs a 2d histogram and returns the
    counts, xx, and yy data flattened.

    Parameters
    ----------
    z: np.array(N, 2)
        Points in the 2D space
    n_bins:(nx_bins, ny_bins)
        List of two elements corresponding to the
        number of bins for the first and second coordinate
    density
        If True, returns the probability density function values

    Returns
    -------
    counts: np.array(nx_bins, ny_bins)
        Counts or PDF
    xx_centers: np.array(nx_bins, ny_bins)
        Centers of the bins for the first coordinate
    yy_centers: np.array(nx_bins, ny_bins)
        Centers of the bins for the second coordinate
    """
    check_2d_input(z, axis=1)
    x, y = z[:, 0], z[:, 1]
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins, density=density)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    # using indexing="ij" so that xx_centers and yy_centers have
    # the same shape as counts, which follows (nx_bins, ny_bins)
    xx_centers, yy_centers = np.meshgrid(x_centers, y_centers, indexing="ij")
    return counts, xx_centers, yy_centers


def reshape_histogram_2d(
    counts: np.ndarray, xx: np.ndarray, yy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattens the output of the `histogram_2d` and combines
    the two coordinates to a single variable with
    shape=(nx_bins * ny_bins, 2) and counts with shape=(nx_bins * ny_bins,).

    NB: this function is used for processing the histogram
        before fitting the pdfs

    Parameters
    ----------
    counts: np.array(nx_bins, ny_bins)
        Counts or PDF
    xx: np.array(nx_bins, ny_bins)
        Centers of the bins for the first coordinate
    yy: np.array(nx_bins, ny_bins)
        Centers of the bins for the second coordinate

    Returns
    -------
    counts: np.array(nx_bins * ny_bins)
        Counts or PDF
    zz: np.array(nx_bins * ny_bins, 2)
        Centers of the bins in 2D
    """
    counts = counts.reshape(-1)
    xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)
    zz = np.concatenate([xx, yy], axis=1)
    return counts, zz
