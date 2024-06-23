import numpy as np
from scipy.special import erf

from .utils import check_2d_input, rotate_data, get_angle


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
    x: np.array(...)
        Points in the 1D space
    mu
        Mean of the first coordinate
    sigma
        Standard deviation of the two coordinates

    Returns
    -------
    prob: np.array(...)
        Values of the probability density function
    """
    prob = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * (x - mu) ** 2 / sigma**2)
    return prob


def simple_1d_gaussian_double_mixture(
    x: np.ndarray,
    mu_0: float,
    mu_1: float,
    sigma: float,
    angle: float,
) -> np.ndarray:
    """
    Probability density function corresponding to the sum of two
    `simple_1d_gaussian`

    Parameters
    ----------
    x: np.array(...)
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
    a_0, a_1 = np.sin(angle) ** 2, np.cos(angle) ** 2

    prob_0 = simple_1d_gaussian(x, mu=mu_0, sigma=sigma)
    prob_1 = simple_1d_gaussian(x, mu=mu_1, sigma=sigma)

    prob = a_0 * prob_0 + a_1 * prob_1

    return prob


def simple_2d_gaussian(
    z: np.ndarray,
    mu_x: float,
    mu_y: float,
    sigma: float,
) -> np.ndarray:
    """
    Probability density function of a 2D Gaussian with
    mean = (mu0, mu1) and covariance matrix = diag(sigma**2, sigma**2)

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
    prob: np.array(...)
        Values of the probability density function
    """
    check_2d_input(z)
    x, y = z[..., 0], z[..., 1]
    x_norm, y_norm = (x - mu_x) / sigma, (y - mu_y) / sigma
    prob = 1 / (2 * np.pi * sigma**2) * np.exp(-0.5 * (x_norm**2 + y_norm**2))
    return prob


def simple_2d_gaussian_double_mixture(
    z: np.ndarray,
    mu_0_x: float,
    mu_0_y: float,
    mu_1_x: float,
    mu_1_y: float,
    sigma: float,
    angle: float,
) -> np.ndarray:
    """
    Probability density function corresponding to the sum of two
    `simple_2d_gaussian`

    Parameters
    ----------
    z: np.array(..., 2)
        Points in the 2D space
    mu_i_x
        Mean of the first coordinate for the i^th Gaussian
    mu_i_y
        Mean of the second coordinate for the i^th Gaussian
    sigma
        Standard deviation of the two coordinates for both Gaussians
    angle
        Weight of the 1st Gaussian is sin(angle)**2 and
        of the 2nd Gaussian is cos(angle)**2 to ensure that
        the PDF is normalized

    Returns
    -------
    prob: np.array(...)
        Values of the probability density function
    """
    check_2d_input(z)
    a_0, a_1 = np.sin(angle) ** 2, np.cos(angle) ** 2

    prob_0 = simple_2d_gaussian(z, mu_x=mu_0_x, mu_y=mu_0_y, sigma=sigma)
    prob_1 = simple_2d_gaussian(z, mu_x=mu_1_x, mu_y=mu_1_y, sigma=sigma)

    prob = a_0 * prob_0 + a_1 * prob_1

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
    `simple_2d_gaussian`

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
    prob
        Values of the probability density function
    """
    a_0, a_1 = np.sin(angle) ** 2, np.cos(angle) ** 2
    C = np.sign(mu_1 - mu_0) * (mu_0 - x) / (np.sqrt(2) * sigma) + sigma / (
        np.sqrt(2) * np.abs(mu_1 - mu_0) * t1_norm
    )
    P = 0.5 * (mu_1 - mu_0) ** 2 / sigma**2
    prob_0 = np.exp(-0.5 * (x - mu_1) ** 2 / sigma**2) / np.sqrt(2 * np.pi * sigma**2)
    prob_1 = (
        np.exp(-0.5 * (x - mu_0) ** 2 / sigma**2 + C**2)
        / (np.sqrt(2 * np.pi * sigma**2) * t1_norm)
        * np.sqrt(np.pi / (4 * P))
        * (erf(C + np.sqrt(P)) - erf(C))
        / (1 - np.exp(-1 / t1_norm))
    )

    prob = a_0 * prob_0 + a_1 * prob_1

    return prob


def decay_amplitude_2d_pdf(
    z: np.ndarray,
    mu_0_x: float,
    mu_0_y: float,
    mu_1_x: float,
    mu_1_y: float,
    sigma: float,
    angle: float,
    t1_norm: float,
) -> np.ndarray:
    """
    Probability density function corresponding to a `decay_amplitude_1d_pdf` in
    the (parallel) projected axis along mu_0 and mu_1 and a `simple_1d_gaussian`
    in the perpendicular dimension

    Parameters
    ----------
    z: np.array(..., 2)
        Points in the 2D space
    mu_i_x
        Mean of the first coordinate for the i^th Gaussian
    mu_i_y
        Mean of the second coordinate for the i^th Gaussian
    sigma
        Standard deviation of the two coordinates for both Gaussians
    angle
        Weight of the 1st Gaussian is sin(angle)**2 and
        of the 2nd Gaussian is cos(angle)**2 to ensure that
        the PDF is normalized
    t1_norm
        T1 normalized with respect to the measurement duration

    Returns
    -------
    prob: np.array(...)
        Values of the probability density function
    """
    check_2d_input(z)
    mu_0, mu_1 = np.array([mu_0_x, mu_0_y]), np.array([mu_1_x, mu_1_y])
    rot_angle = get_angle(mu_1 - mu_0)
    z_rot = rotate_data(z, -rot_angle)
    mu_0_rot, mu_1_rot = rotate_data(mu_0, -rot_angle), rotate_data(mu_1, -rot_angle)

    prob_para = decay_amplitude_1d_pdf(
        z_rot[..., 0],
        mu_0=mu_0_rot[0],
        mu_1=mu_1_rot[0],
        sigma=sigma,
        angle=angle,
        t1_norm=t1_norm,
    )
    prob_perp = simple_1d_gaussian(
        z_rot[..., 1],
        mu=mu_0_rot[1],  # it is the same as mu_1_rot[1]
        sigma=sigma,
    )

    prob = prob_para * prob_perp

    return prob


def pdf_from_hist1d(x, bins, pdf_values):
    """
    Returns the PDF value from the 1D histogram bins closest to `x`.

    Parameters
    ----------
    x: np.array(...)
        Points in the 1D space.
    bins: np.ndarray(n_bins)
        Centers of the bins of the 1D histogram.
    pdf_values: np.ndarray(n_bins)
        Normalized counts of the 1D histogram.

    Returns
    -------
    prob: np.ndarray(...)
        Values of the probability density function.
    """
    idxs = np.searchsorted(bins, x + np.diff(bins)[0] / 2, side="right") - 1
    return pdf_values[idxs]


def pdf_from_hist2d(z, bins_x, bins_y, pdf_values):
    """
    Returns the PDF value from the 2D histogram bins closest to `x`.

    Parameters
    ----------
    z: np.array(..., 2)
        Points in the 2D space.
    bins_x: np.ndarray(n_bins_x)
        Centers of the X-axis bins of the 2D histogram.
    bins_y: np.ndarray(n_bins_y)
        Centers of the Y-axis bins of the 2D histogram.
    pdf_values: np.ndarray(n_bins_x, n_bins_y)
        Normalized counts of the 2D histogram.

    Returns
    -------
    prob: np.ndarray(...)
        Values of the probability density function.
    """
    check_2d_input(z)
    idxs_x = (
        np.searchsorted(bins_x, z[..., 0] + np.diff(bins_x)[0] / 2, side="right") - 1
    )
    idxs_y = (
        np.searchsorted(bins_y, z[..., 1] + np.diff(bins_y)[0] / 2, side="right") - 1
    )
    prob = pdf_values[idxs_x.flatten(), idxs_y.flatten()]
    prob = prob.reshape(np.shape(z)[:-1])
    return prob
