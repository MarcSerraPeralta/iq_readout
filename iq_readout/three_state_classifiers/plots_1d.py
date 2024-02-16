from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..utils import get_angle, rotate_data


def plot_pdf_projected(
    ax: plt.Axes,
    points: Tuple[Tuple[float, float], Tuple[float, float]],
    shots: np.ndarray,
    pdf_func: callable,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """
    Plots the projection of the experimental histogram and the fitted pdf
    to the line defined by the points in the given axes

    Parameters
    ----------
    ax:
        Matplotlib axis
    points: (mean_1, mean_2)
        Points defining the line in which to project the data
    shots: np.ndarray(N, 2)
        Experimental data
    pdf_func:
        2D probability density function for the data
    label:
        Label for the data
    color:
        Color for the data

    Returns
    -------
    ax:
        Matplotlib axis with the data plotted
    """
    if color is None:
        color = "blue"
    if not callable(pdf_func):
        raise ValueError("'pdf_func' must be callable")
    points = np.array(points)
    if points.shape != (2, 2):
        raise ValueError(
            f"Shape of 'points' must be (2,2), but {points.shape} was given"
        )
    if shots.shape[-1] != 2 and len(shots.shape) != 2:
        raise ValueError(
            f"Shape of 'shots' must be (N, 2), but {shots.shape} was given"
        )

    # rotate shots to the given axis
    vector = points[1] - points[0]
    theta = get_angle(vector)
    shift = rotate_data(vector, -theta)[1]
    shots_rot = rotate_data(shots, -theta)

    # select shots inside a small rectangle
    eps = 0.05 * np.linalg.norm(vector)
    shots_proj = shots_rot[
        (shots_rot[:, 1] >= shift - eps) & (shots_rot[:, 1] <= shift + eps)
    ]
    shots_proj = shots_proj[:, 0]

    # plot experimental histogram
    hist, bin_edges = np.histogram(shots_proj, bins=50, density=True)
    ax.stairs(hist, bin_edges, color=color, alpha=0.5, label=label, fill=True)

    # get theoretical data
    # 1) create grid in rectangle
    # 2) unrotate data
    # 3) evaluate pdf
    # 4) integrate over the "eps" size of the rectangle
    x = np.linspace(np.min(shots_proj), np.max(shots_proj), 1_000)
    y = np.linspace(shift - eps, shift + eps, 100)
    xx, yy = np.meshgrid(x, y, indexing="ij")  # follow cartesian indexing
    XX = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=-1)
    XX_unrot = rotate_data(XX, theta)
    pdf = pdf_func(XX_unrot)
    pdf = np.sum(pdf, axis=1) / (np.sum(pdf) * (x[1] - x[0]))

    # plot pdf
    ax.plot(x, pdf, color=color, linestyle="-")

    ax.set_xlabel("projected data, z")
    ax.set_ylabel("PDF(z)")
    ax.set_yscale("log")
    ax.set_ylim(ymin=np.min(hist[hist > 0]) * 0.1)
    if label is not None:
        ax.legend(loc="best")

    return ax
