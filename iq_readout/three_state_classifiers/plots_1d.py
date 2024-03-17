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

    # rotate shots to the given axis
    vector = points[1] - points[0]
    theta = get_angle(vector)
    shift = rotate_data(points[0], -theta)[1]
    shots_rot = rotate_data(shots, -theta)

    # select shots inside a small rectangle
    eps = 0.05 * np.linalg.norm(vector)
    shots_proj = shots_rot[
        (shots_rot[:, 1] >= shift - eps) & (shots_rot[:, 1] <= shift + eps)
    ]
    shots_proj = shots_proj[:, 0]

    # it can happen that there is almost no data to plot
    # e.g. along mu0 and mu1, maybe shots_2 will not have
    # any data
    if len(shots_proj) < 100:
        return ax

    # plot experimental histogram
    hist, bin_edges = np.histogram(shots_proj, bins=50, density=True)
    hist = hist * len(shots_proj) / len(shots)
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
    # renormalize the pdf as done for the experimental data
    pdf = np.sum(pdf, axis=1) / (np.sum(pdf) * (x[1] - x[0]))
    pdf = pdf * len(shots_proj) / len(shots)

    # plot pdf
    ax.plot(x, pdf, color=color, linestyle="-")

    ax.set_xlabel("projected data, z")
    ax.set_ylabel("PDF(z)")
    ax.set_yscale("log")
    ax.set_ylim(ymin=np.min(hist[hist > 0]) * 0.1)
    if label is not None:
        ax.legend(loc="best")

    return ax


def plot_pdfs_projected(
    ax: plt.Axes,
    points: Tuple[Tuple[float, float], Tuple[float, float]],
    shots_0: np.ndarray,
    shots_1: np.ndarray,
    shots_2: np.ndarray,
    classifier,
    labels: Optional[Tuple[str, str, str]] = None,
    colors: Optional[Tuple[str, str, str]] = None,
) -> plt.Axes:
    """
    Plots the projected experimental histogram and the fitted pdf
    in the given axis

    Parameters
    ----------
    ax:
        Matplotlib axis
    points: (mean_1, mean_2)
        Points defining the line in which to project the data
    shots_0: np.ndarray(N, 2)
        Experimental data for state 0
    shots_1: np.ndarray(N, 2)
        Experimental data for state 1
    shots_2: np.ndarray(N, 2)
        Experimental data for state 2
    classifier:
        Class with 'pdf_0', 'pdf_1' and 'pdf_2' functions
    labels: (label_0, label_1, label_2)
        Labels for the state 0, 1, and 2 data
    colors: (color_0, color_1, color_2)
        Colors for the state 0, 1, and 2 data

    Returns
    -------
    ax:
        Matplotlib axis with the data plotted
    """
    if labels is None:
        labels = ["0", "1", "2"]
    if colors is None:
        colors = ["orange", "blue", "green"]
    if len(labels) != 3:
        raise ValueError(
            f"'labels' must contain 3 elements, but {len(labels)} were given"
        )
    if len(colors) != 3:
        raise ValueError(
            f"'colors' must contain 3 elements, but {len(colors)} were given"
        )
    if set(["pdf_0", "pdf_1", "pdf_2"]) > set(dir(classifier)):
        raise ValueError(
            "'classifier' must have the following methods: "
            "'pdf_0', 'pdf_1', and 'pdf_2'; "
            f"but it has {dir(classifier)}"
        )

    shots = [shots_0, shots_1, shots_2]
    pdfs = [classifier.pdf_0, classifier.pdf_1, classifier.pdf_2]

    for shot, pdf, color, label in zip(shots, pdfs, colors, labels):
        ax = plot_pdf_projected(ax, points, shot, pdf, label=label, color=color)

    # avoid automatic axis range problems due to plotting different curves
    # the 5% margin needs to be done in log scale
    # the ylim limit is already specified in 'plot_pdf_projected'
    _, ymax = ax.yaxis.get_data_interval()
    ymin, _ = ax.get_ylim()
    margin = 10 ** (0.05 * (np.log10(ymax) - np.log10(ymin)))
    ax.set_ylim(ymax=ymax * margin)

    return ax