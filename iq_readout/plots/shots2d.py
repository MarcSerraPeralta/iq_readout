"""Functions to plot in 2D.
"""

from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def plot_shots_2d(
    ax: plt.Axes,
    *shots: List[np.ndarray],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
) -> plt.Axes:
    """
    Plots the experimental shots a 2D plane

    Parameters
    ----------
    ax:
        Matplotlib axis
    shots: [np.ndarray(N, 2), np.ndarray(N, 2), ...]
        Experimental data for state 0, 1, ...
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
        labels = [f"{i}" for i, _ in enumerate(shots)]
    if colors is None:
        all_colors = ["orange", "blue", "green"]
        colors = all_colors[: len(shots)]

    # calculate the transparency of the points
    # based on the number of points on top of each other
    max_points = 0
    for shot in shots:
        counts, _, _ = np.histogram2d(shot[:, 0], shot[:, 1], bins=[50, 50])
        counts = np.max(counts)
        if counts > max_points:
            max_points = counts

    for shots, color, label in zip(shots, colors, labels):
        ax.plot(
            shots[..., 0],
            shots[..., 1],
            color=color,
            linestyle="none",
            marker=".",
            alpha=np.min([100 / max_points, 0.1]),
        )
        ax.plot(
            [],
            [],
            label=label,
            color=color,
            linestyle="none",
            marker=".",
        )

    ax.set_xlabel("I [a.u.]")
    ax.set_ylabel("Q [a.u.]")
    ax.legend(loc="best")
    # same scale in x and y axis
    ax.axis("equal")

    return ax


def plot_boundaries_2d(
    ax: plt.Axes,
    classifier,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
) -> plt.Axes:
    """
    Plots the decision boundaries in a 2D plane

    Parameters
    ----------
    ax:
        Matplotlib axis
    xlim: (xmin, xmax)
        Range of the X axis.
        Default: `ax.get_xlim()`.
    ylim: (ymin, ymax)
        Range of the Y axis.
        Default: `ax.get_ylim()`.

    Returns
    -------
    ax:
        Matplotlib axis with the data plotted
    """
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    x, y = np.linspace(*xlim, 1_000), np.linspace(*ylim, 1_000)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    XX = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=-1)
    prediction = classifier.predict(XX)

    ax.contour(
        xx,
        yy,
        prediction,
        levels=np.unique(prediction),
        colors="black",
        linestyles="--",
    )

    ax.set_xlabel("I [a.u.]")
    ax.set_ylabel("Q [a.u.]")

    # same scale in x and y axis
    ax.axis("equal")

    return ax


def plot_contour_pdf_2d(
    ax: plt.Axes,
    pdf_func: callable,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    contour_levels: np.ndarray = [1 / np.e],
) -> plt.Axes:
    """
    Plots the decision boundaries in a 2D plane

    Parameters
    ----------
    ax:
        Matplotlib axis.
    xlim: (xmin, xmax)
        Range of the X axis.
        Default: `ax.get_xlim()`.
    ylim: (ymin, ymax)
        Range of the Y axis.
        Default: `ax.get_ylim()`.
    contour_levels
        Levels of the PDFs to be plotted.

    Returns
    -------
    ax:
        Matplotlib axis with the data plotted
    """
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    x, y = np.linspace(*xlim, 1_000), np.linspace(*ylim, 1_000)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    XX = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=-1)
    pdf = pdf_func(XX)

    ax.contour(
        xx,
        yy,
        pdf,
        levels=contour_levels,
        colors="gray",
        linestyles="--",
    )

    ax.set_xlabel("I [a.u.]")
    ax.set_ylabel("Q [a.u.]")

    # same scale in x and y axis
    ax.axis("equal")

    return ax
