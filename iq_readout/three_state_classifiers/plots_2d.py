from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_shots_2d(
    ax: plt.Axes,
    shots_0: np.ndarray,
    shots_1: np.ndarray,
    shots_2: np.ndarray,
    labels: Optional[Tuple[str, str, str]] = None,
    colors: Optional[Tuple[str, str, str]] = None,
) -> plt.Axes:
    """
    Plots the experimental shots a 2D plane

    Parameters
    ----------
    ax:
        Matplotlib axis
    shots_0: np.ndarray(N, 2)
        Experimental data for state 0
    shots_1: np.ndarray(N, 2)
        Experimental data for state 1
    shots_2: np.ndarray(N, 2)
        Experimental data for state 2
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

    shots = [shots_0, shots_1, shots_2]

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
    ax: plt.Axes, classifier, xlim: Tuple[float, float], ylim: Tuple[float, float]
) -> plt.Axes:
    """
    Plots the decision boundaries in a 2D plane

    Parameters
    ----------
    ax:
        Matplotlib axis
    xlim: (xmin, xmax)
        Range of the X axis
    ylim: (ymin, ymax)
        Range of the Y axis

    Returns
    -------
    ax:
        Matplotlib axis with the data plotted
    """
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

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # same scale in x and y axis
    ax.axis("equal")

    return ax


def plot_contour_pdf_2d(
    ax: plt.Axes,
    classifier,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    contour_levels: np.ndarray = [1 / np.e],
) -> plt.Axes:
    """
    Plots the decision boundaries in a 2D plane

    Parameters
    ----------
    ax:
        Matplotlib axis
    xlim: (xmin, xmax)
        Range of the X axis
    ylim: (ymin, ymax)
        Range of the Y axis
    contour_levels
        Levels of the PDFs to be plotted

    Returns
    -------
    ax:
        Matplotlib axis with the data plotted
    """
    x, y = np.linspace(*xlim, 1_000), np.linspace(*ylim, 1_000)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    XX = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=-1)
    prediction = classifier.predict(XX)

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

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # same scale in x and y axis
    ax.axis("equal")

    return ax
