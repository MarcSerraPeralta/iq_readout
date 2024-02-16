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

    return ax
