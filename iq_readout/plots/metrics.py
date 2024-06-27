"""Functions to plot the readout metrics.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_probs_prep_meas(
    ax: plt.Axes, probs: np.ndarray, colorbar_flag: bool = False
) -> plt.Axes:
    """
    Plots the matrix whose element i,j corresponds to:
    p(measure state j | prepared state i)

    Parameters
    ----------
    ax:
        Matplotlib axis
    probs: np.ndarray(N, N)
        Probability matrix
    colorbar_flag
        If True, adds colorbar to the figure

    Returns
    -------
    ax:
        Matplotlib axis with the data plotted
    """
    fidelity = 0.5 * (probs[0, 0] + probs[1, 1])

    probs = probs * 100  # percentage
    # rotate the matrix so that the diagonal starts from top left
    # instead of bottom left
    probs = np.rot90(probs, 3)
    n_states = len(probs)

    # transpose the matrix because imshow plots it transpose
    heatmap = ax.imshow(probs.T, cmap="Blues", vmin=0, vmax=100)

    # add numbers in matrix
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            color = "white" if n_states - 1 - i == j else "black"
            ax.text(i, j, f"{probs[i, j]:0.2f}", ha="center", va="center", color=color)

    # Add colorbar for reference
    if colorbar_flag:
        fig = ax.get_figure()
        colorbar = fig.colorbar(heatmap, ax=ax)
        colorbar.set_label("probability (%)")

    ax.set_xlabel("outcome")
    ax.set_ylabel("prepared state")

    ax.set_xticks(range(n_states))
    ax.set_xticklabels([f"{i}" for i in range(n_states)])
    ax.set_yticks(range(n_states))
    ax.set_yticklabels([f"|{i}>" for i in range(n_states)][::-1])
    ax.set_xlim(-0.5, n_states - 0.5)
    ax.set_ylim(-0.5, n_states - 0.5)

    ax.set_title(f"Fidelity (states 0&1) = {fidelity*100:0.2f}%")

    return ax
