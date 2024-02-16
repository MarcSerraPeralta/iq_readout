import numpy as np
import matplotlib.pyplot as plt


def probs_prep_meas(
    classifier,
    shots_0: np.ndarray,
    shots_1: np.ndarray,
    shots_2: np.ndarray = None,
) -> np.ndarray:
    """
    Returns matrix whose element i,j corresponds to:
    p(measure state j | prepared state i)

    Parameters
    ----------
    classifier
        A classifier from the iq_readout package
    shots_0: np.ndarray(N, 2)
        Data when preparaing state 0
    shots_1: np.ndarray(N, 2)
        Data when preparaing state 1
    shots_2: np.ndarray(N, 2)
        Data when preparaing state 2

    Returns
    -------
    probs: np.ndarray(2,2) or np.ndarray(3,3)
        The size of the array depends on the number of states that
        the classifier can discriminate.
    """
    if set(["pdf_0", "pdf_1"]) > set(dir(classifier)):
        raise ValueError(
            "'classifier' must have the following methods: "
            f"'pdf_0', 'pdf_1'; but it has {dir(classifier)}"
        )
    if "pdf_2" in dir(classifier) and (shots_2 is None):
        raise ValueError("For 3-state classifiers, one must specify 'shots_2'")

    states = [0, 1]
    shots = [shots_0, shots_1]
    probs = np.zeros((2, 2))
    if "pdf_2" in dir(classifier):
        states = [0, 1, 2]
        shots = [shots_0, shots_1, shots_2]
        probs = np.zeros(3, 3)

    for i, (state, shot) in enumerate(zip(states, shots)):
        prediction = classifier.predict(shot)
        for j, _ in enumerate(states):
            probs[i, j] = np.average(prediction == j)

    return probs


def plot_probs_prep_meas(ax: plt.Axes, probs: np.ndarray) -> plt.Axes:
    """
    Plots the matrix whose element i,j corresponds to:
    p(measure state j | prepared state i)

    Parameters
    ----------
    ax:
        Matplotlib axis
    probs: np.ndarray(N, N)
        Probability matrix

    Returns
    -------
    ax:
        Matplotlib axis with the data plotted
    """
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

    return ax
