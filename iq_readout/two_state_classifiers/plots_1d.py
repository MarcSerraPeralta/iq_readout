from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_pdf_projected(
    ax: plt.Axes,
    shots: np.ndarray,
    pdf_func: callable,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """
    Plots the projected experimental histogram and the fitted pdf
    in the given axis

    Parameters
    ----------
    ax:
        Matplotlib axis
    shots: np.ndarray(N)
        Projected experimental data
    pdf:
        Probability density function for the projected data
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

    # plot experimental histogram
    hist, bin_edges = np.histogram(shots, bins=100, density=True)
    ax.stairs(hist, bin_edges, color=color, alpha=0.5, label=label, fill=True)

    # plot pdf
    zmin, zmax = np.min(shots), np.max(shots)
    z = np.linspace(zmin, zmax, 1_000)
    pdf = pdf_func(z)
    ax.plot(z, pdf, color=color, linestyle="-")

    ax.set_xlabel("projected data, z")
    ax.set_ylabel("PDF(z)")
    ax.set_yscale("log")
    ax.set_ylim(ymin=np.min(hist[hist > 0]) * 0.1)
    if label is not None:
        ax.legend(loc="best")

    return ax


def plot_pdfs_projected(
    ax: plt.Axes,
    classifier,
    shots_0: np.ndarray,
    shots_1: np.ndarray,
    labels: Optional[Tuple[str, str]] = None,
    colors: Optional[Tuple[str, str]] = None,
) -> plt.Axes:
    """
    Plots the projected experimental histogram and the fitted pdf
    in the given axis

    Parameters
    ----------
    ax:
        Matplotlib axis
    shots_0: np.ndarray(N, 2)
        Experimental data for state 0
    shots_1: np.ndarray(N, 2)
        Experimental data for state 1
    classifier:
        Class with 'project', 'pdf_0_projected' and 'pdf_1_projected'
        functions
    labels: (label_0, label_1)
        Labels for the state 0 and state 1 data
    colors: (color_0, color_1)
        Colors for the state 0 and state 1 data

    Returns
    -------
    ax:
        Matplotlib axis with the data plotted
    """
    if labels is None:
        labels = ["0", "1"]
    if colors is None:
        colors = ["orange", "blue"]
    if set(["project", "pdf_0_projected", "pdf_1_projected"]) > set(dir(classifier)):
        raise ValueError(
            "'classifier' must have the following methods: "
            "'project', 'pdf_0_projected', and 'pdf_1_projected'; "
            f"but it has {dir(classifier)}"
        )
    if len(labels) != 2:
        raise ValueError(
            f"'labels' must contain 2 elements, but {len(labels)} were given"
        )
    if len(colors) != 2:
        raise ValueError(
            f"'colors' must contain 2 elements, but {len(colors)} were given"
        )

    # state 1
    # first do state 1 because the minimum ylim
    # is expected to be larger for this state
    z = classifier.project(shots_1)
    ax = plot_pdf_projected(
        ax, z, classifier.pdf_1_projected, label=labels[1], color=colors[1]
    )

    # state 0
    z = classifier.project(shots_0)
    ax = plot_pdf_projected(
        ax, z, classifier.pdf_0_projected, label=labels[0], color=colors[0]
    )

    # avoid automatic axis range problems due to plotting different curves
    # the 5% margin needs to be done in log scale
    # the ylim limit is already specified in 'plot_pdf_projected'
    _, ymax = ax.yaxis.get_data_interval()
    ymin, _ = ax.get_ylim()
    margin = 10 ** (0.05 * (np.log10(ymax) - np.log10(ymin)))
    ax.set_ylim(ymax=ymax * margin)

    return ax
