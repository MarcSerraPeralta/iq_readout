"""Plot summaries for the classifiers.
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..metrics import get_probs_prep_meas
from .shots1d import plot_two_pdfs_projected, plot_several_pdfs_along_line
from .shots2d import plot_shots_2d, plot_boundaries_2d, plot_contour_pdf_2d
from .metrics import plot_probs_prep_meas


def summary(classifier, *shots: List[np.ndarray]) -> plt.Figure:
    """Figure to show a general overview of the performance of the classifier.
    """
    if classifier._num_states == 2:
        fig = two_state_classifier(classifier, *shots)
    elif classifier._num_states == 3:
        fig = three_state_classifier(classifier, *shots)
    else:
        raise ValueError(
            "Only implemented for two- and three-state classifiers, "
            f"but {classifier._num_states}-state classifier was given. "
        )

    return fig


def two_state_classifier(
    classifier,
    shots_0: np.ndarray,
    shots_1: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

    axes[0] = plot_shots_2d(axes[0], shots_0, shots_1)

    probs = get_probs_prep_meas(classifier, shots_0, shots_1)
    axes[1] = plot_probs_prep_meas(axes[1], probs)

    axes[2] = plot_two_pdfs_projected(
        axes[2],
        classifier,
        shots_0,
        shots_1,
    )
    axes[2].set_title("projection: 0-1")

    fig.tight_layout()

    return fig


def three_state_classifier(
    classifier,
    shots_0: np.ndarray,
    shots_1: np.ndarray,
    shots_2: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))

    # (1) 2D plot with the experimental shots
    plot_shots_2d(axes[0, 0], shots_0, shots_1, shots_2)

    # (2) p(m|s) matrix
    probs_prep_meas = get_probs_prep_meas(classifier, shots_0, shots_1, shots_2)
    plot_probs_prep_meas(axes[0, 1], probs_prep_meas)

    # (3) 2D plot with decision boundaries and 1/e countour
    plot_shots_2d(axes[0, 2], shots_0, shots_1, shots_2)
    plot_boundaries_2d(axes[0, 2], classifier)
    plot_contour_pdf_2d(axes[0, 2], classifier.pdf_0)
    plot_contour_pdf_2d(axes[0, 2], classifier.pdf_1)
    plot_contour_pdf_2d(axes[0, 2], classifier.pdf_2)
    axes[0, 2].legend().remove()

    # (4-6) 1D plot along a line
    params = classifier.params
    mu_0 = [params[0]["mu_0_x"], params[0]["mu_0_y"]]
    mu_1 = [params[1]["mu_1_x"], params[1]["mu_1_y"]]
    mu_2 = [params[2]["mu_2_x"], params[2]["mu_2_y"]]

    plot_several_pdfs_along_line(
        axes[1, 0],
        [mu_0, mu_1],
        classifier,
        shots_0,
        shots_1,
        shots_2,
    )
    axes[1, 0].set_title("projection: 0-1")
    axes[1, 0].legend().remove()

    plot_several_pdfs_along_line(
        axes[1, 1],
        [mu_1, mu_2],
        classifier,
        shots_0,
        shots_1,
        shots_2,
    )
    axes[1, 1].set_title("projection: 1-2")
    axes[1, 1].legend().remove()

    plot_several_pdfs_along_line(
        axes[1, 2],
        [mu_0, mu_2],
        classifier,
        shots_0,
        shots_1,
        shots_2,
    )
    axes[1, 2].set_title("projection: 0-2")
    axes[1, 2].legend().remove()

    fig.tight_layout()

    return fig
