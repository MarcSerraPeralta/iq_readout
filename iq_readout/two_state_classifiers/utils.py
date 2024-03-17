import matplotlib.pyplot as plt

from .plots_1d import plot_pdfs_projected
from ..plots.plots_2d import plot_shots_2d
from ..metrics import get_probs_prep_meas, plot_probs_prep_meas


def summary(classifier, shots_0, shots_1):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

    axes[0] = plot_shots_2d(axes[0], shots_0, shots_1)

    probs = get_probs_prep_meas(classifier, shots_0, shots_1)
    axes[1] = plot_probs_prep_meas(axes[1], probs)

    axes[2] = plot_pdfs_projected(
        axes[2],
        classifier,
        shots_0,
        shots_1,
    )
    axes[2].set_title("projection: 0-1")

    fig.tight_layout()

    return fig
