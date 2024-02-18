import matplotlib.pyplot as plt

from .plots_1d import plot_pdfs_projected
from .plots_2d import plot_shots_2d, plot_boundaries_2d
from ..metrics import get_probs_prep_meas, plot_probs_prep_meas


def summary(classifier, shots_0, shots_1, shots_2):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))

    plot_shots_2d(axes[0, 0], shots_0=shots_0, shots_1=shots_1, shots_2=shots_2)

    probs = get_probs_prep_meas(
        classifier=classifier, shots_0=shots_0, shots_1=shots_1, shots_2=shots_2
    )
    plot_probs_prep_meas(axes[0, 1], probs)

    plot_shots_2d(axes[0, 2], shots_0=shots_0, shots_1=shots_1, shots_2=shots_2)
    xlim, ylim = axes[0, 2].get_xlim(), axes[0, 2].get_ylim()
    plot_boundaries_2d(axes[0, 2], classifier=classifier, xlim=xlim, ylim=ylim)
    axes[0, 2].legend().remove()

    params = classifier.params
    mu_0 = [params[0]["mu_0_x"], params[0]["mu_0_y"]]
    mu_1 = [params[1]["mu_1_x"], params[1]["mu_1_y"]]
    mu_2 = [params[2]["mu_2_x"], params[2]["mu_2_y"]]

    plot_pdfs_projected(
        axes[1, 0],
        points=[mu_0, mu_1],
        classifier=classifier,
        shots_0=shots_0,
        shots_1=shots_1,
        shots_2=shots_2,
    )
    axes[1, 0].set_title("projection: 0-1")
    axes[1, 0].legend().remove()

    plot_pdfs_projected(
        axes[1, 1],
        points=[mu_1, mu_2],
        classifier=classifier,
        shots_0=shots_0,
        shots_1=shots_1,
        shots_2=shots_2,
    )
    axes[1, 1].set_title("projection: 1-2")
    axes[1, 1].legend().remove()

    plot_pdfs_projected(
        axes[1, 2],
        points=[mu_0, mu_2],
        classifier=classifier,
        shots_0=shots_0,
        shots_1=shots_1,
        shots_2=shots_2,
    )
    axes[1, 2].set_title("projection: 0-2")
    axes[1, 2].legend().remove()

    # fig.delaxes(axes[0, 2])
    fig.tight_layout()

    return fig
