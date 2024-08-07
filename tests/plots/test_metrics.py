import numpy as np
import matplotlib.pyplot as plt

from iq_readout.plots.metrics import plot_probs_prep_meas


def test_plot_probs_prep_meas(show_figures):
    probs = np.array([[0.9, 0.1, 0.1], [0.4, 0.7, 0.4], [0.2, 0.2, 0.8]])

    fig, ax = plt.subplots()

    plot_probs_prep_meas(ax, probs)

    if show_figures:
        plt.show()
    plt.close()

    return
