import numpy as np
import matplotlib.pyplot as plt

from iq_readout.three_state_classifiers.plots_2d import plot_shots_2d


def test_plot_shots_2d():
    fig, ax = plt.subplots()

    shots_0 = np.random.multivariate_normal([0, 0], np.eye(2), 10_000)
    shots_1 = np.random.multivariate_normal([0, 4], np.eye(2), 10_000)
    shots_2 = np.random.multivariate_normal([2, 2], np.eye(2), 10_000)

    plot_shots_2d(ax, shots_0, shots_1, shots_2)

    return
