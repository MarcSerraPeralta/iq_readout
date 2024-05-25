import numpy as np
import matplotlib.pyplot as plt

from iq_readout.plots.plots_1d import plot_pdf_projected


def test_plot_pdf_projected(show_figures):
    fig, ax = plt.subplots()

    mu0, mu1 = np.array([0, 4])
    shots_0_0 = np.random.normal(mu0, 1, 100_000)
    shots_0_1 = np.random.normal(mu1, 1, 10_000)
    shots_0 = np.concatenate([shots_0_0, shots_0_1], axis=0)
    pdf = lambda x: 0.9 * np.exp(-0.5 * (x - mu0) ** 2) / np.sqrt(
        2 * np.pi
    ) + 0.1 * np.exp(-0.5 * (x - mu1) ** 2) / np.sqrt(2 * np.pi)

    plot_pdf_projected(ax, shots_0, pdf)

    if show_figures:
        plt.show()
    plt.close()

    return
