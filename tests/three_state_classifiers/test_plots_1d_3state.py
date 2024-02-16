import numpy as np
import matplotlib.pyplot as plt

from iq_readout.three_state_classifiers.plots_1d import plot_pdf_projected


def test_plot_pdf_projected():
    fig, ax = plt.subplots()

    mu0, mu1 = np.array([[0, 0], [0, 4]])
    shots_0_0 = np.random.multivariate_normal(mu0, np.eye(2), 100_000)
    shots_0_1 = np.random.multivariate_normal(mu1, np.eye(2), 10_000)
    shots_0 = np.concatenate([shots_0_0, shots_0_1], axis=0)
    pdf = lambda x: 0.9 * np.exp(-0.5 * (x[..., 0] ** 2 + x[..., 1] ** 2)) / (
        2 * np.pi
    ) + 0.1 * np.exp(-0.5 * ((x[..., 0] - mu1[0]) ** 2 + (x[..., 1] - mu1[1]) ** 2)) / (
        2 * np.pi
    )

    plot_pdf_projected(ax, [mu0, mu1], shots_0, pdf)

    # plt.show()
    plt.close()

    return
