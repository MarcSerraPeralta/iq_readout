import numpy as np
import matplotlib.pyplot as plt

from iq_readout.three_state_classifiers import GaussMixClassifier
from iq_readout.two_state_classifiers import GaussMixLinearClassifier
from iq_readout.plots import summary


def test_summary_two_state(show_figures):
    N, M = 100_000, 150_000
    mu0, mu1 = np.array([0, 0]), np.array([2, 0])
    cov = np.array([[0.3, 0], [0, 0.3]])
    p0, p1 = 1, 0.8

    change0, change1 = np.random.rand(N, 1), np.random.rand(M, 1)
    shots_0 = np.where(
        np.tile(change0, 2) < p0,
        np.random.multivariate_normal(mu0, cov, size=N),
        np.random.multivariate_normal(mu1, cov, size=N),
    )
    shots_1 = np.where(
        np.tile(change1, 2) < p1,
        np.random.multivariate_normal(mu1, cov, size=M),
        np.random.multivariate_normal(mu0, cov, size=M),
    )

    cla = GaussMixLinearClassifier.fit(shots_0, shots_1)

    fig = summary(cla, shots_0, shots_1)

    if show_figures:
        plt.show()
    plt.close()

    return


def test_summary_three_state(show_figures):
    N, M, P = 100_000, 150_000, 125_000
    mu0, mu1, mu2 = np.array([0, 0]), np.array([1, 0]), np.array([1, 1])
    cov = np.array([[0.05, 0], [0, 0.05]])
    p0, p1, p2 = (1, 1), (0.02, 1), (0.02, 0.05)

    change0, change1, change2 = (
        np.tile(np.random.rand(N, 1), 2),
        np.tile(np.random.rand(M, 1), 2),
        np.tile(np.random.rand(P, 1), 2),
    )
    shots_0 = np.where(
        change0 < p0[0],
        np.random.multivariate_normal(mu0, cov, size=N),
        np.random.multivariate_normal(mu1, cov, size=N),
    )
    shots_0 = np.where(
        change0 > p0[1],
        np.random.multivariate_normal(mu2, cov, size=N),
        shots_0,
    )
    shots_1 = np.where(
        change1 < p1[0],
        np.random.multivariate_normal(mu0, cov, size=M),
        np.random.multivariate_normal(mu1, cov, size=M),
    )
    shots_1 = np.where(
        change1 > p1[1],
        np.random.multivariate_normal(mu2, cov, size=M),
        shots_1,
    )
    shots_2 = np.where(
        change2 < p2[0],
        np.random.multivariate_normal(mu0, cov, size=P),
        np.random.multivariate_normal(mu1, cov, size=P),
    )
    shots_2 = np.where(
        change2 > p2[1],
        np.random.multivariate_normal(mu2, cov, size=P),
        shots_2,
    )

    cla = GaussMixClassifier.fit(shots_0, shots_1, shots_2)

    fig = summary(cla, shots_0, shots_1, shots_2)

    if show_figures:
        plt.show()
    plt.close()

    return
