import numpy as np
import matplotlib.pyplot as plt

from iq_readout.two_state_classifiers import GaussMixLinearClassifier
from iq_readout.two_state_classifiers import summary


def test_summary():
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

    cla = GaussMixLinearClassifier().fit(shots_0, shots_1)

    fig = summary(cla, shots_0, shots_1)

    # plt.show()
    plt.close()

    return
