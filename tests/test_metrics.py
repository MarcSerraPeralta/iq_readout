import numpy as np
import pytest

from iq_readout.two_state_classifiers import GaussMixLinearClassifier
from iq_readout.metrics import get_probs_prep_meas


def test_get_probs_prep_meas():
    N, M = 100_000, 150_000
    mu0, mu1 = np.array([0, 0]), np.array([1, 0])
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

    PARAMS = {
        0: {
            "mu_0_x": 0,
            "mu_0_y": 0,
            "mu_1_x": 1,
            "mu_1_y": 0,
            "sigma": np.sqrt(0.3),
            "angle": 1.4,
        },
        1: {
            "mu_0_x": 0,
            "mu_0_y": 0,
            "mu_1_x": 1,
            "mu_1_y": 0,
            "sigma": np.sqrt(0.3),
            "angle": 0.2,
        },
    }

    cla = GaussMixLinearClassifier(PARAMS)

    probs = get_probs_prep_meas(cla, shots_0, shots_1)

    assert probs.shape == (2, 2)
    assert probs[1, 0] > probs[0, 1]
    assert pytest.approx(probs.sum(axis=1)) == np.ones(2)

    return
