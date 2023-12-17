import numpy as np
import pytest

from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit

PARAMS = {
    0: [0.0, 0.1, 0.01, 1.4],
    1: [0.0, 0.1, 0.01, 0.4],
    "rot_angle": np.pi / 4,
    "threshold": 0.05,
}


def test_TwoStateLinearClassifierFit():
    N, M = 100_000, 150_000
    mu0, mu1 = np.array([0, 0]), np.array([1, 1])
    cov = np.array([[0.3, 0], [0, 0.3]])
    p0, p1 = 0.99, 0.95

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

    cla = TwoStateLinearClassifierFit().fit(shots_0, shots_1)

    params = cla.params()

    assert pytest.approx(params["rot_angle"], rel=1e-2) == np.pi / 4
    assert pytest.approx(params["threshold"], rel=1e-2) == np.sqrt(2) / 2
    assert (
        pytest.approx(params[0]["mu_0"], abs=1e-2) == 0
    )  # relative comparison with 0 is not correct
    assert pytest.approx(params[0]["mu_1"], rel=1e-2) == np.sqrt(2)
    assert pytest.approx(params[0]["sigma"], rel=1e-2) == np.sqrt(0.3)
    assert pytest.approx(params[0]["angle"], rel=1e-2) == np.arcsin(np.sqrt(p0))
    assert pytest.approx(params[1]["angle"], rel=1e-2) == np.arccos(np.sqrt(p1))
    return


def test_load():
    return


def test_pdfs():
    return


def test_prediction():
    return
