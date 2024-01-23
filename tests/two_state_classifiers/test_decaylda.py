from copy import deepcopy

import numpy as np
import pytest

from iq_readout.two_state_classifiers import DecayLinearClassifierFit

PARAMS = {
    0: {"mu_0": 0.0, "mu_1": 0.1, "sigma": 0.01, "angle": 1.4},
    1: {"mu_0": 0.0, "mu_1": 0.1, "sigma": 0.01, "angle": 0.2, "t1_norm": 0.2},
    "rot_angle": np.pi / 4,
    "threshold": 0.05,
    "rot_shift": 0.0,
}


def test_DecayLinearClassifierFit():
    N, M = 150_000, 200_000
    mu0, mu1 = np.array([1, 0]), np.array([2, 1])
    cov = np.array([[0.3, 0], [0, 0.3]])
    p0, p1 = 0.8, 0.8

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

    cla = DecayLinearClassifierFit().fit(shots_0, shots_1)

    params = cla.params()

    assert pytest.approx(params["rot_angle"], rel=2e-2) == np.pi / 4
    # assert pytest.approx(params["threshold"], rel=2e-2) == np.sqrt(2)
    assert pytest.approx(params["rot_shift"], abs=2e-2) == -np.sqrt(2) / 2

    assert pytest.approx(params[0]["mu_0"], abs=1e-1) == np.sqrt(2) / 2
    assert pytest.approx(params[0]["mu_1"], rel=1e-1) == 3 * np.sqrt(2) / 2
    assert pytest.approx(params[0]["sigma"], rel=1e-1) == np.sqrt(0.3)

    assert pytest.approx(params[1]["mu_0"], abs=1e-1) == np.sqrt(2) / 2
    # assert pytest.approx(params[1]["mu_1"], rel=1e-1) == 3 * np.sqrt(2) / 2
    # assert pytest.approx(params[1]["sigma"], rel=1e-1) == np.sqrt(0.3)

    return


def test_load():
    cla = DecayLinearClassifierFit().load(PARAMS)
    assert cla.params() == PARAMS

    with pytest.raises(ValueError) as e_info:
        cla = DecayLinearClassifierFit().load({})

    params = deepcopy(PARAMS)
    params["rot_angle"] = None
    with pytest.raises(ValueError) as e_info:
        cla = DecayLinearClassifierFit().load(params)
    return


def test_pdfs():
    params = {
        0: {"mu_0": 0.0, "mu_1": 1.0, "sigma": 0.5, "angle": 1.4},
        1: {"mu_0": 0.0, "mu_1": 1.0, "sigma": 0.5, "angle": 0.2, "t1_norm": 0.2},
        "rot_angle": np.pi / 2,
        "threshold": 0.5,
        "rot_shift": 0.3,
    }
    cla = DecayLinearClassifierFit().load(params)

    dx = 0.01
    x0 = np.arange(-5, 7, dx)
    x1 = np.arange(-5, 7, dx)

    x_proj = x1  # because rotation angle is np.pi / 2
    pdf_0_proj, pdf_1_proj = cla.pdf_0_projected(x_proj), cla.pdf_1_projected(x_proj)
    assert pytest.approx(np.sum(pdf_0_proj) * dx, rel=1e-3) == 1
    assert pytest.approx(np.sum(pdf_1_proj) * dx, rel=1e-3) == 1

    return


def test_prediction():
    params = {
        0: {"mu_0": 0.0, "mu_1": 1.0, "sigma": 0.1, "angle": np.pi / 2},
        1: {"mu_0": 0.0, "mu_1": 1.0, "sigma": 0.1, "angle": 0.2, "t1_norm": 0.10},
        "rot_angle": np.pi / 2,  # blob 1 is above blob 0
        "threshold": 0.5,
        "rot_shift": 1.0,
    }
    cla = DecayLinearClassifierFit().load(params)
    x = np.array([[0, 2]])  # blob 1
    assert cla.predict(x) == np.array([1])
    x = np.array([[0, -2]])  # blob 0
    assert cla.predict(x) == np.array([0])
    return


def test_n_bins():
    shots_0, shots_1 = np.zeros((10, 2)), np.zeros((10, 2))
    with pytest.raises(ValueError) as e_info:
        cla = DecayLinearClassifierFit().fit(shots_0, shots_1, n_bins=[1, 1])
    return
