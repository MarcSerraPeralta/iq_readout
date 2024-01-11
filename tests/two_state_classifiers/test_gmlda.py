from copy import deepcopy

import numpy as np
import pytest

from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit

PARAMS = {
    0: {"mu_0": 0.0, "mu_1": 0.1, "sigma": 0.01, "angle": 1.4},
    1: {"mu_0": 0.0, "mu_1": 0.1, "sigma": 0.01, "angle": 0.2},
    "rot_angle": np.pi / 4,
    "threshold": 0.05,
    "rot_shift": 0.0,
}


def test_TwoStateLinearClassifierFit():
    N, M = 100_000, 150_000
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

    cla = TwoStateLinearClassifierFit().fit(shots_0, shots_1)

    params = cla.params()

    assert pytest.approx(params["rot_angle"], rel=2e-2) == np.pi / 4
    assert pytest.approx(params["threshold"], rel=2e-2) == np.sqrt(2)
    assert pytest.approx(params["rot_shift"], abs=2e-2) == -np.sqrt(2) / 2

    assert pytest.approx(params[0]["mu_0"], abs=2e-2) == np.sqrt(2) / 2
    assert pytest.approx(params[0]["mu_1"], rel=2e-2) == 3 * np.sqrt(2) / 2
    assert pytest.approx(params[0]["sigma"], rel=2e-2) == np.sqrt(0.3)
    assert pytest.approx(params[0]["angle"], rel=2e-2) == np.arcsin(np.sqrt(p0))
    assert pytest.approx(params[1]["angle"], rel=2e-2) == np.arccos(np.sqrt(p1))
    return


def test_load():
    cla = TwoStateLinearClassifierFit().load(PARAMS)
    assert cla.params() == PARAMS

    with pytest.raises(ValueError) as e_info:
        cla = TwoStateLinearClassifierFit().load({})

    params = deepcopy(PARAMS)
    params["rot_angle"] = None
    with pytest.raises(ValueError) as e_info:
        cla = TwoStateLinearClassifierFit().load(params)
    return


def test_pdfs():
    params = {
        0: {"mu_0": 0.0, "mu_1": 1.0, "sigma": 0.5, "angle": 1.4},
        1: {"mu_0": 0.0, "mu_1": 1.0, "sigma": 0.5, "angle": 0.2},
        "rot_angle": np.pi / 2,
        "threshold": 0.5,
        "rot_shift": 0.3,
    }
    cla = TwoStateLinearClassifierFit().load(params)

    dx = 0.01
    x0 = np.arange(-3, 3, dx)
    x1 = np.arange(-5, 5, dx)

    xx0, xx1 = np.meshgrid(x0, x1, indexing="ij")
    xxx = np.concatenate([xx0[..., np.newaxis], xx1[..., np.newaxis]], axis=-1)
    pdf_0, pdf_1 = cla.pdf_0(xxx), cla.pdf_1(xxx)
    assert pytest.approx(np.sum(pdf_0) * dx**2, rel=1e-3) == 1
    assert pytest.approx(np.sum(pdf_1) * dx**2, rel=1e-3) == 1

    xx = np.concatenate(
        [np.zeros_like(x1)[..., np.newaxis], x1[..., np.newaxis]], axis=-1
    )  # because rotation angle is np.pi / 2
    pdf_0_proj, pdf_1_proj = cla.pdf_0_projected(xx), cla.pdf_1_projected(xx)
    assert pytest.approx(np.sum(pdf_0_proj) * dx, rel=1e-3) == 1
    assert pytest.approx(np.sum(pdf_1_proj) * dx, rel=1e-3) == 1

    assert pytest.approx(pdf_0.sum(axis=0) * dx, abs=1e-3) == pdf_0_proj
    assert pytest.approx(pdf_1.sum(axis=0) * dx, abs=1e-3) == pdf_1_proj

    return


def test_prediction():
    params = {
        0: {"mu_0": 0.0, "mu_1": 1.0, "sigma": 0.1, "angle": np.pi / 2},
        1: {"mu_0": 0.0, "mu_1": 1.0, "sigma": 0.1, "angle": 0},
        "rot_angle": np.pi / 2,  # blob 1 is above blob 0
        "threshold": 0.5,
        "rot_shift": 1.0,
    }
    cla = TwoStateLinearClassifierFit().load(params)
    x = np.array([[0, 2]])  # blob 1
    assert cla.predict(x) == np.array([1])
    x = np.array([[0, -2]])  # blob 0
    assert cla.predict(x) == np.array([0])
    return


def test_n_bins():
    shots_0, shots_1 = np.zeros((10, 2)), np.zeros((10, 2))
    with pytest.raises(ValueError) as e_info:
        cla = TwoStateLinearClassifierFit().fit(shots_0, shots_1, n_bins=[1, 1])
    return
