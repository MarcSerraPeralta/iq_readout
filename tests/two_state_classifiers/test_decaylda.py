from copy import deepcopy

import numpy as np
import pytest

from iq_readout.two_state_classifiers import DecayLinearClassifier

PARAMS = {
    0: {
        "mu_0_x": 0,
        "mu_0_y": 0,
        "mu_1_x": 1,
        "mu_1_y": 1,
        "sigma": 0.1,
        "angle": 1.4,
    },
    1: {
        "mu_0_x": 0,
        "mu_0_y": 0,
        "mu_1_x": 1,
        "mu_1_y": 1,
        "sigma": 0.1,
        "angle": 0.2,
        "t1_norm": 100,
    },
}


def test_DecayLinearClassifier():
    N, M = 150_000, 200_000
    mu_0, mu_1 = np.array([1, 0]), np.array([2, 1])
    cov = np.array([[0.3, 0], [0, 0.3]])
    p0, p1 = 0.8, 0.8

    change0, change1 = np.random.rand(N, 1), np.random.rand(M, 1)
    shots_0 = np.where(
        np.tile(change0, 2) < p0,
        np.random.multivariate_normal(mu_0, cov, size=N),
        np.random.multivariate_normal(mu_1, cov, size=N),
    )
    shots_1 = np.where(
        np.tile(change1, 2) < p1,
        np.random.multivariate_normal(mu_1, cov, size=M),
        np.random.multivariate_normal(mu_0, cov, size=M),
    )

    cla = DecayLinearClassifier.fit(shots_0, shots_1)

    params = cla.params

    assert pytest.approx(params[0]["mu_0_x"], abs=5e-2) == mu_0[0]
    assert pytest.approx(params[0]["mu_0_y"], abs=5e-2) == mu_0[1]
    assert pytest.approx(params[1]["mu_1_x"], abs=5e-2) == mu_1[0]
    assert pytest.approx(params[1]["mu_1_y"], abs=5e-2) == mu_1[1]
    assert pytest.approx(params[0]["sigma"], rel=5e-2) == np.sqrt(0.3)
    # same means and sigma
    for name in ["mu_0_x", "mu_0_y", "mu_1_x", "mu_1_y", "sigma"]:
        assert pytest.approx(params[0][name]) == params[1][name]

    params_proj = cla.params_proj

    assert pytest.approx(params_proj[0]["mu_0"], rel=5e-2) == np.sqrt(2) / 2
    assert pytest.approx(params_proj[0]["mu_1"], rel=5e-2) == 3 * np.sqrt(2) / 2
    assert pytest.approx(params_proj[0]["sigma"], rel=5e-2) == np.sqrt(0.3)
    assert pytest.approx(params_proj[0]["angle"], rel=5e-2) == np.arcsin(np.sqrt(p0))

    return


def test_load():
    cla = DecayLinearClassifier(PARAMS)
    assert cla.params == PARAMS

    with pytest.raises(ValueError) as e_info:
        cla = DecayLinearClassifier({})

    params = deepcopy(PARAMS)
    params[0]["angle"] = None
    with pytest.raises(TypeError) as e_info:
        cla = DecayLinearClassifier(params)
    return


def test_pdfs():
    # needs rot angle to be 0
    params = {
        0: {
            "mu_0_x": 0,
            "mu_0_y": 0,
            "mu_1_x": 1,
            "mu_1_y": 0,
            "sigma": 0.1,
            "angle": np.pi / 2,
        },
        1: {
            "mu_0_x": 0,
            "mu_0_y": 0,
            "mu_1_x": 1,
            "mu_1_y": 0,
            "sigma": 0.1,
            "angle": 0,
            "t1_norm": 15,
        },
    }
    cla = DecayLinearClassifier(params)

    dx = 0.01
    x = np.arange(-3, 3, dx)
    y = np.arange(-5, 5, dx)

    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=-1)
    pdf_0, pdf_1 = cla.pdf_0(zz), cla.pdf_1(zz)
    assert pytest.approx(np.sum(pdf_0) * dx**2, rel=1e-3) == 1
    assert pytest.approx(np.sum(pdf_1) * dx**2, rel=1e-3) == 1

    x_proj = x  # because rotation angle is 0
    pdf_0_proj, pdf_1_proj = cla.pdf_0_projected(x_proj), cla.pdf_1_projected(x_proj)
    assert pytest.approx(np.sum(pdf_0_proj) * dx, rel=1e-3) == 1
    assert pytest.approx(np.sum(pdf_1_proj) * dx, rel=1e-3) == 1

    assert pytest.approx(pdf_0.sum(axis=1) * dx, abs=1e-3) == pdf_0_proj
    assert pytest.approx(pdf_1.sum(axis=1) * dx, abs=1e-3) == pdf_1_proj

    return


def test_prediction():
    cla = DecayLinearClassifier(PARAMS)
    x = np.array([[1, 1]])  # blob 1
    assert cla.predict(x) == np.array([1])
    x = np.array([[0, 0]])  # blob 0
    assert cla.predict(x) == np.array([0])
    return


def test_n_bins():
    shots_0, shots_1 = np.zeros((10, 2)), np.zeros((10, 2))
    with pytest.raises(ValueError) as e_info:
        cla = DecayLinearClassifier.fit(shots_0, shots_1, n_bins=[1, 1])
    return
