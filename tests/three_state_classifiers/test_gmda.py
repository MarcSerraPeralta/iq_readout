from copy import deepcopy

import numpy as np
import pytest

from iq_readout.three_state_classifiers import GaussMixClassifier

means_cov = {
    "mu_0_x": 0,
    "mu_0_y": 0,
    "mu_1_x": 1,
    "mu_1_y": 1,
    "mu_2_x": 2,
    "mu_2_y": 2,
    "sigma": 0.1,
}
PARAMS = {
    0: {**means_cov, "angle1": np.pi / 2, "angle2": 0},
    1: {**means_cov, "angle1": np.pi / 2, "angle2": np.pi / 2},
    2: {**means_cov, "angle1": 0, "angle2": 0},
}


def test_GaussMixClassifier():
    N, M, P = 100_000, 150_000, 125_000
    mu0, mu1, mu2 = np.array([0, 0]), np.array([1, 1]), np.array([0, 1])
    cov = np.array([[0.3, 0], [0, 0.3]])
    p0, p1, p2 = (0.7, 0.85), (0.15, 0.85), (0.15, 0.3)

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

    params = cla.params

    for k in range(3):
        assert pytest.approx(params[k]["mu_0_x"], abs=5e-2) == 0
        assert pytest.approx(params[k]["mu_0_y"], abs=5e-2) == 0
        assert pytest.approx(params[k]["mu_1_x"], rel=5e-2) == 1
        assert pytest.approx(params[k]["mu_1_y"], rel=5e-2) == 1
        assert pytest.approx(params[k]["mu_2_x"], abs=5e-2) == 0
        assert pytest.approx(params[k]["mu_2_y"], rel=5e-2) == 1
        assert pytest.approx(params[k]["sigma"], rel=5e-2) == np.sqrt(0.3)

    a1, a2 = params[0]["angle1"], params[0]["angle2"]
    assert np.sin(a1) ** 2 * np.cos(a2) ** 2 == pytest.approx(p0[0], abs=5e-2)
    assert np.cos(a1) ** 2 == pytest.approx(1 - p0[1], abs=5e-2)
    a1, a2 = params[1]["angle1"], params[1]["angle2"]
    assert np.sin(a1) ** 2 * np.cos(a2) ** 2 == pytest.approx(p1[0], abs=5e-2)
    assert np.cos(a1) ** 2 == pytest.approx(1 - p1[1], abs=5e-2)
    a1, a2 = params[2]["angle1"], params[2]["angle2"]
    assert np.sin(a1) ** 2 * np.cos(a2) ** 2 == pytest.approx(p2[0], abs=6e-2)
    assert np.cos(a1) ** 2 == pytest.approx(1 - p2[1], abs=6e-2)

    return


def test_load():
    cla = GaussMixClassifier(PARAMS)
    assert cla.params == PARAMS

    with pytest.raises(ValueError) as e_info:
        cla = GaussMixClassifier({})

    return


def test_from_to_yaml(tmp_path):
    cla = GaussMixClassifier(PARAMS)
    cla.to_yaml(tmp_path / "clf.yaml")

    cla_loaded = GaussMixClassifier.from_yaml(tmp_path / "clf.yaml")

    assert cla.params == cla_loaded.params

    return


def test_pdfs():
    cla = GaussMixClassifier(PARAMS)

    dx = 0.01
    x0 = np.arange(-3, 3, dx)
    x1 = np.arange(-5, 5, dx)

    xx0, xx1 = np.meshgrid(x0, x1, indexing="ij")
    xxx = np.concatenate([xx0[..., np.newaxis], xx1[..., np.newaxis]], axis=-1)
    pdf_0, pdf_1, pdf_2 = cla.pdf_0(xxx), cla.pdf_1(xxx), cla.pdf_2(xxx)
    assert pytest.approx(np.sum(pdf_0) * dx**2, rel=1e-3) == 1
    assert pytest.approx(np.sum(pdf_1) * dx**2, rel=1e-3) == 1
    assert pytest.approx(np.sum(pdf_2) * dx**2, rel=1e-3) == 1

    return


def test_prediction():
    cla = GaussMixClassifier(PARAMS)
    x = np.array([[0.1, 0.1]])
    assert cla.predict(x) == np.array([0])
    x = np.array([[1.1, 1.1]])
    assert cla.predict(x) == np.array([1])
    x = np.array([[2.1, 2.1]])
    assert cla.predict(x) == np.array([2])
    return


def test_n_bins():
    shots_0, shots_1, shots_2 = np.zeros((10, 2)), np.zeros((10, 2)), np.zeros((10, 2))
    with pytest.raises(ValueError) as e_info:
        cla = GaussMixClassifier.fit(shots_0, shots_1, shots_2, n_bins=2)
    return
