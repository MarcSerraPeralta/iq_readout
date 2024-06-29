from copy import deepcopy
import numpy as np
import pytest

from iq_readout.two_state_classifiers import MaxFidLinearClassifier


PARAMS = {
    0: {
        "bins_x": np.array([0, 1]),
        "bins_y": np.array([1, 2]),
        "pdf_values": np.array([[0.5, 0.5], [0, 0]]),
    },
    1: {
        "bins_x": np.array([0, 1]),
        "bins_y": np.array([1, 2]),
        "pdf_values": np.array([[0, 0], [0.5, 0.5]]),
    },
}


def test_MaxFidLinearClassifier():
    N, M = 100_000, 150_000
    mu_0, mu_1 = np.array([1, 1]), np.array([2, 2])
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

    cla = MaxFidLinearClassifier.fit(shots_0, shots_1)

    params = cla.params
    assert set(params[0].keys()) == set(["bins_x", "bins_y", "pdf_values"])
    assert set(params[1].keys()) == set(["bins_x", "bins_y", "pdf_values"])
    for state in range(2):
        norm = (
            params[state]["pdf_values"].sum()
            * np.diff(params[state]["bins_x"])[0]
            * np.diff(params[state]["bins_y"])[0]
        )
        assert pytest.approx(norm) == 1

    params_proj = cla.params_proj
    for state in range(2):
        print(params_proj[state]["pdf_values"].sum())
        print(np.diff(params_proj[state]["bins"])[0])
        norm = (
            params_proj[state]["pdf_values"].sum()
            * np.diff(params_proj[state]["bins"])[0]
        )
        assert pytest.approx(norm) == 1

    statistics = cla.statistics
    assert pytest.approx(statistics["rot_angle"], rel=2e-2) == np.pi / 4
    assert pytest.approx(statistics["threshold"], rel=2e-2) == 1.5 * np.sqrt(2)

    return


def test_load():
    cla = MaxFidLinearClassifier(PARAMS)
    assert cla.params == PARAMS

    with pytest.raises(ValueError) as e_info:
        cla = MaxFidLinearClassifier({})

    params = deepcopy(PARAMS)
    params["bins_x"] = [4, 6]
    with pytest.raises(ValueError) as e_info:
        cla = MaxFidLinearClassifier(params)
    return


def test_from_to_yaml(tmp_path):
    cla = MaxFidLinearClassifier(PARAMS)
    cla.to_yaml(tmp_path / "clf.yaml")

    cla_loaded = MaxFidLinearClassifier.from_yaml(tmp_path / "clf.yaml")

    assert cla.params.keys() == cla_loaded.params.keys()
    # the params contain numpy arrays
    for s in cla.params.keys():
        assert cla.params[s].keys() == cla_loaded.params[s].keys()
        for k in cla.params[s].keys():
            assert np.array(cla.params[s][k] == cla_loaded.params[s][k]).all()
    return


def test_prediction():
    params = deepcopy(PARAMS)
    cla = MaxFidLinearClassifier(params)

    x = np.array([[1, 1]])  # blob 1
    assert cla.predict(x) == np.array([1])
    x = np.array([[0, 0]])  # blob 0
    assert cla.predict(x) == np.array([0])

    return


def test_n_bins():
    shots_0, shots_1 = np.zeros((10, 2)), np.zeros((10, 2))
    with pytest.raises(ValueError) as e_info:
        cla = MaxFidLinearClassifier.fit(shots_0, shots_1, n_bins=[1, 1])
    return
