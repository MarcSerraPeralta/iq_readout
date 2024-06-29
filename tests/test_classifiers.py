import numpy as np

from iq_readout import classifiers


GENERAL_ATTRS = ["to_yaml", "from_yaml", "params", "statistics", "fit", "predict"]


def test_TwoStateLinearClassifier():
    ATTRIBUTES = GENERAL_ATTRS + [
        "pdf_0",
        "pdf_1",
        "pdf_0_projected",
        "pdf_1_projected",
        "params_proj",
        "project",
    ]
    assert set(dir(classifiers.TwoStateLinearClassifier)) >= set(ATTRIBUTES)
    return


def test_TwoStateClassifier():
    ATTRIBUTES = GENERAL_ATTRS + ["pdf_0", "pdf_1"]
    assert set(dir(classifiers.TwoStateClassifier)) >= set(ATTRIBUTES)
    return


def test_ThreeStateClassifier():
    ATTRIBUTES = GENERAL_ATTRS + ["pdf_0", "pdf_1", "pdf_2"]
    assert set(dir(classifiers.ThreeStateClassifier)) >= set(ATTRIBUTES)
    return

def test_to_from_yaml(tmp_path):
    clfs = [
        classifiers.TwoStateLinearClassifier, 
        classifiers.TwoStateClassifier,
        classifiers.ThreeStateClassifier,
    ]

    for clf in clfs:
        clf._param_names = {i: ["a"] for i in range(clf._num_states)}
        setattr(clf, "statistics", {"p": np.array([1., 2.])})
        clf_1 = clf({i: {"a": np.core.multiarray.scalar(np.dtype(np.float64))} for i in range(clf._num_states)})
        clf_1.to_yaml(tmp_path / "clf.yaml")

        with open(tmp_path / "clf.yaml", "r") as file:
            data = file.read()
        assert "numpy" not in data

        clf_2 = clf.from_yaml(tmp_path / "clf.yaml")

        assert clf_1.params == clf_2.params
        
