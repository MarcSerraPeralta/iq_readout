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
