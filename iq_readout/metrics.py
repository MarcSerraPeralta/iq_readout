"""Readout metrics.
"""

import numpy as np


def get_probs_prep_meas(
    classifier,
    shots_0: np.ndarray,
    shots_1: np.ndarray,
    shots_2: np.ndarray = None,
) -> np.ndarray:
    """
    Returns matrix whose element i,j corresponds to:
    p(measure state j | prepared state i)

    Parameters
    ----------
    classifier
        A classifier from the iq_readout package
    shots_0: np.ndarray(N, 2)
        Data when preparaing state 0
    shots_1: np.ndarray(N, 2)
        Data when preparaing state 1
    shots_2: np.ndarray(N, 2)
        Data when preparaing state 2

    Returns
    -------
    probs: np.ndarray(2,2) or np.ndarray(3,3)
        The size of the array depends on the number of states that
        the classifier can discriminate.
    """
    if set(["pdf_0", "pdf_1"]) > set(dir(classifier)):
        raise ValueError(
            "'classifier' must have the following methods: "
            f"'pdf_0', 'pdf_1'; but it has {dir(classifier)}"
        )
    if "pdf_2" in dir(classifier) and (shots_2 is None):
        raise ValueError("For 3-state classifiers, one must specify 'shots_2'")

    states = [0, 1]
    shots = [shots_0, shots_1]
    probs = np.zeros((2, 2))
    if "pdf_2" in dir(classifier):
        states = [0, 1, 2]
        shots = [shots_0, shots_1, shots_2]
        probs = np.zeros((3, 3))

    for i, (state, shot) in enumerate(zip(states, shots)):
        prediction = classifier.predict(shot)
        for j, _ in enumerate(states):
            probs[i, j] = np.average(prediction == j)

    return probs
