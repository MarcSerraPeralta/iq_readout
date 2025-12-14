"""Readout metrics."""

from typing import List

import numpy as np

from .classifiers import Classifier


def get_probs_prep_meas(
    classifiers: List[Classifier] | Classifier, *shots: np.ndarray
) -> np.ndarray:
    """
    Returns matrix whose element ``probs[s,o]`` corresponds to:
    p(measured bitstring o | prepared bitstring s),
    with s and o in the corresponding base 2, 3 or 10.

    Parameters
    ----------
    classifiers
        A list of classifiers from the iq_readout package.
        It can also be a single classifier.
    shots: list(np.ndarray(num_qubits, num_shots, 2))
        Tuple of shots when preparing all the possible bistrings.
        The bitstrings should be sorted in the *standard binary order*, e.g.
        ``000``, ``001``, ``010``, ``011``, ...
        In the case that ``classifiers`` is a single classifier (not a list),
        each element of this variable can have shape ``(num_shots, 2)``.

    Returns
    -------
    probs: np.ndarray(2**num_qubits, 2**num_qubits) | np.ndarray(3**num_qubits, 3**num_qubits)
        The size of the array depends on the number of states that
        the classifier can discriminate.
    """
    if isinstance(classifiers, Classifier):
        classifiers = [classifiers]
        shots = [np.array([shots_k]) for shots_k in shots]
    if not (isinstance(classifiers, list) or isinstance(classifiers, tuple)):
        raise TypeError(
            f"'classifiers' must be a list or a tuple, but {type(classifiers)} was given."
        )
    num_qubits = len(classifiers)
    if not isinstance(classifiers[0], Classifier):
        raise TypeError(
            f"The given object is not a classifier, it is a {type(classifiers[0])}."
        )
    num_states = classifiers[0]._num_states
    for clf in classifiers[1:]:
        if not isinstance(clf, Classifier):
            raise TypeError(
                f"The given object is not a classifier, it is a {type(clf)}."
            )
        if clf._num_states != num_states:
            raise TypeError(
                "All given classifiers must have the same number of states."
            )

    if len(shots) != num_states**num_qubits:
        raise ValueError(
            "The number of bitstrings in 'shots' must be num_states**num_qubits, "
            f"but {len(shots)} != {num_states}**{num_qubits} was given."
        )
    for k, shots_k in enumerate(shots):
        if len(shots_k) != num_qubits:
            raise ValueError(
                f"Each bitstring must include all qubits ({num_qubits}), "
                f"but only {len(shots_k)} were given for bitstring index {k}."
            )

    probs = np.zeros((num_states**num_qubits, num_states**num_qubits))
    powers = num_states ** np.arange(num_qubits)[::-1]  # to convert to base 10
    for k, shots_k in enumerate(shots):
        # shots_k.shape = (num_qubits, num_shots, 2)
        num_total = shots_k.shape[1]
        outcomes = np.array(
            [clf.predict(shots_k[q]) for q, clf in enumerate(classifiers)]
        )
        unique, counts = np.unique(outcomes.T, axis=0, return_counts=True)
        for vec, count in zip(unique, counts):
            idx = np.sum(vec * powers)  # base 2 or 3 conversion to base 10
            probs[k, idx] = count / num_total

    return probs
