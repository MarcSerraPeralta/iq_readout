import numpy as np


def check_2d_input(x: np.ndarray, axis=-1):
    if x.shape[axis] != 2:
        raise ValueError(
            "input must be specified with shape (..., 2), "
            f"but the following shape was given: {x.shape}"
        )
    if axis != -1:
        if len(x.shape) != axis + 1:
            raise ValueError(
                f"input must have {axis+1} axis, "
                f"but the following shape was given: {x.shape}"
            )
    return


def get_angle(vector: np.ndarray) -> float:
    """
    The counterclockwise angle from the x-axis in the range (-pi, pi]

    Parameters
    ----------
    vector: np.ndarray(2)
    """
    assert vector.shape == (2,)

    angle = np.arctan2(*vector[::-1])  # arctan(y/x)
    return angle


def rotate_data(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Counterclock-wise rotation of x by an angle theta

    Parameters
    ----------
    x: np.ndarray(N, 2)
    theta: float

    Returns
    -------
    output: np.ndarray(N, 2)
    """
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.einsum("ij,...j->...i", rot, x)
