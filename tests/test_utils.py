import pytest
import numpy as np

from iq_readout.utils import check_2d_input, get_angle, rotate_data, from_complex


def test_check_2d_input():
    correct = np.zeros((4, 3, 2))
    incorrect = np.zeros((4, 3, 4))

    check_2d_input(correct)

    with pytest.raises(ValueError):
        check_2d_input(correct, axis=1)

    with pytest.raises(ValueError):
        check_2d_input(incorrect)

    return


def test_get_angle():
    y_axis = np.array([0, 1])
    angle = get_angle(y_axis)

    assert pytest.approx(angle) == np.pi / 2

    return


def test_rotate_data():
    y_axis = np.array([0, 1])
    angle = get_angle(y_axis)
    rot_data = rotate_data(y_axis, -angle)

    assert pytest.approx(rot_data) == np.array([1, 0])

    return


def test_from_complex():
    complex_shots = np.arange(12, dtype=np.complex64).reshape(1, 3, 4)

    shots = from_complex(complex_shots)

    assert (complex_shots.real == shots[..., 0]).all()
    assert (complex_shots.imag == shots[..., 1]).all()

    non_complex_shots = np.arange(12).reshape(1, 3, 4)
    with pytest.raises(TypeError):
        from_complex(non_complex_shots)

    with pytest.raises(TypeError):
        from_complex(None)

    return
