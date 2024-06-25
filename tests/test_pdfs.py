from inspect import signature
import numpy as np

from iq_readout import pdfs


def generate_parameters(funct):
    # assumes that all PDF functions follow "funct(x, *args)"
    params = signature(funct).parameters
    my_params = {}

    # pdfs built from hist use args which are not float
    if "pdf_from_hist1d" == funct.__name__:
        params = {
                "bins": np.linspace(-1, 2, 10),
                "pdf_values": np.ones(10)/3,
        }
        return params
    if "pdf_from_hist2d" == funct.__name__:
        params = {
                "bins_x": np.linspace(-1, 2, 10),
                "bins_y": np.linspace(-2, 1, 20),
                "pdf_values": np.ones((10, 20))/9,
        }
        return params

    # need to set mu_0 to avoid division by 0 errors in PDF from decay
    for key, var_type in list(params.items())[1:]:
        if "mu_0" in key:
            my_params[key] = 0
        else:
            my_params[key] = 1

    return my_params


def is_normalized_1d(funct):
    dx = 0.01
    x = np.arange(-10, 10, dx)
    args = generate_parameters(funct)

    norm = np.sum(funct(x, **args)) * dx
    return np.isclose(norm, 1)


def is_normalized_2d(funct):
    dx = 0.01
    x = np.arange(-10, 10, dx)
    xx, yy = np.meshgrid(x, x)
    zz = np.concatenate([xx[..., np.newaxis], yy[..., np.newaxis]], axis=-1)
    args = generate_parameters(funct)

    norm = np.sum(funct(zz, **args)) * dx**2
    return np.isclose(norm, 1)


def test_pdf_normalized():
    IMPORTED_FUNCTS = ["erf", "check_2d_input", "rotate_data", "get_angle"]

    for name, pdf in pdfs.__dict__.items():
        if not callable(pdf) or name in IMPORTED_FUNCTS:
            continue
        if "1d" in name:
            assert is_normalized_1d(pdf)
        elif "2d" in name:
            assert is_normalized_2d(pdf)
        else:
            raise ValueError(f"{name} does not contain '1d' or '2d'")
    return
