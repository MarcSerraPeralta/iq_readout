from typing import TypeVar, Type, Union, Dict

import pathlib

import yaml
import numpy as np

from .utils import check_2d_input, rotate_data, get_angle

T2 = TypeVar("T2", bound="TwoStateClassifier")
T3 = TypeVar("T3", bound="ThreeStateClassifier")


class TwoStateClassifier:
    """
    Template for creating two-state classifiers.

    The functions to be rewritten for each specific classifier are:
    * <class attributes>
        specify the pdf functions and parameter names
    * statistics
        compute the statistics
    * fit
        perform the fit

    NB: if the classifier does not use max-likelihood classification,
        then `predict` needs to the overwritten.
    """

    _pdf_func_0 = None
    _pdf_func_1 = None
    # parameter name ordering must match the ordering in the pdf functions
    _param_names = {
        0: [],
        1: [],
    }
    _num_states = 2

    def __init__(self, params: Dict[int, Dict[str, float]]):
        """
        Loads params to `TwoStateClassifier`.

        Parameters
        ----------
        params
            The structure of the dictionary must be:
            {
                0: {"param1": float, ...},
                1: {"param1": float, ...}
            }
        """
        self._check_params(params)

        # param values are stored in a vector to run `curve_fit` easily
        # because it uses `args` and not `kargs` to fit the functions
        self._param_values = {
            state: [params[state][n] for n in self._param_names[state]]
            for state in range(2)
        }

        return

    def to_yaml(self, filename: Union[str, pathlib.Path]):
        """
        Stores parameters to YAML file.
        NB: the file can include extra data, but this data should be removed in
            the `from_yaml` function
        """
        data = {"params": self.params, "extra": self.statistics}

        with open(filename, "w") as file:
            yaml.dump(data, file, default_flow_style=False)
        return

    @classmethod
    def from_yaml(cls: Type[T2], filename: Union[str, pathlib.Path]) -> T2:
        """
        Load `TwoStateClassifier` from YAML file

        NB: this removes any extra data stored in the YAML file from `to_yaml`
        """
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        return cls(data["params"])

    @property
    def params(self) -> Dict[int, Dict[str, float]]:
        """
        Returns the parameters required to set up the classifier.
        The structure of the output dictionary is:
        {
            0: {"param1": float, ...},
            1: {"param1": float, ...}
        }
        """
        params = {}
        for state in range(2):
            params[state] = {
                k: v
                for k, v in zip(self._param_names[state], self._param_values[state])
            }
        return params

    @property
    def statistics(self) -> Dict[str, np.ndarray]:
        """
        Returns dictionary with general statistical data:
        - mu_0: np.array([float, float])
        - mu_1: np.array([float, float])
        - cov_0: np.array([[float, float], [float, float]])
        - cov_1: np.array([[float, float], [float, float]])
        It can also include other information such as rot_angle, rot_shift, ...

        NB: this property is used for plotting and for storing useful
            information in the YAML file
        """
        return {}

    @classmethod
    def fit(cls: Type[T2], shots_0: np.ndarray, shots_1: np.ndarray, **kargs) -> T2:
        """
        Runs fit to the given data

        Parameters
        ----------
        shots_0: np.array(N, 2)
            IQ data when preparing state 0
        shots_1: np.array(M, 2)
            IQ data when preparing state 1

        Returns
        -------
        Loaded `TwoStateClassifier`
        """
        check_2d_input(shots_0, axis=1)
        check_2d_input(shots_1, axis=1)

        # perform fit ...

        params = {}

        return cls(params)

    def predict(self, z: np.ndarray, p_0: float = 1 / 2) -> np.ndarray:
        """
        Classifies the given data to 0 or 1 using maximum likelihood classification:
        - 0 if p(0|z) > p(1|z)
        - 1 otherwise

        Parameters
        ----------
        z: np.array(..., 2)
            Points to classify
        p_0
            Probability to measure outcome 0.
            By default 1/2, thus p(0|z) > p(1|z) is equivalent to p(z|0) > p(z|0)

        Returns
        -------
        prediction: np.array(...)
            Classification of the given data. It only contains 0s and 1s.
        """
        if (p_0 > 1) or (p_0 < 0):
            raise ValueError(
                "The speficied 'p_0' must be a physical probability, "
                f"but p_0={p_0} (and p1={1-p_0}) were given"
            )
        # does not compute p(z) for p(i|z) = p(z|i) * p(i) / p(z)
        # because it is the same for all states and we are
        # only selecting the one with highest probability
        probs = [self.pdf_0(z) * p_0, self.pdf_1(z) * (1 - p_0)]
        return np.argmax(probs, axis=0)

    def pdf_0(self, z: np.ndarray) -> np.ndarray:
        """
        Returns p(z|0)

        Parameters
        ----------
        z: np.array(..., 2)
            IQ points

        Returns
        -------
        prob: np.array(...)
            Probability of the given IQ points
        """
        check_2d_input(z)
        # the pdf functions are class variables (as opposed to instance variables)
        # thus they are available in the class of `self`, not the instance of `self`
        return self.__class__._pdf_func_0(z, *self._param_values[0])

    def pdf_1(self, z: np.ndarray) -> np.ndarray:
        """
        Returns p(z|1)

        Parameters
        ----------
        z: np.array(..., 2)
            IQ points

        Returns
        -------
        prob: np.array(...)
            Probability of the given IQ points
        """
        check_2d_input(z)
        # the pdf functions are class variables (as opposed to instance variables)
        # thus they are available in the class of `self`, not the instance of `self`
        return self.__class__._pdf_func_1(z, *self._param_values[1])

    def _check_params(self, params: Dict[int, Dict[str, float]]):
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be a dict, but {type(params)} was given")
        if set(params) != set([0, 1]):
            raise ValueError(
                f"'params' must have keys [0,1], but {list(params)} were given"
            )

        for state, p in params.items():
            if not isinstance(p, dict):
                raise TypeError(
                    f"'params[{state}]' must be a dict, but {type(p)} was given"
                )
            if set(p) != set(self._param_names[state]):
                raise ValueError(
                    f"'params[{state}]' must have keys {self._param_names[state]}, "
                    f" but {list(p)} were given"
                )

            for key, value in p.items():
                if (not isinstance(value, float)) and (not isinstance(value, int)):
                    raise TypeError(
                        f"'params[{state}][{key}]' must be a float, "
                        f"but {type(value)} was given"
                    )

        return


class TwoStateLinearClassifier(TwoStateClassifier):
    """
    Template for creating two-state linear classifiers.

    The functions to be rewritten for each specific classifier are:
    * <class attributes>
        specify the pdf functions and parameter names
    * statistics
        compute the statistics
    * params_proj
        compute parameters for projected pdf from `params`
    * fit
        perform the fit

    NB: if the classifier does not use max-likelihood classification,
        then `predict` needs to the overwritten.
    """

    _pdf_func_0 = None
    _pdf_func_1 = None
    # parameter name ordering must match the ordering in the pdf functions
    _param_names = {
        0: [],
        1: [],
    }
    _pdf_func_0_proj = None
    _pdf_func_1_proj = None
    # parameter name ordering must match the ordering in the pdf functions
    _param_names_proj = {
        0: [],
        1: [],
    }
    _num_states = 2

    def __init__(self, params: Dict[int, Dict[str, float]]):
        """
        Loads params to `TwoStateLinearClassifier`.

        Parameters
        ----------
        params
            The structure of the dictionary must be:
            {
                0: {"param1": float, ...},
                1: {"param1": float, ...}
            }
        """
        self._check_params(params)

        # param values are stored in a vector to run `curve_fit` easily
        # because it uses `args` and not `kargs` to fit the functions
        self._param_values = {
            state: [params[state][n] for n in self._param_names[state]]
            for state in range(2)
        }

        # compute parameters for the projected pdfs from `params`
        # this step needs to be done after loading the standard `params`
        self._param_values_proj = {
            state: [self.params_proj[state][n] for n in self._param_names_proj[state]]
            for state in range(2)
        }

        return

    def to_yaml(self, filename: Union[str, pathlib.Path]):
        """
        Stores parameters to YAML file.
        NB: the file can include extra data, but this data should be removed in
            the `from_yaml` function
        """
        data = {
            "params": self.params,
            "params_proj": self.params_proj,
            "extra": self.statistics,
        }

        with open(filename, "w") as file:
            yaml.dump(data, file, default_flow_style=False)
        return

    @property
    def params_proj(self) -> Dict[int, Dict[str, float]]:
        """
        Returns the parameters for the projected pdfs, computed
        from `params`.
        The structure of the output dictionary is:
        {
            0: {"param1": float, ...},
            1: {"param1": float, ...}
        }
        """

        # compute `params_proj` from `params` ...

        return params_proj

    def project(self, z: np.ndarray) -> np.ndarray:
        """
        Returns the projection of the given IQ data to
        the mu_0 - mu_1 axis

        Parameters
        ----------
        z: np.array(..., 2)
            IQ points

        Returns
        -------
        z_proj: np.array(...)
            Projection of IQ points to mu_0 - mu_1 axis
        """
        check_2d_input(z)
        mu_0, mu_1 = self.statistics["mu_0"], self.statistics["mu_1"]
        rot_angle = get_angle(mu_1 - mu_0)
        return rotate_data(z, -rot_angle)[..., 0]

    def pdf_0_projected(self, z_proj: np.ndarray) -> np.ndarray:
        """
        Returns p_projected(z_proj|0)

        NB: p_projected(z_proj|0) != p(z|0)

        Parameters
        ----------
        z_proj: np.array(...)
            Projection of IQ points to mu_0 - mu_1 axis.
            See `self.project`.

        Returns
        -------
        prob: np.array(...)
            Probability of the given projected points
        """
        # the pdf functions are class variables (as opposed to instance variables)
        # thus they are available in the class of `self`, not the instance of `self`
        return self.__class__._pdf_func_0_proj(z_proj, *self._param_values_proj[0])

    def pdf_1_projected(self, z_proj: np.ndarray) -> np.ndarray:
        """
        Returns p_projected(z_proj|1)

        NB: p_projected(z_proj|1) != p(z|1)

        Parameters
        ----------
        z_proj: np.array(...)
            Projection of IQ points to mu_0 - mu_1 axis.
            See `self.project`.

        Returns
        -------
        prob: np.array(...)
            Probability of the given projected points
        """
        # the pdf functions are class variables (as opposed to instance variables)
        # thus they are available in the class of `self`, not the instance of `self`
        return self.__class__._pdf_func_1_proj(z_proj, *self._param_values_proj[1])


class ThreeStateClassifier:
    """
    Template for creating three-state classifiers.

    The functions to be rewritten for each specific classifier are:
    * <class attributes>
        specify the pdf functions and parameter names
    * statistics
        compute the statistics
    * fit
        perform the fit

    NB: if the classifier does not use max-likelihood classification,
        then `predict` needs to the overwritten.
    """

    _pdf_func_0 = None
    _pdf_func_1 = None
    _pdf_func_2 = None
    # parameter name ordering must match the ordering in the pdf functions
    _param_names = {
        0: [],
        1: [],
        2: [],
    }
    _num_states = 3

    def __init__(self, params: Dict[int, Dict[str, float]]):
        """
        Loads params to `ThreeStateClassifier`.

        Parameters
        ----------
        params
            The structure of the dictionary must be:
            {
                0: {"param1": float, ...},
                1: {"param1": float, ...},
                2: {"param1": float, ...},
            }
        """
        self._check_params(params)

        # param values are stored in a vector to run `curve_fit` easily
        # because it uses `args` and not `kargs` to fit the functions
        self._param_values = {
            state: [params[state][n] for n in self._param_names[state]]
            for state in range(3)
        }

        return

    def to_yaml(self, filename: Union[str, pathlib.Path]):
        """
        Stores parameters to YAML file.
        NB: the file can include extra data, but this data should be removed in
            the `from_yaml` function
        """
        data = {"params": self.params, "extra": self.statistics}

        with open(filename, "w") as file:
            yaml.dump(data, file, default_flow_style=False)
        return

    @classmethod
    def from_yaml(cls: Type[T3], filename: Union[str, pathlib.Path]) -> T3:
        """
        Load `ThreeStateClassifier` from YAML file

        NB: this removes any extra data stored in the YAML file from `to_yaml`
        """
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        return cls(data["params"])

    @property
    def params(self) -> Dict[int, Dict[str, float]]:
        """
        Returns the parameters required to set up the classifier.
        The structure of the output dictionary is:
        {
            0: {"param1": float, ...},
            1: {"param1": float, ...},
            2: {"param1": float, ...},
        }
        """
        params = {}
        for state in range(3):
            params[state] = {
                k: v
                for k, v in zip(self._param_names[state], self._param_values[state])
            }
        return params

    @property
    def statistics(self) -> Dict[str, np.ndarray]:
        """
        Returns dictionary with general statistical data:
        - mu_0: np.array([float, float])
        - mu_1: np.array([float, float])
        - mu_2: np.array([float, float])
        - cov_0: np.array([[float, float], [float, float]])
        - cov_1: np.array([[float, float], [float, float]])
        - cov_2: np.array([[float, float], [float, float]])
        It can also include other information.

        NB: this property is used for plotting and for storing useful
            information in the YAML file
        """
        return {}

    @classmethod
    def fit(
        cls: Type[T3],
        shots_0: np.ndarray,
        shots_1: np.ndarray,
        shots_2: np.ndarray,
        **kargs,
    ) -> T3:
        """
        Runs fit to the given data

        Parameters
        ----------
        shots_0: np.array(N, 2)
            IQ data when preparing state 0
        shots_1: np.array(M, 2)
            IQ data when preparing state 1
        shots_2: np.array(P, 2)
            IQ data when preparing state 2

        Returns
        -------
        Loaded `ThreeStateClassifier`
        """
        check_2d_input(shots_0, axis=1)
        check_2d_input(shots_1, axis=1)
        check_2d_input(shots_2, axis=1)

        # perform fit ...

        params = {}

        return cls(params)

    def predict(
        self, z: np.ndarray, p_0: float = 1 / 3, p_1: float = 1 / 3
    ) -> np.ndarray:
        """
        Classifies the given data to 0, 1 or 2 using maximum likelihood classification:
        - 0 if p(0|z) > p(1|z), p(2|z)
        - 1 if p(1|z) > p(0|z), p(2|z)
        - 2 otherwise

        Parameters
        ----------
        z: np.array(..., 2)
            Points to classify
        p_0
            Probability to measure outcome 0.
        p_1
            Probability to measure outcome 1.

        By default p_0=p_1=1/3, thus using p(i|z) is equivalent to using p(z|i)

        Returns
        -------
        prediction: np.array(...)
            Classification of the given data. It only contains 0s and 1s.
        """
        if (p_0 + p_1 > 1) or (p_0 < 0) or (p_1 < 0):
            raise ValueError(
                "The speficied 'p_0' and 'p_1' must be physical probabilities, "
                f"but p_0={p_0} and p1={p_1} (and p2={1-p_0-p_1}) were given"
            )
        # does not compute p(z) for p(i|z) = p(z|i) * p(i) / p(z)
        # because it is the same for all states and we are
        # only selecting the one with highest probability
        probs = [
            self.pdf_0(z) * p_0,
            self.pdf_1(z) * p_1,
            self.pdf_2(z) * (1 - p_0 - p_1),
        ]
        return np.argmax(probs, axis=0)

    def pdf_0(self, z: np.ndarray) -> np.ndarray:
        """
        Returns p(z|0)

        Parameters
        ----------
        z: np.array(..., 2)
            IQ points

        Returns
        -------
        prob: np.array(...)
            Probability of the given IQ points
        """
        check_2d_input(z)
        # the pdf functions are class variables (as opposed to instance variables)
        # thus they are available in the class of `self`, not the instance of `self`
        return self.__class__._pdf_func_0(z, *self._param_values[0])

    def pdf_1(self, z: np.ndarray) -> np.ndarray:
        """
        Returns p(z|1)

        Parameters
        ----------
        z: np.array(..., 2)
            IQ points

        Returns
        -------
        prob: np.array(...)
            Probability of the given IQ points
        """
        check_2d_input(z)
        # the pdf functions are class variables (as opposed to instance variables)
        # thus they are available in the class of `self`, not the instance of `self`
        return self.__class__._pdf_func_1(z, *self._param_values[1])

    def pdf_2(self, z: np.ndarray) -> np.ndarray:
        """
        Returns p(z|2)

        Parameters
        ----------
        z: np.array(..., 2)
            IQ points

        Returns
        -------
        prob: np.array(...)
            Probability of the given IQ points
        """
        check_2d_input(z)
        # the pdf functions are class variables (as opposed to instance variables)
        # thus they are available in the class of `self`, not the instance of `self`
        return self.__class__._pdf_func_2(z, *self._param_values[2])

    def _check_params(self, params: Dict[int, Dict[str, float]]):
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be a dict, but {type(params)} was given")
        if set(params) != set([0, 1, 2]):
            raise ValueError(
                f"'params' must have keys [0,1,2], but {list(params)} were given"
            )

        for state, p in params.items():
            if not isinstance(p, dict):
                raise TypeError(
                    f"'params[{state}]' must be a dict, but {type(p)} was given"
                )
            if set(p) != set(self._param_names[state]):
                raise ValueError(
                    f"'params[{state}]' must have keys {self._param_names[state]}, "
                    f" but {list(p)} were given"
                )

            for key, value in p.items():
                if (not isinstance(value, float)) and (not isinstance(value, int)):
                    raise TypeError(
                        f"'params[{state}][{key}]' must be a float, "
                        f"but {type(value)} was given"
                    )

        return
