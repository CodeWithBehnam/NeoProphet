# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from collections import OrderedDict
from contextlib import contextmanager
from enum import Enum
import importlib.resources
import json
import pathlib
import platform
import tempfile
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence, cast

import numpy as np
import pandas as pd

import logging
logger: logging.Logger = logging.getLogger('prophet.models')

if TYPE_CHECKING:
    from cmdstanpy import CmdStanMCMC, CmdStanMLE, CmdStanModel
    from typing_extensions import TypedDict, Unpack, type_check_only, final

    @type_check_only
    @final
    class _IStanBackendOptions(TypedDict, total=False):
        newton_fallback: bool

PLATFORM = "win" if platform.platform().startswith("Win") else "unix"

class TrendIndicator(Enum):
    LINEAR = 0
    LOGISTIC = 1
    FLAT = 2

@dataclass
class ModelInputData:
    T: int
    S: int
    K: int
    tau: float
    trend_indicator: int
    y: Sequence[float] | pd.Series  # length T
    t: Sequence[float] | pd.Series  # length T
    cap: Sequence[float] | np.ndarray | pd.Series  # length T
    t_change: Sequence[float] | np.ndarray  # length S
    s_a: Sequence[int] | pd.Series  # length K
    s_m: Sequence[int] | pd.Series  # length K
    X: Sequence[Sequence[float]] | pd.DataFrame  # shape (T, K)
    sigmas: Sequence[float]  # length K

@dataclass
class ModelParams:
    k: float
    m: float
    delta: Sequence[float] | np.ndarray  # length S
    beta: Sequence[float] | np.ndarray  # length K
    sigma_obs: float


def _to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@contextmanager
def stan_json_files(
    inits: Mapping[str, Any],
    data: Mapping[str, Any],
) -> Iterator[tuple[str, str]]:
    """Write Stan inits and data to temporary JSON files; yield their paths.

    cmdstanpy serialises dictionaries with stanio, which falls back to a
    pure Python chunked encoder when ujson is absent. A single ``json.dumps``
    call uses the C encoder instead and emits the same ``repr`` of each
    float, so the values Stan reads are bit-for-bit unchanged. The files are
    removed on exit, including when optimisation raises.
    """
    with tempfile.TemporaryDirectory(prefix="prophet_stan_") as tmp:
        directory = pathlib.Path(tmp)
        paths = []
        for name, mapping in (("init", inits), ("data", data)):
            path = directory / f"{name}.json"
            path.write_text(
                json.dumps({k: _to_json_value(v) for k, v in mapping.items()}),
                encoding="utf-8",
            )
            paths.append(str(path))
        yield paths[0], paths[1]


class IStanBackend(ABC):
    model: Any
    stan_fit: Any | None
    newton_fallback: bool
    
    def __init__(self) -> None:
        self.model = self.load_model()
        self.stan_fit = None
        self.newton_fallback = True

    def set_options(self, **kwargs: Unpack[_IStanBackendOptions]) -> None:
        """
        Specify model options as kwargs.
         * newton_fallback [bool]: whether to fallback to Newton if L-BFGS fails
        """
        for k, v in kwargs.items():
            if k == 'newton_fallback':
                self.newton_fallback = cast(bool, v)
            else:
                raise ValueError(f'Unknown option {k}')
    
    def cleanup(self) -> None:
        """Clean up temporary files created during model fitting."""
        pass


    @staticmethod
    @abstractmethod
    def get_type() -> str:
        pass

    @abstractmethod
    def load_model(self) -> Any:
        pass

    @abstractmethod
    def fit(
        self,
        stan_init: dict[str, float],
        stan_data: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def sampling(
        self,
        stan_init: dict[str, float],
        stan_data: dict[str, Any],
        samples: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        pass


class CmdStanPyBackend(IStanBackend):
    CMDSTAN_VERSION = "2.37.0"

    model: CmdStanModel
    stan_fit: CmdStanMLE | CmdStanMCMC | None
    
    def __init__(self) -> None:
        import cmdstanpy
        # this must be set before super.__init__() for load_model to work on Windows
        local_cmdstan = cast(
            pathlib.Path,
            importlib.resources.files("prophet") / "stan_model" / f"cmdstan-{self.CMDSTAN_VERSION}",
        )
        if local_cmdstan.exists():
            cmdstanpy.set_cmdstan_path(str(local_cmdstan))
        super().__init__()

    @staticmethod
    def get_type() -> str:
        return StanBackendEnum.CMDSTANPY.name

    def load_model(self) -> CmdStanModel:
        import cmdstanpy
        model_file = importlib.resources.files("prophet") / "stan_model" / "prophet_model.bin"
        return cmdstanpy.CmdStanModel(exe_file=str(model_file))

    def fit(
        self,
        stan_init: dict[str, float], 
        stan_data: dict[str, Any],
        **kwargs: Any,
    ) -> OrderedDict[str, np.ndarray]:
        if 'inits' not in kwargs and 'init' in kwargs:
            stan_init = self.sanitize_custom_inits(stan_init, kwargs['init'])
            del kwargs['init']

        inits_list, data_list = self.prepare_data(stan_init, stan_data)
        with stan_json_files(inits_list, data_list) as (inits_path, data_path):
            args = dict(
                data=data_path,
                inits=inits_path,
                algorithm='Newton' if cast(float, data_list['T']) < 100 else 'LBFGS',
                iter=int(1e4),
            )
            args.update(kwargs)

            try:
                self.stan_fit = self.model.optimize(**args)
            except RuntimeError as e:
                # Fall back on Newton
                if not self.newton_fallback or args['algorithm'] == 'Newton':
                    raise e
                logger.warning('Optimization terminated abnormally. Falling back to Newton.')
                args['algorithm'] = 'Newton'
                self.stan_fit = self.model.optimize(**args)
        params = self.stan_to_dict_numpy(
            self.stan_fit.column_names, self.stan_fit.optimized_params_np)
        for par in params:
            params[par] = params[par].reshape((1, -1))
        return params

    def sampling(
        self,
        stan_init: dict[str, float],
        stan_data: dict[str, Any],
        samples: int,
        **kwargs: Any,
    ) -> OrderedDict[str, np.ndarray]:
        if 'inits' not in kwargs and 'init' in kwargs:
            stan_init = self.sanitize_custom_inits(stan_init, kwargs['init'])
            del kwargs['init']

        inits_list, data_list = self.prepare_data(stan_init, stan_data)
        if 'chains' not in kwargs:
            kwargs['chains'] = 4
        iter_half = samples // 2
        kwargs['iter_sampling'] = iter_half
        if 'iter_warmup' not in kwargs:
            kwargs['iter_warmup'] = iter_half
        with stan_json_files(inits_list, data_list) as (inits_path, data_path):
            args = dict(
                data=data_path,
                inits=inits_path,
            )
            args.update(kwargs)
            self.stan_fit = self.model.sample(**args)
        res = self.stan_fit.draws()
        (samples, c, columns) = res.shape
        res = res.reshape((samples * c, columns))
        params = self.stan_to_dict_numpy(self.stan_fit.column_names, res)

        for par in params:
            s = params[par].shape
            if s[1] == 1:
                params[par] = params[par].reshape((s[0],))

            if par in ['delta', 'beta'] and len(s) < 2:
                params[par] = params[par].reshape((-1, 1))

        return params

    def cleanup(self) -> None:
        import cmdstanpy

        if hasattr(self, "stan_fit") and self.stan_fit is not None:
            fit_result: cmdstanpy.CmdStanMLE | cmdstanpy.CmdStanMCMC = self.stan_fit
            to_remove = (
                fit_result.runset.csv_files +
                fit_result.runset.diagnostic_files +
                fit_result.runset.stdout_files +
                fit_result.runset.profile_files
            )
            for fpath in to_remove:
                if pathlib.Path(fpath).is_file():
                    pathlib.Path(fpath).unlink()

    @staticmethod
    def sanitize_custom_inits(
        default_inits: dict[str, Any],
        custom_inits: dict[str, Any],
    ) -> dict[str, float | Any]:
        """Validate that custom inits have the correct type and shape, otherwise use defaults."""
        sanitized = {}
        for param in ['k', 'm', 'sigma_obs']:
            try:
                sanitized[param] = float(custom_inits[param])
            except Exception:
                sanitized[param] = default_inits[param]
        for param in ['delta', 'beta']:
            if default_inits[param].shape == custom_inits[param].shape:
                sanitized[param] = custom_inits[param]
            else:
                sanitized[param] = default_inits[param]
        return sanitized

    @staticmethod
    def prepare_data(
        init: dict[str, Any],
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Selects the Stan inputs in the form cmdstanpy reads.

        Arrays are deliberately left as numpy objects rather than converted to
        nested Python lists. stanio serialises numpy values with a single
        C-level ``tolist`` call, whereas a list is walked element by element
        in Python, which dominated the cost of fit(). The JSON written is
        identical because ``tolist`` yields the same Python floats.
        """
        cmdstanpy_data = {
            'T': data['T'],
            'S': data['S'],
            'K': data['K'],
            'tau': data['tau'],
            'trend_indicator': data['trend_indicator'],
            'y': np.asarray(data['y']),
            't': np.asarray(data['t']),
            'cap': np.asarray(data['cap']),
            't_change': np.asarray(data['t_change']),
            's_a': np.asarray(data['s_a']),
            's_m': np.asarray(data['s_m']),
            'X': data['X'].to_numpy(),
            'sigmas': data['sigmas']
        }

        cmdstanpy_init = {
            'k': init['k'],
            'm': init['m'],
            'delta': np.asarray(init['delta']),
            'beta': np.asarray(init['beta']),
            'sigma_obs': init['sigma_obs']
        }
        return (cmdstanpy_init, cmdstanpy_data)

    @staticmethod
    def stan_to_dict_numpy(
        column_names: Sequence[str],
        data: np.ndarray,
    ) -> OrderedDict[str, np.ndarray]:
        output: OrderedDict[str, np.ndarray] = OrderedDict()

        prev: str | None = None

        start = 0
        end = 0
        two_dims = len(data.shape) > 1
        for cname in column_names:
            parsed = cname.split(".") if "." in cname else cname.split("[")
            curr = parsed[0]
            if prev is None:
                prev = curr

            if curr != prev:
                if prev in output:
                    raise RuntimeError(
                        "Found repeated column name"
                    )
                if two_dims:
                    output[prev] = np.array(data[:, start:end])
                else:
                    output[prev] = np.array(data[start:end])
                prev = curr
                start = end
            end += 1
        if prev in output:
            raise RuntimeError(
                "Found repeated column name"
            )
        assert prev is not None
        if two_dims:
            output[prev] = np.array(data[:, start:end])
        else:
            output[prev] = np.array(data[start:end])
        return output




class StanBackendEnum(Enum):
    CMDSTANPY = CmdStanPyBackend

    @staticmethod
    def get_backend_class(name: str) -> type[IStanBackend]:
        if name == "MLX":
            # Experimental and MAP-only, so it is opt-in by name and kept out of the
            # enum that Prophet iterates when choosing a default backend. The lazy
            # import also avoids a circular import, as mlx_backend imports this module.
            from prophet.mlx_backend import MLXBackend

            return MLXBackend
        try:
            return StanBackendEnum[name].value
        except KeyError as e:
            raise ValueError(f"Unknown stan backend: {name}") from e
