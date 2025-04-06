# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
from typing import Dict, Any

import numpy as np
import pandas as pd


def regressor_index(m: Any, name: str) -> int:
    """Given the name of a regressor, return its (column) index in the `beta` matrix.

    Parameters
    ----------
    m: Prophet model object, after fitting.
    name: Name of the regressor, as passed into the `add_regressor` function.

    Returns
    -------
    int: The column index of the regressor in the `beta` matrix.
    """
    try:
        index = np.extract(m.train_component_cols[name] == 1, m.train_component_cols.index)[0]
        return index
    except KeyError:
        raise ValueError(f"Regressor '{name}' not found in the model's train_component_cols.")
    except IndexError:
        raise ValueError(f"Regressor '{name}' does not have a valid index.")


def regressor_coefficients(m: Any) -> pd.DataFrame:
    """Summarize the coefficients of the extra regressors used in the model.

    Parameters
    ----------
    m: Prophet model object, after fitting.

    Returns
    -------
    pd.DataFrame: DataFrame containing regressor coefficients and related information.
    """
    if len(m.extra_regressors) == 0:
        raise ValueError("No extra regressors found.")

    coefs = []
    for regressor, params in m.extra_regressors.items():
        beta = m.params["beta"][:, regressor_index(m, regressor)]
        coef = beta * m.y_scale / params["std"] if params["mode"] == "additive" else beta / params["std"]
        percentiles = [(1 - m.interval_width) / 2, 1 - (1 - m.interval_width) / 2]
        coef_bounds = np.quantile(coef, q=percentiles)
        record = {
            "regressor": regressor,
            "regressor_mode": params["mode"],
            "center": params["mu"],
            "coef_lower": coef_bounds[0],
            "coef": np.mean(coef),
            "coef_upper": coef_bounds[1],
        }
        coefs.append(record)

    return pd.DataFrame(coefs)


def warm_start_params(m: Any) -> Dict[str, Any]:
    """
    Retrieve parameters from a trained model in the format used to initialize a new Stan model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    Dict[str, Any]: Dictionary containing retrieved parameters of m.
    """
    res = {}
    param_names = ["k", "m", "sigma_obs", "delta", "beta"]
    for pname in param_names:
        if pname in ["k", "m", "sigma_obs"]:
            res[pname] = m.params[pname][0][0] if m.mcmc_samples == 0 else np.mean(m.params[pname])
        else:
            res[pname] = m.params[pname][0] if m.mcmc_samples == 0 else np.mean(m.params[pname], axis=0)

    return res
