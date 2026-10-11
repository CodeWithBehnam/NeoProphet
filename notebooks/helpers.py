"""Shared set-up for the NeoProphet notebooks.

Every notebook starts with ``from helpers import load_example, quiet_logs``,
so finding the example data and silencing routine log output live in one
place instead of being repeated in each notebook.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import pandas as pd

EXAMPLES = {
    "peyton_manning": "example_wp_log_peyton_manning.csv",
    "wp_log_r": "example_wp_log_R.csv",
    "wp_log_r_outliers1": "example_wp_log_R_outliers1.csv",
    "wp_log_r_outliers2": "example_wp_log_R_outliers2.csv",
    "air_passengers": "example_air_passengers.csv",
    "retail_sales": "example_retail_sales.csv",
    "pedestrians": "example_pedestrians_covid.csv",
    "pedestrians_multivariate": "example_pedestrians_multivariate.csv",
    "yosemite_temps": "example_yosemite_temps.csv",
}


@cache
def repo_root() -> Path:
    """The repository root, found by walking up from the working directory."""
    for path in (Path.cwd(), *Path.cwd().parents):
        if (path / "examples").is_dir() and (path / "python").is_dir():
            return path
    raise FileNotFoundError(
        "Could not find the NeoProphet repository root (a folder with examples/ and "
        "python/). Start Jupyter from inside the repository."
    )


def load_example(name: str) -> pd.DataFrame:
    """Load one of the bundled example datasets with ``ds`` parsed as dates.

    Rows are sorted by ``ds`` because some of the original CSV files are not.
    """
    if name not in EXAMPLES:
        raise KeyError(
            f"Unknown example {name!r}. Choose one of: {', '.join(EXAMPLES)}."
        )
    df = pd.read_csv(repo_root() / "examples" / EXAMPLES[name])
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds", ignore_index=True)


def quiet_logs() -> None:
    """Hide routine fitting messages so notebook output shows only what matters.

    cmdstanpy resets its own level on first use unless a handler already
    exists, so a handler is attached before the level is set.
    """
    cmdstan = logging.getLogger("cmdstanpy")
    if not cmdstan.handlers:
        cmdstan.addHandler(logging.NullHandler())
    cmdstan.setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
