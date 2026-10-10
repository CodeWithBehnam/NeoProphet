"""Tests pinning the behaviour of the fast predict and fit data paths."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet
from prophet.models import CmdStanPyBackend, stan_json_files


@pytest.fixture(scope="module")
def fitted(daily_univariate_ts: pd.DataFrame) -> Prophet:
    return Prophet(uncertainty_samples=200).fit(daily_univariate_ts)


@pytest.fixture(scope="module")
def future(fitted: Prophet) -> pd.DataFrame:
    return fitted.make_future_dataframe(60)


def _reference_samples(
    m: Prophet, df: pd.DataFrame, vectorized: bool
) -> dict[str, np.ndarray]:
    """The original dict-per-sample then column_stack algorithm."""
    df = m.setup_dataframe(df.copy())
    n_iter = m.params["k"].shape[0]
    per_iter = max(1, int(np.ceil(m.uncertainty_samples / float(n_iter))))
    features, _, cols, _ = m.make_all_seasonality_features(df)
    out: dict[str, list[np.ndarray]] = {"yhat": [], "trend": []}
    for i in range(n_iter):
        kwargs = dict(
            df=df,
            seasonal_features=features,
            iteration=i,
            s_a=cols["additive_terms"],
            s_m=cols["multiplicative_terms"],
        )
        if vectorized:
            sims = m.sample_model_vectorized(n_samples=per_iter, **kwargs)
        else:
            sims = [m.sample_model(**kwargs) for _ in range(per_iter)]
        for key in out:
            out[key].extend(np.asarray(sim[key]) for sim in sims)
    return {k: np.column_stack(v) for k, v in out.items()}


@pytest.mark.parametrize("vectorized", [True, False])
def test_predictive_samples_shape(
    fitted: Prophet, future: pd.DataFrame, vectorized: bool
) -> None:
    samples = fitted.predictive_samples(future, vectorized=vectorized)
    assert set(samples) == {"yhat", "trend"}
    for arr in samples.values():
        assert arr.shape == (len(future), fitted.uncertainty_samples)
        assert np.isfinite(arr).all()


@pytest.mark.parametrize("vectorized", [True, False])
def test_predictive_samples_match_reference_algorithm(
    fitted: Prophet, future: pd.DataFrame, vectorized: bool
) -> None:
    np.random.seed(123)
    expected = _reference_samples(fitted, future, vectorized)
    np.random.seed(123)
    actual = fitted.predictive_samples(future, vectorized=vectorized)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_percentile_interval_equals_separate_calls(fitted: Prophet) -> None:
    a = np.random.default_rng(0).normal(size=(50, 300))
    lower, upper = fitted.percentile_interval(a, 10.0, 90.0)
    np.testing.assert_array_equal(lower, np.percentile(a, 10.0, axis=1))
    np.testing.assert_array_equal(upper, np.percentile(a, 90.0, axis=1))


def test_percentile_interval_ignores_nans(fitted: Prophet) -> None:
    a = np.random.default_rng(1).normal(size=(5, 100))
    a[:, 0] = np.nan
    lower, upper = fitted.percentile_interval(a, 5.0, 95.0)
    np.testing.assert_array_equal(lower, np.nanpercentile(a, 5.0, axis=1))
    np.testing.assert_array_equal(upper, np.nanpercentile(a, 95.0, axis=1))


def test_predict_intervals_are_ordered(fitted: Prophet, future: pd.DataFrame) -> None:
    fc = fitted.predict(future)
    assert (fc["yhat_lower"] <= fc["yhat_upper"]).all()
    assert (fc["trend_lower"] <= fc["trend_upper"]).all()
    assert (fc["yhat_lower"] <= fc["yhat"] + 1e-9).mean() > 0.99
    assert (fc["yhat"] <= fc["yhat_upper"] + 1e-9).mean() > 0.99
    for name in ("weekly", "yearly"):
        assert (fc[f"{name}_lower"] <= fc[f"{name}_upper"]).all()


def test_predict_is_reproducible_under_seed(
    fitted: Prophet, future: pd.DataFrame
) -> None:
    np.random.seed(5)
    first = fitted.predict(future)
    np.random.seed(5)
    second = fitted.predict(future)
    pd.testing.assert_frame_equal(first, second)


def test_predict_intervals_match_samples(fitted: Prophet, future: pd.DataFrame) -> None:
    np.random.seed(9)
    fc = fitted.predict(future)
    np.random.seed(9)
    samples = fitted.predictive_samples(future)
    p = 100 * (1 - fitted.interval_width) / 2
    np.testing.assert_array_equal(
        fc["yhat_lower"].to_numpy(), np.percentile(samples["yhat"], p, axis=1)
    )
    np.testing.assert_array_equal(
        fc["trend_upper"].to_numpy(), np.percentile(samples["trend"], 100 - p, axis=1)
    )


def test_prepare_data_keeps_numpy_arrays(fitted: Prophet) -> None:
    stan_init, stan_data = fitted.stan_backend.prepare_data(
        {
            "k": 0.0,
            "m": 0.0,
            "sigma_obs": 1.0,
            "delta": np.zeros(3),
            "beta": np.zeros(2),
        },
        {
            "T": 2,
            "S": 1,
            "K": 2,
            "tau": 0.05,
            "trend_indicator": 0,
            "y": pd.Series([1.0, 2.0]),
            "t": pd.Series([0.0, 1.0]),
            "cap": pd.Series([0.0, 0.0]),
            "t_change": np.array([0.5]),
            "s_a": pd.Series([1, 0]),
            "s_m": pd.Series([0, 1]),
            "X": pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
            "sigmas": [10.0, 10.0],
        },
    )
    assert isinstance(stan_data["y"], np.ndarray)
    assert stan_data["X"].shape == (2, 2)
    assert isinstance(stan_init["delta"], np.ndarray)


def test_stan_json_files_round_trip_and_cleanup() -> None:
    data = {
        "T": 3,
        "tau": 0.05,
        "y": np.array([0.1, 1 / 3, 1e-300]),
        "X": np.array([[1.0, 2.0], [3.0, 4.5], [5.0, 6.0]]),
        "flag": np.int64(7),
    }
    inits = {"k": 0.1 + 0.2, "delta": np.array([np.pi, -np.e])}
    with stan_json_files(inits, data) as (init_path, data_path):
        assert pathlib.Path(init_path).parent == pathlib.Path(data_path).parent
        loaded_data = json.loads(pathlib.Path(data_path).read_text())
        loaded_init = json.loads(pathlib.Path(init_path).read_text())
        directory = pathlib.Path(data_path).parent
    assert loaded_data["y"] == data["y"].tolist()
    assert loaded_data["X"] == data["X"].tolist()
    assert loaded_data["flag"] == 7
    assert loaded_init == {"k": 0.1 + 0.2, "delta": [np.pi, -np.e]}
    assert not directory.exists()


def test_stan_json_files_cleanup_on_error() -> None:
    directory = None
    with pytest.raises(RuntimeError):
        with stan_json_files({"k": 1.0}, {"T": 1}) as (init_path, _):
            directory = pathlib.Path(init_path).parent
            raise RuntimeError("optimiser failed")
    assert directory is not None and not directory.exists()


def test_fit_matches_cmdstanpy_dict_input(daily_univariate_ts: pd.DataFrame) -> None:
    """File based input gives the same optimum as handing cmdstanpy the dicts."""
    m = Prophet()
    backend = m.stan_backend
    assert isinstance(backend, CmdStanPyBackend)
    original = backend.model.optimize
    seen: dict[str, object] = {}

    def spy(**kwargs):
        seen["kwargs"] = dict(kwargs)
        seen["data"] = json.loads(pathlib.Path(kwargs["data"]).read_text())
        seen["inits"] = json.loads(pathlib.Path(kwargs["inits"]).read_text())
        return original(**kwargs)

    backend.model.optimize = spy
    try:
        m.fit(daily_univariate_ts)
    finally:
        backend.model.optimize = original

    kwargs = dict(seen["kwargs"])
    assert not pathlib.Path(kwargs["data"]).parent.exists()
    kwargs.update(data=seen["data"], inits=seen["inits"])
    reference = original(**kwargs)
    np.testing.assert_array_equal(
        m.stan_fit.optimized_params_np, reference.optimized_params_np
    )
