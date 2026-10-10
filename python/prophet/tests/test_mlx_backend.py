"""Tests for the experimental MLX backend (skipped when mlx is unavailable)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlx.core")

from prophet import Prophet  # noqa: E402
from prophet.mlx_backend import (  # noqa: E402
    MLXBackend,
    fit_many,
    negative_log_posterior,
)


def _stan_inputs(m: Prophet, df: pd.DataFrame) -> tuple[dict, dict]:
    inputs = m.preprocess(df)
    init = dataclasses.asdict(m.calculate_initial_params(inputs.K))
    return init, dict(vars(inputs))


def _objective_of(m: Prophet, data: dict) -> float:
    p = m.params
    return negative_log_posterior(
        data,
        p["k"].item(),
        p["m"].item(),
        p["delta"].ravel(),
        p["beta"].ravel(),
        p["sigma_obs"].item(),
    )


@pytest.fixture(scope="module")
def short_ts(daily_univariate_ts: pd.DataFrame) -> pd.DataFrame:
    return daily_univariate_ts.head(300).reset_index(drop=True)


def test_objective_matches_stan_lp(short_ts: pd.DataFrame) -> None:
    """-lp__ reported by CmdStan optimize equals our objective at the same point."""
    m = Prophet()
    m.fit(short_ts)
    _, data = _stan_inputs(Prophet(), short_ts)
    assert _objective_of(m, data) == pytest.approx(-m.params["lp__"].item(), abs=1e-3)


def test_mlx_optimum_matches_stan(short_ts: pd.DataFrame) -> None:
    """Stan's L-BFGS stalls at the Laplace kinks; its Newton run is the fair reference."""
    lbfgs, newton = Prophet(), Prophet()
    lbfgs.fit(short_ts)
    newton.fit(short_ts, algorithm="Newton", iter=100000)
    (mlx_model,) = fit_many([short_ts], device="cpu")
    _, data = _stan_inputs(Prophet(), short_ts)
    assert _objective_of(mlx_model, data) <= _objective_of(lbfgs, data) + 1e-3
    assert _objective_of(mlx_model, data) == pytest.approx(
        _objective_of(newton, data), abs=0.05
    )
    y = short_ts["y"]
    a = newton.predict(newton.history)["yhat"].to_numpy()
    b = mlx_model.predict(mlx_model.history)["yhat"].to_numpy()
    assert np.abs(a - b).max() / (y.max() - y.min()) < 5e-3


def test_backend_fit_param_shapes(short_ts: pd.DataFrame) -> None:
    m = Prophet()
    init, data = _stan_inputs(m, short_ts)
    params = MLXBackend().fit(init, data)
    T, S, K = data["T"], data["S"], data["K"]
    expected = {
        "lp__": 1,
        "k": 1,
        "m": 1,
        "delta": S,
        "sigma_obs": 1,
        "beta": K,
        "trend": T,
    }
    assert list(params) == list(expected)
    for name, width in expected.items():
        assert params[name].shape == (1, width), name


def test_prophet_fit_with_mlx_backend(short_ts: pd.DataFrame) -> None:
    m = Prophet()
    m.stan_backend = MLXBackend()
    m.fit(short_ts)
    forecast = m.predict(m.make_future_dataframe(10))
    assert len(forecast) == len(short_ts) + 10
    assert forecast["yhat"].notna().all()


def test_fit_many_predicts_and_preserves_order(
    daily_univariate_ts: pd.DataFrame,
) -> None:
    dfs = [
        daily_univariate_ts.head(200),
        daily_univariate_ts.head(400).assign(y=lambda d: d["y"] * 2.0 + 5.0),
        daily_univariate_ts.head(300),
    ]
    models = fit_many(dfs, yearly_seasonality=False)
    assert len(models) == 3
    for df, m in zip(dfs, models, strict=True):
        assert m.history is not None and len(m.history) == len(df)
        out = m.predict(m.make_future_dataframe(5))
        assert out["yhat"].notna().all()
        assert out["yhat_lower"].notna().all()


def test_fit_many_batch_matches_single_fit(short_ts: pd.DataFrame) -> None:
    other = short_ts.assign(y=short_ts["y"] * 1.5 + 2.0)
    batch = fit_many([short_ts, other], device="cpu")
    (single,) = fit_many([short_ts], device="cpu")
    a = batch[0].predict(batch[0].history)["yhat"].to_numpy()
    b = single.predict(single.history)["yhat"].to_numpy()
    scale = short_ts["y"].max() - short_ts["y"].min()
    assert np.abs(a - b).max() / scale < 1e-4


def test_gpu_matches_cpu(short_ts: pd.DataFrame) -> None:
    (gpu,) = fit_many([short_ts], device="gpu")
    (cpu,) = fit_many([short_ts], device="cpu")
    scale = short_ts["y"].max() - short_ts["y"].min()
    a = gpu.predict(gpu.history)["yhat"].to_numpy()
    b = cpu.predict(cpu.history)["yhat"].to_numpy()
    assert np.abs(a - b).max() / scale < 5e-3


def test_flat_and_multiplicative(short_ts: pd.DataFrame) -> None:
    flat, mult = (
        fit_many([short_ts], device="cpu", growth="flat")[0],
        fit_many(
            [short_ts.assign(y=short_ts["y"] + 50.0)],
            device="cpu",
            seasonality_mode="multiplicative",
        )[0],
    )
    assert flat.predict(flat.history)["yhat"].notna().all()
    assert mult.predict(mult.history)["yhat"].notna().all()


def test_constant_series_and_no_changepoints(short_ts: pd.DataFrame) -> None:
    constant = short_ts.assign(y=3.0)
    (m,) = fit_many([constant])
    assert m.predict(m.history)["yhat"].to_numpy() == pytest.approx(3.0)
    (m0,) = fit_many([short_ts], n_changepoints=0, device="cpu")
    assert np.all(m0.params["delta"] == 0)


def test_unsupported_options_raise(short_ts: pd.DataFrame) -> None:
    logistic = short_ts.assign(cap=short_ts["y"].max() * 2)
    with pytest.raises(NotImplementedError, match="logistic"):
        fit_many([logistic], growth="logistic")
    with pytest.raises(NotImplementedError, match="MAP"):
        fit_many([short_ts], mcmc_samples=100)
    with pytest.raises(ValueError, match="stan_backend"):
        fit_many([short_ts], stan_backend="CMDSTANPY")
    with pytest.raises(ValueError, match="empty"):
        fit_many([])
    with pytest.raises(NotImplementedError, match="MCMC|MAP"):
        MLXBackend().sampling({}, {}, 100)
    init, data = _stan_inputs(Prophet(growth="logistic"), short_ts.assign(cap=100.0))
    with pytest.raises(NotImplementedError, match="logistic"):
        MLXBackend().fit(init, data)
    init, data = _stan_inputs(Prophet(), short_ts)
    with pytest.raises(ValueError, match="Unsupported fit options"):
        MLXBackend().fit(init, data, algorithm="Newton")
    with pytest.raises(ValueError, match="device"):
        MLXBackend().set_options(device="tpu")


def test_backend_selected_by_name(short_ts: pd.DataFrame) -> None:
    m = Prophet(stan_backend="MLX", uncertainty_samples=0)
    assert m.stan_backend is not None
    assert m.stan_backend.get_type() == "MLX"
    m.fit(short_ts)
    assert len(m.predict(m.make_future_dataframe(periods=7))) == len(short_ts) + 7


def test_default_backend_is_not_mlx() -> None:
    m = Prophet()
    assert m.stan_backend is not None
    assert m.stan_backend.get_type() == "CMDSTANPY"


def test_cross_validation_keeps_mlx_backend(short_ts: pd.DataFrame) -> None:
    from prophet.diagnostics import cross_validation

    m = Prophet(stan_backend="MLX", uncertainty_samples=0).fit(short_ts)
    df_cv = cross_validation(
        m, horizon="30 days", period="60 days", initial="180 days", disable_tqdm=True
    )
    assert not df_cv.empty
