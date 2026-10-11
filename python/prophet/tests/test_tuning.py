import dataclasses
import logging
import math

import pandas as pd
import pytest

from prophet import tuning
from prophet.tuning import (
    CVWindows,
    Trial,
    TuningResult,
    default_param_grid,
    derive_windows,
    expand_grid,
    infer_step,
    longest_seasonality,
    rank_trials,
    with_baseline,
    scores_frame,
    trial_score,
    tune,
    validate_param_grid,
)


def _dates(days: int, freq: str = "D") -> pd.Series:
    return pd.Series(pd.date_range("2020-01-01", periods=days, freq=freq), name="ds")


# --- grid --------------------------------------------------------------


def test_default_grid_is_valid_and_has_32_combinations():
    grid = validate_param_grid(default_param_grid())
    assert len(expand_grid(grid)) == 32


def test_expand_grid_order_is_deterministic():
    combos = expand_grid({"a": (1, 2), "b": ("x", "y")})
    assert combos == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


@pytest.mark.parametrize(
    ("grid", "kwargs", "message"),
    [
        ({}, None, "at least one"),
        ({"changepoint_prior_scal": [0.1]}, None, "Unknown Prophet parameter"),
        ({"self": [1]}, None, "Unknown Prophet parameter"),
        ({"changepoint_prior_scale": []}, None, "is empty"),
        ({"changepoint_prior_scale": 0.1}, None, "list or tuple"),
        ({"seasonality_mode": "additive"}, None, "list or tuple"),
        ({"changepoint_prior_scale": [0.1]}, {"changepoint_prior_scale": 0.2}, "both"),
    ],
)
def test_validate_param_grid_errors(grid, kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_param_grid(grid, kwargs)


# --- windows -----------------------------------------------------------


def test_infer_step_uses_median_gap():
    ds = pd.Series(
        pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-10"])
    )
    assert infer_step(ds) == pd.Timedelta(days=1)


def test_longest_seasonality_follows_prophet_auto_rules():
    day = pd.Timedelta(days=1)
    assert longest_seasonality(pd.Timedelta(days=800), day) == pd.Timedelta(days=365.25)
    assert longest_seasonality(pd.Timedelta(days=100), day) == pd.Timedelta(days=7)
    assert longest_seasonality(
        pd.Timedelta(days=100), pd.Timedelta(days=30)
    ) == pd.Timedelta(days=30)
    assert longest_seasonality(pd.Timedelta(days=3), pd.Timedelta(hours=1)) == day


def test_derive_windows_defaults_on_long_daily_series():
    w = derive_windows(_dates(3000))
    assert w.horizon == pd.Timedelta(days=300)  # span / 10, below the one year cap
    assert w.initial == 3 * w.horizon
    assert 3 <= w.n_folds <= 10
    assert (
        w.cutoffs[-1]
        == pd.Timestamp("2020-01-01") + pd.Timedelta(days=2999) - w.horizon
    )
    assert list(w.cutoffs) == sorted(w.cutoffs)


def test_derive_windows_caps_horizon_at_one_year():
    w = derive_windows(_dates(20000))
    assert w.horizon == pd.Timedelta(days=366)  # 365.25 days rounded up to a whole step


def test_derive_windows_caps_fold_count():
    w = derive_windows(
        _dates(6000),
        horizon="30 days",
        period="1 days",
        max_folds=10,
        initial="365 days",
    )
    # Explicit period is respected, so the cap does not apply.
    assert w.n_folds > 10
    auto = derive_windows(
        _dates(6000), horizon="30 days", initial="365 days", max_folds=10
    )
    assert auto.n_folds <= 10


def test_derive_windows_relaxes_omitted_arguments_when_history_is_short():
    # 400 days: the default initial (>= one year seasonality) leaves too few folds.
    w = derive_windows(_dates(400))
    assert w.n_folds >= 3


def test_derive_windows_respects_explicit_arguments():
    w = derive_windows(
        _dates(400), horizon="20 days", initial="100 days", period="40 days"
    )
    assert (w.horizon, w.initial, w.period) == tuple(
        pd.Timedelta(x) for x in ("20 days", "100 days", "40 days")
    )


def test_derive_windows_monthly_data():
    ds = pd.Series(pd.date_range("1949-01-01", periods=144, freq="MS"))
    w = derive_windows(ds)
    assert w.n_folds >= 3
    assert w.horizon >= pd.Timedelta(days=28)


def test_derive_windows_too_short_history_fails_clearly():
    with pytest.raises(ValueError, match="too short"):
        derive_windows(_dates(3))
    with pytest.raises(ValueError, match="too short"):
        derive_windows(_dates(60), horizon="90 days")


def test_derive_windows_rejects_non_positive_and_bad_fold_bounds():
    with pytest.raises(ValueError, match="positive"):
        derive_windows(_dates(400), horizon="0 days")
    with pytest.raises(ValueError, match="min_folds"):
        derive_windows(_dates(400), min_folds=5, max_folds=2)


def test_with_baseline_appends_prophet_defaults_once():
    combos = expand_grid({"changepoint_prior_scale": (0.01, 0.5)})
    out, idx = with_baseline(combos, ["changepoint_prior_scale"])
    assert out[idx] == {"changepoint_prior_scale": 0.05}
    assert len(out) == 3
    again, idx2 = with_baseline(out, ["changepoint_prior_scale"])
    assert again == out and idx2 == idx


# --- ranking -----------------------------------------------------------


def _trial(cp: float, **metrics: float) -> Trial:
    return Trial(params={"changepoint_prior_scale": cp}, metrics=metrics, seconds=0.1)


def test_rank_trials_sorts_ascending_with_stable_ties_and_nan_last():
    trials = [
        _trial(0.1, rmse=2.0),
        _trial(0.2, rmse=math.nan),
        _trial(0.3, rmse=1.0),
        _trial(0.4, rmse=2.0),
    ]
    ranked = rank_trials(trials, "rmse")
    assert [t.params["changepoint_prior_scale"] for t in ranked] == [0.3, 0.1, 0.4, 0.2]


def test_coverage_is_ranked_by_distance_from_interval_width():
    trials = [
        _trial(0.1, coverage=0.95),
        _trial(0.2, coverage=0.78),
        _trial(0.3, coverage=0.5),
    ]
    ranked = rank_trials(trials, "coverage", interval_width=0.8)
    assert [t.params["changepoint_prior_scale"] for t in ranked] == [0.2, 0.1, 0.3]
    assert trial_score({"coverage": 0.7}, "coverage", 0.8) == pytest.approx(0.1)


def test_scores_frame_is_tidy():
    ranked = rank_trials(
        [_trial(0.1, rmse=2.0, mae=1.0), _trial(0.2, rmse=1.0, mae=3.0)], "rmse"
    )
    frame = scores_frame(ranked, "rmse")
    assert list(frame.columns) == [
        "changepoint_prior_scale",
        "rmse",
        "mae",
        "score",
        "rank",
        "seconds",
        "is_default",
    ]
    assert frame["rank"].tolist() == [1, 2]
    assert frame["changepoint_prior_scale"].tolist() == [0.2, 0.1]


def test_results_are_frozen():
    windows = CVWindows(
        pd.Timedelta(days=1), pd.Timedelta(days=2), pd.Timedelta(days=1), ()
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        windows.horizon = pd.Timedelta(days=5)  # type: ignore[misc]
    trial = Trial(params={}, metrics={}, seconds=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        trial.seconds = 1.0  # type: ignore[misc]
    assert "best_params" in {f.name for f in dataclasses.fields(TuningResult)}


# --- tune() argument validation ---------------------------------------


def test_tune_rejects_unknown_metric(daily_univariate_ts):
    with pytest.raises(ValueError, match="Unknown metric"):
        tune(daily_univariate_ts, metric="rmsle")


def test_tune_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required"):
        tune(pd.DataFrame({"ds": _dates(100), "value": range(100)}))


def test_tune_rejects_bad_grid_before_fitting(daily_univariate_ts):
    with pytest.raises(ValueError, match="Unknown Prophet parameter"):
        tune(daily_univariate_ts, param_grid={"nope": [1]})


def test_tune_rejects_mape_when_y_has_zeros():
    df = pd.DataFrame({"ds": _dates(200), "y": [0.0, 1.0] * 100})
    with pytest.raises(ValueError, match="mape"):
        tune(df, metric="mape")


def test_tune_fails_clearly_on_short_history():
    df = pd.DataFrame({"ds": _dates(5), "y": range(5)})
    with pytest.raises(ValueError, match="too short"):
        tune(df, param_grid={"changepoint_prior_scale": [0.1]})


# --- end to end --------------------------------------------------------


GRID = {"changepoint_prior_scale": [0.01, 0.5], "seasonality_prior_scale": [0.1, 10.0]}


def test_tune_end_to_end_sequential(daily_univariate_ts):
    result = tune(
        daily_univariate_ts,
        param_grid=GRID,
        horizon="30 days",
        initial="365 days",
        period="120 days",
        parallel=False,
    )
    assert len(result.scores) == 5  # 2 x 2 grid plus Prophet's defaults
    assert result.scores["is_default"].sum() == 1
    assert result.scores["rank"].tolist() == [1, 2, 3, 4, 5]
    assert result.scores["rmse"].is_monotonic_increasing
    assert set(result.best_params) == set(GRID)
    top = result.scores.iloc[0]
    assert all(top[k] == v for k, v in result.best_params.items())
    assert result.windows.n_folds == len(result.cutoffs) >= 2
    assert (
        "coverage" not in result.scores.columns
    )  # intervals were switched off while scoring

    model = result.best_model()
    assert (
        model.changepoint_prior_scale == result.best_params["changepoint_prior_scale"]
    )
    assert model.uncertainty_samples > 0  # restored for the final model
    future = model.make_future_dataframe(periods=5)
    assert "yhat_lower" in model.predict(future)


def test_tune_is_reproducible_and_parallel_matches_sequential(daily_univariate_ts):
    kwargs = {
        "param_grid": {"changepoint_prior_scale": [0.01, 0.5]},
        "horizon": "30 days",
        "initial": "365 days",
        "period": "180 days",
    }
    first = tune(daily_univariate_ts, parallel=False, include_defaults=False, **kwargs)
    second = tune(
        daily_univariate_ts,
        parallel=True,
        max_workers=2,
        include_defaults=False,
        **kwargs,
    )
    cols = ["changepoint_prior_scale", "rmse", "mae"]
    pd.testing.assert_frame_equal(first.scores[cols], second.scores[cols])
    assert first.best_params == second.best_params


def test_tune_coverage_metric_keeps_intervals(daily_univariate_ts):
    result = tune(
        daily_univariate_ts,
        param_grid={"changepoint_prior_scale": [0.01, 0.5]},
        metric="coverage",
        horizon="30 days",
        initial="365 days",
        period="180 days",
        parallel=False,
        prophet_kwargs={"uncertainty_samples": 100},
    )
    assert "coverage" in result.scores.columns
    assert result.scores["score"].is_monotonic_increasing


def test_module_exposes_expected_api():
    for name in ("tune", "TuningResult", "derive_windows", "default_param_grid"):
        assert hasattr(tuning, name)


def _record(name: str, level: int, msg: str) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 0, msg, None, None)


def test_quiet_filters_drop_expected_fallback_noise_only() -> None:
    from prophet.tuning import _keep_cmdstan_record, _keep_models_record

    assert not _keep_cmdstan_record(
        _record("cmdstanpy", logging.INFO, "Chain [1] start processing")
    )
    assert not _keep_cmdstan_record(
        _record(
            "cmdstanpy",
            logging.ERROR,
            "Chain [1] error: code '1' Operation not permitted",
        )
    )
    assert _keep_cmdstan_record(
        _record("cmdstanpy", logging.CRITICAL, "binary missing")
    )
    assert not _keep_models_record(
        _record(
            "prophet.models",
            logging.WARNING,
            "Optimization terminated abnormally. Falling back to Newton.",
        )
    )
    assert _keep_models_record(
        _record("prophet.models", logging.WARNING, "something else")
    )


def test_quiet_cmdstan_removes_its_filters() -> None:
    from prophet.tuning import _quiet_cmdstan

    cmdstan, models = (
        logging.getLogger("cmdstanpy"),
        logging.getLogger("prophet.models"),
    )
    before = (list(cmdstan.filters), list(models.filters))
    with _quiet_cmdstan():
        assert len(cmdstan.filters) == len(before[0]) + 1
        assert len(models.filters) == len(before[1]) + 1
    assert (cmdstan.filters, models.filters) == before
