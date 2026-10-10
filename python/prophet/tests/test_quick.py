import logging
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from prophet import (  # noqa: E402
    AllNullValuesError,
    ColumnSelectionError,
    DuplicateTimestampsError,
    ForecastInputError,
    FrequencyInferenceError,
    HorizonError,
    InsufficientDataError,
    NonNumericValuesError,
    UnparseableDatesError,
    forecast,
    infer_frequency,
    normalise_input,
)
from prophet.quick import horizon_to_periods, infer_frequency_details  # noqa: E402

N = 10
DATES = pd.date_range("2024-01-01", periods=N, freq="D")
VALUES = np.arange(N, dtype=float)


def expected() -> pd.DataFrame:
    return pd.DataFrame({"ds": DATES, "y": VALUES})


# ---- input shapes ---------------------------------------------------------


def test_ds_y_frame():
    out = normalise_input(expected())
    pd.testing.assert_frame_equal(out, expected())


def test_any_names_inferred():
    df = pd.DataFrame({"when": DATES, "sales": VALUES, "label": list("abcdefghij")})
    pd.testing.assert_frame_equal(normalise_input(df), expected())


def test_string_dates_inferred():
    df = pd.DataFrame({"day": DATES.strftime("%Y-%m-%d"), "n": VALUES})
    pd.testing.assert_frame_equal(normalise_input(df), expected())


def test_explicit_columns():
    df = pd.DataFrame(
        {"a": DATES, "b": DATES + pd.Timedelta(days=99), "c": VALUES, "d": VALUES * 2}
    )
    out = normalise_input(df, date_col="a", value_col="c")
    pd.testing.assert_frame_equal(out, expected())


def test_datetime_index_frame():
    df = pd.DataFrame({"sales": VALUES}, index=DATES)
    pd.testing.assert_frame_equal(normalise_input(df), expected())


def test_series():
    pd.testing.assert_frame_equal(
        normalise_input(pd.Series(VALUES, index=DATES)), expected()
    )


def test_unsorted_input_is_sorted():
    shuffled = expected().iloc[::-1]
    pd.testing.assert_frame_equal(normalise_input(shuffled), expected())


def test_null_rows_dropped_and_logged(caplog: pytest.LogCaptureFixture):
    df = expected()
    df.loc[3, "y"] = np.nan
    with caplog.at_level(logging.WARNING, logger="prophet.quick"):
        out = normalise_input(df)
    assert len(out) == N - 1
    assert "Dropped 1 row" in caplog.text


# ---- timezone -------------------------------------------------------------


def test_tz_wall_time_default(caplog: pytest.LogCaptureFixture):
    aware = pd.Series(
        VALUES, index=DATES.tz_localize("Europe/London").tz_convert("Asia/Tokyo")
    )
    with caplog.at_level(logging.WARNING, logger="prophet.quick"):
        out = normalise_input(aware)
    assert out["ds"].dt.tz is None
    assert out["ds"].iloc[0] == pd.Timestamp("2024-01-01 09:00")  # Tokyo wall time
    assert "wall-clock" in caplog.text


def test_tz_utc(caplog: pytest.LogCaptureFixture):
    aware = pd.Series(VALUES, index=DATES.tz_localize("Asia/Tokyo"))
    with caplog.at_level(logging.WARNING, logger="prophet.quick"):
        out = normalise_input(aware, tz="utc")
    assert out["ds"].iloc[0] == pd.Timestamp("2023-12-31 15:00")
    assert "UTC" in caplog.text


def test_bad_tz_option():
    with pytest.raises(ForecastInputError):
        normalise_input(expected(), tz="local")  # type: ignore[arg-type]


# ---- errors ---------------------------------------------------------------


def test_error_types_are_value_errors():
    for exc in (
        ColumnSelectionError,
        UnparseableDatesError,
        DuplicateTimestampsError,
        InsufficientDataError,
        AllNullValuesError,
        NonNumericValuesError,
        FrequencyInferenceError,
        HorizonError,
    ):
        assert issubclass(exc, ValueError)
        assert issubclass(exc, ForecastInputError)
    assert issubclass(AllNullValuesError, InsufficientDataError)


def test_missing_explicit_column():
    with pytest.raises(ColumnSelectionError, match="nope"):
        normalise_input(expected(), value_col="nope")


def test_no_date_column():
    with pytest.raises(ColumnSelectionError, match="date"):
        normalise_input(pd.DataFrame({"a": VALUES, "b": VALUES}))


def test_ambiguous_date_columns():
    df = pd.DataFrame({"a": DATES, "b": DATES, "c": VALUES})
    with pytest.raises(ColumnSelectionError, match="date_col"):
        normalise_input(df)


def test_ambiguous_value_columns():
    df = pd.DataFrame({"a": DATES, "b": VALUES, "c": VALUES})
    with pytest.raises(ColumnSelectionError, match="value_col"):
        normalise_input(df)


def test_no_numeric_column():
    with pytest.raises(ColumnSelectionError, match="numeric"):
        normalise_input(pd.DataFrame({"a": DATES}))


def test_series_without_datetime_index():
    with pytest.raises(ColumnSelectionError, match="DatetimeIndex"):
        normalise_input(pd.Series(VALUES))


def test_wrong_type():
    with pytest.raises(ForecastInputError):
        normalise_input([1, 2, 3])  # type: ignore[arg-type]


def test_unparseable_dates():
    df = pd.DataFrame(
        {"ds": ["2024-01-01", "not a date", "2024-01-03"], "y": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(UnparseableDatesError):
        normalise_input(df)


def test_missing_dates():
    df = expected()
    df.loc[2, "ds"] = pd.NaT
    with pytest.raises(UnparseableDatesError, match="missing"):
        normalise_input(df)


def test_numeric_dates_rejected():
    with pytest.raises(UnparseableDatesError):
        normalise_input(pd.DataFrame({"ds": [1, 2, 3], "y": [1.0, 2.0, 3.0]}))


def test_duplicate_timestamps():
    df = pd.concat([expected(), expected().iloc[:1]])
    with pytest.raises(DuplicateTimestampsError, match="2024-01-01"):
        normalise_input(df)


def test_fewer_than_two_values():
    df = expected()
    df["y"] = np.nan
    df.loc[0, "y"] = 1.0
    with pytest.raises(InsufficientDataError):
        normalise_input(df)


def test_all_null():
    df = expected()
    df["y"] = np.nan
    with pytest.raises(AllNullValuesError):
        normalise_input(df)


def test_non_numeric_y():
    df = pd.DataFrame({"ds": DATES, "y": list("abcdefghij")})
    with pytest.raises(NonNumericValuesError):
        normalise_input(df)


def test_bool_and_inf_y_rejected():
    with pytest.raises(NonNumericValuesError, match="boolean"):
        normalise_input(pd.DataFrame({"ds": DATES, "y": [True, False] * 5}))
    inf = expected()
    inf.loc[1, "y"] = np.inf
    with pytest.raises(NonNumericValuesError, match="infinite"):
        normalise_input(inf)


def test_numeric_strings_in_y_are_coerced():
    df = pd.DataFrame({"ds": DATES, "y": [str(v) for v in VALUES]})
    pd.testing.assert_frame_equal(normalise_input(df), expected())


# ---- frequency ------------------------------------------------------------


@pytest.mark.parametrize("alias", ["D", "h", "W-SUN", "ME", "MS", "15min", "B"])
def test_regular_frequencies(alias: str):
    dates = pd.date_range("2024-01-01", periods=30, freq=alias)
    details = infer_frequency_details(dates)
    assert details.freq == alias
    assert details.regular
    # Usable for make_future_dataframe style extension.
    assert len(pd.date_range(dates[-1], periods=3, freq=details.freq)) == 3


def test_irregular_daily_uses_modal_difference():
    dates = pd.date_range("2024-01-01", periods=40, freq="D").delete([5, 6, 20])
    details = infer_frequency_details(dates)
    assert details.freq == "D"
    assert not details.regular


def test_irregular_weekly_and_hourly():
    weekly = pd.date_range("2024-01-01", periods=30, freq="7D").delete([4, 9])
    assert infer_frequency(weekly) == "7D"
    hourly = pd.date_range("2024-01-01", periods=50, freq="h").delete([10, 11, 30])
    assert infer_frequency(hourly) == "h"


def test_irregular_monthly_gives_month_alias():
    ends = pd.date_range("2023-01-31", periods=20, freq="ME").delete([3, 8])
    starts = pd.date_range("2023-01-01", periods=20, freq="MS").delete([3, 8])
    assert infer_frequency(ends) == "ME"
    assert infer_frequency(starts) == "MS"


def test_two_points_and_too_few():
    assert infer_frequency(pd.date_range("2024-01-01", periods=2, freq="D")) == "D"
    with pytest.raises(FrequencyInferenceError):
        infer_frequency(pd.date_range("2024-01-01", periods=1))


# ---- horizon --------------------------------------------------------------


def test_horizon_int():
    assert horizon_to_periods(7, pd.Timestamp("2024-01-01"), "D") == 7


@pytest.mark.parametrize(
    ("horizon", "last", "freq", "periods"),
    [
        ("12W", "2024-01-07", "W-SUN", 12),
        ("90D", "2024-01-01", "D", 90),
        ("2 weeks", "2024-01-01", "D", 14),
        ("6 months", "2024-01-31", "ME", 6),
        ("1y", "2024-01-31", "ME", 12),
        ("1Q", "2024-01-01", "MS", 3),
        ("48h", "2024-01-01", "h", 48),
        ("1D", "2024-01-01", "W-SUN", 1),  # never fewer than one period
    ],
)
def test_horizon_strings(horizon: str, last: str, freq: str, periods: int):
    assert horizon_to_periods(horizon, pd.Timestamp(last), freq) == periods


@pytest.mark.parametrize(
    "bad", [0, -3, True, 2.5, None, "abc", "12", "5m", "3parsecs", "0D"]
)
def test_bad_horizon(bad: object):
    with pytest.raises(HorizonError):
        horizon_to_periods(bad, pd.Timestamp("2024-01-01"), "D")  # type: ignore[arg-type]


# ---- end to end -----------------------------------------------------------


@pytest.fixture(scope="module")
def weekly_series() -> pd.Series:
    idx = pd.date_range("2022-01-02", periods=80, freq="W-SUN")
    t = np.arange(80)
    return pd.Series(10 + 0.1 * t + np.sin(t / 8), index=idx, name="sales")


def test_forecast_end_to_end(weekly_series: pd.Series):
    result = forecast(weekly_series, horizon="6W", uncertainty_samples=50)
    assert result.freq == "W-SUN"
    assert result.horizon_periods == 6
    fc = result.forecast
    assert list(fc.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(fc) == 6
    assert fc["ds"].min() > weekly_series.index.max()
    assert fc["yhat"].notna().all()
    assert len(result.to_frame(include_history=True)) == 86
    summary = result.summary()
    assert summary.n_rows_used == 80
    assert summary.horizon_periods == 6
    assert "W-SUN" in str(summary)
    # Frozen
    with pytest.raises(AttributeError):
        result.freq = "D"  # type: ignore[misc]


def test_forecast_plots_and_options(weekly_series: pd.Series):
    df = weekly_series.rename_axis("when").reset_index()
    result = forecast(
        df,
        horizon=4,
        include_history=True,
        interval_width=0.9,
        country_holidays="GB",
        uncertainty_samples=50,
        yearly_seasonality=False,
    )
    assert result.model.interval_width == 0.9
    assert result.model.country_holidays == "GB"
    assert len(result.forecast) == 84
    assert result.plot() is not None
    assert result.plot_components() is not None


def test_forecast_explicit_freq_and_irregular_warning(caplog: pytest.LogCaptureFixture):
    idx = pd.date_range("2024-01-01", periods=60, freq="D").delete([10, 11, 30])
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    with caplog.at_level(logging.WARNING, logger="prophet.quick"):
        result = forecast(s, horizon=3, uncertainty_samples=0)
    assert "irregular" in caplog.text
    assert result.freq == "D"
    assert "yhat_lower" not in result.forecast.columns
    assert forecast(s, horizon=3, freq="D", uncertainty_samples=0).freq_method == "user"


def test_forecast_rejects_bad_inputs(weekly_series: pd.Series):
    with pytest.raises(FrequencyInferenceError):
        forecast(weekly_series, horizon=2, freq="nonsense")
    with pytest.raises(HorizonError):
        forecast(weekly_series, horizon="soon")
    with pytest.raises(ForecastInputError, match="logistic"):
        forecast(weekly_series, horizon=2, growth="logistic")


class _FakeTuning:
    def __init__(self, best_params: dict[str, Any]) -> None:
        self.best_params = best_params


def test_forecast_tune_uses_tuned_params_but_explicit_args_win(
    daily_univariate_ts: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prophet.tuning as tuning

    calls: dict[str, Any] = {}

    def fake_tune(df: pd.DataFrame, **kwargs: Any) -> _FakeTuning:
        calls.update(kwargs)
        return _FakeTuning(
            {"changepoint_prior_scale": 0.5, "seasonality_mode": "multiplicative"}
        )

    monkeypatch.setattr(tuning, "tune", fake_tune)
    result = forecast(
        daily_univariate_ts,
        horizon=7,
        tune=True,
        seasonality_mode="additive",
        country_holidays="GB",
        uncertainty_samples=0,
    )
    assert "seasonality_mode" not in calls["param_grid"]
    assert "changepoint_prior_scale" in calls["param_grid"]
    assert "holidays" in calls["prophet_kwargs"]
    assert result.model.changepoint_prior_scale == 0.5
    assert result.model.seasonality_mode == "additive"
    assert result.model.holidays is None
    assert result.tuning is not None


def test_forecast_tune_skipped_when_every_param_is_explicit(
    daily_univariate_ts: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prophet.tuning as tuning

    def fail_tune(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("tune should not run")

    monkeypatch.setattr(tuning, "tune", fail_tune)
    result = forecast(
        daily_univariate_ts,
        horizon=7,
        tune=True,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=1.0,
        seasonality_mode="additive",
        uncertainty_samples=0,
    )
    assert result.tuning is None


def test_forecast_without_tune_has_no_tuning(daily_univariate_ts: pd.DataFrame) -> None:
    result = forecast(daily_univariate_ts, horizon=7, uncertainty_samples=0)
    assert result.tuning is None
