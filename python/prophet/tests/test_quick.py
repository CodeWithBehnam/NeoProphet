import logging
import subprocess
import sys
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
    MissingRegressorValuesError,
    NonNumericValuesError,
    RegressorError,
    UnparseableDatesError,
    forecast,
    infer_frequency,
    normalise_input,
)
from prophet.quick import (  # noqa: E402
    _plan_weather,
    _temperature_for,
    horizon_to_periods,
    infer_frequency_details,
)

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


# ---- regressors ------------------------------------------------------------


def promo_frame(n: int = 120) -> pd.DataFrame:
    """A daily series that a binary promo lifts by 8, plus an unrelated price column."""
    rng = np.random.default_rng(7)
    ds = pd.date_range("2024-01-01", periods=n, freq="D")
    promo = (np.arange(n) % 9 == 0).astype(float)
    y = 20 + 0.05 * np.arange(n) + 8 * promo + rng.normal(0, 0.4, n)
    return pd.DataFrame({"ds": ds, "y": y, "promo": promo, "price": rng.normal(size=n)})


def future_frame(df: pd.DataFrame, periods: int = 7, **columns: float) -> pd.DataFrame:
    ds = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=periods, freq="D")
    return pd.DataFrame({"ds": ds, **{k: v for k, v in columns.items()}})


def test_regressor_columns_with_future_values_and_options():
    df = promo_frame()
    future = future_frame(df, 7, promo=1.0, price=0.0)
    result = forecast(
        df,
        horizon=7,
        regressors=["promo", "price"],
        future=future,
        regressor_options={
            "promo": {"mode": "multiplicative", "prior_scale": 5.0},
            "price": {"standardize": False},
        },
        uncertainty_samples=0,
    )
    extra = result.model.extra_regressors
    assert set(extra) == {"promo", "price"}
    assert extra["promo"]["mode"] == "multiplicative"
    assert extra["promo"]["prior_scale"] == 5.0
    assert extra["price"]["standardize"] is False
    assert extra["price"]["mode"] == "additive"
    assert result.regressors == ("promo", "price")
    assert len(result.forecast) == 7
    values = result.regressor_values
    assert values is not None
    assert list(values.columns) == ["ds", "promo", "price"]
    assert len(values) == len(df) + 7
    assert values["promo"].iloc[-7:].eq(1.0).all()
    assert "regressors: promo, price" in str(result.summary())
    # The promo effect is learnt: promo days forecast well above the baseline.
    baseline = forecast(
        df.drop(columns=["promo", "price"]), horizon=7, uncertainty_samples=0
    )
    assert result.forecast["yhat"].mean() > baseline.forecast["yhat"].mean() + 3


def test_regressor_columns_do_not_confuse_column_detection():
    df = promo_frame().drop(columns="price").rename(columns={"ds": "when", "y": "sales"})
    future = future_frame(df.rename(columns={"when": "ds"}), 3, promo=0.0)
    result = forecast(
        df, horizon=3, regressors=["promo"], future=future, uncertainty_samples=0
    )
    assert result.history.columns.tolist() == ["ds", "y"]
    assert result.regressors == ("promo",)


def test_regressor_series_mapping_covering_history_and_horizon():
    df = promo_frame()
    full = pd.date_range(df["ds"].min(), periods=len(df) + 5, freq="D")
    promo = pd.Series((np.arange(len(full)) % 9 == 0).astype(float), index=full)
    price = pd.Series(np.zeros(len(full)), index=full)
    result = forecast(
        df[["ds", "y"]],
        horizon=5,
        regressors={"promo": promo, "price": price},
        uncertainty_samples=0,
    )
    assert result.regressors == ("promo", "price")
    assert len(result.forecast) == 5
    assert result.regressor_values is not None
    assert len(result.regressor_values) == len(df) + 5


def test_regressor_dataframe_with_ds_column_and_unordered_rows():
    df = promo_frame()
    full = pd.date_range(df["ds"].min(), periods=len(df) + 4, freq="D")
    table = pd.DataFrame({"ds": full, "promo": (np.arange(len(full)) % 9 == 0) * 1.0})
    result = forecast(
        df[["ds", "y"]],
        horizon=4,
        regressors=table.sample(frac=1.0, random_state=0),
        uncertainty_samples=0,
    )
    assert result.regressors == ("promo",)
    assert result.regressor_values is not None
    assert result.regressor_values["ds"].is_monotonic_increasing


def test_boolean_regressor_values_become_flags():
    df = promo_frame().assign(promo=lambda d: d["promo"].astype(bool))
    future = future_frame(df, 3, promo=True)
    result = forecast(
        df, horizon=3, regressors=["promo"], future=future, uncertainty_samples=0
    )
    assert result.regressor_values is not None
    assert result.regressor_values["promo"].dtype == np.float64


def test_missing_future_values_name_the_timestamps():
    df = promo_frame()
    with pytest.raises(MissingRegressorValuesError) as info:
        forecast(df, horizon=14, regressors=["promo"], uncertainty_samples=0)
    message = str(info.value)
    assert "'promo'" in message
    assert "14 of 14 forecast timestamps" in message
    assert "2024-04-30" in message  # the first forecast day
    assert "and 9 more" in message
    assert "future=" in message
    assert isinstance(info.value, RegressorError)
    assert isinstance(info.value, ValueError)


def test_partly_missing_future_values_are_counted():
    df = promo_frame()
    future = future_frame(df, 5, promo=1.0).iloc[[0, 1, 3]]
    with pytest.raises(MissingRegressorValuesError) as info:
        forecast(
            df, horizon=5, regressors=["promo"], future=future, uncertainty_samples=0
        )
    message = str(info.value)
    assert "2 of 5 forecast timestamps" in message
    assert "2024-04-32" not in message
    assert "2024-05-02" in message and "2024-05-04" in message


def test_gaps_in_history_say_how_many_rows():
    df = promo_frame()
    df.loc[[10, 11, 12], "promo"] = np.nan
    future = future_frame(df, 2, promo=0.0)
    with pytest.raises(MissingRegressorValuesError) as info:
        forecast(
            df, horizon=2, regressors=["promo"], future=future, uncertainty_samples=0
        )
    message = str(info.value)
    assert "3 of 120 training timestamps" in message
    assert "2024-01-11" in message


def test_mapping_that_stops_at_the_end_of_history_fails_on_the_horizon():
    df = promo_frame()
    promo = pd.Series(df["promo"].to_numpy(), index=df["ds"])
    with pytest.raises(MissingRegressorValuesError, match="7 of 7 forecast"):
        forecast(
            df[["ds", "y"]],
            horizon=7,
            regressors={"promo": promo},
            uncertainty_samples=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"regressors": ["yhat"]}, "reserved"),
        ({"regressors": ["trend"]}, "reserved"),
        ({"regressors": ["nope"]}, "not a column.*Available columns"),
        ({"regressors": ["promo", "promo"]}, "unique"),
        ({"regressors": [3]}, "column names"),
        ({"regressors": 3.5}, "list of column names"),
        ({"regressors": ["y"]}, "reserved"),
        ({"regressors": ["promo"], "regressor_options": {"price": {}}}, "not regressors"),
        (
            {"regressors": ["promo"], "regressor_options": {"promo": {"colour": 1}}},
            "unknown setting",
        ),
        (
            {"regressors": ["promo"], "regressor_options": {"promo": {"mode": "x"}}},
            "mode",
        ),
        (
            {"regressors": ["promo"], "regressor_options": {"promo": {"prior_scale": -1}}},
            "Prior scale",
        ),
        ({"regressors": ["promo"], "regressor_options": {"promo": 5}}, "mapping"),
    ],
)
def test_regressor_validation_errors(kwargs: dict[str, Any], match: str):
    df = promo_frame()
    future = future_frame(df, 3, promo=0.0, price=0.0)
    with pytest.raises(RegressorError, match=match):
        forecast(df, horizon=3, future=future, uncertainty_samples=0, **kwargs)


def test_regressor_values_must_be_numeric_and_finite():
    df = promo_frame()
    words = df.assign(promo=["a", "b"] * 60)
    with pytest.raises(RegressorError, match="not numeric"):
        forecast(words, horizon=3, regressors=["promo"], uncertainty_samples=0)
    infinite = df.assign(promo=np.where(np.arange(120) == 5, np.inf, 0.0))
    with pytest.raises(RegressorError, match="infinite"):
        forecast(infinite, horizon=3, regressors=["promo"], uncertainty_samples=0)
    stamps = pd.date_range("2024-01-01", periods=130, freq="D")
    with pytest.raises(RegressorError, match="DatetimeIndex"):
        forecast(
            df[["ds", "y"]],
            horizon=3,
            regressors={"promo": pd.Series(np.zeros(130))},
            uncertainty_samples=0,
        )
    with pytest.raises(RegressorError, match="pandas Series"):
        forecast(
            df[["ds", "y"]],
            horizon=3,
            regressors={"promo": np.zeros(130)},  # type: ignore[dict-item]
            uncertainty_samples=0,
        )
    dup = pd.Series(np.zeros(130), index=stamps.insert(0, stamps[0])[:130])
    with pytest.raises(RegressorError, match="duplicate"):
        forecast(
            df[["ds", "y"]],
            horizon=3,
            regressors={"promo": dup},
            uncertainty_samples=0,
        )


def test_column_regressors_need_a_dataframe():
    series = pd.Series(np.arange(30.0), index=pd.date_range("2024-01-01", periods=30))
    with pytest.raises(RegressorError, match="Series"):
        forecast(series, horizon=3, regressors=["promo"], uncertainty_samples=0)


def test_future_without_regressors_is_ignored_with_a_warning(
    caplog: pytest.LogCaptureFixture,
):
    df = promo_frame()[["ds", "y"]]
    with caplog.at_level(logging.WARNING, logger="prophet.quick"):
        forecast(df, horizon=3, future=future_frame(df, 3), uncertainty_samples=0)
    assert "future= ignored" in caplog.text


def test_tune_receives_regressor_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import prophet.tuning as tuning

    calls: dict[str, Any] = {}

    def fake_tune(df: pd.DataFrame, **kwargs: Any) -> Any:
        calls.update(kwargs)
        calls["columns"] = list(df.columns)

        class Best:
            best_params = {"changepoint_prior_scale": 0.1}

        return Best()

    monkeypatch.setattr(tuning, "tune", fake_tune)
    df = promo_frame()
    forecast(
        df,
        horizon=3,
        tune=True,
        regressors=["promo"],
        future=future_frame(df, 3, promo=0.0),
        uncertainty_samples=0,
    )
    assert calls["regressors"] == ("promo",)
    assert "promo" in calls["columns"]


# ---- weather ----------------------------------------------------------------


class FakeWeather:
    """Stands in for prophet.weather.fetch_temperature and records its calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, location: Any, start: Any, end: Any, *, freq: str = "daily", **_: Any
    ) -> Any:
        from prophet.weather import Location, WeatherResult

        self.calls.append(
            {
                "location": location,
                "start": pd.Timestamp(start),
                "end": pd.Timestamp(end),
                "freq": freq,
            }
        )
        if freq == "daily":
            ds = pd.date_range(
                pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D"
            )
            temperature = (ds - pd.Timestamp("2024-01-01")).days.to_numpy() * 0.1
        else:
            first = pd.Timestamp(start).floor("h")
            last = pd.Timestamp(end).ceil("h")
            if last == last.normalize():
                last = last + pd.Timedelta(hours=23)
            ds = pd.date_range(first, last, freq="h")
            temperature = (ds - first).total_seconds().to_numpy() / 3600.0
        source = np.where(ds >= pd.Timestamp("2024-04-25"), "climatology", "archive")
        return WeatherResult(
            location=Location("Testville, England", 51.5, -0.1, "Europe/London"),
            frame=pd.DataFrame(
                {"ds": ds, "temperature": temperature, "source": source}
            ),
        )


@pytest.fixture
def fake_weather(monkeypatch: pytest.MonkeyPatch) -> FakeWeather:
    weather = pytest.importorskip("prophet.weather")
    fake = FakeWeather()
    monkeypatch.setattr(weather, "fetch_temperature", fake)
    return fake


def test_weather_adds_a_temperature_regressor_and_keeps_the_result(
    fake_weather: FakeWeather,
):
    df = promo_frame()[["ds", "y"]]
    result = forecast(df, horizon=10, weather="Testville", uncertainty_samples=0)
    assert result.regressors == ("temperature",)
    assert "temperature" in result.model.extra_regressors
    assert len(result.forecast) == 10
    call = fake_weather.calls[0]
    assert call["location"] == "Testville"
    assert call["freq"] == "daily"
    assert call["start"] == df["ds"].min()
    assert call["end"] == df["ds"].max() + pd.Timedelta(days=10)
    # The WeatherResult is kept whole, so charts can tell measured from climatology.
    assert result.weather is not None
    assert result.weather.location.name == "Testville, England"
    assert set(result.weather.frame["source"]) == {"archive", "climatology"}
    values = result.regressor_values
    assert values is not None
    assert len(values) == len(df) + 10  # history rows and forecast rows
    expected = result.weather.frame.set_index("ds")["temperature"].reindex(values["ds"])
    np.testing.assert_allclose(values["temperature"].to_numpy(), expected.to_numpy())
    text = str(result.summary())
    assert "temperature" in text and "Testville, England" in text
    assert result.summary().weather_location == "Testville, England"


def test_weather_accepts_coordinates_and_combines_with_regressors(
    fake_weather: FakeWeather,
):
    df = promo_frame()
    future = future_frame(df, 5, promo=0.0)
    result = forecast(
        df[["ds", "y", "promo"]],
        horizon=5,
        weather=(51.5, -0.12),
        regressors=["promo"],
        future=future,
        regressor_options={"temperature": {"prior_scale": 2.0}},
        uncertainty_samples=0,
    )
    assert fake_weather.calls[0]["location"] == (51.5, -0.12)
    assert result.regressors == ("promo", "temperature")
    assert result.model.extra_regressors["temperature"]["prior_scale"] == 2.0


def test_weather_with_skipped_days_only_uses_kept_timestamps(
    fake_weather: FakeWeather,
):
    df = promo_frame()[["ds", "y"]]
    result = forecast(
        df, horizon=5, weather="Testville", skip_days="weekends", uncertainty_samples=0
    )
    assert not result.forecast["ds"].dt.dayofweek.isin({5, 6}).any()
    assert result.regressor_values is not None
    assert not result.regressor_values["ds"].dt.dayofweek.isin({5, 6}).any()
    assert result.regressors == ("temperature",)


def test_weather_for_sub_daily_data_is_hourly_and_interpolated(
    fake_weather: FakeWeather,
):
    ds = pd.date_range("2024-03-04", periods=24 * 14, freq="30min")
    series = pd.Series(
        10 + np.sin(np.arange(len(ds)) / 6.0), index=ds
    )
    result = forecast(series, horizon=6, weather="Testville", uncertainty_samples=0)
    assert fake_weather.calls[0]["freq"] == "hourly"
    values = result.regressor_values
    assert values is not None
    half_hour = values[values["ds"].dt.minute == 30]["temperature"].to_numpy()
    on_hour = values[values["ds"].dt.minute == 0]["temperature"].to_numpy()
    # A half hour lies midway between its hourly neighbours (the fake rises 1 per hour).
    np.testing.assert_allclose(half_hour[:5], on_hour[:5] + 0.5)


def test_weather_for_weekly_data_averages_each_week(fake_weather: FakeWeather):
    weekly = pd.Series(
        np.arange(40.0), index=pd.date_range("2024-01-07", periods=40, freq="W-SUN")
    )
    result = forecast(weekly, horizon=2, weather="Testville", uncertainty_samples=0)
    assert fake_weather.calls[0]["freq"] == "daily"
    values = result.regressor_values
    assert values is not None
    # The fake temperature is 0.1 per day since 1 January; the week ending Sunday 7 January
    # covers 1 to 7 January, so its mean is the value for 4 January.
    assert values["temperature"].iloc[0] == pytest.approx(0.3)
    assert values["temperature"].iloc[1] == pytest.approx(1.0)


def test_weather_plans_for_month_labels():
    month_end = pd.DatetimeIndex(pd.date_range("2024-01-31", periods=3, freq="ME"))
    plan = _plan_weather(month_end, "ME")
    assert plan.kind == "daily"
    assert plan.start == pd.Timestamp("2024-01-01")
    assert plan.end == pd.Timestamp("2024-03-31")
    month_start = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="MS"))
    plan = _plan_weather(month_start, "MS")
    assert plan.start == pd.Timestamp("2024-01-01")
    assert plan.end == pd.Timestamp("2024-03-31")
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", "2024-03-31", freq="D"),
            "temperature": 1.0,
        }
    )
    assert _temperature_for(plan, frame).tolist() == [1.0, 1.0, 1.0]


def test_weather_errors_and_clashes():
    df = promo_frame()
    with pytest.raises(RegressorError, match="place name"):
        forecast(df, horizon=3, weather=42, uncertainty_samples=0)  # type: ignore[arg-type]
    with pytest.raises(RegressorError, match="place name"):
        forecast(df, horizon=3, weather="  ", uncertainty_samples=0)
    with pytest.raises(RegressorError, match="clashes"):
        forecast(
            df.rename(columns={"price": "temperature"}),
            horizon=3,
            weather="London",
            regressors=["temperature"],
            uncertainty_samples=0,
        )


def test_missing_weather_module_fails_only_when_weather_is_used(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "prophet.weather", None)
    df = promo_frame()[["ds", "y"]]
    assert len(forecast(df, horizon=3, uncertainty_samples=0).forecast) == 3
    with pytest.raises(ImportError, match="prophet.weather"):
        forecast(df, horizon=3, weather="London", uncertainty_samples=0)


def test_importing_prophet_does_not_import_the_weather_module():
    code = "import sys, prophet; assert 'prophet.weather' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
