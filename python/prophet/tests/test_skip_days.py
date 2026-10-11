"""skip_days / skip_hours: parsing, training and forecast frames, and the closed-gap axis."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.dates import date2num  # noqa: E402

from prophet import SkipDaysError, forecast, skip_axis  # noqa: E402
from prophet.plot import (  # noqa: E402
    SkipSpecError,
    parse_skip_days,
    parse_skip_hours,
    plot_plotly,
)
from prophet.quick import (  # noqa: E402
    ForecastInputError,
    HorizonError,
    InsufficientDataError,
    _future_dates,
    _parse_skip,
)

WEEKEND = {5, 6}


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def weekday_series(periods: int = 140, seed: int = 0) -> pd.Series:
    """A trading-style series: Monday high, Friday low, Saturdays and Sundays absent."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    dow = index.dayofweek.to_numpy()
    values = (
        50
        + 0.05 * np.arange(periods)
        + np.where(dow == 0, 6, np.where(dow == 4, -5, 0))
        + rng.normal(0, 0.5, periods)
    )
    return pd.Series(values, index=index, name="sales")


def every_day_series() -> pd.Series:
    return weekday_series()


def weekdays_only() -> pd.Series:
    series = every_day_series()
    return series[~series.index.dayofweek.isin(WEEKEND)]


@pytest.fixture(scope="module")
def weekend_result():  # noqa: ANN201
    return forecast(
        every_day_series(), horizon="3W", skip_days="weekends", uncertainty_samples=30
    )


# ---- parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("Sat", {5}),
        ("Saturday", {5}),
        ("sat", {5}),
        ("  SATURDAY ", {5}),
        ("Saturdays", {5}),
        (5, {5}),
        (np.int64(5), {5}),
        ("weekends", {5, 6}),
        ("weekend", {5, 6}),
        (["Sat", "Sun"], {5, 6}),
        (("mon", 2, "friday"), {0, 2, 4}),
        (range(5, 7), {5, 6}),
        (None, set()),
        ([], set()),
        ("tues", {1}),
        ("Thu", {3}),
    ],
)
def test_parse_skip_days_accepts_names_numbers_and_shortcut(
    spec: Any, expected: set[int]
) -> None:
    assert parse_skip_days(spec) == frozenset(expected)


@pytest.mark.parametrize(
    "bad", ["Funday", 7, -1, True, "5", 3.5, [None], "weekdays", list(range(7))]
)
def test_parse_skip_days_rejects_bad_values(bad: Any) -> None:
    with pytest.raises(SkipSpecError):
        parse_skip_days(bad)


def test_all_seven_days_cannot_be_skipped() -> None:
    with pytest.raises(SkipSpecError, match="every day"):
        parse_skip_days(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])


def test_parse_skip_hours() -> None:
    assert parse_skip_hours(range(0, 8)) == frozenset(range(8))
    assert parse_skip_hours(3) == frozenset({3})
    assert parse_skip_hours(None) == frozenset()
    for bad in (24, -1, "night", True, 2.5):
        with pytest.raises(SkipSpecError):
            parse_skip_hours(bad)
    with pytest.raises(SkipSpecError, match="every hour"):
        parse_skip_hours(range(24))


def test_forecast_wraps_bad_specs_in_the_input_error_family() -> None:
    with pytest.raises(SkipDaysError, match="Funday") as info:
        forecast(every_day_series(), horizon=3, skip_days="Funday")
    assert isinstance(info.value, ForecastInputError)
    assert isinstance(info.value, ValueError)


# ---- the axis mapping -------------------------------------------------------


def test_positions_are_contiguous_across_a_weekend() -> None:
    days = pd.date_range("2024-01-01", periods=21)
    axis = skip_axis(days, skip_days="weekends")
    kept = days[axis.keeps(days)]
    x = np.asarray(axis.to_x(kept))
    assert len(kept) == 15
    np.testing.assert_allclose(np.diff(x), 1.0)  # Friday to Monday is one step
    # A skipped timestamp sits where the next kept time starts.
    saturday = float(axis.to_x(pd.Timestamp("2024-01-06 12:00")))
    assert saturday == float(axis.to_x(pd.Timestamp("2024-01-08")))


def test_positions_round_trip_and_follow_hours_and_gaps() -> None:
    hours = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    axis = skip_axis(hours, skip_days=["Sat", "Sun"], skip_hours=range(0, 8))
    kept = hours[axis.keeps(hours)]
    assert len(kept) == 8 * 16  # eight weekdays of sixteen kept hours
    x = np.asarray(axis.to_x(kept))
    assert (np.diff(x) > 0).all()
    np.testing.assert_allclose(np.diff(x[:5]), 1 / 24)
    back = axis.from_x(x)
    assert list(back) == list(kept)
    # 07:00 on Tuesday and 08:00 on Tuesday are one kept hour apart.
    gap = float(axis.to_x(pd.Timestamp("2024-01-02 08:00"))) - float(
        axis.to_x(pd.Timestamp("2024-01-02 07:00"))
    )
    assert gap == pytest.approx(0.0)
    # Last kept hour of Monday to first kept hour of Tuesday: the night is closed.
    mon_last = float(axis.to_x(pd.Timestamp("2024-01-01 23:00")))
    tue_first = float(axis.to_x(pd.Timestamp("2024-01-02 08:00")))
    assert tue_first - mon_last == pytest.approx(1 / 24)


def test_a_real_gap_that_is_not_skipped_stays_open() -> None:
    days = pd.date_range("2024-01-01", periods=30)
    axis = skip_axis(days, skip_days="weekends")
    # Wednesday 3rd to Wednesday 10th: five kept days, not seven.
    span = float(axis.to_x(pd.Timestamp("2024-01-10"))) - float(
        axis.to_x(pd.Timestamp("2024-01-03"))
    )
    assert span == pytest.approx(5.0)


def test_skip_axis_errors_and_limits() -> None:
    days = pd.date_range("2024-01-06", periods=2)  # Saturday and Sunday only
    with pytest.raises(SkipSpecError, match="nothing is left"):
        skip_axis(days, skip_days="weekends")
    axis = skip_axis(pd.date_range("2024-01-01", periods=14), skip_days="weekends")
    low, high = axis.limits
    assert high - low == pytest.approx(9.0)
    assert axis.active
    assert not skip_axis(days).active


def test_keeps_accepts_timezone_aware_input_without_shifting_wall_time() -> None:
    axis = skip_axis(pd.date_range("2024-01-01", periods=3), skip_days="Mon")
    aware = pd.date_range("2024-01-01 00:30", periods=2, freq="D", tz="Asia/Tokyo")
    assert list(axis.keeps(aware)) == [False, True]


def test_apply_warps_a_matplotlib_axis_and_keeps_dates_as_data() -> None:
    days = pd.date_range("2024-01-01", periods=21)
    axis = skip_axis(days, skip_days="weekends")
    fig, ax = plt.subplots()
    ax.plot(days[axis.keeps(days)], np.arange(15))
    ax.set_xlim(days[0], days[-1])
    before = ax.get_xlim()
    axis.apply(ax)
    assert ax.get_xlim() == before
    assert ax.get_xscale() == "function"
    fig.canvas.draw()
    pixels = ax.transData.transform(
        [
            (date2num(d), 0.0)
            for d in pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"])
        ]
    )[:, 0]
    assert pixels[2] - pixels[1] == pytest.approx(pixels[1] - pixels[0], rel=1e-6)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels and all(labels)


def test_position_mode_locator_and_formatter_label_kept_dates_only() -> None:
    days = pd.date_range("2024-01-01", periods=84)
    axis = skip_axis(days, skip_days="weekends")
    kept = days[axis.keeps(days)]
    fig, ax = plt.subplots()
    ax.plot(axis.to_x(kept), np.arange(len(kept)))
    ax.xaxis.set_major_locator(axis.tick_locator())
    ax.xaxis.set_major_formatter(axis.tick_formatter())
    fig.canvas.draw()
    ticks = ax.get_xticks()
    stamps = axis.from_x(ticks)
    assert len(stamps) >= 3
    assert axis.keeps(stamps).all()
    assert all(t.get_text() for t in ax.get_xticklabels() if t.get_text() != "")


# ---- forecast() --------------------------------------------------------------


def test_weekends_are_excluded_from_training_and_forecast(weekend_result: Any) -> None:
    result = weekend_result
    assert not result.history["ds"].dt.dayofweek.isin(WEEKEND).any()
    assert not result.forecast["ds"].dt.dayofweek.isin(WEEKEND).any()
    assert not result.predictions["ds"].dt.dayofweek.isin(WEEKEND).any()
    assert (
        not result.to_frame(include_history=True)["ds"].dt.dayofweek.isin(WEEKEND).any()
    )
    assert result.skip_days == (5, 6)
    assert result.skip_hours == ()
    assert result.rows_skipped == 40  # 20 weekends in 140 days
    assert len(result.forecast) == 15  # three weeks of weekdays
    assert result.horizon_periods == 15
    summary = result.summary()
    assert summary.skip_days == ("Sat", "Sun")
    assert summary.rows_skipped == 40
    assert "skipping Sat, Sun (40 rows dropped)" in str(summary)


def test_dropped_rows_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="prophet.quick"):
        forecast(
            every_day_series(),
            horizon=3,
            skip_days=["Sat", "Sunday"],
            uncertainty_samples=0,
        )
    assert "Dropped 40 of 140 row(s)" in caplog.text


def test_weekday_only_data_with_skipped_weekends_is_not_called_irregular(
    caplog: pytest.LogCaptureFixture,
) -> None:
    series = weekdays_only()
    # The data never has Wednesdays either, which looks gappy only because of the skip.
    series = series[series.index.dayofweek != 2]
    with caplog.at_level(logging.WARNING, logger="prophet.quick"):
        result = forecast(
            series, horizon=5, skip_days=["weekends", "Wed"], uncertainty_samples=0
        )
    assert "irregular" not in caplog.text
    assert result.rows_skipped == 0
    assert not result.forecast["ds"].dt.dayofweek.isin({2, 5, 6}).any()


def test_a_genuine_gap_still_warns_when_skipping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    series = every_day_series().drop(pd.Timestamp("2024-02-14"))
    with caplog.at_level(logging.WARNING, logger="prophet.quick"):
        forecast(series, horizon=3, skip_days="weekends", uncertainty_samples=0)
    assert "irregular" in caplog.text


def test_integer_horizon_counts_rows_and_duration_counts_calendar_time() -> None:
    last = pd.Timestamp("2024-05-17")  # a Friday
    axis = _parse_skip("weekends", None)
    rows = _future_dates(last, "D", 5, axis)
    assert list(rows.strftime("%a")) == ["Mon", "Tue", "Wed", "Thu", "Fri"]
    window = _future_dates(last, "D", "1W", axis)
    assert list(window.strftime("%d")) == [
        "20",
        "21",
        "22",
        "23",
        "24",
    ]  # 18 to 24 May minus Sat and Sun
    # Without skipping nothing changes.
    plain = _future_dates(last, "D", 5, _parse_skip(None, None))
    assert list(plain.strftime("%a")) == ["Sat", "Sun", "Mon", "Tue", "Wed"]


def test_a_horizon_that_only_covers_skipped_days_is_an_error() -> None:
    with pytest.raises(HorizonError, match="skipped"):
        forecast(
            weekdays_only(),
            horizon="2D",
            freq="D",
            skip_days="weekends",
            uncertainty_samples=0,
        )


def test_skip_hours_needs_sub_daily_data() -> None:
    with pytest.raises(SkipDaysError, match="sub-daily"):
        forecast(
            every_day_series(),
            horizon=3,
            skip_hours=range(0, 8),
            uncertainty_samples=0,
        )


def test_skipping_that_leaves_no_training_rows_is_an_error() -> None:
    series = weekday_series(periods=14)
    with pytest.raises(InsufficientDataError, match="after skipping"):
        # Only one Sunday is left in the first eight days.
        forecast(
            series.iloc[:8],
            horizon=3,
            skip_days=range(0, 6),
            uncertainty_samples=0,
        )


def test_a_frequency_that_is_always_skipped_has_no_forecast_to_make() -> None:
    weekly = pd.Series(
        np.arange(30.0), index=pd.date_range("2023-01-01", periods=30, freq="W-SUN")
    )
    with pytest.raises(SkipDaysError, match="nothing to forecast"):
        forecast(
            weekly, horizon=3, skip_days="Sat", freq="W-SAT", uncertainty_samples=0
        )


def test_sub_daily_skip_days_and_hours() -> None:
    index = pd.date_range("2024-03-04", periods=24 * 21, freq="h")
    values = 10 + 5 * np.sin(2 * np.pi * (index.hour.to_numpy() - 6) / 24)
    series = pd.Series(values, index=index)
    result = forecast(
        series,
        horizon=30,
        skip_days="weekends",
        skip_hours=range(0, 8),
        uncertainty_samples=0,
    )
    for frame in (result.history, result.forecast):
        assert not frame["ds"].dt.dayofweek.isin(WEEKEND).any()
        assert (frame["ds"].dt.hour >= 8).all()
    assert len(result.forecast) == 30
    assert result.skip_hours == tuple(range(8))
    assert "00:00 to 08:00" in str(result.summary())


# ---- charts ------------------------------------------------------------------


def _observed_positions(ax: plt.Axes) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Pixel x of the observed dots (the first line drawn) and their dates."""
    line = ax.lines[0]
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(line.get_xdata())))
    ax.figure.canvas.draw()  # pyrefly:ignore[missing-attribute]
    pixels = ax.transData.transform(
        np.column_stack([date2num(dates), np.zeros(len(dates))])
    )[:, 0]
    return pixels, dates


def test_chart_closes_the_weekend_gap(weekend_result: Any) -> None:
    fig = weekend_result.plot()
    ax = fig.axes[0]
    assert ax.get_xscale() == "function"
    pixels, dates = _observed_positions(ax)
    steps = np.diff(pixels)
    # Every step between consecutive weekdays is the same, including Friday to Monday.
    np.testing.assert_allclose(steps, steps[0], rtol=1e-6)
    friday_to_monday = dates[1:][dates[:-1].dayofweek == 4]
    assert len(friday_to_monday) > 10
    assert (friday_to_monday.dayofweek == 0).all()
    assert any("leaves out Saturdays and Sundays" in t.get_text() for t in fig.texts)


def test_chart_axis_can_be_made_continuous_again(weekend_result: Any) -> None:
    ax = weekend_result.plot(skip_days=()).axes[0]
    assert ax.get_xscale() == "linear"


def test_components_close_the_gap_and_the_weekly_profile_omits_skipped_days(
    weekend_result: Any,
) -> None:
    fig = weekend_result.plot_components()
    trend_ax, weekly_ax = fig.axes[0], fig.axes[1]
    assert trend_ax.get_xscale() == "function"
    labels = [t.get_text() for t in weekly_ax.get_xticklabels()]
    assert labels == ["Mon", "Tue", "Wed", "Thu", "Fri"]


def test_sub_daily_components_blank_the_skipped_hours() -> None:
    index = pd.date_range("2024-03-04", periods=24 * 28, freq="h")
    rng = np.random.default_rng(3)
    values = 10 + 5 * np.sin(2 * np.pi * (index.hour.to_numpy() - 6) / 24)
    series = pd.Series(values + rng.normal(0, 0.3, len(index)), index=index)
    result = forecast(
        series,
        horizon=24,
        skip_hours=range(0, 8),
        daily_seasonality=True,
        uncertainty_samples=0,
    )
    fig = result.plot_components()
    daily_ax = fig.axes[-1]
    line = [ln for ln in daily_ax.lines if len(ln.get_xdata()) > 50][0]
    y = np.asarray(line.get_ydata(), dtype=float)
    x = pd.to_datetime(np.asarray(line.get_xdata()))
    assert np.isnan(y[x.hour < 8]).all()
    assert np.isfinite(y[x.hour >= 8]).all()
    assert result.plot() is not None


def test_plotly_closes_gaps_with_range_breaks(weekend_result: Any) -> None:
    pytest.importorskip("plotly.graph_objs")
    fig = plot_plotly(
        weekend_result.model, weekend_result.predictions, skip_days="weekends"
    )
    breaks = fig.layout.xaxis.rangebreaks
    assert [list(b.bounds) for b in breaks] == [["sat", "mon"]]
    nights = plot_plotly(
        weekend_result.model,
        weekend_result.predictions,
        skip_days=["Sun", "Mon"],
        skip_hours=[22, 23, 0, 1],
    )
    got = [(b.pattern, list(b.bounds)) for b in nights.layout.xaxis.rangebreaks]
    assert (None, ["sun", "tue"]) in got
    assert ("hour", [22, 2]) in got
