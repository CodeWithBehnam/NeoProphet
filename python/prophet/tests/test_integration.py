"""The one-call API wired to backtests, weather charts, reports, dashboards and skipped days."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from prophet import ForecastResult, forecast  # noqa: E402
from prophet import visuals  # noqa: E402
from prophet.quick import weather_source_labels  # noqa: E402
from prophet.visuals import (  # noqa: E402
    Backtest,
    NoRegressorsError,
    SourceRun,
    dashboard_html,
    plot_backtest,
    plot_regressor_effects,
    plot_report,
    run_backtest,
    weather_ahead_note,
    weather_source_phrase,
)

WEEKEND = {5, 6}


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def texts(fig: Figure) -> list[str]:
    """Every text on the figure, with wrapped lines joined back into one line."""
    raw = [t.get_text() for t in fig.texts] + [
        t.get_text() for ax in fig.axes for t in ax.texts
    ]
    return [" ".join(t.split()) for t in raw]


def compact(page: str) -> str:
    return page.replace(" ", "").replace("\n", "")


# ---- fake weather -----------------------------------------------------------


def fake_fetch(forecast_from: str, climatology_from: str) -> Callable[..., Any]:
    """A stand-in for ``fetch_temperature``: archive, then weather forecast, then climatology.

    The two boundaries are timestamps, so daily and hourly requests are labelled by the same rule.
    """
    from prophet.weather import Location, WeatherResult

    forecast_start = pd.Timestamp(forecast_from)
    climatology_start = pd.Timestamp(climatology_from)

    def fetch(
        location: Any, start: Any, end: Any, *, freq: str = "daily", **_: Any
    ) -> WeatherResult:
        if freq == "daily":
            ds = pd.date_range(
                pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D"
            )
        else:
            first = pd.Timestamp(start).floor("h")
            last = pd.Timestamp(end).ceil("h")
            if last == last.normalize():
                last = last + pd.Timedelta(hours=23)
            ds = pd.date_range(first, last, freq="h")
        source = np.where(
            ds < forecast_start,
            "archive",
            np.where(ds < climatology_start, "forecast", "climatology"),
        )
        temperature = 10.0 - 8.0 * np.cos(2 * np.pi * ds.dayofyear.to_numpy() / 365.25)
        return WeatherResult(
            location=Location("Testville, England", 51.5, -0.1, "Europe/London"),
            frame=pd.DataFrame(
                {"ds": ds, "temperature": temperature, "source": source}
            ),
        )

    return fetch


@pytest.fixture
def weather_boundaries(monkeypatch: pytest.MonkeyPatch) -> Callable[[str, str], None]:
    weather = pytest.importorskip("prophet.weather")

    def install(forecast_from: str, climatology_from: str) -> None:
        monkeypatch.setattr(
            weather, "fetch_temperature", fake_fetch(forecast_from, climatology_from)
        )

    return install


def make_series(
    end: str, periods: int, freq: str = "D", seed: int = 0, weekdays_only: bool = False
) -> pd.Series:
    index = pd.date_range(end=end, periods=periods, freq=freq)
    rng = np.random.default_rng(seed)
    dow = index.dayofweek.to_numpy()
    y = (
        100
        + 0.05 * np.arange(periods)
        + np.where(dow == 0, 6, np.where(dow == 4, -5, 0))
        + rng.normal(0, 0.8, periods)
    )
    series = pd.Series(y, index=index, name="orders")
    return series[~series.index.dayofweek.isin(WEEKEND)] if weekdays_only else series


# ---- a weekday-only shop with weather, shared by many tests -------------------


@pytest.fixture(scope="module")
def combined() -> Iterator[ForecastResult]:
    weather = pytest.importorskip("prophet.weather")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            weather, "fetch_temperature", fake_fetch("2024-04-29", "2024-05-08")
        )
        yield forecast(
            make_series("2024-04-30", 160, weekdays_only=True),
            horizon="6W",
            weather="Testville",
            skip_days="weekends",
            uncertainty_samples=30,
        )


def test_training_frame_is_the_history_with_each_regressor(combined: ForecastResult):
    frame = combined.training_frame
    assert list(frame.columns) == ["ds", "y", "temperature"]
    assert len(frame) == len(combined.history)
    assert frame["ds"].equals(combined.history["ds"])
    assert frame["temperature"].notna().all()
    values = combined.regressor_values
    assert values is not None
    expected = values.set_index("ds")["temperature"].reindex(frame["ds"])
    np.testing.assert_allclose(frame["temperature"].to_numpy(), expected.to_numpy())


def test_training_frame_without_regressors_is_a_copy_of_the_history():
    result = forecast(make_series("2024-04-30", 60), horizon=7, uncertainty_samples=0)
    frame = result.training_frame
    pd.testing.assert_frame_equal(frame, result.history)
    assert frame is not result.history


def test_training_frame_keeps_regressors_passed_in_as_columns():
    series = make_series("2024-04-30", 60)
    data = pd.DataFrame({"ds": series.index, "y": series.to_numpy()})
    data["promo"] = (np.arange(len(data)) % 5 == 0).astype(float)
    future = pd.DataFrame(
        {"ds": pd.date_range("2024-05-01", periods=7), "promo": np.zeros(7)}
    )
    result = forecast(
        data, horizon=7, regressors=["promo"], future=future, uncertainty_samples=0
    )
    frame = result.training_frame
    assert list(frame.columns) == ["ds", "y", "promo"]
    np.testing.assert_array_equal(frame["promo"].to_numpy(), data["promo"].to_numpy())
    assert isinstance(result.backtest(5), Backtest)


# ---- weather source alignment -------------------------------------------------


def test_weather_sources_are_none_without_weather():
    result = forecast(make_series("2024-04-30", 60), horizon=7, uncertainty_samples=0)
    assert result.weather_sources is None


def test_weather_sources_daily(weather_boundaries: Callable[[str, str], None]):
    weather_boundaries("2024-04-29", "2024-05-08")
    result = forecast(
        make_series("2024-04-30", 100), horizon=14, weather="x", uncertainty_samples=0
    )
    labels = result.weather_sources
    assert labels is not None
    assert labels.index.equals(result.predictions.index)
    assert len(labels) == len(result.predictions)
    ds = result.predictions["ds"]
    expected = np.where(
        ds < "2024-04-29",
        "archive",
        np.where(ds < "2024-05-08", "forecast", "climatology"),
    )
    assert labels.tolist() == expected.tolist()
    # The last two observed days already carry a forecast label: the future starts mid-run.
    assert labels[ds == "2024-04-30"].item() == "forecast"
    assert labels.iloc[-1] == "climatology"


def test_weather_sources_follow_the_rows_that_are_kept_when_days_are_skipped(
    combined: ForecastResult,
):
    labels = combined.weather_sources
    assert labels is not None
    assert labels.index.equals(combined.predictions.index)
    assert not combined.predictions["ds"].dt.dayofweek.isin(WEEKEND).any()
    future = labels[combined.predictions["ds"] > combined.history["ds"].max()]
    assert set(future) == {"forecast", "climatology"}
    assert labels.notna().all()


def test_weather_sources_hourly_use_the_hour_of_each_timestamp(
    weather_boundaries: Callable[[str, str], None],
):
    weather_boundaries("2024-05-01 06:00", "2024-05-01 18:00")
    result = forecast(
        make_series("2024-04-30 23:00", 120, freq="h"),
        horizon=24,
        weather="x",
        uncertainty_samples=0,
    )
    labels = result.weather_sources
    assert labels is not None
    ds = result.predictions["ds"]
    assert labels[ds == "2024-05-01 05:00"].item() == "archive"
    assert labels[ds == "2024-05-01 06:00"].item() == "forecast"
    assert labels[ds == "2024-05-01 17:00"].item() == "forecast"
    assert labels[ds == "2024-05-01 18:00"].item() == "climatology"
    assert labels.notna().all()


def test_weather_sources_between_hours_use_the_nearest_earlier_hour(
    weather_boundaries: Callable[[str, str], None],
):
    weather_boundaries("2024-05-01 00:00", "2024-05-01 02:00")
    result = forecast(
        make_series("2024-04-30 23:30", 200, freq="30min"),
        horizon=8,
        weather="x",
        uncertainty_samples=0,
    )
    labels = result.weather_sources
    assert labels is not None
    ds = result.predictions["ds"]
    # 23:30 sits in the 23:00 hour (archive) and 01:30 in the 01:00 hour (forecast), not the later hour.
    assert labels[ds == "2024-04-30 23:30"].item() == "archive"
    assert labels[ds == "2024-05-01 00:00"].item() == "forecast"
    assert labels[ds == "2024-05-01 01:30"].item() == "forecast"
    assert labels[ds == "2024-05-01 02:00"].item() == "climatology"


def test_weather_sources_monthly_use_the_latest_day_of_each_period(
    weather_boundaries: Callable[[str, str], None],
):
    weather_boundaries("2024-03-21", "2024-04-11")
    result = forecast(
        make_series("2024-03-01", 48, freq="MS"),
        horizon=3,
        weather="x",
        uncertainty_samples=0,
    )
    labels = result.weather_sources
    assert labels is not None
    ds = result.predictions["ds"]
    # March's last day (31 Mar) is a forecast day although the 1st was measured; April's last day is climatology.
    assert labels[ds == "2024-02-01"].item() == "archive"
    assert labels[ds == "2024-03-01"].item() == "forecast"
    assert labels[ds == "2024-04-01"].item() == "climatology"
    assert labels.iloc[-1] == "climatology"


def test_weather_sources_weekly_use_the_label_day_of_each_period(
    weather_boundaries: Callable[[str, str], None],
):
    weather_boundaries("2024-03-25", "2024-04-04")
    result = forecast(
        make_series("2024-03-31", 40, freq="W"),
        horizon=3,
        freq="W-SUN",
        weather="x",
        uncertainty_samples=0,
    )
    labels = result.weather_sources
    assert labels is not None
    ds = result.predictions["ds"]
    assert labels[ds == "2024-03-24"].item() == "archive"
    assert labels[ds == "2024-03-31"].item() == "forecast"
    # The week of 1 to 7 April starts on a forecast day, but its last day (7 April) is climatology.
    assert labels[ds == "2024-04-07"].item() == "climatology"


def test_weather_source_labels_leave_gaps_as_missing():
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-02", periods=2),
            "temperature": [1.0, 2.0],
            "source": ["archive", "forecast"],
        }
    )
    ds = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04"])
    labels = weather_source_labels(ds, "D", frame)
    assert pd.isna(labels.iloc[0]) and pd.isna(labels.iloc[2])
    assert labels.iloc[1] == "archive"


# ---- backtest methods -----------------------------------------------------------


def test_backtest_defaults_to_the_forecasts_own_horizon(combined: ForecastResult):
    run = combined.backtest()
    assert isinstance(run, Backtest)
    assert len(run.holdout) == combined.horizon_periods
    assert run.skip_days == (5, 6)
    # 30 weekday rows are six weeks, and the wording says so rather than "30 days".
    assert "week" in run.span
    # Weekend dates are neither trained on nor held out.
    for part in (run.train, run.holdout):
        assert not part["ds"].dt.dayofweek.isin(WEEKEND).any()
    assert run.metrics.n == len(run.holdout)


def test_backtest_accepts_a_duration_horizon(combined: ForecastResult):
    run = combined.backtest("21D")
    assert run.span == "the last 21 days"
    assert 14 <= len(run.holdout) <= 16  # 21 calendar days hold 15 weekdays
    assert not run.holdout["ds"].dt.dayofweek.isin(WEEKEND).any()


def test_default_backtest_horizon_keeps_half_the_history_for_training(
    caplog: pytest.LogCaptureFixture,
):
    result = forecast(make_series("2024-04-30", 30), horizon=40, uncertainty_samples=0)
    with caplog.at_level("INFO", logger="prophet.quick"):
        run = result.backtest()
    assert len(run.holdout) == 15 and len(run.train) == 15
    assert any("cut from 40 to 15" in r.getMessage() for r in caplog.records)


def test_backtest_horizon_errors_stay_specific():
    result = forecast(make_series("2024-04-30", 30), horizon=7, uncertainty_samples=0)
    with pytest.raises(visuals.BacktestError):
        result.backtest(60)


def test_plot_backtest_draws_on_a_closed_axis_and_reuses_a_result(
    combined: ForecastResult, monkeypatch: pytest.MonkeyPatch
):
    run = combined.backtest()
    calls: list[int] = []
    original = visuals.run_backtest

    def counting(*args: Any, **kwargs: Any) -> Backtest:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(visuals, "run_backtest", counting)
    fig = combined.plot_backtest(backtest=run)
    assert calls == []
    assert fig.axes[0].get_xscale() == "function"
    assert any("Saturdays and Sundays" in t for t in texts(fig))
    combined.plot_backtest(horizon=10)
    assert calls == [1]
    flat = combined.plot_backtest(backtest=run, skip_days=())
    assert flat.axes[0].get_xscale() == "linear"


# ---- regressor effects, report and dashboard methods -----------------------------


def test_regressor_effects_label_where_the_weather_came_from(combined: ForecastResult):
    fig = combined.plot_regressor_effects()
    words = " ".join(texts(fig))
    assert "Weather forecast" in words and "Typical season" in words
    # The sentence counts calendar days: five weekday rows of weather forecast span seven days, not five.
    assert "weather forecast for 7 days" in words
    assert "climatology" in words
    assert fig.axes[0].get_xscale() == "function"
    # The ribbon above the panel shares the panel's closed axis.
    assert {ax.get_xscale() for ax in fig.axes} == {"function"}
    assert "Saturdays and Sundays" in words


def test_regressor_effects_need_regressors():
    result = forecast(make_series("2024-04-30", 60), horizon=7, uncertainty_samples=0)
    with pytest.raises(NoRegressorsError):
        result.plot_regressor_effects()


def test_report_runs_one_backtest_and_passes_its_metrics(
    combined: ForecastResult, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[Any, ...]] = []
    original = visuals.run_backtest

    def counting(*args: Any, **kwargs: Any) -> Backtest:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(visuals, "run_backtest", counting)
    fig = combined.report()
    assert len(calls) == 1
    words = texts(fig)
    assert "HELD BACK" in words
    assert any("Saturdays and Sundays" in t for t in words)
    # The weather note names the climatology that fills the horizon.
    assert any("Future temperature: weather forecast for" in t for t in words)
    assert any("climatology" in t for t in words)
    time_axes = [ax for ax in fig.axes if ax.get_xscale() == "function"]
    assert (
        len(time_axes) >= 4
    )  # forecast, trend, holidays and temperature panels at least
    weekly = [
        ax
        for ax in fig.axes
        if [t.get_text() for t in ax.get_xticklabels()]
        == ["Mon", "Tue", "Wed", "Thu", "Fri"]
    ]
    assert len(weekly) == 1  # only the days that remain


def test_report_can_skip_or_reuse_the_backtest(
    combined: ForecastResult, monkeypatch: pytest.MonkeyPatch
):
    run = combined.backtest()
    monkeypatch.setattr(
        visuals,
        "run_backtest",
        lambda *a, **k: pytest.fail("the backtest must not run again"),
    )
    assert "HELD BACK" in texts(combined.report(backtest=run))
    assert "HELD BACK" not in texts(combined.report(backtest=False))


def test_dashboard_closes_skipped_days_and_notes_the_weather(
    combined: ForecastResult, tmp_path: Path
):
    pytest.importorskip("plotly")
    out = combined.dashboard(tmp_path / "out" / "dash.html")
    page = out.read_text(encoding="utf-8")
    assert out == tmp_path / "out" / "dash.html"
    assert compact(page).count('"rangebreaks":[{"bounds":["sat","mon"]}]') >= 2
    assert "Future temperature: weather forecast for" in page
    assert "Saturdays and Sundays" in page
    assert "HELD BACK" in page
    skipped = combined.dashboard(tmp_path / "plain.html", backtest=False)
    assert "HELD BACK" not in skipped.read_text(encoding="utf-8")


def test_forecast_chart_says_how_much_of_the_weather_is_climatology(
    combined: ForecastResult,
):
    words = " ".join(texts(combined.plot()))
    assert "Future temperature: weather forecast for 7 days, then climatology" in words
    # An explicit credit line replaces the default, including the note.
    assert "climatology" not in " ".join(texts(combined.plot(source="Mine")))


def test_forecast_chart_stays_quiet_when_no_climatology_is_used(
    weather_boundaries: Callable[[str, str], None],
):
    weather_boundaries("2024-04-29", "2030-01-01")
    result = forecast(
        make_series("2024-04-30", 80), horizon=7, weather="x", uncertainty_samples=0
    )
    assert "climatology" not in " ".join(texts(result.plot()))
    labels = result.weather_sources
    assert labels is not None and set(labels) == {"archive", "forecast"}


def test_plain_results_have_every_view(tmp_path: Path):
    pytest.importorskip("plotly")
    result = forecast(
        make_series("2024-04-30", 120), horizon=14, uncertainty_samples=30
    )
    assert isinstance(result.plot_backtest(), Figure)
    assert "HELD BACK" in texts(result.report())
    page = result.dashboard(tmp_path / "plain.html").read_text(encoding="utf-8")
    assert '"bounds":["sat","mon"]' not in compact(
        page
    )  # plotly's own bundle mentions rangebreaks
    assert "Future temperature" not in page


# ---- skipped hours ----------------------------------------------------------------


def test_skipped_hours_reach_the_backtest_report_and_dashboard(tmp_path: Path):
    pytest.importorskip("plotly")
    index = pd.date_range("2024-04-01", periods=24 * 20, freq="h")
    keep = index[(index.hour >= 8)]
    rng = np.random.default_rng(1)
    series = pd.Series(
        50
        + 10 * np.sin(2 * np.pi * keep.hour.to_numpy() / 24)
        + rng.normal(0, 1, len(keep)),
        index=keep,
    )
    result = forecast(
        series, horizon=32, skip_hours=range(0, 8), uncertainty_samples=20
    )
    run = result.backtest()
    assert run.skip_hours == tuple(range(8))
    assert (run.holdout["ds"].dt.hour >= 8).all()
    fig = result.plot_backtest(backtest=run)
    assert fig.axes[0].get_xscale() == "function"
    assert any("00:00 to 08:00" in t for t in texts(fig))
    report = result.report(backtest=run)
    assert any(ax.get_xscale() == "function" for ax in report.axes)
    page = result.dashboard(tmp_path / "hours.html", backtest=run).read_text(
        encoding="utf-8"
    )
    assert '"pattern":"hour"' in compact(page)
    assert '"bounds":[0,8]' in compact(page)


# ---- visuals called directly with skipped days -----------------------------------


@pytest.fixture(scope="module")
def full_calendar(
    combined: ForecastResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """History and a forecast frame that still contain the weekend rows the model never saw."""
    weather = fake_fetch("2024-04-29", "2024-05-08")
    model = combined.model
    ds = pd.date_range(combined.history["ds"].min(), "2024-06-14", freq="D")
    frame = weather("x", ds.min(), ds.max()).frame.set_index("ds").reindex(ds)
    inputs = pd.DataFrame({"ds": ds, "temperature": frame["temperature"].to_numpy()})
    fcst = model.predict(inputs)
    rng = np.random.default_rng(2)
    df = inputs[inputs["ds"] <= combined.history["ds"].max()].assign(
        y=lambda d: 100 + 0.2 * d["temperature"] + rng.normal(0, 1, len(d))
    )
    sources = pd.Series(frame["source"].to_numpy(), index=fcst.index)
    return df, fcst, sources


def test_run_backtest_leaves_skipped_days_out_of_the_holdout(
    combined: ForecastResult,
    full_calendar: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
):
    df, _, _ = full_calendar
    plain = run_backtest(combined.model, df, 28)
    skipped = run_backtest(combined.model, df, 28, skip_days="weekends")
    assert plain.holdout["ds"].dt.dayofweek.isin(WEEKEND).any()
    assert not skipped.holdout["ds"].dt.dayofweek.isin(WEEKEND).any()
    assert not skipped.train["ds"].dt.dayofweek.isin(WEEKEND).any()
    assert len(skipped.holdout) == 28  # a row count counts only the rows that remain
    assert skipped.skip_days == (5, 6) and plain.skip_days == ()
    assert skipped.span == "the last 6 weeks"


def test_run_backtest_rejects_a_horizon_that_leaves_nothing_after_skipping(
    combined: ForecastResult,
    full_calendar: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
):
    df, _, _ = full_calendar
    with pytest.raises(visuals.BacktestError):
        run_backtest(combined.model, df, 10_000, skip_days="weekends")
    with pytest.raises(ValueError):
        run_backtest(combined.model, df, 10, skip_days="Funday")


def test_plot_backtest_takes_skip_settings_or_inherits_them(
    combined: ForecastResult,
    full_calendar: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
):
    df, _, _ = full_calendar
    fig = plot_backtest(combined.model, df, 20, skip_days="weekends")
    assert fig.axes[0].get_xscale() == "function"
    run = run_backtest(combined.model, df, 20, skip_days="weekends")
    inherited = plot_backtest(combined.model, df, 20, backtest=run)
    assert inherited.axes[0].get_xscale() == "function"
    override = plot_backtest(combined.model, df, 20, backtest=run, skip_days=())
    assert override.axes[0].get_xscale() == "linear"
    assert plot_backtest(combined.model, df, 20).axes[0].get_xscale() == "linear"


def test_regressor_effects_drop_skipped_rows_and_keep_sources_paired(
    combined: ForecastResult,
    full_calendar: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
):
    _, fcst, sources = full_calendar
    fig = plot_regressor_effects(
        combined.model, fcst, weather_sources=sources, skip_days="weekends"
    )
    panel = next(ax for ax in fig.axes if ax.get_xscale() == "function" and ax.lines)
    dated = [
        np.asarray(line.get_xdata())
        for line in panel.lines
        if np.issubdtype(np.asarray(line.get_xdata()).dtype, np.datetime64)
    ]
    assert dated, "the panel draws its contribution against timestamps"
    days = pd.DatetimeIndex(np.concatenate(dated)).dayofweek
    assert not np.isin(days, list(WEEKEND)).any()
    words = " ".join(texts(fig))
    assert "Weather forecast" in words and "Typical season" in words
    # Weekday rows only: the nine days of forecast weather (6 May to 7 May) are counted in calendar days.
    assert "climatology" in words


def test_report_and_dashboard_accept_frames_that_still_hold_skipped_rows(
    combined: ForecastResult,
    full_calendar: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    tmp_path: Path,
):
    pytest.importorskip("plotly")
    df, fcst, sources = full_calendar
    fig = plot_report(
        combined.model, fcst, df=df, weather_sources=sources, skip_days="weekends"
    )
    assert any(ax.get_xscale() == "function" for ax in fig.axes)
    words = texts(fig)
    assert any("Saturdays and Sundays" in t for t in words)
    # Summary numbers ignore the dropped rows: the history tile counts weekday points only.
    assert any(
        f"{len(combined.history):,} points" in t.replace("\n", " ") for t in words
    )
    page = dashboard_html(
        combined.model,
        fcst,
        tmp_path / "d.html",
        df=df,
        weather_sources=sources,
        skip_days="weekends",
    ).read_text(encoding="utf-8")
    assert '"bounds":["sat","mon"]' in compact(page) and "Saturdays and Sundays" in page


def test_everything_skipped_is_an_error_not_an_empty_chart(
    combined: ForecastResult,
    full_calendar: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
):
    _, fcst, _ = full_calendar
    with pytest.raises(ValueError):
        plot_report(combined.model, fcst, skip_days=range(7))


# ---- pure helpers -----------------------------------------------------------------


def run(source: str, start: str, end: str, n: int) -> SourceRun:
    return SourceRun(source, pd.Timestamp(start), pd.Timestamp(end), n)


def test_weather_ahead_note_only_speaks_when_climatology_is_involved():
    after = pd.Timestamp("2024-04-30")
    day = pd.Timedelta(days=1)
    mixed = [
        run("archive", "2024-04-01", "2024-04-28", 28),
        run("forecast", "2024-05-01", "2024-05-09", 9),
        run("climatology", "2024-05-10", "2024-05-30", 21),
    ]
    assert weather_ahead_note(mixed, after, day) == (
        "Future temperature: weather forecast for 9 days, then climatology (typical seasonal values) for 21 days."
    )
    assert weather_ahead_note(mixed[2:], after, day, name="rainfall") == (
        "Future rainfall: climatology (typical seasonal values) for 21 days."
    )
    assert weather_ahead_note(mixed[:2], after, day) == ""
    assert weather_ahead_note([], after, day) == ""


def test_calendar_wording_counts_elapsed_time_not_rows():
    after = pd.Timestamp("2024-04-30")
    day = pd.Timedelta(days=1)
    # Nine weekday rows from Wed 1 May to Mon 13 May cover 13 days.
    runs = [
        run("forecast", "2024-05-01", "2024-05-13", 9),
        run("climatology", "2024-05-14", "2024-05-31", 14),
    ]
    rows = weather_source_phrase(runs, after, day)
    elapsed = weather_source_phrase(runs, after, day, calendar=True)
    assert "weather forecast for 9 days" in rows
    assert "weather forecast for 13 days" in elapsed
    note = weather_ahead_note(runs, after, day, calendar=True)
    assert "weather forecast for 13 days" in note


def test_runs_that_start_before_the_forecast_are_cut_to_their_future_rows():
    ds = pd.date_range("2024-04-27", periods=8, freq="D")  # history ends 30 April
    runs = (
        run("archive", "2024-04-27", "2024-04-28", 2),
        run("forecast", "2024-04-29", "2024-05-02", 4),
        run("climatology", "2024-05-03", "2024-05-04", 2),
    )
    cut = visuals._runs_after(runs, ds, pd.Timestamp("2024-04-30"))
    assert [(r.source, r.start, r.n) for r in cut] == [
        ("forecast", pd.Timestamp("2024-05-01"), 2),
        ("climatology", pd.Timestamp("2024-05-03"), 2),
    ]
    assert visuals._runs_after((runs[0],), ds, pd.Timestamp("2024-04-30")) == ()


def test_horizon_wording_uses_the_calendar_only_when_given():
    step = pd.Timedelta(days=1)
    assert visuals._horizon_phrase(30, 30, step) == "the last 30 days"
    assert (
        visuals._horizon_phrase(30, 30, step, pd.Timedelta(days=42))
        == "the last 6 weeks"
    )
    # A duration is always quoted in the caller's own words.
    assert (
        visuals._horizon_phrase("90D", 64, step, pd.Timedelta(days=90))
        == "the last 90 days"
    )


def test_wrap_keeps_to_its_line_limit_even_when_words_wrap_badly():
    text = "Thursdays are highest (+0.797), Saturdays lowest (-1.53)"
    wrapped = visuals._wrap(text, 2.2, 9.0, 2)
    assert wrapped.count("\n") <= 1
    assert "(-1.53)" in wrapped


def test_flat_axes_get_enough_decimals_to_tell_ticks_apart():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [281.7, 282.3])
    ax.yaxis.set_major_formatter(visuals.FuncFormatter(visuals._fmt_tick))
    fig.canvas.draw()
    before = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert len(set(before)) < len(before)
    visuals._distinct_ticks(ax)
    fig.canvas.draw()
    after = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert len(set(after)) == len(after)
    # Axes whose ticks already differ keep their formatter.
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [10, 500])
    ax2.yaxis.set_major_formatter(visuals.FuncFormatter(visuals._fmt_tick))
    formatter = ax2.yaxis.get_major_formatter()
    visuals._distinct_ticks(ax2)
    assert ax2.yaxis.get_major_formatter() is formatter
