import logging
import warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from prophet import Prophet  # noqa: E402
from prophet import plot as plotmod  # noqa: E402
from prophet.diagnostics import performance_metrics  # noqa: E402
from prophet.plot import (  # noqa: E402
    add_changepoints_to_plot,
    plot,
    plot_components,
    plot_cross_validation_metric,
    plot_forecast_component,
    plot_seasonality,
    plot_weekly,
    plot_yearly,
)

go = pytest.importorskip("plotly.graph_objs")

from prophet.plot import (  # noqa: E402
    plot_components_plotly,
    plot_forecast_component_plotly,
    plot_plotly,
    plot_seasonality_plotly,
)

DATA = Path(__file__).parent
CASES = [
    "linear",
    "logistic",
    "flat",
    "multiplicative",
    "custom_seasonality",
    "holidays",
    "regressors",
    "no_uncertainty",
    "no_changepoints",
    "subdaily",
]


@contextmanager
def no_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@lru_cache(maxsize=None)
def fit_case(name: str) -> tuple[Prophet, pd.DataFrame]:
    """Fit a small model for one configuration of the matrix and predict 30 days (or 48 steps) ahead."""
    daily = pd.read_csv(DATA / "data.csv", parse_dates=["ds"])
    df = daily
    kwargs: dict[str, Any] = {}
    future_kwargs: dict[str, Any] = {"periods": 30}
    if name == "logistic":
        df = daily.assign(cap=daily["y"].max() * 1.5, floor=0.0)
        kwargs = {"growth": "logistic"}
    elif name == "flat":
        kwargs = {"growth": "flat"}
    elif name == "multiplicative":
        kwargs = {"seasonality_mode": "multiplicative"}
    elif name == "holidays":
        kwargs = {"holidays": pd.DataFrame({
            "holiday": "sale", "ds": pd.to_datetime(["2012-12-24", "2013-12-24", "2014-06-10"])})}
    elif name == "regressors":
        df = daily.assign(x=np.sin(np.arange(len(daily)) / 7.0), z=np.cos(np.arange(len(daily)) / 11.0))
    elif name == "no_uncertainty":
        kwargs = {"uncertainty_samples": 0}
    elif name == "no_changepoints":
        kwargs = {"n_changepoints": 0}
    elif name == "subdaily":
        df = pd.read_csv(DATA / "data2.csv", parse_dates=["ds"])
        future_kwargs = {"periods": 48, "freq": "5min"}
    m = Prophet(**kwargs)
    if name == "custom_seasonality":
        m.add_seasonality(name="monthly", period=30.5, fourier_order=3)
    if name == "subdaily":
        m.add_seasonality(name="hourly", period=1 / 24, fourier_order=2)
    if name == "regressors":
        m.add_regressor("x")
        m.add_regressor("z", mode="multiplicative")
    m.fit(df)
    future = m.make_future_dataframe(**future_kwargs)
    if name == "logistic":
        future = future.assign(cap=df["cap"].iloc[0], floor=0.0)
    if name == "regressors":
        future = future.assign(x=np.sin(np.arange(len(future)) / 7.0), z=np.cos(np.arange(len(future)) / 11.0))
    return m, m.predict(future)


def _fit_synthetic(y_of: pd.Series, **kwargs: Any) -> tuple[Prophet, pd.DataFrame]:
    m = Prophet(uncertainty_samples=0, **kwargs)
    m.fit(pd.DataFrame({"ds": y_of.index, "y": y_of.to_numpy()}))
    return m, m.predict(m.make_future_dataframe(30))


def _texts(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.texts]


# ---- pure wording helpers -------------------------------------------------


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        (pd.Timedelta(days=365), "the next 12 months"),
        (pd.Timedelta(days=1826), "the next 5 years"),
        (pd.Timedelta(days=30), "the next 4 weeks"),
        (pd.Timedelta(days=5), "the next 5 days"),
        (pd.Timedelta(days=1), "the next 24 hours"),
        (pd.Timedelta(hours=6), "the next 6 hours"),
        (pd.Timedelta(minutes=30), "the next 30 minutes"),
        (pd.Timedelta(hours=1), "the next 60 minutes"),
        (pd.Timedelta(days=3), "the next 3 days"),
    ],
)
def test_span_phrase_follows_the_forecast_span(span: pd.Timedelta, expected: str) -> None:
    assert plotmod._span_phrase(span, "the next") == expected


def test_span_phrase_singular() -> None:
    assert plotmod._span_phrase(pd.Timedelta(minutes=1), "the next") == "the next minute"
    assert plotmod._span_phrase(pd.Timedelta(days=2), "the last") == "the last 2 days"


def test_compare_phrase_uses_percentages_only_for_a_clear_positive_base() -> None:
    assert plotmod._compare_phrase(112.0, 100.0, 120.0) == "about 12% higher than"
    assert plotmod._compare_phrase(90.0, 100.0, 120.0) == "about 10% lower than"
    assert plotmod._compare_phrase(101.0, 100.0, 120.0) == "about the same as"


@pytest.mark.parametrize(("new", "old"), [(0.9, 0.0), (0.9, 0.001), (-2.0, -3.0), (3.0, -1.0)])
def test_compare_phrase_falls_back_to_differences(new: float, old: float) -> None:
    phrase = plotmod._compare_phrase(new, old, scale=10.0)
    assert "%" not in phrase


def test_number_and_percentage_formatting() -> None:
    assert plotmod._fmt_number(8.523) == "8.52"
    assert plotmod._fmt_number(1234.4) == "1,234"
    assert plotmod._fmt_number(12345) == "12.3K"
    assert plotmod._fmt_number(2.5e6) == "2.5M"
    assert plotmod._fmt_pct(0.065) == "6.5%"
    assert plotmod._fmt_pct(0.123) == "12%"
    assert plotmod._signed_effect(-0.04, True) == "-4%"
    assert plotmod._signed_effect(1.25, False) == "+1.25"


def test_text_switches() -> None:
    def auto() -> tuple[str, str]:
        return "Auto title", "Auto subtitle"

    assert plotmod._headline_texts(None, None, auto) == ("Auto title", "Auto subtitle")
    assert plotmod._headline_texts("Mine", None, auto) == ("Mine", "Auto subtitle")
    assert plotmod._headline_texts(False, None, auto) == (None, None)
    assert plotmod._headline_texts(None, False, auto) == ("Auto title", None)
    assert plotmod._headline_texts(False, "Kept", auto) == (None, "Kept")


def test_fonts_end_in_the_bundled_families_and_are_cached() -> None:
    assert plotmod._serif()[-1] == "DejaVu Serif"
    assert plotmod._sans()[-1] == "DejaVu Sans"
    plotmod._installed_families.cache_clear()
    plotmod._sans()
    plotmod._sans()
    assert plotmod._installed_families.cache_info().hits >= 1


# ---- matplotlib across the configuration matrix ---------------------------


@pytest.mark.parametrize("case", CASES)
def test_plot_across_configurations(case: str) -> None:
    m, fcst = fit_case(case)
    with no_warnings():
        fig = plot(m, fcst)
        fig_plain = plot(m, fcst, annotate=False, title=False, include_legend=True)
        fig_titled = plot(m, fcst, title="Mine", subtitle=False, source=False, xlabel="when", ylabel="count")
    assert isinstance(fig, Figure)
    assert isinstance(fig_plain, Figure)
    assert "Mine" in _texts(fig_titled)
    assert _texts(fig)[0].startswith("What happens next: the next ")


@pytest.mark.parametrize("case", CASES)
def test_plot_components_across_configurations(case: str) -> None:
    m, fcst = fit_case(case)
    with no_warnings():
        fig = plot_components(m, fcst)
        fig_plain = plot_components(m, fcst, annotate=False, panel_titles=False, title=False, source=False)
    assert isinstance(fig, Figure)
    assert isinstance(fig_plain, Figure)
    assert len(fig.axes) == len(fig_plain.axes) >= 1
    assert _texts(fig_plain) == []
    assert _texts(fig)[0].startswith("What shapes the forecast: trend")


@pytest.mark.parametrize("case", CASES)
def test_component_helpers_return_artists_and_title_the_axes(case: str) -> None:
    m, fcst = fit_case(case)
    with no_warnings():
        for name in ("trend", *(["holidays"] if case == "holidays" else [])):
            fig, ax = plt.subplots()
            artists = plot_forecast_component(m, fcst, name, ax=ax, plot_cap=True)
            assert len(artists) >= 1
            assert ax.get_title(loc="left") != ""
            assert fig.texts == []
        for name, props in m.seasonalities.items():
            fig, ax = plt.subplots()
            if name == "yearly":
                artists = plot_yearly(m, ax=ax)
            elif name == "weekly" and case != "subdaily":
                artists = plot_weekly(m, ax=ax)
            else:
                artists = plot_seasonality(m, name, ax=ax)
            assert len(artists) >= 1, (case, name, props["period"])
            assert ax.get_title(loc="left") != ""
            assert fig.texts == []


@pytest.mark.parametrize("case", ["linear", "multiplicative", "no_uncertainty", "subdaily"])
def test_user_axes_get_no_figure_level_additions(case: str) -> None:
    m, fcst = fit_case(case)
    fig, ax = plt.subplots(figsize=(8, 4))
    with no_warnings():
        returned = plot(m, fcst, ax=ax)
    assert returned is fig
    assert fig.texts == []
    assert fig.artists == []
    assert ax.get_title(loc="left").startswith("What happens next")
    assert ax.yaxis.get_label().get_text() == ""


def test_owned_figure_has_headline_credit_and_paper_ground() -> None:
    m, fcst = fit_case("linear")
    fig = plot(m, fcst)
    assert fig.get_facecolor()[:3] == matplotlib.colors.to_rgb(plotmod.PAPER)
    texts = _texts(fig)
    assert any(t.startswith("NeoProphet forecast:") for t in texts)
    assert plot(m, fcst, source="Custom credit").texts[-1].get_text() == "Custom credit"


def test_changepoint_helper_returns_lines() -> None:
    m, fcst = fit_case("linear")
    fig = plot(m, fcst)
    with no_warnings():
        artists = add_changepoints_to_plot(fig.gca(), m, fcst)
        assert all(isinstance(a, Line2D) for a in artists)
        assert len(artists) >= 1
        assert add_changepoints_to_plot(fig.gca(), m, fcst, trend=False, threshold=1e9) == []


def test_no_significant_changepoints_means_no_callouts() -> None:
    m, fcst = fit_case("no_changepoints")
    fig = plot(m, fcst)
    assert not any("trend bends" in t.get_text() for t in fig.axes[0].texts)
    assert plotmod._top_changepoints(m, fcst).empty


def test_changepoint_callouts_are_neutral_and_switchable() -> None:
    m, fcst = fit_case("linear")
    with_callouts = plot(m, fcst)
    callouts = [t.get_text() for t in with_callouts.axes[0].texts if "trend bends" in t.get_text()]
    assert 1 <= len(callouts) <= 3
    assert all(c.endswith(("upwards", "downwards")) for c in callouts)
    without = plot(m, fcst, annotate=False)
    assert not any("trend bends" in t.get_text() for t in without.axes[0].texts)
    assert not any(t.get_text() in ("HISTORY", "FORECAST") for t in without.axes[0].texts)


def test_history_only_prediction_and_short_history() -> None:
    m, _ = fit_case("linear")
    assert m.history is not None
    in_sample = m.predict(m.history[["ds"]])
    with no_warnings():
        fig = plot(m, in_sample)
        components = plot_components(m, in_sample)
    assert _texts(fig)[0] == "How the model fits the history"
    assert not any(t.get_text() in ("HISTORY", "FORECAST") for t in fig.axes[0].texts)
    assert isinstance(components, Figure)

    short = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=8), "y": np.arange(8.0) + 5})
    m_short = Prophet(uncertainty_samples=20)
    m_short.fit(short)
    fc_short = m_short.predict(m_short.make_future_dataframe(5))
    with no_warnings():
        assert isinstance(plot(m_short, fc_short), Figure)
        assert isinstance(plot_components(m_short, fc_short), Figure)


def test_plots_do_not_touch_global_matplotlib_state() -> None:
    before = dict(matplotlib.rcParams)
    m, fcst = fit_case("holidays")
    plot(m, fcst)
    plot_components(m, fcst)
    assert dict(matplotlib.rcParams) == before


def test_no_font_warnings(caplog: pytest.LogCaptureFixture) -> None:
    m, fcst = fit_case("holidays")
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        plot(m, fcst)
        plot_components(m, fcst)
    assert [r for r in caplog.records if "findfont" in r.getMessage()] == []


# ---- derived wording on constructed data ----------------------------------


def test_weekly_title_names_the_known_peak_day() -> None:
    dates = pd.date_range("2023-01-02", periods=210)
    y = pd.Series(100.0 + 25.0 * (dates.dayofweek == 5) + 0.01 * np.arange(len(dates)), index=dates)
    m, _ = _fit_synthetic(y, yearly_seasonality=False)
    fig, ax = plt.subplots()
    plot_weekly(m, ax=ax)
    assert ax.get_title(loc="left") == "Weekly pattern: Saturdays are highest"
    assert any("Saturdays are" in text.get_text() and "usual level" in text.get_text() for text in ax.texts)
    fig_off, ax_off = plt.subplots()
    plot_weekly(m, ax=ax_off, title=False, annotate=False)
    assert ax_off.get_title(loc="left") == ""
    fig_custom, ax_custom = plt.subplots()
    plot_weekly(m, ax=ax_custom, title="Day by day")
    assert ax_custom.get_title(loc="left") == "Day by day"


def test_weekly_title_when_the_week_is_flat() -> None:
    dates = pd.date_range("2023-01-02", periods=120)
    m, _ = _fit_synthetic(pd.Series(50.0 + 0.0 * np.arange(len(dates)), index=dates), yearly_seasonality=False)
    fig, ax = plt.subplots()
    plot_weekly(m, ax=ax)
    assert ax.get_title(loc="left") == "Weekly pattern: little variation"


def test_yearly_title_names_the_known_peak_month() -> None:
    dates = pd.date_range("2020-01-01", periods=3 * 365)
    day = np.asarray(dates.dayofyear, dtype=float)
    y = pd.Series(100.0 + 30.0 * np.cos(2 * np.pi * (day - 196) / 365.25), index=dates)  # peak near 14 July
    m, _ = _fit_synthetic(y, weekly_seasonality=False)
    fig, ax = plt.subplots()
    plot_yearly(m, ax=ax)
    assert ax.get_title(loc="left") == "Yearly pattern: peaks in July"


def test_multiplicative_effects_are_percentages_additive_are_units() -> None:
    m_mult, _ = fit_case("multiplicative")
    fig, ax = plt.subplots()
    plot_weekly(m_mult, ax=ax)
    ax.figure.canvas.draw()
    assert ax.get_yticklabels()[1].get_text().endswith("%")
    assert "%" in ax.texts[0].get_text()

    m_add, _ = fit_case("linear")
    fig, ax = plt.subplots()
    plot_weekly(m_add, ax=ax)
    ax.figure.canvas.draw()
    assert not any(label.get_text().endswith("%") for label in ax.get_yticklabels())


def test_forecast_headline_names_the_horizon_in_the_right_unit() -> None:
    m, fcst = fit_case("linear")
    assert plotmod._forecast_headline(m, fcst, True)[0] == "What happens next: the next 4 weeks"
    m_sub, fcst_sub = fit_case("subdaily")
    assert plotmod._forecast_headline(m_sub, fcst_sub, True)[0] == "What happens next: the next 4 hours"


def test_forecast_headline_avoids_percentages_near_zero() -> None:
    dates = pd.date_range("2023-01-01", periods=140)
    y = pd.Series(np.sin(np.arange(140) / 3.0) * 2.0, index=dates)
    m, fcst = _fit_synthetic(y, weekly_seasonality=False, yearly_seasonality=False)
    title, sub = plotmod._forecast_headline(m, fcst, show_band=False)
    assert title.startswith("What happens next")
    assert "%" not in sub


def test_trend_title_handles_flat_growth() -> None:
    m, fcst = fit_case("flat")
    assert plotmod._trend_headline(m, fcst)[0] == "Trend: broadly flat"


def test_logistic_cap_and_floor_are_drawn_and_labelled() -> None:
    m, fcst = fit_case("logistic")
    fig = plot(m, fcst)
    labels = {line.get_label() for line in fig.axes[0].get_lines()}
    assert {"Maximum capacity", "Minimum capacity"} <= labels
    assert {"Maximum capacity", "Minimum capacity"} <= {t.get_text() for t in fig.axes[0].texts}


def test_holiday_headline_names_the_biggest_holiday() -> None:
    m, fcst = fit_case("holidays")
    title, sub = plotmod._holiday_headline(m, fcst)
    assert title == "Holidays: sale has the largest effect"
    assert "usual level" in sub


def test_regressor_headline_lists_the_regressors() -> None:
    m, fcst = fit_case("regressors")
    assert plotmod._regressor_headline(m, "extra_regressors_additive", fcst["extra_regressors_additive"])[0] == (
        "Extra regressors: x")
    assert plotmod._regressor_headline(
        m, "extra_regressors_multiplicative", fcst["extra_regressors_multiplicative"])[0] == "Extra regressors: z"


# ---- cross-validation -----------------------------------------------------


def make_cv(seed: int = 0, horizon_days: int = 60, outlier: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for cutoff in pd.date_range("2022-01-01", periods=4, freq="30D"):
        ds = cutoff + pd.to_timedelta(np.arange(1, horizon_days + 1), unit="D")
        y = 100 + rng.normal(0, 5, horizon_days)
        error_scale = np.linspace(1, 6, horizon_days)  # error grows with the horizon
        frames.append(pd.DataFrame({
            "ds": ds,
            "yhat": y + rng.normal(0, 1, horizon_days) * error_scale,
            "yhat_lower": y - 10,
            "yhat_upper": y + 10,
            "y": y,
            "cutoff": cutoff,
        }))
    df = pd.concat(frames, ignore_index=True)
    if outlier:
        df.loc[0, "y"] = 1e-6
    return df


@pytest.mark.parametrize("metric", ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"])
def test_cross_validation_plot_for_every_metric(metric: str) -> None:
    df_cv = make_cv()
    with no_warnings():
        fig = plot_cross_validation_metric(df_cv, metric)
        plain = plot_cross_validation_metric(df_cv, metric, annotate=False, title=False, source=False)
    assert isinstance(fig, Figure)
    assert isinstance(plain, Figure)
    assert _texts(plain) == []
    assert _texts(fig)[0] != ""


def test_cross_validation_wording_is_plain_english() -> None:
    fig = plot_cross_validation_metric(make_cv(), "mape")
    texts = _texts(fig)
    subtitle = next(t for t in texts if t.startswith("Typical error "))
    assert " ahead: " in subtitle and "%" in subtitle
    labels = [t.get_text() for t in fig.axes[0].texts if " ahead: " in t.get_text()]
    assert 1 <= len(labels) <= 3
    assert all("week" in lab or "day" in lab or "month" in lab for lab in labels)
    assert texts[0] == "The further ahead we predict, the bigger the miss"


def test_cross_validation_user_axes_and_colours() -> None:
    fig, ax = plt.subplots()
    returned = plot_cross_validation_metric(make_cv(), "rmse", ax=ax, color="tab:green", point_color="black")
    assert returned is fig
    assert fig.texts == []
    assert ax.get_title(loc="left").startswith("The further ahead")
    assert ax.lines[-1].get_color() == "tab:green" or ax.lines[1].get_color() == "tab:green"


def test_cross_validation_clips_extreme_points_and_says_so() -> None:
    df_cv = make_cv(outlier=True)
    metrics = performance_metrics(df_cv, metrics=["mape"], rolling_window=-1)
    assert metrics is not None and metrics["mape"].max() > 1e3
    fig = plot_cross_validation_metric(df_cv, "mape")
    assert "run off the top of the chart" in " ".join(_texts(fig))
    assert fig.axes[0].get_ylim()[1] < 10
    assert "Typical error" in fig.axes[0].get_ylabel()
    unclipped = plot_cross_validation_metric(df_cv, "mape", annotate=False)
    assert unclipped.axes[0].get_ylim()[1] > 100


def test_cross_validation_sub_daily_horizons_use_hours() -> None:
    frames = []
    for cutoff in pd.date_range("2022-01-01", periods=3, freq="12h"):
        ds = cutoff + pd.to_timedelta(np.arange(1, 7), unit="h")
        frames.append(pd.DataFrame({"ds": ds, "y": 10.0, "yhat": 10.0 + np.arange(1, 7) * 0.2,
                                    "yhat_lower": 9.0, "yhat_upper": 11.0, "cutoff": cutoff}))
    fig = plot_cross_validation_metric(pd.concat(frames, ignore_index=True), "mae")
    assert "minutes" in fig.axes[0].get_xlabel()
    assert any("hour" in t.get_text() for t in fig.axes[0].texts)


# ---- plotly ---------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(daily_univariate_ts: pd.DataFrame) -> tuple[Prophet, pd.DataFrame]:
    m = Prophet()
    m.fit(daily_univariate_ts)
    forecast = m.predict(m.make_future_dataframe(periods=30))
    return m, forecast


def test_plot_plotly(fitted: tuple[Prophet, pd.DataFrame]) -> None:
    m, forecast = fitted
    fig = plot_plotly(m, forecast)
    assert isinstance(fig, go.Figure)
    names = {trace.name for trace in fig.data}
    assert {"Actual", "Predicted"} <= names


def test_plot_plotly_without_changepoints(fitted: tuple[Prophet, pd.DataFrame]) -> None:
    m, forecast = fitted
    fig = plot_plotly(m, forecast, changepoints=False, trend=False)
    assert isinstance(fig, go.Figure)


def test_plot_components_plotly(fitted: tuple[Prophet, pd.DataFrame]) -> None:
    m, forecast = fitted
    fig = plot_components_plotly(m, forecast)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("case", CASES)
def test_plotly_across_configurations_without_warnings(case: str) -> None:
    m, fcst = fit_case(case)
    with no_warnings():
        figs = [
            plot_plotly(m, fcst, trend=True, changepoints=True),
            plot_plotly(m, fcst, annotate=False, title=False),
            plot_components_plotly(m, fcst),
            plot_components_plotly(m, fcst, panel_titles=False, title=False),
            plot_forecast_component_plotly(m, fcst, "trend", plot_cap=True),
        ]
        for name in m.seasonalities:
            figs.append(plot_seasonality_plotly(m, name))
        for fig in figs:
            assert isinstance(fig, go.Figure)
            fig.to_json()


def test_plotly_uses_the_editorial_palette_and_fonts(fitted: tuple[Prophet, pd.DataFrame]) -> None:
    m, forecast = fitted
    fig = plot_plotly(m, forecast)
    assert fig.layout.paper_bgcolor == plotmod.PAPER
    assert "DejaVu Sans" in fig.layout.font.family
    assert "DejaVu Serif" in fig.layout.title.font.family
    assert fig.layout.title.text.startswith("What happens next")
    predicted = next(t for t in fig.data if t.name == "Predicted")
    assert predicted.line.color == plotmod.ACCENT
    assert len(fig.layout.shapes) >= 3  # two zones and the boundary
    assert any(a.text == "<b>Forecast</b>" for a in fig.layout.annotations)


def test_plotly_switches(fitted: tuple[Prophet, pd.DataFrame]) -> None:
    m, forecast = fitted
    fig = plot_plotly(m, forecast, title=False, annotate=False)
    assert fig.layout.title.text is None
    assert len(fig.layout.annotations) == 0
    assert plot_plotly(m, forecast, title="Mine").layout.title.text == "Mine"
    comp = plot_components_plotly(m, forecast, panel_titles=False, title=False)
    assert all("<br>" not in a.text for a in comp.layout.annotations if a.text.startswith("<b>"))


def test_plotly_components_use_add_trace_with_row_and_col(fitted: tuple[Prophet, pd.DataFrame]) -> None:
    m, forecast = fitted
    fig = plot_components_plotly(m, forecast)
    axes_used = {trace.yaxis for trace in fig.data}
    assert len(axes_used) == len(
        [c for c in ["trend", *m.seasonalities]]
    )


def test_prophet_methods_forward_style_options() -> None:
    m, fcst = fit_case("linear")
    with no_warnings():
        fig = m.plot(fcst, title=False, annotate=False, source=False)
        fig_components = m.plot_components(fcst, title="Parts", source=False)
    assert not any(t.startswith("What happens next") for t in _texts(fig))
    assert "Parts" in _texts(fig_components)
