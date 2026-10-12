import re
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

from prophet import Prophet  # noqa: E402
from prophet import visuals  # noqa: E402
from prophet.visuals import (  # noqa: E402
    Backtest,
    BacktestError,
    BacktestMetrics,
    NoRegressorsError,
    SourceRun,
    backtest_headline,
    backtest_metrics,
    dashboard_html,
    describe_periods,
    plot_backtest,
    plot_regressor_effects,
    plot_report,
    regressor_effects,
    regressor_sentence,
    run_backtest,
    split_holdout,
    weather_source_phrase,
    weather_source_runs,
)

DATA = Path(__file__).parent
CASES = [
    "linear",
    "logistic",
    "flat",
    "multiplicative",
    "subdaily",
    "no_uncertainty",
    "regressors",
    "no_regressors_custom_seasonality",
]
REGRESSOR_CASES = ["regressors", "temperature"]
HORIZONS = {"subdaily": 40}


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
def fit_case(name: str) -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    """Fit a small model for one configuration of the matrix: model, training data and a 30-step forecast."""
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"])
    kwargs: dict[str, Any] = {}
    future_kwargs: dict[str, Any] = {"periods": 30}
    if name == "logistic":
        df = df.assign(cap=df["y"].max() * 1.5, floor=0.0)
        kwargs = {"growth": "logistic"}
    elif name == "flat":
        kwargs = {"growth": "flat"}
    elif name == "multiplicative":
        kwargs = {"seasonality_mode": "multiplicative"}
    elif name == "no_uncertainty":
        kwargs = {"uncertainty_samples": 0}
    elif name == "subdaily":
        df = pd.read_csv(DATA / "data2.csv", parse_dates=["ds"])
        future_kwargs = {"periods": 48, "freq": "5min"}
    elif name == "regressors":
        df = df.assign(
            x=np.sin(np.arange(len(df)) / 7.0), z=np.cos(np.arange(len(df)) / 11.0)
        )
    elif name == "temperature":
        df = df.assign(
            temperature=10 - 8 * np.cos(2 * np.pi * df["ds"].dt.dayofyear / 365.25),
            promo=(np.arange(len(df)) % 5 == 0).astype(float),
        )
    m = Prophet(**kwargs)
    if name == "no_regressors_custom_seasonality":
        m.add_seasonality(name="monthly", period=30.5, fourier_order=3)
    if name == "subdaily":
        m.add_seasonality(name="hourly", period=1 / 24, fourier_order=2)
    if name == "regressors":
        m.add_regressor("x")
        m.add_regressor("z", mode="multiplicative")
    if name == "temperature":
        m.add_regressor("temperature")
        m.add_regressor("promo", mode="multiplicative")
    m.fit(df)
    future = m.make_future_dataframe(**future_kwargs)
    if name == "logistic":
        future = future.assign(cap=df["cap"].iloc[0], floor=0.0)
    if name == "regressors":
        future = future.assign(
            x=np.sin(np.arange(len(future)) / 7.0),
            z=np.cos(np.arange(len(future)) / 11.0),
        )
    if name == "temperature":
        future = future.assign(
            temperature=10 - 8 * np.cos(2 * np.pi * future["ds"].dt.dayofyear / 365.25),
            promo=(np.arange(len(future)) % 5 == 0).astype(float),
        )
    return m, df, m.predict(future)


@lru_cache(maxsize=None)
def backtest_case(name: str) -> Backtest:
    m, df, _ = fit_case(name)
    return run_backtest(m, df, HORIZONS.get(name, 30))


def texts(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.texts] + [
        t.get_text() for ax in fig.axes for t in ax.texts
    ]


def weather_labels(fcst: pd.DataFrame, history_end: pd.Timestamp) -> pd.Series:
    future = (fcst["ds"] > history_end).to_numpy()
    n_forecast = np.cumsum(future) <= 9
    labels = np.where(
        ~future, "archive", np.where(n_forecast, "forecast", "climatology")
    )
    return pd.Series(labels, index=fcst.index)


# ---- pure: metrics ---------------------------------------------------------


def test_metrics_values() -> None:
    m = backtest_metrics(
        [10, 20, 30, 40],
        [12, 18, 30, 44],
        [8, 15, 25, 35],
        [14, 22, 28, 48],
        interval_width=0.8,
    )
    assert m.n == 4
    assert m.mae == pytest.approx(2.0)
    assert m.rmse == pytest.approx(np.sqrt((4 + 4 + 0 + 16) / 4))
    assert m.mape == pytest.approx(np.mean([0.2, 0.1, 0.0, 0.1]))
    assert m.coverage == pytest.approx(0.75)
    assert m.interval_width == 0.8
    assert m.mean_actual == pytest.approx(25.0)


@pytest.mark.parametrize(
    "actual", [[0.0, 5.0, 6.0], [-1.0, 5.0, 6.0], [0.01, 5.0, 6.0]]
)
def test_mape_is_dropped_when_percentages_are_meaningless(actual: list[float]) -> None:
    assert backtest_metrics(actual, [1.0, 5.0, 6.0]).mape is None


def test_metrics_without_interval_have_no_coverage() -> None:
    m = backtest_metrics([1.0, 2.0], [1.0, 3.0], interval_width=0.8)
    assert m.coverage is None
    assert m.interval_width is None


def test_metrics_ignore_missing_pairs_and_reject_empty_input() -> None:
    assert backtest_metrics([1.0, np.nan, 3.0], [1.0, 2.0, 5.0]).n == 2
    with pytest.raises(BacktestError, match="No holdout point"):
        backtest_metrics([np.nan], [1.0])
    with pytest.raises(BacktestError, match="same length"):
        backtest_metrics([1.0, 2.0], [1.0])


def test_headline_states_the_accuracy_in_plain_english() -> None:
    metrics = BacktestMetrics(
        n=90,
        mae=1.0,
        rmse=1.5,
        mape=0.065,
        coverage=0.74,
        interval_width=0.8,
        mean_actual=20.0,
    )
    assert backtest_headline(metrics, "the last 90 days") == (
        "Backtest: typical error 6.5% over the last 90 days; 74% of actuals inside the 80% band"
    )
    plain = BacktestMetrics(
        n=5,
        mae=2.3,
        rmse=3.0,
        mape=None,
        coverage=None,
        interval_width=None,
        mean_actual=0.4,
    )
    assert (
        backtest_headline(plain, "the last 5 days")
        == "Backtest: forecasts missed by 2.3 on average over the last 5 days"
    )
    assert "—" not in backtest_headline(metrics, "x") and "–" not in backtest_headline(
        metrics, "x"
    )


# ---- pure: horizons and spacing --------------------------------------------


def make_frame(n: int, freq: str = "D") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=n, freq=freq),
            "y": np.arange(n, dtype=float),
        }
    )


def test_split_holdout_by_rows_and_by_duration() -> None:
    train, hold = split_holdout(make_frame(100), 10)
    assert (len(train), len(hold)) == (90, 10)
    train, hold = split_holdout(make_frame(100), "30D")
    assert (len(train), len(hold)) == (70, 30)
    train, hold = split_holdout(make_frame(36, "MS"), "1Y")
    assert len(hold) == 12
    train, hold = split_holdout(make_frame(100).sample(frac=1.0, random_state=1), 5)
    assert train["ds"].is_monotonic_increasing and hold["ds"].min() > train["ds"].max()


@pytest.mark.parametrize(
    "horizon", [0, -3, True, 1.5, "soon", "10m", "10x", "0D", 100, 99]
)
def test_split_holdout_rejects_bad_horizons(horizon: Any) -> None:
    with pytest.raises(BacktestError):
        split_holdout(make_frame(100), horizon)


def test_split_holdout_needs_columns() -> None:
    with pytest.raises(BacktestError, match="ds"):
        split_holdout(pd.DataFrame({"x": [1, 2, 3]}), 1)


@pytest.mark.parametrize(
    ("n", "step", "expected"),
    [
        (90, pd.Timedelta(days=1), "the last 90 days"),
        (1, pd.Timedelta(days=1), "the last day"),
        (12, pd.Timedelta(days=30), "the last 12 months"),
        (8, pd.Timedelta(days=7), "the last 8 weeks"),
        (48, pd.Timedelta(hours=1), "the last 48 hours"),
        (12, pd.Timedelta(minutes=5), "the last 60 minutes"),
        (4, pd.Timedelta(days=91), "the last 4 quarters"),
        (10, pd.Timedelta(days=3), "the last 4 weeks"),
    ],
)
def test_describe_periods(n: int, step: pd.Timedelta, expected: str) -> None:
    assert describe_periods(n, step) == expected


def test_horizon_phrase_uses_the_callers_words_for_durations() -> None:
    assert (
        visuals._horizon_phrase("90D", 90, pd.Timedelta(days=1)) == "the last 90 days"
    )
    assert (
        visuals._horizon_phrase("12W", 12, pd.Timedelta(days=7)) == "the last 12 weeks"
    )
    assert (
        visuals._horizon_phrase("6M", 6, pd.Timedelta(days=30)) == "the last 6 months"
    )
    assert visuals._horizon_phrase(30, 30, pd.Timedelta(days=1)) == "the last 30 days"


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (pd.Timedelta(days=1), "Daily"),
        (pd.Timedelta(hours=1), "Hourly"),
        (pd.Timedelta(days=30), "Monthly"),
        (pd.Timedelta(minutes=5), "Every 5 minutes"),
    ],
)
def test_frequency_words(step: pd.Timedelta, expected: str) -> None:
    assert visuals.frequency_words(step)[0] == expected


# ---- pure: regressor wording -------------------------------------------------


def test_regressor_sentences() -> None:
    assert (
        regressor_sentence("temperature", "additive", 0.8, unit="°C")
        == "each extra °C adds 0.8 to y"
    )
    assert (
        regressor_sentence("temperature", "additive", -0.8, unit="°C")
        == "each extra °C takes 0.8 off y"
    )
    assert (
        regressor_sentence("ad_spend", "additive", 2.0)
        == "each extra unit of ad spend adds 2 to y"
    )
    assert regressor_sentence("promo", "multiplicative", 0.021) == (
        "each extra unit of promo raises y by 2.1% of the trend level"
    )
    assert regressor_sentence("promo", "multiplicative", -0.05, binary=True) == (
        "when promo is on (1 rather than 0), y is 5% of the trend level lower"
    )
    assert (
        regressor_sentence("promo", "additive", 3.0, binary=True)
        == "when promo is on (1 rather than 0), y is 3 higher"
    )


def test_negligible_regressor_effects_are_called_that() -> None:
    assert "almost no effect" in regressor_sentence("x", "additive", 0.001, scale=100.0)
    assert "almost no effect" in regressor_sentence("x", "multiplicative", 0.001)
    assert "barely changes" in regressor_sentence(
        "x", "additive", 0.001, binary=True, scale=100.0
    )


def test_regressor_effects_need_regressors() -> None:
    m, _, fcst = fit_case("linear")
    with pytest.raises(NoRegressorsError, match="no extra regressors"):
        regressor_effects(m, fcst)


def test_regressor_effects_describe_each_regressor() -> None:
    m, _, fcst = fit_case("temperature")
    effects = {e.name: e for e in regressor_effects(m, fcst)}
    assert effects["temperature"].unit == "°C"
    assert effects["temperature"].sentence.startswith("each extra °C")
    assert effects["promo"].binary and effects["promo"].mode == "multiplicative"
    assert "when promo is on" in effects["promo"].sentence


def test_regressors_are_ranked_in_the_units_of_y_whatever_their_mode() -> None:
    # x is additive and moves y by about 0.6; z is multiplicative and moves it by
    # 20% of a trend near 40, which is about 8. Ranked on the raw numbers z (0.2)
    # would sit below x (0.6).
    m, _, fcst = fit_case("regressors")
    assert fcst["x"].abs().max() > fcst["z"].abs().max()
    assert (fcst["z"].abs() * fcst["trend"]).max() > 10 * fcst["x"].abs().max()
    assert [e.name for e in regressor_effects(m, fcst)] == ["z", "x"]


def test_the_reach_of_a_regressor_is_scaled_by_the_trend_only_when_multiplicative() -> (
    None
):
    fcst = pd.DataFrame(
        {"trend": [10.0, 20.0, 40.0], "r": [0.5, -0.25, 0.1], "ds": [1, 2, 3]}
    )
    assert visuals._reach_in_y(Prophet(), fcst, "r", "additive") == 0.5
    assert visuals._reach_in_y(Prophet(), fcst, "r", "multiplicative") == 5.0
    assert (
        visuals._reach_in_y(
            Prophet(), fcst.assign(r=[np.nan] * 3), "r", "multiplicative"
        )
        == 0.0
    )


# ---- pure: weather sources ---------------------------------------------------


def test_weather_runs_and_phrase() -> None:
    ds = pd.date_range("2024-01-01", periods=10, freq="D")
    labels = ["archive"] * 4 + ["forecast"] * 2 + ["climatology"] * 4
    runs = weather_source_runs(ds, labels)
    assert [(r.source, r.n) for r in runs] == [
        ("archive", 4),
        ("forecast", 2),
        ("climatology", 4),
    ]
    assert runs[1].start == pd.Timestamp("2024-01-05") and runs[1].end == pd.Timestamp(
        "2024-01-06"
    )
    assert weather_source_phrase(
        runs, pd.Timestamp("2024-01-04"), pd.Timedelta(days=1)
    ) == (
        "Ahead, it comes from the weather forecast for 2 days, then typical seasonal values (climatology) for 4 days."
    )
    assert (
        weather_source_phrase(
            runs[:1], pd.Timestamp("2024-01-04"), pd.Timedelta(days=1)
        )
        == ""
    )
    only = [
        SourceRun(
            "climatology", pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-03"), 3
        )
    ]
    assert weather_source_phrase(
        only, pd.Timestamp("2024-01-31"), pd.Timedelta(days=1)
    ) == ("Ahead, all of it comes from typical seasonal values (climatology).")


def test_weather_runs_skip_unlabelled_rows() -> None:
    runs = weather_source_runs(
        pd.date_range("2024-01-01", periods=4), ["archive", None, None, "forecast"]
    )
    assert [r.source for r in runs] == ["archive", "forecast"]


def test_align_sources() -> None:
    fcst = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=3)}, index=[10, 11, 12]
    )
    by_position = visuals._align_sources(fcst, ["archive", "forecast", "climatology"])
    assert list(by_position.index) == [10, 11, 12]
    by_date = visuals._align_sources(
        fcst,
        pd.Series(
            ["forecast", "archive", "climatology"],
            index=pd.DatetimeIndex(fcst["ds"][::-1]),
        ),
    )
    assert list(by_date) == ["climatology", "archive", "forecast"]
    with pytest.raises(ValueError, match="Unknown weather source"):
        visuals._align_sources(fcst, ["archive", "rain", "forecast"])
    with pytest.raises(ValueError, match="3 rows"):
        visuals._align_sources(fcst, ["archive"])


# ---- backtest across the configuration matrix -------------------------------


@pytest.mark.parametrize("case", CASES)
def test_backtest_across_configurations(case: str) -> None:
    m, df, _ = fit_case(case)
    with no_warnings():
        bt = backtest_case(case)
        fig = plot_backtest(m, df, HORIZONS.get(case, 30), backtest=bt)
        plain = plot_backtest(
            m, df, HORIZONS.get(case, 30), backtest=bt, annotate=False, title=False
        )
    assert isinstance(fig, Figure) and isinstance(plain, Figure)
    assert bt.metrics.n == len(bt.holdout)
    assert bt.metrics.mae >= 0 and bt.metrics.rmse >= bt.metrics.mae - 1e-9
    assert len(bt.train) + len(bt.holdout) == len(df)
    title = " ".join(texts(fig)[0].split())
    assert title.startswith("Backtest: ")
    if case == "no_uncertainty":
        assert bt.metrics.coverage is None and "band" not in title
    else:
        assert (
            bt.metrics.coverage is not None
            and "of actuals inside the 80% band" in title
        )
    assert "—" not in "".join(texts(fig))


def test_backtest_title_and_text_overrides() -> None:
    m, df, _ = fit_case("linear")
    bt = backtest_case("linear")
    fig = plot_backtest(
        m, df, 30, backtest=bt, title="Mine", subtitle="Sub", source="Src"
    )
    assert {"Mine", "Sub", "Src"} <= set(texts(fig))
    fig = plot_backtest(m, df, 30, backtest=bt, title="Mine")
    assert "Mine" in texts(fig) and not any("refitted" in t for t in texts(fig))


def test_backtest_on_user_axes() -> None:
    m, df, _ = fit_case("linear")
    fig, ax = plt.subplots(figsize=(9, 5))
    with no_warnings():
        out = plot_backtest(m, df, 30, ax=ax, backtest=backtest_case("linear"))
    assert out is fig
    assert ax.get_title(loc="left").startswith("Backtest")


def test_backtest_refits_a_copy_and_leaves_the_model_alone() -> None:
    m, df, _ = fit_case("regressors")
    history_rows = len(m.history)
    bt = run_backtest(m, df, "30D")
    assert len(m.history) == history_rows
    assert bt.span == "the last 30 days"
    limit = df["ds"].max() - pd.Timedelta(days=30)
    assert bt.cutoff == df.loc[df["ds"] <= limit, "ds"].max()
    assert len(bt.holdout) == int(
        (df["ds"] > df["ds"].max() - pd.Timedelta(days=30)).sum()
    )
    assert {"x", "z"} <= set(bt.forecast.columns)
    assert "30 days" in backtest_headline(bt.metrics, bt.span)


def test_backtest_keeps_the_model_configuration() -> None:
    m, df, _ = fit_case("logistic")
    bt = run_backtest(m, df, 20)
    assert (
        "cap" in bt.forecast and bt.forecast["yhat"].max() <= df["cap"].iloc[0] + 1e-6
    )


def test_backtest_errors_are_specific() -> None:
    m, df, _ = fit_case("regressors")
    with pytest.raises(BacktestError, match=r"lacks the column\(s\) \['x', 'z'\]"):
        run_backtest(m, df.drop(columns=["x", "z"]), 10)
    with pytest.raises(BacktestError, match="fitted"):
        run_backtest(Prophet(), df, 10)
    with pytest.raises(BacktestError, match="fewer than 2 rows"):
        run_backtest(m, df, len(df) - 1)


# ---- regressor effects ---------------------------------------------------------


@pytest.mark.parametrize("case", REGRESSOR_CASES)
def test_regressor_effects_figure(case: str) -> None:
    m, _, fcst = fit_case(case)
    with no_warnings():
        fig = plot_regressor_effects(m, fcst)
        quiet = plot_regressor_effects(
            m, fcst, annotate=False, title=False, uncertainty=False
        )
    assert isinstance(fig, Figure) and isinstance(quiet, Figure)
    shown = texts(fig)
    assert shown[0].startswith("Extra regressors: ")
    assert shown.count("FITTED EFFECT") == len(m.extra_regressors)
    assert any(
        "adds" in t or "raises" in t or "takes" in t or "lowers" in t for t in shown
    )


def test_regressor_effects_say_the_coefficient_in_words() -> None:
    m, _, fcst = fit_case("temperature")
    shown = "\n".join(texts(plot_regressor_effects(m, fcst)))
    assert "per °C" in shown and "Each extra °C" in shown
    assert "when on" in shown and "When promo is on" in shown


def test_regressor_effects_without_regressors_fail_clearly() -> None:
    m, _, fcst = fit_case("linear")
    with pytest.raises(NoRegressorsError, match="no extra regressors"):
        plot_regressor_effects(m, fcst)


def test_regressor_effects_need_the_regressor_columns() -> None:
    m, _, fcst = fit_case("regressors")
    with pytest.raises(NoRegressorsError, match="no column"):
        plot_regressor_effects(m, fcst.drop(columns=["x"]))


def test_regressor_effects_label_the_weather_sources() -> None:
    m, df, fcst = fit_case("temperature")
    sources = weather_labels(fcst, df["ds"].max())
    with no_warnings():
        fig = plot_regressor_effects(m, fcst, weather_sources=sources)
    shown = " ".join(" ".join(texts(fig)).split())
    assert "Observed weather" in shown
    assert (
        "Ahead, it comes from the weather forecast for 9 days, then typical seasonal values (climatology)"
        in shown
    )
    assert (
        "Weather forecast" in shown
    )  # keyed beside the headline: its stretch is too narrow to carry a label


def test_weather_sources_need_a_weather_regressor() -> None:
    m, df, fcst = fit_case("regressors")
    with pytest.raises(ValueError, match="no 'temperature' regressor"):
        plot_regressor_effects(
            m, fcst, weather_sources=weather_labels(fcst, df["ds"].max())
        )
    m, df, fcst = fit_case("temperature")
    with pytest.raises(ValueError, match="Unknown weather source"):
        plot_regressor_effects(m, fcst, weather_sources=["rain"] * len(fcst))
    with pytest.raises(ValueError, match="rows"):
        plot_regressor_effects(m, fcst, weather_sources=["archive"] * 3)


def test_weather_regressor_name_is_configurable() -> None:
    m, df, fcst = fit_case("regressors")
    with no_warnings():
        fig = plot_regressor_effects(
            m,
            fcst,
            weather_sources=weather_labels(fcst, df["ds"].max()),
            weather_regressor="x",
        )
    assert "Observed weather" in "\n".join(texts(fig))


# ---- report ----------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES + ["temperature"])
def test_report_across_configurations(case: str) -> None:
    m, df, fcst = fit_case(case)
    with no_warnings():
        fig = plot_report(m, fcst, df=df)
        with_metrics = plot_report(
            m,
            fcst,
            df=df,
            metrics=backtest_case(case).metrics if case in CASES else None,
        )
    assert isinstance(fig, Figure) and isinstance(with_metrics, Figure)
    shown = texts(fig)
    assert shown[0].startswith("Forecast report: ")
    for label in ("AT A GLANCE", "THE FORECAST", "HISTORY", "FORECAST AVERAGE"):
        assert label in shown
    assert ("EXTRA REGRESSORS" in shown) == (case in ("regressors", "temperature"))
    if case in CASES:
        assert "TYPICAL ERROR" in texts(with_metrics) or "AVERAGE MISS" in texts(
            with_metrics
        )
    assert "—" not in "".join(shown)


def test_report_is_a_single_page_pdf(tmp_path: Path) -> None:
    m, df, fcst = fit_case("temperature")
    fig = plot_report(
        m, fcst, df=df, weather_sources=weather_labels(fcst, df["ds"].max())
    )
    out = tmp_path / "report.pdf"
    with no_warnings():
        fig.savefig(out)
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page(?!s)", data)) == 1
    width, height = fig.get_size_inches()
    assert width == pytest.approx(8.27) and 11.69 <= height <= 16.5


def test_report_page_stays_a4_when_light_and_grows_when_dense() -> None:
    m, df, fcst = fit_case("linear")
    assert plot_report(m, fcst, df=df).get_size_inches()[1] == pytest.approx(11.69)
    m, df, fcst = fit_case("temperature")
    dense = plot_report(
        m, fcst, df=df, metrics=backtest_metrics([1.0, 2.0], [1.0, 2.5])
    )
    assert dense.get_size_inches()[1] > 11.69


def test_report_caps_the_regressors_it_charts() -> None:
    m, df, fcst = fit_case("temperature")
    shown = texts(plot_report(m, fcst, df=df, max_regressors=1))
    assert shown.count("FITTED EFFECT") == 1
    assert any(t.startswith("Also in the model, with smaller effects:") for t in shown)


def test_report_text_overrides() -> None:
    m, df, fcst = fit_case("linear")
    shown = texts(
        plot_report(m, fcst, df=df, title="Mine", subtitle="Sub", source="Src")
    )
    assert {"Mine", "Sub", "Src"} <= set(shown)
    assert not any(t.startswith("Forecast report") for t in shown)


def test_report_summary_numbers_come_from_the_data() -> None:
    m, df, fcst = fit_case("linear")
    summary = visuals._summarise(m, fcst, df)
    assert summary.n_obs == len(df)
    assert summary.last_value == pytest.approx(df["y"].iloc[-1])
    assert summary.n_future == 30
    assert summary.forecast_mean == pytest.approx(fcst["yhat"].iloc[-30:].mean())
    tiles = {t.label: t for t in visuals._summary_tiles(summary)}
    assert tiles["FREQUENCY"].value == "Daily"
    assert tiles["HORIZON"].value == "4 weeks"


def test_the_observed_range_is_part_of_the_report_summary() -> None:
    m, df, fcst = fit_case("linear")
    summary = visuals._summarise(m, fcst, df)
    assert summary.y_min == pytest.approx(df["y"].min())
    assert summary.y_max == pytest.approx(df["y"].max())


def summary_with(mean: float | None, low: float = 25.0, high: float = 72.0) -> Any:
    ds = pd.Timestamp("2024-01-31")
    return visuals._Summary(
        n_obs=31,
        start=pd.Timestamp("2024-01-01"),
        end=ds,
        step=pd.Timedelta(days=1),
        last_value=40.0,
        n_future=28,
        future_end=ds + pd.Timedelta(days=28),
        forecast_mean=mean,
        recent_mean=40.0,
        window=pd.Timedelta(days=28),
        scale=72.0,
        y_min=low,
        y_max=high,
    )


def test_a_forecast_average_far_outside_the_data_is_flagged_in_the_headline() -> None:
    low = visuals._range_warning(summary_with(-8.18))
    assert low == (
        "That average is below every value in the history (the lowest is 25), "
        "so check the model before relying on the forecast."
    )
    high = visuals._range_warning(summary_with(120.0))
    assert high.startswith(
        "That average is above every value in the history (the highest is 72)"
    )
    # Inside the range, or only a little outside it, a trend can explain the average.
    assert visuals._range_warning(summary_with(50.0)) == ""
    assert visuals._range_warning(summary_with(20.0)) == ""
    assert visuals._range_warning(summary_with(80.0)) == ""
    assert visuals._range_warning(summary_with(None)) == ""
    assert visuals._range_warning(summary_with(10.0, low=float("nan"))) == ""
    # Data that never goes below zero cannot average below it.
    assert visuals._range_warning(summary_with(-1.0, low=0.0, high=100.0)) != ""
    assert visuals._range_warning(summary_with(-1.0, low=-50.0, high=100.0)) == ""
    m, _, _ = fit_case("linear")
    _, sub = visuals._report_headline(m, summary_with(-8.18))
    assert (
        "The forecast averages -8.18 over the next 4 weeks. That average is below"
        in sub
    )
    _, quiet = visuals._report_headline(m, summary_with(50.0))
    assert "check the model" not in quiet


def test_dates_in_prose_do_not_carry_a_leading_zero() -> None:
    day = pd.Timedelta(days=1)
    stamp = pd.Timestamp("2014-03-05 09:05")
    assert visuals._fmt_day(stamp, day) == "5 Mar 2014"
    assert visuals._fmt_day(stamp, pd.Timedelta(minutes=5)) == "5 Mar 2014 09:05"
    assert visuals._fmt_date(stamp, pd.Timedelta(days=100)) == "5 Mar 2014"
    assert visuals._fmt_date(stamp, pd.Timedelta(days=3)) == "5 Mar 09:05"
    assert visuals._fmt_date(stamp, pd.Timedelta(days=900)) == "Mar 2014"
    assert visuals._day_month(pd.Timestamp("2014-12-25")) == "25 Dec"


def test_report_without_a_forecast_says_so() -> None:
    m, df, fcst = fit_case("linear")
    in_sample = fcst[fcst["ds"] <= df["ds"].max()]
    with no_warnings():
        fig = plot_report(m, in_sample, df=df)
    assert texts(fig)[0].startswith("Model report")


# ---- dashboard ---------------------------------------------------------------------


def test_dashboard_requires_plotly_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("plotly"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    m, _, fcst = fit_case("linear")
    with pytest.raises(ImportError, match="pip install plotly"):
        dashboard_html(m, fcst, tmp_path / "x.html")
    assert not (tmp_path / "x.html").exists()


@pytest.mark.parametrize(
    "case",
    [
        "linear",
        "logistic",
        "multiplicative",
        "subdaily",
        "no_uncertainty",
        "temperature",
    ],
)
def test_dashboard_html(case: str, tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case(case)
    sources = weather_labels(fcst, df["ds"].max()) if case == "temperature" else None
    out = tmp_path / "nested" / "dash.html"
    with no_warnings():
        path = dashboard_html(m, fcst, out, df=df, weather_sources=sources)
    assert path == out and out.exists()
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert (
        "plotly.js" in page and "Plotly.newPlot" in page and len(page) > 2_000_000
    )  # the bundle is embedded
    assert '<script src="' not in page  # nothing is fetched from a CDN
    assert "<title>Forecast report: " in page
    for heading in ("Forecast", "Components", "At a glance"):
        assert f">{heading}<" in page
    assert ("<h2>Extra regressors</h2>" in page) == (case == "temperature")
    if case == "temperature":
        assert "+3" in page or "per °C" in page
        assert "Ahead, it comes from the weather forecast for 9 days" in page
    assert "—" not in page.split("<script", 1)[0]


def test_dashboard_title_overrides_and_escaping(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    metrics = backtest_metrics(
        [1.0, 2.0, 3.0], [1.0, 2.5, 3.0], [0.0] * 3, [4.0] * 3, interval_width=0.8
    )
    path = dashboard_html(
        m,
        fcst,
        tmp_path / "d.html",
        title="Sales <2025>",
        subtitle="A & B",
        metrics=metrics,
    )
    page = path.read_text(encoding="utf-8")
    assert "<title>Sales &lt;2025&gt;</title>" in page
    assert "A &amp; B" in page
    assert "How well it works on held-back data" in page and "INSIDE THE BAND" in page


def test_dashboard_rejects_weather_sources_without_regressors(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    with pytest.raises(ValueError, match="no extra regressors"):
        dashboard_html(
            m, fcst, tmp_path / "d.html", weather_sources=["archive"] * len(fcst)
        )


def test_frequency_wording_names_weekday_only_series() -> None:
    from prophet.plot import skip_axis

    ds = pd.Series(pd.bdate_range("2024-01-01", periods=30))
    day = pd.Timedelta(days=1)
    weekends = skip_axis(ds, skip_days="weekends")
    sundays = skip_axis(ds, skip_days="Sun")
    assert visuals._frequency_with_skips(day, None) == ("Daily", "one point per day")
    assert visuals._frequency_with_skips(day, weekends) == (
        "Weekdays",
        "one point per weekday",
    )
    assert (
        visuals._frequency_with_skips(day, sundays)[1]
        == "one point per day on the days kept"
    )
