"""Tests for the "Accuracy" dashboard section.

The wording and numbers come from pure functions, tested here on small handmade
frames. The smoke tests build the whole section over the fixture matrix of
``test_visuals`` and need plotly.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from prophet import Prophet  # noqa: E402
from prophet.dashboard import accuracy  # noqa: E402
from prophet.dashboard._context import build_context  # noqa: E402
from prophet.dashboard._palette import DARK_SWAP  # noqa: E402
from prophet.diagnostics import cross_validation  # noqa: E402
from prophet.plot import skip_axis  # noqa: E402
from prophet.tests.test_visuals import (  # noqa: E402
    CASES,
    backtest_case,
    fit_case,
    no_warnings,
)
from prophet.visuals import (  # noqa: E402
    Backtest,
    BacktestMetrics,
    backtest_metrics,
    run_backtest,
)

DATA = Path(__file__).parent
#: The configuration matrix of test_visuals, plus the weather regressors.
MATRIX = [*CASES, "temperature"]
DAY = pd.Timedelta(days=1)
HOUR = pd.Timedelta(hours=1)


# ---- handmade inputs --------------------------------------------------------


def make_cv(
    cutoffs: int = 3,
    days: int = 5,
    *,
    intervals: bool = True,
    y: float = 100.0,
    start: str = "2024-01-10",
) -> pd.DataFrame:
    """Cross-validation output where the forecast is off by exactly ``h`` at ``h`` days ahead.

    So the MAE and RMSE at horizon ``h`` are ``h``, the MAPE is ``h / y`` and the
    interval (plus or minus 3) catches the actual only up to three days ahead.
    """
    rows = []
    for k in range(cutoffs):
        cutoff = pd.Timestamp(start) + k * 10 * DAY
        for h in range(1, days + 1):
            row: dict[str, Any] = {
                "ds": cutoff + h * DAY,
                "cutoff": cutoff,
                "y": y,
                "yhat": y + h,
            }
            if intervals:
                row |= {"yhat_lower": y + h - 3, "yhat_upper": y + h + 3}
            rows.append(row)
    return pd.DataFrame(rows)


def make_backtest(
    n_train: int = 60,
    n_hold: int = 12,
    *,
    intervals: bool = True,
    span: str | None = None,
) -> Backtest:
    """A backtest whose forecast is 2 too high early, 4 in the middle and 8 late in the holdout."""
    ds = pd.date_range("2024-01-01", periods=n_train + n_hold, freq="D")
    y = 100.0 + np.arange(n_train + n_hold)
    train = pd.DataFrame({"ds": ds[:n_train], "y": y[:n_train]})
    holdout = pd.DataFrame({"ds": ds[n_train:], "y": y[n_train:]})
    third = n_hold // 3
    error = np.r_[
        np.full(third, 2.0), np.full(n_hold - 2 * third, 4.0), np.full(third, 8.0)
    ]
    forecast = holdout[["ds"]].assign(yhat=holdout["y"].to_numpy() + error)
    if intervals:
        forecast = forecast.assign(
            yhat_lower=forecast["yhat"] - 5.0, yhat_upper=forecast["yhat"] + 5.0
        )
    metrics = backtest_metrics(
        holdout["y"],
        forecast["yhat"],
        forecast.get("yhat_lower"),
        forecast.get("yhat_upper"),
        interval_width=0.8,
    )
    return Backtest(
        cutoff=pd.Timestamp(ds[n_train - 1]),
        span=span or f"the last {n_hold} days",
        train=train,
        holdout=holdout,
        forecast=forecast,
        metrics=metrics,
    )


@lru_cache(maxsize=None)
def cv_case(name: str) -> pd.DataFrame:
    """A cheap cross-validation of one fixture model."""
    m, _, _ = fit_case(name)
    if name == "subdaily":
        return cross_validation(
            m, horizon="2 hours", period="12 hours", initial="1 day", disable_tqdm=True
        )
    return cross_validation(
        m, horizon="10 days", period="90 days", initial="400 days", disable_tqdm=True
    )


@lru_cache(maxsize=None)
def weekday_backtest() -> Backtest:
    m, df, _ = fit_case("linear")
    return run_backtest(m, df, 30, skip_days="weekends")


@lru_cache(maxsize=None)
def short_case() -> tuple[Prophet, pd.DataFrame, pd.DataFrame, Backtest, pd.DataFrame]:
    """About four months of weekday data: model, data, forecast, backtest and cross-validation."""
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"]).head(80)
    m = Prophet()
    m.fit(df)
    fcst = m.predict(m.make_future_dataframe(periods=10))
    cv = cross_validation(
        m, horizon="7 days", period="14 days", initial="42 days", disable_tqdm=True
    )
    return m, df, fcst, run_backtest(m, df, 10), cv


def figure_colours(fig: Any) -> set[str]:
    """Every colour string a figure carries, normalised the way DARK_SWAP keys are."""
    spec = fig.to_dict()
    spec["layout"].pop("template", None)
    text = json.dumps(spec, default=str)
    found = re.findall(r"#[0-9a-fA-F]{6}\b", text) + re.findall(r"rgba?\([^)]*\)", text)
    return {c.upper().replace(" ", "") for c in found}


def assert_figure_rules(fig: Any) -> None:
    """The dashboard's chart rules: palette colours, hover on every trace, no second y axis."""
    assert figure_colours(fig) <= set(DARK_SWAP)
    for trace in fig.data:
        assert trace.hovertemplate or trace.hoverinfo == "skip"
    for axis in fig.select_yaxes():
        assert axis.overlaying is None and axis.side is None


# ---- units and wording ------------------------------------------------------


@pytest.mark.parametrize(
    ("shortest", "longest", "unit"),
    [
        (DAY, 30 * DAY, "day"),
        (DAY, 2 * DAY, "day"),
        (7 * DAY, 28 * DAY, "day"),
        (5 * pd.Timedelta(minutes=1), 4 * DAY, "day"),
        (5 * pd.Timedelta(minutes=1), 2 * DAY, "hour"),
        (5 * pd.Timedelta(minutes=1), 2 * HOUR, "hour"),
        (5 * pd.Timedelta(minutes=1), 90 * pd.Timedelta(minutes=1), "minute"),
    ],
)
def test_horizon_scale_picks_the_natural_unit(
    shortest: pd.Timedelta, longest: pd.Timedelta, unit: str
) -> None:
    assert accuracy.horizon_scale(shortest, longest).unit == unit


def test_horizon_scale_converts_and_words_durations() -> None:
    scale = accuracy.horizon_scale(pd.Timedelta(minutes=5), 6 * HOUR)
    assert scale.of(90 * pd.Timedelta(minutes=1)) == pytest.approx(1.5)
    assert scale.phrase(3 * HOUR) == "3 hours"
    assert scale.phrase(20 * pd.Timedelta(minutes=1)) == "20 minutes"
    assert accuracy.horizon_scale(DAY, 30 * DAY).phrase(12 * HOUR) == "12 hours"
    assert accuracy.horizon_scale(DAY, 30 * DAY).phrase(DAY) == "1 day"
    column = scale.column(pd.Series([HOUR, 2 * HOUR, 30 * pd.Timedelta(minutes=1)]))
    assert column.tolist() == [1.0, 2.0, 0.5]


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1.0, "day", "1 day"),
        (3.0, "day", "3 days"),
        (2.5, "hour", "2.5 hours"),
        (0.5, "day", "0.5 days"),
        (12.7, "day", "13 days"),
        (9.96, "day", "10 days"),
    ],
)
def test_amount_phrase(value: float, unit: str, expected: str) -> None:
    assert accuracy.amount_phrase(value, unit) == expected


@pytest.mark.parametrize(
    ("new", "old", "expected"),
    [
        (10.0, 10.0, "about the same as"),
        (11.0, 10.0, "about the same as"),
        (14.0, 10.0, "about 40% larger than"),
        (25.0, 10.0, "about 2.5 times as large as"),
        (6.0, 10.0, "about 40% smaller than"),
        (5.0, 0.0, ""),
        (5.0, -1.0, ""),
        (float("nan"), 10.0, ""),
    ],
)
def test_change_phrase(new: float, old: float, expected: str) -> None:
    assert accuracy.change_phrase(new, old) == expected


def test_coverage_sentence_says_whether_the_band_is_calibrated() -> None:
    assert accuracy.coverage_sentence(None, 0.8, 100) == ""
    assert accuracy.coverage_sentence(0.8, None, 100) == ""
    close = accuracy.coverage_sentence(0.78, 0.8, 100)
    assert (
        close
        == "78% of actuals fell inside the 80% band, close to the 80% it is meant to catch."
    )
    narrow = accuracy.coverage_sentence(0.5, 0.8, 100)
    assert "50% of actuals" in narrow and narrow.endswith("so the band is too narrow.")
    wide = accuracy.coverage_sentence(0.97, 0.8, 100)
    assert "wider than it needs to be" in wide
    rough = accuracy.coverage_sentence(0.8, 0.8, 7)
    assert rough.endswith("With only 7 points this is a rough guide.")
    assert "rough guide" not in accuracy.coverage_sentence(0.8, 0.8, 20)
    assert accuracy.coverage_sentence(1.0, 0.8, 1).endswith(
        "With only 1 point this is a rough guide."
    )


def test_coverage_tolerance_widens_when_there_are_few_points() -> None:
    assert accuracy.calibration_tolerance(0.8, 10_000) == accuracy.CALIBRATION_TOLERANCE
    assert accuracy.calibration_tolerance(0.8, 30) == pytest.approx(
        2 * (0.8 * 0.2 / 30) ** 0.5
    )
    assert accuracy.calibration_tolerance(0.8, 30) > accuracy.calibration_tolerance(
        0.8, 100
    )
    assert accuracy.calibration_tolerance(0.8, 0) == 1.0


def test_coverage_sentence_does_not_blame_the_band_for_noise() -> None:
    # A calibrated 80% band measured on 30 points is 70% or lower about one run in six.
    for n in (25, 30, 50):
        noisy = accuracy.coverage_sentence(0.7, 0.8, n)
        assert noisy.endswith("though a gap this size could be chance.")
        assert "too narrow" not in noisy
    assert "though a gap this size could be chance" in accuracy.coverage_sentence(
        0.92, 0.8, 30
    )
    # The same gap is a verdict once there are enough points to trust it.
    assert accuracy.coverage_sentence(0.7, 0.8, 300).endswith(
        "so the band is too narrow."
    )
    assert "wider than it needs to be" in accuracy.coverage_sentence(0.9, 0.8, 300)
    # A gap beyond the noise at 30 points is still called.
    assert accuracy.coverage_sentence(0.45, 0.8, 30).endswith(
        "so the band is too narrow."
    )


def test_padded_range_leaves_room_above_for_labels() -> None:
    assert accuracy.padded_range([]) is None
    assert accuracy.padded_range([np.array([np.nan])]) is None
    low, high = accuracy.padded_range(
        [np.array([0.0, 10.0]), pd.Series([5.0, np.nan])]
    ) or (0, 0)
    assert low == pytest.approx(-0.6) and high == pytest.approx(11.4)
    flat = accuracy.padded_range([[5.0, 5.0]])
    assert flat is not None and flat[0] < 5.0 < flat[1]


def test_thin_keeps_the_ends_and_the_limit() -> None:
    frame = pd.DataFrame({"v": range(10_000)})
    thinned = accuracy.thin(frame, 100)
    assert len(thinned) <= 100
    assert thinned["v"].iloc[0] == 0 and thinned["v"].iloc[-1] == 9_999
    small = pd.DataFrame({"v": range(5)})
    assert accuracy.thin(small, 100) is small


def test_date_labels_follow_the_resolution() -> None:
    ds = pd.Series(pd.to_datetime(["2024-03-05 14:30:00"]))
    assert accuracy.date_labels(ds, DAY) == ["05 Mar 2024"]
    assert accuracy.date_labels(ds, HOUR) == ["05 Mar 2024 14:30"]


# ---- held-back test: pure parts -------------------------------------------


def test_holdout_errors_align_by_date_and_sign_the_error() -> None:
    bt = make_backtest()
    shuffled = Backtest(
        **{**bt.__dict__, "forecast": bt.forecast.sample(frac=1.0, random_state=1)}
    )
    frame = accuracy.holdout_errors(shuffled)
    assert frame["ds"].is_monotonic_increasing and len(frame) == 12
    # Actual minus forecast, as the diagnostics section reads a miss: the forecast ran high.
    assert frame["error"].tolist() == [-2.0] * 4 + [-4.0] * 4 + [-8.0] * 4
    assert frame["abs_error"].tolist() == [2.0] * 4 + [4.0] * 4 + [8.0] * 4
    assert frame["ahead"].iloc[0] == DAY and frame["ahead"].iloc[-1] == 12 * DAY
    assert (
        frame["inside"].tolist() == [True] * 8 + [False] * 4
    )  # the band is plus or minus 5


def test_holdout_errors_without_interval_or_with_gaps() -> None:
    bt = make_backtest(intervals=False)
    frame = accuracy.holdout_errors(bt)
    assert "inside" not in frame and "yhat_lower" not in frame
    gappy = bt.holdout.copy()
    gappy.loc[3, "y"] = np.nan
    dropped = accuracy.holdout_errors(Backtest(**{**bt.__dict__, "holdout": gappy}))
    assert len(dropped) == 11
    partial = make_backtest()
    forecast = partial.forecast.copy()
    forecast.loc[0, "yhat_lower"] = np.nan
    no_band = accuracy.holdout_errors(
        Backtest(**{**partial.__dict__, "forecast": forecast})
    )
    assert "inside" not in no_band  # an incomplete interval is not drawn or scored


@pytest.mark.parametrize(
    ("n_hold", "n_train", "expected"),
    [(30, 1000, 90), (2, 1000, 20), (30, 50, 50), (1000, 100_000, 1500), (5, 0, 0)],
)
def test_context_rows(n_hold: int, n_train: int, expected: int) -> None:
    assert accuracy.context_rows(n_hold, n_train) == expected


def test_growth_sentence_compares_the_last_third_with_the_first() -> None:
    errors = [2.0] * 4 + [4.0] * 4 + [8.0] * 4
    assert accuracy.growth_sentence(errors) == (
        "Misses in the last third of the holdout were about 4.0 times as large as those in the first third."
    )
    assert "about the same as those" in accuracy.growth_sentence([3.0] * 9)
    assert "smaller than" in accuracy.growth_sentence([8.0] * 3 + [4.0] * 3 + [2.0] * 3)
    assert accuracy.growth_sentence([1.0, 2.0, 3.0, 4.0, 5.0]) == ""
    assert accuracy.growth_sentence([0.0] * 9) == ""


def test_bias_sentence_ignores_leans_that_noise_would_give() -> None:
    assert accuracy.bias_sentence([]) == ""
    assert accuracy.bias_sentence([0.0, 0.0]) == ""
    assert accuracy.bias_sentence([3.0, -3.0, 2.0, -2.0]) == (
        "The forecast did not lean consistently high or low."
    )
    # The misses are actual minus forecast: positive means the forecast ran low.
    assert (
        accuracy.bias_sentence([3.0] * 5)
        == "On average the forecast ran 3 below the actuals."
    )
    assert (
        accuracy.bias_sentence([-1.5] * 5)
        == "On average the forecast ran 1.5 above the actuals."
    )


def test_backtest_wording_states_the_numbers() -> None:
    bt = make_backtest()
    step = DAY
    lede = accuracy.backtest_lede(bt, bt.metrics, step)
    assert lede.startswith(
        "Refitted on data up to 29 Feb 2024 and asked to forecast the last 12 days, "
        "the model had a typical error of"
    )
    assert "an average miss of 4.67." in lede
    assert "of actuals fell inside the 80% band" in lede and "rough guide" in lede
    summary = accuracy.backtest_summary(bt, bt.metrics)
    assert summary.startswith("Held back the last 12 days: typical error ")
    assert summary.endswith("of actuals inside the 80% band.")
    for text in (lede, summary):
        assert "—" not in text and "\n" not in text


def test_backtest_wording_without_percentages_or_intervals() -> None:
    bt = make_backtest(intervals=False)
    flat = BacktestMetrics(
        n=12,
        mae=4.0,
        rmse=5.0,
        mape=None,
        coverage=None,
        interval_width=None,
        mean_actual=0.0,
    )
    lede = accuracy.backtest_lede(bt, flat, DAY)
    assert lede.endswith("the model missed by 4 on average.")
    assert accuracy.backtest_summary(bt, flat) == (
        "Held back the last 12 days: forecasts missed by 4 on average."
    )
    hidden = accuracy.visible_metrics(make_backtest().metrics, False)
    assert hidden.coverage is None and hidden.interval_width is None
    assert hidden.mae == make_backtest().metrics.mae


def test_span_phrase_rewords_a_row_count_that_covers_more_calendar() -> None:
    # 30 daily rows that fall two days apart cover eight weeks, not "30 days".
    gappy = make_backtest(n_hold=30, span="the last 30 days")
    days = pd.date_range("2024-03-01", periods=30, freq="2D")
    holdout = pd.DataFrame({"ds": days, "y": 1.0})
    stretched = Backtest(
        **{**gappy.__dict__, "holdout": holdout, "forecast": holdout.assign(yhat=1.0)}
    )
    assert accuracy.span_phrase(stretched) == "the last 8 weeks"
    # Rows with no gaps keep the backtest's own wording.
    assert accuracy.span_phrase(gappy) == "the last 30 days"
    # So does a span the caller wrote themselves, even when the rows run short of it.
    own = Backtest(**{**stretched.__dict__, "span": "the last 90 days"})
    assert accuracy.span_phrase(own) == "the last 90 days"
    # And one already measured on the calendar for skipped days.
    weekdays = Backtest(**{**stretched.__dict__, "span": "the last 6 weeks"})
    assert accuracy.span_phrase(weekdays) == "the last 6 weeks"
    one = make_backtest(n_hold=1)
    assert accuracy.span_phrase(one) == one.span


def test_largest_misses_are_biggest_first() -> None:
    frame = accuracy.holdout_errors(make_backtest())
    top = accuracy.largest_misses(frame, 3)
    assert top["abs_error"].tolist() == [8.0, 8.0, 8.0]
    assert len(accuracy.largest_misses(frame)) == 5


# ---- cross-validation: pure parts ---------------------------------------


def test_prepare_cv_validates_and_cleans() -> None:
    with pytest.raises(ValueError, match=r"lacks the column\(s\) \['cutoff'\]"):
        accuracy.prepare_cv(make_cv().drop(columns="cutoff"))
    cv = make_cv()
    cv.loc[0, "yhat"] = np.nan
    cv.loc[1, "ds"] = cv.loc[1, "cutoff"]  # no distance ahead, so not a forecast
    cleaned = accuracy.prepare_cv(cv)
    assert len(cleaned) == len(cv) - 2
    assert cleaned["ds"].dtype.kind == "M" and cleaned["cutoff"].dtype.kind == "M"


def test_prepare_cv_drops_skipped_days() -> None:
    cv = make_cv(
        cutoffs=2, days=7, start="2024-01-01"
    )  # Mondays; day 5 and 6 are weekend
    skip = skip_axis(cv["ds"], skip_days="weekends")
    kept = accuracy.prepare_cv(cv, skip)
    assert not kept["ds"].dt.dayofweek.isin([5, 6]).any()
    assert len(kept) == 2 * 5


@pytest.mark.parametrize(
    ("y", "expected"),
    [
        ([10.0, 12.0, 9.0], True),
        ([10.0, 0.0], False),
        ([100.0, 1.0], False),
        ([], False),
    ],
)
def test_mape_is_meaningful(y: list[float], expected: bool) -> None:
    assert accuracy.mape_is_meaningful(y) is expected


def test_cv_summary() -> None:
    summary = accuracy.cv_summary(make_cv())
    assert (
        summary.cutoffs == 3 and summary.shortest == DAY and summary.longest == 5 * DAY
    )
    assert summary.first == pd.Timestamp("2024-01-10")
    assert summary.last == pd.Timestamp("2024-01-30")


def test_horizon_table_matches_the_hand_calculation() -> None:
    perf = accuracy.horizon_table(make_cv(), mape=True, coverage=True)
    assert perf is not None
    assert perf["horizon"].tolist() == [h * DAY for h in range(1, 6)]
    assert perf["mae"].tolist() == pytest.approx([1, 2, 3, 4, 5])
    assert np.sqrt(perf["mse"]).tolist() == pytest.approx([1, 2, 3, 4, 5])
    assert perf["mape"].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])
    assert perf["coverage"].tolist() == pytest.approx([1, 1, 1, 0, 0])
    assert accuracy.with_rmse(perf)["rmse"].tolist() == pytest.approx([1, 2, 3, 4, 5])
    bare = accuracy.horizon_table(make_cv(intervals=False), mape=False, coverage=False)
    assert bare is not None and set(bare.columns) == {"horizon", "mse", "mae"}


def test_bin_horizons_merges_neighbours_and_keeps_rmse_honest() -> None:
    hours = pd.timedelta_range("1h", periods=1000, freq="h")
    perf = pd.DataFrame(
        {"horizon": hours, "mse": np.arange(1000.0) ** 2, "mae": np.arange(1000.0)}
    )
    binned = accuracy.bin_horizons(perf, 100)
    assert len(binned) == 100
    assert binned["horizon"].is_monotonic_increasing
    assert binned["horizon"].iloc[0] < binned["horizon"].iloc[1]
    first_ten = perf.iloc[:10]
    assert binned["mae"].iloc[0] == pytest.approx(first_ten["mae"].mean())
    assert binned["horizon"].iloc[0] == first_ten["horizon"].mean()
    # RMSE comes from the averaged squared error, not from averaging RMSEs.
    assert accuracy.with_rmse(binned)["rmse"].iloc[0] == pytest.approx(
        np.sqrt(first_ten["mse"].mean())
    )
    assert accuracy.bin_horizons(perf.head(50), 100).equals(perf.head(50))


def test_edge_reading_and_headline_say_how_error_changes_with_distance() -> None:
    perf = accuracy.horizon_table(make_cv(), mape=True, coverage=True)
    assert perf is not None
    scale = accuracy.horizon_scale(DAY, 5 * DAY)
    reading = accuracy.edge_reading(perf, "mape")
    assert reading is not None
    assert (reading.near, reading.far) == pytest.approx((0.01, 0.05))
    assert (reading.near_edge, reading.far_edge) == (DAY, 5 * DAY)
    assert accuracy.cv_headline(3, reading, percent=True, scale=scale) == (
        "Across 3 cutoffs the typical error grows from 1% for forecasts up to 1 day ahead "
        "to 5% for forecasts 5 days or more ahead."
    )
    mae = accuracy.edge_reading(perf, "mae")
    assert mae is not None
    assert accuracy.cv_headline(1, mae, percent=False, scale=scale).startswith(
        "Across 1 cutoff the average miss grows from 1 for forecasts"
    )
    assert accuracy.edge_reading(perf, "nonexistent") is None
    assert accuracy.edge_reading(perf.iloc[:0], "mae") is None


def test_cv_headline_covers_every_shape() -> None:
    scale = accuracy.horizon_scale(DAY, 30 * DAY)
    edge = accuracy.EdgeReading
    falling = accuracy.cv_headline(
        4, edge(10.0, 3 * DAY, 4.0, 28 * DAY), percent=False, scale=scale
    )
    assert "average miss falls from 10 for forecasts up to 3 days ahead to 4" in falling
    flat = accuracy.cv_headline(
        4, edge(10.0, 3 * DAY, 10.4, 28 * DAY), percent=False, scale=scale
    )
    assert (
        "barely changes with distance: 10 for forecasts up to 3 days ahead and 10.4"
        in flat
    )
    single = accuracy.cv_headline(
        2, edge(0.07, 7 * DAY, 0.07, 7 * DAY), percent=True, scale=scale
    )
    assert single == "Across 2 cutoffs the typical error was 7% at 7 days ahead."
    from_zero = accuracy.cv_headline(
        2, edge(0.0, DAY, 3.0, 9 * DAY), percent=False, scale=scale
    )
    assert "grows from 0" in from_zero
    assert "—" not in "".join([falling, flat, single, from_zero])


def test_edge_reading_of_one_horizon_is_a_single_point() -> None:
    perf = accuracy.horizon_table(make_cv(days=1), mape=False, coverage=False)
    assert perf is not None and len(perf) == 1
    reading = accuracy.edge_reading(perf, "mae")
    assert reading is not None and reading.near_edge == reading.far_edge == DAY


def test_metric_gloss_names_only_the_panels_drawn() -> None:
    assert accuracy.metric_gloss(["horizon", "mse", "mae", "rmse", "mape"]) == (
        "MAE is the average miss, RMSE weights big misses more and MAPE is the "
        "typical error as a share of the actual."
    )
    assert accuracy.metric_gloss(["mae"]) == "MAE is the average miss."
    assert accuracy.metric_gloss(["horizon", "coverage"]) == ""


def test_cutoff_table_and_rows() -> None:
    cv = make_cv()
    table = accuracy.cutoff_table(cv, mape=True, coverage=True)
    assert table["points"].tolist() == [5, 5, 5]
    assert table["mae"].tolist() == pytest.approx([3.0] * 3)
    assert table["rmse"].tolist() == pytest.approx(
        [np.sqrt(11.0)] * 3
    )  # mean of 1, 4, 9, 16, 25
    assert table["mape"].tolist() == pytest.approx([0.03] * 3)
    assert table["coverage"].tolist() == pytest.approx([0.6] * 3)
    headers, rows = accuracy.cutoff_rows(table, DAY)
    assert headers == ["Cutoff", "Points", "MAE", "RMSE", "MAPE", "Coverage"]
    assert rows[0] == ["10 Jan 2024", "5", "3", "3.32", "3%", "60%"]
    plain = accuracy.cutoff_table(make_cv(intervals=False), mape=False, coverage=False)
    assert list(plain.columns) == ["cutoff", "points", "mae", "rmse"]
    assert accuracy.cutoff_rows(plain, DAY)[0] == ["Cutoff", "Points", "MAE", "RMSE"]


def test_cv_tiles_lead_with_the_run_and_drop_the_holdout_tile() -> None:
    cv = make_cv()
    overall = backtest_metrics(
        cv["y"], cv["yhat"], cv["yhat_lower"], cv["yhat_upper"], interval_width=0.8
    )
    scale = accuracy.horizon_scale(DAY, 5 * DAY)
    tiles = accuracy.cv_tiles(overall, accuracy.cv_summary(cv), scale, DAY)
    labels = [t.label for t in tiles]
    assert labels[:2] == ["CUTOFFS", "HORIZON"] and "HELD BACK" not in labels
    assert labels[-1] == "INSIDE THE BAND"
    assert tiles[0].value == "3" and tiles[0].detail == "10 Jan 2024 to 30 Jan 2024"
    assert tiles[1].value == "5 days" and "15 points scored" in tiles[1].detail
    one = accuracy.cv_tiles(
        overall, accuracy.cv_summary(make_cv(cutoffs=1)), scale, DAY
    )
    assert one[0].detail == "on 10 Jan 2024"
    single = backtest_metrics([100.0], [101.0], interval_width=0.8)
    point = accuracy.cv_tiles(single, accuracy.cv_summary(make_cv(1, 1)), scale, DAY)
    assert point[1].detail.endswith("1 point scored")
    assert "1 points" not in point[1].detail


def test_mape_is_not_meaningful_for_tiny_actuals() -> None:
    assert accuracy.mape_is_meaningful([1e-10, 1e-10]) is False
    assert accuracy.mape_is_meaningful([1e-3, 1e-3]) is True


@pytest.mark.parametrize(("n", "cols"), [(1, 1), (2, 2), (3, 3), (4, 2), (5, 2)])
def test_panel_grid_tiles_the_plot_without_overlap(n: int, cols: int) -> None:
    cells = accuracy.panel_grid(n, cols, 600)
    assert len(cells) == n
    for cell in cells:
        assert 0.0 <= cell.x[0] < cell.x[1] <= 1.0 + 1e-9
        assert -1e-9 <= cell.y[0] < cell.y[1] <= 1.0 + 1e-9
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            apart_x = a.x[1] <= b.x[0] + 1e-9 or b.x[1] <= a.x[0] + 1e-9
            apart_y = a.y[1] <= b.y[0] + 1e-9 or b.y[1] <= a.y[0] + 1e-9
            assert apart_x or apart_y
    assert [c.bottom for c in cells][-cols:] == [True] * min(cols, n)
    assert not any(c.bottom for c in cells[: max(0, n - cols)])


# ---- smoke tests over the fixture matrix -------------------------------------


@pytest.fixture
def plotly() -> Any:
    return pytest.importorskip("plotly")


def has_header(body: str, name: str) -> bool:
    return re.search(rf"<th[^>]*>{name}</th>", body) is not None


def smoke(section: Any) -> None:
    assert section is not None
    assert section.key == "accuracy" and section.title == "Accuracy"
    assert "—" not in section.body and "—" not in section.summary
    assert section.summary and "\n" not in section.summary
    assert len(section.body) < 1_000_000
    assert "NaN" not in re.sub(r"<script.*?</script>", "", section.body, flags=re.S)


@pytest.mark.parametrize("case", MATRIX)
def test_backtest_part_across_configurations(case: str, plotly: Any) -> None:
    m, df, fcst = fit_case(case)
    bt = backtest_case(case)
    ctx = build_context(m, fcst, df=df, backtest=bt)
    with no_warnings():
        section = accuracy.build(ctx)
    smoke(section)
    assert section is not None
    body = section.body
    for heading in (
        "Held-back test",
        "Misses by distance from the cutoff",
        "Largest misses",
    ):
        assert f"<h3>{heading}</h3>" in body
    assert "Cross-validation" not in body
    assert section.summary.startswith(f"Held back {accuracy.span_phrase(bt)}: ")
    # The page shell shows the held-back key numbers once, above the sections.
    assert "AVERAGE MISS" not in body and "BIG MISSES" not in body
    assert 'class="tiles"' not in body
    if case == "no_uncertainty":
        assert "band" not in section.summary
        assert "Inside the band" not in body
    else:
        assert "of actuals inside the 80% band" in section.summary
        assert has_header(body, "Inside the band")
    unit = "hours" if case == "subdaily" else "days"
    assert f"{unit} after the cutoff" in body


@pytest.mark.parametrize("case", MATRIX)
def test_backtest_figures_follow_the_chart_rules(case: str, plotly: Any) -> None:
    m, df, fcst = fit_case(case)
    bt = backtest_case(case)
    ctx = build_context(m, fcst, df=df, backtest=bt)
    frame = accuracy.holdout_errors(bt)
    intervals = "inside" in frame
    main = accuracy._backtest_figure(ctx, bt, frame, intervals)
    scale = accuracy.horizon_scale(
        pd.Timedelta(frame["ahead"].min()), pd.Timedelta(frame["ahead"].max())
    )
    errors = accuracy._error_figure(ctx, frame, scale, bt.metrics.mae)
    for fig in (main, errors):
        assert_figure_rules(fig)
    names = [t.name for t in main.data if t.name]
    assert (
        names[0] == "Training data"
        and "Forecast" in names
        and "Held-back actuals" in names
    )
    assert (f"{ctx.m.interval_width:.0%} interval" in names) is intervals
    assert len(main.data) == len(names) + (
        1 if intervals else 0
    )  # the lower edge of the band has no name
    assert main.layout.xaxis.type == "date" and errors.layout.xaxis.type == "linear"
    assert errors.data[-1].name.startswith("Overall average")  # the reference line


def test_backtest_part_without_a_model_future_or_past(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    bt = backtest_case("linear")
    end = df["ds"].max()
    for part in (fcst[fcst["ds"] > end], fcst[fcst["ds"] <= end]):
        ctx = build_context(m, part, df=df, backtest=bt)
        with no_warnings():
            smoke(accuracy.build(ctx))


def test_section_does_not_apply_without_a_backtest_or_cv(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    assert accuracy.build(build_context(m, fcst, df=df)) is None
    empty = pd.DataFrame(columns=["ds", "cutoff", "y", "yhat"])
    assert accuracy.build(build_context(m, fcst, df=df, cv=empty)) is None
    unusable = make_cv().assign(yhat=np.nan)
    assert accuracy.build(build_context(m, fcst, df=df, cv=unusable)) is None
    with pytest.raises(ValueError, match="lacks the column"):
        accuracy.build(build_context(m, fcst, df=df, cv=make_cv().drop(columns="y")))


def test_intervals_are_left_out_when_the_caller_does_not_want_them(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    ctx = build_context(
        m,
        fcst,
        df=df,
        backtest=backtest_case("linear"),
        cv=cv_case("linear"),
        uncertainty=False,
    )
    section = accuracy.build(ctx)
    smoke(section)
    assert section is not None
    assert "band" not in section.body.lower() and "interval" not in section.body.lower()
    assert not has_header(section.body, "Coverage")


def test_backtest_text_is_escaped(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    bt = backtest_case("linear")
    hostile = Backtest(**{**bt.__dict__, "span": "the last <b>30</b> days"})
    section = accuracy.build(build_context(m, fcst, df=df, backtest=hostile))
    assert section is not None
    assert "&lt;b&gt;30&lt;/b&gt;" in section.body and "<b>30</b>" not in section.body


def assert_hover_reads_for_any_count(fig: Any) -> None:
    """No hover label puts a plural unit after the distance ('1 days'), or drops into exponents."""
    for trace in fig.data:
        template = trace.hovertemplate or ""
        assert not re.search(r"%\{x[^}]*\}(</b>)? (minute|hour|day)s", template)
        for spec in re.findall(r"%\{x:([^}|]+)\}", template):
            assert spec == ".6~g"  # six digits before d3 turns to 1e+6 notation


def test_hover_labels_read_for_one_day_and_for_a_thousand(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    ctx = build_context(m, fcst, df=df)
    bt = make_backtest()
    frame = accuracy.holdout_errors(bt)
    scale = accuracy.horizon_scale(DAY, 12 * DAY)
    assert_hover_reads_for_any_count(accuracy._error_figure(ctx, frame, scale, 4.0))
    table = accuracy.horizon_table(make_cv(), mape=True, coverage=True)
    assert table is not None
    fig = accuracy._horizon_figure(ctx, accuracy.with_rmse(table), scale, 0.8)
    assert_hover_reads_for_any_count(fig)
    assert any("Horizon (days)" in (t.hovertemplate or "") for t in fig.data)


@pytest.mark.parametrize("panels", [1, 2, 3, 4])
def test_panels_stay_wide_enough_for_a_phone(panels: int) -> None:
    cols, height = accuracy.panel_layout(panels)
    assert cols <= 2 and height >= 330
    cells = accuracy.panel_grid(panels, cols, height)
    plot_px = 335 - 64 - 24  # a 375 px screen less the page gutters, the figure margins
    assert min((c.x[1] - c.x[0]) * plot_px for c in cells) >= 100


def column_classes(body: str, table_id: str) -> list[str]:
    """The class of each header cell of a table, in order ('' when it has none)."""
    table = re.search(rf'<table class="tbl" id="{table_id}">(.*?)</thead>', body, re.S)
    assert table is not None
    return re.findall(r"<th(?: class=\"(\w+)\")?>", table.group(1))


def test_misses_table_aligns_every_figure_column_to_the_right(plotly: Any) -> None:
    # The page CSS pads only right-aligned cells on the left, so a left-aligned
    # column beside a numeric one runs into it ("+10.2No").
    m, df, fcst = fit_case("linear")
    for intervals in (True, False):
        bt = make_backtest(intervals=intervals)
        section = accuracy.build(build_context(m, fcst, df=df, backtest=bt))
        assert section is not None
        classes = column_classes(section.body, "accuracy-misses")
        assert len(classes) == (5 if intervals else 4)
        assert classes[0] == "" and set(classes[1:]) == {"num"}


def test_long_holdout_stays_light(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    big = make_backtest(n_train=6000, n_hold=4000)
    section = accuracy.build(build_context(m, fcst, df=df, backtest=big))
    smoke(section)
    assert section is not None and len(section.body) < 900_000
    ctx = build_context(m, fcst, df=df, backtest=big)
    frame = accuracy.holdout_errors(big)
    fig = accuracy._backtest_figure(ctx, big, frame, True)
    assert max(len(t.x) for t in fig.data if t.x is not None) <= accuracy.MAX_POINTS


@pytest.mark.parametrize("case", MATRIX)
def test_cv_part_across_configurations(case: str, plotly: Any) -> None:
    m, df, fcst = fit_case(case)
    cv = cv_case(case)
    ctx = build_context(m, fcst, df=df, cv=cv)
    with no_warnings():
        section = accuracy.build(ctx)
    smoke(section)
    assert section is not None
    body = section.body
    assert "Held-back test" not in body
    for heading in ("Cross-validation", "By cutoff"):
        assert f"<h3>{heading}</h3>" in body
    assert 'id="accuracy-cutoffs"' in body
    assert section.summary.startswith(f"Across {cv['cutoff'].nunique()} cutoffs the ")
    assert (has_header(body, "Coverage")) is (case != "no_uncertainty")
    assert ("Inside the 80% band" in body) is (case != "no_uncertainty")
    assert has_header(body, "MAPE") is (case != "subdaily")  # its actuals include zeros
    unit = "hours" if case == "subdaily" else "days"
    assert f"{unit} ahead" in body


@pytest.mark.parametrize("case", ["linear", "no_uncertainty", "subdaily"])
def test_cv_figures_follow_the_chart_rules(case: str, plotly: Any) -> None:
    m, df, fcst = fit_case(case)
    cv = accuracy.prepare_cv(cv_case(case))
    ctx = build_context(m, fcst, df=df, cv=cv)
    intervals = accuracy.has_interval(cv)
    percent = accuracy.mape_is_meaningful(cv["y"].to_numpy())
    perf = accuracy.horizon_table(cv, mape=percent, coverage=intervals)
    assert perf is not None
    summary = accuracy.cv_summary(cv)
    scale = accuracy.horizon_scale(summary.shortest, summary.longest)
    fig = accuracy._horizon_figure(
        ctx, accuracy.with_rmse(perf), scale, 0.8 if intervals else None
    )
    assert_figure_rules(fig)
    wanted = 2 + int(percent) + int(intervals)
    panels = [t for t in fig.data if t.xaxis is not None]
    assert len(panels) == wanted + int(
        intervals
    )  # the coverage panel also holds its target line
    assert (
        len({t.yaxis for t in panels}) == wanted
    )  # one y axis per metric: small multiples
    assert not fig.layout.showlegend
    shared = [a for a in fig.select_xaxes() if a.matches]
    assert len(shared) == wanted - 1
    by_cutoff = accuracy.cutoff_table(cv, mape=percent, coverage=intervals)
    assert_figure_rules(accuracy._cutoff_figure(ctx, by_cutoff, 3.0))
    if intervals:
        labels = [a.text for a in fig.layout.annotations]
        assert "80% target" in labels and "<b>Inside the 80% band</b>" in labels
        coverage = [t for t in panels if t.name is None and list(t.y) == [0.8, 0.8]]
        assert len(coverage) == 1


def test_cv_without_percentages_when_actuals_include_zero(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    cv = make_cv(y=0.0)
    section = accuracy.build(build_context(m, fcst, df=df, cv=cv))
    smoke(section)
    assert section is not None
    assert not has_header(section.body, "MAPE") and "TYPICAL ERROR" not in section.body
    assert "the average miss grows from" in section.summary


def test_cv_with_one_cutoff_and_one_horizon(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    section = accuracy.build(
        build_context(m, fcst, df=df, cv=make_cv(cutoffs=1, days=1))
    )
    smoke(section)
    assert section is not None and section.summary.startswith("Across 1 cutoff the ")
    assert "was" in section.summary  # a single distance cannot grow


def test_many_horizons_are_merged_and_many_cutoffs_capped(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    rows = []
    rng = np.random.default_rng(0)
    for k in range(accuracy.MAX_CUTOFFS + 20):
        cutoff = pd.Timestamp("2020-01-01") + k * HOUR * 6
        for step in range(1, 61):
            y = 50.0 + rng.normal()
            rows.append(
                {
                    "ds": cutoff + step * pd.Timedelta(minutes=5),
                    "cutoff": cutoff,
                    "y": y,
                    "yhat": y + step * 0.01,
                }
            )
    cv = pd.DataFrame(rows)
    section = accuracy.build(build_context(m, fcst, df=df, cv=cv))
    smoke(section)
    assert section is not None
    assert (
        f"Showing the latest {accuracy.MAX_CUTOFFS} of {accuracy.MAX_CUTOFFS + 20} cutoffs."
        in section.body
    )
    assert "hours ahead" in section.body

    wide = pd.DataFrame(
        [
            {
                "ds": c + h * pd.Timedelta(minutes=5),
                "cutoff": c,
                "y": 10.0 + h % 7,
                "yhat": 11.0,
            }
            for c in pd.date_range("2020-01-01", periods=2, freq="D")
            for h in range(1, 301)
        ]
    )
    merged = accuracy.build(build_context(m, fcst, df=df, cv=wide))
    assert merged is not None and "merging neighbouring distances" in merged.body
    perf = accuracy.horizon_table(wide, mape=True, coverage=False)
    assert perf is not None and len(perf) == 300
    assert len(accuracy.bin_horizons(perf)) == accuracy.MAX_HORIZONS


def test_backtest_and_cv_together_lead_with_the_backtest(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    bt = backtest_case("linear")
    with no_warnings():
        section = accuracy.build(
            build_context(m, fcst, df=df, backtest=bt, cv=cv_case("linear"))
        )
    smoke(section)
    assert section is not None
    assert section.body.index("Held-back test") < section.body.index("Cross-validation")
    assert section.summary.startswith("Held back ")


def test_the_key_finding_adds_the_cross_validation_result_to_the_held_back_one(
    plotly: Any,
) -> None:
    m, df, fcst = fit_case("linear")
    bt = backtest_case("linear")
    cv = cv_case("linear")
    both = accuracy.build(build_context(m, fcst, df=df, backtest=bt, cv=cv))
    held_only = accuracy.build(build_context(m, fcst, df=df, backtest=bt))
    cv_only = accuracy.build(build_context(m, fcst, df=df, cv=cv))
    assert both is not None and held_only is not None and cv_only is not None
    assert both.summary.startswith(held_only.summary)
    follow = both.summary[len(held_only.summary) :].strip()
    assert follow.startswith(
        f"Cross-validation over {cv['cutoff'].nunique()} cutoffs: typical error "
    )
    assert follow.endswith(".") and "of actuals inside the 80% band" in follow
    # Alone, the cross-validation keeps its own lead sentence about distance.
    assert cv_only.summary.startswith(f"Across {cv['cutoff'].nunique()} cutoffs the ")
    assert "Cross-validation over" not in cv_only.summary


def test_cross_validation_wording_matches_the_held_back_wording() -> None:
    metrics = BacktestMetrics(
        n=63,
        mae=2.1,
        rmse=2.7,
        mape=0.038,
        coverage=0.73,
        interval_width=0.8,
        mean_actual=50.0,
    )
    assert accuracy.metrics_clause(metrics) == (
        "typical error 3.8%, with 73% of actuals inside the 80% band"
    )
    assert accuracy.cv_follow(metrics, 3) == (
        "Cross-validation over 3 cutoffs: typical error 3.8%, with 73% of actuals inside the 80% band."
    )
    plain = BacktestMetrics(
        n=5,
        mae=2.3,
        rmse=2.9,
        mape=None,
        coverage=None,
        interval_width=None,
        mean_actual=0.0,
    )
    assert accuracy.metrics_clause(plain) == "forecasts missed by 2.3 on average"
    assert accuracy.cv_follow(plain, 1) == (
        "Cross-validation over 1 cutoff: forecasts missed by 2.3 on average."
    )


def test_a_chart_of_a_few_cutoffs_is_left_to_the_table(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")

    def drawn(cutoffs: int) -> tuple[int, str]:
        cv = make_cv(cutoffs=cutoffs)
        section = accuracy.build(build_context(m, fcst, df=df, cv=cv))
        assert section is not None
        return section.body.count("Average miss at each cutoff"), section.body

    few, body = drawn(accuracy.MIN_CUTOFFS_DRAWN - 1)
    assert few == 0 and 'id="accuracy-cutoffs"' in body
    enough, body = drawn(accuracy.MIN_CUTOFFS_DRAWN)
    assert enough >= 1 and 'id="accuracy-cutoffs"' in body


def test_the_misses_table_and_the_diagnostics_section_read_a_miss_the_same_way(
    plotly: Any,
) -> None:
    m, df, fcst = fit_case("linear")
    bt = backtest_case("linear")
    ctx = build_context(m, fcst, df=df, backtest=bt)
    section = accuracy.build(ctx)
    assert section is not None
    assert has_header(section.body, "Actual minus forecast")
    assert "Forecast minus actual" not in section.body
    frame = accuracy.holdout_errors(bt)
    worst = accuracy.largest_misses(frame, 1).iloc[0]
    assert worst["error"] == pytest.approx(worst["y"] - worst["yhat"])
    sign = "+" if worst["error"] > 0 else "-"
    row = re.search(r'id="accuracy-misses".*?<tbody><tr>(.*?)</tr>', section.body, re.S)
    assert row is not None
    assert f'<td class="num">{sign}' in row.group(1)
    assert (
        "the forecast ran low" in section.body
        and "the forecast ran high" in section.body
    )


def test_weekday_only_data_closes_the_gaps_on_every_time_axis(plotly: Any) -> None:
    m, df, fcst = fit_case("linear")
    skip = skip_axis(df["ds"], skip_days="weekends")
    weekdays = fcst[skip.keeps(fcst["ds"])]
    bt = weekday_backtest()
    ctx = build_context(
        m, weekdays, df=df, skip=skip, backtest=bt, cv=cv_case("linear")
    )
    with no_warnings():
        section = accuracy.build(ctx)
    smoke(section)
    assert section is not None
    # Only the held-back chart plots observations; the by-cutoff chart plots cutoffs.
    assert section.body.count('"rangebreaks"') == 1
    frame = accuracy.holdout_errors(bt)
    assert accuracy._backtest_figure(ctx, bt, frame, True).layout.xaxis.rangebreaks
    cv = accuracy.prepare_cv(cv_case("linear"), skip)
    by_cutoff = accuracy.cutoff_table(cv, mape=True, coverage=True)
    assert not accuracy._cutoff_figure(ctx, by_cutoff, 1.0).layout.xaxis.rangebreaks
    # A weekend row that slipped into the cross-validation frame is not scored.
    weekend = cv_case("linear").iloc[[0]].assign(ds=pd.Timestamp("2014-05-24"))
    mixed = pd.concat([cv_case("linear"), weekend], ignore_index=True)
    assert len(accuracy.prepare_cv(mixed, skip)) == len(cv_case("linear"))


def test_cutoffs_on_skipped_days_are_still_drawn(plotly: Any) -> None:
    # cross_validation steps cutoffs by calendar time, so on weekday-only data many
    # land on a Saturday or Sunday. Rangebreaks would hide those points, leaving the
    # chart short of the table beside it.
    m, df, fcst = fit_case("linear")
    skip = skip_axis(df["ds"], skip_days="weekends")
    starts = ["2024-01-06", "2024-01-13", "2024-01-22", "2024-01-23", "2024-02-03"]
    frames = [make_cv(1, 5, start=start) for start in starts]
    cv = pd.concat(frames, ignore_index=True)
    cv = cv[skip.keeps(cv["ds"])]
    cutoffs = sorted(cv["cutoff"].unique())
    assert any(pd.Timestamp(c).dayofweek >= 5 for c in cutoffs)
    ctx = build_context(m, fcst[skip.keeps(fcst["ds"])], df=df, skip=skip, cv=cv)
    table = accuracy.cutoff_table(
        accuracy.prepare_cv(cv, skip), mape=True, coverage=True
    )
    fig = accuracy._cutoff_figure(ctx, table, 3.0)
    assert not fig.layout.xaxis.rangebreaks and fig.layout.xaxis.type == "date"
    assert len(fig.data[0].x) == len(table) == len(cutoffs)
    assert_figure_rules(fig)
    section = accuracy.build(ctx)
    smoke(section)
    assert section is not None and "By cutoff" in section.body
    assert section.body.count('"rangebreaks"') == 0


def test_a_tiny_positive_actual_drops_percentages_everywhere(plotly: Any) -> None:
    # performance_metrics skips MAPE below 1e-8; the headline, the tiles, the panels
    # and the by-cutoff table must all agree with it.
    m, df, fcst = fit_case("linear")
    cv = make_cv(y=1e-10, intervals=False)
    with no_warnings():
        section = accuracy.build(build_context(m, fcst, df=df, cv=cv))
    smoke(section)
    assert section is not None
    assert "typical error" not in section.summary
    assert section.summary.startswith("Across 3 cutoffs the average miss ")
    assert "TYPICAL ERROR" not in section.body and not has_header(section.body, "MAPE")


def test_short_history(plotly: Any) -> None:
    m, df, fcst, bt, cv = short_case()
    assert len(df) == 80
    with no_warnings():
        section = accuracy.build(build_context(m, fcst, df=df, backtest=bt, cv=cv))
    smoke(section)
    assert section is not None
    assert "Held-back test" in section.body and "Cross-validation" in section.body
