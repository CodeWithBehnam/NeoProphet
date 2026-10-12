"""Tests for the "trend" section of the dashboard (prophet.dashboard.trend)."""

from __future__ import annotations

import copy
import dataclasses
import re
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet
from prophet.dashboard import _palette, trend
from prophet.plot import skip_axis
from prophet.tests.test_visuals import CASES, fit_case, no_warnings
from prophet.visuals import _fmt_number

DATA = Path(__file__).parent
DAY = trend.Period("day", pd.Timedelta(days=1))
EM_DASH = chr(0x2014)  # written as a code point so that this file has none either
COLOUR = re.compile(r"#[0-9A-Fa-f]{6}|rgba?\([^)]*\)")


def moving_capacity(top: float, ds: pd.Series) -> np.ndarray:
    """A capacity that climbs with the date, from 10% above the highest observation to about twice it.

    It depends on the date alone, so the training rows and the forecast rows agree about it.
    """
    days = (ds - pd.Timestamp("2012-05-18")) / pd.Timedelta(days=1)
    return np.asarray(top * (1.1 + 0.9 * days / 770.0), dtype=float)


@lru_cache(maxsize=None)
def extra_fit(name: str) -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    """Models the shared fixtures lack: logistic without a floor, min-max scaling, no changepoints, a long series."""
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"])
    kwargs: dict[str, Any] = {}
    future_kwargs: dict[str, Any] = {"periods": 30}
    if name == "logistic_no_floor":
        df = df.assign(cap=df["y"].max() * 1.5)
        kwargs = {"growth": "logistic"}
    elif name == "logistic_moving_cap":
        df = df.assign(cap=moving_capacity(df["y"].max(), df["ds"]), floor=0.0)
        kwargs = {"growth": "logistic"}
    elif name == "minmax":
        kwargs = {"scaling": "minmax"}
    elif name == "no_changepoints":
        kwargs = {"n_changepoints": 0}
    elif name == "long":
        n = 2500
        ds = pd.date_range("2020-01-01", periods=n, freq="5min")
        wave = np.sin(np.arange(n) / 300.0) + np.arange(n) / 3000.0
        df = pd.DataFrame({"ds": ds, "y": wave})
        future_kwargs = {"periods": 400, "freq": "5min"}
    m = Prophet(**kwargs).fit(df)
    future = m.make_future_dataframe(**future_kwargs)
    if name == "logistic_no_floor":
        future = future.assign(cap=df["cap"].iloc[0])
    elif name == "logistic_moving_cap":
        future = future.assign(
            cap=moving_capacity(df["y"].max(), future["ds"]), floor=0.0
        )
    return m, df, m.predict(future)


def stamp(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def facts(**changes: Any) -> trend.TrendFacts:
    base: dict[str, Any] = {
        "growth": "linear",
        "start": stamp("2012-05-18"),
        "end": stamp("2014-05-30"),
        "first": 41.1,
        "last": 66.6,
        "scale": 72.0,
        "step": pd.Timedelta(days=1),
        "period": DAY,
        "candidates": 25,
        "significant": 10,
        "biggest": trend.Pace(stamp("2012-09-13"), -0.164, -0.0584),
        "ahead": trend.Ahead(stamp("2014-06-29"), 69.7, 69.4, 70.0, 0.8),
        "capacity": None,
    }
    return trend.TrendFacts(**{**base, **changes})


def days_between(ds: pd.Series) -> np.ndarray:
    return np.asarray((ds - ds.iloc[0]) / pd.Timedelta(days=1), dtype=float)


def scaled_time(m: Any, ds: pd.Series) -> np.ndarray:
    return np.asarray((ds - m.start) / m.t_scale, dtype=float)


# ---- rates ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (pd.Timedelta(minutes=5), "hour"),
        (pd.Timedelta(hours=1), "hour"),
        (pd.Timedelta(hours=12), "hour"),
        (pd.Timedelta(hours=20), "day"),
        (pd.Timedelta(days=1), "day"),
        (pd.Timedelta(days=2), "day"),
        (pd.Timedelta(days=7), "week"),
        (pd.Timedelta(days=30), "month"),
        (pd.Timedelta(days=91), "quarter"),
        (pd.Timedelta(days=365), "year"),
    ],
)
def test_natural_period(step: pd.Timedelta, expected: str) -> None:
    assert trend.natural_period(step).name == expected


def test_rate_factor_converts_scaled_rates_to_y_per_period() -> None:
    # A scaled rate of 1 climbs y_scale over the whole history.
    history = pd.Timedelta(days=100)
    assert trend.rate_factor(50.0, history, DAY) == pytest.approx(0.5)
    week = trend.Period("week", pd.Timedelta(days=7))
    assert trend.rate_factor(50.0, history, week) == pytest.approx(3.5)


def test_changepoint_effects_match_the_numerical_derivative_of_the_fitted_trend() -> (
    None
):
    m, _, fcst = fit_case("linear")
    effects = trend.changepoint_effects(m, DAY)
    assert len(effects) == len(m.changepoints)
    # The conversion from scaled units, written out.
    delta = np.nanmean(m.params["delta"], axis=0)
    per_day = m.y_scale / (m.t_scale / pd.Timedelta(days=1))
    np.testing.assert_allclose(effects["change"], delta * per_day, rtol=1e-12)
    # And the same numbers measured from the trend column of the forecast.
    secant = np.diff(fcst["trend"]) / np.diff(days_between(fcst["ds"]))
    ds = fcst["ds"].to_numpy()
    for row in effects.itertuples():
        i = int(np.flatnonzero(ds == np.datetime64(row.ds))[0])
        assert 0 < i < len(fcst) - 1
        assert row.after == pytest.approx(secant[i], abs=1e-9)
        assert row.before == pytest.approx(secant[i - 1], abs=1e-9)
        assert row.change == pytest.approx(secant[i] - secant[i - 1], abs=1e-9)


def test_linear_slopes_follow_the_trend_column_everywhere() -> None:
    m, _, fcst = fit_case("linear")
    slopes = trend.slopes_per_period(m, scaled_time(m, fcst["ds"]), None, DAY)
    secant = np.diff(fcst["trend"]) / np.diff(days_between(fcst["ds"]))
    np.testing.assert_allclose(slopes[:-1], secant, atol=1e-9)
    # The closed form and the forward difference of the model's own function agree.
    numeric = trend.numeric_slopes(m, scaled_time(m, fcst["ds"]), None, DAY)
    np.testing.assert_allclose(slopes, numeric, atol=1e-7)


def test_slopes_are_quoted_per_hour_for_subdaily_data() -> None:
    m, _, fcst = fit_case("subdaily")
    hour = trend.Period("hour", pd.Timedelta(hours=1))
    slopes = trend.slopes_per_period(m, scaled_time(m, fcst["ds"]), None, hour)
    hours = days_between(fcst["ds"]) * 24
    secant = np.diff(fcst["trend"]) / np.diff(hours)
    np.testing.assert_allclose(slopes[:-1], secant, atol=1e-9)


def test_logistic_slopes_are_measured_from_the_trend_function() -> None:
    m, _, fcst = fit_case("logistic")
    cap = ((fcst["cap"] - fcst["floor"]) / m.y_scale).to_numpy(dtype=float)
    slopes = trend.slopes_per_period(m, scaled_time(m, fcst["ds"]), cap, DAY)
    days = days_between(fcst["ds"])
    secant = np.diff(fcst["trend"]) / np.diff(days)
    tolerance = 0.01 * np.abs(slopes).max()
    np.testing.assert_allclose(slopes[:-1], secant, atol=tolerance)
    # Adding the slope up over time rebuilds the rise of the trend.
    rebuilt = float((slopes[:-1] * np.diff(days)).sum())
    assert rebuilt == pytest.approx(
        fcst["trend"].iloc[-1] - fcst["trend"].iloc[0], rel=0.02
    )
    # Changepoint effects use the capacity of the training rows.
    history = m.history
    capacity = (history["t"].to_numpy(), history["cap_scaled"].to_numpy())
    effects = trend.changepoint_effects(m, DAY, capacity)
    ds = fcst["ds"].to_numpy()
    for row in effects.itertuples():
        i = int(np.flatnonzero(ds == np.datetime64(row.ds))[0])
        assert row.after == pytest.approx(secant[i], abs=tolerance)
        assert row.before == pytest.approx(secant[i - 1], abs=tolerance)
    with pytest.raises(ValueError, match="capacities"):
        trend.changepoint_effects(m, DAY)


def test_capacity_curve_joins_points_and_carries_the_end_segments_on() -> None:
    curve = trend.capacity_curve(
        np.array([0.0, 1.0, 3.0]), np.array([10.0, 20.0, 30.0])
    )
    assert curve is not None
    np.testing.assert_allclose(curve(np.array([0.5, 2.0])), [15.0, 25.0])
    np.testing.assert_allclose(curve(np.array([-1.0, 4.0])), [0.0, 35.0])
    one = trend.capacity_curve(np.array([0.2]), np.array([7.0]))
    assert one is not None and one(np.array([0.0, 5.0])).tolist() == [7.0, 7.0]
    # Unusable points are ignored, and without a usable point there is no curve.
    some = trend.capacity_curve(np.array([0.0, np.nan, 1.0]), np.array([1.0, 5.0, 3.0]))
    assert some is not None and some(np.array([0.5]))[0] == pytest.approx(2.0)
    assert trend.capacity_curve(np.array([np.nan]), np.array([1.0])) is None
    assert trend.capacity_curve(np.empty(0), np.empty(0)) is None


def test_logistic_slopes_follow_a_capacity_that_moves() -> None:
    m, _, fcst = extra_fit("logistic_moving_cap")
    assert fcst["cap"].iloc[-1] > 1.5 * fcst["cap"].iloc[0]  # it climbs a long way
    cap = ((fcst["cap"] - fcst["floor"]) / m.y_scale).to_numpy(dtype=float)
    slopes = trend.slopes_per_period(m, scaled_time(m, fcst["ds"]), cap, DAY)
    secant = np.diff(fcst["trend"]) / np.diff(days_between(fcst["ds"]))
    tolerance = 0.01 * np.abs(slopes).max()
    # The pace is that of the plotted trend line, not that of the curve under a frozen capacity.
    np.testing.assert_allclose(slopes[:-1], secant, atol=tolerance)
    history = m.history
    capacity = (history["t"].to_numpy(), history["cap_scaled"].to_numpy())
    effects = trend.changepoint_effects(m, DAY, capacity)
    ds = fcst["ds"].to_numpy()
    assert len(effects)
    for row in effects.itertuples():
        i = int(np.flatnonzero(ds == np.datetime64(row.ds))[0])
        assert row.after == pytest.approx(secant[i], abs=tolerance)
        assert row.before == pytest.approx(secant[i - 1], abs=tolerance)
    # The sign of the pace, which the wording turns into rising or falling, agrees with the line.
    clear = np.abs(secant) > 5 * tolerance
    assert (np.sign(slopes[:-1])[clear] == np.sign(secant)[clear]).all()


def test_logistic_slopes_need_a_capacity() -> None:
    m, _, fcst = fit_case("logistic")
    t = scaled_time(m, fcst["ds"])
    with pytest.raises(ValueError, match="capacity"):
        trend.slopes_per_period(m, t, None, DAY)
    with pytest.raises(ValueError, match="capacities"):
        trend.changepoint_effects(m, DAY, (np.array([np.nan]), np.array([np.nan])))


def test_flat_growth_has_no_changepoints_and_no_slope() -> None:
    m, _, fcst = fit_case("flat")
    assert trend.changepoint_deltas(m).empty
    assert trend.changepoint_effects(m, DAY).empty
    slopes = trend.slopes_per_period(m, scaled_time(m, fcst["ds"]), None, DAY)
    assert not slopes.any()


def test_changepoint_deltas_average_mcmc_samples_and_follow_time_order() -> None:
    dates = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"])
    samples = np.array([[1.0, 3.0, -2.0], [3.0, 5.0, 0.0], [2.0, 4.0, -1.0]])
    model = SimpleNamespace(
        growth="linear",
        changepoints=pd.Series(dates),
        changepoints_t=np.array([0.2, 0.5, 0.8]),
        params={"delta": samples},
    )
    frame = trend.changepoint_deltas(model)  # type: ignore[arg-type]
    assert list(frame["ds"]) == sorted(dates)
    np.testing.assert_allclose(frame["t"], [0.2, 0.5, 0.8])
    np.testing.assert_allclose(frame["delta"], [2.0, 4.0, -1.0])


@pytest.mark.parametrize(
    "changes",
    [
        {"growth": "flat"},
        {"changepoints": pd.Series(pd.to_datetime([]))},
        {"changepoints": None},
        {"params": {}},
        {"params": {"delta": np.zeros((1, 2))}},
    ],
)
def test_changepoint_deltas_are_empty_when_there_is_nothing_to_pair(
    changes: dict[str, Any],
) -> None:
    base: dict[str, Any] = {
        "growth": "linear",
        "changepoints": pd.Series(
            pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
        ),
        "changepoints_t": np.array([0.2, 0.5, 0.8]),
        "params": {"delta": np.zeros((1, 3))},
    }
    model = SimpleNamespace(**{**base, **changes})
    assert trend.changepoint_deltas(model).empty  # type: ignore[arg-type]


def test_mcmc_shaped_params_give_the_same_section_as_their_mean() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = fit_case("linear")
    spread = np.random.default_rng(1).normal(size=(10, 25))
    spread = np.vstack([spread, -spread])  # mean zero, so averaging gives back the fit
    mcmc = copy.copy(m)
    mcmc.params = {
        "k": np.repeat(m.params["k"], 20, axis=0),
        "m": np.repeat(m.params["m"], 20, axis=0),
        "delta": m.params["delta"] + 0.05 * spread,
    }
    assert trend.changepoint_deltas(mcmc).shape == (25, 3)
    pd.testing.assert_frame_equal(
        trend.changepoint_effects(mcmc, DAY), trend.changepoint_effects(m, DAY)
    )
    with no_warnings():
        a = trend.build(build_context(m, fcst, df=df))
        b = trend.build(build_context(mcmc, fcst, df=df))
    assert a is not None and b is not None
    assert a.summary == b.summary


def test_thin_index_keeps_the_ends_and_the_corners() -> None:
    assert list(trend.thin_index(5, 10)) == [0, 1, 2, 3, 4]
    kept = trend.thin_index(10_000, 100, anchors=[1234, 4321])
    assert kept[0] == 0 and kept[-1] == 9_999
    assert {1234, 4321} <= set(kept)
    assert len(kept) <= 102
    assert (np.diff(kept) > 0).all()
    assert trend.thin_index(10_000, 100, anchors=[-5, 20_000])[-1] == 9_999


def test_bar_width_is_most_of_the_spacing() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=5, freq="10D"))
    assert trend.bar_width_ms(dates, pd.Timedelta(days=1)) == pytest.approx(
        0.7 * 10 * 86_400_000
    )
    one = trend.bar_width_ms(dates.iloc[:1], pd.Timedelta(days=2))
    assert one == pytest.approx(0.7 * 2 * 86_400_000)


# ---- wording ----------------------------------------------------------------


def test_movement_is_flat_percent_or_absolute() -> None:
    flat = trend.movement(100.0, 101.0, 100.0)
    assert flat.direction == "flat"
    assert trend.movement_clause(flat) == "stayed broadly flat"
    rose = trend.movement(40.0, 60.0, 72.0)
    assert (rose.direction, rose.size, rose.relative) == ("rose", "50%", True)
    assert trend.movement_clause(rose) == "rose 50%"
    # A base near zero has no honest percentage.
    small = trend.movement(1.0, 9.0, 72.0)
    assert (small.direction, small.size, small.relative) == ("rose", "8", False)
    assert trend.movement_clause(small) == "rose by 8"
    below = trend.movement(-5.0, -15.0, 72.0)
    assert (below.direction, below.relative) == ("fell", False)
    assert trend.movement_clause(below) == "fell by 10"


@pytest.mark.parametrize(
    ("before", "after", "verb"),
    [
        (0.2, 0.5, "speeds up"),
        (0.5, 0.2, "slows down"),
        (0.0, 0.3, "speeds up"),
        (-0.2, -0.5, "falls faster"),
        (-0.5, -0.2, "falls more slowly"),
        (0.3, -0.1, "turns from rising to falling"),
        (-0.3, 0.1, "turns from falling to rising"),
    ],
)
def test_pace_verb(before: float, after: float, verb: str) -> None:
    assert trend.pace_verb(before, after) == verb


def test_pace_clause_gives_both_paces_and_the_unit() -> None:
    pace = trend.Pace(stamp("2013-02-01"), 0.4, 0.7)
    assert trend.pace_clause(pace, DAY) == "speeds up (from 0.4 to 0.7 per day)"


def test_lede_reads_the_trend_and_leaves_the_numbers_to_the_tiles() -> None:
    text = trend.lede_text(facts())
    assert text == (
        "Over the history the trend rose, bending at several of the dates where it could. "
        "By the end of the forecast the trend is about 4.7% higher than where the history ends."
    )
    # Nothing the tiles under the lede already carry is written out twice.
    tiles = " ".join(f"{t.value} {t.detail}" for t in trend.trend_tiles(facts()))
    for number in (
        "41.1",
        "66.6",
        "13 Sep 2012",
        "-0.164",
        "69.7",
        "10 of 25",
        "29 Jun 2014",
    ):
        assert number in tiles and number not in text
    assert "logistic" not in text and "noticeabl" not in text


def test_lede_variants() -> None:
    assert "stayed broadly flat" in trend.lede_text(
        facts(first=66.0, last=66.5, significant=0, biggest=None)
    )
    assert "ended close to where it started" in trend.lede_text(
        facts(first=66.0, last=66.5)
    )
    assert "Over the history the trend fell," in trend.lede_text(
        facts(first=66.6, last=41.1)
    )
    assert "at close to one pace throughout" in trend.lede_text(
        facts(significant=0, biggest=None)
    )
    assert "allowed no changepoints" in trend.lede_text(
        facts(candidates=0, significant=0, biggest=None)
    )
    no_future = trend.lede_text(facts(ahead=None))
    assert "end of the forecast" not in no_future
    steady = trend.CapacityFacts(108.0, 108.0, 108.0, stamp("2014-06-29"))
    logistic = trend.lede_text(facts(growth="logistic", capacity=steady))
    assert "nears the capacity of 108." in logistic


@pytest.mark.parametrize(
    ("candidates", "significant", "expected"),
    [
        (1, 1, ", bending at the one date where it could"),
        (25, 20, ", bending at many of the dates where it could"),
        (25, 10, ", bending at several of the dates where it could"),
        (25, 3, ", bending at only a few of the dates where it could"),
    ],
)
def test_how_often_the_trend_bends_is_worded_without_counting(
    candidates: int, significant: int, expected: str
) -> None:
    clause = trend.bending_clause(facts(candidates=candidates, significant=significant))
    assert clause == expected
    assert not any(ch.isdigit() for ch in clause)
    assert "the 1 " not in clause and "1 dates" not in clause


def test_a_slight_change_of_pace_is_not_called_a_bend() -> None:
    slight = trend.Pace(stamp("2021-02-08"), 0.00204, 0.00199)
    assert trend.is_slight(slight)
    assert not trend.is_slight(trend.Pace(stamp("2021-02-08"), 0.4, 0.7))
    assert not trend.is_slight(trend.Pace(stamp("2021-02-08"), 0.0, 0.3))
    clause = trend.bending_clause(facts(biggest=slight))
    assert (
        clause
        == ", at close to one pace throughout, as no change of pace is large enough to matter"
    )
    assert "noticeabl" not in trend.lede_text(facts(biggest=slight))


def test_the_biggest_change_is_a_tile_that_says_what_the_trend_does() -> None:
    tile = {
        t.label: t
        for t in trend.trend_tiles(
            facts(biggest=trend.Pace(stamp("2013-02-01"), 0.4, 0.7))
        )
    }["BIGGEST CHANGE"]
    assert tile.value == "1 Feb 2013"
    assert tile.detail == "speeds up, from 0.4 to 0.7 per day"
    turn = {
        t.label: t
        for t in trend.trend_tiles(
            facts(biggest=trend.Pace(stamp("2013-02-01"), 0.3, -0.1))
        )
    }["BIGGEST CHANGE"]
    assert turn.detail == "turns from rising to falling, from 0.3 to -0.1 per day"


def test_a_range_that_reads_the_same_at_both_ends_is_left_out_of_the_tile() -> None:
    same = trend.Ahead(stamp("2022-08-29"), 105.1, 104.9, 105.3, 0.8)  # 105 to 105
    tile = {t.label: t for t in trend.trend_tiles(facts(ahead=same))}["HEADING FOR"]
    assert tile.detail == "by 29 Aug 2022"
    wider = trend.Ahead(stamp("2022-08-29"), 105.1, 98.0, 112.0, 0.8)
    tile = {t.label: t for t in trend.trend_tiles(facts(ahead=wider))}["HEADING FOR"]
    assert "running from 98 to 112" in tile.detail


def test_an_interval_needs_real_width_against_the_scale_of_the_data() -> None:
    zeros: Any = SimpleNamespace(history=pd.DataFrame({"y": [0.0, 0.0, 0.0]}))
    assert not trend.has_width(zeros, 5.5e-9, 0.0)  # numerical noise around zero
    assert not trend.has_width(zeros, float("nan"), 0.0)
    assert trend.has_width(zeros, 0.01, 0.0)
    big: Any = SimpleNamespace(history=pd.DataFrame({"y": [1e6, 2e6]}))
    assert not trend.has_width(big, 0.5, 1.5e6)
    assert trend.has_width(big, 50.0, 1.5e6)


def test_a_moving_capacity_is_described_as_a_range() -> None:
    moving = trend.CapacityFacts(144.0, 100.0, 144.0, stamp("2014-06-29"))
    text = trend.capacity_sentence(facts(growth="logistic", capacity=moving))
    assert text == (
        "Growth is logistic, so the trend flattens as it nears the capacity, which "
        "moves between 100 and 144 and stands at 144 by 29 Jun 2014."
    )
    still = trend.CapacityFacts(144.0, 144.0, 144.0, stamp("2014-06-29"))
    assert trend.capacity_sentence(facts(growth="logistic", capacity=still)) == (
        "Growth is logistic, so the trend flattens as it nears the capacity of 144."
    )
    assert trend.capacity_sentence(facts(capacity=moving)) == ""


def test_flat_growth_is_described_as_held_flat() -> None:
    flat = facts(
        growth="flat",
        first=37.6,
        last=37.6,
        candidates=0,
        significant=0,
        biggest=None,
        ahead=None,
    )
    text = trend.lede_text(flat)
    assert text.startswith("The model holds the trend flat, so everything that moves")
    assert "37.6" not in text  # the tile has the level
    assert trend.summary_text(flat) == "The trend is held flat at 37.6."
    assert [t.label for t in trend.trend_tiles(flat)] == ["TREND LEVEL"]


def test_summary_is_one_plain_sentence() -> None:
    text = trend.summary_text(facts())
    assert text == (
        "The trend rose 62% between May 2012 and May 2014, changed pace at 10 dates "
        "and is heading for 69.7 by 29 Jun 2014."
    )
    assert trend.summary_text(facts(significant=1, ahead=None)).endswith(
        "and changed pace once."
    )
    assert trend.summary_text(facts(significant=0, ahead=None)).endswith("May 2014.")
    for variant in (
        facts(),
        facts(significant=0),
        facts(ahead=None),
        facts(first=70.0, last=40.0),
    ):
        summary = trend.summary_text(variant)
        assert EM_DASH not in summary and "\n" not in summary
        assert summary.count(". ") == 0


def test_tiles_carry_the_key_numbers() -> None:
    tiles = {t.label: t for t in trend.trend_tiles(facts())}
    assert tiles["OVERALL CHANGE"].value == "+62%"
    assert tiles["CHANGES OF PACE"].value == "10 of 25"
    assert tiles["BIGGEST CHANGE"].value == "13 Sep 2012"
    assert tiles["HEADING FOR"].value == "69.7"
    assert "69.4 to 70" in tiles["HEADING FOR"].detail
    assert tiles["CHANGES OF PACE"].detail == "candidate dates where the pace shifts"
    fall = {t.label: t for t in trend.trend_tiles(facts(first=70.0, last=40.0))}
    assert fall["OVERALL CHANGE"].value == "-43%"
    quiet = trend.trend_tiles(
        facts(candidates=0, significant=0, biggest=None, ahead=None)
    )
    assert [t.label for t in quiet] == ["OVERALL CHANGE"]


def test_sentences_use_the_time_resolution_of_the_data() -> None:
    hourly = facts(
        start=stamp("2017-01-01 00:05"),
        end=stamp("2017-01-03 23:55"),
        step=pd.Timedelta(minutes=5),
        period=trend.Period("hour", pd.Timedelta(hours=1)),
        biggest=trend.Pace(stamp("2017-01-01 13:50"), -0.115, 0.0913),
        ahead=trend.Ahead(stamp("2017-01-04 03:55"), 5.0, None, None, 0.8),
    )
    tiles = {t.label: t for t in trend.trend_tiles(hourly)}
    assert tiles["BIGGEST CHANGE"].value == "1 Jan 2017 13:50"
    assert "per hour" in tiles["BIGGEST CHANGE"].detail
    assert tiles["HEADING FOR"].detail == "by 4 Jan 2017 03:55"
    assert trend.summary_text(hourly).endswith("is heading for 5 by 4 Jan 2017 03:55.")
    assert "between 1 Jan 2017 00:05 and 3 Jan 2017 23:55" in trend.summary_text(hourly)


# ---- year over year -----------------------------------------------------------


def daily_frame(start: str, end: str, levels: dict[int, float]) -> pd.DataFrame:
    ds = pd.date_range(start, end)
    return pd.DataFrame({"ds": ds, "y": [levels[d.year] for d in ds]})


def test_daily_means_drop_thin_days_and_average_the_rest() -> None:
    ds = pd.Series(
        pd.to_datetime(["2020-01-01 00:00", "2020-01-01 12:00", "2020-01-02 00:00"])
    )
    values = pd.Series([1.0, 3.0, 10.0])
    out = trend.daily_means(ds, values, min_count=2)
    assert out.to_dict() == {pd.Timestamp("2020-01-01"): 2.0}
    assert len(trend.daily_means(ds, values, min_count=1)) == 2


def test_observed_daily_ignores_the_partial_days_of_subdaily_data() -> None:
    ds = pd.date_range("2020-01-01 18:00", "2020-01-10 05:00", freq="h")
    frame = pd.DataFrame({"ds": ds, "y": np.arange(len(ds), dtype=float)})
    daily, usual = trend.observed_daily(frame)
    assert usual == 24.0
    assert list(daily.index) == list(pd.date_range("2020-01-02", "2020-01-09"))
    assert daily.iloc[0] == pytest.approx(frame["y"].iloc[6:30].mean())


def test_observed_daily_without_observations() -> None:
    frame = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=3), "y": np.nan})
    daily, usual = trend.observed_daily(frame)
    assert daily.empty and usual == 1.0


def test_smoothing_applies_to_daily_data_only() -> None:
    series = pd.Series(
        np.arange(30, dtype=float), index=pd.date_range("2020-01-01", periods=30)
    )
    smoothed, applied = trend.smooth(series, pd.Timedelta(days=1))
    assert applied and smoothed.iloc[10] == pytest.approx(10.0)
    same, applied = trend.smooth(series, pd.Timedelta(days=7))
    assert not applied and same is series


def test_usable_years_need_three_days_over_two_weeks() -> None:
    days = pd.DatetimeIndex(
        [
            "2011-12-30",
            "2011-12-31",
            "2012-05-18",
            "2012-05-19",
            "2012-06-30",
            "2013-01-01",
            "2013-01-02",
            "2013-01-03",
        ]
    )
    daily = pd.Series(1.0, index=days)
    assert trend.usable_years(daily) == [2012]


def test_reference_dates_put_every_year_on_one_leap_year() -> None:
    days = pd.DatetimeIndex(["2013-03-01", "2016-02-29", "2012-12-31"])
    out = trend.reference_dates(days)
    assert list(out) == [stamp("2000-03-01"), stamp("2000-02-29"), stamp("2000-12-31")]


def test_like_for_like_compares_only_the_days_both_years_cover() -> None:
    before = daily_frame("2012-05-18", "2012-12-31", {2012: 10.0})
    after = daily_frame("2013-01-01", "2013-12-31", {2013: 5.0})
    after.loc[after["ds"] >= "2013-05-18", "y"] = 15.0
    assert trend.like_for_like(before, after) == pytest.approx(0.5)
    # No shared days, or a base of zero: no percentage.
    early = daily_frame("2012-01-01", "2012-03-31", {2012: 10.0})
    late = daily_frame("2013-06-01", "2013-12-31", {2013: 10.0})
    assert trend.like_for_like(early, late) is None
    zero = before.assign(y=0.0)
    assert trend.like_for_like(zero, after) is None


def test_year_stats_label_partial_years_and_compare_like_for_like() -> None:
    frame = daily_frame(
        "2012-05-18", "2014-05-30", {2012: 10.0, 2013: 20.0, 2014: 30.0}
    )
    stats = trend.year_stats(frame, pd.Timedelta(days=1), [2012, 2013, 2014])
    assert [trend.year_label(s) for s in stats] == [
        "2012 (from 18 May)",
        "2013",
        "2014 (to 30 May)",
    ]
    assert [s.change for s in stats] == [None, pytest.approx(1.0), pytest.approx(0.5)]
    assert stats[1].count == 365
    assert stats[1].mean == pytest.approx(20.0)
    assert stats[1].total == pytest.approx(7300.0)
    rows = trend.year_rows(stats)
    assert [r[0] for r in rows] == ["2014 (to 30 May)", "2013", "2012 (from 18 May)"]
    assert rows[1] == ["2013", "365", "20", "7,300", "+100%"]
    assert rows[2][-1] == "n/a"
    sentence = trend.year_sentence(stats)
    assert sentence.startswith("2014 (to 30 May) averaged 30, 50% above 2013")
    assert EM_DASH not in sentence


def test_a_year_without_its_predecessor_has_no_change() -> None:
    frame = daily_frame(
        "2012-01-01", "2014-12-31", {2012: 10.0, 2013: 99.0, 2014: 30.0}
    )
    frame = frame[frame["ds"].dt.year != 2013]
    stats = trend.year_stats(frame, pd.Timedelta(days=1), [2012, 2014])
    assert [s.change for s in stats] == [None, None]
    assert trend.year_sentence(stats) == ""
    assert [trend.year_label(s) for s in stats] == ["2012", "2014"]


def test_year_lines_join_the_forecast_to_the_last_observation() -> None:
    observed = pd.Series(1.0, index=pd.date_range("2012-05-18", "2014-05-30"))
    forecast = pd.Series(2.0, index=pd.date_range("2014-05-31", periods=30))
    lines = trend.year_lines(observed, forecast, [2012, 2013, 2014])
    assert [ln.year for ln in lines] == [2012, 2013, 2014]
    latest = lines[-1]
    assert latest.forecast.index[0] == stamp("2014-05-30")  # the joining point
    assert latest.forecast.iloc[0] == 1.0 and latest.forecast.iloc[1] == 2.0
    assert len(latest.forecast) == 31
    assert lines[0].forecast.empty and lines[1].forecast.empty


def test_year_lines_draw_a_forecast_year_only_when_it_is_mostly_forecast() -> None:
    observed = pd.Series(1.0, index=pd.date_range("2012-05-18", "2014-12-14"))
    short = pd.Series(
        2.0, index=pd.date_range("2014-12-15", periods=40)
    )  # 23 days of 2015
    assert [
        ln.year for ln in trend.year_lines(observed, short, [2012, 2013, 2014])
    ] == [2012, 2013, 2014]
    long = pd.Series(
        2.0, index=pd.date_range("2014-12-15", periods=60)
    )  # 43 days of 2015
    lines = trend.year_lines(observed, long, [2012, 2013, 2014])
    assert [ln.year for ln in lines] == [2012, 2013, 2014, 2015]
    assert lines[-1].observed.empty and len(lines[-1].forecast) == 43


def test_a_forecast_year_needs_days_of_forecast_however_coarse_the_data() -> None:
    weekly_seen = pd.Series(
        1.0, index=pd.date_range("2020-01-01", "2022-12-25", freq="7D")
    )
    weekly = pd.Series(2.0, index=pd.date_range("2023-01-01", periods=26, freq="7D"))
    drawn = [
        ln.year for ln in trend.year_lines(weekly_seen, weekly, [2020, 2021, 2022])
    ]
    assert drawn == [2020, 2021, 2022, 2023]  # 26 points, but 175 days
    two_weeks = weekly.iloc[:3]
    drawn = [
        ln.year for ln in trend.year_lines(weekly_seen, two_weeks, [2020, 2021, 2022])
    ]
    assert drawn == [2020, 2021, 2022]
    daily_seen = pd.Series(1.0, index=pd.date_range("2020-01-01", "2022-12-31"))
    for days, expected in ((27, 3), (28, 4)):
        ahead = pd.Series(2.0, index=pd.date_range("2023-01-01", periods=days))
        lines = trend.year_lines(daily_seen, ahead, [2020, 2021, 2022])
        assert len(lines) == expected
    assert trend.year_lines(daily_seen, trend.no_days(), [2022])[-1].year == 2022


def test_year_lines_cap_the_number_of_years() -> None:
    observed = pd.Series(1.0, index=pd.date_range("2000-01-01", "2020-12-31"))
    lines = trend.year_lines(
        observed, trend.no_days(), list(range(2000, 2021)), max_years=5
    )
    assert [ln.year for ln in lines] == [2016, 2017, 2018, 2019, 2020]


def test_minmax_scaling_keeps_the_slopes_and_adds_its_floor_to_the_levels() -> None:
    m, _, fcst = extra_fit("minmax")
    assert m.scaling == "minmax" and m.y_min > 0
    slopes = trend.slopes_per_period(m, scaled_time(m, fcst["ds"]), None, DAY)
    secant = np.diff(fcst["trend"]) / np.diff(days_between(fcst["ds"]))
    np.testing.assert_allclose(slopes[:-1], secant, atol=1e-9)
    effects = trend.changepoint_effects(m, DAY)
    ds = fcst["ds"].to_numpy()
    for row in effects.itertuples():
        i = int(np.flatnonzero(ds == np.datetime64(row.ds))[0])
        assert row.change == pytest.approx(secant[i] - secant[i - 1], abs=1e-9)


def test_a_model_without_changepoints_has_one_steady_pace() -> None:
    m, _, fcst = extra_fit("no_changepoints")
    assert trend.changepoint_deltas(m).empty
    assert trend.changepoint_effects(m, DAY).empty
    slopes = trend.slopes_per_period(m, scaled_time(m, fcst["ds"]), None, DAY)
    assert np.ptp(slopes) == pytest.approx(0.0, abs=1e-12)
    secant = np.diff(fcst["trend"]) / np.diff(days_between(fcst["ds"]))
    np.testing.assert_allclose(slopes[:-1], secant, atol=1e-9)


# ---- the section --------------------------------------------------------------


def record(ctx: Any) -> tuple[Any, list[Any]]:
    """Build the section with an embed that keeps the figures instead of rendering them."""
    figures: list[Any] = []

    def embed(fig: Any) -> str:
        figures.append(fig)
        return f'<div class="plotly-graph-div">figure {len(figures)}</div>'

    section = trend.build(dataclasses.replace(ctx, embed=embed))
    return section, figures


def case_context(name: str, **changes: Any) -> Any:
    from prophet.dashboard._context import build_context

    m, df, fcst = fit_case(name)
    return build_context(m, fcst, df=df, **changes)


def colours(node: Any) -> set[str]:
    """Every colour string in a figure, written the way DARK_SWAP keys are, leaving out the template."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key != "template":
                found |= colours(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            found |= colours(value)
    elif isinstance(node, str) and COLOUR.fullmatch(node):
        found.add(node.upper().replace(" ", ""))
    return found


def check_figures(figures: list[Any]) -> None:
    for fig in figures:
        assert fig.data, "an empty figure"
        for trace in fig.data:
            # Only helper traces kept out of the legend (the lower edge of a band) may skip the hover.
            assert trace.hovertemplate or (
                trace.hoverinfo == "skip" and trace.showlegend is False
            ), trace.name
        assert colours(fig.to_plotly_json()) <= set(_palette.DARK_SWAP)


@pytest.mark.parametrize("case", [*CASES, "temperature"])
def test_section_across_configurations(case: str) -> None:
    pytest.importorskip("plotly")
    ctx = case_context(case)
    with no_warnings():
        section, figures = record(ctx)
    assert section is not None
    assert (section.key, section.title) == ("trend", "Trend")
    assert section.summary.endswith(".") and EM_DASH not in section.summary
    assert section.summary.startswith("The trend")
    assert EM_DASH not in section.body and "<script" not in section.body
    assert 'class="lede"' in section.body and 'class="tiles"' in section.body
    check_figures(figures)
    number = r"-?\d[\d,]*(?:\.\d+)?(?:e[+-]\d+)?[KMB]?"
    for low, high in re.findall(
        rf"running from ({number}) to ({number})", section.body
    ):
        assert low != high
    m = ctx.m
    if case == "flat":
        assert len(figures) == 2  # the trend and the years
        assert "How much the pace changes" not in section.body
        assert "Growth rate over time" not in section.body
    elif case == "subdaily":
        assert len(figures) == 3  # three days of data cannot be compared by year
        assert "Year over year" not in section.body
        assert "per hour" in section.body
    else:
        assert len(figures) == 4
        assert "Year over year" in section.body
        assert "per day" in section.body
    # The rates chart gets a second panel when something multiplies the trend and the
    # trend stays above zero (the "regressors" fixture dips below it).
    rates = [f for f in figures if f.data and f.data[0].name == "Change per day"]
    if case in ("multiplicative", "temperature"):
        assert rates and "yaxis2" in rates[0].layout.to_plotly_json()
    elif rates:
        assert "yaxis2" not in rates[0].layout.to_plotly_json()
    if m.growth == "logistic":
        names = [t.name for t in figures[0].data]
        assert "Maximum capacity" in names and "Growth is logistic" in section.body


def test_no_share_of_the_trend_when_the_trend_is_not_positive() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("regressors")
    assert (ctx.fcst["trend"] <= 0).any() and ctx.fcst[
        "multiplicative_terms"
    ].abs().max() > 0
    _, figures = record(ctx)
    assert "yaxis2" not in figures[2].layout.to_plotly_json()


@pytest.mark.parametrize("case", ["linear", "logistic", "multiplicative", "regressors"])
def test_embedded_data_stays_small(case: str) -> None:
    pytest.importorskip("plotly")
    section = trend.build(case_context(case))
    assert section is not None
    assert len(section.body) < 1_000_000


def test_numbers_in_the_lede_come_from_the_fitted_trend() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("linear")
    fitted = np.asarray(ctx.m.predict_trend(ctx.history), dtype=float)
    last_row = ctx.future.iloc[-1]
    section, _ = record(ctx)
    body = section.body
    assert f"{_fmt_number(fitted[0])} to {_fmt_number(fitted[-1])} since" in body
    assert "HEADING FOR" in body
    assert _fmt_number(last_row["trend"]) in body
    assert (
        f"{_fmt_number(last_row['trend_lower'])} to {_fmt_number(last_row['trend_upper'])}"
        in body
    )
    effects = trend.changepoint_effects(ctx.m, DAY)
    assert f"{(effects['significant']).sum()} of {len(effects)}" in body


def test_the_flat_section_does_not_repeat_its_own_title() -> None:
    pytest.importorskip("plotly")
    flat, _ = record(case_context("flat"))
    assert flat.title == "Trend"
    assert "<h3>Trend level against the data</h3>" in flat.body
    assert "<h3>Trend</h3>" not in flat.body and "<h3>The trend</h3>" not in flat.body
    linear, _ = record(case_context("linear"))
    assert "<h3>Trend and changepoints</h3>" in linear.body


def test_the_changepoint_note_names_the_unit_the_lede_and_tiles_use() -> None:
    pytest.importorskip("plotly")
    daily, _ = record(case_context("linear"))
    assert "in the units of the data per day." in daily.body
    assert "per period" not in daily.body
    hourly, _ = record(case_context("subdaily"))
    assert "in the units of the data per hour." in hourly.body
    assert "per hour" in hourly.body


def test_flat_growth_draws_no_band_around_a_trend_with_no_uncertainty() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("flat")
    assert ctx.show_band  # intervals are on, but the flat trend has none to show
    section, figures = record(ctx)
    assert [t.name for t in figures[0].data] == ["Observed", "Trend"]
    assert "shaded band" not in section.body


def test_logistic_without_a_floor_column() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = extra_fit("logistic_no_floor")
    assert not m.logistic_floor and "floor" not in fcst
    ctx = build_context(m, fcst, df=df)
    rows = trend._forecast_rows(ctx)
    n = len(ctx.history)
    np.testing.assert_allclose(
        rows["cap_scaled"].iloc[:n], ctx.history["cap_scaled"], rtol=1e-12
    )
    with no_warnings():
        section, figures = record(ctx)
    assert section is not None and len(figures) == 4
    names = [t.name for t in figures[0].data]
    assert "Maximum capacity" in names and "Minimum capacity" not in names
    check_figures(figures)


def test_a_moving_capacity_is_drawn_and_described() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = extra_fit("logistic_moving_cap")
    with no_warnings():
        section, figures = record(build_context(m, fcst, df=df))
    assert section is not None and len(figures) == 4
    assert "which moves between" in section.body and "stands at" in section.body
    cap = next(t for t in figures[0].data if t.name == "Maximum capacity")
    assert np.ptp(np.asarray(cap.y, dtype=float)) > 0
    check_figures(figures)


def test_minmax_scaling_builds() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = extra_fit("minmax")
    ctx = build_context(m, fcst, df=df)
    with no_warnings():
        section, figures = record(ctx)
    assert section is not None
    fitted = np.asarray(m.predict_trend(ctx.history), dtype=float)
    assert (
        f"{_fmt_number(fitted[0])} to {_fmt_number(fitted[-1])} since" in section.body
    )
    check_figures(figures)


def test_a_model_without_changepoints_says_so() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = extra_fit("no_changepoints")
    with no_warnings():
        section, figures = record(build_context(m, fcst, df=df))
    assert section is not None
    assert "allowed no changepoints" in section.body
    assert "How much the pace changes" not in section.body
    assert "Growth rate over time" in section.body
    assert "Change of pace" not in [t.name for t in figures[0].data]
    check_figures(figures)


def test_long_series_are_thinned_before_they_are_embedded() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = extra_fit("long")
    assert len(df) > trend.MAX_OBSERVED and len(fcst) > trend.MAX_POINTS
    ctx = build_context(m, fcst, df=df)
    with no_warnings():
        section, figures = record(ctx)
    assert section is not None
    check_figures(figures)
    by_name = {t.name: t for t in figures[0].data}
    assert len(by_name["Observed"].x) <= trend.MAX_OBSERVED
    assert len(by_name["Trend"].x) <= trend.MAX_POINTS + len(m.changepoints)
    rates = figures[2].data[0]
    assert len(rates.x) <= trend.MAX_POINTS + len(m.changepoints)
    # The thinned trend still passes through every changepoint it marks.
    marks = by_name["Change of pace"]
    thinned = set(pd.to_datetime(by_name["Trend"].x))
    assert set(pd.to_datetime(marks.x)) <= thinned
    assert len(section.body) < 1_000_000
    assert "per hour" in section.body


def test_trend_chart_marks_the_significant_changepoints_with_words() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("linear")
    _, figures = record(ctx)
    fig = figures[0]
    names = [t.name for t in fig.data]
    assert names[0] == "Observed" and names.count("Trend") == 1
    marks = next(t for t in fig.data if t.name == "Change of pace")
    effects = trend.changepoint_effects(ctx.m, DAY)
    assert len(marks.x) == int(effects["significant"].sum())
    verbs = set(marks.customdata[:, 0])
    assert verbs <= {
        "Speeds up",
        "Slows down",
        "Falls faster",
        "Falls more slowly",
        "Turns from rising to falling",
        "Turns from falling to rising",
    }
    assert all(" per day" in text for text in marks.customdata[:, 1])
    # The trend says its pace in the hover, in y units per day, as the rates chart does.
    line = next(t for t in fig.data if t.name == "Trend")
    assert "per day" in line.hovertemplate
    # The observed dots say what they are; the nearest point wins the hover, so the line still can.
    dots = next(t for t in fig.data if t.name == "Observed")
    assert dots.hovertemplate and "Observed" in dots.hovertemplate
    slopes = trend.slopes_per_period(
        ctx.m, scaled_time(ctx.m, ctx.fcst["ds"]), None, DAY
    )
    np.testing.assert_allclose(np.asarray(line.customdata, dtype=float), slopes)
    assert names.index("Trend") < names.index("Change of pace")
    # One dotted line per mark, the history and forecast tints, and the divider.
    assert len(fig.layout.shapes) == len(marks.x) + 3
    band = [t for t in fig.data if t.name and "range of the trend" in t.name]
    assert len(band) == 1 and band[0].fillcolor == _palette.BAND


def test_a_lone_changepoint_gets_a_bar_you_can_see() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("linear")
    effects = trend.changepoint_effects(ctx.m, DAY).iloc[:1]
    fig = trend._effects_figure(ctx, effects, DAY)
    span = (ctx.fcst["ds"].max() - ctx.fcst["ds"].min()).total_seconds() * 1000
    assert fig.data[0].width >= 0.01 * span  # a day would be a tenth of a percent


def test_changepoints_on_skipped_days_are_left_out_of_every_chart() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = fit_case("linear")
    weekday = int(m.changepoints.dt.weekday.mode().iloc[0])
    axis = skip_axis(df["ds"], skip_days=[weekday])
    on_skipped = int((~np.asarray(axis.keeps(m.changepoints), dtype=bool)).sum())
    assert on_skipped >= 2
    with no_warnings():
        section, figures = record(
            build_context(
                m, fcst[axis.keeps(fcst["ds"])], df=df[axis.keeps(df["ds"])], skip=axis
            )
        )
    assert section is not None
    bars = [d for t in figures[1].data for d in pd.to_datetime(t.x)]
    assert len(bars) == len(m.changepoints) - on_skipped
    assert np.asarray(axis.keeps(pd.Series(bars)), dtype=bool).all()
    marks = next(t for t in figures[0].data if t.name == "Change of pace")
    assert np.asarray(axis.keeps(pd.Series(pd.to_datetime(marks.x))), dtype=bool).all()
    assert (
        f"{on_skipped} changepoints fall on days the dashboard leaves out"
        in section.body
    )


def test_without_skipped_days_no_changepoint_is_left_out() -> None:
    pytest.importorskip("plotly")
    section, _ = record(case_context("linear"))
    assert "leaves out" not in section.body
    assert trend.skipped_note(0) == ""
    assert trend.skipped_note(1).startswith("One changepoint falls on a day")


def test_effects_chart_has_a_bar_per_changepoint_in_the_two_poles() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("linear")
    _, figures = record(ctx)
    fig = figures[1]
    effects = trend.changepoint_effects(ctx.m, DAY)
    assert sum(len(t.x) for t in fig.data) == len(effects)
    by_name = {t.name: t for t in fig.data}
    assert by_name["Pace rises"].marker.color == _palette.DIVERGING[-1][1]
    assert by_name["Pace falls"].marker.color == _palette.DIVERGING[0][1]
    assert (np.asarray(by_name["Pace rises"].y) >= 0).all()
    assert (np.asarray(by_name["Pace falls"].y) < 0).all()
    faint = [o for t in fig.data for o in t.marker.opacity if o < 1]
    assert len(faint) == int((~effects["significant"]).sum())


def test_rates_chart_is_flat_between_changepoints() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("linear")
    _, figures = record(ctx)
    rates = figures[2]
    trace = rates.data[0]
    assert trace.line.shape == "hv" and trace.name == "Change per day"
    # Every step of the line is a changepoint: the values change at most at the candidate dates.
    values = np.asarray(trace.y, dtype=float)
    assert len(np.unique(values)) <= len(ctx.m.changepoints) + 1


def test_rates_subplot_titles_fit_a_phone_width_figure() -> None:
    pytest.importorskip("plotly")
    _, figures = record(case_context("multiplicative"))
    titles = [a.text for a in figures[2].layout.annotations]
    assert len(titles) == 2  # the share of the trend is drawn for multiplicative fits
    # Plotly never wraps a subplot title; at 15px serif, 40 characters fit a 360px figure.
    assert max(len(t) for t in titles) <= 40


def test_years_chart_colours_the_latest_year_and_dashes_the_forecast() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("linear")
    _, figures = record(ctx)
    traces = {t.name: t for t in figures[3].data}
    assert set(traces) == {"2014", "2014 forecast", "2013", "Earlier years"}
    assert traces["2014"].line.color == _palette.SERIES[0]
    assert traces["2014 forecast"].line.dash == "dash"
    assert traces["2013"].line.color != traces["2014"].line.color
    assert traces["Earlier years"].opacity < 1
    axis = figures[3].layout.xaxis
    assert axis.tickformat == "%b" and not axis.rangebreaks


def legend_names(fig: Any) -> list[str]:
    return [t.name for t in fig.data if t.showlegend]


def test_years_chart_draws_the_forecast_that_crosses_a_year_end() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("linear")
    observed = pd.Series(1.0, index=pd.date_range("2012-05-18", "2013-12-15"))
    forecast = pd.Series(2.0, index=pd.date_range("2013-12-16", periods=60))
    lines = trend.year_lines(observed, forecast, trend.usable_years(observed))
    assert [ln.year for ln in lines] == [2012, 2013, 2014]
    fig = trend._year_figure(ctx, lines, "Value")
    by_name = {t.name: t for t in fig.data}
    assert set(by_name) == {"Earlier years", "2013", "2013 forecast", "2014 forecast"}
    # The last days of 2013 are drawn, dashed, from the last observed point to the year end.
    tail = by_name["2013 forecast"]
    assert tail.line.dash == "dash" and tail.line.color == _palette.INK_SOFT
    days = pd.to_datetime(tail.x)
    assert (days.min(), days.max()) == (stamp("2000-12-15"), stamp("2000-12-31"))
    # The new year starts its own dashed line on 1 January, with no line back across the chart.
    head = by_name["2014 forecast"]
    days = pd.to_datetime(head.x)
    assert (days.min(), days.max()) == (stamp("2000-01-01"), stamp("2000-02-13"))
    assert list(head.x) == sorted(head.x)
    # Every forecast day is on the chart, and the 2013 line is one legend entry.
    shown = {d.strftime("%m-%d") for t in fig.data for d in pd.to_datetime(t.x)}
    assert {d.strftime("%m-%d") for d in forecast.index} <= shown
    assert legend_names(fig) == ["Earlier years", "2013", "2014 forecast"]
    assert by_name["2013 forecast"].legendgroup == by_name["2013"].legendgroup


def test_years_chart_draws_the_forecast_of_older_and_forecast_only_years() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("linear")
    observed = pd.Series(1.0, index=pd.date_range("2012-05-18", "2013-06-30"))
    forecast = pd.Series(2.0, index=pd.date_range("2013-07-01", "2015-02-10"))
    lines = trend.year_lines(observed, forecast, trend.usable_years(observed))
    assert [ln.year for ln in lines] == [2012, 2013, 2014, 2015]
    fig = trend._year_figure(ctx, lines, "Value")
    # 2014 is only forecast, so its dashes carry its legend entry.
    assert legend_names(fig) == ["Earlier years", "2014 forecast", "2015 forecast"]
    earlier = [t for t in fig.data if t.legendgroup == "earlier"]
    assert len(earlier) == 3  # 2012, 2013 and the forecast part of 2013
    assert sum(1 for t in earlier if t.line.dash == "dash") == 1
    assert all(t.opacity < 1 for t in earlier)


def test_no_uncertainty_draws_no_band_and_quotes_no_range() -> None:
    pytest.importorskip("plotly")
    section, figures = record(case_context("no_uncertainty"))
    assert not [t for t in figures[0].data if t.name and "range of the trend" in t.name]
    assert "range running" not in section.body


def test_uncertainty_can_be_switched_off() -> None:
    pytest.importorskip("plotly")
    section, figures = record(case_context("linear", uncertainty=False))
    assert not [t for t in figures[0].data if t.name and "range of the trend" in t.name]
    assert "range running" not in section.body


def test_forecast_with_only_future_rows() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = fit_case("linear")
    future = fcst[fcst["ds"] > m.history["ds"].max()]
    with no_warnings():
        section, figures = record(build_context(m, future, df=df))
    assert section is not None
    check_figures(figures)
    assert "HEADING FOR" in section.body and "since May 2012" in section.body


def test_forecast_with_only_in_sample_rows() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = fit_case("linear")
    past = fcst[fcst["ds"] <= m.history["ds"].max()]
    with no_warnings():
        section, figures = record(build_context(m, past, df=df))
    assert section is not None
    check_figures(figures)
    assert "heading for" not in section.body
    assert not [t for t in figures[0].data if t.name and "range of the trend" in t.name]
    assert not figures[0].layout.shapes or all(
        s.type == "line" for s in figures[0].layout.shapes
    )
    assert not [t for t in figures[3].data if t.name and "forecast" in t.name]


def test_weekday_only_data_closes_the_weekend_gaps() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = fit_case("linear")
    axis = skip_axis(df["ds"], skip_days="weekends")
    kept = fcst[axis.keeps(fcst["ds"])]
    with no_warnings():
        section, figures = record(
            build_context(m, kept, df=df[axis.keeps(df["ds"])], skip=axis)
        )
    assert section is not None
    check_figures(figures)
    assert len(figures) == 4
    for fig in figures[:3]:
        assert fig.layout.xaxis.rangebreaks, (
            "a time axis must close the skipped weekdays"
        )
    # The years share a calendar axis made of one reference year, which is not a time axis of the data.
    assert not figures[3].layout.xaxis.rangebreaks
    assert "per day" in section.body


def test_a_short_history_has_no_years_and_still_builds() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("subdaily")
    assert (ctx.history_end - ctx.history["ds"].min()) < pd.Timedelta(days=7)
    with no_warnings():
        section, figures = record(ctx)
    assert section is not None and len(figures) == 3
    assert "Year over year" not in section.body


def test_build_leaves_the_context_frames_alone() -> None:
    pytest.importorskip("plotly")
    ctx = case_context("logistic")
    fcst, history = ctx.fcst.copy(), ctx.history.copy()
    params = copy.deepcopy(ctx.m.params)
    trend.build(ctx)
    pd.testing.assert_frame_equal(ctx.fcst, fcst)
    pd.testing.assert_frame_equal(ctx.history, history)
    assert all(np.array_equal(ctx.m.params[k], params[k]) for k in params)


def test_nothing_to_show_without_forecast_rows() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard._context import build_context

    m, df, fcst = fit_case("linear")
    assert trend.build(build_context(m, fcst.iloc[0:0], df=df)) is None
    assert trend.build(build_context(m, fcst.drop(columns="trend"), df=df)) is None
