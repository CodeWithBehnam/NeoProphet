"""Tests for the "Holidays" section of the interactive dashboard."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet, skip_axis
from prophet.dashboard import holidays as hol
from prophet.dashboard._context import DashboardContext, Section, build_context
from prophet.dashboard._palette import DARK_SWAP
from prophet.dashboard.holidays import (
    Ahead,
    HolidayDay,
    HolidayEffect,
    WindowDay,
)
from prophet.tests.test_visuals import CASES, fit_case, no_warnings

DATA = Path(__file__).parent
EM_DASH = "\u2014"
EN_DASH = "\u2013"


# ----------------------------------------------------------------------------
# Small hand-made values
# ----------------------------------------------------------------------------


def effect(
    name: str,
    value: float | None,
    window: tuple[int, int] = (0, 0),
    profile: tuple[WindowDay, ...] | None = None,
    occurrences: int = 2,
) -> HolidayEffect:
    if profile is None:
        profile = () if value is None else (WindowDay(0, value, occurrences),)
    return HolidayEffect(
        name=name,
        effect=value,
        occurrences=0 if value is None else occurrences,
        days=0 if value is None else occurrences,
        window=window,
        profile=profile,
    )


def day(date: str, name: str, offset: int = 0, value: float | None = 1.0) -> HolidayDay:
    return HolidayDay(pd.Timestamp(date), name, offset, value)


# ----------------------------------------------------------------------------
# Parsing and wording
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("Christmas Day_delim_+0", ("Christmas Day", 0)),
        ("Christmas Day_delim_-1", ("Christmas Day", -1)),
        ("New Year's Day (observed)_delim_+12", ("New Year's Day (observed)", 12)),
        ("a_delim_b_delim_+3", ("a_delim_b", 3)),
        ("yearly_delim_3", None),
        ("weekly_delim_1", None),
        ("temperature", None),
    ],
)
def test_parse_column(column: str, expected: tuple[str, int] | None) -> None:
    assert hol.parse_column(column) == expected


def test_offset_and_window_phrases() -> None:
    assert hol.offset_phrase(0) == "the day itself"
    assert hol.offset_phrase(-1) == "1 day before"
    assert hol.offset_phrase(3) == "3 days after"
    assert hol.window_phrase((0, 0)) == "Single day"
    assert hol.window_phrase((-1, 1)) == "Day plus 1 day before and 1 day after"
    assert hol.window_phrase((0, 2)) == "Day plus 2 days after"
    assert hol.window_phrase((-3, 0)) == "Day plus 3 days before"


def test_effect_phrase_and_negligible() -> None:
    assert hol.effect_phrase(41.2, False) == "adds 41.2 to y"
    assert hol.effect_phrase(-18.0, False) == "takes 18 off y"
    assert hol.effect_phrase(0.12, True) == "raises y by 12% of the trend level"
    assert hol.effect_phrase(-0.08, True) == "lowers y by 8% of the trend level"
    assert hol.is_negligible(0.1, False, 100.0)
    assert not hol.is_negligible(1.0, False, 100.0)
    assert hol.is_negligible(0.004, True, 100.0)
    assert not hol.is_negligible(0.02, True, 100.0)


def test_percentage_ticks_have_the_decimals_the_effects_need() -> None:
    assert hol.pct_tickformat(0.5) == ".1~%"
    assert hol.pct_tickformat(0.02) == ".2~%"
    assert hol.pct_tickformat(0.0004) == ".4~%"
    assert hol.pct_tickformat(0.0) == hol.pct_tickformat(float("nan")) == ".1~%"
    # Plotly steps its ticks 1, 2 or 5 times a power of ten, at least a tenth of
    # the range apart. An axis spanning at most 2.2 times the largest effect
    # therefore has a step of 22 * largest percentage points or more, and the
    # power of ten in that step is at least a fifth of it, 4.4 * largest. The
    # decimals must reach that digit.
    for largest in (0.9, 0.2, 0.04, 0.012, 0.0031, 0.0009, 0.0004, 0.00002):
        decimals = int(hol.pct_tickformat(largest)[1:].split("~")[0])
        assert 10.0**-decimals <= 4.4 * largest, largest


def test_bar_range_leaves_little_air_beyond_the_bars() -> None:
    low, high = hol.bar_range([-4.0, 1.0, 2.0])
    assert low < -4.0 < 2.0 < high
    assert high - 2.0 < 0.06 * 6.0 and -4.0 - low < 0.06 * 6.0
    assert hol.bar_range([2.0, 3.0])[0] == 0.0
    assert hol.bar_range([-2.0, -3.0])[1] == 0.0
    assert hol.bar_range([]) == [-1.0, 1.0]


# ----------------------------------------------------------------------------
# The effect of each holiday
# ----------------------------------------------------------------------------


def indicator_frame() -> tuple[pd.Series, pd.DataFrame, dict[str, float]]:
    """Ten days with a three-day holiday 'a', a one-day holiday 'b' and a holiday 'z' that never occurs."""
    dates = pd.Series(pd.date_range("2024-03-01", periods=10))
    on = {
        "a_delim_-1": [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        "a_delim_+0": [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        "a_delim_+1": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        "b_delim_+0": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "z_delim_+0": [0] * 10,
    }
    coefficients = {
        "a_delim_-1": 2.0,
        "a_delim_+0": 10.0,
        "a_delim_+1": 4.0,
        "b_delim_+0": -3.0,
        "z_delim_+0": 7.0,
    }
    return dates, pd.DataFrame(on), coefficients


def test_holiday_effects_average_over_the_days_it_applies() -> None:
    dates, indicators, coefficients = indicator_frame()
    a, b, z, missing = hol.holiday_effects(
        dates, indicators, coefficients, ["a", "b", "z", "nowhere"]
    )
    assert a.effect == pytest.approx((2 + 10 + 4) / 3)
    assert (a.occurrences, a.days, a.window) == (2, 6, (-1, 1))
    assert [(d.offset, d.effect, d.days) for d in a.profile] == [
        (-1, 2.0, 2),
        (0, 10.0, 2),
        (1, 4.0, 2),
    ]
    assert a.multi_day
    assert a.peak is not None and a.peak.offset == 0
    assert b.effect == -3.0 and not b.multi_day and b.peak is None
    # A coefficient is only an effect where the history shows the day; the prior alone is not.
    assert z.effect is None and z.profile == () and z.occurrences == 0
    assert missing.effect is None and missing.window == (0, 0)


def test_holiday_effects_count_a_holiday_once_per_calendar_day() -> None:
    dates = pd.Series(pd.date_range("2024-03-01", periods=9, freq="8h"))
    flags = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    indicators = pd.DataFrame({"h_delim_+0": flags})
    (h,) = hol.holiday_effects(dates, indicators, {"h_delim_+0": 5.0}, ["h"])
    assert (h.effect, h.occurrences, h.days) == (5.0, 1, 1)


def test_occurrences_come_from_the_holiday_dates_not_the_columns() -> None:
    # Weekday-only data: a Saturday holiday leaves its Friday, a Sunday one its Monday.
    dates = pd.Series(
        pd.to_datetime(["2024-03-01", "2024-03-04"])  # Friday and Monday
    )
    indicators = pd.DataFrame({"a_delim_-1": [1, 0], "a_delim_+1": [0, 1]})
    coefficients = {"a_delim_-1": 3.0, "a_delim_+1": 5.0}
    saturday_and_sunday = pd.DataFrame(
        {
            "holiday": ["a", "a"],
            "ds": pd.to_datetime(["2024-03-02", "2024-03-03"]),
            "lower_window": [-1, -1],
            "upper_window": [1, 1],
        }
    )
    counts = hol.occurrence_counts(saturday_and_sunday, dates)
    assert counts == {"a": 2}
    (without,) = hol.holiday_effects(dates, indicators, coefficients, ["a"])
    assert without.occurrences == 1  # what the columns alone can say: a lower bound
    (h,) = hol.holiday_effects(dates, indicators, coefficients, ["a"], counts)
    assert (h.occurrences, h.days) == (2, 2)
    assert h.effect == pytest.approx(4.0)


def test_occurrence_counts_follow_the_model_reading_of_the_frame() -> None:
    dates = pd.Series(pd.date_range("2024-03-01", periods=5))
    frame = pd.DataFrame(
        {
            "holiday": ["a", "a", "a", "b", "never", "c"],
            "ds": pd.to_datetime(
                [
                    "2024-03-02",
                    "2024-03-02",  # a duplicate row counts once
                    "2024-03-05",
                    "2024-03-08",  # outside the dates, and its window is too
                    "2030-01-01",
                    "2030-01-01",
                ]
            ),
            "lower_window": [
                0,
                0,
                0,
                np.nan,
                0,
                0,
            ],  # NaN means no window, as in the model
            "upper_window": [0, 0, 0, 2, 0, 0],
        }
    )
    # "c" is a name the model knew at fit time but that has no date now.
    frame.loc[frame["holiday"] == "c", "ds"] = pd.NaT
    assert hol.occurrence_counts(frame, dates) == {"a": 2}
    assert hol.occurrence_counts(frame.iloc[:0], dates) == {}
    assert hol.occurrence_counts(frame.drop(columns="holiday"), dates) == {}
    # A window that reaches into the dates counts, wherever the date itself is.
    reaching = pd.DataFrame(
        {
            "holiday": ["r"],
            "ds": pd.to_datetime(["2024-03-07"]),
            "lower_window": [-2],
            "upper_window": [0],
        }
    )
    assert hol.occurrence_counts(reaching, dates) == {"r": 1}
    # No window columns at all means a single day.
    plain = reaching.drop(columns=["lower_window", "upper_window"])
    assert hol.occurrence_counts(plain, dates) == {}


def test_holiday_effects_without_any_columns() -> None:
    dates = pd.Series(pd.date_range("2024-03-01", periods=3))
    (h,) = hol.holiday_effects(dates, pd.DataFrame(index=range(3)), {}, ["h"])
    assert h.effect is None


def test_chart_rows_keep_the_largest_and_say_how_many_are_left_out() -> None:
    effects = [effect(f"h{i:02d}", float(i - 15)) for i in range(30)] + [
        effect("unseen", None)
    ]
    rows, left_out = hol.chart_rows(effects, limit=20)
    assert len(rows) == 20 and left_out == 10
    values = [e.effect or 0.0 for e in rows]
    assert values == sorted(values)  # smallest first, so the largest ends on top
    kept = {abs(v) for v in values}
    assert min(kept) >= 5.0  # the ten smallest in size (|v| < 5) are the ones dropped
    assert all(e.name != "unseen" for e in rows)
    assert hol.chart_rows([effect("only", 1.0)]) == ((effect("only", 1.0),), 0)


def test_leaders_ignore_negligible_and_same_sign_effects() -> None:
    effects = [
        effect("up", 5.0),
        effect("down", -9.0),
        effect("tiny", 0.1),
        effect("x", None),
    ]
    top, bottom = hol.leaders(effects, False, 100.0)
    assert top is not None and top.name == "up"
    assert bottom is not None and bottom.name == "down"
    top, bottom = hol.leaders([effect("up", 5.0), effect("up2", 3.0)], False, 100.0)
    assert top is not None and top.name == "up" and bottom is None
    assert hol.leaders([effect("tiny", 0.1), effect("tiny2", -0.2)], False, 100.0) == (
        None,
        None,
    )
    assert hol.leaders([], False, 100.0) == (None, None)


# ----------------------------------------------------------------------------
# Holiday days, the forecast window and the line through time
# ----------------------------------------------------------------------------


def test_holiday_days_report_each_window_day_once() -> None:
    dates = pd.Series(pd.date_range("2024-12-22", periods=6).repeat(2))
    indicators = pd.DataFrame(
        {
            "a_delim_-1": np.repeat([0, 0, 0, 1, 0, 0], 2),
            "a_delim_+0": np.repeat([0, 0, 0, 0, 1, 0], 2),
        }
    )
    seen = [effect("a", 4.0, window=(-1, 0), profile=(WindowDay(0, 4.0, 1),))]
    found = hol.holiday_days(
        dates, indicators, {"a_delim_-1": 3.0, "a_delim_+0": 4.0}, seen
    )
    assert [(d.date.day, d.offset, d.effect) for d in found] == [
        (25, -1, None),  # the day before was never seen in the history
        (26, 0, 4.0),
    ]
    assert {d.holiday_date.day for d in found} == {26}


def span(first: str, last: str) -> hol.ForecastSpan:
    return hol.ForecastSpan(pd.Timestamp(first), pd.Timestamp(last))


def test_forecast_span_is_the_calendar_days_with_forecast_rows() -> None:
    assert hol.forecast_span(pd.Series([], dtype="datetime64[ns]")) is None
    # A history that ends at noon leaves the rest of that day to the forecast.
    ds = pd.Series(pd.date_range("2017-01-02 12:05", "2017-01-03 03:00", freq="5min"))
    assert hol.forecast_span(ds) == span("2017-01-02", "2017-01-03")
    only = hol.forecast_span(pd.Series(pd.to_datetime(["2025-01-05"])))
    assert only == span("2025-01-05", "2025-01-05")
    assert only is not None and only.contains(pd.Timestamp("2025-01-05"))
    assert not only.contains(pd.Timestamp("2025-01-04"))


def test_ahead_summary_counts_holiday_dates_inside_the_forecast_window() -> None:
    window = span("2024-12-25", "2025-01-10")  # the history ends on the 24th
    days = [
        day("2024-12-25", "a", -1),  # a holiday on the 26th
        day("2024-12-26", "a", 0),
        day("2024-12-25", "old", 5),  # a holiday on the 20th, its tail runs in
        day("2025-01-01", "b"),
        day("2024-12-20", "past"),  # wholly before the window
    ]
    assert hol.ahead_summary(days, window) == Ahead(
        2, "a", pd.Timestamp("2024-12-26"), ("old",)
    )
    assert hol.ahead_summary([], window) == Ahead(0, None, None)
    assert hol.ahead_summary(days, None) is None
    assert hol.next_dates(days, window) == {
        "a": hol.NextDate(pd.Timestamp("2024-12-26"), True),
        "old": hol.NextDate(pd.Timestamp("2024-12-20"), False),
        "b": hol.NextDate(pd.Timestamp("2025-01-01"), True),
    }
    assert hol.next_dates(days, None) == {}


def test_a_holiday_just_after_the_window_is_not_inside_it() -> None:
    # The forecast ends on the 6th; "promo" is on the 7th with a day of run-up on the 6th.
    window = span("2013-03-08", "2013-05-06")
    days = [day("2013-05-06", "promo", -1, 1.5)]
    ahead = hol.ahead_summary(days, window)
    assert ahead == Ahead(0, None, None, ("promo",))
    assert hol.next_dates(days, window) == {
        "promo": hol.NextDate(pd.Timestamp("2013-05-07"), False)
    }
    # Once the window reaches the 7th the holiday is inside it.
    inside = hol.ahead_summary(days, span("2013-03-08", "2013-05-07"))
    assert inside == Ahead(1, "promo", pd.Timestamp("2013-05-07"))


def test_the_rest_of_a_holiday_day_after_the_history_is_ahead() -> None:
    # Sub-daily history ending at noon on the holiday: its date is midnight, before the end.
    window = span("2017-01-02", "2017-01-02")
    days = [day("2017-01-01", "new year"), day("2017-01-02", "bank")]
    assert hol.ahead_summary(days, window) == Ahead(
        1, "bank", pd.Timestamp("2017-01-02")
    )
    headers, rows, left = hol.upcoming_table(days, window, False)
    assert [r[1] for r in rows] == ["bank"] and left == 0
    assert hol.next_dates(days, window) == {
        "bank": hol.NextDate(pd.Timestamp("2017-01-02"), True)
    }


def test_day_lines_escape_names_and_say_what_is_unknown() -> None:
    assert (
        hol.day_line(day("2024-12-26", "R&D <day>", 0, 1.0), False)
        == "R&amp;D &lt;day&gt;: +1"
    )
    assert (
        hol.day_line(day("2024-12-25", "a", -1, None), True)
        == "a (1 day before): not seen"
    )
    labels = hol.day_labels(
        [day("2024-12-26", "a", 0, 0.5), day("2024-12-26", "b", 0, -0.25)], True
    )
    assert labels == {pd.Timestamp("2024-12-26"): "<br>a: +50%<br>b: -25%"}


def test_spike_series_keeps_spikes_and_both_ends() -> None:
    ds = pd.Series(pd.date_range("2024-01-01", periods=10))
    total = np.zeros(10)
    total[[4, 7]] = [3.0, -2.0]
    series = hol.spike_series(ds, total, [""] * 10)
    assert [d.day for d in series["ds"]] == [1, 4, 5, 6, 7, 8, 9, 10]
    assert series["total"].tolist() == [0, 0, 3.0, 0, 0, -2.0, 0, 0]


def test_spike_series_keeps_a_sub_daily_holiday_flat_through_its_day() -> None:
    # One holiday day in three, every 8 hours: a ramp from the first row of the
    # day to the first row of the next would show the effect fading through it.
    ds = pd.Series(pd.date_range("2024-01-01", periods=9, freq="8h"))
    total = np.array([0, 0, 0, 2, 2, 2, 0, 0, 0], dtype=float)
    series = hol.spike_series(ds, total, list("abcdefghi"))
    assert series["ds"].tolist() == [ds[0], ds[2], ds[3], ds[5], ds[6], ds[8]]
    assert series["total"].tolist() == [0, 0, 2, 2, 0, 0]
    assert series["label"].tolist() == ["a", "c", "d", "f", "g", "i"]
    assert hol.spike_series(ds.iloc[:0], np.array([]), []).empty


def test_spike_series_has_a_vertical_edge_at_midnight_for_hourly_data() -> None:
    ds = pd.Series(pd.date_range("2024-03-01", periods=24 * 5, freq="h"))
    total = np.where(ds.dt.day == 3, 7.0, 0.0)
    series = hol.spike_series(ds, total, [""] * len(ds))
    spike = series[series["ds"].dt.day == 3]
    assert spike["total"].tolist() == [7.0, 7.0]
    assert spike["ds"].dt.hour.tolist() == [0, 23]
    # Neighbouring days are drawn at both ends too, so the edges are steps, not ramps.
    assert series["ds"].dt.day.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert len(series) == 10 < len(ds)


# ----------------------------------------------------------------------------
# Sentences
# ----------------------------------------------------------------------------


def window_effect() -> HolidayEffect:
    return effect(
        "Christmas Day",
        41.0,
        window=(-1, 1),
        profile=(WindowDay(-1, 12.0, 2), WindowDay(0, 80.0, 2), WindowDay(1, 31.0, 2)),
    )


def test_lede_names_the_biggest_raising_and_lowering_holiday() -> None:
    effects = [window_effect(), effect("Easter Monday", -18.0), effect("Lost", None)]
    ahead = Ahead(3, "Boxing Day", pd.Timestamp("2025-12-26"))
    text = hol.holiday_lede(effects, False, 100.0, ahead)
    assert text.startswith("The model includes 3 holidays.")
    assert (
        "1 holiday never falls on a date in the history, so it could not be measured."
        in text
    )
    assert (
        "Christmas Day lifts y the most" in text and "adds 41 to y on average" in text
    )
    assert "Its biggest day is the day itself, at +80." in text
    assert (
        "Easter Monday pulls y down the most" in text
        and "takes 18 off y on average" in text
    )
    assert (
        "Easter Monday pulls y down the most: on the days it applies, it takes 18 off y on average. 3 holidays"
        in text
    )
    assert text.endswith(
        "3 holidays fall inside the forecast window, the first being Boxing Day on 26 Dec 2025."
    )


def test_ahead_sentence_names_holidays_whose_window_reaches_in() -> None:
    inside = Ahead(1, "Boxing Day", pd.Timestamp("2025-12-26"), ("Christmas Day",))
    assert hol.ahead_sentence(inside) == (
        "One holiday falls inside the forecast window: Boxing Day on 26 Dec 2025. "
        "Christmas Day is dated outside it, but its window reaches in."
    )
    assert hol.ahead_sentence(Ahead(0, None, None, ("promo",))) == (
        "No holiday falls inside the forecast window. "
        "promo is dated outside it, but its window reaches in."
    )
    assert hol.ahead_sentence(Ahead(0, None, None, ("a", "b"))).endswith(
        "The windows of 2 holidays dated outside it reach in."
    )
    assert hol.ahead_sentence(Ahead(0, None, None)) == (
        "No holiday falls inside the forecast window."
    )
    assert hol.ahead_sentence(None) == ""


def test_lede_cautions_about_a_holiday_seen_once() -> None:
    effects = [effect("Rare", 9.0, occurrences=1), effect("Usual", -5.0)]
    text = hol.holiday_lede(effects, False, 100.0, None)
    assert "Rare lifts y the most" in text
    assert (
        text.count("It occurs only once in the history, so its size is a rough guide.")
        == 1
    )


def test_lede_in_multiplicative_mode_speaks_in_shares_of_the_trend() -> None:
    text = hol.holiday_lede(
        [effect("Sale", 0.12), effect("Dip", -0.08)], True, 100.0, None
    )
    assert "raises y by 12% of the trend level" in text
    assert "lowers y by 8% of the trend level" in text
    assert "forecast window" not in text


def test_lede_when_nothing_matters_or_nothing_was_seen() -> None:
    quiet = hol.holiday_lede(
        [effect("a", 0.1), effect("b", -0.2)], False, 100.0, Ahead(0, None, None)
    )
    assert (
        "barely move this forecast" in quiet
        and "the largest effect, b, takes 0.2 off y" in quiet
    )
    assert quiet.endswith("No holiday falls inside the forecast window.")
    lost = hol.holiday_lede([effect("a", None), effect("b", None)], False, 100.0, None)
    assert lost == (
        "The model includes 2 holidays, but none of them falls on a date in the history, "
        "so there was nothing to measure."
    )
    single = hol.holiday_lede([effect("a", None)], False, 100.0, None)
    assert "but it does not fall on a date in the history" in single


def test_summary_is_one_plain_sentence() -> None:
    effects = [window_effect(), effect("Easter Monday", -18.0)]
    text = hol.holiday_summary(
        effects, False, 100.0, Ahead(2, "x", pd.Timestamp("2025-01-01"))
    )
    assert text == (
        "Christmas Day raises y the most (+41) and Easter Monday lowers it the most (-18); "
        "2 holidays fall in the forecast window."
    )
    assert hol.holiday_summary(
        [effect("a", 5.0)], False, 100.0, Ahead(0, None, None)
    ) == ("a raises y the most (+5); no holiday falls in the forecast window.")
    assert (
        hol.holiday_summary([effect("a", -0.2)], True, 100.0, None)
        == "a lowers y the most (-20%)."
    )
    assert hol.holiday_summary(
        [effect("a", 5.0)], False, 100.0, Ahead(0, None, None, ("a",))
    ) == (
        "a raises y the most (+5); no holiday falls in the forecast window, "
        "though 1 holiday dated outside it has a window that reaches in."
    )
    assert hol.holiday_summary(
        [effect("a", 5.0)],
        False,
        100.0,
        Ahead(2, "a", pd.Timestamp("2025-01-01"), ("a", "b")),
    ) == (
        "a raises y the most (+5); 2 holidays fall in the forecast window, "
        "and 2 holidays dated outside it have windows that reach in."
    )
    assert hol.holiday_summary([effect("a", 0.1)], False, 100.0, None).startswith(
        "Holidays barely move"
    )
    assert "could not be measured" in hol.holiday_summary(
        [effect("a", None)], False, 100.0, None
    )
    assert hol.holiday_summary(
        [effect("a", None), effect("b", None)], False, 100.0, None
    ).startswith("None of the 2 holidays")


def test_summary_says_when_the_biggest_effect_rests_on_one_occurrence() -> None:
    effects = [
        effect("Rare", 9.0, occurrences=1),
        effect("Usual", -5.0),
        effect("Once down", -12.0, occurrences=1),
    ]
    assert hol.holiday_summary(effects, False, 100.0, None) == (
        "Rare raises y the most (+9, seen once) and Once down lowers it the most "
        "(-12, seen once)."
    )
    # Holidays seen more than once carry no such tag.
    assert hol.holiday_summary(effects[:2], False, 100.0, None) == (
        "Rare raises y the most (+9, seen once) and Usual lowers it the most (-5)."
    )
    assert "seen once" not in hol.holiday_summary(
        [effect("a", 5.0), effect("b", -3.0)], False, 100.0, None
    )


def test_wording_has_no_dashes() -> None:
    effects = [window_effect(), effect("Easter Monday", -18.0), effect("Lost", None)]
    ahead = Ahead(1, "Boxing Day", pd.Timestamp("2025-12-26"))
    texts = [
        hol.holiday_lede(effects, False, 100.0, ahead),
        hol.holiday_summary(effects, False, 100.0, ahead),
        hol.bar_hover(window_effect(), False),
        hol.window_phrase((-2, 3)),
    ]
    for text in texts:
        assert EM_DASH not in text and EN_DASH not in text


def test_bar_hover_lists_the_window_days() -> None:
    text = hol.bar_hover(window_effect(), False)
    assert text.startswith(
        "Seen 2 times in the history<br>Window: Day plus 1 day before and 1 day after"
    )
    assert (
        "1 day before: +12" in text
        and "The day itself: +80" in text
        and "1 day after: +31" in text
    )
    assert text.endswith("Biggest: the day itself")
    assert (
        hol.bar_hover(effect("one", 3.0, occurrences=1), False)
        == "Seen once in the history"
    )
    assert hol.peak_cell(window_effect()) == "The day itself"
    assert hol.peak_cell(effect("one", 3.0)) == "-"


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------


def test_all_holidays_table_orders_by_size_and_puts_unseen_last() -> None:
    effects = [
        effect("small", 1.0),
        effect("unseen", None),
        window_effect(),
        effect("down", -9.0),
    ]
    nexts = {"down": hol.NextDate(pd.Timestamp("2025-03-03"), True)}
    headers, rows = hol.all_holidays_table(effects, nexts, False)
    assert headers == [
        "Holiday",
        "Window",
        "Biggest day",
        "Next in forecast",
        "Seen in history",
        "Average effect",
    ]
    assert [r[0] for r in rows] == ["Christmas Day", "down", "small", "unseen"]
    assert rows[1] == ["down", "Single day", "-", "Mon 3 Mar 2025", "2", "-9"]
    assert (
        rows[0][1] == "Day plus 1 day before and 1 day after"
        and rows[0][2] == "The day itself"
    )
    assert rows[-1][-1] == "not seen" and rows[-1][3] == "none"
    headers, rows = hol.all_holidays_table(effects, None, False)
    assert "Next in forecast" not in headers and all(len(r) == 5 for r in rows)
    assert headers[-2:] == ["Seen in history", "Average effect"]


def test_columns_that_say_nothing_are_left_out_of_the_table() -> None:
    effects = [effect("small", 1.0), effect("down", -9.0), effect("unseen", None)]
    # No holiday has a window: no Window and no Biggest day column.
    headers, rows = hol.all_holidays_table(effects, None, False)
    assert headers == ["Holiday", "Seen in history", "Average effect"]
    assert rows[0] == ["down", "2", "-9"]
    # Nothing reaches the forecast (an empty mapping) is the same as no forecast window.
    assert hol.all_holidays_table(effects, {}, False)[0] == headers
    nexts = {"down": hol.NextDate(pd.Timestamp("2025-03-03"), True)}
    headers, rows = hol.all_holidays_table(effects, nexts, False)
    assert headers == [
        "Holiday",
        "Next in forecast",
        "Seen in history",
        "Average effect",
    ]
    assert rows[0] == ["down", "Mon 3 Mar 2025", "2", "-9"]
    # A window anywhere brings back both window columns.
    headers, _ = hol.all_holidays_table([*effects, window_effect()], None, False)
    assert headers[:3] == ["Holiday", "Window", "Biggest day"]


def test_next_cell_says_when_only_the_window_reaches_the_forecast() -> None:
    assert hol.next_cell(None) == "none"
    inside = hol.NextDate(pd.Timestamp("2025-03-03"), True)
    assert hol.next_cell(inside) == "Mon 3 Mar 2025"
    outside = hol.NextDate(pd.Timestamp("2025-03-03"), False)
    assert hol.next_cell(outside) == "Mon 3 Mar 2025 (window reaches in)"


def test_upcoming_table_adds_a_window_column_only_when_needed() -> None:
    window = span("2024-12-21", "2026-01-01")
    plain = [
        day("2024-12-26", "Boxing Day", 0, 0.34),
        day("2025-01-01", "New Year", 0, None),
    ]
    headers, rows, left = hol.upcoming_table(plain, window, False)
    assert headers == ["Date", "Holiday", "Predicted effect"] and left == 0
    assert rows == [
        ["Thu 26 Dec 2024", "Boxing Day", "+0.34"],
        ["Wed 1 Jan 2025", "New Year", "not seen"],
    ]
    windowed = [*plain, day("2024-12-24", "Christmas Day", -1, 0.1)]
    headers, rows, _ = hol.upcoming_table(
        sorted(windowed, key=lambda d: d.date), window, True
    )
    assert (
        headers[2] == "Window day"
        and rows[0][2] == "1 day before"
        and rows[0][3] == "+10%"
    )
    assert (
        hol.upcoming_table([day("2024-12-01", "past", 0, 1.0)], window, False)[1] == []
    )
    many = [day("2025-01-01", f"h{i}") for i in range(hol.MAX_LISTED + 5)]
    assert hol.upcoming_table(many, window, False)[2] == 5


# ----------------------------------------------------------------------------
# Fitted models: small, cached
# ----------------------------------------------------------------------------


def holiday_frame() -> pd.DataFrame:
    """A three-day window holiday, a holiday with a late window, one that never occurs and one that needs escaping."""
    return pd.DataFrame(
        {
            "holiday": [
                "launch",
                "launch",
                "sale",
                "sale",
                "never",
                "R&D <day>",
                "R&D <day>",
            ],
            "ds": pd.to_datetime(
                [
                    "2012-12-25",
                    "2013-06-03",
                    "2012-11-23",
                    "2013-11-29",
                    "2030-01-01",
                    "2012-07-17",
                    "2013-07-31",
                ]
            ),
            "lower_window": [-1, -1, 0, 0, 0, 0, 0],
            "upper_window": [1, 1, 2, 2, 0, 0, 0],
        }
    )


def edge_holidays() -> pd.DataFrame:
    """Two holidays that sit next to the forecast window rather than in it.

    On the first 200 rows of ``data.csv`` the history ends on 7 March 2013. "tail"
    falls on that last day and runs three days on, into the forecast. "runup" falls
    on 7 May 2013, the day after a 60 day forecast ends, and starts a day early.
    Each also occurs once in the history, so each has an effect to report.
    """
    return pd.DataFrame(
        {
            "holiday": ["tail", "tail", "runup", "runup"],
            "ds": pd.to_datetime(
                ["2012-08-01", "2013-03-07", "2012-09-05", "2013-05-07"]
            ),
            "lower_window": [0, 0, -1, -1],
            "upper_window": [3, 3, 0, 0],
        }
    )


@lru_cache(maxsize=None)
def holiday_case(name: str) -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    """Model, training data and forecast for one holiday configuration."""
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"])
    future_kwargs: dict[str, Any] = {"periods": 400}
    kwargs: dict[str, Any] = {"holidays": holiday_frame()}
    if name == "multiplicative":
        kwargs["holidays_mode"] = "multiplicative"
    elif name == "logistic":
        df = df.assign(cap=df["y"].max() * 1.5, floor=0.0)
        kwargs["growth"] = "logistic"
    elif name == "short":
        df = df.iloc[:200]
        future_kwargs = {"periods": 60}
        kwargs = {
            "holidays": holiday_frame().iloc[:2],
            "uncertainty_samples": 0,
        }
    elif name == "mcmc":
        df = df.iloc[:200]
        future_kwargs = {"periods": 60}
        kwargs = {"holidays": holiday_frame().iloc[:2], "mcmc_samples": 40}
    elif name == "edges":
        df = df.iloc[:200]
        future_kwargs = {"periods": 60}
        kwargs = {"holidays": edge_holidays(), "uncertainty_samples": 0}
    elif name == "many":
        future_kwargs = {"periods": 30}
        kwargs = {
            "holidays": pd.DataFrame(
                {
                    "holiday": [f"holiday {i:02d}" for i in range(25)],
                    "ds": df["ds"].iloc[10::20].reset_index(drop=True),
                    "lower_window": -1,
                    "upper_window": 1,
                }
            ),
            "uncertainty_samples": 0,
        }
    elif name == "midday":
        # The history stops at noon on a holiday: the afternoon of it is forecast.
        df = pd.read_csv(DATA / "data2.csv", parse_dates=["ds"])
        df = df[df["ds"] < pd.Timestamp("2017-01-02 12:00")]
        future_kwargs = {"periods": 48, "freq": "5min"}
        kwargs = {
            "holidays": pd.DataFrame(
                {
                    "holiday": ["new year", "bank holiday"],
                    "ds": pd.to_datetime(["2017-01-01", "2017-01-02"]),
                }
            )
        }
    elif name == "subdaily":
        df = pd.read_csv(DATA / "data2.csv", parse_dates=["ds"])
        future_kwargs = {"periods": 48, "freq": "5min"}
        kwargs = {
            "holidays": pd.DataFrame(
                {
                    "holiday": ["new year", "bank holiday"],
                    "ds": pd.to_datetime(["2017-01-01", "2017-01-02"]),
                }
            )
        }
    m = Prophet(**kwargs)
    if name == "additive":
        m.add_country_holidays("GB")
    elif name == "short":
        m.add_country_holidays("UK")
    m.fit(df)
    future = m.make_future_dataframe(**future_kwargs)
    if name == "logistic":
        future = future.assign(cap=df["cap"].iloc[0], floor=0.0)
    return m, df, m.predict(future)


def context(
    name: str, *, part: str = "all", skip_weekends: bool = False
) -> DashboardContext:
    pytest.importorskip("plotly")
    m, df, fcst = holiday_case(name)
    assert m.history is not None
    end = m.history["ds"].max()
    if part == "future":
        fcst = fcst[fcst["ds"] > end]
    elif part == "history":
        fcst = fcst[fcst["ds"] <= end]
    skip = skip_axis(df["ds"], skip_days="weekends") if skip_weekends else None
    if skip is not None:
        fcst = fcst[skip.keeps(fcst["ds"])]
    return build_context(m, fcst, df=df, skip=skip)


def built(ctx: DashboardContext) -> Section:
    with no_warnings():
        section = hol.build(ctx)
    assert section is not None
    return section


def colours_in(figure: Any) -> set[str]:
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif isinstance(value, str) and re.match(r"^(#[0-9a-fA-F]{6}|rgba?\()", value):
            found.add(value.upper().replace(" ", ""))

    payload = figure.to_plotly_json()
    payload["layout"].pop("template", None)  # the plotly default theme is not ours
    walk(payload)
    return found


MODELS = [
    "additive",
    "multiplicative",
    "logistic",
    "short",
    "subdaily",
    "mcmc",
    "midday",
    "edges",
    "many",
]


@pytest.mark.parametrize("name", MODELS)
def test_coefficients_reproduce_the_forecast_columns(name: str) -> None:
    ctx = context(name)
    inputs = hol.read_model(ctx)
    assert inputs is not None
    indicators = hol.holiday_indicators(ctx.m, ctx.fcst["ds"])
    vector = np.array([inputs.coefficients.get(c, 0.0) for c in indicators.columns])
    contribution = indicators.to_numpy(dtype=float) @ vector
    np.testing.assert_allclose(contribution, ctx.fcst["holidays"].to_numpy(), atol=1e-9)
    for holiday in inputs.names:
        mine = sum(
            indicators[c].to_numpy(dtype=float) * inputs.coefficients.get(c, 0.0)
            for c in indicators.columns
            if (p := hol.parse_column(str(c))) is not None and p[0] == holiday
        )
        np.testing.assert_allclose(mine, ctx.fcst[holiday].to_numpy(), atol=1e-9)
    assert inputs.multiplicative == (name == "multiplicative")
    if name == "mcmc":
        # The coefficients are the mean over the posterior samples, as predict uses.
        assert ctx.m.params["beta"].shape[0] > 1


def test_additive_effects_match_the_fitted_model() -> None:
    ctx = context("additive")
    inputs = hol.read_model(ctx)
    assert inputs is not None
    effects = {
        e.name: e
        for e in hol.holiday_effects(
            ctx.history["ds"],
            inputs.history,
            inputs.coefficients,
            inputs.names,
            inputs.occurrences,
        )
    }
    assert set(effects) == set(inputs.names)
    # The mean of the model's own column over the days that holiday is active.
    for name, e in effects.items():
        column = ctx.fcst.loc[ctx.fcst["ds"].isin(ctx.history["ds"]), name]
        active = column[column != 0]
        if e.effect is None:
            assert active.empty
        else:
            assert e.effect == pytest.approx(float(active.mean()), abs=1e-9)
    assert effects["never"].effect is None and effects["never"].occurrences == 0
    launch = effects["launch"]
    assert launch.window == (-1, 1) and launch.multi_day and launch.peak is not None
    assert launch.occurrences == 2
    assert effects["R&D <day>"].occurrences == 2
    assert (
        effects["Christmas Day"].effect is None
    )  # 25 December 2012 is not in the data


def test_the_section_applies_only_to_models_with_holidays() -> None:
    pytest.importorskip("plotly")
    for case in CASES:
        m, df, fcst = fit_case(case)
        with no_warnings():
            assert hol.build(build_context(m, fcst, df=df)) is None


@pytest.mark.parametrize("name", MODELS)
def test_section_is_built_for_each_holiday_model(name: str) -> None:
    section = built(context(name))
    assert (section.key, section.title) == ("holidays", "Holidays")
    assert section.summary and "\n" not in section.summary
    assert section.summary.endswith(".")
    assert 'class="lede"' in section.body and "holidays-all" in section.body
    assert EM_DASH not in section.body and EM_DASH not in section.summary
    assert len(section.body) < 1_000_000
    assert "<day>" not in section.body  # names are escaped wherever they are written
    assert hol.KEY == section.key and hol.TITLE == section.title


def test_additive_section_content() -> None:
    section = built(context("additive"))
    body = section.body
    for heading in (
        "Effect of each holiday",
        "Days within a window",
        "Holidays through time",
        "Holidays inside the forecast window",
        "All holidays",
    ):
        assert f"<h3>{heading}</h3>" in body
    assert body.count("plotly-graph-div") == 3
    assert "R&amp;D &lt;day&gt;" in body
    # The upcoming table already says which holidays reach the forecast, and when.
    assert "Next in forecast" not in body and "not seen" in body
    assert "<th>Window</th>" in body and "<th>Biggest day</th>" in body
    assert "fall inside the forecast window" in body
    assert "Effects are in the units of y." in body
    assert (
        "raises y the most" in section.summary and "forecast window" in section.summary
    )


def test_multiplicative_section_speaks_in_percent() -> None:
    section = built(context("multiplicative"))
    assert "of the trend level" in section.body
    assert "Effects are shares of the trend level" in section.body
    assert re.search(r"[+-]\d+(\.\d+)?%", section.summary)


def test_in_sample_forecast_has_no_forecast_window() -> None:
    section = built(context("additive", part="history"))
    assert "Holidays inside the forecast window" not in section.body
    assert "Next in forecast" not in section.body
    assert "forecast window" not in section.summary
    assert "forecast window" not in section.body.split("<h3>")[0]


def test_future_only_forecast_still_lists_the_holidays_ahead() -> None:
    section = built(context("additive", part="future"))
    assert "Holidays inside the forecast window" in section.body
    assert "fall inside the forecast window" in section.body


def test_weekday_data_with_skipped_weekends_and_a_short_history() -> None:
    ctx = context("short", skip_weekends=True)
    assert ctx.breaks and ctx.skip is not None
    section = built(ctx)
    assert "rangebreaks" in section.body
    assert section.body.count("plotly-graph-div") >= 1


def test_sub_daily_data_counts_holiday_days_not_rows() -> None:
    ctx = context("subdaily")
    inputs = hol.read_model(ctx)
    assert inputs is not None
    effects = {
        e.name: e
        for e in hol.holiday_effects(
            ctx.history["ds"], inputs.history, inputs.coefficients, inputs.names
        )
    }
    assert effects["new year"].occurrences == 1 and effects["new year"].days == 1
    section = built(ctx)
    assert "No holiday falls inside the forecast window." in section.body


def test_model_without_holiday_definitions_gives_no_section() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    ctx = build_context(m, fcst, df=df)
    assert hol.read_model(ctx) is None
    assert hol.build(ctx) is None


def section_parts(body: str) -> dict[str, str]:
    """The HTML after each ``<h3>``, by heading."""
    chunks = body.split("<h3>")[1:]
    return {c.split("</h3>")[0]: c.split("</h3>", 1)[1] for c in chunks}


def test_history_ending_inside_a_holiday_window_is_reported_consistently() -> None:
    ctx = context("edges")
    assert ctx.history_end == pd.Timestamp("2013-03-07")
    assert ctx.fcst["ds"].max() == pd.Timestamp("2013-05-06")
    section = built(ctx)
    parts = section_parts(section.body)
    lede = section.body.split("</p>")[0]
    # Neither holiday is dated inside the window, so the lede must not say it is.
    assert "No holiday falls inside the forecast window." in lede
    assert "One holiday falls" not in lede and "holidays fall inside" not in lede
    assert "The windows of 2 holidays dated outside it reach in." in lede
    assert section.summary.endswith(
        "no holiday falls in the forecast window, though 2 holidays dated outside it "
        "have windows that reach in."
    )
    # The table lists the three tail days and the run-up day, as the lede says.
    upcoming = parts["Holidays inside the forecast window"]
    for date in ("Fri 8 Mar 2013", "Sat 9 Mar 2013", "Sun 10 Mar 2013"):
        assert date in upcoming
    assert "Mon 6 May 2013" in upcoming and "1 day before" in upcoming
    assert upcoming.count("tail") == 3 and upcoming.count("runup") == 1
    # The table above says when each holiday reaches the forecast, so the full table does not.
    everything = parts["All holidays"]
    assert "Next in forecast" not in everything
    assert "(window reaches in)" not in everything
    assert "<td>none</td>" not in everything


def test_occurrences_in_a_built_section_come_from_the_dates() -> None:
    ctx = context("edges")
    inputs = hol.read_model(ctx)
    assert inputs is not None
    assert dict(inputs.occurrences) == {"tail": 2, "runup": 1}
    effects = {
        e.name: e
        for e in hol.holiday_effects(
            ctx.history["ds"],
            inputs.history,
            inputs.coefficients,
            inputs.names,
            inputs.occurrences,
        )
    }
    assert effects["tail"].occurrences == 2 and effects["runup"].occurrences == 1
    everything = section_parts(built(ctx).body)["All holidays"]
    seen = {
        cells[0]: cells[-2]
        for row in re.findall(r"<tr><td>(.*?)</tr>", everything)
        if len(cells := re.split(r"</td><td[^>]*>", row.removesuffix("</td>"))) > 2
    }
    assert seen == {"tail": "2", "runup": "1"}


def test_the_rest_of_a_holiday_after_a_midday_history_end_is_inside_the_window() -> (
    None
):
    ctx = context("midday")
    assert ctx.history_end == pd.Timestamp("2017-01-02 11:55")
    section = built(ctx)
    lede = section.body.split("</p>")[0]
    assert (
        "One holiday falls inside the forecast window: bank holiday on 2 Jan 2017."
        in lede
    )
    assert "No holiday falls" not in section.body
    assert section.summary.endswith("1 holiday falls in the forecast window.")
    upcoming = section_parts(section.body)["Holidays inside the forecast window"]
    assert "Mon 2 Jan 2017" in upcoming and "bank holiday" in upcoming
    assert "new year" not in upcoming


def test_the_heatmap_says_how_many_windowed_holidays_it_leaves_out() -> None:
    section = built(context("many"))
    parts = section_parts(section.body)
    assert section.body.count("plotly-graph-div") == 3
    bar_note = parts["Effect of each holiday"].split('<p class="note">')[1]
    heat_note = parts["Days within a window"].split('<p class="note">')[1]
    assert (
        "Showing the 20 largest effects; the other 5 are in the table below."
        in bar_note
    )
    assert "Showing the 20 largest; the other 5 are in the table below." in heat_note
    assert hol._profile_note(20, 0).endswith("never saw.")
    assert "other 3 are" in hol._profile_note(20, 3)


def numeric_flags(table_html: str) -> list[list[bool]]:
    """For each row, whether each cell carries the ``num`` class."""
    rows = re.findall(r"<tr>(.*?)</tr>", table_html)
    return [
        [' class="num"' in cell for cell in re.findall(r"<t[dh]([^>]*)>", row)]
        for row in rows
    ]


@pytest.mark.parametrize("name", ["additive", "multiplicative", "short", "edges"])
def test_numeric_columns_never_sit_before_a_text_column(name: str) -> None:
    # The page pads a numeric cell on its left only, so a text cell after one
    # would run into its figures ("-1.77Single day").
    section = built(context(name))
    tables = re.findall(r'<table class="tbl".*?</table>', section.body)
    assert len(tables) >= 1
    for table in tables:
        for flags in numeric_flags(table):
            assert any(flags)
            first = flags.index(True)
            assert all(flags[first:]), flags


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def test_effect_figure_uses_the_diverging_poles_and_a_legend_for_two_series() -> None:
    ctx = context("additive")
    rows = (effect("down", -2.0), effect("up", 3.0), effect("up2", 5.0))
    fig = hol.effect_figure(ctx, rows, False)
    by_name = {t.name: t for t in fig.data}
    assert set(by_name) == {"Raises y", "Lowers y"}
    assert by_name["Raises y"].marker.color == hol.RAISES
    assert by_name["Lowers y"].marker.color == hol.LOWERS
    assert fig.layout.showlegend is True
    assert fig.layout.yaxis.categoryarray == ("down", "up", "up2")
    assert all(t.hovertemplate for t in fig.data)
    only_up = hol.effect_figure(ctx, rows[1:], False)
    assert len(only_up.data) == 1 and only_up.layout.showlegend is False


def test_effect_figure_handles_the_largest_chart() -> None:
    ctx = context("additive")
    effects = [effect(f"holiday {i}", float(i - 14)) for i in range(40)]
    rows, left_out = hol.chart_rows(effects)
    fig = hol.effect_figure(ctx, rows, False)
    assert sum(len(t.y) for t in fig.data) == hol.MAX_BARS == 20
    assert left_out == 20
    note = hol._bars_note(len(rows), left_out, False)
    assert (
        "Showing the 20 largest effects; the other 20 are in the table below." in note
    )


def test_value_labels_are_written_out_and_match_the_tables() -> None:
    ctx = context("additive")
    rows = (effect("neg", -4190.0), effect("big", 12345.6), effect("small", 0.5))
    fig = hol.effect_figure(ctx, rows, False)
    labels = sorted(t for trace in fig.data for t in trace.text)
    assert labels == ["+0.5", "+12.3K", "-4,190"]
    assert all("e+" not in t and "e-" not in t for t in labels)
    assert [trace.textposition for trace in fig.data] == ["outside"] * len(fig.data)
    tiny = hol.effect_figure(ctx, (effect("a", 0.0042), effect("b", -0.12)), True)
    assert sorted(t for trace in tiny.data for t in trace.text) == ["+0.42%", "-12%"]
    _, table_rows = hol.all_holidays_table(list(rows), None, False)
    assert {r[0]: r[-1] for r in table_rows} == {
        "big": "+12.3K",
        "neg": "-4,190",
        "small": "+0.5",
    }


def test_window_cells_are_written_out_too() -> None:
    ctx = context("additive")
    rows = [
        effect(
            "a", 1.0, (-1, 1), (WindowDay(-1, 1234.5, 1), WindowDay(1, -45678.0, 1))
        ),
    ]
    fig = hol.profile_figure(ctx, rows, False)
    assert [list(r) for r in fig.data[0].text] == [["+1,234", "", "-45.7K"]]
    assert fig.data[0].texttemplate == "%{text}"
    window = (WindowDay(0, 0.0123, 1), WindowDay(2, -0.0004, 1))
    pct = hol.profile_figure(ctx, [effect("a", 0.01, (0, 2), window)], True)
    assert [list(r) for r in pct.data[0].text] == [["+1.2%", "", "-0.04%"]]


def test_names_of_long_negative_bars_keep_clear_of_their_value_labels() -> None:
    ctx = context("additive")
    mixed = hol.effect_figure(ctx, (effect("down", -2.0), effect("up", 3.0)), False)
    assert mixed.layout.yaxis.ticks == "outside"
    assert mixed.layout.yaxis.ticklen == hol.LABEL_GAP
    # A gap, not a mark drawn through the value label of the longest bar (the
    # ticks are paper-coloured and sit under the traces), and no axis rule at
    # the left edge for that label to run into either.
    assert mixed.layout.yaxis.tickcolor == hol.PAPER
    assert mixed.layout.yaxis.layer == "below traces"
    assert mixed.layout.yaxis.showline is False
    only_up = hol.effect_figure(ctx, (effect("a", 1.0), effect("b", 2.0)), False)
    assert not only_up.layout.yaxis.ticks


def test_percentage_ticks_follow_the_size_of_small_effects() -> None:
    ctx = context("multiplicative")
    small = (effect("a", 0.0021), effect("b", -0.0055))
    bars = hol.effect_figure(ctx, small, True)
    assert bars.layout.xaxis.tickformat == hol.pct_tickformat(0.0055) == ".2~%"
    series = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=3),
            "total": [0.0, 0.004, 0.0],
            "label": "",
        }
    )
    line = hol.timeline_figure(ctx, series, True)
    assert line.layout.yaxis.tickformat == hol.pct_tickformat(0.004)
    window = [
        effect("a", 0.011, (0, 1), (WindowDay(0, 0.011, 1), WindowDay(1, 0.002, 1)))
    ]
    heat = hol.profile_figure(ctx, window, True)
    assert heat.data[0].colorbar.tickformat == hol.pct_tickformat(0.011) != ".0%"
    large = hol.effect_figure(ctx, (effect("a", 0.4), effect("b", -0.3)), True)
    assert large.layout.xaxis.tickformat == ".1~%"


def test_sub_daily_timeline_is_a_step_through_the_holiday_day() -> None:
    ctx = context("subdaily")
    labels = [""] * len(ctx.fcst)
    series = hol.spike_series(ctx.fcst["ds"], ctx.fcst["holidays"].to_numpy(), labels)
    new_year = series[series["ds"].dt.normalize() == pd.Timestamp("2017-01-01")]
    assert len(new_year) == 2 and new_year["total"].nunique() == 1
    assert new_year["total"].iloc[0] != 0
    assert new_year["ds"].dt.hour.tolist() == [0, 23]  # first and last row of the day
    fig = hol.timeline_figure(ctx, series, False)
    plotted = pd.Series(
        list(fig.data[0].y), index=pd.DatetimeIndex(list(fig.data[0].x))
    )
    day = plotted[
        pd.DatetimeIndex(plotted.index).normalize() == pd.Timestamp("2017-01-01")
    ]
    assert day.nunique() == 1


def test_figures_use_only_colours_the_dark_theme_can_swap() -> None:
    for name in ("additive", "multiplicative"):
        ctx = context(name)
        inputs = hol.read_model(ctx)
        assert inputs is not None
        effects = hol.holiday_effects(
            ctx.history["ds"], inputs.history, inputs.coefficients, inputs.names
        )
        rows, _ = hol.chart_rows(effects)
        windowed = [e for e in effects if e.effect is not None and e.multi_day]
        series = pd.DataFrame(
            {"ds": ctx.fcst["ds"], "total": ctx.fcst["holidays"], "label": ""}
        )
        for fig in (
            hol.effect_figure(ctx, rows, inputs.multiplicative),
            hol.profile_figure(ctx, windowed, inputs.multiplicative),
            hol.timeline_figure(ctx, series, inputs.multiplicative),
        ):
            assert colours_in(fig) <= set(DARK_SWAP), colours_in(fig) - set(DARK_SWAP)
            assert all(t.hovertemplate for t in fig.data)


def test_timeline_closes_the_gaps_of_skipped_days_and_marks_the_forecast() -> None:
    ctx = context("short", skip_weekends=True)
    series = pd.DataFrame(
        {"ds": ctx.fcst["ds"], "total": ctx.fcst["holidays"], "label": ""}
    )
    fig = hol.timeline_figure(ctx, series, False)
    assert fig.layout.xaxis.type == "date"
    assert len(fig.layout.xaxis.rangebreaks) == len(ctx.breaks) > 0
    assert len(fig.layout.shapes) >= 2  # history and forecast tints
    assert "%d %b %Y" in fig.data[0].hovertemplate


def test_timeline_marks_spikes_and_hides_markers_on_zero_days() -> None:
    ctx = context("additive")
    series = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=3),
            "total": [0.0, 2.0, 0.0],
            "label": ["", "<br>a: +2", ""],
        }
    )
    fig = hol.timeline_figure(ctx, series, False)
    assert list(fig.data[0].marker.size) == [0, 5, 0]
    assert fig.data[0].customdata[1] == "<br>a: +2"


def test_profile_figure_leaves_unseen_days_blank() -> None:
    ctx = context("additive")
    rows = [
        effect(
            "a",
            1.0,
            (-1, 1),
            (WindowDay(-1, 0.5, 1), WindowDay(0, 1.0, 1), WindowDay(1, 1.5, 1)),
        ),
        effect("b", -2.0, (0, 1), (WindowDay(0, -2.0, 1),)),
    ]
    fig = hol.profile_figure(ctx, rows, False)
    z = fig.data[0].z
    assert list(fig.data[0].x) == [-1, 0, 1]
    assert [list(r) for r in z] == [[0.5, 1.0, 1.5], [None, -2.0, None]]
    assert fig.data[0].zmin == -fig.data[0].zmax


def test_section_stays_small_and_does_not_touch_the_context() -> None:
    ctx = context("additive")
    before = (ctx.fcst.copy(), ctx.history.copy())
    section = built(ctx)
    pd.testing.assert_frame_equal(ctx.fcst, before[0])
    pd.testing.assert_frame_equal(ctx.history, before[1])
    assert len(section.body) < 400_000
