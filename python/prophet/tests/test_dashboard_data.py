"""The "data" section of the dashboard: coverage, data facts, the forecast table and its CSV."""

import io
import json
import re
import shutil
import subprocess
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet, skip_axis
from prophet.dashboard import data
from prophet.dashboard._context import DashboardContext, build_context
from prophet.plot import _plotly_rangebreaks, parse_skip_hours
from prophet.tests.test_visuals import CASES, fit_case, no_warnings
from prophet.visuals import _fmt_day

DATA = Path(__file__).parent
EM_DASH = "\u2014"
CSV_BLOCK = re.compile(r'<script type="text/csv" class="csv-data">(.*?)</script>', re.S)


def days(start: str, periods: int, drop: tuple[int, ...] = ()) -> pd.Series:
    """Daily timestamps from ``start`` without the positions listed in ``drop``."""
    index = pd.date_range(start, periods=periods, freq="D")
    keep = [i for i in range(periods) if i not in drop]
    return pd.Series(index[keep])


# ---- pure: coverage ---------------------------------------------------------


def test_coverage_of_a_complete_daily_series() -> None:
    cov = data.coverage(days("2024-01-01", 10))
    assert (cov.rows, cov.expected, cov.present, cov.missing) == (10, 10, 10, 0)
    assert cov.gaps == ()
    assert cov.longest is None
    assert cov.share_missing == 0.0


def test_coverage_finds_the_gap_and_its_neighbours() -> None:
    cov = data.coverage(days("2024-01-01", 10, drop=(4, 5, 6)))
    assert (cov.rows, cov.expected, cov.missing) == (7, 10, 3)
    assert cov.share_missing == pytest.approx(0.3)
    (gap,) = cov.gaps
    assert gap == data.Gap(
        before=pd.Timestamp("2024-01-04"),
        after=pd.Timestamp("2024-01-08"),
        missing=3,
        slot=3,
    )
    assert cov.longest == gap


def test_longest_gap_is_the_earliest_of_equals() -> None:
    cov = data.coverage(days("2024-01-01", 20, drop=(3, 4, 12, 13)))
    assert [g.missing for g in cov.gaps] == [2, 2]
    assert cov.longest is not None
    assert cov.longest.before == pd.Timestamp("2024-01-03")


def test_coverage_counts_unsorted_input_and_ignores_nulls() -> None:
    ds = days("2024-01-01", 6, drop=(2,)).sample(frac=1.0, random_state=1)
    cov = data.coverage(pd.concat([ds, pd.Series([pd.NaT])]))
    assert (cov.rows, cov.expected, cov.missing) == (5, 6, 1)


def test_weekends_are_gaps_until_they_are_skipped() -> None:
    weekdays = pd.Series(pd.bdate_range("2024-01-01", periods=15))
    plain = data.coverage(weekdays)
    assert (plain.expected, plain.missing, len(plain.gaps)) == (19, 4, 2)
    skip = skip_axis(weekdays, skip_days="weekends")
    closed = data.coverage(weekdays, skip)
    assert (closed.expected, closed.missing, closed.gaps) == (15, 0, ())


def test_a_missing_weekday_is_found_when_weekends_are_skipped() -> None:
    weekdays = pd.Series(pd.bdate_range("2024-01-01", periods=15))
    weekdays = weekdays[weekdays != pd.Timestamp("2024-01-10")]
    cov = data.coverage(weekdays, skip_axis(weekdays, skip_days="weekends"))
    (gap,) = cov.gaps
    assert gap.missing == 1
    assert (gap.before, gap.after) == (
        pd.Timestamp("2024-01-09"),
        pd.Timestamp("2024-01-11"),
    )


def test_skipped_hours_do_not_count_as_missing() -> None:
    ds = pd.Series(pd.date_range("2024-01-01", periods=24 * 12 * 2, freq="5min"))
    skip = skip_axis(ds, skip_hours=range(0, 8))
    kept = ds[skip.keeps(ds)].reset_index(drop=True)
    assert data.coverage(kept, skip).missing == 0
    assert data.coverage(kept).missing > 0


def test_months_of_unequal_length_stay_on_the_grid() -> None:
    months = pd.Series(pd.date_range("2020-01-01", periods=24, freq="MS"))
    assert data.coverage(months).missing == 0
    cov = data.coverage(months.drop(index=10).reset_index(drop=True))
    assert (cov.expected, cov.missing) == (24, 1)


def test_repeated_timestamps_count_once_on_the_grid() -> None:
    ds = pd.concat([days("2024-01-01", 5), days("2024-01-03", 1)])
    cov = data.coverage(ds)
    assert (cov.rows, cov.present, cov.expected, cov.missing) == (6, 5, 5, 0)


def test_one_timestamp_has_no_gaps_and_nothing_raises() -> None:
    cov = data.coverage(days("2024-01-01", 1))
    assert (cov.rows, cov.expected, cov.gaps) == (1, 1, ())


def test_coverage_rejects_an_empty_series() -> None:
    with pytest.raises(ValueError, match="no timestamps"):
        data.coverage(pd.Series([], dtype="datetime64[ns]"))


# ---- pure: calendar spacing, the grid and uneven spacing ----------------------


def calendar_series(
    freq: str, drop: tuple[int, ...] = ()
) -> tuple[pd.DatetimeIndex, pd.Series]:
    """240 timestamps from 2000 at ``freq`` and the same without the positions in ``drop``."""
    index = pd.date_range("2000-01-01", periods=240, freq=freq)
    return index, pd.Series(index[[i for i in range(240) if i not in drop]])


@pytest.mark.parametrize("freq", ["MS", "ME", "QS", "YS"])
@pytest.mark.parametrize("length", [1, 3, 12, 24, 36, 60])
def test_long_gaps_in_calendar_data_count_every_period(freq: str, length: int) -> None:
    drop = (*range(50, 50 + length), *range(150, 150 + length))
    _, ds = calendar_series(freq, drop)
    cov = data.coverage(ds)
    assert cov.regular
    assert (cov.expected, cov.missing) == (240, 2 * length)
    assert [g.missing for g in cov.gaps] == [length, length]


def test_typical_step_is_the_mean_of_the_jumps_around_the_median() -> None:
    jumps = np.array([31, 28, 31, 30, 31, 30, 62.0, 31])
    assert data.typical_step(jumps) == pytest.approx(
        np.mean([31, 28, 31, 30, 31, 30, 31])
    )
    assert data.typical_step(np.array([])) == 1.0
    assert data.typical_step(np.array([0.0, 0.0])) == 1.0
    assert data.typical_step(np.array([1.0, 3.0])) == 2.0


def test_whole_months_needs_a_calendar_step_and_a_steady_day() -> None:
    month = 30.4375
    starts = pd.date_range("2000-01-01", periods=36, freq="MS")
    assert data.whole_months(month, starts, None) == 1
    assert (
        data.whole_months(
            3 * month, pd.date_range("2000-01-01", periods=12, freq="QS"), None
        )
        == 3
    )
    assert (
        data.whole_months(
            12 * month, pd.date_range("2000-01-01", periods=5, freq="YS"), None
        )
        == 12
    )
    assert (
        data.whole_months(
            month, pd.date_range("2000-01-31", periods=12, freq="ME"), None
        )
        == 1
    )
    thirty = pd.DatetimeIndex(
        pd.Timestamp("2000-01-01") + pd.to_timedelta(30 * np.arange(30), unit="D")
    )
    assert data.whole_months(30.0, thirty, None) == 0
    assert data.whole_months(28.0, thirty, None) == 0
    assert data.whole_months(7.0, thirty, None) == 0
    weekdays = skip_axis(starts, skip_days="weekends")
    assert data.whole_months(month, starts, weekdays) == 0


@pytest.mark.parametrize("freq", ["MS", "ME", "QS", "YS", "W", "D"])
def test_grid_labels_follow_the_calendar_through_gaps(freq: str) -> None:
    index, ds = calendar_series(freq, (*range(50, 74), 150, 151, 152))
    cov = data.coverage(ds)
    assert cov.grid.labels(np.arange(240)).tolist() == index.tolist()


def test_grid_times_do_not_drift_off_monthly_data() -> None:
    index, ds = calendar_series("MS", tuple(range(100, 140)))
    grid = data.coverage(ds).grid
    times = grid.times(np.arange(240))
    assert times.is_monotonic_increasing
    assert (times[grid.slots] == index[grid.slots]).all()
    assert abs((times - index) / pd.Timedelta(days=1)).max() <= 2
    before, after = grid.times([-0.5, 239.5])
    assert index[0] - before < pd.Timedelta(days=16) and after - index[
        -1
    ] < pd.Timedelta(days=16)


def test_uneven_spacing_is_not_called_gappy() -> None:
    rng = np.random.default_rng(1)
    offsets = np.sort(rng.random(300)) * 365
    events = pd.Series(pd.Timestamp("2022-01-01") + pd.to_timedelta(offsets, unit="D"))
    cov = data.coverage(events)
    assert not cov.regular
    assert (cov.rows, cov.expected, cov.present, cov.missing, cov.gaps) == (
        300,
        300,
        300,
        0,
        (),
    )
    kept = rng.random(600) < 0.3
    assert not data.coverage(
        days("2022-01-01", 600)[kept].reset_index(drop=True)
    ).regular


@pytest.mark.parametrize("freq", ["D", "h", "5min", "MS", "W", "QS"])
def test_regular_series_stay_regular(freq: str) -> None:
    index = pd.date_range("2020-01-01", periods=300, freq=freq)
    assert data.coverage(pd.Series(index)).regular
    rng = np.random.default_rng(4)
    thinned = pd.Series(index[rng.random(300) < 0.7])
    cov = data.coverage(thinned)
    assert cov.regular and cov.missing > 0


def test_uneven_spacing_is_described_not_counted() -> None:
    rng = np.random.default_rng(1)
    offsets = np.sort(rng.random(300)) * 365
    cov = data.coverage(
        pd.Series(pd.Timestamp("2022-01-01") + pd.to_timedelta(offsets, unit="D"))
    )
    step = pd.Timedelta(hours=20)
    lede = data.lede_text(cov, step, None, "1 year")
    assert lede == (
        "The observations are unevenly spaced, so there is no regular series "
        "to count missing timestamps against."
    )
    pairs = dict(data.coverage_pairs(cov, step, None, "1 year"))
    assert pairs["Spacing"] == "Uneven, typically about 20 hours apart"
    assert pairs["Timestamps missing"] == "Not counted: the spacing is uneven"
    assert "Timestamps expected" not in pairs
    summary = data.summary_text(cov, step, None, None, None)
    assert summary == "Timestamps are unevenly spaced, so gaps were not counted."
    assert data.missing_pattern(cov, step, None) is None


# ---- pure: fixed hours of the day ---------------------------------------------

HOUR = pd.Timedelta(hours=1)
OFFICE_HOURS = (*range(0, 9), *range(17, 24))


def business_hours(days_: int = 30) -> pd.Series:
    hours = pd.date_range("2022-01-01", periods=24 * days_, freq="h")
    return pd.Series(hours[(hours.hour >= 9) & (hours.hour < 17)])


def test_fixed_hours_of_the_day_are_recognised_and_skipping_them_clears_the_gaps() -> (
    None
):
    ds = business_hours()
    cov = data.coverage(ds)
    assert cov.missing > 0.6 * cov.expected
    assert data.unused_hours(cov, HOUR, None) == OFFICE_HOURS
    skip = skip_axis(ds, skip_hours=OFFICE_HOURS)
    closed = data.coverage(ds, skip)
    assert (closed.missing, closed.gaps) == (0, ())
    assert data.unused_hours(closed, HOUR, skip) == ()
    assert data.unused_hours(cov, HOUR, skip) == ()
    pattern = data.missing_pattern(cov, HOUR, None)
    assert pattern is not None and pattern.kind == "hours"
    assert "falls from 17:00 to 09:00" in pattern.hint
    assert "skip_hours=[*range(0, 9), *range(17, 24)]" in pattern.hint
    summary = data.summary_text(cov, HOUR, None, None, pattern)
    assert summary.startswith("66% of the expected timestamps are missing, nearly all")


def test_the_skip_hours_argument_means_the_hours_it_was_made_from() -> None:
    for hours in [
        set(range(0, 8)),
        set(OFFICE_HOURS),
        {22, 23, 0, 1},
        {3},
        set(range(6, 22)),
    ]:
        argument = data.skip_hours_argument(sorted(hours))
        spans = re.findall(r"range\((\d+), (\d+)\)", argument)
        assert spans, argument
        assert parse_skip_hours(
            [h for a, b in spans for h in range(int(a), int(b))]
        ) == frozenset(hours)
    assert data.skip_hours_argument([0, 1, 2]) == "range(0, 3)"
    assert data.hours_text([22, 23, 0, 1]) == "from 22:00 to 02:00"
    assert data.hours_text([1, 2, 20]) == "from 01:00 to 03:00 and from 20:00 to 21:00"


def test_holes_spread_over_every_hour_are_not_a_pattern() -> None:
    rng = np.random.default_rng(5)
    hours = pd.date_range("2022-01-01", periods=24 * 40, freq="h")
    cov = data.coverage(pd.Series(hours[rng.random(len(hours)) > 0.1]))
    assert cov.gaps
    assert data.unused_hours(cov, HOUR, None) == ()
    assert data.missing_pattern(cov, HOUR, None) is None
    daily = data.coverage(days("2024-01-01", 30, drop=(4, 5)))
    assert data.unused_hours(daily, pd.Timedelta(days=1), None) == ()


def test_missing_slots_list_every_hole_or_a_fixed_sample() -> None:
    holes = [5, 10, 11, 20, 21, 22]
    cov = data.coverage(days("2024-01-01", 60, drop=tuple(holes)))
    assert data.missing_slots(cov).tolist() == holes
    sample = data.missing_slots(cov, limit=4)
    assert len(sample) == 4 and set(sample.tolist()) <= set(holes)
    assert sample.tolist() == data.missing_slots(cov, limit=4).tolist()
    assert data.missing_slots(data.coverage(days("2024-01-01", 5))).size == 0


# ---- pure: the data the caller passed ---------------------------------------


def test_observed_facts_count_each_kind_of_oddity() -> None:
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2023-12-31",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-09",
                ]
            ),
            "y": [1.0, 0.0, -2.0, np.nan, 5.0, 7.0],
        }
    )
    facts = data.observed_facts(
        df, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")
    )
    assert facts == data.ObservedFacts(
        rows=6, duplicates=1, missing_values=1, zeros=1, negatives=1, before=1, after=1
    )


def test_the_row_count_is_listed_once_when_it_is_the_training_row_count() -> None:
    facts = data.ObservedFacts(510, 0, 0, 0, 0, 0, 0)
    same = dict(data.observed_pairs(facts, 510))
    assert "Rows" not in same and same["Repeated timestamps"] == "None"
    # A frame that is not the training frame (rows were dropped) keeps its count.
    other = dict(data.observed_pairs(facts, 480))
    assert other["Rows"] == "510"
    assert dict(data.observed_pairs(facts))["Rows"] == "510"


def test_the_section_names_its_blocks_without_repeating_the_training_rows() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    section = data.build(build_context(m, fcst, df=df))
    assert section is not None
    assert (
        "<h3>Your data</h3>" in section.body and "<h3>Missing data</h3>" in section.body
    )
    assert "The data you passed" not in section.body
    assert section.body.count(f"<dd>{len(m.history):,}</dd>") == 1


def test_observed_facts_tolerate_missing_columns_and_bad_values() -> None:
    first, last = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")
    assert data.observed_facts(pd.DataFrame({"y": [1.0]}), first, last) is None
    only_dates = data.observed_facts(
        pd.DataFrame({"ds": ["2024-01-02", "oops"]}), first, last
    )
    assert only_dates == data.ObservedFacts(2, 0, 0, 0, 0, 0, 0)
    text_values = data.observed_facts(
        pd.DataFrame({"ds": ["2024-01-02", "2024-01-03"], "y": ["3", "n/a"]}),
        first,
        last,
    )
    assert text_values is not None
    assert text_values.missing_values == 1


def test_observed_facts_compare_timezone_aware_dates_by_wall_clock() -> None:
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-02", "2024-01-09"]).tz_localize("UTC"),
            "y": [1.0, 2.0],
        }
    )
    facts = data.observed_facts(
        df, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")
    )
    assert facts is not None
    assert (facts.before, facts.after) == (0, 1)


# ---- pure: wording ----------------------------------------------------------


def test_points_and_gaps_are_phrased_in_the_unit_of_the_data() -> None:
    day, weekend = (
        pd.Timedelta(days=1),
        skip_axis(days("2024-01-01", 14), skip_days="weekends"),
    )
    assert data.points_phrase(1, day, None) == "1 day"
    assert data.points_phrase(9, day, None) == "9 days"
    assert data.points_phrase(5, day, weekend) == "5 weekdays"
    assert data.points_phrase(36, pd.Timedelta(minutes=5), None) == "36 points"
    gap = data.Gap(pd.Timestamp("2024-03-02"), pd.Timestamp("2024-03-12"), 9, 0)
    assert (
        data.gap_phrase(gap, day, None)
        == "9 days missing between 2 Mar 2024 and 12 Mar 2024"
    )


def test_spacing_clause_names_the_spacing() -> None:
    assert data.spacing_clause(pd.Timedelta(days=1), None) == "one point per day"
    assert (
        data.spacing_clause(pd.Timedelta(minutes=5), None)
        == "one point every 5 minutes"
    )
    weekend = skip_axis(days("2024-01-01", 14), skip_days="weekends")
    assert data.spacing_clause(pd.Timedelta(days=1), weekend) == "one point per weekday"


def test_missing_weekday_counts_walk_the_days_of_each_gap() -> None:
    friday = pd.Timestamp("2024-01-05")
    gap = data.Gap(friday, pd.Timestamp("2024-01-08"), 2, 0)  # Saturday, Sunday
    assert data.missing_weekday_counts([gap]).tolist() == [0, 0, 0, 0, 0, 1, 1]
    long = data.Gap(friday, pd.Timestamp("2024-01-20"), 14, 0)  # two full weeks
    assert data.missing_weekday_counts([long]).tolist() == [2] * 7
    odd = data.Gap(friday, pd.Timestamp("2024-01-13"), 7 + 2, 0)
    assert data.missing_weekday_counts([odd]).tolist() == [1, 1, 1, 1, 1, 2, 2]


def test_weekday_series_is_recognised_only_without_a_skip() -> None:
    weekdays = pd.Series(pd.bdate_range("2024-01-01", periods=40))
    cov = data.coverage(weekdays)
    day = pd.Timedelta(days=1)
    assert data.looks_like_weekdays_only(cov, day, None)
    assert not data.looks_like_weekdays_only(
        cov, day, skip_axis(weekdays, skip_days="weekends")
    )
    assert not data.looks_like_weekdays_only(cov, pd.Timedelta(hours=1), None)
    scattered = data.coverage(days("2024-01-01", 60, drop=(3, 4, 10, 11, 22, 23, 30)))
    assert not data.looks_like_weekdays_only(scattered, day, None)
    assert not data.looks_like_weekdays_only(
        data.coverage(days("2024-01-01", 10)), day, None
    )


def test_lede_reads_the_numbers_it_leaves_to_the_list_beside_it() -> None:
    day = pd.Timedelta(days=1)
    cov = data.coverage(days("2024-01-01", 10, drop=(4, 5, 6)))
    text = data.lede_text(cov, day, None, "10 days")
    assert text == (
        "A large share of the timestamps expected at this spacing are missing, "
        "all in one gap."
    )
    # What the list says (rows, range, spacing, counts) is not said twice.
    pairs = dict(data.coverage_pairs(cov, day, None, "10 days"))
    assert pairs["Rows used for training"] == "7"
    assert not any(ch.isdigit() for ch in text)
    whole = data.lede_text(data.coverage(days("2024-01-01", 10)), day, None, "10 days")
    assert whole == (
        "The series is complete: every timestamp expected at this spacing is present."
    )


@pytest.mark.parametrize(
    ("share", "words"),
    [
        (0.004, "Hardly any"),
        (0.03, "A few"),
        (0.12, "Some"),
        (0.31, "A large share"),
        (0.66, "Most"),
    ],
)
def test_the_share_missing_is_worded_without_the_percentage(
    share: float, words: str
) -> None:
    assert data.share_words(share) == words


def test_gaps_are_one_long_one_few_or_many() -> None:
    one = data.coverage(days("2024-01-01", 40, drop=tuple(range(10, 20))))
    assert data.gap_character(one) == "all in one gap"
    mostly = data.coverage(days("2024-01-01", 60, drop=(5, *range(20, 40), 50)))
    assert data.gap_character(mostly) == "mostly in one long gap"
    few = data.coverage(days("2024-01-01", 60, drop=(5, 6, 20, 21, 40, 41)))
    assert data.gap_character(few) == "in a few gaps"
    many = data.coverage(days("2024-01-01", 60, drop=(3, 8, 13, 18, 23, 28, 33)))
    assert data.gap_character(many) == "spread over many short gaps"
    assert data.gap_character(data.coverage(days("2024-01-01", 10))) == ""


def test_summary_is_empty_for_clean_data_and_names_the_problems_otherwise() -> None:
    day = pd.Timedelta(days=1)
    clean = data.coverage(days("2024-01-01", 30))
    assert data.summary_text(clean, day, None, None, None) == ""
    gappy = data.coverage(days("2024-01-01", 30, drop=tuple(range(5, 15))))
    facts = data.ObservedFacts(20, 2, 1, 0, 0, 0, 0)
    assert data.summary_text(gappy, day, None, facts, None) == (
        "33% of the expected timestamps are missing (longest gap 10 days); "
        "2 timestamps repeated; 1 value missing."
    )
    weekends = data.Pattern(
        "weekends", data.WEEKEND_HINT, ", nearly all of them weekend days"
    )
    assert "weekend" in data.summary_text(gappy, day, None, None, weekends)
    one_gap_in_a_thousand = data.coverage(days("2020-01-01", 1000, drop=(500,)))
    assert data.summary_text(one_gap_in_a_thousand, day, None, None, None) == ""


# ---- pure: the gap chart ----------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "size"),
    [(1, 1), (90, 1), (91, 2), (100, 2), (743, 10), (1000, 14), (10**6, 11112)],
)
def test_bucket_size_keeps_bars_few_and_round(expected: int, size: int) -> None:
    assert data.bucket_size(expected) == size
    assert -(-expected // size) <= max(data.MAX_BARS, 1) or size > 5840


def test_buckets_add_up_to_the_series() -> None:
    cov = data.coverage(days("2024-01-01", 250, drop=tuple(range(40, 60))))
    bars = data.buckets(cov)
    assert bars.size == 3
    assert int(bars.observed.sum()) == cov.present
    assert int(bars.possible.sum()) == cov.expected
    assert (bars.observed <= bars.possible).all()
    assert len(bars.observed) <= data.MAX_BARS
    assert bars.possible[-1] == cov.expected - bars.size * (len(bars.observed) - 1)


def test_marked_gaps_keep_the_longest_in_time_order() -> None:
    cov = data.coverage(days("2024-01-01", 60, drop=(5, 10, 11, 20, 21, 22, 40, 41)))
    assert [g.missing for g in data.marked_gaps(cov.gaps, limit=1)] == [3]
    kept = data.marked_gaps(cov.gaps, limit=3)
    assert sorted(g.missing for g in kept) == [2, 2, 3]
    assert [g.slot for g in kept] == sorted(g.slot for g in kept)
    assert len(data.marked_gaps(cov.gaps)) == len(cov.gaps)


def test_equal_gaps_are_picked_from_the_whole_series_not_its_start() -> None:
    dropped = tuple(range(3, 400, 4))  # one missing day every four days
    cov = data.coverage(days("2024-01-01", 400, drop=dropped))
    assert len(cov.gaps) == len(dropped) - 1 and {g.missing for g in cov.gaps} == {1}
    picked = data.marked_gaps(cov.gaps, limit=10)
    assert len(picked) == 10
    slots = [g.slot for g in picked]
    assert slots == sorted(slots)
    assert max(slots) - min(slots) > 0.6 * cov.expected
    assert data.marked_gaps(cov.gaps, limit=10) == picked


# ---- pure: table and CSV ----------------------------------------------------


@pytest.mark.parametrize(
    ("values", "decimals"),
    [
        ([6.0, 11.0], 2),
        ([60.0, 110.0], 1),
        ([600.0, 1100.0], 0),
        ([0.2, 0.7], 3),
        ([5.0, 5.0], 2),
        ([np.nan], 2),
    ],
)
def test_decimals_follow_the_spread(values: list[float], decimals: int) -> None:
    assert data.decimals_for(values) == decimals


def test_values_are_formatted_for_reading() -> None:
    assert data.format_value(1234567.891, 1) == "1,234,567.9"
    assert data.format_value(float("nan"), 2) == ""


def test_shown_rows_are_the_future_or_everything() -> None:
    fcst = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=6), "yhat": range(6)}
    )
    assert len(data.shown_rows(fcst, pd.Timestamp("2024-01-04"))) == 2
    assert len(data.shown_rows(fcst, pd.Timestamp("2024-01-06"))) == 6


def test_table_content_formats_columns_and_marks_numeric_ones() -> None:
    rows = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "yhat": [10.123, 11.5],
            "yhat_lower": [9.0, 10.0],
            "yhat_upper": [11.0, 13.0],
            "trend": [10.0, 10.5],
            "y": [10.0, np.nan],
        }
    )
    headers, cells, numeric = data.table_content(rows, pd.Timedelta(days=1))
    assert headers == ["Date", "Forecast", "Lower", "Upper", "Trend", "Observed"]
    assert numeric == [1, 2, 3, 4, 5]
    assert cells[0] == ["02 Jan 2024", "10.12", "9.00", "11.00", "10.00", "10.00"]
    assert cells[1][-1] == ""
    plain = data.table_content(rows[["ds", "yhat", "trend"]], pd.Timedelta(days=1))
    assert plain[0] == ["Date", "Forecast", "Trend"]


def test_with_observed_adds_y_only_when_something_is_known() -> None:
    rows = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=3), "yhat": [1.0, 2.0, 3.0]}
    )
    known = pd.DataFrame({"ds": [pd.Timestamp("2024-01-02")], "y": [2.5]})
    merged = data.with_observed(rows, known)
    assert merged["y"].isna().tolist() == [True, False, True]
    unknown = pd.DataFrame({"ds": [pd.Timestamp("2030-01-01")], "y": [1.0]})
    assert "y" not in data.with_observed(rows, unknown)


def test_iso_stamps_keep_dates_short_and_add_times_when_needed() -> None:
    whole = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert data.iso_stamps(whole).tolist() == ["2024-01-02", "2024-01-03"]
    timed = pd.Series([pd.Timestamp("2024-01-02 03:04:05"), pd.Timestamp("2024-01-03")])
    assert data.iso_stamps(timed).tolist() == [
        "2024-01-02T03:04:05",
        "2024-01-03T00:00:00",
    ]
    fine = pd.Series(pd.to_datetime(["2024-01-02 03:04:05.250000"]))
    assert data.iso_stamps(fine).tolist() == ["2024-01-02T03:04:05.250000"]


def test_forecast_csv_is_exact_and_iso() -> None:
    yhat = np.array([0.1 + 0.2, 1 / 3, 1e-7])
    fcst = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "trend": [1.0, 2.0, 3.0],
            "yhat_lower": yhat - 1,
            "yhat_upper": yhat + 1,
            "yhat": yhat,
            "other": [9, 9, 9],
        }
    )
    observed = pd.DataFrame({"ds": [pd.Timestamp("2024-01-02")], "y": [0.25]})
    text = data.forecast_csv(fcst, observed)
    lines = text.split("\n")
    assert lines[0] == "ds,yhat,yhat_lower,yhat_upper,trend,y"
    assert lines[1].startswith("2024-01-02,0.30000000000000004,")
    assert lines[1].endswith(",1.0,0.25")
    assert lines[2].endswith(",2.0,")
    assert text.endswith("\n") and "\r" not in text
    parsed = pd.read_csv(io.StringIO(text), float_precision="round_trip")
    assert np.array_equal(parsed["yhat"].to_numpy(), yhat)
    no_interval = data.forecast_csv(fcst.drop(columns=["yhat_lower", "yhat_upper"]))
    assert no_interval.split("\n")[0] == "ds,yhat,trend"


def test_csv_rows_keep_the_future_then_the_latest_history() -> None:
    fcst = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=100), "yhat": range(100)}
    )
    end = pd.Timestamp("2024-03-01")  # 39 rows lie after it
    assert data.csv_rows(fcst, end, limit=100) is fcst
    mixed = data.csv_rows(fcst, end, limit=50)
    assert mixed["yhat"].tolist() == list(range(50, 100))
    assert mixed["ds"].is_monotonic_increasing
    future_only = data.csv_rows(fcst, end, limit=10)
    assert future_only["yhat"].tolist() == list(range(61, 71))


def test_table_note_says_what_the_csv_holds() -> None:
    args = (500, 900, True, "01 Mar 2024")
    whole = data.table_note(*args, csv=1200, all_rows=1200)
    assert whole == (
        "The table lists the dates after the last training date (01 Mar 2024). "
        "It shows the first 500 of 900 rows. The CSV holds all 1,200 rows, "
        "including the dates before the last training date."
    )
    capped = data.table_note(*args, csv=1000, all_rows=1200)
    assert "The CSV is capped at 1,000 of the 1,200 rows" in capped
    only = data.table_note(30, 30, False, "01 Mar 2024", csv=30, all_rows=30)
    assert only.endswith("The CSV holds all 30 rows.")
    assert "every date" in only and "first" not in only


def test_script_escaping_is_reversible_and_hides_every_angle_bracket() -> None:
    nasty = "a</script><!--<script>\\u003c\\\\ \\</x"
    escaped = data.escape_for_script(nasty)
    assert "<" not in escaped
    assert data.unescape_from_script(escaped) == nasty
    assert (
        data.unescape_from_script(data.escape_for_script("1,2\n3,4\n")) == "1,2\n3,4\n"
    )


# ---- the section ------------------------------------------------------------


def context(name: str, **overrides: Any) -> DashboardContext:
    m, df, fcst = fit_case(name)
    return build_context(m, fcst, **{"df": df, **overrides})


def embedded_csv(body: str) -> str:
    match = CSV_BLOCK.search(body)
    assert match is not None
    return data.unescape_from_script(match.group(1))


def assert_well_formed(body: str) -> None:
    assert EM_DASH not in body
    assert body.count('<script type="text/csv"') == 1
    block = CSV_BLOCK.search(body)
    assert block is not None
    assert "<" not in block.group(1)
    assert body.count(">Download CSV</button>") == 1
    assert body.count(">Copy CSV</button>") == 1
    assert 'aria-live="polite"' in body
    assert 'id="forecast-table"' in body


@pytest.mark.parametrize("name", CASES + ["temperature"])
def test_section_survives_every_fixture(name: str) -> None:
    pytest.importorskip("plotly")
    ctx = context(name)
    with no_warnings():
        section = data.build(ctx)
    assert section is not None
    assert (section.key, section.title) == ("data", "Data")
    assert_well_formed(section.body)
    assert len(section.body) < 400_000
    assert "\n" not in section.summary and EM_DASH not in section.summary
    csv = embedded_csv(section.body)
    parsed = pd.read_csv(io.StringIO(csv), float_precision="round_trip")
    assert len(parsed) == len(ctx.fcst)
    assert np.array_equal(parsed["yhat"].to_numpy(), ctx.fcst["yhat"].to_numpy())
    assert ("yhat_lower" in parsed) == ("yhat_lower" in ctx.fcst)
    assert pd.to_datetime(parsed["ds"]).tolist() == ctx.fcst["ds"].tolist()
    rows = re.findall(r"<tr>", section.body.split('id="forecast-table"')[1])
    assert len(rows) - 1 == min(len(ctx.future), data.MAX_TABLE_ROWS)
    assert section.body.count("plotly-graph-div") == (
        1 if ctx.history["ds"].diff().max() > ctx.step * 1.5 else 0
    )


def test_csv_holds_observed_values_where_they_are_known() -> None:
    pytest.importorskip("plotly")
    ctx = context("linear")
    section = data.build(ctx)
    assert section is not None
    parsed = pd.read_csv(io.StringIO(embedded_csv(section.body)))
    known = parsed.dropna(subset=["y"])
    assert len(known) == len(ctx.history)
    assert parsed["y"].isna().sum() == len(ctx.future)


def test_section_reports_the_data_the_caller_passed() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    messy = pd.concat([df, df.iloc[[3]], df.iloc[[9]].assign(y=np.nan)]).reset_index(
        drop=True
    )
    messy = pd.concat(
        [messy, pd.DataFrame({"ds": [pd.Timestamp("2015-01-01")], "y": [0.0]})]
    )
    section = data.build(build_context(m, fcst, df=messy))
    assert section is not None
    text = re.sub(r"<[^>]+>", " ", section.body)
    assert re.search(r"Repeated timestamps\s+2\b", text)
    assert re.search(r"Missing values\s+1\b", text)
    assert re.search(r"Zero values\s+1\b", text)
    assert "1 after the last" in text
    assert "repeated" in section.summary and "missing" in section.summary


def test_section_without_df_says_how_to_check_more() -> None:
    pytest.importorskip("plotly")
    m, _, fcst = fit_case("linear")
    section = data.build(build_context(m, fcst))
    assert section is not None
    assert "Pass the observed data as df" in section.body
    assert "Repeated timestamps" not in section.body


def test_weekday_data_is_flagged_and_then_cleared_by_skipping() -> None:
    pytest.importorskip("plotly")
    m, df, fcst, skip = weekday_case()
    plain = data.build(build_context(m, fcst, df=df))
    assert plain is not None
    assert data.WEEKEND_HINT in re.sub(r"&quot;", '"', plain.body)
    assert "weekend" in plain.summary
    with no_warnings():
        skipped = data.build(build_context(m, fcst, df=df, skip=skip))
    assert skipped is not None
    assert data.WEEKEND_HINT not in skipped.body
    assert "weekdays" in skipped.body
    assert re.search(r"Timestamps missing</dt><dd>\d", skipped.body)
    assert_well_formed(skipped.body)
    assert not re.search(r"weekend", skipped.summary)


def test_complete_business_days_have_no_gaps_once_weekends_are_skipped() -> None:
    pytest.importorskip("plotly")
    m, df, fcst, skip = business_case()
    section = data.build(build_context(m, fcst, df=df, skip=skip))
    assert section is not None
    assert "Nothing is missing" in section.body
    assert "plotly-graph-div" not in section.body
    assert section.summary == ""
    assert "One point per weekday" in section.body


def test_one_missing_business_day_is_a_gap_of_one_weekday() -> None:
    pytest.importorskip("plotly")
    m, df, fcst, skip = business_case(drop=30)
    section = data.build(build_context(m, fcst, df=df, skip=skip))
    assert section is not None
    assert "1 weekday missing between" in section.body
    assert "plotly-graph-div" in section.body
    assert (
        section.summary == ""
    )  # one day in 200 is under the 1% worth lifting to the key findings


def test_future_only_forecast_lists_every_row() -> None:
    pytest.importorskip("plotly")
    ctx = context("linear")
    future_only = ctx.fcst[ctx.fcst["ds"] > ctx.history_end]
    section = data.build(build_context(ctx.m, future_only, df=ctx.df))
    assert section is not None
    assert_well_formed(section.body)
    assert len(embedded_csv(section.body).strip().split("\n")) - 1 == len(future_only)


def test_in_sample_forecast_lists_every_row_with_observed_values() -> None:
    pytest.importorskip("plotly")
    ctx = context("linear")
    in_sample = ctx.fcst[ctx.fcst["ds"] <= ctx.history_end]
    section = data.build(build_context(ctx.m, in_sample, df=ctx.df))
    assert section is not None
    assert "this forecast has none after the training data" in section.body
    shown = section.body.split('id="forecast-table"')[1]
    assert "Observed</th>" in shown
    assert len(re.findall(r"<tr>", shown)) - 1 == data.MAX_TABLE_ROWS
    assert f"first {data.MAX_TABLE_ROWS} of {len(in_sample):,} rows" in section.body


def test_short_history_still_builds() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = short_case()
    with no_warnings():
        section = data.build(build_context(m, fcst, df=df))
    assert section is not None
    assert_well_formed(section.body)


def test_section_is_none_without_training_rows() -> None:
    pytest.importorskip("plotly")
    ctx = context("linear")
    empty = DashboardContext(**{**ctx.__dict__, "history": ctx.history.iloc[0:0]})
    assert data.build(empty) is None


def series_context(ds: pd.Series, skip: Any = None, ahead: int = 5) -> DashboardContext:
    """The linear fixture's context with its history replaced by the timestamps ``ds``."""
    ds = ds.sort_values().reset_index(drop=True)
    step = pd.Timedelta(ds.diff().median())
    future = ds.max() + step * np.arange(1, ahead + 1)
    fcst = pd.DataFrame(
        {
            "ds": pd.DatetimeIndex([*ds, *future]),
            "yhat": 1.0,
            "yhat_lower": 0.5,
            "yhat_upper": 1.5,
            "trend": 1.0,
        }
    )
    breaks = (
        ()
        if skip is None
        else tuple(_plotly_rangebreaks(skip.skip_days, skip.skip_hours))
    )
    return replace(
        context("linear"),
        history=pd.DataFrame({"ds": ds, "y": 1.0}),
        fcst=fcst,
        df=None,
        step=step,
        skip=skip,
        breaks=breaks,
    )


def label_ends(text: str, fmt: str = "%d %b %Y") -> list[pd.Timestamp]:
    return [pd.to_datetime(part, format=fmt) for part in text.split(" to ")]


def test_monthly_gaps_are_counted_and_named_on_the_calendar() -> None:
    pytest.importorskip("plotly")
    index, ds = calendar_series("MS", (*range(50, 74), *range(150, 152)))
    section = data.build(series_context(ds))
    assert section is not None
    text = re.sub(r"<[^>]+>", " ", section.body)
    assert re.search(r"Timestamps expected\s+240\b", text)
    assert re.search(r"Timestamps missing\s+26 \(11%\)", text)
    fmt = lambda i: _fmt_day(index[i], pd.Timedelta(days=31))  # noqa: E731
    assert f"24 months missing between {fmt(49)} and {fmt(74)}" in text
    assert re.search(r"Longest gaps.*\b2 months\b", text, re.S)


def test_complete_monthly_data_has_no_gap_chart() -> None:
    pytest.importorskip("plotly")
    _, ds = calendar_series("MS")
    section = data.build(series_context(ds))
    assert section is not None
    assert (
        "Nothing is missing" in section.body and "plotly-graph-div" not in section.body
    )
    assert section.summary == ""


def test_monthly_gap_chart_puts_every_mark_on_the_dates_of_the_data() -> None:
    pytest.importorskip("plotly")
    index, ds = calendar_series("MS", (*range(100, 120), *range(180, 183)))
    ctx = series_context(ds)
    cov = data.coverage(ds)
    fig = data.gap_figure(ctx, cov)
    bars, triangles = fig.data
    for text in (row[0] for row in bars.customdata):
        first, *rest = label_ends(text)
        assert index[0] <= first <= (rest[0] if rest else first) <= index[-1]
        assert all(t.day == 1 for t in [first, *rest])
    assert bars.customdata[-1][0].endswith(_fmt_day(index[-1], ctx.step))
    assert pd.Timestamp(bars.x[-1]) <= index[-1]
    for gap, x in zip(data.marked_gaps(cov.gaps), triangles.x):
        assert gap.before < pd.Timestamp(x) < gap.after
    shaded = data.marked_gaps(cov.gaps, data.MAX_SHADED_GAPS)
    assert len(shaded) == len(fig.layout.shapes) == 2
    for gap, shape in zip(shaded, fig.layout.shapes):
        assert gap.before < pd.Timestamp(shape.x0) < pd.Timestamp(shape.x1) < gap.after


def test_subdaily_bar_labels_are_the_exact_times() -> None:
    pytest.importorskip("plotly")
    index = pd.date_range("2021-01-01", periods=96 * 7, freq="15min")
    rng = np.random.default_rng(0)
    dropped = set(np.flatnonzero(rng.random(len(index)) < 0.1)) - {0, len(index) - 1}
    ds = pd.Series(index[[i for i in range(len(index)) if i not in dropped]])
    cov = data.coverage(ds)
    assert cov.expected == len(index)
    assert cov.grid.labels(np.arange(len(index))).tolist() == index.tolist()
    ctx = series_context(ds)
    fig = data.gap_figure(ctx, cov)
    bars = data.buckets(cov)
    for i, row in enumerate(fig.data[0].customdata):
        start = i * bars.size
        last = index[start + int(bars.possible[i]) - 1]
        expected = f"{_fmt_day(index[start], ctx.step)} to {_fmt_day(last, ctx.step)}"
        assert row[0] == expected


def test_bar_labels_never_name_a_skipped_hour() -> None:
    pytest.importorskip("plotly")
    index = pd.date_range("2021-01-01", periods=24 * 21, freq="h")
    skip = skip_axis(index, skip_days="weekends", skip_hours=range(0, 8))
    kept = index[skip.keeps(index)]
    rng = np.random.default_rng(2)
    dropped = set(np.flatnonzero(rng.random(len(kept)) < 0.1)) - {0, len(kept) - 1}
    ds = pd.Series(kept[[i for i in range(len(kept)) if i not in dropped]])
    cov = data.coverage(ds, skip)
    assert cov.expected == len(kept)
    assert cov.grid.labels(np.arange(len(kept))).tolist() == kept.tolist()
    fig = data.gap_figure(series_context(ds, skip), cov)
    for row in fig.data[0].customdata:
        stamps = pd.DatetimeIndex(label_ends(row[0], "%d %b %Y %H:%M"))
        assert skip.keeps(stamps).all(), row[0]


def test_uneven_data_gets_no_gap_claims_and_no_chart() -> None:
    pytest.importorskip("plotly")
    rng = np.random.default_rng(1)
    offsets = np.sort(rng.random(300)) * 365
    ds = pd.Series(pd.Timestamp("2022-01-01") + pd.to_timedelta(offsets, unit="D"))
    section = data.build(series_context(ds.dt.floor("s")))
    assert section is not None
    assert "unevenly spaced" in section.body
    assert "Not drawn: the points are not evenly spaced" in section.body
    assert "plotly-graph-div" not in section.body
    assert "Timestamps expected" not in section.body
    assert (
        section.summary == "Timestamps are unevenly spaced, so gaps were not counted."
    )
    assert_well_formed(section.body)


def test_business_hours_data_is_told_to_skip_the_other_hours() -> None:
    pytest.importorskip("plotly")
    section = data.build(series_context(business_hours()))
    assert section is not None
    assert "skip_hours=[*range(0, 9), *range(17, 24)]" in section.body
    assert "outside the hours the data covers" in section.summary
    skip = skip_axis(business_hours(), skip_hours=OFFICE_HOURS)
    with no_warnings():
        closed = data.build(series_context(business_hours(), skip))
    assert closed is not None
    assert "skip_hours=" not in closed.body and closed.summary == ""
    assert "Nothing is missing" in closed.body


@pytest.mark.parametrize("future_only", [False, True])
def test_a_very_long_forecast_is_capped_in_the_csv(future_only: bool) -> None:
    pytest.importorskip("plotly")
    ctx = context("linear")
    inside = len(ctx.history)
    # Either the future fits in the cap with room for some history, or it fills the cap.
    n = data.MAX_CSV_ROWS + (inside + 500 if future_only else inside // 2)
    ds = pd.date_range(ctx.history["ds"].min(), periods=n, freq="D")
    values = np.linspace(0.0, 1.0, n)
    big = pd.DataFrame(
        {
            "ds": ds,
            "yhat": values,
            "yhat_lower": values - 1,
            "yhat_upper": values + 1,
            "trend": values,
        }
    )
    section = data.build(replace(ctx, fcst=big))
    assert section is not None
    assert_well_formed(section.body)
    parsed = pd.read_csv(io.StringIO(embedded_csv(section.body)))
    assert len(parsed) == data.MAX_CSV_ROWS
    dates = pd.to_datetime(parsed["ds"])
    assert dates.is_monotonic_increasing
    if future_only:
        assert dates.iloc[0] > ctx.history_end and dates.iloc[-1] < ds[-1]
    else:
        assert dates.iloc[0] < ctx.history_end
        assert dates.iloc[-1] == ds[-1]
        assert (dates > ctx.history_end).sum() == n - (ds <= ctx.history_end).sum()
    assert f"capped at {data.MAX_CSV_ROWS:,} of the {n:,} rows" in section.body
    assert len(section.body) < 3_000_000
    shown = section.body.split('id="forecast-table"')[1]
    assert len(re.findall(r"<tr>", shown)) - 1 == data.MAX_TABLE_ROWS


def test_timezone_aware_observed_dates_still_build() -> None:
    pytest.importorskip("plotly")
    ctx = context("linear")
    assert ctx.df is not None
    aware = ctx.df.assign(ds=pd.to_datetime(ctx.df["ds"]).dt.tz_localize("UTC"))
    section = data.build(replace(ctx, df=aware))
    assert section is not None
    parsed = pd.read_csv(io.StringIO(embedded_csv(section.body)))
    assert parsed["y"].notna().sum() == len(ctx.history)
    assert data.observed_frame(replace(ctx, df=aware))["ds"].dt.tz is None


def test_gap_chart_marks_a_few_and_shades_many_and_says_so() -> None:
    pytest.importorskip("plotly")
    ctx = context("linear")
    cov = data.coverage(ctx.history["ds"])
    assert data.MAX_MARKED_GAPS < len(cov.gaps) < data.MAX_SHADED_GAPS
    fig = data.gap_figure(ctx, cov)
    bars, triangles = fig.data
    assert len(triangles.x) == data.MAX_MARKED_GAPS
    assert len(fig.layout.shapes) == len(cov.gaps)
    assert len(bars.x) <= data.MAX_BARS
    note = data.gap_chart_note(cov, ctx.step, ctx.skip)
    assert f"{data.MAX_MARKED_GAPS} of the longest of the {len(cov.gaps)} gaps" in note
    assert "Only" not in note


def test_gap_chart_shades_only_the_longest_when_there_are_very_many() -> None:
    pytest.importorskip("plotly")
    ctx = context("linear")
    many = days("2020-01-01", 2000, drop=tuple(range(4, 2000, 5)))
    cov = data.coverage(many)
    assert len(cov.gaps) > data.MAX_SHADED_GAPS
    fig = data.gap_figure(ctx, cov)
    assert len(fig.layout.shapes) == data.MAX_SHADED_GAPS
    assert (
        f"Only the {data.MAX_SHADED_GAPS} longest gaps are shaded."
        in data.gap_chart_note(cov, ctx.step, ctx.skip)
    )


def test_gap_chart_with_few_gaps_invites_hovering() -> None:
    cov = data.coverage(days("2024-01-01", 30, drop=(5, 6, 20)))
    note = data.gap_chart_note(cov, pd.Timedelta(days=1), None)
    assert note.startswith("Each bar is one day and shows the share of expected points")
    assert "Hover a triangle for the dates of a gap." in note


def test_chart_uses_only_palette_colours() -> None:
    pytest.importorskip("plotly")
    from prophet.dashboard import _palette

    ctx = context("linear")
    fig = data.gap_figure(ctx, data.coverage(ctx.history["ds"]))
    allowed = {key.lower() for key in _palette.DARK_SWAP}
    payload = fig.to_plotly_json()
    payload["layout"].pop("template", None)
    used = set(
        re.findall(r"#[0-9a-fA-F]{6}|rgba\([^)]*\)", json.dumps(payload, default=str))
    )
    assert used, "the figure carries colours"
    assert {c.lower().replace(" ", "") for c in used} <= {
        a.replace(" ", "") for a in allowed
    }
    assert all(trace.hovertemplate for trace in fig.data)
    assert len(fig.data) == 2 and fig.layout.showlegend


# ---- the page script, run for real when node is installed --------------------

HARNESS = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const log = { status: [], blobs: [], clipboard: [], links: [] };
const status = { textContent: '' };
const payload = { textContent: input.payload };
let handler = null;
const button = (action) => ({
  getAttribute: () => action,
  closest() { return this; },
});
const root = {
  querySelector: (sel) => (sel === 'script.csv-data' ? payload : sel === '.csv-status' ? status : null),
  getAttribute: () => input.filename,
  addEventListener: (_, fn) => { handler = fn; },
  contains: () => true,
};
const script = { closest: () => root };
global.document = {
  currentScript: script,
  body: { appendChild() {}, removeChild() {} },
  createElement: (tag) => ({ tag, style: {}, setAttribute() {}, click() { log.links.push(this.download); }, select() {} }),
  execCommand: () => input.execCommand,
};
global.Blob = class { constructor(parts) { log.blobs.push(parts.join('')); } };
global.URL = { createObjectURL: () => 'blob:x', revokeObjectURL() {} };
Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: input.clipboard
    ? { clipboard: { writeText: (t) => { log.clipboard.push(t); return input.clipboardFails ? Promise.reject(new Error('no')) : Promise.resolve(); } } }
    : {},
});
eval(input.script);
(async () => {
  handler({ target: button('download') });
  log.status.push(status.textContent);
  handler({ target: button('copy') });
  await new Promise((r) => setTimeout(r, 20));
  log.status.push(status.textContent);
  console.log(JSON.stringify(log));
})();
"""


def run_page_script(csv: str, **options: object) -> dict[str, list[str]]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    section_html = data.csv_tools_html(csv)
    script = re.findall(r"<script>(.*?)</script>", section_html, re.S)[0]
    payload = CSV_BLOCK.search(section_html)
    assert payload is not None
    request = {
        "script": script,
        "payload": payload.group(1),
        "filename": data.CSV_FILENAME,
        "clipboard": True,
        "clipboardFails": False,
        "execCommand": True,
        **options,
    }
    result = subprocess.run(
        [node, "-e", HARNESS],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


def test_page_script_downloads_and_copies_the_exact_csv() -> None:
    csv = "ds,yhat\n2024-01-02,0.30000000000000004\n2024-01-03,<b>\\x</script>\n"
    log = run_page_script(csv)
    assert log["blobs"] == [csv]
    assert log["links"] == ["forecast.csv"]
    assert log["clipboard"] == [csv]
    assert log["status"] == [
        "Downloaded forecast.csv with 2 rows.",
        "Copied 2 rows to the clipboard.",
    ]


def test_page_script_falls_back_when_the_clipboard_is_missing_or_refused() -> None:
    csv = "ds,yhat\n2024-01-02,1.5\n"
    for options in ({"clipboard": False}, {"clipboardFails": True}):
        assert (
            run_page_script(csv, **options)["status"][-1]
            == "Copied 1 row to the clipboard."
        )
    refused = run_page_script(csv, clipboard=False, execCommand=False)
    assert (
        refused["status"][-1]
        == "The browser blocked copying. Try Download CSV instead."
    )


# ---- fixtures ----------------------------------------------------------------


@lru_cache(maxsize=None)
def weekday_case() -> tuple[Prophet, pd.DataFrame, pd.DataFrame, object]:
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"])
    df = df[df["ds"].dt.dayofweek < 5].reset_index(drop=True)
    m = Prophet(uncertainty_samples=100).fit(df)
    future = m.make_future_dataframe(periods=30)
    future = future[future["ds"].dt.dayofweek < 5].reset_index(drop=True)
    fcst = m.predict(future)
    return m, df, fcst, skip_axis(fcst["ds"], skip_days="weekends")


@lru_cache(maxsize=None)
def business_case(
    drop: int | None = None,
) -> tuple[Prophet, pd.DataFrame, pd.DataFrame, object]:
    """Forty weeks of business days, optionally without the day at position ``drop``."""
    ds = pd.bdate_range("2023-01-02", periods=200)
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {"ds": ds, "y": 50 + 0.05 * np.arange(200) + rng.normal(0, 1, 200)}
    )
    if drop is not None:
        df = df.drop(index=drop).reset_index(drop=True)
    m = Prophet(uncertainty_samples=50).fit(df)
    future = m.make_future_dataframe(periods=14)
    future = future[future["ds"].dt.dayofweek < 5].reset_index(drop=True)
    fcst = m.predict(future)
    return m, df, fcst, skip_axis(fcst["ds"], skip_days="weekends")


@lru_cache(maxsize=None)
def short_case() -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"]).head(60)
    m = Prophet(uncertainty_samples=100).fit(df)
    return m, df, m.predict(m.make_future_dataframe(periods=14))
