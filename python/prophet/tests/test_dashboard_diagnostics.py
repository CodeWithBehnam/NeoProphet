"""Tests for the "Model diagnostics" section of the dashboard."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from prophet.dashboard import diagnostics as dg
from prophet.dashboard._context import DashboardContext, build_context
from prophet.plot import skip_axis
from prophet.tests.test_visuals import fit_case, no_warnings

EM_DASH = "\u2014"
DAY = pd.Timedelta(days=1)
HOUR = pd.Timedelta(hours=1)
MINUTE = pd.Timedelta(minutes=1)
FIVE_MINUTES = pd.Timedelta(minutes=5)
FIFTEEN_MINUTES = pd.Timedelta(minutes=15)
WEEKENDS = skip_axis(pd.date_range("2024-01-01", periods=30), skip_days="weekends")


def frame_of(
    y: list[float] | np.ndarray,
    yhat: list[float] | np.ndarray,
    *,
    band: float | None = None,
    start: str = "2024-01-01",
    freq: str = "D",
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ds": pd.date_range(start, periods=len(y), freq=freq),
            "y": np.asarray(y, dtype=float),
            "yhat": np.asarray(yhat, dtype=float),
        }
    )
    if band is not None:
        frame["yhat_lower"] = frame["yhat"] - band
        frame["yhat_upper"] = frame["yhat"] + band
    return frame


# ---- pure: error statistics --------------------------------------------------


def test_error_stats_match_hand_computed_values() -> None:
    # Residuals -2, 2, 0, -4: mean -1, deviations -1, 3, 1, -3.
    stats = dg.error_stats([10, 20, 30, 40], [12, 18, 30, 44])
    assert stats.n == 4
    assert stats.bias == pytest.approx(-1.0)
    assert stats.mae == pytest.approx(2.0)
    assert stats.rmse == pytest.approx(np.sqrt(6.0))
    assert stats.mape == pytest.approx((0.2 + 0.1 + 0.0 + 0.1) / 4)
    assert stats.std == pytest.approx(np.sqrt(5.0))
    assert stats.skewness == pytest.approx(0.0)
    assert stats.excess_kurtosis == pytest.approx(41 / 25 - 3)


def test_skewness_and_kurtosis_of_one_large_miss() -> None:
    # Residuals 0, 0, 0, 0, 10: m2 = 16, m3 = 96, m4 = 832.
    stats = dg.error_stats([0, 0, 0, 0, 10], [0, 0, 0, 0, 0])
    assert stats.skewness == pytest.approx(96 / 16**1.5)
    assert stats.excess_kurtosis == pytest.approx(832 / 16**2 - 3)


def test_mape_is_left_out_when_actuals_include_zero() -> None:
    assert dg.error_stats([0, 10, 20], [1, 9, 21]).mape is None


def test_constant_residuals_have_no_shape() -> None:
    stats = dg.error_stats([5.1, 5.1, 5.1, 5.1], [5.0, 5.0, 5.0, 5.0])
    assert stats.std == pytest.approx(0.0, abs=1e-12)
    assert (stats.skewness, stats.excess_kurtosis) == (0.0, 0.0)


def test_error_stats_rejects_empty_and_mismatched_input() -> None:
    with pytest.raises(ValueError, match="no points"):
        dg.error_stats([], [])
    with pytest.raises(ValueError, match="equal length"):
        dg.error_stats([1.0, 2.0], [1.0])


# ---- pure: coverage ------------------------------------------------------------


def test_interval_coverage_splits_misses_above_and_below() -> None:
    y = np.arange(1, 11)
    cov = dg.interval_coverage(y, [2] * 10, [8] * 10, 0.8)
    assert cov is not None
    assert cov.n == 10
    assert cov.below == pytest.approx(0.1)  # y = 1
    assert cov.above == pytest.approx(0.2)  # y = 9, 10
    assert cov.inside == pytest.approx(0.7)
    assert cov.outside == pytest.approx(0.3)
    assert cov.expected_outside == pytest.approx(0.2)


def test_interval_coverage_ignores_rows_without_an_interval() -> None:
    y = np.arange(1, 5)
    cov = dg.interval_coverage(y, [0, np.nan, 0, 0], [5, 5, np.nan, 5], 0.8)
    assert cov is not None and cov.n == 2
    assert dg.interval_coverage(y, [np.nan] * 4, [np.nan] * 4, 0.8) is None
    with pytest.raises(ValueError, match="equal length"):
        dg.interval_coverage(y, [0, 0], [1, 1], 0.8)


@pytest.mark.parametrize(
    ("n", "above", "below", "verdict"),
    [
        (500, 0.04, 0.03, "wide"),
        (500, 0.20, 0.20, "narrow"),
        (500, 0.10, 0.11, "ok"),
        (20, 0.15, 0.15, "ok"),  # few points: a 10 point gap is within the noise
    ],
)
def test_coverage_verdict(n: int, above: float, below: float, verdict: str) -> None:
    cov = dg.Coverage(n, 1 - above - below, above, below, 0.8)
    assert dg.coverage_verdict(cov) == verdict


def test_effective_count_discounts_positive_correlation_only() -> None:
    assert dg.effective_count(100, 0.0) == 100
    assert dg.effective_count(100, -0.5) == 100  # never credited with more than n
    assert dg.effective_count(510, 0.92) == pytest.approx(510 * 0.08 / 1.92)
    assert dg.effective_count(10, 0.999) == 1.0


def test_interval_coverage_counts_independent_points() -> None:
    y = np.arange(1, 11)
    assert dg.interval_coverage(y, [2] * 10, [8] * 10, 0.8).effective_n == 10  # type: ignore[union-attr]
    cov = dg.interval_coverage(y, [2] * 10, [8] * 10, 0.8, rho1=0.5)
    assert cov is not None
    assert cov.effective_n == pytest.approx(10 / 3)
    assert cov.independent == pytest.approx(10 / 3)


def test_neighbouring_misses_widen_the_tolerance_of_the_verdict() -> None:
    # 25.5% outside an 80% band at 510 points: narrow if the points were unrelated.
    plain = dg.Coverage(510, 0.745, 0.2, 0.055, 0.8)
    assert dg.coverage_verdict(plain) == "narrow"
    # With a lag-1 correlation of 0.92 they are worth about 21 independent points.
    related = dg.Coverage(
        510, 0.745, 0.2, 0.055, 0.8, effective_n=dg.effective_count(510, 0.92)
    )
    assert dg.coverage_verdict(related) == "ok"
    assert "fewer independent checks than points" in dg.coverage_sentence(related)
    assert "fewer independent checks" not in dg.coverage_sentence(plain)


def test_a_clear_verdict_needs_no_caveat_however_related_the_misses() -> None:
    cov = dg.Coverage(20000, 0.98, 0.01, 0.01, 0.8, effective_n=1000.0)
    assert dg.coverage_verdict(cov) == "wide"
    assert "fewer independent checks" not in dg.coverage_sentence(cov)


def test_coverage_sentence_for_a_band_that_is_too_wide() -> None:
    cov = dg.Coverage(500, 0.93, 0.04, 0.03, 0.8)
    text = dg.coverage_sentence(cov)
    assert text.startswith(
        "93% of points fall inside the 80% interval, against 80% expected"
    )
    assert "wider than it needs to be" in text
    assert "Most of the misses" not in text  # 57% above is not lopsided


def test_coverage_sentence_names_the_lopsided_side() -> None:
    cov = dg.Coverage(100, 0.70, 0.28, 0.02, 0.8)
    text = dg.coverage_sentence(cov)
    assert "narrower than it should be" in text
    assert (
        "Most of the misses (93%) are above the band, so the model tends to run low"
        in text
    )
    low = dg.Coverage(100, 0.70, 0.02, 0.28, 0.8)
    assert "below the band, so the model tends to run high" in dg.coverage_sentence(low)
    few = dg.Coverage(100, 0.96, 0.04, 0.0, 0.95)
    assert dg.miss_split_sentence(few) == ""  # four misses are too few to call lopsided


# ---- pure: autocorrelation -------------------------------------------------------


def test_autocorrelation_of_an_alternating_series() -> None:
    acf = dg.autocorrelation([1, -1, 1, -1, 1, -1, 1, -1], 3)
    np.testing.assert_allclose(acf, [-7 / 8, 6 / 8, -5 / 8])


def test_autocorrelation_edge_cases() -> None:
    np.testing.assert_array_equal(dg.autocorrelation([2.0] * 6, 3), np.zeros(3))
    with pytest.raises(ValueError, match="max_lag"):
        dg.autocorrelation([1.0, 2.0, 3.0], 3)
    with pytest.raises(ValueError, match="max_lag"):
        dg.autocorrelation([1.0, 2.0, 3.0], 0)


def test_white_noise_stays_mostly_inside_the_bound() -> None:
    noise = np.random.default_rng(7).normal(size=2000)
    acf = dg.autocorrelation(noise, 40)
    bound = dg.acf_bound(2000)
    assert np.mean(np.abs(acf) > bound) < 0.15
    assert dg.clear_pattern(acf, bound) is None


def test_lags_bound_and_clear_pattern() -> None:
    assert [dg.default_max_lag(n) for n in (6, 40, 100, 1000)] == [1, 10, 25, 40]
    assert dg.acf_bound(100) == pytest.approx(0.196)
    assert dg.clear_pattern(np.array([0.05, 0.35, -0.1]), 0.1) == (2, 0.35)
    assert dg.clear_pattern(np.array([-0.4, 0.2]), 0.1) == (1, -0.4)
    # Beyond the bound but not twice the bound: left to chance.
    assert dg.clear_pattern(np.array([0.15, 0.1]), 0.1) is None
    # Statistically clear on a long series, yet too small to matter.
    assert dg.clear_pattern(np.array([0.08]), 0.02) is None
    assert dg.clear_pattern(np.array([]), 0.1) is None


# ---- pure: naming lags -----------------------------------------------------------


@pytest.mark.parametrize(
    ("lag", "step", "skip", "regular", "expected"),
    [
        (1, DAY, None, True, "one day"),
        (3, DAY, None, True, "3 days"),
        (7, DAY, None, True, "one week"),
        (14, DAY, None, True, "2 weeks"),
        (1, DAY, WEEKENDS, True, "one working day"),
        (2, DAY, WEEKENDS, True, "2 working days"),
        (5, DAY, WEEKENDS, True, "one working week"),
        (10, DAY, WEEKENDS, True, "2 working weeks"),
        (1, HOUR, None, True, "one hour"),
        (6, HOUR, None, True, "6 hours"),
        (24, HOUR, None, True, "one day"),
        (168, HOUR, None, True, "one week"),
        (1, FIVE_MINUTES, None, True, "5 minutes"),
        (12, FIVE_MINUTES, None, True, "one hour"),
        (288, FIVE_MINUTES, None, True, "one day"),
        (1, pd.Timedelta(days=30.4375), None, True, "one month"),
        (12, pd.Timedelta(days=30.4375), None, True, "one year"),
        (1, DAY, None, False, "one observation"),
        (7, DAY, None, False, "7 observations"),
        # A lag that only comes close to a unit is not named after it.
        (23, HOUR, None, True, "23 hours"),
        (25, HOUR, None, True, "25 hours"),
        (49, HOUR, None, True, "49 hours"),
        (60, MINUTE, None, True, "one hour"),
        (61, MINUTE, None, True, "61 minutes"),
        (63, MINUTE, None, True, "63 minutes"),
        (24, FIVE_MINUTES, None, True, "2 hours"),
        (25, FIVE_MINUTES, None, True, "125 minutes"),
        (30, FIVE_MINUTES, None, True, "150 minutes"),
        (36, FIVE_MINUTES, None, True, "3 hours"),
        (8, FIFTEEN_MINUTES, None, True, "2 hours"),
        (9, FIFTEEN_MINUTES, None, True, "135 minutes"),
        # Spacing that is no unit of its own is described by the exact span.
        (7, 2 * DAY, None, True, "2 weeks"),
        (8, 2 * DAY, None, True, "16 days"),
        (9, 2 * DAY, None, True, "18 days"),
        (4, 7 * DAY, None, True, "4 weeks"),
        # Months vary in length, so monthly spacing is counted in months.
        (2, pd.Timedelta(days=30.4375), None, True, "2 months"),
        (3, pd.Timedelta(days=30.4375), None, True, "3 months"),
        (24, pd.Timedelta(days=30.4375), None, True, "2 years"),
        (1, 31 * DAY, None, True, "one month"),
        (2, 31 * DAY, None, True, "2 months"),
        (13, 31 * DAY, None, True, "13 months"),
        (12, 31 * DAY, None, True, "one year"),
        (1, 91 * DAY, None, True, "one quarter"),
        (3, 91 * DAY, None, True, "3 quarters"),
        (4, 91 * DAY, None, True, "one year"),
        # A spacing near an hour but not exactly one is still counted in hours.
        (24, pd.Timedelta(minutes=62), None, True, "24 hours"),
        # A spacing in no whole unit falls back to observations.
        (3, pd.Timedelta(seconds=450), None, True, "3 observations"),
    ],
)
def test_lag_phrase(
    lag: int, step: pd.Timedelta, skip, regular: bool, expected: str
) -> None:
    assert dg.lag_phrase(lag, step, skip, regular) == expected


def test_lag_phrase_closes_skipped_hours_and_days() -> None:
    hours = pd.date_range("2024-01-01", periods=24 * 14, freq="h")
    skip = skip_axis(hours, skip_days="weekends", skip_hours=range(0, 8))
    assert dg.steps_per_day(HOUR, skip) == 16
    assert dg.steps_per_week(HOUR, skip) == 80
    assert dg.lag_phrase(16, HOUR, skip) == "one working day"
    assert dg.lag_phrase(80, HOUR, skip) == "one working week"


def test_lag_phrase_rejects_lag_zero() -> None:
    with pytest.raises(ValueError, match="at least one"):
        dg.lag_phrase(0, DAY)


def test_regular_spacing_follows_skipped_days() -> None:
    days = pd.Series(pd.date_range("2024-01-01", periods=60))
    weekdays = days[days.dt.dayofweek < 5].reset_index(drop=True)
    assert dg.regular_spacing(days, DAY)
    assert not dg.regular_spacing(weekdays, DAY)
    assert dg.regular_spacing(weekdays, DAY, WEEKENDS)
    assert dg.regular_spacing(days.head(2), DAY)


# ---- pure: windows, thinning, histogram -------------------------------------------


def test_rolling_window_is_a_week_of_days_or_a_day_of_points() -> None:
    assert dg.rolling_window(500, DAY) == 7
    assert dg.rolling_window(500, DAY, WEEKENDS) == 5
    assert dg.rolling_window(5000, HOUR) == 24
    assert dg.rolling_window(5000, FIVE_MINUTES) == 288
    assert dg.rolling_window(863, FIVE_MINUTES) == 172  # a fifth of the series
    assert dg.rolling_window(20, DAY) == 4
    assert dg.rolling_window(500, pd.Timedelta(days=7)) == 3


def test_rolling_mean_is_centred_and_uses_the_window_it_has_at_the_ends() -> None:
    out = dg.rolling_mean(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    np.testing.assert_allclose(out, [1.5, 2.0, 3.0, 4.0, 4.5])


def test_rolling_mean_can_leave_out_the_partial_windows_at_the_ends() -> None:
    out = dg.rolling_mean(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3, partial=False)
    np.testing.assert_allclose(out, [np.nan, 2.0, 3.0, 4.0, np.nan])


def test_thin_indices_caps_rows_and_keeps_the_flagged_ones() -> None:
    assert list(dg.thin_indices(5, 10)) == [0, 1, 2, 3, 4]
    keep = np.zeros(1000, dtype=bool)
    keep[[3, 500, 997]] = True
    idx = dg.thin_indices(1000, 100, keep=keep)
    assert len(idx) == 100
    assert len(set(idx)) == 100
    assert list(idx) == sorted(idx)
    assert {3, 500, 997} <= set(idx)
    many = dg.thin_indices(1000, 50, keep=np.ones(1000, dtype=bool))
    assert len(many) == 50
    unflagged = dg.thin_indices(1000, 50, keep=np.zeros(1000, dtype=bool))
    assert len(unflagged) == 50 and list(unflagged) == sorted(unflagged)


@pytest.mark.parametrize("rows", [8760, 12500, 20000, 50000])
def test_thinned_scatter_keeps_both_colours_in_their_true_proportion(rows: int) -> None:
    keep = np.zeros(rows, dtype=bool)
    keep[np.random.default_rng(1).choice(rows, rows // 5, replace=False)] = True
    idx = dg.thin_indices(rows, 2500, keep=keep)
    assert len(idx) <= 2500 and len(set(idx)) == len(idx)
    drawn = keep[idx]
    assert drawn.any() and not drawn.all()  # no colour disappears
    assert drawn.mean() == pytest.approx(0.2, abs=0.02)


def test_thinned_scatter_keeps_every_rare_extreme_miss() -> None:
    rows = 50000
    keep = np.zeros(rows, dtype=bool)
    rare = np.random.default_rng(2).choice(rows, dg.FLAGGED_FLOOR - 20, replace=False)
    keep[rare] = True
    idx = dg.thin_indices(rows, 2500, keep=keep)
    assert set(rare) <= set(idx)
    assert len(idx) == 2500


def test_chance_ceiling_rises_with_the_number_of_comparisons() -> None:
    assert dg.chance_ceiling(1, 3.0) == 3.0
    assert dg.chance_ceiling(0, 3.5) == 3.5
    assert dg.chance_ceiling(168, 3.5) == pytest.approx(np.sqrt(2 * np.log(168)) + 1)
    assert dg.chance_ceiling(1000, 3.0) > dg.chance_ceiling(100, 3.0) > 3.0


def test_residual_histogram_counts_every_point() -> None:
    resid = np.random.default_rng(1).normal(size=400)
    hist = dg.residual_histogram(resid)
    assert hist.counts.sum() == 400
    assert len(hist.edges) == len(hist.counts) + 1
    assert 8 <= len(hist.counts) <= 60
    flat = dg.residual_histogram(np.zeros(20))
    assert flat.counts.sum() == 20


def test_normal_counts_peak_and_total() -> None:
    x = np.linspace(-6, 6, 4001)
    width = 12 / 4000
    curve = dg.normal_counts(x, 0.0, 1.0, 500, width)
    assert curve.max() == pytest.approx(500 * width / np.sqrt(2 * np.pi))
    assert curve.sum() == pytest.approx(500.0, rel=1e-3)
    with pytest.raises(ValueError, match="positive"):
        dg.normal_counts(x, 0.0, 0.0, 10, 1.0)


def test_spread_by_level_sees_misses_that_grow_with_the_level() -> None:
    level = np.linspace(10, 100, 200)
    sign = np.where(np.arange(200) % 2 == 0, 1.0, -1.0)
    grows = dg.spread_by_level(level, 0.1 * level * sign)
    assert grows is not None
    ratio = dg.spread_ratio(grows[1])
    assert ratio is not None and 5 < ratio < 12
    steady = dg.spread_by_level(level, sign)
    assert steady is not None
    assert dg.spread_ratio(steady[1]) == pytest.approx(1.0)
    assert dg.spread_by_level(level[:39], sign[:39]) is None
    assert dg.spread_ratio(np.array([0.0, 1.0])) is None


# ---- pure: the largest misses ---------------------------------------------------------


def test_largest_misses_are_ordered_by_size_then_date() -> None:
    # Misses (observed minus fitted): day 2 is -4, day 4 is +3, days 6 and 12 are +4.
    yhat = [10, 14, 10, 7, 10, 6, 10, 10, 10, 10, 10, 6]
    frame = frame_of([10.0] * 12, yhat, band=2.5)
    top = dg.largest_misses(frame, 4)
    assert [m.ds.day for m in top] == [2, 6, 12, 4]  # ties keep the earlier date first
    assert [m.residual for m in top] == [-4.0, 4.0, 4.0, 3.0]
    assert [m.fitted for m in top] == [14.0, 6.0, 6.0, 7.0]
    assert [m.observed for m in top] == [10.0] * 4
    assert [m.position for m in top] == ["below", "above", "above", "above"]


def test_a_miss_inside_the_band_is_marked_inside() -> None:
    top = dg.largest_misses(
        frame_of([10.0, 10.0, 10.0], [10.0, 12.0, 10.0], band=3.0), 1
    )
    assert top[0].residual == -2.0 and top[0].position == "inside"


def test_largest_misses_without_an_interval_have_no_position() -> None:
    top = dg.largest_misses(frame_of([1, 5, 3], [1, 1, 1]), 2)
    assert [m.position for m in top] == [None, None]
    assert [m.residual for m in top] == [4.0, 2.0]


# ---- pure: the calendar heatmap ---------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "days", "expected"),
    [
        (FIVE_MINUTES, 20, "weekday_hour"),
        (HOUR, 14, "weekday_hour"),
        (HOUR, 13, None),
        (FIVE_MINUTES, 3, None),
        (DAY, 100, "month_weekday"),
        (DAY, 59, None),
        (pd.Timedelta(days=7), 700, None),
    ],
)
def test_heatmap_kind(step: pd.Timedelta, days: int, expected: str | None) -> None:
    assert dg.heatmap_kind(step, pd.Timedelta(days=days)) == expected


def hourly_frame(days: int = 28) -> pd.DataFrame:
    return pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=24 * days, freq="h")}
    )


def test_weekday_hour_grid_finds_the_cell_that_misses() -> None:
    frame = hourly_frame()
    noise = np.random.default_rng(3).normal(scale=1.0, size=len(frame))
    monday_nine = (frame["ds"].dt.dayofweek == 0) & (frame["ds"].dt.hour == 9)
    resid = noise + np.where(monday_nine, 6.0, 0.0)
    grid = dg.mean_grid(frame["ds"], resid, "weekday_hour")
    assert grid is not None
    assert grid.rows == tuple(range(7)) and grid.cols == tuple(range(24))
    assert grid.mean.shape == (7, 24)
    assert np.all(grid.count == 4)
    assert grid.scale.shape == grid.mean.shape
    assert np.all((grid.scale > 0.5) & (grid.scale < 1.5))  # about the noise of 1
    best = dg.strongest_cell(grid)
    assert best is not None and best[:2] == (0, 9) and best[2] > dg.standout_bar(grid)
    text = dg.grid_sentence(grid)
    assert "on Mondays at 09:00" in text
    assert "above the fit" in text and "(4 points)" in text


def test_grid_without_a_systematic_miss_says_so() -> None:
    frame = hourly_frame()
    resid = np.random.default_rng(5).normal(size=len(frame))
    grid = dg.mean_grid(frame["ds"], resid, "weekday_hour")
    assert grid is not None
    assert dg.grid_sentence(grid) == (
        "No single weekday and hour combination stands out from the rest of the grid."
    )
    flat = dg.mean_grid(frame["ds"], np.zeros(len(frame)), "weekday_hour")
    assert flat is not None and dg.strongest_cell(flat) is None
    assert dg.grid_sentence(flat).startswith("No single weekday and hour")


def ar_noise(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(size=n)
    out = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = phi * acc + eps[i]
        out[i] = acc
    return out


def share_flagged(ds: pd.Series, kind: dg.GridKind, make, seeds: int = 20) -> float:
    """Share of seeded draws of calendar-free misses for which a cell is called out."""
    hits = 0
    for seed in range(seeds):
        resid = make(np.random.default_rng(seed))
        grid = dg.mean_grid(ds, resid, kind)
        assert grid is not None
        hits += not dg.grid_sentence(grid).startswith("No single")
    return hits / seeds


def test_misses_that_follow_each_other_are_not_a_calendar_effect() -> None:
    # Drifting misses (autocorrelated) used to be reported as a systematic cell in
    # a third to all of the draws, since each point was counted as independent.
    daily = pd.Series(pd.date_range("2022-01-01", periods=3 * 365, freq="D"))
    assert (
        share_flagged(daily, "month_weekday", lambda r: ar_noise(len(daily), 0.9, r))
        <= 0.1
    )
    five = pd.Series(pd.date_range("2024-01-01", periods=288 * 28, freq="5min"))
    assert (
        share_flagged(five, "weekday_hour", lambda r: ar_noise(len(five), 0.95, r))
        <= 0.1
    )


def test_noise_that_grows_with_the_level_is_not_a_calendar_effect() -> None:
    hourly = hourly_frame(42)["ds"]
    level = 100 + 50 * np.sin(2 * np.pi * (hourly.dt.hour.to_numpy() - 6) / 24)
    assert (
        share_flagged(
            hourly, "weekday_hour", lambda r: 0.1 * level * r.normal(size=len(hourly))
        )
        <= 0.1
    )


def test_misses_with_no_pattern_are_rarely_called_out() -> None:
    hourly = hourly_frame(42)["ds"]
    assert (
        share_flagged(hourly, "weekday_hour", lambda r: r.normal(size=len(hourly)))
        <= 0.1
    )


def test_a_cell_is_judged_against_the_noise_of_its_own_hour() -> None:
    frame = hourly_frame(42)
    level = 100 + 50 * np.sin(2 * np.pi * (frame["ds"].dt.hour.to_numpy() - 6) / 24)
    resid = 0.1 * level * np.random.default_rng(8).normal(size=len(frame))
    grid = dg.mean_grid(frame["ds"], resid, "weekday_hour")
    assert grid is not None
    # The level peaks at noon (150, so a noise of 15) and bottoms out at midnight (50, so 5).
    assert grid.scale[0, 12] == pytest.approx(15.0, rel=0.2)
    assert grid.scale[0, 0] == pytest.approx(5.0, rel=0.2)
    assert np.all(grid.scale[:, 12] == grid.scale[0, 12])  # pooled along the hour


def test_a_whole_weekday_that_misses_is_named_rather_than_one_cell() -> None:
    ds = pd.Series(pd.date_range("2022-01-01", "2023-12-31", freq="D"))
    ds = ds[ds.dt.dayofweek < 5].reset_index(drop=True)
    noise = np.random.default_rng(6).normal(scale=1.0, size=len(ds))
    resid = noise - np.where(ds.dt.dayofweek == 4, 6.0, 0.0)
    grid = dg.mean_grid(ds, resid, "month_weekday")
    assert grid is not None
    best = dg.strongest_cell(grid)
    assert best is not None and grid.cols[best[1]] == 4  # a Friday cell stands out ...
    effect = dg.line_effect(grid, best[0], best[1])
    assert effect is not None and effect.axis == "col" and effect.mean < -5
    text = dg.grid_sentence(grid)  # ... but the sentence is about all Fridays
    assert text.startswith("The clearest systematic miss is on Fridays, where")
    assert "below the fit" in text and "Fridays in" not in text


def test_a_cell_that_is_alone_keeps_its_own_sentence() -> None:
    grid = dg.MeanGrid(
        kind="weekday_hour",
        rows=(0, 1, 2),
        cols=(8, 9, 10, 11),
        mean=np.array(
            [[0.1, 6.0, -0.1, 0.0], [0.0, 0.2, 0.1, -0.2], [0.1, -0.1, 0.0, 0.1]]
        ),
        count=np.full((3, 4), 5.0),
        scale=np.ones((3, 4)),
    )
    assert dg.line_effect(grid, 0, 1) is None
    assert "on Mondays at 09:00" in dg.grid_sentence(grid)


def test_an_hour_or_a_month_can_be_the_line() -> None:
    hours = dg.MeanGrid(
        kind="weekday_hour",
        rows=(0, 1, 2, 3),
        cols=(8, 9, 10),
        mean=np.array(
            [[0.0, 5.0, 0.1], [0.1, 5.5, 0.0], [0.0, 4.5, -0.1], [-0.1, 6.0, 0.0]]
        ),
        count=np.full((4, 3), 5.0),
        scale=np.ones((4, 3)),
    )
    assert "around 09:00" in dg.grid_sentence(hours)
    months = dg.MeanGrid(
        kind="month_weekday",
        rows=(3, 4, 5, 6),
        cols=(0, 1, 2, 3),
        mean=np.array(
            [
                [0.0, 0.1, 0.0, -0.1],
                [-5.0, -5.5, -4.5, -6.0],
                [0.1, 0.0, -0.1, 0.0],
                [0.0, 0.1, 0.0, 0.1],
            ]
        ),
        count=np.full((4, 4), 5.0),
        scale=np.ones((4, 4)),
    )
    text = dg.grid_sentence(months)  # the second row is month 4
    assert "miss is in April, where" in text and "below the fit" in text


def test_grid_leaves_out_weekdays_and_hours_that_never_occur() -> None:
    frame = hourly_frame()
    keep = skip_axis(frame["ds"], skip_days="weekends", skip_hours=range(0, 8)).keeps(
        frame["ds"]
    )
    kept = frame.loc[keep].reset_index(drop=True)
    resid = np.random.default_rng(2).normal(size=len(kept))
    grid = dg.mean_grid(kept["ds"], resid, "weekday_hour")
    assert grid is not None
    assert grid.rows == (0, 1, 2, 3, 4)
    assert grid.cols == tuple(range(8, 24))


def test_month_weekday_grid_names_the_month_and_weekday() -> None:
    ds = pd.Series(pd.date_range("2022-01-01", "2023-12-31", freq="D"))
    ds = ds[ds.dt.dayofweek < 5].reset_index(drop=True)
    resid = np.random.default_rng(4).normal(scale=0.5, size=len(ds))
    tuesday_march = (ds.dt.dayofweek == 1) & (ds.dt.month == 3)
    resid = resid - np.where(tuesday_march, 4.0, 0.0)
    grid = dg.mean_grid(ds, resid, "month_weekday")
    assert grid is not None
    assert grid.rows == tuple(range(1, 13)) and grid.cols == (0, 1, 2, 3, 4)
    text = dg.grid_sentence(grid)
    assert "on Tuesdays in March" in text and "below the fit" in text


def test_grid_needs_enough_points_per_cell() -> None:
    ds = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    assert dg.mean_grid(ds, np.zeros(10), "month_weekday") is None


# ---- pure: diagnosis and wording -----------------------------------------------------------


def ar_series(n: int = 300, phi: float = 0.8, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.normal()
    return out


def test_diagnose_flags_a_left_over_pattern() -> None:
    resid = ar_series()
    frame = frame_of(100 + resid, [100.0] * len(resid), band=2.0)
    diag = dg.diagnose(frame, 0.8)
    assert diag.pattern is not None and diag.pattern[0] == 1 and diag.pattern[1] > 0.6
    assert diag.coverage is not None and diag.coverage.nominal == 0.8
    text = dg.lede_text(diag, "one day")
    assert "Neighbouring misses move together" in text
    assert "so some pattern is left over" in text
    # The lag stays in the section; the key finding says only that something is left.
    assert dg.summary_text(diag).endswith("; some pattern is left in the misses.")
    assert "lag" not in dg.summary_text(diag)


def test_neighbouring_misses_are_said_once_in_the_lede() -> None:
    from dataclasses import replace

    diag = dg.diagnose(frame_of(100 + ar_series(), [100.0] * 300, band=2.0), 0.8)
    cov = dg.Coverage(300, 0.8, 0.1, 0.1, 0.8, effective_n=30.0)
    assert dg.rough_verdict(cov)
    # A pattern at a lag of one carries the caveat on the band in its own sentence.
    merged = dg.lede_text(replace(diag, coverage=cov, pattern=(1, 0.9)), "one day")
    assert merged.count("Neighbouring misses move together") == 1
    assert (
        "so some pattern is left over, and there are fewer independent checks than "
        "points, so the verdict on the band is a rough one." in merged
    )
    # Any other lag leaves the caveat where it was, and the two sentences differ.
    apart = dg.lede_text(replace(diag, coverage=cov, pattern=(7, 0.3)), "one week")
    assert apart.count("Neighbouring misses move together") == 1
    assert "this verdict is a rough one" in apart
    assert "resemble the miss from one week earlier" in apart
    # With no pattern the caveat stands alone; with a clear verdict there is none.
    alone = dg.lede_text(replace(diag, coverage=cov, pattern=None), None)
    assert "this verdict is a rough one" in alone
    clear = dg.Coverage(300, 0.97, 0.01, 0.02, 0.8, effective_n=30.0)
    assert not dg.rough_verdict(clear)
    assert "rough one" not in dg.lede_text(replace(diag, coverage=clear), "one day")
    assert "fewer independent" not in dg.pattern_sentence(1, 0.9, 0.09, "one day")


def test_diagnose_counts_independent_points_for_the_interval() -> None:
    related = dg.diagnose(frame_of(100 + ar_series(), [100.0] * 300, band=2.0), 0.8)
    assert related.coverage is not None
    assert related.coverage.effective_n is not None
    assert related.coverage.effective_n < 0.25 * related.coverage.n
    noise = np.random.default_rng(21).normal(size=300)
    unrelated = dg.diagnose(frame_of(50 + noise, [50.0] * 300, band=1.0), 0.8)
    assert unrelated.coverage is not None
    assert unrelated.coverage.independent > 0.8 * unrelated.coverage.n


def test_diagnose_stays_quiet_about_noise() -> None:
    resid = np.random.default_rng(21).normal(size=300)
    diag = dg.diagnose(frame_of(50 + resid, [50.0] * 300), None)
    assert diag.pattern is None
    assert diag.coverage is None
    text = dg.lede_text(diag)
    assert "pattern" not in text and "interval" not in text
    assert text.startswith("On the training data the average miss is ")
    assert dg.summary_text(diag) == (
        f"The fit misses by {dg._fmt_number(diag.stats.mae)} on average."
    )


def test_pattern_sentence_reads_each_case() -> None:
    assert dg.pattern_sentence(1, -0.4, 0.09, "one day").startswith(
        "Neighbouring misses tend to alternate in sign (correlation -0.40"
    )
    assert dg.pattern_sentence(7, 0.3, 0.09, "one week").startswith(
        "Each miss tends to resemble the miss from one week earlier (correlation +0.30"
    )
    assert dg.pattern_sentence(5, -0.3, 0.09, "5 observations").startswith(
        "Each miss tends to have the opposite sign to the miss from 5 observations earlier"
    )
    assert "against about ±0.09 for pure noise" in dg.pattern_sentence(
        7, 0.3, 0.09, "x"
    )


def test_accuracy_and_bias_sentences() -> None:
    stats = dg.error_stats([10, 20, 30, 40], [12, 18, 30, 44])
    assert dg.accuracy_sentence(stats) == (
        "On the training data the average miss is 2 (MAE) and the typical error "
        "is 10% of the observed value (MAPE)."
    )
    assert dg.accuracy_sentence(
        dg.error_stats([0, 20, 30, 40], [1, 18, 30, 44])
    ).endswith("(MAE).")  # a zero among the actuals leaves no percentage
    assert dg.bias_sentence(stats) == "On average the fit runs high by 1."
    even = dg.error_stats([1, 2, 3, 4], [1.5, 1.5, 3.5, 3.5])
    assert dg.bias_sentence(even) == ""
    low = dg.error_stats([5, 6, 7], [4, 5, 6])
    assert dg.bias_sentence(low) == "On average the fit runs low by 1."


def test_spread_sentence_and_lede_hint() -> None:
    assert "grow with the level" in dg.spread_sentence(2.5)
    assert "about 2.5 times" in dg.spread_sentence(2.5)
    assert "shrink as the level rises" in dg.spread_sentence(0.4)
    assert "similar across" in dg.spread_sentence(1.1)
    assert dg.spread_sentence(None) == ""
    assert dg.spread_sentence(0.0) == ""
    level = np.linspace(10, 100, 200)
    sign = np.where(np.arange(200) % 2 == 0, 1.0, -1.0)
    frame = frame_of(level + 0.1 * level * sign, level)
    diag = dg.diagnose(frame, None)
    assert "Multiplicative seasonality may suit" in dg.lede_text(diag, additive=True)
    assert "Multiplicative seasonality" not in dg.lede_text(diag, additive=False)


def test_shape_sentence() -> None:
    assert "roughly symmetric" in dg.shape_sentence(0.1, 0.2)
    assert "tails close to a normal curve" in dg.shape_sentence(0.1, 0.2)
    assert "skewed towards points above the fit" in dg.shape_sentence(0.8, 0.0)
    assert "skewed towards points below the fit" in dg.shape_sentence(-0.8, 0.0)
    assert "heavier tails" in dg.shape_sentence(0.0, 2.3)
    assert "lighter tails" in dg.shape_sentence(0.0, -1.4)
    assert "(skewness +0.1, excess kurtosis +2.3)" in dg.shape_sentence(0.1, 2.3)


def test_values_that_round_to_zero_print_without_a_sign() -> None:
    assert "(skewness 0.0, excess kurtosis 0.0)" in dg.shape_sentence(-0.04, -0.02)
    assert "(skewness 0.0, excess kurtosis 0.0)" in dg.shape_sentence(0.0, 0.0)
    assert "(skewness -0.5, excess kurtosis +0.1)" in dg.shape_sentence(-0.5, 0.1)


def test_a_bias_that_is_rounding_noise_prints_as_plain_zero() -> None:
    assert dg.signed_number(-4.74e-06, 2.0) == "0"
    assert dg.signed_number(7.38e-05, 0.5) == "0"
    assert dg.signed_number(0.0, 0.0) == "0"
    assert dg.signed_number(-0.5, 2.0) == "-0.5"
    assert dg.signed_number(0.02, 2.0) == "0"  # exactly 1% of the MAE still counts as 0
    assert dg.signed_number(0.03, 2.0) == "+0.03"


def test_the_bias_tile_shows_zero_for_a_fit_that_averages_out() -> None:
    def bias(y: list[float]) -> str:
        tiles = dg._tiles(dg.diagnose(frame_of(y, [10.0] * len(y)), None))
        return next(t.value for t in tiles if t.label == "BIAS")

    assert bias([10.0, 12.0, 8.0, 11.0, 9.0]) == "0"
    assert bias([11.0, 13.0, 9.0]) == "+1"


def test_acf_note_counts_lags_beyond_chance() -> None:
    acf = np.array([0.5, 0.01, -0.3, 0.02] + [0.0] * 36)
    text = dg.acf_note(acf, 0.1)
    assert "2 of 40 lags are beyond them, against about 2 by chance alone" in text


def noisy_with(
    n: int, bump: dict[int, float], seed: int = 3, scale: float = 1.0
) -> np.ndarray:
    resid = np.random.default_rng(seed).normal(scale=scale, size=n)
    for i, v in bump.items():
        resid[i] += v
    return resid


def test_drift_sentence_names_a_stretch_that_runs_away_from_zero() -> None:
    n, window = 120, 7
    ds = pd.Series(pd.date_range("2024-01-01", periods=n))
    resid = noisy_with(n, {i: 4.0 for i in range(60, 67)})  # a week, centred on day 64
    hump = dg.drift_sentence(ds, resid, window, DAY)
    assert "strays furthest from zero around" in hump
    assert "Mar 2024" in hump and "runs low" in hump and "(+" in hump
    dip = dg.drift_sentence(ds, -resid, window, DAY)
    assert "runs high" in dip and "(-" in dip


def test_drift_sentence_stays_calm_for_noise() -> None:
    # Pure noise has a largest rolling average too; it must not be called drift.
    # The old threshold (half the MAE) fired for every noise series.
    calm = 0
    for seed in range(60):
        n = 200
        ds = pd.Series(pd.date_range("2024-01-01", periods=n))
        window = dg.rolling_window(n, DAY)
        s = dg.drift_sentence(ds, noisy_with(n, {}, seed=seed), window, DAY)
        calm += "stays close to zero" in s
    assert calm >= 58  # about one draw in a hundred may still be called drift


def test_drift_ignores_the_noisy_partial_windows_at_the_ends() -> None:
    n, window = 100, 7
    ds = pd.Series(pd.date_range("2024-01-01", periods=n))
    # A spike on the very first day. The partial window at the start averages it over
    # four points and used to win; only windows of seven points are compared now.
    resid = noisy_with(n, {0: 8.0})
    s = dg.drift_sentence(ds, resid, window, DAY)
    assert "01 Jan 2024" not in s and "02 Jan 2024" not in s and "03 Jan 2024" not in s


def test_drift_sentence_is_silent_without_anything_to_compare() -> None:
    ds = pd.Series(pd.date_range("2024-01-01", periods=6))
    assert dg.drift_sentence(ds, np.zeros(6), 3, DAY) == ""  # no scatter
    assert dg.drift_sentence(ds, np.arange(6.0), 7, DAY) == ""  # no full window


def test_drift_bar_rises_with_the_length_of_the_series() -> None:
    assert dg.drift_bar(3) == dg.DRIFT_Z
    assert dg.drift_bar(5000) > dg.drift_bar(100) > dg.DRIFT_Z


def test_notes_describe_only_what_is_drawn() -> None:
    level = np.linspace(10, 100, 200)
    sign = np.where(np.arange(200) % 2 == 0, 1.0, -1.0)
    long = dg.diagnose(frame_of(level + 0.1 * level * sign, level), None)
    assert "dark lines" in dg.level_note(long)
    short = dg.diagnose(frame_of(level[:25] + sign[:25], level[:25]), None)
    assert short.spread is None
    note = dg.level_note(short)
    assert "dark lines" not in note and "too few points" in note
    varying = dg.error_stats([1.0, 2.0, 4.0, 3.0], [1.5, 1.5, 3.0, 3.5])
    assert "red curve" in dg.histogram_note(varying)
    perfect = dg.error_stats([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert perfect.std == 0
    assert "red curve" not in dg.histogram_note(perfect)


def test_no_wording_contains_an_em_dash() -> None:
    resid = ar_series()
    diag = dg.diagnose(frame_of(100 + resid, [100.0] * len(resid), band=2.0), 0.8)
    texts = [
        dg.lede_text(diag, "one day", additive=True),
        dg.summary_text(diag),
        dg.acf_note(diag.acf, diag.bound),
        dg.shape_sentence(diag.stats.skewness, diag.stats.excess_kurtosis),
        dg.spread_sentence(2.0),
        dg.spread_sentence(0.3),
    ]
    assert not any(EM_DASH in t for t in texts)


# ---- the section ---------------------------------------------------------------------------

CASES = [
    "linear",
    "logistic",
    "flat",
    "multiplicative",
    "subdaily",
    "no_uncertainty",
    "regressors",
    "temperature",
    "no_regressors_custom_seasonality",
]


def context(name: str, **kwargs) -> DashboardContext:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case(name)
    return build_context(m, fcst, df=df, **kwargs)


def build_section(ctx: DashboardContext) -> dg.Section:
    with no_warnings():
        section = dg.build(ctx)
    assert section is not None
    return section


@pytest.mark.parametrize("name", CASES)
def test_section_builds_for_every_fixture(name: str) -> None:
    section = build_section(context(name))
    assert (section.key, section.title) == ("diagnostics", "Model diagnostics")
    assert section.body.count("Plotly.newPlot") >= 4
    assert EM_DASH not in section.body and EM_DASH not in section.summary
    assert len(section.body) < 1_000_000
    assert section.summary.startswith("The fit misses by ")
    assert section.summary.endswith(".") and "\n" not in section.summary
    assert len(section.summary.split()) <= 25 and "lag" not in section.summary
    assert "largest-misses" in section.body
    assert "has already seen" in section.body
    banded = name != "no_uncertainty"
    assert ("INSIDE THE BAND" in section.body) is banded
    assert ('<th class="num">80% interval</th>' in section.body) is banded
    assert ("80% band is " in section.summary) is banded


def test_calendar_heatmap_follows_the_data() -> None:
    daily = build_section(context("linear"))
    assert "Misses on the calendar" in daily.body
    assert '"Mon","Tue","Wed","Thu","Fri"' in daily.body.replace(" ", "")
    assert (
        '"Sat"' not in daily.body
    )  # the data have no weekend, so neither does the chart
    short = build_section(context("subdaily"))  # three days of 5-minute data
    assert "Misses on the calendar" not in short.body
    assert "Misses against the fitted level" in short.body


def test_section_is_none_without_in_sample_rows() -> None:
    ctx = context("linear")
    assert dg.build(dataclasses.replace(ctx, fcst=ctx.future)) is None


def test_section_works_with_in_sample_rows_only() -> None:
    ctx = context("linear")
    past = ctx.fcst[ctx.fcst["ds"] <= ctx.history_end]
    assert len(past) == len(ctx.history)
    section = build_section(dataclasses.replace(ctx, fcst=past))
    assert section.body.count("Plotly.newPlot") >= 4


def test_weekday_only_data_closes_gaps_and_counts_a_working_week() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    skip = skip_axis(fcst["ds"], skip_days="weekends")
    kept = fcst[skip.keeps(fcst["ds"])]
    section = build_section(build_context(m, kept, df=df, skip=skip))
    assert "rangebreaks" in section.body
    assert "Rolling average (one working week)" in section.body
    assert "Lag, in observations (1 = one working day)" in section.body
    plain = build_section(context("linear"))
    # Without the skip the weekend gaps are real, so lags are counted in rows.
    assert "(1 = " not in plain.body
    assert "Rolling average (7 observations)" in plain.body


@pytest.mark.parametrize("rows", [12, 25, 60])
def test_short_histories_still_build(rows: int) -> None:
    ctx = context("linear")
    short = dataclasses.replace(
        ctx, history=ctx.history.head(rows), fcst=ctx.fcst.head(rows)
    )
    section = build_section(short)
    assert f'<div class="v">{rows:,}</div>' in section.body  # POINTS CHECKED tile


def test_short_history_notes_match_the_figures() -> None:
    ctx = context("linear")
    short = dataclasses.replace(
        ctx, history=ctx.history.head(25), fcst=ctx.fcst.head(25)
    )
    body = build_section(short).body
    assert (
        "Typical miss (RMS)" not in body
    )  # fewer than 40 points are not split by level
    assert "dark lines" not in body and "too few points to measure" in body
    full = build_section(ctx).body
    assert "Typical miss (RMS)" in full and "dark lines" in full


def test_a_perfect_fit_draws_no_normal_curve_and_does_not_mention_one() -> None:
    ctx = context("linear")
    y = ctx.history["y"].to_numpy(dtype=float)
    fcst = pd.DataFrame(
        {
            "ds": ctx.history["ds"],
            "yhat": y,
            "yhat_lower": y - 1.0,
            "yhat_upper": y + 1.0,
        }
    )
    body = build_section(dataclasses.replace(ctx, fcst=fcst)).body
    assert "Normal curve" not in body and "red curve" not in body
    assert (
        "stays close to zero" not in body
    )  # no scatter to measure the average against


@pytest.mark.parametrize("name", CASES)
def test_the_bias_tile_is_plain_text(name: str) -> None:
    body = build_section(context(name)).body
    tile = re.search(r'BIAS</div><div class="v">([^<]*)</div>', body)
    assert tile is not None
    assert re.fullmatch(r"0|[+-]\d[\d.,]*[KMB]?", tile.group(1)), tile.group(1)
    assert "e-0" not in tile.group(1)


def test_strongly_related_misses_do_not_make_the_band_look_too_narrow() -> None:
    # The logistic fixture has 25% of 510 points outside an 80% band, but a lag-1
    # correlation of 0.92 leaves it worth about 22 independent points. Counting every
    # point as independent called the band "narrower than it should be".
    section = build_section(context("logistic"))
    assert "narrower than it should be" not in section.body
    assert "about the right width" in section.body
    assert "fewer independent checks than points" in section.body


def test_section_needs_ten_in_sample_rows() -> None:
    ctx = context("linear")
    nine = dataclasses.replace(ctx, history=ctx.history.head(9), fcst=ctx.fcst.head(9))
    assert dg.build(nine) is None


def hourly_context(*, skipped: bool) -> DashboardContext:
    base = context("linear")
    ds = pd.date_range("2024-01-01", periods=24 * 28, freq="h")
    skip = None
    if skipped:
        skip = skip_axis(ds, skip_days="weekends", skip_hours=range(0, 8))
        ds = ds[skip.keeps(ds)]
    rng = np.random.default_rng(9)
    monday_nine = (ds.dayofweek == 0) & (ds.hour == 9)
    y = 100 + rng.normal(size=len(ds)) + np.where(monday_nine, 6.0, 0.0)
    fcst = pd.DataFrame(
        {"ds": ds, "yhat": 100.0, "yhat_lower": 97.0, "yhat_upper": 103.0}
    )
    history = pd.DataFrame({"ds": ds, "y": y})
    breaks = base.breaks
    if skip is not None:
        from prophet.plot import _plotly_rangebreaks

        breaks = tuple(_plotly_rangebreaks(skip.skip_days, skip.skip_hours))
    return dataclasses.replace(
        base, fcst=fcst, history=history, step=HOUR, skip=skip, breaks=breaks
    )


def test_hourly_data_gets_a_weekday_by_hour_heatmap() -> None:
    section = build_section(hourly_context(skipped=False))
    assert "Misses on the calendar" in section.body
    assert "Mondays at 09:00" in section.body
    assert "Lag, in observations (1 = one hour)" in section.body
    assert "Rolling average (one day)" in section.body


def test_hourly_data_with_skips_leaves_out_skipped_weekdays_and_hours() -> None:
    section = build_section(hourly_context(skipped=True))
    body = section.body.replace(" ", "")
    assert "rangebreaks" in body
    assert '"08:00"' in body and '"00:00"' not in body and '"07:00"' not in body
    assert '"Sat"' not in body and '"Sun"' not in body
    assert "Rolling average (one working day)" in section.body
    assert "Mondays at 09:00" in section.body


def test_largest_misses_table_lists_ten_rows_with_their_side_of_the_band() -> None:
    section = build_section(context("linear"))
    table = section.body[section.body.index('id="largest-misses"') :]
    table = table[: table.index("</table>")]
    assert table.count("<tr>") == dg.TOP_MISSES + 1  # header and rows
    assert (
        '<td class="num">Below</td>' in table or '<td class="num">Above</td>' in table
    )
    assert re.search(r"\d{2}\u00a0[A-Z][a-z]{2}\u00a0\d{4}", table)  # one-line dates
    plain = build_section(context("no_uncertainty"))
    assert "Above" not in plain.body[plain.body.index('id="largest-misses"') :]


def test_a_dashboard_without_intervals_has_no_coverage() -> None:
    section = build_section(context("linear", uncertainty=False))
    assert "INSIDE THE BAND" not in section.body
    assert "interval" not in section.summary
    assert "Inside the 80% interval" not in section.body


def test_a_year_of_hourly_rows_stays_light() -> None:
    base = context("linear")
    ds = pd.date_range("2022-01-01", periods=24 * 365, freq="h")
    y = 100 + np.random.default_rng(0).normal(size=len(ds)) * 5
    fcst = pd.DataFrame(
        {"ds": ds, "yhat": 100.0, "yhat_lower": 92.0, "yhat_upper": 108.0}
    )
    big = dataclasses.replace(
        base, fcst=fcst, history=pd.DataFrame({"ds": ds, "y": y}), step=HOUR
    )
    section = build_section(big)
    assert len(section.body) < 700_000  # scatters are thinned to a few thousand points
    assert "8,760" in section.body


def test_long_series_keep_both_colours_in_the_scatters() -> None:
    # With a fifth of 20,000 hourly points outside the band, thinning used to keep only
    # the outside ones and leave the "inside" trace empty.
    base = context("linear")
    ds = pd.date_range("2022-01-01", periods=20000, freq="h")
    y = 100 + np.random.default_rng(0).normal(size=len(ds)) * 5
    fcst = pd.DataFrame(
        {"ds": ds, "yhat": 100.0, "yhat_lower": 93.6, "yhat_upper": 106.4}
    )
    big = dataclasses.replace(
        base, fcst=fcst, history=pd.DataFrame({"ds": ds, "y": y}), step=HOUR
    )
    frame = dg._usable(dg.fitted_frame(big))
    view = dg._view(big, frame)
    diag = dg.diagnose(frame, 0.8)
    roll = dg.rolling_mean(view.resid, 24)
    figures = [
        dg._time_figure(big, view, roll, "Rolling average"),
        dg._level_figure(big, view, diag),
    ]
    for fig in figures:
        inside, outside = len(fig.data[0].x), len(fig.data[1].x)
        assert inside > 0 and outside > 0
        assert inside + outside <= dg.SCATTER_CAP
        assert outside / (inside + outside) == pytest.approx(0.2, abs=0.03)
