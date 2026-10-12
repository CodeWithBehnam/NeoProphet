"""Dashboard section "Model diagnostics": how the fit behaves on the training data.

The section looks at the residuals (observed minus fitted) of the rows the model
was trained on: how large they are, whether the interval covers as much as it
promises, whether neighbouring misses are related, whether the spread grows
with the level, and where in the calendar the model misses systematically.

The numbers and the wording come from small pure functions (statistics first,
then sentences); :func:`build` only assembles figures and HTML. In-sample
errors flatter a model, so the page says so and leaves the fair test to a
backtest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from prophet.dashboard._context import DashboardContext, Section, fitted_frame
from prophet.dashboard._palette import DIVERGING, colorscale
from prophet.dashboard._style import (
    SANS,
    axes,
    grid2,
    h3,
    hover_date,
    layout,
    lede,
    note,
    table,
    time_axis,
)
from prophet.plot import ACCENT, INK, INK_SOFT, OBSERVED
from prophet.visuals import (
    _UNIT_STEPS,
    Tile,
    _fmt_day,
    _fmt_number,
    _fmt_pct,
    _mape_is_meaningful,
    _signed,
    _step_unit,
    _tiles_html,
)

if TYPE_CHECKING:
    from prophet.plot import SkipAxis

KEY = "diagnostics"
TITLE = "Model diagnostics"

#: Fewest in-sample rows for which the section is drawn.
MIN_ROWS = 10
#: Most lags in the autocorrelation chart (fewer for short series).
MAX_LAGS = 40
#: Points drawn in a scatter and in a line, so that long series stay light.
SCATTER_CAP = 2500
LINE_CAP = 1500
#: Points outside the interval that a thinned scatter always keeps (up to this many),
#: so that rare extreme misses are not lost; beyond it they keep their true share.
FLAGGED_FLOOR = 100
#: Rows of the table of the largest misses.
TOP_MISSES = 10
#: An autocorrelation is "clear" beyond this multiple of the chance bound ...
CLEAR_FACTOR = 2.0
#: ... and beyond this absolute size, so that tiny but significant values stay quiet.
CLEAR_MIN = 0.1
#: Share of the MAE the mean miss must reach before the bias is worth a sentence.
BIAS_SHARE = 0.2
#: Ratio of the miss size between the highest and lowest fitted values that counts as a trend.
SPREAD_RATIO = 1.8
#: Fewest points in a heatmap cell, and the least score of a cell that counts as
#: systematic (see :func:`cell_scores`; the bar rises with the number of cells compared,
#: see :func:`chance_ceiling`). At 5, drifting misses are named in under 1 in 100 draws
#: at an autocorrelation of 0.9, and a cell that sits 3 noise levels off is always named.
CELL_MIN = 3
SYSTEMATIC_Z = 5.0
#: Share of a cell's effect that the rest of its row or column must carry for the
#: row or column, not the single cell, to be named.
LINE_SHARE = 0.6
#: Least z-score of the rolling average that counts as drift, before the bar rises with
#: the number of independent windows in the series.
DRIFT_Z = 3.0
#: Share of the MAE below which a mean miss prints as plain 0 rather than as noise.
ZERO_SHARE = 0.01
#: Share of the points a lag-1 correlation may leave independent before the verdict on the
#: interval is called rough.
INDEPENDENT_SHARE = 0.5

_SUBDAILY = pd.Timedelta(hours=20)
_WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

Verdict = Literal["wide", "narrow", "ok"]
GridKind = Literal["weekday_hour", "month_weekday"]
Position = Literal["inside", "above", "below"]

# ----------------------------------------------------------------------------
# Pure statistics
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorStats:
    """Size, direction and shape of the residuals (observed minus fitted).

    ``mape`` is a fraction and None when the observed values include zeros or
    values tiny next to their average. ``skewness`` and ``excess_kurtosis`` use
    population moments, so a normal distribution has 0 for both.
    """

    n: int
    bias: float
    mae: float
    rmse: float
    mape: float | None
    std: float
    skewness: float
    excess_kurtosis: float


def central_moments(resid: np.ndarray) -> tuple[float, float]:
    """Skewness and excess kurtosis; both 0 when the values do not spread."""
    centred = resid - resid.mean()
    variance = float(np.mean(centred**2))
    scale = float(np.max(np.abs(resid)))
    if variance <= (1e-9 * scale) ** 2:
        return 0.0, 0.0
    skew = float(np.mean(centred**3) / variance**1.5)
    kurtosis = float(np.mean(centred**4) / variance**2 - 3.0)
    return skew, kurtosis


def error_stats(
    y: Sequence[float] | np.ndarray, yhat: Sequence[float] | np.ndarray
) -> ErrorStats:
    """Statistics of ``y - yhat``. Raises ValueError for empty or mismatched input."""
    actual = np.asarray(y, dtype=float)
    fitted = np.asarray(yhat, dtype=float)
    if actual.ndim != 1 or actual.shape != fitted.shape:
        raise ValueError("y and yhat must be one-dimensional and of equal length.")
    if actual.size == 0:
        raise ValueError("There are no points to describe.")
    resid = actual - fitted
    skew, kurtosis = central_moments(resid)
    return ErrorStats(
        n=int(actual.size),
        bias=float(resid.mean()),
        mae=float(np.abs(resid).mean()),
        rmse=float(np.sqrt(np.mean(resid**2))),
        mape=(
            float(np.mean(np.abs(resid) / np.abs(actual)))
            if _mape_is_meaningful(actual)
            else None
        ),
        std=float(resid.std()),
        skewness=skew,
        excess_kurtosis=kurtosis,
    )


@dataclass(frozen=True)
class Coverage:
    """Share of points inside the interval and of those outside, above and below it.

    All shares are fractions of the ``n`` points that had an interval.
    ``nominal`` is the width the interval promises (``interval_width``).
    ``effective_n`` is how many independent points the ``n`` are worth when
    neighbouring misses are related (None means they count in full).
    """

    n: int
    inside: float
    above: float
    below: float
    nominal: float
    effective_n: float | None = None

    @property
    def outside(self) -> float:
        return self.above + self.below

    @property
    def expected_outside(self) -> float:
        return 1.0 - self.nominal

    @property
    def independent(self) -> float:
        """Points that count as separate evidence about the interval."""
        return (
            float(self.n) if self.effective_n is None else min(self.effective_n, self.n)
        )


def effective_count(n: int, rho1: float) -> float:
    """What ``n`` values with a lag-1 autocorrelation ``rho1`` are worth as independent ones.

    Values that follow each other (a positive correlation) carry less evidence
    than the same number of unrelated ones, by the factor (1 - rho1) / (1 + rho1).
    A negative correlation is not credited: the result never exceeds ``n``.
    """
    if rho1 <= 0.0:
        return float(n)
    return max(1.0, n * (1.0 - rho1) / (1.0 + rho1))


def interval_coverage(
    y: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    nominal: float,
    rho1: float = 0.0,
) -> Coverage | None:
    """In-sample coverage of an interval, or None when no point has one.

    ``rho1`` is the lag-1 autocorrelation of the misses; it sets how many of the
    points count as independent evidence (see :func:`effective_count`).
    """
    actual = np.asarray(y, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if not actual.shape == low.shape == high.shape:
        raise ValueError("y, lower and upper must be of equal length.")
    complete = np.isfinite(actual) & np.isfinite(low) & np.isfinite(high)
    n = int(complete.sum())
    if n == 0:
        return None
    actual, low, high = actual[complete], low[complete], high[complete]
    above = float(np.mean(actual > high))
    below = float(np.mean(actual < low))
    return Coverage(
        n=n,
        inside=float(np.mean((actual >= low) & (actual <= high))),
        above=above,
        below=below,
        nominal=nominal,
        effective_n=effective_count(n, rho1),
    )


def coverage_verdict(cov: Coverage) -> Verdict:
    """Whether the interval is too wide, too narrow or about right.

    The tolerance is two standard errors of the expected miss share, counted on
    the independent points (neighbouring misses that move together tell less
    than the same number of unrelated ones), and at least two percentage points.
    """
    p = cov.expected_outside
    tolerance = max(0.02, 2.0 * math.sqrt(max(p * (1.0 - p), 0.0) / cov.independent))
    gap = cov.outside - p
    if gap < -tolerance:
        return "wide"
    if gap > tolerance:
        return "narrow"
    return "ok"


def default_max_lag(n: int) -> int:
    """How many lags to inspect for ``n`` residuals: about a quarter, at most 40."""
    return max(1, min(MAX_LAGS, n // 4))


def acf_bound(n: int) -> float:
    """Approximate 95% bound of the autocorrelation of ``n`` independent values."""
    return 1.96 / math.sqrt(n)


def autocorrelation(resid: Sequence[float] | np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..``max_lag`` (zeros when the values do not spread)."""
    values = np.asarray(resid, dtype=float)
    if not 1 <= max_lag < values.size:
        raise ValueError(
            f"max_lag must be between 1 and {values.size - 1}, got {max_lag}."
        )
    centred = values - values.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        return np.zeros(max_lag)
    return np.array(
        [float(centred[k:] @ centred[:-k]) / denominator for k in range(1, max_lag + 1)]
    )


def clear_pattern(acf: np.ndarray, bound: float) -> tuple[int, float] | None:
    """The lag and size of the strongest autocorrelation, when it is clearly beyond chance.

    Over 40 lags a few bars poke past the 95% bound by luck, so "clearly" means
    twice the bound and at least :data:`CLEAR_MIN`.
    """
    if acf.size == 0:
        return None
    k = int(np.argmax(np.abs(acf)))
    r = float(acf[k])
    if abs(r) > max(CLEAR_FACTOR * bound, CLEAR_MIN):
        return k + 1, r
    return None


def _kept_days(skip: SkipAxis | None) -> int:
    return 7 - (len(skip.skip_days) if skip is not None else 0)


def _kept_hours(skip: SkipAxis | None) -> int:
    return 24 - (len(skip.skip_hours) if skip is not None else 0)


def steps_per_day(step: pd.Timedelta, skip: SkipAxis | None = None) -> float:
    """Observations in one kept day (sub-daily data respects skipped hours)."""
    seconds = step.total_seconds()
    if seconds <= 0:
        raise ValueError("The step must be positive.")
    kept = _kept_hours(skip) * 3600.0 if step < _SUBDAILY else 86400.0
    return kept / seconds


def steps_per_week(step: pd.Timedelta, skip: SkipAxis | None = None) -> float:
    """Observations in one week of the data's calendar (five for weekdays only)."""
    return steps_per_day(step, skip) * _kept_days(skip)


def _is_whole(x: float) -> bool:
    return abs(x - round(x)) < 1e-6 * max(1.0, abs(x))


def _count_phrase(n: int, unit: str) -> str:
    return f"one {unit}" if n == 1 else f"{n} {unit}s"


#: Units a lag is named in. Quarters are left out: "3 months" says the same more plainly.
_LAG_UNITS = tuple((name, size) for name, size in _UNIT_STEPS if name != "quarter")
_UNIT_SIZE = dict(_UNIT_STEPS)


def _exact_span_phrase(span: pd.Timedelta) -> str | None:
    """``span`` as a whole count of the largest unit that divides it exactly, else None.

    Nothing is rounded: 25 hours stays "25 hours" and 125 minutes "125 minutes".
    """
    for name, size in reversed(_LAG_UNITS):
        ratio = span / size
        if ratio >= 1.0 - 1e-9 and _is_whole(ratio):
            return _count_phrase(round(ratio), name)
    return None


def _unit_phrase(lag: int, unit: str) -> str:
    """``lag`` steps of the spacing's unit, as years when months or quarters make whole ones."""
    if unit == "month" and lag % 12 == 0:
        return _count_phrase(lag // 12, "year")
    if unit == "quarter" and lag % 4 == 0:
        return _count_phrase(lag // 4, "year")
    return _count_phrase(lag, unit)


def lag_phrase(
    lag: int,
    step: pd.Timedelta,
    skip: SkipAxis | None = None,
    regular: bool = True,
) -> str:
    """A lag of ``lag`` observations in plain words: ``one day``, ``one working week``.

    With skipped weekdays a week has fewer observations, so five steps of
    weekday-only data are one working week. Spacing that is not regular is
    counted in observations, since rows then do not map to elapsed time.

    A named unit is used only when the lag fills it exactly. Spacing that only
    roughly matches a unit (calendar months, which vary in length) is counted in
    that unit, and any other spacing is described by the exact span it covers,
    or by observations when that span is no whole number of any unit.
    """
    if lag < 1:
        raise ValueError("A lag is at least one observation.")
    if not regular:
        return _count_phrase(lag, "observation")
    working = skip is not None and set(skip.skip_days) == {5, 6}
    week = steps_per_week(step, skip)
    day = steps_per_day(step, skip)
    if week >= 1 - 1e-9 and _is_whole(lag / week):
        return _count_phrase(round(lag / week), "working week" if working else "week")
    if day >= 1 - 1e-9 and _is_whole(lag / day):
        return _count_phrase(round(lag / day), "working day" if working else "day")
    unit = _step_unit(step)
    if unit is not None and abs(step / _UNIT_SIZE[unit] - 1.0) > 1e-9:
        return _unit_phrase(lag, unit)
    exact = _exact_span_phrase(step * lag)
    return exact if exact is not None else _count_phrase(lag, "observation")


def regular_spacing(
    ds: pd.Series, step: pd.Timedelta, skip: SkipAxis | None = None
) -> bool:
    """Whether nearly every row sits one step after the previous one.

    Skipped weekdays and hours are closed up first, so a weekday-only series
    with ``skip`` counts as regular and the same series without it does not.
    """
    if len(ds) < 3:
        return True
    if skip is not None and skip.active:
        gaps = np.diff(np.asarray(skip.to_x(ds), dtype=float)) * 86400.0
    else:
        stamps = pd.DatetimeIndex(ds)
        gaps = np.asarray((stamps[1:] - stamps[:-1]).total_seconds(), dtype=float)
    near = np.abs(gaps / step.total_seconds() - 1.0) < 0.25
    return bool(near.mean() >= 0.9)


def rolling_window(n: int, step: pd.Timedelta, skip: SkipAxis | None = None) -> int:
    """Points in the rolling average: a week for daily data, a day for finer data.

    Never fewer than 3 points and never more than a fifth of the series.
    """
    target = (
        steps_per_week(step, skip) if step >= _SUBDAILY else steps_per_day(step, skip)
    )
    return int(min(max(3, round(target)), max(3, n // 5)))


def rolling_mean(resid: np.ndarray, window: int, *, partial: bool = True) -> np.ndarray:
    """Centred rolling mean.

    With ``partial`` the first and last points use the part of the window they
    have, which keeps the drawn line whole; without it they are NaN, so that
    every value averages a full window.
    """
    return (
        pd.Series(resid)
        .rolling(
            window, center=True, min_periods=max(1, window // 2) if partial else window
        )
        .mean()
        .to_numpy()
    )


def chance_ceiling(count: int, floor: float) -> float:
    """A z-score that the largest of ``count`` independent ones passes only very rarely.

    The largest of ``count`` standard normal values sits near sqrt(2 ln count),
    so one more unit above that, and never below ``floor``, keeps a search over
    many cells or windows from naming one that is only the luckiest.
    """
    return max(floor, math.sqrt(2.0 * math.log(max(count, 1))) + 1.0)


def _stride(positions: np.ndarray, count: int) -> np.ndarray:
    """``count`` of the ``positions``, evenly spread."""
    if count >= positions.size:
        return positions
    if count <= 0:
        return positions[:0]
    return positions[np.linspace(0, positions.size - 1, count).astype(int)]


def thin_indices(n: int, cap: int, keep: np.ndarray | None = None) -> np.ndarray:
    """Sorted row positions of at most ``cap`` of ``n`` rows.

    Rows flagged in ``keep`` (the misses outside the interval) and the others
    are each thinned by an even stride, so that the colours of the drawn points
    stay in the proportion of the data. Up to :data:`FLAGGED_FLOOR` flagged
    rows are always kept, since a handful of extreme misses is what the eye
    looks for.
    """
    everything = np.arange(n)
    if n <= cap:
        return everything
    if keep is None or not keep.any():
        return _stride(everything, cap)
    flagged = np.flatnonzero(keep)
    plain = np.flatnonzero(~keep)
    share = round(cap * flagged.size / n)
    flagged_cap = min(flagged.size, max(share, FLAGGED_FLOOR), cap)
    return np.sort(
        np.concatenate(
            [_stride(flagged, flagged_cap), _stride(plain, cap - flagged_cap)]
        )
    )


@dataclass(frozen=True, eq=False)
class Histogram:
    """Counts per bin of the residuals; ``edges`` has one more entry than ``counts``."""

    edges: np.ndarray
    counts: np.ndarray


def residual_histogram(resid: np.ndarray, max_bins: int = 60) -> Histogram:
    """Histogram with Freedman-Diaconis bins, between 8 and ``max_bins`` of them."""
    n = resid.size
    q75, q25 = np.percentile(resid, [75, 25])
    width = 2.0 * float(q75 - q25) / n ** (1.0 / 3.0)
    span = float(np.ptp(resid))
    bins = math.ceil(span / width) if width > 0 and span > 0 else 8
    counts, edges = np.histogram(resid, bins=int(np.clip(bins, 8, max_bins)))
    return Histogram(edges=edges, counts=counts)


def normal_counts(
    x: np.ndarray, mean: float, std: float, n: int, bin_width: float
) -> np.ndarray:
    """Counts a normal curve of the given mean and spread expects per bin of ``bin_width``."""
    if std <= 0:
        raise ValueError("A normal curve needs a positive spread.")
    density = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2.0 * math.pi))
    return n * bin_width * density


def spread_by_level(
    fitted: np.ndarray, resid: np.ndarray, groups: int | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Typical miss (root mean square) in groups of rising fitted value.

    Returns the median fitted value and the RMS residual of each group, or None
    when there are fewer than 40 points to split.
    """
    n = fitted.size
    if n < 40:
        return None
    count = groups if groups is not None else int(np.clip(n // 20, 2, 10))
    order = np.argsort(fitted, kind="stable")
    chunks = np.array_split(order, count)
    level = np.array([float(np.median(fitted[c])) for c in chunks])
    rms = np.array([math.sqrt(float(np.mean(resid[c] ** 2))) for c in chunks])
    return level, rms


def spread_ratio(rms: np.ndarray) -> float | None:
    """Miss size in the group with the highest fitted values over the lowest."""
    if rms.size < 2 or rms[0] <= 0 or rms[-1] <= 0:
        return None
    return float(rms[-1] / rms[0])


@dataclass(frozen=True)
class Miss:
    ds: pd.Timestamp
    observed: float
    fitted: float
    residual: float
    #: Where the observation sits against the interval; None without an interval.
    position: Position | None


def largest_misses(frame: pd.DataFrame, count: int = TOP_MISSES) -> list[Miss]:
    """The ``count`` rows with the largest absolute residual, the earliest first on ties.

    ``frame`` has ``ds``, ``y``, ``yhat`` and optionally ``yhat_lower`` and
    ``yhat_upper``.
    """
    y = frame["y"].to_numpy(dtype=float)
    fitted = frame["yhat"].to_numpy(dtype=float)
    resid = y - fitted
    banded = "yhat_lower" in frame and "yhat_upper" in frame
    low = frame["yhat_lower"].to_numpy(dtype=float) if banded else None
    high = frame["yhat_upper"].to_numpy(dtype=float) if banded else None
    order = np.argsort(-np.abs(resid), kind="stable")[:count]

    def position(i: int) -> Position | None:
        if low is None or high is None:
            return None
        if y[i] > high[i]:
            return "above"
        if y[i] < low[i]:
            return "below"
        return "inside"

    return [
        Miss(
            ds=pd.Timestamp(frame["ds"].iloc[i]),
            observed=float(y[i]),
            fitted=float(fitted[i]),
            residual=float(resid[i]),
            position=position(int(i)),
        )
        for i in order
    ]


@dataclass(frozen=True, eq=False)
class MeanGrid:
    """Mean residual by two calendar keys; cells with too few points are NaN.

    For ``weekday_hour`` the rows are weekdays (Monday is 0) and the columns
    hours of the day; for ``month_weekday`` the rows are months (1 to 12) and
    the columns weekdays. Only keys that occur in the data are listed.

    ``scale`` is the scatter of single misses around their own cell mean,
    pooled along the line that carries the level of the series (the hour for
    ``weekday_hour``, the month for ``month_weekday``). A cell at a busy hour or
    in a busy month is then judged against its own, larger, noise.
    """

    kind: GridKind
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    mean: np.ndarray
    count: np.ndarray
    scale: np.ndarray


def heatmap_kind(step: pd.Timedelta, span: pd.Timedelta) -> GridKind | None:
    """Which calendar heatmap suits the data, or None when none does.

    Sub-daily data with at least two weeks gets weekday by hour; daily data with
    at least two months gets month by weekday. Coarser data gets none.
    """
    if step < _SUBDAILY:
        return "weekday_hour" if span >= pd.Timedelta(days=14) else None
    if step < pd.Timedelta(days=3) and span >= pd.Timedelta(days=60):
        return "month_weekday"
    return None


def mean_grid(
    ds: pd.Series, resid: np.ndarray, kind: GridKind, min_count: int = CELL_MIN
) -> MeanGrid | None:
    """Mean residual per calendar cell, or None when fewer than four cells are filled."""
    stamps = pd.DatetimeIndex(ds)
    if kind == "weekday_hour":
        row_key, col_key = stamps.dayofweek, stamps.hour
    else:
        row_key, col_key = stamps.month, stamps.dayofweek
    frame = pd.DataFrame(
        {"row": np.asarray(row_key), "col": np.asarray(col_key), "e": resid}
    )
    cells = frame.groupby(["row", "col"])["e"]
    frame["dev2"] = (frame["e"] - cells.transform("mean")) ** 2
    stats = cells.agg(["mean", "count"])
    line = "col" if kind == "weekday_hour" else "row"
    dof = (stats["count"] - 1).groupby(level=line).sum()
    pooled = (
        math.sqrt(float(frame["dev2"].sum()) / float(dof.sum()))
        if dof.sum() > 0
        else 0.0
    )
    variance = frame.groupby(line)["dev2"].sum() / dof.where(dof > 0)
    by_line = variance.pow(0.5).fillna(pooled)
    cell_scale = pd.Series(
        by_line.reindex(stats.index.get_level_values(line)).to_numpy(),
        index=stats.index,
    )
    means = stats["mean"].unstack()
    counts = stats["count"].unstack().fillna(0)
    scales = cell_scale.unstack()
    means = means.where(counts >= min_count)
    means = means.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if means.size == 0 or int(means.notna().to_numpy().sum()) < 4:
        return None
    counts = counts.reindex(index=means.index, columns=means.columns).fillna(0)
    scales = scales.reindex(index=means.index, columns=means.columns).fillna(0.0)
    return MeanGrid(
        kind=kind,
        rows=tuple(int(r) for r in means.index),
        cols=tuple(int(c) for c in means.columns),
        mean=means.to_numpy(dtype=float),
        count=counts.to_numpy(dtype=float),
        scale=scales.to_numpy(dtype=float),
    )


def _key_name(axis: Literal["weekday", "month", "hour"], key: int) -> str:
    if axis == "weekday":
        return _WEEKDAYS[key]
    if axis == "month":
        return _MONTHS[key - 1]
    return f"{key:02d}:00"


def _axes_of(
    grid: MeanGrid,
) -> tuple[Literal["weekday", "month"], Literal["hour", "weekday"]]:
    return ("weekday", "hour") if grid.kind == "weekday_hour" else ("month", "weekday")


def _axis_label(axis: Literal["weekday", "month", "hour"], key: int) -> str:
    name = _key_name(axis, key)
    return name if axis == "hour" else name[:3]


def cell_scores(grid: MeanGrid) -> tuple[np.ndarray, np.ndarray]:
    """Two z-scores per cell, ``alone`` and ``relative`` (NaN where a cell cannot be scored).

    ``alone`` is the cell mean over its standard error, which takes the misses in
    a cell for unrelated values with the scatter of their line. They rarely are:
    misses that follow each other make every cell mean wander further than that.
    ``relative`` therefore re-expresses ``alone`` against how far the cells
    typically stray from the middle one (1.4826 times the median absolute
    deviation, never below 1), so that a cell has to stand out from the rest of
    the grid and not only from zero.
    """
    usable = np.isfinite(grid.mean) & (grid.count > 0) & (grid.scale > 0)
    alone = np.full(grid.mean.shape, np.nan)
    alone[usable] = grid.mean[usable] * np.sqrt(grid.count[usable]) / grid.scale[usable]
    relative = np.full(grid.mean.shape, np.nan)
    if int(usable.sum()) >= 4:
        centre = float(np.median(alone[usable]))
        spread = max(1.0, 1.4826 * float(np.median(np.abs(alone[usable] - centre))))
        relative[usable] = (alone[usable] - centre) / spread
    return alone, relative


def strongest_cell(grid: MeanGrid) -> tuple[int, int, float] | None:
    """Row, column and score of the cell that stands out most, or None.

    The score of a cell is the smaller of its two z-scores (see
    :func:`cell_scores`), so it is high only when the cell is far from zero and
    far from the other cells. None when no cell can be scored.
    """
    alone, relative = cell_scores(grid)
    score = np.fmin(np.abs(alone), np.abs(relative))
    if not np.isfinite(score).any():
        return None
    r, c = np.unravel_index(int(np.nanargmax(score)), score.shape)
    return int(r), int(c), float(score[r, c])


def standout_bar(grid: MeanGrid) -> float:
    """The score a cell must reach to stand out, which rises with the number of cells compared."""
    return chance_ceiling(int(np.isfinite(grid.mean).sum()), SYSTEMATIC_Z)


@dataclass(frozen=True)
class LineEffect:
    """A whole row or column of the grid that leans the same way as a standout cell."""

    axis: Literal["row", "col"]
    #: Position of the line in the grid.
    index: int
    #: Mean miss over the whole line, weighted by the points in each cell, and its points.
    mean: float
    points: int


def line_effect(grid: MeanGrid, r: int, c: int) -> LineEffect | None:
    """The row or column of cell (``r``, ``c``) when it, not the single cell, carries the effect.

    The other filled cells of the line (at least two) must lean the same way as
    the cell and average at least :data:`LINE_SHARE` of its size. Of the row and
    the column the one whose other cells lean harder is returned.
    """
    value = grid.mean[r, c]
    best: tuple[float, Literal["row", "col"]] | None = None
    for axis in ("row", "col"):
        line_mean = grid.mean[r, :] if axis == "row" else grid.mean[:, c]
        line_count = grid.count[r, :] if axis == "row" else grid.count[:, c]
        others = np.isfinite(line_mean)
        others[c if axis == "row" else r] = False
        if int(others.sum()) < 2:
            continue
        rest = float(np.average(line_mean[others], weights=line_count[others]))
        if rest * value > 0 and abs(rest) >= LINE_SHARE * abs(value):
            if best is None or abs(rest) > best[0]:
                best = (abs(rest), axis)
    if best is None:
        return None
    axis = best[1]
    line_mean = grid.mean[r, :] if axis == "row" else grid.mean[:, c]
    line_count = grid.count[r, :] if axis == "row" else grid.count[:, c]
    filled = np.isfinite(line_mean)
    return LineEffect(
        axis=axis,
        index=r if axis == "row" else c,
        mean=float(np.average(line_mean[filled], weights=line_count[filled])),
        points=int(line_count[filled].sum()),
    )


def _line_place(grid: MeanGrid, line: LineEffect) -> str:
    """Where a whole row or column is: ``on Fridays``, ``around 09:00``, ``in May``."""
    row_axis, col_axis = _axes_of(grid)
    if line.axis == "row":
        axis, key = row_axis, grid.rows[line.index]
    else:
        axis, key = col_axis, grid.cols[line.index]
    name = _key_name(axis, key)
    if axis == "weekday":
        return f"on {name}s"
    return f"around {name}" if axis == "hour" else f"in {name}"


def grid_sentence(grid: MeanGrid) -> str:
    """Plain words for the calendar heatmap: the clearest systematic miss, or its absence.

    A miss is called systematic only when its cell is far from zero and from the
    other cells by more than the luckiest of that many cells would be. When the
    rest of its row or column leans the same way, the row or column is named
    instead of the cell, since the cell is then only the extreme of a broader
    pattern.
    """
    pair = "weekday and hour" if grid.kind == "weekday_hour" else "month and weekday"
    best = strongest_cell(grid)
    if best is None or best[2] < standout_bar(grid):
        return f"No single {pair} combination stands out from the rest of the grid."
    r, c, _ = best
    line = line_effect(grid, r, c)
    if line is not None:
        where, value, points = _line_place(grid, line), line.mean, line.points
    else:
        row_axis, col_axis = _axes_of(grid)
        row, col = _key_name(row_axis, grid.rows[r]), _key_name(col_axis, grid.cols[c])
        where = (
            f"on {row}s at {col}"
            if grid.kind == "weekday_hour"
            else f"on {col}s in {row}"
        )
        value, points = float(grid.mean[r, c]), int(grid.count[r, c])
    side = "above" if value > 0 else "below"
    return (
        f"The clearest systematic miss is {where}, where the data sit on "
        f"average {_fmt_number(abs(value))} {side} the fit ({points} points)."
    )


@dataclass(frozen=True, eq=False)
class Diagnosis:
    """Every statistic the section reports about one set of residuals."""

    stats: ErrorStats
    coverage: Coverage | None
    acf: np.ndarray
    bound: float
    #: Lag and size of a clearly non-random autocorrelation, or None.
    pattern: tuple[int, float] | None
    #: Median fitted value and RMS miss per group of rising fitted value, or None.
    spread: tuple[np.ndarray, np.ndarray] | None
    spread_ratio: float | None


def diagnose(frame: pd.DataFrame, nominal: float | None) -> Diagnosis:
    """The statistics of ``frame`` (``ds``, ``y``, ``yhat`` and optionally the interval).

    ``nominal`` is the width the interval promises; None skips the coverage.
    """
    y = frame["y"].to_numpy(dtype=float)
    fitted = frame["yhat"].to_numpy(dtype=float)
    stats = error_stats(y, fitted)
    resid = y - fitted
    acf = autocorrelation(resid, default_max_lag(stats.n))
    coverage = (
        interval_coverage(
            y,
            frame["yhat_lower"].to_numpy(dtype=float),
            frame["yhat_upper"].to_numpy(dtype=float),
            nominal,
            rho1=float(acf[0]),
        )
        if nominal is not None and "yhat_lower" in frame and "yhat_upper" in frame
        else None
    )
    bound = acf_bound(stats.n)
    spread = spread_by_level(fitted, resid)
    return Diagnosis(
        stats=stats,
        coverage=coverage,
        acf=acf,
        bound=bound,
        pattern=clear_pattern(acf, bound),
        spread=spread,
        spread_ratio=spread_ratio(spread[1]) if spread is not None else None,
    )


# ----------------------------------------------------------------------------
# Pure wording
# ----------------------------------------------------------------------------

_VERDICT_PHRASE: Mapping[Verdict, str] = {
    "wide": "wider than it needs to be",
    "narrow": "narrower than it should be",
    "ok": "about the right width",
}


def accuracy_sentence(stats: ErrorStats) -> str:
    """The average miss and, when percentages mean something, the typical error: the words of the tiles."""
    text = f"On the training data the average miss is {_fmt_number(stats.mae)} (MAE)"
    if stats.mape is not None:
        text += f" and the typical error is {_fmt_pct(stats.mape)} of the observed value (MAPE)"
    return text + "."


def miss_split_sentence(cov: Coverage) -> str:
    """Which side of the band the misses fall on, when it is lopsided; else empty."""
    if cov.outside <= 0 or cov.outside * cov.n < 5:
        return ""
    above = cov.above / cov.outside
    if above >= 0.7:
        return (
            f"Most of the misses ({_fmt_pct(above)}) are above the band, "
            "so the model tends to run low there."
        )
    if above <= 0.3:
        return (
            f"Most of the misses ({_fmt_pct(1.0 - above)}) are below the band, "
            "so the model tends to run high there."
        )
    return ""


def rough_verdict(cov: Coverage) -> bool:
    """The band looks about right, but only on few independent points."""
    return coverage_verdict(cov) == "ok" and cov.independent < INDEPENDENT_SHARE * cov.n


def coverage_sentence(cov: Coverage, *, caveat: bool = True) -> str:
    """How the interval fared, with a caveat when "about right" rests on few independent points.

    ``caveat=False`` leaves the caveat to the sentence about neighbouring misses,
    which says the same thing (see :func:`pattern_sentence`).
    """
    lead = (
        f"{_fmt_pct(cov.inside)} of points fall inside the {cov.nominal:.0%} "
        f"interval, against {_fmt_pct(cov.nominal)} expected, "
        f"so the band is {_VERDICT_PHRASE[coverage_verdict(cov)]}."
    )
    split = miss_split_sentence(cov)
    rough = (
        "Neighbouring misses move together, so there are fewer independent checks "
        "than points and this verdict is a rough one."
        if caveat and rough_verdict(cov)
        else ""
    )
    return " ".join(part for part in (lead, split, rough) if part)


def bias_sentence(stats: ErrorStats) -> str:
    """The average direction of the misses, only when it is a notable share of the MAE."""
    if stats.mae <= 0 or abs(stats.bias) < BIAS_SHARE * stats.mae:
        return ""
    side = "low" if stats.bias > 0 else "high"
    return f"On average the fit runs {side} by {_fmt_number(abs(stats.bias))}."


def pattern_sentence(
    lag: int, r: float, bound: float, lag_text: str, *, rough: bool = False
) -> str:
    """The left-over pattern in the misses, for an autocorrelation clearly beyond chance.

    ``rough`` adds that the verdict on the band rests on fewer independent points
    than there are points, which neighbouring misses that move together imply.
    """
    if lag == 1:
        how = (
            "Neighbouring misses move together"
            if r > 0
            else "Neighbouring misses tend to alternate in sign"
        )
    elif r > 0:
        how = f"Each miss tends to resemble the miss from {lag_text} earlier"
    else:
        how = (
            "Each miss tends to have the opposite sign to the miss from "
            f"{lag_text} earlier"
        )
    tail = "so some pattern is left over"
    if rough:
        tail += ", and there are fewer independent checks than points, so the verdict on the band is a rough one"
    return f"{how} (correlation {r:+.2f}, against about ±{bound:.2f} for pure noise), {tail}."


def spread_sentence(ratio: float | None) -> str:
    """Whether the misses grow or shrink with the level of the series."""
    if ratio is None or ratio <= 0:
        return ""
    if ratio >= SPREAD_RATIO:
        return (
            f"The misses grow with the level: they are about {ratio:.1f} times as "
            "large where the fit is highest as where it is lowest."
        )
    if ratio <= 1.0 / SPREAD_RATIO:
        return (
            f"The misses shrink as the level rises: they are about {1.0 / ratio:.1f} "
            "times as large where the fit is lowest as where it is highest."
        )
    return "The size of the misses is similar across the range of fitted values."


def _signed_one(x: float) -> str:
    """One decimal with a sign; a value that rounds to zero prints as plain 0.0."""
    return "0.0" if round(x, 1) == 0 else f"{x:+.1f}"


def signed_number(v: float, scale: float) -> str:
    """A signed compact number; a value within :data:`ZERO_SHARE` of ``scale`` prints as 0.

    Fitted values average out to almost exactly zero miss, so the leftover
    (such as -4.7e-06) is rounding noise and is not shown as a number.
    """
    return "0" if abs(v) <= ZERO_SHARE * scale else _signed(v, False)


def shape_sentence(skewness: float, excess_kurtosis: float) -> str:
    """The shape of the miss distribution against a normal curve."""
    if abs(skewness) < 0.5:
        lean = "roughly symmetric"
    elif skewness > 0:
        lean = "skewed towards points above the fit"
    else:
        lean = "skewed towards points below the fit"
    if excess_kurtosis > 1.0:
        tails = (
            "heavier tails than a normal curve, so now and then a miss is much "
            "larger than the typical one"
        )
    elif excess_kurtosis < -1.0:
        tails = "lighter tails than a normal curve, so extreme misses are rare"
    else:
        tails = "tails close to a normal curve"
    return (
        f"The misses are {lean}, with {tails} "
        f"(skewness {_signed_one(skewness)}, excess kurtosis {_signed_one(excess_kurtosis)})."
    )


def acf_note(acf: np.ndarray, bound: float) -> str:
    """How many lags poke past the chance bound, against how many would by luck."""
    beyond = int((np.abs(acf) > bound).sum())
    expected = 0.05 * acf.size
    return (
        "Each bar is the correlation between a miss and the one that many "
        "observations earlier. Bars inside the dashed lines are what pure noise "
        f"produces: {beyond} of {acf.size} lags are beyond them, against about "
        f"{_fmt_number(expected)} by chance alone."
    )


def drift_bar(n: int) -> float:
    """How many standard errors the rolling average must stray from zero to count as drift.

    A window of ``window`` unrelated misses averages to within one standard
    error (the spread of a miss over the square root of ``window``) of zero. The
    average is looked at in every one of ``n`` overlapping positions, and the
    largest of them strays further than a single one would, hence
    :func:`chance_ceiling`. Over series of 30 to 2,000 points this calls noise
    drift in about one draw in a hundred.
    """
    return chance_ceiling(n, DRIFT_Z)


def drift_sentence(
    ds: pd.Series, resid: np.ndarray, window: int, step: pd.Timedelta
) -> str:
    """Where the rolling average of the misses strays furthest from zero, if it is more than chance.

    Only windows that hold all ``window`` misses are compared (the ends of the
    series have fewer, and their averages are noisier, so they would win too
    often). An empty string means there is nothing to say: no full window, or no
    scatter to measure the average against.
    """
    spread = float(resid.std())
    if spread <= 0 or window < 1 or resid.size < window:
        return ""
    roll = rolling_mean(resid, window, partial=False)
    if not np.isfinite(roll).any():
        return ""
    peak = int(np.nanargmax(np.abs(roll)))
    stray = abs(float(roll[peak])) / (spread / math.sqrt(window))
    if stray < drift_bar(resid.size):
        return (
            "The rolling average stays close to zero (within what chance alone "
            "would produce), so the model does not drift away from the data anywhere."
        )
    side = "low" if roll[peak] > 0 else "high"
    when = _fmt_day(pd.Timestamp(ds.iloc[peak]), step)
    return (
        f"The rolling average strays furthest from zero around {when} "
        f"({_signed(float(roll[peak]), False)}), a stretch where the model runs {side}."
    )


def level_note(diag: Diagnosis) -> str:
    """The note under the miss-against-fitted chart, describing only what is drawn."""
    text = "Each point is a miss against the value the model fitted there."
    if diag.spread is not None:
        text += " The dark lines mark the typical size of a miss at each level."
    else:
        text += (
            " There are too few points to measure the typical size of a miss "
            "at each level."
        )
    return f"{text} {spread_sentence(diag.spread_ratio)}".strip()


def histogram_note(stats: ErrorStats) -> str:
    """The note under the histogram; the red curve is mentioned only when it is drawn."""
    text = shape_sentence(stats.skewness, stats.excess_kurtosis)
    if stats.std > 0:
        text += " The red curve is a normal curve of the same average and spread."
    return text


def lede_text(
    diag: Diagnosis, lag_text: str | None = None, additive: bool = False
) -> str:
    """The opening paragraph: accuracy, coverage, and what is left over, from the numbers.

    ``lag_text`` is :func:`lag_phrase` of the clear autocorrelation lag. A hint
    towards multiplicative seasonality is added only for an additive model
    whose misses grow with the level.
    """
    sentences = [accuracy_sentence(diag.stats)]
    # The caveat on the band and the sentence on the pattern both open with "Neighbouring
    # misses move together" when the pattern is at a lag of one: say it once.
    merged = (
        diag.coverage is not None
        and diag.pattern is not None
        and lag_text is not None
        and diag.pattern[0] == 1
        and diag.pattern[1] > 0
        and rough_verdict(diag.coverage)
    )
    if diag.coverage is not None:
        sentences.append(coverage_sentence(diag.coverage, caveat=not merged))
    sentences.append(bias_sentence(diag.stats))
    if diag.pattern is not None and lag_text is not None:
        lag, r = diag.pattern
        sentences.append(pattern_sentence(lag, r, diag.bound, lag_text, rough=merged))
    ratio = diag.spread_ratio
    if (
        ratio is not None
        and ratio > 0
        and not 1.0 / SPREAD_RATIO < ratio < SPREAD_RATIO
    ):
        sentences.append(spread_sentence(ratio))
        if additive and ratio >= SPREAD_RATIO:
            sentences.append("Multiplicative seasonality may suit the series better.")
    return " ".join(s for s in sentences if s)


def summary_text(diag: Diagnosis) -> str:
    """One short sentence for the key findings at the top of the page; the lag and the statistics stay in the section."""
    text = f"The fit misses by {_fmt_number(diag.stats.mae)} on average"
    cov = diag.coverage
    if cov is not None:
        text += f" and its {cov.nominal:.0%} band is {_VERDICT_PHRASE[coverage_verdict(cov)]}"
    if diag.pattern is not None:
        text += "; some pattern is left in the misses"
    return text + "."


# ----------------------------------------------------------------------------
# Figures and page
# ----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _View:
    """In-sample rows as plain arrays, ready for the figures."""

    ds: np.ndarray
    y: np.ndarray
    yhat: np.ndarray
    resid: np.ndarray
    #: True where the observation lies outside the interval (all False without one).
    outside: np.ndarray
    #: ``80% interval`` or None when no interval is drawn.
    band: str | None
    #: Hover details per row: observed, fitted and the date as text.
    custom: np.ndarray


def _usable(frame: pd.DataFrame) -> pd.DataFrame:
    finite = np.isfinite(frame["y"].to_numpy(dtype=float)) & np.isfinite(
        frame["yhat"].to_numpy(dtype=float)
    )
    return frame.loc[finite].reset_index(drop=True)


def _view(ctx: DashboardContext, frame: pd.DataFrame) -> _View:
    y = frame["y"].to_numpy(dtype=float)
    yhat = frame["yhat"].to_numpy(dtype=float)
    banded = "yhat_lower" in frame and "yhat_upper" in frame
    outside = (
        (y > frame["yhat_upper"].to_numpy(dtype=float))
        | (y < frame["yhat_lower"].to_numpy(dtype=float))
        if banded
        else np.zeros(len(frame), dtype=bool)
    )
    when = frame["ds"].dt.strftime(hover_date(ctx)).to_numpy()
    custom = np.empty((len(frame), 3), dtype=object)
    custom[:, 0], custom[:, 1], custom[:, 2] = y, yhat, when
    return _View(
        ds=frame["ds"].to_numpy(),
        y=y,
        yhat=yhat,
        resid=y - yhat,
        outside=outside,
        band=f"{ctx.m.interval_width:.0%} interval" if banded else None,
        custom=custom,
    )


def _point_traces(
    go: Any, view: _View, x: np.ndarray, idx: np.ndarray, hover: str
) -> list[Any]:
    """Residual points, inside the interval in grey and outside it in the accent."""

    def trace(
        sel: np.ndarray, name: str, colour: str, size: int, opacity: float
    ) -> Any:
        return go.Scatter(
            name=name,
            x=x[sel],
            y=view.resid[sel],
            mode="markers",
            marker={"color": colour, "size": size, "opacity": opacity},
            customdata=view.custom[sel],
            hovertemplate=hover,
        )

    flagged = view.outside[idx]
    if view.band is None:
        return [trace(idx, "Miss", OBSERVED, 4, 0.75)]
    traces = [trace(idx[~flagged], f"Inside the {view.band}", OBSERVED, 4, 0.75)]
    if flagged.any():
        traces.append(trace(idx[flagged], f"Outside the {view.band}", ACCENT, 5, 0.9))
    return traces


def _residual_axis_title() -> dict[str, Any]:
    return {"text": "Observed minus fitted", "font": {"size": 12}}


def _time_figure(
    ctx: DashboardContext, view: _View, roll: np.ndarray, roll_name: str
) -> Any:
    go = ctx.go
    date = hover_date(ctx)
    fig = go.Figure()
    idx = thin_indices(view.resid.size, SCATTER_CAP, keep=view.outside)
    hover = (
        f"%{{x|{date}}}<br>Observed %{{customdata[0]:,.4g}}, fitted "
        "%{customdata[1]:,.4g}<br>Miss <b>%{y:+,.4g}</b><extra></extra>"
    )
    for trace in _point_traces(go, view, view.ds, idx, hover):
        fig.add_trace(trace)
    line = thin_indices(view.resid.size, LINE_CAP)
    fig.add_trace(
        go.Scatter(
            name=roll_name,
            x=view.ds[line],
            y=roll[line],
            mode="lines",
            line={"color": INK, "width": 2},
            hovertemplate=f"%{{x|{date}}}<br>Rolling average <b>%{{y:+,.3g}}</b><extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": INK, "width": 1, "dash": "dot"})
    _, y_style = axes()
    fig.update_layout(
        **layout(
            340,
            hovermode="closest",
            hoverdistance=20,
            showlegend=True,
            xaxis=time_axis(ctx),
            yaxis={**y_style, "title": _residual_axis_title()},
        )
    )
    return fig


def _histogram_figure(ctx: DashboardContext, view: _View, stats: ErrorStats) -> Any:
    go = ctx.go
    hist = residual_histogram(view.resid)
    widths = np.diff(hist.edges)
    centres = hist.edges[:-1] + widths / 2.0
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Misses",
            x=centres,
            y=hist.counts,
            width=widths,
            marker={"color": OBSERVED, "opacity": 0.8, "line": {"width": 0}},
            customdata=np.column_stack([hist.edges[:-1], hist.edges[1:]]),
            hovertemplate=(
                "%{customdata[0]:,.3g} to %{customdata[1]:,.3g}"
                "<br><b>%{y:,}</b> points<extra></extra>"
            ),
        )
    )
    if stats.std > 0:
        grid = np.linspace(hist.edges[0], hist.edges[-1], 120)
        mean = float(view.resid.mean())
        fig.add_trace(
            go.Scatter(
                name="Normal curve",
                x=grid,
                y=normal_counts(grid, mean, stats.std, stats.n, float(np.mean(widths))),
                mode="lines",
                line={"color": ACCENT, "width": 2},
                hovertemplate=(
                    "%{x:,.3g}<br>A normal curve expects <b>%{y:,.1f}</b> "
                    "points here<extra></extra>"
                ),
            )
        )
    x_style, y_style = axes()
    fig.update_layout(
        **layout(
            320,
            showlegend=True,
            bargap=0.04,
            margin={"l": 64, "r": 24, "t": 36, "b": 56},
            xaxis={**x_style, "title": _residual_axis_title(), "zeroline": False},
            yaxis={**y_style, "title": {"text": "Points", "font": {"size": 12}}},
        )
    )
    return fig


def _acf_figure(
    ctx: DashboardContext,
    diag: Diagnosis,
    lag_names: Sequence[str],
    regular: bool,
    unit_text: str,
) -> Any:
    go = ctx.go
    acf, bound = diag.acf, diag.bound
    lags = np.arange(1, acf.size + 1)
    names = np.array(lag_names)
    beyond = np.abs(acf) > bound
    hover = "Lag %{x} (%{customdata})<br>Correlation <b>%{y:+.2f}</b><extra></extra>"
    fig = go.Figure()
    for flag, name, colour in (
        (False, "Within chance level", OBSERVED),
        (True, "Beyond chance level", ACCENT),
    ):
        sel = beyond == flag
        if sel.any():
            fig.add_trace(
                go.Bar(
                    name=name,
                    x=lags[sel],
                    y=acf[sel],
                    marker={"color": colour},
                    customdata=names[sel],
                    hovertemplate=hover,
                )
            )
    edges = [0.5, acf.size + 0.5]
    for sign, shown in ((1.0, True), (-1.0, False)):
        fig.add_trace(
            go.Scatter(
                name="95% chance bound",
                x=edges,
                y=[sign * bound] * 2,
                mode="lines",
                line={"color": INK_SOFT, "width": 1.2, "dash": "dash"},
                legendgroup="bound",
                showlegend=shown,
                customdata=[bound, bound],
                hovertemplate=(
                    "Pure noise stays within ±%{customdata:.2f} "
                    "95% of the time<extra></extra>"
                ),
            )
        )
    x_style, y_style = axes()
    reach = max(float(np.max(np.abs(acf))), bound) * 1.15
    tick = 1 if acf.size <= 12 else 2 if acf.size <= 24 else 5
    title = "Lag, in observations" + (f" (1 = {unit_text})" if regular else "")
    fig.update_layout(
        **layout(
            320,
            showlegend=True,
            bargap=0.35,
            margin={"l": 64, "r": 24, "t": 36, "b": 56},
            xaxis={
                **x_style,
                "range": [0.4, acf.size + 0.6],
                "dtick": tick,
                "title": {"text": title, "font": {"size": 12}},
            },
            yaxis={
                **y_style,
                "range": [-reach, reach],
                "zeroline": True,
                "zerolinecolor": INK_SOFT,
                "zerolinewidth": 1,
                "title": {"text": "Correlation", "font": {"size": 12}},
            },
        )
    )
    return fig


def _level_figure(ctx: DashboardContext, view: _View, diag: Diagnosis) -> Any:
    go = ctx.go
    hover = (
        "%{customdata[2]}<br>Fitted %{x:,.4g}, observed %{customdata[0]:,.4g}"
        "<br>Miss <b>%{y:+,.4g}</b><extra></extra>"
    )
    fig = go.Figure()
    idx = thin_indices(view.resid.size, SCATTER_CAP, keep=view.outside)
    for trace in _point_traces(go, view, view.yhat, idx, hover):
        fig.add_trace(trace)
    if diag.spread is not None:
        level, rms = diag.spread
        for sign, shown in ((1.0, True), (-1.0, False)):
            fig.add_trace(
                go.Scatter(
                    name="Typical miss (RMS)",
                    x=level,
                    y=sign * rms,
                    mode="lines",
                    line={"color": INK, "width": 2},
                    legendgroup="rms",
                    showlegend=shown,
                    hovertemplate=(
                        "Around a fitted value of %{x:,.4g}<br>"
                        "Typical miss <b>±%{customdata:,.3g}</b><extra></extra>"
                    ),
                    customdata=rms,
                )
            )
    fig.add_hline(y=0, line={"color": INK_SOFT, "width": 1, "dash": "dot"})
    x_style, y_style = axes()
    fig.update_layout(
        **layout(
            340,
            hovermode="closest",
            hoverdistance=20,
            showlegend=True,
            margin={"l": 64, "r": 24, "t": 36, "b": 56},
            xaxis={**x_style, "title": {"text": "Fitted value", "font": {"size": 12}}},
            yaxis={**y_style, "title": _residual_axis_title()},
        )
    )
    return fig


def _heatmap_figure(ctx: DashboardContext, grid: MeanGrid) -> Any:
    go = ctx.go
    row_axis, col_axis = _axes_of(grid)
    rows = [_axis_label(row_axis, k) for k in grid.rows]
    cols = [_axis_label(col_axis, k) for k in grid.cols]
    row_title = "Weekday" if row_axis == "weekday" else "Month"
    col_title = "Hour of day" if col_axis == "hour" else "Weekday"
    fig = go.Figure(
        go.Heatmap(
            x=cols,
            y=rows,
            z=grid.mean,
            customdata=grid.count,
            colorscale=colorscale(DIVERGING),
            zmid=0,
            xgap=1,
            ygap=1,
            hoverongaps=False,
            hovertemplate=(
                f"{row_title} %{{y}}, {col_title.lower()} %{{x}}"
                "<br>Mean miss <b>%{z:+,.3g}</b><br>%{customdata:.0f} points"
                "<extra></extra>"
            ),
            colorbar={
                "thickness": 10,
                "outlinewidth": 0,
                "tickfont": {"family": SANS, "color": INK_SOFT, "size": 11},
                "title": {
                    "text": "Mean miss",
                    "side": "right",
                    "font": {"family": SANS, "color": INK_SOFT, "size": 12},
                },
            },
        )
    )
    x_style, _ = axes()
    fig.update_layout(
        **layout(
            340,
            margin={"l": 72, "r": 24, "t": 24, "b": 56},
            xaxis={
                **x_style,
                "type": "category",
                "title": {"text": col_title, "font": {"size": 12}},
            },
            yaxis={
                "type": "category",
                "autorange": "reversed",
                "showgrid": False,
                "ticks": "",
                "tickfont": x_style["tickfont"],
                "title": {"text": row_title, "font": {"size": 12}},
            },
        )
    )
    return fig


def _tiles(diag: Diagnosis) -> list[Tile]:
    stats, cov = diag.stats, diag.coverage
    tiles = [
        Tile(
            "AVERAGE MISS",
            _fmt_number(stats.mae),
            "mean absolute error (MAE), in the units of the data",
        ),
        Tile(
            "BIG MISSES",
            _fmt_number(stats.rmse),
            "RMSE, which weights large misses more",
        ),
    ]
    if stats.mape is not None:
        tiles.append(
            Tile(
                "TYPICAL ERROR",
                _fmt_pct(stats.mape),
                "average miss as a share of the observed value (MAPE)",
            )
        )
    tiles.append(
        Tile(
            "BIAS",
            signed_number(stats.bias, stats.mae),
            "average of observed minus fitted; positive means the fit runs low",
        )
    )
    if cov is not None:
        tiles.append(
            Tile(
                "INSIDE THE BAND",
                _fmt_pct(cov.inside),
                f"of points fell inside the {cov.nominal:.0%} interval "
                f"({_fmt_pct(cov.nominal)} expected)",
            )
        )
    tiles.append(
        Tile(
            "POINTS CHECKED",
            f"{stats.n:,}",
            "training rows, which the model has already seen",
        )
    )
    return tiles


_POSITION_TEXT: Mapping[Position, str] = {
    "inside": "Inside",
    "above": "Above",
    "below": "Below",
}
#: Keeps a date on one line when the table is squeezed on a phone.
_NBSP = "\u00a0"


def _misses_table(misses: Sequence[Miss], step: pd.Timedelta, band: str | None) -> str:
    """Date, observed, fitted and miss; with an interval, where the point sat against it."""
    headers = ["Date", "Observed", "Fitted", "Miss"] + ([band] if band else [])

    def row(m: Miss) -> list[str]:
        cells = [
            _fmt_day(m.ds, step).replace(" ", _NBSP),
            _fmt_number(m.observed),
            _fmt_number(m.fitted),
            _signed(m.residual, False),
        ]
        if band is not None:
            cells.append("" if m.position is None else _POSITION_TEXT[m.position])
        return cells

    return table(
        headers,
        [row(m) for m in misses],
        numeric=range(1, len(headers)),
        table_id="largest-misses",
    )


def build(ctx: DashboardContext) -> Section | None:
    """The section, or None when it does not apply to this model."""
    frame = _usable(fitted_frame(ctx))
    if len(frame) < MIN_ROWS:
        return None
    view = _view(ctx, frame)
    diag = diagnose(frame, ctx.m.interval_width if view.band else None)
    step, skip = ctx.step, ctx.skip
    regular = regular_spacing(frame["ds"], step, skip)
    lag_names = [
        lag_phrase(k, step, skip, regular) for k in range(1, diag.acf.size + 1)
    ]
    unit_text = lag_names[0]
    pattern_text = lag_names[diag.pattern[0] - 1] if diag.pattern is not None else None

    window = rolling_window(diag.stats.n, step, skip)
    roll = rolling_mean(view.resid, window)
    roll_name = f"Rolling average ({lag_phrase(window, step, skip, regular)})"
    over_time = (
        "Each point is observed minus fitted. "
        + (
            f"Red points lie outside the {view.band}. "
            if view.band is not None and view.outside.any()
            else ""
        )
        + "The dark line is the rolling average, which should hover around zero. "
        + drift_sentence(frame["ds"], view.resid, window, step)
    ).strip()

    level = (
        h3("Misses against the fitted level")
        + ctx.embed(_level_figure(ctx, view, diag))
        + note(level_note(diag))
    )

    span = pd.Timestamp(frame["ds"].iloc[-1]) - pd.Timestamp(frame["ds"].iloc[0])
    kind = heatmap_kind(step, span)
    grid = mean_grid(frame["ds"], view.resid, kind) if kind is not None else None

    parts = [
        lede(
            lede_text(diag, pattern_text, additive=ctx.m.seasonality_mode == "additive")
        ),
        _tiles_html(_tiles(diag)),
        h3("Misses over time"),
        ctx.embed(_time_figure(ctx, view, roll, roll_name)),
        note(over_time),
        grid2(
            h3("Size of the misses")
            + ctx.embed(_histogram_figure(ctx, view, diag.stats))
            + note(histogram_note(diag.stats)),
            h3("Pattern left in the misses")
            + ctx.embed(_acf_figure(ctx, diag, lag_names, regular, unit_text))
            + note(acf_note(diag.acf, diag.bound)),
        ),
    ]
    if grid is None:
        parts.append(level)
    else:
        parts.append(
            grid2(
                level,
                h3("Misses on the calendar")
                + ctx.embed(_heatmap_figure(ctx, grid))
                + note(
                    "Blue cells are where the data sit below the fit, red cells "
                    f"where they sit above it. {grid_sentence(grid)}"
                ),
            )
        )
    parts += [
        h3("Largest misses"),
        _misses_table(largest_misses(frame, TOP_MISSES), step, view.band),
        note(
            "These figures describe the training data, which the model has already "
            "seen, so they flatter it. A backtest on held-back data is the fairer test."
        ),
    ]
    return Section(
        key=KEY,
        title=TITLE,
        body="".join(parts),
        summary=summary_text(diag),
    )
