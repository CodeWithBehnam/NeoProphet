"""Dashboard section "Trend": trend with its changepoints, growth rate over time and year-over-year comparison.

The numbers and the wording come from small pure functions (rates per natural
period, changepoint effects, year statistics, sentences); :func:`build` only
assembles figures and HTML from them.

Rates. The model fits the trend on scaled data: ``y`` is divided by
``y_scale`` and time is ``(ds - start) / t_scale``. A rate of one in those
units therefore climbs ``y_scale`` over the whole history (``t_scale``), so
``y_scale * period / t_scale`` converts a scaled rate into "y units per
period". For linear growth the changepoint deltas are exactly such rates. For
logistic growth ``k`` is the steepness of the curve, not a slope, so the slope
is measured from the model's own trend function instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from prophet.dashboard import _palette as palette
from prophet.dashboard._context import DashboardContext, Section
from prophet.dashboard._style import (
    SERIF,
    axes,
    h3,
    hover_date,
    layout,
    lede,
    note,
    table,
    time_axis,
    zones,
)
from prophet.plot import ACCENT, INK, INK_SOFT, OBSERVED
from prophet.visuals import (
    Tile,
    _compare_phrase,
    _day_month,
    _fmt_date,
    _fmt_day,
    _fmt_number,
    _fmt_pct,
    _tiles_html,
    _y_scale,
)

if TYPE_CHECKING:
    from prophet.forecaster import Prophet

KEY = "trend"
TITLE = "Trend"

#: Smallest mean rate change (in scaled units) that counts as a changepoint:
#: the default threshold of ``prophet.plot.add_changepoints_to_plot``. It is
#: absolute, so on its own it does not say that a change of pace is large.
THRESHOLD = 0.01
#: A change of pace under this share of the larger of the two paces is slight.
SLIGHT = 0.05
#: An interval narrower than this share of the data's scale (or of the trend
#: level, if larger) is not worth drawing or quoting.
MIN_BAND = 1e-6
#: Points drawn per line and per scatter, so that long series stay small.
MAX_POINTS = 2000
MAX_OBSERVED = 1500
#: Changepoints marked on the trend chart, largest first.
MAX_MARKED = 40
#: Years drawn in the year-over-year chart.
MAX_YEARS = 12
#: A year needs this many days of data (and three points) to be compared.
MIN_YEAR_DAYS = 14
#: A year with no observations needs a forecast spanning this many days to be drawn.
MIN_FORECAST_DAYS = 28
SMOOTH_WINDOW = "7D"
#: Step in scaled time for slopes measured from the trend function.
_EPS = 1e-6
#: Reference leap year that every date is moved to, so that all days share one axis.
_REFERENCE_YEAR = 2000

RISE = palette.DIVERGING[-1][1]
FALL = palette.DIVERGING[0][1]


# ----------------------------------------------------------------------------
# Rates: pure parts
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Period:
    """The unit a rate is quoted in, for example ``Period("day", 1 day)``."""

    name: str
    length: pd.Timedelta


_PERIODS: tuple[Period, ...] = (
    Period("hour", pd.Timedelta(hours=1)),
    Period("day", pd.Timedelta(days=1)),
    Period("week", pd.Timedelta(days=7)),
    Period("month", pd.Timedelta(days=30.4375)),
    Period("quarter", pd.Timedelta(days=91.3125)),
    Period("year", pd.Timedelta(days=365.25)),
)


def natural_period(step: pd.Timedelta) -> Period:
    """Per day for daily data, per hour for sub-daily data, per week for weekly data and so on.

    It is the longest unit that is no more than a quarter longer than the
    spacing of the data; anything finer than an hour is quoted per hour.
    """
    fits = [p for p in _PERIODS if p.length <= step * 1.25]
    return fits[-1] if fits else _PERIODS[0]


@dataclass(frozen=True)
class Scales:
    """How the model scaled the training data."""

    y: float
    t: pd.Timedelta
    start: pd.Timestamp


def model_scales(m: Prophet) -> Scales:
    if m.y_scale is None or m.t_scale is None or m.start is None:
        raise ValueError("The model has not been fitted.")
    if m.t_scale <= pd.Timedelta(0):
        raise ValueError("The training data spans no time, so it has no trend rate.")
    return Scales(float(m.y_scale), pd.Timedelta(m.t_scale), pd.Timestamp(m.start))


def rate_factor(y_scale: float, t_scale: pd.Timedelta, period: Period) -> float:
    """Multiply a rate in scaled units by this to get y units per ``period``."""
    return y_scale * (period.length / t_scale)


def changepoint_deltas(m: Prophet) -> pd.DataFrame:
    """The model's candidate changepoints in time order.

    Columns ``ds``, ``t`` (scaled time) and ``delta`` (the rate change in
    scaled units, averaged over the samples when the model was fitted with
    MCMC). Empty for flat growth and for a model without changepoints.
    """
    empty = pd.DataFrame(
        {"ds": pd.to_datetime([]), "t": np.empty(0), "delta": np.empty(0)}
    )
    changepoints, times, params = m.changepoints, m.changepoints_t, m.params
    if (
        m.growth == "flat"
        or changepoints is None
        or len(changepoints) == 0
        or times is None
        or not params
        or "delta" not in params
    ):
        return empty
    dates = pd.DatetimeIndex(pd.to_datetime(changepoints.to_numpy())).sort_values()
    delta = np.nanmean(np.atleast_2d(params["delta"]), axis=0)
    if not len(dates) == len(delta) == len(times):
        return empty
    return pd.DataFrame(
        {"ds": dates, "t": np.asarray(times, dtype=float), "delta": delta}
    )


def mean_rate(m: Prophet) -> float:
    """The base rate ``k`` in scaled units, averaged over samples."""
    return float(np.nanmean(m.params["k"]))


def linear_slopes(
    k: float, changepoints: pd.DataFrame, t: np.ndarray, factor: float
) -> np.ndarray:
    """Slope of a linear trend at scaled times ``t``, in y units per period.

    A changepoint applies from its own time onwards, as in
    ``Prophet.piecewise_linear``. ``changepoints`` is the output of
    :func:`changepoint_deltas`.
    """
    cumulative = np.concatenate(
        ([0.0], np.cumsum(changepoints["delta"].to_numpy(dtype=float)))
    )
    passed = np.searchsorted(changepoints["t"].to_numpy(dtype=float), t, side="right")
    return (k + cumulative[passed]) * factor


def capacity_curve(
    times: np.ndarray, capacities: np.ndarray
) -> Callable[[np.ndarray], np.ndarray] | None:
    """The scaled capacity as a function of scaled time, joining the known points with straight lines.

    The model's logistic trend is ``capacity(t)`` times a curve, so a capacity
    that moves over time moves the trend too. Slopes must therefore see the
    capacity at both ends of their step. Beyond the first and last point the
    nearest segment is carried on. None when no point is usable.
    """
    t = np.asarray(times, dtype=float)
    cap = np.asarray(capacities, dtype=float)
    usable = np.isfinite(t) & np.isfinite(cap)
    if not usable.any():
        return None
    order = np.argsort(t[usable], kind="stable")
    t, cap = t[usable][order], cap[usable][order]
    if len(t) == 1:
        return lambda x: np.full(np.shape(x), cap[0])
    first = (cap[1] - cap[0]) / (t[1] - t[0]) if t[1] > t[0] else 0.0
    last = (cap[-1] - cap[-2]) / (t[-1] - t[-2]) if t[-1] > t[-2] else 0.0

    def curve(x: np.ndarray) -> np.ndarray:
        at = np.asarray(x, dtype=float)
        inside = np.interp(at, t, cap)
        return (
            inside
            + np.where(at < t[0], (at - t[0]) * first, 0.0)
            + np.where(at > t[-1], (at - t[-1]) * last, 0.0)
        )

    return curve


def numeric_slopes(
    m: Prophet,
    t: np.ndarray,
    capacity: Callable[[np.ndarray], np.ndarray] | None,
    period: Period,
) -> np.ndarray:
    """Slope of the model's own trend function to the right of ``t``, in y units per period.

    A forward difference of ``Prophet.predict_trend``, so it holds for any
    growth. ``capacity`` (see :func:`capacity_curve`) is evaluated at both ends
    of the step, which makes the slope that of the plotted trend line, a moving
    capacity included. The floor cancels out of a difference.
    """
    if len(t) == 0:
        return np.empty(0)
    scales = model_scales(m)

    def trend_at(x: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame({"t": x, "floor": 0.0})
        if capacity is not None:
            frame["cap_scaled"] = capacity(x)
        return np.asarray(m.predict_trend(frame), dtype=float)

    per_unit = (trend_at(t + _EPS) - trend_at(t)) / _EPS
    return per_unit * (period.length / scales.t)


def slopes_per_period(
    m: Prophet, t: np.ndarray, cap_scaled: np.ndarray | None, period: Period
) -> np.ndarray:
    """Slope of the trend at scaled times ``t`` in y units per period (zero for flat growth).

    ``cap_scaled`` holds the scaled capacity at each ``t`` (logistic growth
    only) and ``t`` must be in time order.
    """
    if m.growth == "flat":
        return np.zeros(len(t))
    if m.growth == "linear":
        scales = model_scales(m)
        return linear_slopes(
            mean_rate(m),
            changepoint_deltas(m),
            t,
            rate_factor(scales.y, scales.t, period),
        )
    curve = None if cap_scaled is None else capacity_curve(t, cap_scaled)
    if curve is None:
        raise ValueError("Logistic growth needs the capacity at each time.")
    return numeric_slopes(m, t, curve, period)


def changepoint_effects(
    m: Prophet,
    period: Period,
    capacity: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Every changepoint with the pace of the trend before and after it.

    Columns of :func:`changepoint_deltas` plus ``before`` and ``after`` (y
    units per period), ``change`` (``after - before``) and ``significant``
    (``|delta| >= THRESHOLD``). For logistic growth ``capacity`` must hold the
    scaled times and scaled capacities of the training rows, which the
    numerical slopes need.
    """
    cps = changepoint_deltas(m)
    scales = model_scales(m)
    factor = rate_factor(scales.y, scales.t, period)
    steps = cps["delta"].to_numpy(dtype=float)
    if cps.empty:
        before = after = change = np.empty(0)
    elif m.growth == "linear":
        # The change is the delta itself; taking it as a difference of two sums would lose digits.
        change = steps * factor
        after = (mean_rate(m) + np.cumsum(steps)) * factor
        before = after - change
    else:
        curve = None if capacity is None else capacity_curve(*capacity)
        if curve is None:
            raise ValueError(
                "Logistic growth needs the capacities of the training rows."
            )
        t = cps["t"].to_numpy(dtype=float)
        before = numeric_slopes(m, t - 2 * _EPS, curve, period)
        after = numeric_slopes(m, t, curve, period)
        change = after - before
    return cps.assign(
        before=before,
        after=after,
        change=change,
        significant=np.abs(steps) >= THRESHOLD,
    )


def thin_index(
    n: int, limit: int, anchors: Sequence[int] | np.ndarray = ()
) -> np.ndarray:
    """Positions of at most about ``limit`` of ``n`` rows, evenly spread, always with ``anchors``.

    The first and last row are always kept. Anchors are the rows where a line
    bends, so that thinning never rounds off a corner.
    """
    if n <= limit:
        return np.arange(n)
    base = np.linspace(0, n - 1, limit).round().astype(int)
    extra = np.clip(np.asarray(anchors, dtype=int), 0, n - 1)
    return np.unique(np.concatenate((base, extra)))


def bar_width_ms(dates: pd.Series, fallback: pd.Timedelta) -> float:
    """Width in milliseconds for bars at ``dates``: 70% of their typical spacing."""
    gaps = pd.Series(pd.to_datetime(dates)).sort_values().diff().dropna()
    spacing = pd.Timedelta(gaps.median()) if len(gaps) else fallback
    return 0.7 * spacing.total_seconds() * 1000.0


# ----------------------------------------------------------------------------
# Wording: pure parts
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Movement:
    """How far the trend moved: ``direction`` is ``flat``, ``rose`` or ``fell``."""

    direction: str
    size: str
    relative: bool


def movement(first: float, last: float, scale: float) -> Movement:
    """Describe a move from ``first`` to ``last``.

    A move under 3% of the starting level (or of a twentieth of the data's
    scale for a start near zero) is flat. Percentages are only used when the
    start is clearly positive; otherwise the absolute change is the honest
    size.
    """
    diff = last - first
    if abs(diff) < 0.03 * max(abs(first), 0.05 * scale):
        return Movement("flat", "", False)
    direction = "fell" if diff < 0 else "rose"
    if first > 0 and last >= 0 and first >= 0.2 * scale:
        return Movement(direction, _fmt_pct(diff / first), True)
    return Movement(direction, _fmt_number(abs(diff)), False)


def movement_clause(mv: Movement) -> str:
    """'rose 43%', 'fell by 3.4', 'stayed broadly flat'."""
    if mv.direction == "flat":
        return "stayed broadly flat"
    return (
        f"{mv.direction} {mv.size}" if mv.relative else f"{mv.direction} by {mv.size}"
    )


def pace_verb(before: float, after: float) -> str:
    """What a changepoint does to the pace: 'speeds up', 'turns from rising to falling'."""
    if before >= 0 and after >= 0:
        return "speeds up" if after > before else "slows down"
    if before < 0 and after < 0:
        return "falls faster" if after < before else "falls more slowly"
    return (
        "turns from rising to falling"
        if before >= 0
        else "turns from falling to rising"
    )


@dataclass(frozen=True)
class Pace:
    """The pace of the trend before and after one changepoint, in y units per period."""

    ds: pd.Timestamp
    before: float
    after: float


def pace_range(pace: Pace, period: Period) -> str:
    """'from 0.4 to 0.7 per day'."""
    return f"from {_fmt_number(pace.before)} to {_fmt_number(pace.after)} per {period.name}"


def pace_clause(pace: Pace, period: Period) -> str:
    """'speeds up (from 0.4 to 0.7 per day)'."""
    return f"{pace_verb(pace.before, pace.after)} ({pace_range(pace, period)})"


@dataclass(frozen=True)
class Ahead:
    """Where the trend ends: the last forecast date, its value and interval if present."""

    ds: pd.Timestamp
    value: float
    lower: float | None
    upper: float | None
    interval_width: float


@dataclass(frozen=True)
class CapacityFacts:
    """The capacity of a logistic model: where it ends and how far it moves on the way."""

    last: float
    low: float
    high: float
    #: Date of ``last``, the final forecast date.
    ds: pd.Timestamp


@dataclass(frozen=True)
class TrendFacts:
    """Every number the opening sentences and tiles of the section are written from."""

    growth: str
    start: pd.Timestamp
    end: pd.Timestamp
    first: float
    last: float
    scale: float
    step: pd.Timedelta
    period: Period
    candidates: int
    significant: int
    biggest: Pace | None
    ahead: Ahead | None
    capacity: CapacityFacts | None


def _span_date(ts: pd.Timestamp, facts: TrendFacts) -> str:
    """A date worded to fit the span from the start of the history to the later of its end and ``ts``."""
    span = max(facts.end, ts) - facts.start
    if span < pd.Timedelta(days=60):
        return _fmt_day(ts, facts.step)
    return _fmt_date(ts, span)


def _day(ts: pd.Timestamp, facts: TrendFacts) -> str:
    """A date down to the day (or the minute for sub-daily data)."""
    return _fmt_day(ts, facts.step)


def _range_phrase(ahead: Ahead) -> str:
    """'the 80% range running from 69.4 to 70', or nothing when both ends read the same."""
    if ahead.lower is None or ahead.upper is None:
        return ""
    low, high = _fmt_number(ahead.lower), _fmt_number(ahead.upper)
    if low == high:
        return ""
    return f"the {ahead.interval_width:.0%} range running from {low} to {high}"


def course_sentence(f: TrendFacts) -> str:
    """How the trend moved over the history and how often it bent, in words.

    The numbers (the change, the count of changepoints, the biggest shift) are in the
    tiles under the lede, so they are not repeated here.
    """
    if f.growth == "flat":
        return (
            "The model holds the trend flat, so everything that moves in the forecast "
            "comes from its seasonal patterns, holidays and regressors."
        )
    mv = movement(f.first, f.last, f.scale)
    steady = f.candidates == 0 or f.significant == 0 or _barely_bends(f)
    if mv.direction == "flat":
        moved = "stayed broadly flat" if steady else "ended close to where it started"
    else:
        moved = mv.direction
    return f"Over the history the trend {moved}{bending_clause(f)}."


def _barely_bends(f: TrendFacts) -> bool:
    return f.biggest is not None and is_slight(f.biggest)


def bending_clause(f: TrendFacts) -> str:
    """', bending at many of the dates where it could': how often the pace changed, without counting."""
    if f.candidates == 0:
        return (
            ", and the model allowed no changepoints, so it keeps one pace throughout"
        )
    if f.significant == 0 or _barely_bends(f):
        return ", at close to one pace throughout, as no change of pace is large enough to matter"
    if f.candidates == 1:
        return ", bending at the one date where it could"
    share = f.significant / f.candidates
    how = "many" if share >= 0.5 else "several" if share >= 0.2 else "only a few"
    return f", bending at {how} of the dates where it could"


def is_slight(pace: Pace) -> bool:
    """The change of pace is under :data:`SLIGHT` of the larger of the two paces."""
    return abs(pace.after - pace.before) < SLIGHT * max(
        abs(pace.before), abs(pace.after)
    )


def ahead_sentence(f: TrendFacts) -> str:
    """Where the trend ends the forecast against where the history ends; its value and range are in the tiles."""
    a = f.ahead
    if a is None or f.growth == "flat":
        return ""
    return (
        "By the end of the forecast the trend is "
        f"{_compare_phrase(a.value, f.last, f.scale)} where the history ends."
    )


def capacity_sentence(f: TrendFacts) -> str:
    c = f.capacity
    if f.growth != "logistic" or c is None:
        return ""
    lead = "Growth is logistic, so the trend flattens as it nears the capacity"
    low, high = _fmt_number(c.low), _fmt_number(c.high)
    if low == high:
        return f"{lead} of {_fmt_number(c.last)}."
    return (
        f"{lead}, which moves between {low} and {high} and stands at "
        f"{_fmt_number(c.last)} by {_day(c.ds, f)}."
    )


def lede_text(f: TrendFacts) -> str:
    """The opening of the section: what the trend did, how often its pace changed and how the forecast ends.

    It carries what the tiles under it cannot, which is the reading of the numbers.
    """
    parts = (course_sentence(f), ahead_sentence(f), capacity_sentence(f))
    return " ".join(p for p in parts if p)


def summary_text(f: TrendFacts) -> str:
    """One sentence for the key findings at the top of the page."""
    if f.growth == "flat":
        return f"The trend is held flat at {_fmt_number(f.last)}."
    head = (
        f"The trend {movement_clause(movement(f.first, f.last, f.scale))} "
        f"between {_span_date(f.start, f)} and {_span_date(f.end, f)}"
    )
    tail: list[str] = []
    if f.significant == 1:
        tail.append("changed pace once")
    elif f.significant > 1:
        tail.append(f"changed pace at {f.significant} dates")
    if f.ahead is not None:
        tail.append(
            f"is heading for {_fmt_number(f.ahead.value)} by {_day(f.ahead.ds, f)}"
        )
    if not tail:
        return f"{head}."
    if len(tail) == 1:
        return f"{head} and {tail[0]}."
    return f"{head}, {tail[0]} and {tail[1]}."


def trend_tiles(f: TrendFacts) -> list[Tile]:
    """Key numbers under the opening sentences."""
    span = f.end - f.start
    tiles: list[Tile] = []
    if f.growth == "flat":
        tiles.append(Tile("TREND LEVEL", _fmt_number(f.last), "held flat by the model"))
    else:
        mv = movement(f.first, f.last, f.scale)
        sign = "-" if mv.direction == "fell" else "+"
        value = "Flat" if mv.direction == "flat" else f"{sign}{mv.size}"
        tiles.append(
            Tile(
                "OVERALL CHANGE",
                value,
                f"{_fmt_number(f.first)} to {_fmt_number(f.last)} since {_fmt_date(f.start, span)}",
            )
        )
        if f.candidates:
            tiles.append(
                Tile(
                    "CHANGES OF PACE",
                    f"{f.significant} of {f.candidates}",
                    "candidate dates where the pace shifts",
                )
            )
        if f.biggest is not None:
            tiles.append(
                Tile(
                    "BIGGEST CHANGE",
                    _day(f.biggest.ds, f),
                    f"{pace_verb(f.biggest.before, f.biggest.after)}, "
                    f"{pace_range(f.biggest, f.period)}",
                )
            )
    if f.ahead is not None:
        detail = f"by {_day(f.ahead.ds, f)}"
        range_text = _range_phrase(f.ahead)
        tiles.append(
            Tile(
                "HEADING FOR",
                _fmt_number(f.ahead.value),
                f"{detail}, {range_text}" if range_text else detail,
            )
        )
    return tiles


# ----------------------------------------------------------------------------
# Year over year: pure parts
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class YearStat:
    """One calendar year of observations."""

    year: int
    count: int
    first: pd.Timestamp
    last: pd.Timestamp
    mean: float
    total: float
    late_start: bool
    early_end: bool
    #: Change in the mean on the previous year as a fraction, over the days both cover.
    change: float | None


def no_days() -> pd.Series:
    """An empty series of values by day."""
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def daily_means(ds: pd.Series, values: pd.Series, min_count: float) -> pd.Series:
    """Mean of ``values`` per calendar day, leaving out days with fewer than ``min_count`` points."""
    grouped = values.groupby(ds.dt.normalize())
    means, counts = grouped.mean(), grouped.count()
    return means[counts >= min_count].astype(float)


def observed_daily(history: pd.DataFrame) -> tuple[pd.Series, float]:
    """Observed values as one mean per day, and the usual number of points per day.

    Days with under half the usual number of points (the first and last day
    of sub-daily data) are left out, since their mean would describe only
    part of the day.
    """
    frame = history.dropna(subset=["y"])
    if frame.empty:
        return no_days(), 1.0
    usual = float(frame["y"].groupby(frame["ds"].dt.normalize()).count().median())
    return daily_means(frame["ds"], frame["y"], 0.5 * usual), usual


def smooth(series: pd.Series, step: pd.Timedelta) -> tuple[pd.Series, bool]:
    """A centred 7-day mean for data at least daily, so that years can be compared by eye."""
    if series.empty or step >= pd.Timedelta(days=2):
        return series, False
    smoothed = series.rolling(SMOOTH_WINDOW, center=True, min_periods=3).mean()
    return smoothed.dropna(), True


def _years(days: pd.Series) -> pd.Index:
    """The calendar year of each date in the index of ``days``."""
    return pd.DatetimeIndex(days.index).year


def usable_years(daily: pd.Series) -> list[int]:
    """Calendar years with at least three days of data spread over two weeks or more."""
    years: list[int] = []
    for year, part in daily.groupby(_years(daily)):
        spread = part.index.max() - part.index.min()
        if len(part) >= 3 and spread >= pd.Timedelta(days=MIN_YEAR_DAYS):
            years.append(int(str(year)))
    return years


def reference_dates(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The same month and day in a leap reference year, so that every year shares one axis."""
    parts = pd.DataFrame(
        {"year": _REFERENCE_YEAR, "month": days.month, "day": days.day}
    )
    return pd.DatetimeIndex(pd.to_datetime(parts))


def _month_day(ds: pd.Series) -> pd.Series:
    return ds.dt.month * 100 + ds.dt.day


def like_for_like(previous: pd.DataFrame, current: pd.DataFrame) -> float | None:
    """Change in the mean of ``y`` from one year to the next, over the days both years cover.

    Both frames have columns ``ds`` and ``y``. A partial year is therefore
    never set against a whole one. None when the years share no days or the
    percentage would be meaningless (a non-positive base).
    """
    before, after = _month_day(previous["ds"]), _month_day(current["ds"])
    low = max(before.min(), after.min())
    high = min(before.max(), after.max())
    if low > high:
        return None
    base = previous.loc[(before >= low) & (before <= high), "y"].mean()
    now = current.loc[(after >= low) & (after <= high), "y"].mean()
    if not (np.isfinite(base) and np.isfinite(now)) or base <= 0 or now < 0:
        return None
    return float(now / base - 1.0)


def year_stats(
    history: pd.DataFrame, step: pd.Timedelta, years: Sequence[int]
) -> list[YearStat]:
    """Statistics per calendar year in ``years``, oldest first."""
    frame = history.dropna(subset=["y"])[["ds", "y"]]
    tolerance = max(pd.Timedelta(days=7), step * 1.5)
    by_year = {int(str(y)): part for y, part in frame.groupby(frame["ds"].dt.year)}
    stats: list[YearStat] = []
    for year in sorted(years):
        part = by_year[year]
        first, last = pd.Timestamp(part["ds"].min()), pd.Timestamp(part["ds"].max())
        before = by_year[year - 1] if (year - 1) in years else None
        stats.append(
            YearStat(
                year=year,
                count=len(part),
                first=first,
                last=last,
                mean=float(part["y"].mean()),
                total=float(part["y"].sum()),
                late_start=first > pd.Timestamp(year=year, month=1, day=1) + tolerance,
                early_end=last < pd.Timestamp(year=year, month=12, day=31) - tolerance,
                change=None if before is None else like_for_like(before, part),
            )
        )
    return stats


def year_label(stat: YearStat) -> str:
    """The year, with the dates it covers when it is only part of a year."""
    if stat.late_start and stat.early_end:
        return f"{stat.year} ({_day_month(stat.first)} to {_day_month(stat.last)})"
    if stat.late_start:
        return f"{stat.year} (from {_day_month(stat.first)})"
    if stat.early_end:
        return f"{stat.year} (to {_day_month(stat.last)})"
    return str(stat.year)


def _signed_pct(fraction: float) -> str:
    return f"{'-' if fraction < 0 else '+'}{_fmt_pct(fraction)}"


def year_rows(stats: Sequence[YearStat]) -> list[list[str]]:
    """Table rows, newest year first."""
    return [
        [
            year_label(s),
            f"{s.count:,}",
            _fmt_number(s.mean),
            _fmt_number(s.total),
            "n/a" if s.change is None else _signed_pct(s.change),
        ]
        for s in reversed(stats)
    ]


def year_sentence(stats: Sequence[YearStat]) -> str:
    """How the latest year compares with the one before, when both can be compared."""
    for stat in reversed(stats):
        if stat.change is None:
            continue
        previous = stat.year - 1
        if abs(stat.change) < 0.005:
            relation = "about the same as"
        else:
            relation = (
                f"{_fmt_pct(stat.change)} {'below' if stat.change < 0 else 'above'}"
            )
        return (
            f"{year_label(stat)} averaged {_fmt_number(stat.mean)}, {relation} {previous} "
            "over the days the two years share."
        )
    return ""


@dataclass(frozen=True)
class YearLine:
    """Smoothed daily values of one year: the observed part and the forecast part."""

    year: int
    observed: pd.Series
    forecast: pd.Series


def _forecast_years(forecast: pd.Series) -> list[int]:
    """Calendar years in which the forecast spans at least :data:`MIN_FORECAST_DAYS` days."""
    bounds = forecast.index.to_series().groupby(_years(forecast)).agg(["min", "max"])
    long_enough = (bounds["max"] - bounds["min"]) >= pd.Timedelta(
        days=MIN_FORECAST_DAYS - 1
    )
    return [int(str(y)) for y in bounds.index[long_enough.to_numpy()]]


def year_lines(
    observed: pd.Series,
    forecast: pd.Series,
    years: Sequence[int],
    max_years: int = MAX_YEARS,
) -> list[YearLine]:
    """The lines of the year-over-year chart, oldest year first.

    A year is drawn when it has enough observations (``years``) or when its
    forecast spans at least :data:`MIN_FORECAST_DAYS` days, however coarse the
    data. The forecast part of a year starts at its last observed point, so the
    dashed line joins the solid one.
    """
    drawn = sorted(set(years) | set(_forecast_years(forecast)))[-max_years:]
    lines: list[YearLine] = []
    for year in drawn:
        seen = observed[_years(observed) == year]
        ahead = forecast[_years(forecast) == year]
        if len(seen) and len(ahead):
            ahead = pd.concat([seen.iloc[-1:], ahead])
        lines.append(YearLine(year, seen, ahead))
    return lines


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _date_template(ctx: DashboardContext) -> str:
    return f"%{{x|{hover_date(ctx)}}}"


def _forecast_rows(ctx: DashboardContext) -> pd.DataFrame:
    """Rows of ``fcst`` with scaled time ``t`` and, for logistic growth, ``cap_scaled``."""
    m = ctx.m
    scales = model_scales(m)
    rows = ctx.fcst.copy()
    rows["t"] = (
        rows["ds"] - scales.start
    ).dt.total_seconds() / scales.t.total_seconds()
    if m.growth == "logistic" and "cap" in rows:
        floor = rows["floor"] if "floor" in rows else (m.y_min or 0.0)
        rows["cap_scaled"] = (rows["cap"] - floor) / scales.y
    return rows


def _capacity_of_history(ctx: DashboardContext) -> tuple[np.ndarray, np.ndarray] | None:
    """Scaled times and capacities of the training rows, for logistic growth."""
    if ctx.m.growth != "logistic" or "cap_scaled" not in ctx.history:
        return None
    history = ctx.history.dropna(subset=["cap_scaled", "t"])
    if history.empty:
        return None
    return (
        history["t"].to_numpy(dtype=float),
        history["cap_scaled"].to_numpy(dtype=float),
    )


def _unskipped(ctx: DashboardContext, effects: pd.DataFrame) -> pd.DataFrame:
    """Changepoints that do not fall on a skipped weekday or hour.

    A model fitted on every day can be shown with some days left out; a bar or a
    mark on such a day would sit in a closed gap.
    """
    if effects.empty or ctx.skip is None:
        return effects
    return effects[np.asarray(ctx.skip.keeps(effects["ds"]), dtype=bool)]


def _kept(ctx: DashboardContext, effects: pd.DataFrame) -> pd.DataFrame:
    """Changepoints on dates the trend chart shows (inside the forecast frame, not on skipped days)."""
    shown = _unskipped(ctx, effects)
    if shown.empty:
        return shown
    inside = (
        (shown["ds"] >= ctx.fcst["ds"].min()) & (shown["ds"] <= ctx.fcst["ds"].max())
    ).to_numpy()
    return shown[inside]


def skipped_note(count: int) -> str:
    """Says how many changepoints are left off the charts because their day is skipped."""
    if count <= 0:
        return ""
    if count == 1:
        return (
            "One changepoint falls on a day the dashboard leaves out and is not drawn."
        )
    return (
        f"{count} changepoints fall on days the dashboard leaves out and are not drawn."
    )


def has_width(m: Prophet, width: float, level: float) -> bool:
    """An interval ``width`` wide around a trend at ``level`` is wide enough to draw and to quote.

    Measured against the data's scale, not just the level, so that numerical
    noise around a trend of zero does not pass for an interval.
    """
    return bool(np.isfinite(width) and width > MIN_BAND * max(_y_scale(m), abs(level)))


def _band_rows(ctx: DashboardContext, rows: pd.DataFrame) -> pd.DataFrame | None:
    """Rows from the end of the history on that carry a trend interval of some width.

    None when intervals are not wanted or absent, when there is no forecast, or
    when the interval has no width (the trend of a flat model has no uncertainty).
    """
    if not ctx.show_band or not {"trend_lower", "trend_upper"} <= set(rows.columns):
        return None
    future = rows[rows["ds"] >= ctx.history_end]
    if not (future["ds"] > ctx.history_end).any():
        return None
    width = float((future["trend_upper"] - future["trend_lower"]).abs().max())
    level = float(future["trend"].abs().max())
    return future if has_width(ctx.m, width, level) else None


def _trend_figure(
    ctx: DashboardContext,
    rows: pd.DataFrame,
    marked: pd.DataFrame,
    band: pd.DataFrame | None,
    slopes: np.ndarray | None,
    period: Period,
) -> Any:
    go = ctx.go
    m = ctx.m
    template = _date_template(ctx)
    fig = go.Figure()
    zones(ctx, fig)

    observed = ctx.history[["ds", "y"]].dropna()
    observed = observed.iloc[thin_index(len(observed), MAX_OBSERVED)]
    fig.add_trace(
        go.Scatter(
            name="Observed",
            x=observed["ds"],
            y=observed["y"],
            mode="markers",
            marker={"color": OBSERVED, "size": 3, "opacity": 0.35},
            # Plotly shows the nearest point, so the trend still wins when the cursor is on its line.
            hovertemplate=f"{template}<br>Observed <b>%{{y:,.4g}}</b><extra></extra>",
        )
    )

    if band is not None:
        future = band.iloc[thin_index(len(band), MAX_POINTS)]
        fig.add_trace(
            go.Scatter(
                x=future["ds"],
                y=future["trend_lower"],
                mode="lines",
                line={"width": 0},
                legendgroup="trend-band",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                name=f"{m.interval_width:.0%} range of the trend",
                x=future["ds"],
                y=future["trend_upper"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=palette.BAND,
                legendgroup="trend-band",
                customdata=future["trend_lower"],
                hovertemplate=(
                    f"{template}<br>Trend range <b>%{{customdata:,.4g}} to "
                    "%{y:,.4g}</b><extra></extra>"
                ),
            )
        )

    for column, name in (("cap", "Maximum capacity"), ("floor", "Minimum capacity")):
        if (
            m.growth == "logistic"
            and column in rows
            and (column == "cap" or m.logistic_floor)
        ):
            shown = rows.iloc[thin_index(len(rows), MAX_POINTS)]
            fig.add_trace(
                go.Scatter(
                    name=name,
                    x=shown["ds"],
                    y=shown[column],
                    mode="lines",
                    line={"color": INK_SOFT, "width": 1.4, "dash": "dash"},
                    hovertemplate=f"{template}<br>{name} <b>%{{y:,.4g}}</b><extra></extra>",
                )
            )

    anchors = np.searchsorted(
        rows["t"].to_numpy(dtype=float), marked["t"].to_numpy(dtype=float)
    )
    keep = thin_index(len(rows), MAX_POINTS, anchors)
    shown = rows.iloc[keep]
    pace = (
        ""
        if slopes is None
        else f"<br>Pace <b>%{{customdata:+,.3g}}</b> per {period.name}"
    )
    fig.add_trace(
        go.Scatter(
            name="Trend",
            x=shown["ds"],
            y=shown["trend"],
            mode="lines",
            line={"color": ACCENT, "width": 2},
            customdata=None if slopes is None else slopes[keep],
            hovertemplate=f"{template}<br>Trend <b>%{{y:,.4g}}</b>{pace}<extra></extra>",
        )
    )

    if not marked.empty:
        level = np.interp(
            marked["t"].to_numpy(dtype=float),
            rows["t"].to_numpy(dtype=float),
            rows["trend"].to_numpy(dtype=float),
        )
        details = np.column_stack(
            [
                [
                    pace_verb(b, a).capitalize()
                    for b, a in zip(marked["before"], marked["after"])
                ],
                [
                    pace_range(Pace(d, b, a), period)
                    for d, b, a in zip(marked["ds"], marked["before"], marked["after"])
                ],
            ]
        )
        fig.add_trace(
            go.Scatter(
                name="Change of pace",
                x=marked["ds"],
                y=level,
                mode="markers",
                marker={"color": INK, "symbol": "diamond", "size": 9},
                customdata=details,
                hovertemplate=(
                    f"{template}<br><b>%{{customdata[0]}}</b><br>"
                    "%{customdata[1]}<extra></extra>"
                ),
            )
        )
        for ds in marked["ds"]:
            fig.add_vline(
                x=ds.to_pydatetime(),
                line={"color": INK_SOFT, "width": 1, "dash": "dot"},
                layer="below",
            )

    _, y_style = axes()
    fig.update_layout(
        **layout(
            440,
            hovermode="closest",
            hoverdistance=30,
            xaxis=time_axis(ctx),
            yaxis=y_style,
        )
    )
    return fig


def _effects_figure(
    ctx: DashboardContext, effects: pd.DataFrame, period: Period
) -> Any:
    go = ctx.go
    template = _date_template(ctx)
    # A lone changepoint has no spacing to borrow, so its bar gets a share of the chart's span.
    span = ctx.fcst["ds"].max() - ctx.fcst["ds"].min()
    width = bar_width_ms(effects["ds"], max(ctx.step, span * 0.02))
    fig = go.Figure()
    for name, colour, mask in (
        ("Pace rises", RISE, effects["change"] >= 0),
        ("Pace falls", FALL, effects["change"] < 0),
    ):
        part = effects[mask]
        if part.empty:
            continue
        details = np.column_stack(
            [
                [
                    pace_verb(b, a).capitalize()
                    for b, a in zip(part["before"], part["after"])
                ],
                [
                    pace_range(Pace(d, b, a), period)
                    for d, b, a in zip(part["ds"], part["before"], part["after"])
                ],
            ]
        )
        fig.add_trace(
            go.Bar(
                name=name,
                x=part["ds"],
                y=part["change"],
                width=width,
                marker={
                    "color": colour,
                    "opacity": np.where(part["significant"], 1.0, 0.35).tolist(),
                },
                customdata=details,
                hovertemplate=(
                    f"{template}<br><b>%{{customdata[0]}}</b><br>%{{customdata[1]}}"
                    f"<br>Change <b>%{{y:+,.4g}}</b> per {period.name}<extra></extra>"
                ),
            )
        )
    _, y_style = axes()
    fig.update_layout(
        **layout(
            320,
            hovermode="closest",
            hoverdistance=30,
            bargap=0,
            xaxis=time_axis(ctx),
            yaxis={
                **y_style,
                "zeroline": True,
                "zerolinecolor": INK_SOFT,
                "zerolinewidth": 1,
            },
        )
    )
    return fig


def _rates_figure(
    ctx: DashboardContext,
    rows: pd.DataFrame,
    slopes: np.ndarray,
    changepoints: pd.DataFrame,
    period: Period,
    percent: bool,
) -> Any:
    # Imported here, as the regressor figure does: it is the only plotly name needed beyond ctx.go.
    from plotly.subplots import make_subplots

    template = _date_template(ctx)
    n_rows = 2 if percent else 1
    titles = [f"Change in the trend per {period.name}"]
    if percent:
        # Short enough to fit a phone-width figure: plotly does not wrap subplot titles.
        titles.append(f"As a share of the trend, per {period.name}")
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.16,
        subplot_titles=titles,
    )
    for annotation in fig.layout.annotations:
        annotation.update(
            x=0,
            xref="paper",
            xanchor="left",
            align="left",
            font={"family": SERIF, "size": 15, "color": INK},
        )
    anchors = np.searchsorted(
        rows["t"].to_numpy(dtype=float), changepoints["t"].to_numpy(dtype=float)
    )
    keep = thin_index(len(rows), MAX_POINTS, anchors)
    shape = "hv" if ctx.m.growth == "linear" else "linear"
    x = rows["ds"].iloc[keep]
    series = [
        (
            slopes[keep],
            f"{template}<br>Moves <b>%{{y:+,.4g}}</b> per {period.name}<extra></extra>",
        )
    ]
    if percent:
        share = slopes / rows["trend"].to_numpy(dtype=float)
        series.append(
            (
                share[keep],
                f"{template}<br>Moves <b>%{{y:+.3%}}</b> of the trend per {period.name}<extra></extra>",
            )
        )
    for i, (values, hover) in enumerate(series, start=1):
        fig.add_trace(
            ctx.go.Scatter(
                name=f"Change per {period.name}" if i == 1 else "Share of the trend",
                x=x,
                y=values,
                mode="lines",
                line={"color": ACCENT, "width": 2, "shape": shape},
                showlegend=False,
                hovertemplate=hover,
            ),
            row=i,
            col=1,
        )
        zones(ctx, fig, row=i)
    x_style = time_axis(ctx)
    _, y_style = axes()
    fig.update_xaxes(**x_style)
    fig.update_yaxes(
        **{**y_style, "zeroline": True, "zerolinecolor": INK_SOFT, "zerolinewidth": 1}
    )
    if percent:
        fig.update_yaxes(tickformat=".1%", row=2, col=1)
    fig.update_layout(
        **layout(
            300 if n_rows == 1 else 520,
            hovermode="closest",
            hoverdistance=30,
        )
    )
    return fig


def _year_figure(ctx: DashboardContext, lines: Sequence[YearLine], label: str) -> Any:
    go = ctx.go
    fig = go.Figure()
    x_style, y_style = axes()
    latest = lines[-1].year
    before = lines[-2].year if len(lines) > 1 else None
    #: Legend groups that already have an entry; a group is one entry, whatever it holds.
    listed: set[str] = set()

    def add(
        line_year: int,
        values: pd.Series,
        name: str,
        group: str,
        what: str,
        **style: Any,
    ) -> None:
        if values.empty:
            return
        first = group not in listed
        listed.add(group)
        fig.add_trace(
            go.Scatter(
                name=name,
                x=reference_dates(pd.DatetimeIndex(values.index)),
                y=values.to_numpy(dtype=float),
                mode="lines",
                legendgroup=group,
                showlegend=first,
                hovertemplate=(
                    f"<b>{line_year}</b> {what}<br>%{{x|%d %b}}<br>"
                    f"{label} <b>%{{y:,.4g}}</b><extra></extra>"
                ),
                **style,
            )
        )

    for line in lines:
        year = str(line.year)
        if line.year == latest:
            solid: dict[str, Any] = {"line": {"color": ACCENT, "width": 2.4}}
            name, group, forecast_group = year, year, f"{year} forecast"
        elif line.year == before:
            solid = {"line": {"color": INK_SOFT, "width": 1.8}}
            name, group, forecast_group = year, year, year
        else:
            solid = {"line": {"color": OBSERVED, "width": 1.2}, "opacity": 0.6}
            name, group, forecast_group = "Earlier years", "earlier", "earlier"
        dashed = {**solid, "line": {**solid["line"], "dash": "dash"}}
        add(line.year, line.observed, name, group, "", **solid)
        # A year other than the latest has a forecast part only when the forecast crosses a year end.
        forecast_name = f"{year} forecast" if forecast_group != "earlier" else name
        add(
            line.year,
            line.forecast,
            forecast_name,
            forecast_group,
            "forecast",
            **dashed,
        )
    fig.update_layout(
        **layout(
            400,
            hovermode="closest",
            hoverdistance=30,
            xaxis={
                **x_style,
                "type": "date",
                "range": [f"{_REFERENCE_YEAR}-01-01", f"{_REFERENCE_YEAR}-12-31"],
                "tickformat": "%b",
                "dtick": "M1",
            },
            yaxis=y_style,
        )
    )
    return fig


# ----------------------------------------------------------------------------
# Section
# ----------------------------------------------------------------------------


def _fitted_trend(ctx: DashboardContext) -> np.ndarray:
    """The trend at every training row, from the model's own parameters."""
    return np.asarray(ctx.m.predict_trend(ctx.history), dtype=float)


def _effects(ctx: DashboardContext, period: Period) -> pd.DataFrame:
    """:func:`changepoint_effects`, or the changepoints without pace when logistic capacity is missing."""
    capacity = _capacity_of_history(ctx)
    if ctx.m.growth == "logistic" and capacity is None:
        cps = changepoint_deltas(ctx.m)
        return cps.assign(
            before=np.nan,
            after=np.nan,
            change=np.nan,
            significant=cps["delta"].abs() >= THRESHOLD,
        )
    return changepoint_effects(ctx.m, period, capacity)


def _largest(effects: pd.DataFrame, n: int) -> pd.DataFrame:
    """The ``n`` changepoints with the largest rate change, in time order."""
    order = effects["delta"].abs().sort_values(ascending=False).index[:n]
    return effects.loc[order].sort_values("ds")


def _ahead(ctx: DashboardContext) -> Ahead | None:
    future = ctx.future
    if future.empty or "trend" not in future:
        return None
    last = future.iloc[-1]
    lower = upper = None
    if ctx.show_band and "trend_lower" in future and "trend_upper" in future:
        low, high = float(last["trend_lower"]), float(last["trend_upper"])
        # A band of no width (a trend without uncertainty) is not worth a sentence.
        if has_width(ctx.m, high - low, float(last["trend"])):
            lower, upper = low, high
    return Ahead(
        ds=pd.Timestamp(last["ds"]),
        value=float(last["trend"]),
        lower=lower,
        upper=upper,
        interval_width=float(ctx.m.interval_width),
    )


def _capacity_facts(ctx: DashboardContext) -> CapacityFacts | None:
    """The capacity over the whole forecast frame, for logistic growth."""
    if ctx.m.growth != "logistic" or "cap" not in ctx.fcst:
        return None
    cap = ctx.fcst["cap"].to_numpy(dtype=float)
    finite = np.flatnonzero(np.isfinite(cap))
    if not len(finite):
        return None
    return CapacityFacts(
        last=float(cap[finite[-1]]),
        low=float(cap[finite].min()),
        high=float(cap[finite].max()),
        ds=pd.Timestamp(ctx.fcst["ds"].iloc[finite[-1]]),
    )


def trend_facts(
    ctx: DashboardContext, effects: pd.DataFrame, period: Period
) -> TrendFacts:
    """Gather the numbers the opening sentences are written from."""
    fitted = _fitted_trend(ctx)
    significant = effects[effects["significant"]] if len(effects) else effects
    biggest: Pace | None = None
    change = significant["change"].to_numpy(dtype=float)
    if len(change) and np.isfinite(change).any():
        top = int(np.nanargmax(np.abs(change)))
        biggest = Pace(
            pd.Timestamp(significant["ds"].to_numpy()[top]),
            float(significant["before"].to_numpy(dtype=float)[top]),
            float(significant["after"].to_numpy(dtype=float)[top]),
        )
    capacity = _capacity_facts(ctx)
    return TrendFacts(
        growth=str(ctx.m.growth),
        start=pd.Timestamp(ctx.history["ds"].iloc[0]),
        end=ctx.history_end,
        first=float(fitted[0]),
        last=float(fitted[-1]),
        scale=_y_scale(ctx.m),
        step=ctx.step,
        period=period,
        candidates=len(effects),
        significant=len(significant),
        biggest=biggest,
        ahead=None if ctx.m.growth == "flat" else _ahead(ctx),
        capacity=capacity,
    )


def _trend_note(
    ctx: DashboardContext,
    f: TrendFacts,
    has_band: bool,
    n_marked: int,
    n_significant: int,
    n_skipped: int = 0,
) -> str:
    text = (
        "Faint dots are the observed values; the line is the trend alone, before "
        "seasonality, holidays and regressors are added."
    )
    if f.growth == "flat":
        text += " The model holds the trend flat, so it is a straight horizontal line."
    if has_band:
        text += (
            f" The shaded band is the range the trend should stay within {ctx.m.interval_width:.0%} "
            "of the time, widening as the forecast looks further ahead."
        )
    if n_marked:
        text += (
            " Diamonds and dotted lines mark the changepoints where the model changes the pace "
            "by more than the small threshold Prophet's own plot uses; hover for what changed."
        )
        if n_significant > n_marked:
            text += f" The {n_marked} largest of {n_significant} are marked."
    if f.growth == "logistic" and f.capacity is not None:
        text += " Dashed lines are the capacity the trend cannot pass."
    if n_skipped:
        text += f" {skipped_note(n_skipped)}"
    return text


def _effects_note(m: Prophet, period: Period, n_skipped: int = 0) -> str:
    text = (
        "Each bar is how much the pace of the trend changes at one changepoint, in the units of "
        f"the data per {period.name}. Claret bars make the trend rise faster (or fall more slowly) and "
        "blue bars make it rise more slowly (or fall faster). Faint bars are too small to count "
        "and are not marked on the trend chart."
    )
    if m.growth == "logistic":
        text += (
            " With logistic growth the model changes how steeply the curve climbs towards its "
            "capacity, so the bars measure the change in the slope of the trend on that date, "
            "which also depends on how close the trend already is to capacity, and on any "
            "movement of the capacity itself."
        )
    if n_skipped:
        text += f" {skipped_note(n_skipped)}"
    return text


def _rates_note(m: Prophet, percent: bool) -> str:
    if m.growth == "logistic":
        text = (
            "With logistic growth the pace is the slope of the trend line itself, which shrinks "
            "as the trend nears capacity and follows the capacity when that moves. It is measured "
            "from the model's own trend function."
        )
    else:
        text = (
            "Between changepoints the trend moves at a constant pace, so the line is level except "
            "where a changepoint bends it. Beyond the history it carries on at the last pace."
        )
    if percent:
        text += (
            " Part of the model multiplies the trend (seasonality, holidays or regressors in "
            "multiplicative mode), so the lower panel restates the pace as a share of the "
            "trend itself."
        )
    return text


def _year_note(
    smoothed: bool,
    sub_daily: bool,
    n_years: int,
    n_drawn: int,
    lines: Sequence[YearLine],
) -> str:
    text = (
        "One line per year on a shared calendar. The latest year is in claret, the year "
        "before in grey and older years in lighter grey."
    )
    if any(not ln.forecast.empty for ln in lines):
        text += " Dashed lines are the forecast."
    if sub_daily:
        text += " Data finer than a day are averaged by day first."
    if smoothed:
        text += " Lines are 7-day averages."
    if n_years > n_drawn:
        text += f" Only the latest {n_drawn} of {n_years} years are drawn."
    return text


def _year_section(ctx: DashboardContext) -> str | None:
    """The year-over-year chart and table, or None when fewer than two years can be compared."""
    daily, usual = observed_daily(ctx.history)
    if daily.empty:
        return None
    years = usable_years(daily)
    if len(years) < 2:
        return None
    smoothed_obs, smoothed = smooth(daily, ctx.step)
    future = ctx.future
    if "yhat" in future and not future.empty:
        ahead = daily_means(future["ds"], future["yhat"], 0.5 * usual)
        smoothed_ahead, _ = smooth(ahead, ctx.step)
    else:
        smoothed_ahead = no_days()
    lines = year_lines(smoothed_obs, smoothed_ahead, years)
    stats = year_stats(ctx.history, ctx.step, years)
    label = "7-day average" if smoothed else "Value"
    parts = [
        h3("Year over year"),
        ctx.embed(_year_figure(ctx, lines, label)),
        note(
            _year_note(
                smoothed,
                ctx.step < pd.Timedelta(hours=20),
                len(years),
                len([ln for ln in lines if ln.year in years]),
                lines,
            )
        ),
    ]
    sentence = year_sentence(stats)
    if sentence:
        parts.append(note(sentence))
    parts.append(
        table(
            ["Year", "Observations", "Mean", "Total", "Change on previous year"],
            year_rows(stats),
            numeric=[1, 2, 3, 4],
            table_id="trend-years",
        )
    )
    parts.append(
        note(
            "Years with under two weeks of data are left out. The change on the previous year "
            "compares only the days both years cover, so a partial year is never set against "
            "a whole one."
        )
    )
    return "".join(parts)


def _row_slopes(
    ctx: DashboardContext, rows: pd.DataFrame, period: Period
) -> np.ndarray | None:
    """The pace of the trend at every forecast row.

    None for flat growth (there is no pace to speak of) and for logistic growth
    when the forecast carries no capacities.
    """
    m = ctx.m
    if m.growth == "flat":
        return None
    cap = rows["cap_scaled"].to_numpy(dtype=float) if "cap_scaled" in rows else None
    if m.growth == "logistic" and (cap is None or not np.isfinite(cap).any()):
        return None
    return slopes_per_period(m, rows["t"].to_numpy(dtype=float), cap, period)


def build(ctx: DashboardContext) -> Section | None:
    """The section, or None when it does not apply to this model."""
    m = ctx.m
    if ctx.history.empty or ctx.fcst.empty or "trend" not in ctx.fcst:
        return None
    if m.t_scale is None or m.t_scale <= pd.Timedelta(0):
        return None

    period = natural_period(ctx.step)
    effects = _effects(ctx, period)
    facts = trend_facts(ctx, effects, period)
    rows = _forecast_rows(ctx)

    band = _band_rows(ctx, rows)
    slopes = _row_slopes(ctx, rows, period)
    shown = _unskipped(ctx, effects)
    kept = _kept(ctx, effects)
    significant = kept[kept["significant"]]
    marked = _largest(significant.dropna(subset=["change"]), MAX_MARKED)
    left_out = len(effects) - len(shown)
    marks_left_out = int(effects["significant"].sum() - shown["significant"].sum())

    parts = [
        lede(lede_text(facts)),
        _tiles_html(trend_tiles(facts)),
        h3(
            "Trend level against the data"
            if m.growth == "flat"
            else "Trend and changepoints"
        ),
        ctx.embed(_trend_figure(ctx, rows, marked, band, slopes, period)),
        note(
            _trend_note(
                ctx,
                facts,
                band is not None,
                len(marked),
                len(significant),
                marks_left_out,
            )
        ),
    ]

    bars = shown.dropna(subset=["change"])
    if not bars.empty:
        parts += [
            h3("Change of pace at each changepoint"),
            ctx.embed(_effects_figure(ctx, bars, period)),
            note(_effects_note(m, period, left_out)),
        ]

    if slopes is not None:
        mixes_in_trend = "multiplicative_terms" in rows and bool(
            (rows["multiplicative_terms"].abs() > 0).any()
        )
        percent = mixes_in_trend and bool((rows["trend"] > 0).all())
        parts += [
            h3("Growth rate over time"),
            ctx.embed(
                _rates_figure(ctx, rows, slopes, changepoint_deltas(m), period, percent)
            ),
            note(_rates_note(m, percent)),
        ]

    years = _year_section(ctx)
    if years is not None:
        parts.append(years)

    return Section(KEY, TITLE, "".join(parts), summary_text(facts))
