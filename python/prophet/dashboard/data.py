"""Dashboard section "Data": data quality (gaps, missing values, duplicates) and the forecast table with CSV download."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from prophet.dashboard import _style
from prophet.dashboard._context import DashboardContext, Section
from prophet.dashboard._palette import ACCENT, BAND, OBSERVED
from prophet.plot import _hour_runs
from prophet.visuals import (
    _duration_phrase,
    _fmt_day,
    _fmt_pct,
    _frequency_with_skips,
    _plural,
    _step_unit,
)

if TYPE_CHECKING:
    from prophet.plot import SkipAxis

KEY = "data"
TITLE = "Data"

#: Rows the on-page forecast table shows.
MAX_TABLE_ROWS = 500
#: Rows in the embedded CSV. Past this the page would carry megabytes of text, and
#: Copy CSV would put all of it on the clipboard, so the CSV keeps the future and
#: the latest history.
MAX_CSV_ROWS = 20_000
#: Bars in the gap chart.
MAX_BARS = 90
#: Gaps that get a triangle to hover, and gaps that get a shaded stripe.
MAX_MARKED_GAPS = 12
MAX_SHADED_GAPS = 300
#: Rows in the "longest gaps" table.
MAX_GAP_ROWS = 5
#: Share of its slot a bar fills; the rest is the space between bars.
BAR_FILL = 0.85
CSV_FILENAME = "forecast.csv"

# Observations per bar, so that bars cover a round number of points.
_BUCKET_SIZES = (1, 2, 3, 4, 5, 6, 7, 10, 14, 15, 20, 21, 28, 30, 45, 60, 90) + (
    120,
    180,
    365,
    730,
    1460,
    2920,
    5840,
)
_WEEKEND_DAYS = (5, 6)
#: Share of missing timestamps that must fall on weekends (or on hours the data never
#: has) before we suggest skipping them.
_WEEKEND_SHARE = 0.9
_HOURS_SHARE = 0.9
#: Missing timestamps looked at when judging which hours of the day are missing.
_HOURS_SAMPLE = 20_000

# ----------------------------------------------------------------------------
# Pure: coverage of the timestamps
# ----------------------------------------------------------------------------

#: Added to a position before it is turned back into a time, so that a position
#: that float error left a hair under a kept hour does not resolve to the skipped
#: period before it. About 86 microseconds, which the millisecond rounding removes.
_EDGE_DAYS = 1e-9
#: Jumps this close to the median (as a share of it) make up the typical step.
_NEAR_MEDIAN = (0.75, 1.25)
#: Share of timestamps that may land on a grid point already taken before the
#: spacing is called uneven.
_UNEVEN_SHARE = 0.05
_MONTH_DAYS = 30.4375
_MONTH_STEPS = (1, 2, 3, 4, 6, 12)
_MONTH_TOLERANCE = 0.02


@dataclass(frozen=True)
class Clock:
    """Timestamps on a number line of days where skipped days and hours take no room."""

    skip: SkipAxis | None
    origin: pd.Timestamp

    def to_pos(self, dates: pd.DatetimeIndex) -> np.ndarray:
        if self.skip is not None:
            return np.asarray(self.skip.to_x(dates), dtype=float)
        return np.asarray((dates - self.origin) / pd.Timedelta(days=1), dtype=float)

    def to_time(self, pos: np.ndarray | Sequence[float]) -> pd.DatetimeIndex:
        """Timestamps for positions, rounded to the millisecond to drop float noise."""
        pos = np.asarray(pos, dtype=float)
        if self.skip is not None:
            return self.skip.from_x(pos + _EDGE_DAYS)
        return pd.DatetimeIndex(self.origin + pd.to_timedelta(pos, unit="D")).round(
            "ms"
        )


@dataclass(frozen=True)
class Grid:
    """The regular grid of expected timestamps, numbered from 0 at the first one.

    ``slots`` are the grid numbers the observations fall on, sorted and unique;
    ``positions`` and ``stamps`` are where and when the first observation of
    each slot sits. Numbering by the rounded ratio of each jump to the typical
    spacing keeps months of unequal length on the grid. Turning a grid number
    back into a time interpolates between the observations around it instead
    of multiplying by the typical step, so a month of 28 days or one of 31 does
    not push later dates off by a few days.
    """

    clock: Clock
    step: float
    slots: np.ndarray = field(compare=False, repr=False)
    positions: np.ndarray = field(compare=False, repr=False)
    stamps: pd.DatetimeIndex = field(compare=False, repr=False)
    #: Every timestamp is at midnight, so labels need no time of day.
    whole_days: bool = True
    #: Months per step for monthly, quarterly or yearly data that sits on the same
    #: day of the month (or on month ends), else 0. Labels then follow the calendar.
    months: int = 0
    month_end: bool = False

    def position(self, slots: Sequence[float] | np.ndarray) -> np.ndarray:
        """Clock positions of grid numbers; fractions land between two grid points."""
        wanted = np.asarray(slots, dtype=float)
        first, last = float(self.slots[0]), float(self.slots[-1])
        inside = np.interp(wanted, self.slots, self.positions)
        before = self.positions[0] + (wanted - first) * self.step
        after = self.positions[-1] + (wanted - last) * self.step
        return np.where(wanted < first, before, np.where(wanted > last, after, inside))

    def times(self, slots: Sequence[float] | np.ndarray) -> pd.DatetimeIndex:
        """Timestamps of grid numbers, for placing marks."""
        return self.clock.to_time(self.position(slots))

    def labels(self, slots: Sequence[float] | np.ndarray) -> pd.DatetimeIndex:
        """Timestamps of whole grid numbers, for naming them.

        A grid number with an observation gives that observation's own timestamp.
        One inside a gap is the date the series would have had: on the calendar
        for monthly data, else interpolated, to whole days when the data has no
        time of day.
        """
        wanted = np.asarray(slots, dtype=float)
        found = self.times(wanted)
        out = list(found.round("D") if self.whole_days else found)
        last_seen = np.searchsorted(self.slots, wanted, side="right") - 1
        for j in np.flatnonzero(wanted == np.rint(wanted)):
            i = max(int(last_seen[j]), 0)
            if self.slots[i] == wanted[j]:
                out[j] = self.stamps[i]
            elif self.months:
                out[j] = self._on_calendar(i, int(wanted[j]) - int(self.slots[i]))
        return pd.DatetimeIndex(out)

    def _on_calendar(self, i: int, steps: int) -> pd.Timestamp:
        """The timestamp ``steps`` steps after the observation ``i`` of monthly data."""
        anchor, months = self.stamps[i], steps * self.months
        if self.month_end:
            return anchor + pd.offsets.MonthEnd(months)
        return anchor + pd.DateOffset(months=months)


@dataclass(frozen=True)
class Gap:
    """A run of expected timestamps with no observation."""

    #: The last observation before the run and the first after it.
    before: pd.Timestamp
    after: pd.Timestamp
    missing: int
    #: Grid number of the observation before the run.
    slot: int


@dataclass(frozen=True)
class Coverage:
    """How many timestamps there are against how many a regular series would have.

    When the spacing is uneven (``regular`` is False) there is no regular series
    to compare with: ``expected`` and ``present`` are both the number of distinct
    timestamps and there are no gaps.
    """

    rows: int
    first: pd.Timestamp
    last: pd.Timestamp
    expected: int
    present: int
    gaps: tuple[Gap, ...]
    grid: Grid = field(compare=False, repr=False)
    regular: bool = True

    @property
    def missing(self) -> int:
        return self.expected - self.present

    @property
    def share_missing(self) -> float:
        return self.missing / self.expected if self.expected else 0.0

    @property
    def longest(self) -> Gap | None:
        """The longest gap; the earliest of equally long ones."""
        return max(self.gaps, key=lambda g: (g.missing, -g.slot), default=None)


def typical_step(jumps: np.ndarray) -> float:
    """The usual jump between neighbours: the mean of the jumps close to the median.

    The median alone is one real spacing, 31 days for monthly data. The mean of
    the jumps around it (30.4 days for months) is the spacing that a long run of
    missing months adds up to, so a gap of two years counts 24 months.
    """
    positive = jumps[jumps > 0]
    if not positive.size:
        return 1.0
    median = float(np.median(positive))
    low, high = _NEAR_MEDIAN
    near = positive[(positive >= low * median) & (positive <= high * median)]
    return float(near.mean()) if near.size else median


def whole_months(step: float, dates: pd.DatetimeIndex, skip: SkipAxis | None) -> int:
    """Months per step when ``dates`` are monthly, quarterly, half-yearly or yearly, else 0.

    The step must be close to a whole number of months and every date must fall
    on the same day of its month (or every one on a month end), so that 60 days
    between points or 28 days between points are not taken for months.
    """
    if skip is not None:
        return 0
    n = round(step / _MONTH_DAYS)
    if n not in _MONTH_STEPS or abs(step / (n * _MONTH_DAYS) - 1) > _MONTH_TOLERANCE:
        return 0
    same_day = bool((dates.day == dates[0].day).all())
    return n if same_day or bool(dates.is_month_end.all()) else 0


def coverage(ds: pd.Series, skip: SkipAxis | None = None) -> Coverage:
    """Gaps in a series of timestamps, judged against its own typical spacing.

    The spacing is the typical jump between neighbours (see :func:`typical_step`),
    measured with skipped weekdays and hours closed up, so a weekday series is
    not full of gaps once ``skip`` says its weekends never appear. Timestamps
    that fall on the same grid point count once. When more than a few of them
    do although their times differ, the spacing is uneven (or more than half
    the points are missing, which looks the same), and no gaps are claimed.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(ds).dropna()).sort_values()
    if dates.empty:
        raise ValueError("There are no timestamps to describe.")
    clock = Clock(skip, dates[0])
    pos = clock.to_pos(dates)
    jumps = np.diff(pos)
    step = typical_step(jumps)
    slot = np.concatenate(([0.0], np.cumsum(np.rint(jumps / step)))).astype(np.int64)
    slots, first = np.unique(slot, return_index=True)
    months = whole_months(step, dates, skip)
    grid = Grid(
        clock,
        step,
        slots,
        pos[first],
        dates[first],
        whole_days=bool((dates == dates.normalize()).all()),
        months=months,
        month_end=bool(dates.is_month_end.all()),
    )
    distinct = len(np.unique(pos))
    if distinct - len(slots) > _UNEVEN_SHARE * distinct:
        return Coverage(
            rows=len(dates),
            first=dates[0],
            last=dates[-1],
            expected=distinct,
            present=distinct,
            gaps=(),
            grid=grid,
            regular=False,
        )
    holes = np.diff(slots) - 1
    gaps = tuple(
        Gap(
            before=dates[int(first[i + 1]) - 1],
            after=dates[int(first[i + 1])],
            missing=int(holes[i]),
            slot=int(slots[i]),
        )
        for i in np.flatnonzero(holes > 0)
    )
    return Coverage(
        rows=len(dates),
        first=dates[0],
        last=dates[-1],
        expected=int(slots[-1]) + 1,
        present=len(slots),
        gaps=gaps,
        grid=grid,
    )


# ----------------------------------------------------------------------------
# Pure: the data the caller passed
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedFacts:
    """Problems in the observed data frame, counted in rows."""

    rows: int
    duplicates: int
    missing_values: int
    zeros: int
    negatives: int
    before: int
    after: int


def naive_dates(values: pd.Series) -> pd.Series:
    """Timestamps as datetimes without a timezone (wall-clock time is kept).

    Values that cannot be read become NaT instead of raising, as Prophet has
    already accepted the frame they come from for training.
    """
    ds = pd.to_datetime(values, errors="coerce")
    return ds.dt.tz_localize(None) if ds.dt.tz is not None else ds


def observed_facts(
    df: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp
) -> ObservedFacts | None:
    """Count what is odd about ``df`` against the training range ``first`` to ``last``.

    ``duplicates`` counts rows that repeat an earlier timestamp. Returns None
    when ``df`` has no ``ds`` column. Dates and values that cannot be read are
    treated as missing instead of raising.
    """
    if "ds" not in df:
        return None
    ds = naive_dates(df["ds"])
    if "y" in df:
        y = pd.to_numeric(df["y"], errors="coerce")
        missing_values = int(y.isna().sum())
        zeros = int((y == 0).sum())
        negatives = int((y < 0).sum())
    else:
        missing_values = zeros = negatives = 0
    return ObservedFacts(
        rows=len(df),
        duplicates=int(ds.dropna().duplicated().sum()),
        missing_values=missing_values,
        zeros=zeros,
        negatives=negatives,
        before=int((ds < first).sum()),
        after=int((ds > last).sum()),
    )


# ----------------------------------------------------------------------------
# Pure: wording
# ----------------------------------------------------------------------------


def _count(n: int) -> str:
    return f"{n:,}" if n else "None"


def spacing_clause(step: pd.Timedelta, skip: SkipAxis | None) -> str:
    """'one point per day', 'one point per weekday', 'one point every 5 minutes'."""
    _, detail = _frequency_with_skips(step, skip)
    if detail.startswith("one point"):
        return detail
    return f"one point every {_duration_phrase(step)}"


def unit_word(step: pd.Timedelta, skip: SkipAxis | None) -> str:
    """'day', 'weekday', 'hour' or, when the spacing matches no calendar unit, 'point'."""
    unit = _step_unit(step)
    if unit == "day" and skip is not None and set(skip.skip_days) == set(_WEEKEND_DAYS):
        return "weekday"
    return unit or "point"


def points_phrase(n: int, step: pd.Timedelta, skip: SkipAxis | None) -> str:
    """'9 days', '5 weekdays', '36 points': a count in the unit of the data's spacing."""
    return _plural(n, unit_word(step, skip))


def gap_phrase(gap: Gap, step: pd.Timedelta, skip: SkipAxis | None) -> str:
    return (
        f"{points_phrase(gap.missing, step, skip)} missing between "
        f"{_fmt_day(gap.before, step)} and {_fmt_day(gap.after, step)}"
    )


def missing_weekday_counts(gaps: Sequence[Gap]) -> np.ndarray:
    """How many missing days fall on each weekday (Monday is 0), for daily data."""
    counts = np.zeros(7, dtype=np.int64)
    for gap in gaps:
        full, rest = divmod(gap.missing, 7)
        counts += full
        for offset in range(1, rest + 1):
            counts[(gap.before.dayofweek + offset) % 7] += 1
    return counts


def looks_like_weekdays_only(
    cov: Coverage, step: pd.Timedelta, skip: SkipAxis | None
) -> bool:
    """Daily data whose missing days are nearly all Saturdays and Sundays."""
    if skip is not None or _step_unit(step) != "day" or not cov.gaps:
        return False
    counts = missing_weekday_counts(cov.gaps)
    weekend = int(counts[5] + counts[6])
    if weekend < _WEEKEND_SHARE * counts.sum():
        return False
    return bool(min(counts[5], counts[6]) >= 0.25 * weekend)


def missing_slots(cov: Coverage, limit: int = _HOURS_SAMPLE) -> np.ndarray:
    """Grid numbers of the missing timestamps; a fixed random sample when there are more than ``limit``."""
    lengths = np.array([g.missing for g in cov.gaps], dtype=np.int64)
    starts = np.array([g.slot + 1 for g in cov.gaps], dtype=np.int64)
    total = int(lengths.sum())
    if total <= limit:
        pick = np.arange(total)
    else:
        pick = np.sort(np.random.default_rng(0).integers(0, total, size=limit))
    ends = np.cumsum(lengths)
    which = np.searchsorted(ends, pick, side="right")
    return starts[which] + (pick - (ends[which] - lengths[which]))


def unused_hours(
    cov: Coverage, step: pd.Timedelta, skip: SkipAxis | None
) -> tuple[int, ...]:
    """Hours of the day the data never has, when nearly every missing timestamp falls in them.

    That is data from fixed hours, such as business hours, passed without
    ``skip_hours``. Empty when the data is not sub-daily, already skips hours,
    or its holes are spread over hours that do have data.
    """
    if not cov.regular or not cov.gaps or step > pd.Timedelta(hours=12):
        return ()
    if skip is not None and skip.skip_hours:
        return ()
    seen = np.bincount(cov.grid.stamps.hour, minlength=24) > 0
    if seen.all():
        return ()
    holes = np.bincount(cov.grid.times(missing_slots(cov)).hour, minlength=24)
    unseen = np.flatnonzero(~seen)
    if holes[unseen].sum() < _HOURS_SHARE * holes.sum():
        return ()
    return tuple(int(h) for h in unseen)


def skip_hours_argument(hours: Sequence[int]) -> str:
    """Source for a ``skip_hours`` argument: ``range(0, 8)`` or ``[*range(0, 9), *range(17, 24)]``."""
    spans: list[tuple[int, int]] = []
    for first, last in _hour_runs(sorted(hours)):
        if first <= last:
            spans.append((first, last + 1))
        else:
            spans += [(first, 24), (0, last + 1)]
    parts = [f"range({a}, {b})" for a, b in sorted(spans)]
    return (
        parts[0] if len(parts) == 1 else "[" + ", ".join(f"*{p}" for p in parts) + "]"
    )


def hours_text(hours: Sequence[int]) -> str:
    """``from 17:00 to 09:00``, one phrase per run of hours, joined with ``and``."""
    return " and ".join(
        f"from {first:02d}:00 to {(last + 1) % 24:02d}:00"
        for first, last in _hour_runs(sorted(hours))
    )


WEEKEND_HINT = (
    "Nearly every missing day is a Saturday or a Sunday, so this looks like a "
    'weekday series. Pass skip_days="weekends" (to forecast() or '
    "dashboard_html()) and they will no longer count as missing."
)


@dataclass(frozen=True)
class Pattern:
    """A regular shape in what is missing, and what to tell the reader about it."""

    kind: Literal["weekends", "hours"]
    #: A paragraph that says what to pass to skip it.
    hint: str
    #: The end of the summary clause: ``", nearly all of them weekend days"``.
    clause: str


def missing_pattern(
    cov: Coverage, step: pd.Timedelta, skip: SkipAxis | None
) -> Pattern | None:
    """Weekends or hours of the day that account for nearly everything missing, or None."""
    if looks_like_weekdays_only(cov, step, skip):
        return Pattern("weekends", WEEKEND_HINT, ", nearly all of them weekend days")
    hours = unused_hours(cov, step, skip)
    if not hours:
        return None
    hint = (
        f"Nearly every missing timestamp falls {hours_text(hours)}, and the data has "
        "no point at those hours on any day, so this looks like data from fixed "
        f"hours of the day. Pass skip_hours={skip_hours_argument(hours)} (to "
        "forecast() or dashboard_html()) and they will no longer count as missing."
    )
    return Pattern(
        "hours", hint, ", nearly all of them outside the hours the data covers"
    )


def share_words(share: float) -> str:
    """How much of the expected timestamps is missing, without the number the list beside it gives."""
    if share < 0.01:
        return "Hardly any"
    if share < 0.05:
        return "A few"
    if share < 0.2:
        return "Some"
    if share < 0.5:
        return "A large share"
    return "Most"


#: Gaps at least this many make a "many short gaps" series, unless one gap holds most of what is missing.
MANY_GAPS = 5
#: One gap holding this share of the missing timestamps is "one long gap".
ONE_LONG_GAP = 0.5


def gap_character(cov: Coverage) -> str:
    """Whether the holes are one long gap, a few gaps or many short ones."""
    longest = cov.longest
    if longest is None or not cov.missing:
        return ""
    if len(cov.gaps) == 1:
        return "all in one gap"
    if longest.missing >= ONE_LONG_GAP * cov.missing:
        return "mostly in one long gap"
    if len(cov.gaps) >= MANY_GAPS:
        return "spread over many short gaps"
    return "in a few gaps"


def lede_text(
    cov: Coverage, step: pd.Timedelta, skip: SkipAxis | None, span: str
) -> str:
    """The opening paragraph: how complete the series is and what the holes look like.

    The counts (rows, range, spacing, expected, missing, gaps) are in the list under it.
    """
    if not cov.regular:
        return (
            "The observations are unevenly spaced, so there is no regular series "
            "to count missing timestamps against."
        )
    if not cov.missing:
        return "The series is complete: every timestamp expected at this spacing is present."
    return (
        f"{share_words(cov.share_missing)} of the timestamps expected at this spacing "
        f"are missing, {gap_character(cov)}."
    )


def summary_text(
    cov: Coverage,
    step: pd.Timedelta,
    skip: SkipAxis | None,
    facts: ObservedFacts | None,
    pattern: Pattern | None,
) -> str:
    """One sentence for the key findings, empty when the data looks complete."""
    clauses: list[str] = []
    longest = cov.longest
    if not cov.regular:
        clauses.append("timestamps are unevenly spaced, so gaps were not counted")
    elif cov.share_missing >= 0.01 and longest is not None:
        clause = f"{_fmt_pct(cov.share_missing)} of the expected timestamps are missing"
        if pattern is not None:
            clause += pattern.clause
        else:
            clause += f" (longest gap {points_phrase(longest.missing, step, skip)})"
        clauses.append(clause)
    if facts is not None:
        if facts.duplicates:
            clauses.append(f"{_plural(facts.duplicates, 'timestamp')} repeated")
        if facts.missing_values:
            clauses.append(f"{_plural(facts.missing_values, 'value')} missing")
    if not clauses:
        return ""
    text = "; ".join(clauses)
    return f"{text[:1].upper()}{text[1:]}."


def coverage_pairs(
    cov: Coverage, step: pd.Timedelta, skip: SkipAxis | None, span: str
) -> list[tuple[str, str]]:
    first = [
        ("Rows used for training", f"{cov.rows:,}"),
        (
            "Date range",
            f"{_fmt_day(cov.first, step)} to {_fmt_day(cov.last, step)} ({span})",
        ),
    ]
    if not cov.regular:
        return [
            *first,
            ("Spacing", f"Uneven, typically about {_duration_phrase(step)} apart"),
            ("Timestamps missing", "Not counted: the spacing is uneven"),
        ]
    clause = spacing_clause(step, skip)
    longest = cov.longest
    return [
        *first,
        ("Spacing", clause[:1].upper() + clause[1:]),
        ("Timestamps expected", f"{cov.expected:,}"),
        (
            "Timestamps missing",
            f"{cov.missing:,} ({_fmt_pct(cov.share_missing)})"
            if cov.missing
            else "None",
        ),
        ("Gaps", _count(len(cov.gaps))),
        (
            "Longest gap",
            gap_phrase(longest, step, skip) if longest is not None else "None",
        ),
    ]


def observed_pairs(
    facts: ObservedFacts, training_rows: int | None = None
) -> list[tuple[str, str]]:
    """What is odd about the frame passed in; its row count only when it is not the training row count already listed."""
    outside = (
        f"{facts.before:,} before the first training date, {facts.after:,} after the last"
        if facts.before or facts.after
        else "None"
    )
    rows = [] if facts.rows == training_rows else [("Rows", f"{facts.rows:,}")]
    return [
        *rows,
        ("Repeated timestamps", _count(facts.duplicates)),
        ("Missing values", _count(facts.missing_values)),
        ("Zero values", _count(facts.zeros)),
        ("Negative values", _count(facts.negatives)),
        ("Rows outside the training range", outside),
    ]


# ----------------------------------------------------------------------------
# Pure: the gap chart
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Buckets:
    """Grid points grouped into equal runs, each with how many have an observation."""

    size: int
    observed: np.ndarray
    possible: np.ndarray


def bucket_size(expected: int, max_bars: int = MAX_BARS) -> int:
    """The smallest round number of grid points per bar that keeps the bars within ``max_bars``."""
    need = -(-expected // max_bars)
    return next((s for s in _BUCKET_SIZES if s >= need), need)


def buckets(cov: Coverage, max_bars: int = MAX_BARS) -> Buckets:
    size = bucket_size(cov.expected, max_bars)
    n = -(-cov.expected // size)
    observed = np.bincount(cov.grid.slots // size, minlength=n)
    possible = np.full(n, size, dtype=np.int64)
    possible[-1] = cov.expected - size * (n - 1)
    return Buckets(size=size, observed=observed, possible=possible)


def _scramble(slot: int) -> int:
    """A fixed pseudo-random number per gap, so that equal gaps are not all picked from the start."""
    return (slot * 2654435761) % 4294967296


def marked_gaps(gaps: Sequence[Gap], limit: int = MAX_MARKED_GAPS) -> list[Gap]:
    """The ``limit`` longest gaps, in time order. Gaps of equal length are picked evenly through time."""
    longest = sorted(gaps, key=lambda g: (-g.missing, _scramble(g.slot)))[:limit]
    return sorted(longest, key=lambda g: g.slot)


def gap_figure(ctx: DashboardContext, cov: Coverage) -> Any:
    """Bars of the share of expected points present, with each missing stretch shaded.

    ``cov`` must be regular: for uneven spacing there are no holes to draw.
    """
    go = ctx.go
    step, skip = ctx.step, ctx.skip
    grid = cov.grid
    bars = buckets(cov)
    starts = np.arange(len(bars.observed)) * bars.size
    first = grid.labels(starts)
    last = grid.labels(starts + bars.possible - 1)
    # A bar starts at the left edge of its first grid point and runs in real time to
    # nearly the edge of its last. On an axis that closes up skipped days the stretch
    # between those edges shrinks with it, so bars stay as wide as the points they hold.
    lo = grid.times(starts - 0.5)
    hi = grid.times(starts + bars.possible - 0.5)
    widths = BAR_FILL * ((hi - lo) / pd.Timedelta(milliseconds=1))
    labels = [
        [
            f"{_fmt_day(a, step)} to {_fmt_day(b, step)}"
            if a != b
            else _fmt_day(a, step),
            int(seen),
            int(total),
        ]
        for a, b, seen, total in zip(first, last, bars.observed, bars.possible)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Share of points present",
            x=[t.isoformat() for t in lo],
            y=bars.observed / bars.possible,
            offset=0,
            width=np.asarray(widths, dtype=float),
            customdata=labels,
            marker={"color": OBSERVED, "line": {"width": 0}},
            hovertemplate=(
                "%{customdata[0]}<br><b>%{y:.0%}</b> present "
                "(%{customdata[1]} of %{customdata[2]} points)<extra></extra>"
            ),
        )
    )
    shown = marked_gaps(cov.gaps)
    shaded = marked_gaps(cov.gaps, MAX_SHADED_GAPS)
    slots = np.array([g.slot for g in shown], dtype=float)
    lengths = np.array([g.missing for g in shown], dtype=float)
    fig.add_trace(
        go.Scatter(
            name="Missing stretch",
            mode="markers",
            x=[t.isoformat() for t in grid.times(slots + (lengths + 1) / 2)],
            y=[1.0] * len(shown),
            customdata=[
                [
                    points_phrase(g.missing, step, skip),
                    _fmt_day(g.before, step),
                    _fmt_day(g.after, step),
                ]
                for g in shown
            ],
            marker={"color": ACCENT, "symbol": "triangle-down", "size": 9},
            hovertemplate=(
                "<b>%{customdata[0]} missing</b><br>"
                "between %{customdata[1]} and %{customdata[2]}<extra></extra>"
            ),
        )
    )
    shade_slots = np.array([g.slot for g in shaded], dtype=float)
    shade_lengths = np.array([g.missing for g in shaded], dtype=float)
    shapes = [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": a.isoformat(),
            "x1": b.isoformat(),
            "y0": 0,
            "y1": 1,
            "fillcolor": BAND,
            "opacity": 1,
            "line": {"width": 0},
            "layer": "below",
        }
        for a, b in zip(
            grid.times(shade_slots + 0.5),
            grid.times(shade_slots + shade_lengths + 0.5),
        )
    ]
    _, y_style = _style.axes()
    fig.update_layout(
        **_style.layout(
            240,
            shapes=shapes,
            showlegend=True,
            xaxis=_style.time_axis(ctx),
            yaxis={
                **y_style,
                "tickformat": ".0%",
                "tickvals": [0, 0.5, 1],
                "range": [0, 1.14],
            },
        )
    )
    return fig


def gap_chart_note(cov: Coverage, step: pd.Timedelta, skip: SkipAxis | None) -> str:
    size = bucket_size(cov.expected)
    each = (
        f"Each bar is one {unit_word(step, skip)}"
        if size == 1
        else f"Each bar covers {points_phrase(size, step, skip)}"
    )
    text = (
        f"{each} and shows the share of expected points that are present. "
        "Shaded stripes are the missing runs."
    )
    n = len(cov.gaps)
    if n > MAX_MARKED_GAPS:
        text += f" Triangles mark {MAX_MARKED_GAPS} of the longest of the {n:,} gaps; hover one for its dates."
    else:
        text += " Hover a triangle for the dates of a gap."
    if n > MAX_SHADED_GAPS:
        text += f" Only the {MAX_SHADED_GAPS} longest gaps are shaded."
    return text


# ----------------------------------------------------------------------------
# Pure: the forecast table and its CSV
# ----------------------------------------------------------------------------

_VALUE_COLUMNS = (
    ("yhat", "Forecast"),
    ("yhat_lower", "Lower"),
    ("yhat_upper", "Upper"),
    ("trend", "Trend"),
)


def shown_rows(fcst: pd.DataFrame, history_end: pd.Timestamp) -> pd.DataFrame:
    """Rows after the last training date, or every row when there are none."""
    future = fcst[fcst["ds"] > history_end]
    return future if len(future) else fcst


def csv_rows(
    fcst: pd.DataFrame, history_end: pd.Timestamp, limit: int = MAX_CSV_ROWS
) -> pd.DataFrame:
    """Every row of ``fcst``; past ``limit`` rows, the future and then the latest history.

    The dates after the last training date are what a forecast is for, so they
    are kept first. ``fcst`` is sorted by date, and so is the result.
    """
    if len(fcst) <= limit:
        return fcst
    future = fcst[fcst["ds"] > history_end]
    if len(future) >= limit:
        return future.head(limit)
    past = fcst[fcst["ds"] <= history_end].tail(limit - len(future))
    return pd.concat([past, future])


def decimals_for(values: Sequence[float] | np.ndarray) -> int:
    """Decimal places that keep about three figures of the spread of ``values``."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    spread = float(finite.max() - finite.min()) if finite.size else 0.0
    if spread <= 0:
        return 2
    return int(min(6, max(0, 2 - np.floor(np.log10(spread)))))


def format_value(value: float, decimals: int) -> str:
    return f"{value:,.{decimals}f}" if np.isfinite(value) else ""


def observed_frame(ctx: DashboardContext) -> pd.DataFrame:
    """Known y by timestamp: from the observed data when it was given, else the training rows."""
    source = (
        ctx.df if ctx.df is not None and {"ds", "y"} <= set(ctx.df) else ctx.history
    )
    frame = pd.DataFrame(
        {
            "ds": naive_dates(source["ds"]),
            "y": pd.to_numeric(source["y"], errors="coerce"),
        }
    ).dropna()
    return frame.drop_duplicates("ds", keep="last")


def with_observed(rows: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """``rows`` with a ``y`` column where a value is known (the column is left out if none is)."""
    merged = rows.merge(observed[["ds", "y"]], on="ds", how="left")
    return merged if merged["y"].notna().any() else merged.drop(columns="y")


def _table_day(ts: pd.Timestamp, step: pd.Timedelta) -> str:
    """A date for the forecast table, which keeps its leading zero so that the column lines up."""
    day = f"{ts:%d %b %Y}"
    return day if step >= pd.Timedelta(hours=20) else f"{day} {ts:%H:%M}"


def table_content(
    rows: pd.DataFrame, step: pd.Timedelta
) -> tuple[list[str], list[list[str]], list[int]]:
    """Headers, formatted cells and the numeric column positions for the forecast table."""
    columns = [(c, h) for c, h in _VALUE_COLUMNS if c in rows]
    if "y" in rows:
        columns.append(("y", "Observed"))
    values = [rows[c].to_numpy(dtype=float) for c, _ in columns]
    decimals = decimals_for(np.concatenate(values))
    body = [
        [_table_day(ts, step), *(format_value(float(v[i]), decimals) for v in values)]
        for i, ts in enumerate(rows["ds"])
    ]
    return ["Date"] + [h for _, h in columns], body, list(range(1, len(columns) + 1))


def iso_stamps(ds: pd.Series) -> pd.Series:
    """ISO 8601 text for timestamps: a date alone for whole days, else with the time."""
    midnight = ds == ds.dt.normalize()
    if bool(midnight.all()):
        return ds.dt.strftime("%Y-%m-%d")
    fraction = bool((ds.dt.microsecond != 0).any())
    return ds.dt.strftime("%Y-%m-%dT%H:%M:%S" + (".%f" if fraction else ""))


def forecast_csv(fcst: pd.DataFrame, observed: pd.DataFrame | None = None) -> str:
    """Every row of ``fcst`` as CSV at full float precision, with observed ``y`` where known."""
    columns = [c for c, _ in _VALUE_COLUMNS if c in fcst]
    out = fcst[["ds", *columns]]
    if observed is not None:
        out = with_observed(out, observed)
    out = out.assign(ds=iso_stamps(out["ds"]))
    return out.to_csv(index=False).replace("\r\n", "\n")


def escape_for_script(text: str) -> str:
    """Make ``text`` safe inside a ``<script>`` block: no ``<`` survives, and it is reversible."""
    return text.replace("\\", "\\\\").replace("<", "\\u003c")


def unescape_from_script(text: str) -> str:
    """Inverse of :func:`escape_for_script`, the same job the page script does."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and text.startswith("\\\\", i):
            out.append("\\")
            i += 2
        elif text.startswith("\\u003c", i):
            out.append("<")
            i += 6
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------

_CSV_SCRIPT = r"""
(function () {
  var script = document.currentScript;
  var root = script && script.closest('.csv-tools');
  if (!root) { return; }
  var data = root.querySelector('script.csv-data');
  var status = root.querySelector('.csv-status');
  var name = root.getAttribute('data-filename') || 'forecast.csv';

  function csv() {
    return data.textContent.replace(/\\(u003c|\\)/g, function (_, c) {
      return c === '\\' ? '\\' : '<';
    });
  }
  function rows(text) {
    var n = Math.max(text.trim().split('\n').length - 1, 0);
    return n + (n === 1 ? ' row' : ' rows');
  }
  function say(text) { status.textContent = text; }

  function download() {
    var text = csv();
    try {
      var url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }));
      var link = document.createElement('a');
      link.href = url;
      link.download = name;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      say('Downloaded ' + name + ' with ' + rows(text) + '.');
    } catch (err) {
      say('The download did not start. Try Copy CSV instead.');
    }
  }

  function copyWithTextarea(text) {
    var area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.top = '0';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
    document.body.removeChild(area);
    return ok;
  }

  function copy() {
    var text = csv();
    var done = function () { say('Copied ' + rows(text) + ' to the clipboard.'); };
    var fail = function () { say('The browser blocked copying. Try Download CSV instead.'); };
    var fallback = function () { if (copyWithTextarea(text)) { done(); } else { fail(); } };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else {
      fallback();
    }
  }

  root.addEventListener('click', function (event) {
    var button = event.target.closest ? event.target.closest('button[data-action]') : null;
    if (!button || !root.contains(button)) { return; }
    if (button.getAttribute('data-action') === 'download') { download(); } else { copy(); }
  });
})();
"""


def csv_tools_html(csv_text: str) -> str:
    """Download and copy buttons with the CSV embedded for them to use."""
    return (
        f'<div class="csv-tools" data-filename="{html.escape(CSV_FILENAME)}">'
        '<button type="button" class="btn" data-action="download">Download CSV</button> '
        '<button type="button" class="btn" data-action="copy">Copy CSV</button> '
        '<span class="note csv-status" role="status" aria-live="polite"></span>'
        f'<script type="text/csv" class="csv-data">{escape_for_script(csv_text)}</script>'
        f"<script>{_CSV_SCRIPT}</script>"
        "</div>"
    )


def table_note(
    shown: int, total: int, has_future: bool, history_end: str, csv: int, all_rows: int
) -> str:
    """What the table lists, and how much of the forecast the CSV holds."""
    where = (
        f"the dates after the last training date ({history_end})"
        if has_future
        else "every date, because this forecast has none after the training data"
    )
    text = f"The table lists {where}."
    if shown < total:
        text += f" It shows the first {shown:,} of {total:,} rows."
    if csv < all_rows:
        text += (
            f" The CSV is capped at {csv:,} of the {all_rows:,} rows: the dates after the "
            "last training date first, then the latest ones before it."
        )
    elif all_rows > total:
        text += (
            f" The CSV holds all {all_rows:,} rows, including the dates before the "
            "last training date."
        )
    else:
        text += f" The CSV holds all {all_rows:,} rows."
    return text


def _gaps_table(cov: Coverage, step: pd.Timedelta, skip: SkipAxis | None) -> str:
    longest = sorted(cov.gaps, key=lambda g: (-g.missing, g.slot))[:MAX_GAP_ROWS]
    return _style.table(
        ["Last point before", "Next point after", "Missing"],
        [
            [
                _fmt_day(g.before, step),
                _fmt_day(g.after, step),
                points_phrase(g.missing, step, skip),
            ]
            for g in longest
        ],
        numeric=[2],
    )


def _missing_part(
    ctx: DashboardContext, cov: Coverage, pattern: Pattern | None
) -> list[str]:
    """The note about weekends or hours, then the gap chart or the reason there is none."""
    parts = [] if pattern is None else [_style.note(pattern.hint)]
    parts.append(_style.h3("Missing data"))
    if not cov.regular:
        parts.append(
            _style.note(
                "Not drawn: the points are not evenly spaced, so there is no regular "
                "series to find holes in."
            )
        )
    elif cov.gaps:
        parts.append(ctx.embed(gap_figure(ctx, cov)))
        parts.append(_style.note(gap_chart_note(cov, ctx.step, ctx.skip)))
        if len(cov.gaps) > 1:
            parts.append(_style.h3("Longest gaps"))
            parts.append(_gaps_table(cov, ctx.step, ctx.skip))
    else:
        parts.append(
            _style.note(
                "Nothing is missing: every timestamp expected at this spacing is present."
            )
        )
    return parts


def _forecast_part(ctx: DashboardContext) -> list[str]:
    """The forecast table with its CSV buttons."""
    rows = shown_rows(ctx.fcst, ctx.history_end)
    observed = observed_frame(ctx)
    headers, cells, numeric = table_content(
        with_observed(rows.head(MAX_TABLE_ROWS), observed), ctx.step
    )
    export = csv_rows(ctx.fcst, ctx.history_end)
    note = table_note(
        len(cells),
        len(rows),
        len(ctx.future) > 0,
        _fmt_day(ctx.history_end, ctx.step),
        len(export),
        len(ctx.fcst),
    )
    return [
        _style.h3("Forecast table"),
        _style.note(note),
        csv_tools_html(forecast_csv(export, observed)),
        _style.table(
            headers, cells, numeric=numeric, scroll=True, table_id="forecast-table"
        ),
    ]


def build(ctx: DashboardContext) -> Section | None:
    """The section, or None when the model has no training rows to describe."""
    if ctx.history.empty:
        return None
    cov = coverage(ctx.history["ds"], ctx.skip)
    span = _duration_phrase(cov.last - cov.first)
    facts = observed_facts(ctx.df, cov.first, cov.last) if ctx.df is not None else None
    pattern = missing_pattern(cov, ctx.step, ctx.skip)

    left = _style.h3("Coverage") + _style.kv(
        coverage_pairs(cov, ctx.step, ctx.skip, span)
    )
    parts = [_style.lede(lede_text(cov, ctx.step, ctx.skip, span))]
    if facts is not None:
        right = _style.h3("Your data") + _style.kv(observed_pairs(facts, cov.rows))
        parts.append(_style.grid2(left, right))
    else:
        parts.append(left)
        parts.append(
            _style.note(
                "Pass the observed data as df to dashboard_html() to also check for "
                "repeated timestamps, missing values, zeros and rows outside the training range."
            )
        )
    parts += _missing_part(ctx, cov, pattern)
    parts += _forecast_part(ctx)
    return Section(
        key=KEY,
        title=TITLE,
        body="".join(parts),
        summary=summary_text(cov, ctx.step, ctx.skip, facts, pattern),
    )
