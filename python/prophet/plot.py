# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Plotting for Prophet forecasts (matplotlib and plotly).

The default look is an editorial one: warm paper, a serif headline over a sans
subtitle, a single claret accent for the forecast, warm grey for what was
observed, horizontal gridlines only and direct labels instead of legends.
History and forecast sit in lightly tinted zones, and headlines are written in
plain English from the data (for example "Weekly pattern: Saturdays are
highest"). Every automatic title and annotation can be switched off with the
keyword-only ``title`` and ``annotate`` parameters.

Nothing here changes global matplotlib state: artists are styled directly.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
import pandas as pd

# TODO: separate performance_metrics into a different module. there is an implicit circular import between forecaster.py and diagnostics.py
from prophet.diagnostics import performance_metrics

if TYPE_CHECKING:
    from typing import Iterable, Literal, Sequence, TypeVar, type_check_only
    from typing_extensions import NotRequired, TypedDict

    from prophet.forecaster import Prophet

    import matplotlib.pyplot as plt
    import plotly.graph_objs as go

    _AxT = TypeVar('_AxT', bound=plt.Axes)
    _SkipSpec = str | int | Iterable[str | int] | None

    @type_check_only
    class _PlotlyProps(TypedDict):
        traces: list[go.Scatter]
        xaxis: go.layout.XAxis
        yaxis: go.layout.YAxis
        title: NotRequired[str]
        subtitle: NotRequired[str]


logger: logging.Logger = logging.getLogger('prophet.plot')


try:
    import matplotlib as mpl
    from matplotlib import font_manager, patheffects
    from matplotlib import pyplot as plt
    from matplotlib.dates import (
        AutoDateLocator,
        ConciseDateFormatter,
        MonthLocator,
        date2num,
        num2date,
    )
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    from matplotlib.ticker import Formatter, FuncFormatter, Locator
    from matplotlib.transforms import ScaledTranslation

    from pandas.plotting import deregister_matplotlib_converters
    deregister_matplotlib_converters()
except ImportError:
    logger.error('Importing matplotlib failed. Plotting will not work.')

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ImportError:
    logger.error('Importing plotly failed. Interactive plots will not work.')


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------

PAPER = '#FFF7EC'  # warm off-white ground
INK = '#1F1A17'  # headlines, baseline rule
INK_SOFT = '#5C534C'  # subtitles, tick labels (about 7:1 on paper)
GRID = '#E3D8CA'  # horizontal gridlines
OBSERVED = '#8C8178'  # warm grey for what was observed
OBSERVED_TEXT = '#6E645B'  # darker grey for text on the tinted zones
ACCENT = '#990F3D'  # claret: the single accent, for the forecast
HISTORY_TINT = '#F6EDE0'
FORECAST_TINT = '#FBE9E4'
BAND_ALPHA = 0.25

# The last family in each stack ships with matplotlib, so rendering never fails.
_SANS_STACK = ('Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans')
_SERIF_STACK = ('Georgia', 'Charter', 'Times New Roman', 'DejaVu Serif')
_PLOTLY_SANS = '"Helvetica Neue", Helvetica, Arial, "DejaVu Sans", sans-serif'
_PLOTLY_SERIF = 'Georgia, Charter, "Times New Roman", "DejaVu Serif", serif'
# d3 format for multiplicative axes: two decimals at most, trailing zeros trimmed (6%, 6.5%, not 6.000000%).
_PERCENT_TICKS = '.2~%'
_PLOTLY_BAND = 'rgba(153, 15, 61, 0.25)'  # ACCENT at BAND_ALPHA

_TITLE_PT = 17.0
_PANEL_TITLE_PT = 13.0
_SUBTITLE_PT = 10.0
_TICK_PT = 9.5
_LABEL_PT = 10.0
_NOTE_PT = 9.0
_SOURCE_PT = 8.5
_MARGIN_RIGHT_IN = 0.45
_TEXT_LEFT_IN = 0.4


@lru_cache(maxsize=None)
def _installed_families(stack: tuple[str, ...]) -> tuple[str, ...]:
    """Drop families that are not installed so matplotlib does not warn about each."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    return tuple(f for f in stack if f in available or f == stack[-1])


def _sans() -> list[str]:
    return list(_installed_families(_SANS_STACK))


def _serif() -> list[str]:
    return list(_installed_families(_SERIF_STACK))


# ----------------------------------------------------------------------------
# Skipped days and hours
# ----------------------------------------------------------------------------
#
# When a series never has data on some weekdays or hours (weekends, nights),
# a continuous time axis shows a hole there. ``skip_axis`` builds a mapping
# from timestamps to a compressed position in which the skipped periods take
# no room, so Friday is joined to Monday. The mapping is analytic: it only
# needs the weekly pattern of kept hours, so it has no range limits and no
# lookup tables. Positions are measured in "kept days" (24 kept hours each).

class SkipSpecError(ValueError):
    """``skip_days`` or ``skip_hours`` cannot be understood."""


_WEEKDAY_NAMES = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
_WEEKDAY_ALIASES: dict[str, int] = {
    **{name: i for i, name in enumerate(_WEEKDAY_NAMES)},
    **{name[:3]: i for i, name in enumerate(_WEEKDAY_NAMES)},
    'tues': 1, 'weds': 2, 'thur': 3, 'thurs': 3,
}
_WEEKEND = frozenset({5, 6})
# A Monday, so the hour of the week is simply the hours elapsed modulo 168.
_MONDAY = pd.Timestamp('2000-01-03')
_HOURS_PER_WEEK = 168


def _weekday_numbers(item: object) -> frozenset[int]:
    if isinstance(item, (bool, np.bool_)):
        raise SkipSpecError(f'skip_days entries must be weekday names or numbers, got {item!r}.')
    if isinstance(item, (int, np.integer)):
        if 0 <= int(item) <= 6:
            return frozenset({int(item)})
        raise SkipSpecError(f'Weekday number {int(item)} is out of range; use 0 (Monday) to 6 (Sunday).')
    if isinstance(item, str):
        key = item.strip().lower()
        if key in ('weekend', 'weekends'):
            return _WEEKEND
        number = _WEEKDAY_ALIASES.get(key, _WEEKDAY_ALIASES.get(key.removesuffix('s')))
        if number is not None:
            return frozenset({number})
        raise SkipSpecError(
            f'Unknown weekday {item!r}. Use names such as "Sat" or "Saturday", numbers from 0 (Monday) '
            'to 6 (Sunday), or "weekends".')
    raise SkipSpecError(f'skip_days entries must be weekday names or numbers, got {type(item).__name__}.')


def _entries(spec: object) -> list[object]:
    if isinstance(spec, (str, int, np.integer)):
        return [spec]
    try:
        return list(cast('Iterable[object]', spec))
    except TypeError:
        raise SkipSpecError(f'Expected a name, a number or a list of them, got {type(spec).__name__}.') from None


def parse_skip_days(spec: object) -> frozenset[int]:
    """Weekday numbers (Monday is 0, Sunday is 6) from a flexible description.

    Accepts one entry or a collection of: full or short names in any case
    ("Saturday", "sat", "Sat"), numbers 0 to 6, and the shortcut "weekends".

    Raises
    ------
    SkipSpecError: For an unknown name, an out-of-range number, or when every
        day of the week would be skipped.

    Examples
    --------
    >>> sorted(parse_skip_days("weekends"))
    [5, 6]
    >>> sorted(parse_skip_days(["Mon", 2, "friday"]))
    [0, 2, 4]
    """
    if spec is None:
        return frozenset()
    days = frozenset().union(*(_weekday_numbers(item) for item in _entries(spec)))
    if len(days) == 7:
        raise SkipSpecError('skip_days covers every day of the week, which leaves nothing to draw or forecast.')
    return days


def parse_skip_hours(spec: object) -> frozenset[int]:
    """Hours of the day (0 to 23) from one number or a collection such as ``range(0, 8)``.

    Raises
    ------
    SkipSpecError: For a non-integer or out-of-range hour, or when every hour
        of the day would be skipped.
    """
    if spec is None:
        return frozenset()
    hours: set[int] = set()
    for item in _entries(spec):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise SkipSpecError(f'skip_hours entries must be whole hours from 0 to 23, got {item!r}.')
        if not 0 <= int(item) <= 23:
            raise SkipSpecError(f'Hour {int(item)} is out of range; use 0 to 23.')
        hours.add(int(item))
    if len(hours) == 24:
        raise SkipSpecError('skip_hours covers every hour of the day, which leaves nothing to draw or forecast.')
    return frozenset(hours)


@lru_cache(maxsize=None)
def _week_profile(skip_days: frozenset[int], skip_hours: frozenset[int]) -> tuple[np.ndarray, np.ndarray, float]:
    """Kept hours accumulated before each hour of the week (length 169), whether each hour is kept, and the total."""
    keep = np.array([(h // 24) not in skip_days and (h % 24) not in skip_hours for h in range(_HOURS_PER_WEEK)])
    cumulative = np.concatenate([[0.0], np.cumsum(keep)])
    cumulative.flags.writeable = False
    keep.flags.writeable = False
    return cumulative, keep, float(cumulative[-1])


def _kept_hours(hours: np.ndarray, skip_days: frozenset[int], skip_hours: frozenset[int]) -> np.ndarray:
    """Kept hours elapsed since the origin Monday, for hours elapsed in real time. Non-finite values stay NaN."""
    cumulative, keep, per_week = _week_profile(skip_days, skip_hours)
    hours = np.asarray(hours, dtype=float)
    finite = np.isfinite(hours)
    safe = np.where(finite, hours, 0.0)
    week = np.floor(safe / _HOURS_PER_WEEK)
    rest = safe - week * _HOURS_PER_WEEK
    slot = np.minimum(np.floor(rest).astype(int), _HOURS_PER_WEEK - 1)
    kept = week * per_week + cumulative[slot] + (rest - slot) * keep[slot]
    return np.where(finite, kept, np.nan)


def _real_hours(kept: np.ndarray, skip_days: frozenset[int], skip_hours: frozenset[int]) -> np.ndarray:
    """Inverse of :func:`_kept_hours`. A position inside a skipped period resolves to the next kept time."""
    cumulative, _, per_week = _week_profile(skip_days, skip_hours)
    kept = np.asarray(kept, dtype=float)
    finite = np.isfinite(kept)
    safe = np.where(finite, kept, 0.0)
    week = np.floor(safe / per_week)
    rest = safe - week * per_week
    slot = np.searchsorted(cumulative[:-1], rest, side='right') - 1
    hours = week * _HOURS_PER_WEEK + slot + (rest - cumulative[slot])
    return np.where(finite, hours, np.nan)


def _naive_index(dates: object) -> pd.DatetimeIndex:
    """Any date-like input as a timezone-naive DatetimeIndex (wall-clock time is kept)."""
    if isinstance(dates, (pd.Series, pd.DatetimeIndex)):
        index = pd.DatetimeIndex(dates)
    else:
        index = pd.DatetimeIndex(np.atleast_1d(np.asarray(dates)))
    return index.tz_localize(None) if index.tz is not None else index


@lru_cache(maxsize=None)
def _ticker_classes() -> tuple[type, type, type]:
    """Matplotlib tick helpers, defined lazily so importing this module never needs matplotlib."""

    class SnappedDateLocator(AutoDateLocator):
        """Calendar ticks on a warped date axis, one per drawn position (a tick in a skipped period is dropped
        when a kept tick sits at the same place)."""

        def __init__(self, axis: SkipAxis) -> None:
            super().__init__(minticks=3, maxticks=8)
            self._skip = axis

        def __call__(self) -> np.ndarray:
            return self._skip._distinct_date_numbers(np.asarray(super().__call__(), dtype=float))

    class PositionLocator(Locator):
        """Calendar ticks for an axis that holds :meth:`SkipAxis.to_x` positions."""

        def __init__(self, axis: SkipAxis) -> None:
            super().__init__()
            self._skip = axis
            self._inner = AutoDateLocator(minticks=3, maxticks=8)

        def __call__(self) -> Sequence[float]:
            low, high = self.axis.get_view_interval()  # pyrefly:ignore[missing-attribute]
            return self.tick_values(low, high)

        def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:
            first, last = self._skip.from_x([vmin, vmax]).to_pydatetime()
            nums = np.asarray(self._inner.get_locator(first, last).tick_values(first, last), dtype=float)
            stamps = pd.DatetimeIndex(num2date(nums)).tz_localize(None)
            # A label is derived from the position, so a tick inside a skipped period could only be
            # labelled with the next kept date. Dropping it keeps every label true.
            return np.unique(self._skip.to_x(stamps[self._skip.keeps(stamps)])).tolist()

    class PositionFormatter(Formatter):
        """Date labels for an axis that holds :meth:`SkipAxis.to_x` positions."""

        def __init__(self, axis: SkipAxis) -> None:
            super().__init__()
            self._skip = axis
            self._concise = ConciseDateFormatter(AutoDateLocator(minticks=3, maxticks=8))

        def format_ticks(self, values: Sequence[float]) -> list[str]:
            stamps = self._skip.from_x(np.asarray(values, dtype=float))
            return self._concise.format_ticks(date2num(stamps.to_pydatetime()))

        def __call__(self, x: float, pos: int | None = None) -> str:
            return self.format_ticks([x])[0]

        def get_offset(self) -> str:
            return self._concise.get_offset()

    return SnappedDateLocator, PositionLocator, PositionFormatter


@dataclass(frozen=True)
class SkipAxis:
    """A time axis that leaves out whole weekdays and hours of the day.

    Build one with :func:`skip_axis`. The mapping from time to position is
    piecewise linear: one kept hour is one 24th of a position unit, and a
    skipped hour has no width, so a Friday and the following Monday are one
    day apart. Positions are therefore measured in kept days.

    Two ways to draw with it:

    * Warped axis, data stay timestamps: call :meth:`apply` on a matplotlib
      axes before or after plotting. Everything that takes dates (lines,
      ``axvspan``, ``axvline``, text) keeps working unchanged.
    * Plain numbers: plot ``to_x(dates)`` on an ordinary axis and install
      :meth:`tick_locator` and :meth:`tick_formatter`, which turn positions
      back into calendar dates.

    Attributes
    ----------
    skip_days: Weekday numbers left out, Monday is 0.
    skip_hours: Hours of the day left out.
    first, last: First and last kept timestamp of the data the axis was
        built from (None when built without data).
    """

    skip_days: frozenset[int] = frozenset()
    skip_hours: frozenset[int] = frozenset()
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None

    @property
    def active(self) -> bool:
        """False when nothing is skipped (positions then equal elapsed days)."""
        return bool(self.skip_days or self.skip_hours)

    @property
    def limits(self) -> tuple[float, float]:
        """Positions of the first and last kept timestamp."""
        if self.first is None or self.last is None:
            raise SkipSpecError('This axis was built without data, so it has no limits.')
        return float(self.to_x(self.first)), float(self.to_x(self.last))

    def keeps(self, dates: object) -> np.ndarray:
        """Boolean mask: True where a timestamp is not on a skipped day or hour."""
        index = _naive_index(dates)
        mask = np.ones(len(index), dtype=bool)
        if self.skip_days:
            mask &= ~np.isin(index.dayofweek, sorted(self.skip_days))
        if self.skip_hours:
            mask &= ~np.isin(index.hour, sorted(self.skip_hours))
        return mask

    def to_x(self, dates: object) -> np.ndarray | float:
        """Compressed position of each timestamp, in kept days since an arbitrary Monday.

        A timestamp inside a skipped period gets the position where the next
        kept time starts. Returns a float for a single timestamp, otherwise an
        array.
        """
        hours = ((_naive_index(dates) - _MONDAY) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
        x = _kept_hours(hours, self.skip_days, self.skip_hours) / 24.0
        return float(x[0]) if np.ndim(dates) == 0 else x

    def from_x(self, x: object) -> pd.DatetimeIndex:
        """Timestamps for compressed positions (the inverse of :meth:`to_x`), rounded to the millisecond."""
        hours = _real_hours(np.atleast_1d(np.asarray(x, dtype=float)) * 24.0, self.skip_days, self.skip_hours)
        return pd.DatetimeIndex(_MONDAY + pd.to_timedelta(hours, unit='h')).round('ms')

    def tick_locator(self) -> Locator:
        """Locator for an axis holding :meth:`to_x` positions: calendar ticks, placed by position."""
        return _ticker_classes()[1](self)

    def tick_formatter(self) -> Formatter:
        """Formatter for an axis holding :meth:`to_x` positions: dates in matplotlib's concise style."""
        return _ticker_classes()[2](self)

    def apply(self, ax: plt.Axes) -> plt.Axes:
        """Warp the x axis of ``ax`` so skipped periods take no room; data stay timestamps.

        Sets a function scale, calendar tick locations (a tick that falls in
        a skipped period moves to the next kept time) and concise date labels.
        """
        origin = float(date2num(_MONDAY))

        def forward(datenum: object) -> np.ndarray:
            hours = (np.asarray(datenum, dtype=float) - origin) * 24.0
            return _kept_hours(hours, self.skip_days, self.skip_hours) / 24.0

        def inverse(position: object) -> np.ndarray:
            hours = _real_hours(np.asarray(position, dtype=float) * 24.0, self.skip_days, self.skip_hours)
            return hours / 24.0 + origin

        ax.set_xscale('function', functions=(forward, inverse))
        locator = _ticker_classes()[0](self)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        return ax

    def _distinct_date_numbers(self, ticks: np.ndarray) -> np.ndarray:
        """One tick per drawn position. A tick in a skipped period (the 1st of a month that is a Saturday)
        keeps its true date and is drawn where the skipped period was, unless a kept tick is already there."""
        if ticks.size == 0:
            return ticks
        hours = (ticks - float(date2num(_MONDAY))) * 24.0
        position = np.round(_kept_hours(hours, self.skip_days, self.skip_hours), 6)
        _, keep, _ = _week_profile(self.skip_days, self.skip_hours)
        slot = np.minimum(np.floor(np.mod(hours, _HOURS_PER_WEEK)).astype(int), _HOURS_PER_WEEK - 1)
        order = np.lexsort((~keep[slot], position))  # by position, kept ticks first
        first = np.concatenate([[True], position[order][1:] != position[order][:-1]])
        return np.sort(ticks[order][first])


def skip_axis(ds: object, *, skip_days: _SkipSpec = None, skip_hours: _SkipSpec = None) -> SkipAxis:
    """Build the :class:`SkipAxis` for timestamps ``ds`` with some weekdays or hours left out.

    Parameters
    ----------
    ds: The timestamps a chart covers (a Series, DatetimeIndex or list). Only
        used to record the first and last kept timestamp and to check that
        something remains after skipping.
    skip_days: Weekdays to leave out: names ("Sat", "Saturday", "sat"),
        numbers (Monday is 0, so Saturday is 5), the shortcut "weekends", or a
        collection of these.
    skip_hours: Hours of the day to leave out, for example ``range(0, 8)``.

    Raises
    ------
    SkipSpecError: When a specification is invalid or every timestamp falls
        on a skipped day or hour.

    Examples
    --------
    >>> import pandas as pd
    >>> axis = skip_axis(pd.date_range("2024-01-01", periods=14), skip_days="weekends")
    >>> x = axis.to_x(pd.to_datetime(["2024-01-05", "2024-01-08"]))
    >>> float(x[1] - x[0])  # Friday to Monday is one step
    1.0
    """
    shell = SkipAxis(parse_skip_days(skip_days), parse_skip_hours(skip_hours))
    index = _naive_index(ds)
    index = index.dropna()
    kept = index[shell.keeps(index)]
    if kept.empty:
        raise SkipSpecError('Every timestamp falls on a skipped day or hour, so nothing is left to draw.')
    return replace(shell, first=kept.min(), last=kept.max())


def _chart_axis(ds: object, skip_days: object, skip_hours: object) -> SkipAxis | None:
    """The skip axis for a chart, or None when nothing is skipped."""
    days, hours = parse_skip_days(skip_days), parse_skip_hours(skip_hours)
    if not days and not hours:
        return None
    return skip_axis(ds, skip_days=days, skip_hours=hours)


def _hour_runs(hours: Sequence[int]) -> list[tuple[int, int]]:
    """Consecutive runs of hours as (first, last), wrapping past midnight."""
    present = set(hours)
    starts = [h for h in sorted(present) if (h - 1) % 24 not in present]
    runs = []
    for start in starts:
        end = start
        while (end + 1) % 24 in present:
            end = (end + 1) % 24
        runs.append((start, end))
    return runs


def _skip_note(axis: SkipAxis) -> str:
    """A sentence for the credit line saying what the compressed time axis leaves out."""
    parts = []
    if axis.skip_days:
        days = [f'{_WEEKDAY_NAMES[d].capitalize()}s' for d in sorted(axis.skip_days)]
        parts.append(days[0] if len(days) == 1 else ', '.join(days[:-1]) + ' and ' + days[-1])
    if axis.skip_hours:
        parts.append(', '.join(f'{a:02d}:00 to {(b + 1) % 24:02d}:00' for a, b in _hour_runs(sorted(axis.skip_hours))))
    return f'The time axis leaves out {" and ".join(parts)}.'


# ----------------------------------------------------------------------------
# Pure helpers: numbers, durations and plain-English wording
# ----------------------------------------------------------------------------

def _fmt_number(v: float) -> str:
    """Compact, unit-free number: 8.52, 1,235, 12.3K, 1.2M."""
    a = abs(v)
    if not np.isfinite(v):
        return 'n/a'
    if a >= 1e9:
        return f'{v / 1e9:.3g}B'
    if a >= 1e6:
        return f'{v / 1e6:.3g}M'
    if a >= 1e4:
        return f'{v / 1e3:.3g}K'
    if a >= 100:
        return f'{v:,.0f}'
    return f'{v + 0.0:.3g}'


def _fmt_tick(v: float, pos: int | None = None) -> str:
    a = abs(v)
    if a >= 1e9:
        return f'{v / 1e9:.4g}B'
    if a >= 1e6:
        return f'{v / 1e6:.4g}M'
    if a >= 1e4:
        return f'{v / 1e3:.4g}K'
    if a >= 100:
        return f'{v:,.0f}'
    return f'{v + 0.0:.4g}'


def _fmt_pct(fraction: float) -> str:
    """Percentage of a fraction with sensible precision: 0.45%, 6.5%, 12%."""
    a = abs(fraction) * 100
    return f'{a:.0f}%' if a >= 10 else f'{a:.2g}%'


def _signed_effect(v: float, multiplicative: bool) -> str:
    sign = '-' if v < 0 else '+'
    return f'{sign}{_fmt_pct(v)}' if multiplicative else f'{sign}{_fmt_number(abs(v))}'


def _above_below(v: float, multiplicative: bool) -> str:
    amount = _fmt_pct(v) if multiplicative else _fmt_number(abs(v))
    if amount in ('0%', '0'):
        return 'at'
    return f'{amount} {"below" if v < 0 else "above"}'


def _plural(n: int, unit: str) -> str:
    return f'{n} {unit}' if n == 1 else f'{n} {unit}s'


def _duration_parts(td: pd.Timedelta) -> tuple[int, str]:
    """Round a span to a whole number of the most natural unit."""
    seconds = max(td.total_seconds(), 1.0)
    days = seconds / 86400.0
    if seconds < 2 * 3600:
        return max(1, round(seconds / 60)), 'minute'
    if days < 2:
        return max(1, round(seconds / 3600)), 'hour'
    if days < 14:
        return max(1, round(days)), 'day'
    if days < 70:
        return max(1, round(days / 7)), 'week'
    if days < 730:
        return max(1, round(days / 30.44)), 'month'
    return max(1, round(days / 365.25)), 'year'


def _duration_phrase(td: pd.Timedelta) -> str:
    n, unit = _duration_parts(td)
    return _plural(n, unit)


def _span_phrase(td: pd.Timedelta, lead: str) -> str:
    """'the next 12 months', 'the last day'."""
    n, unit = _duration_parts(td)
    return f'{lead} {unit}' if n == 1 else f'{lead} {n} {unit}s'


def _fmt_date(ts: pd.Timestamp, span: pd.Timedelta) -> str:
    """Date wording that fits the span of the data: month for years, time for hours."""
    if span >= pd.Timedelta(days=730):
        return f'{ts:%b %Y}'
    if span >= pd.Timedelta(days=60):
        return f'{ts:%d %b %Y}'
    return f'{ts:%d %b %H:%M}'


def _pretty(name: str) -> str:
    text = name.replace('_', ' ')
    return text[:1].upper() + text[1:]


def _y_scale(m: Prophet) -> float:
    """Typical magnitude of y, used to tell a negligible difference from a real one."""
    y = np.abs(cast('pd.DataFrame', m.history)['y'].to_numpy(dtype=float))
    scale = float(np.nanmax(y)) if y.size else 0.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def _compare_phrase(new: float, old: float, scale: float) -> str:
    """'about 12% higher than', 'about the same as', '0.9 lower than'.

    Percentages are only used when the base is clearly positive; with a base
    near zero or negative the absolute difference is the honest comparison.
    """
    diff = new - old
    if old > 0 and new >= 0 and old >= 0.2 * scale:
        rel = diff / old
        if abs(rel) < 0.03:
            return 'about the same as'
        return f'about {_fmt_pct(rel)} {"lower" if rel < 0 else "higher"} than'
    if abs(diff) < 0.03 * scale:
        return 'about the same as'
    return f'{_fmt_number(abs(diff))} {"lower" if diff < 0 else "higher"} than'


def _forecast_headline(m: Prophet, fcst: pd.DataFrame, show_band: bool) -> tuple[str, str]:
    history = cast('pd.DataFrame', m.history)
    hist_start, hist_end = history['ds'].min(), history['ds'].max()
    hist_span = hist_end - hist_start
    band = (
        f'The shaded band is the range the real value should land in {m.interval_width:.0%} of the time.'
        if show_band else ''
    )
    future = fcst[fcst['ds'] > hist_end]
    if future.empty:
        sub = (
            f'{len(history):,} observations from {_fmt_date(hist_start, hist_span)} to '
            f'{_fmt_date(hist_end, hist_span)}, with the fitted values drawn over them.'
        )
        return 'How the model fits the history', f'{sub} {band}'.strip()
    span = future['ds'].max() - hist_end
    title = f'What happens next: {_span_phrase(span, "the next")}'
    nxt = float(future['yhat'].mean())
    window = min(span, hist_span)
    sub = f'The forecast averages {_fmt_number(nxt)} over {_span_phrase(span, "the next")}'
    if window > pd.Timedelta(0) and window >= span * 0.5:
        recent = history.loc[history['ds'] > hist_end - window, 'y']
        last = float(recent.mean())
        sub += (
            f', {_compare_phrase(nxt, last, _y_scale(m))} {_span_phrase(window, "the last")} '
            f'({_fmt_number(last)}).'
        )
    else:
        sub += '.'
    return title, f'{sub} {band}'.strip()


def _trend_headline(m: Prophet, fcst: pd.DataFrame) -> tuple[str, str]:
    trend = fcst.set_index('ds')['trend'].sort_index()
    hist_end = cast('pd.DataFrame', m.history)['ds'].max()
    inside = trend[trend.index <= hist_end]
    if inside.empty:
        inside = trend
    first, now, end = float(inside.iloc[0]), float(inside.iloc[-1]), float(trend.iloc[-1])
    span = trend.index[-1] - trend.index[0]
    scale = _y_scale(m)
    start_label = _fmt_date(inside.index[0], span)
    now_label = _fmt_date(inside.index[-1], span)
    diff = now - first
    if abs(diff) < 0.03 * max(abs(first), 0.05 * scale):
        title = 'Trend: broadly flat'
    elif first > 0 and now >= 0 and first >= 0.2 * scale:
        title = f'Trend: {"down" if diff < 0 else "up"} {_fmt_pct(diff / first)} since {start_label}'
    else:
        title = f'Trend: {"falling" if diff < 0 else "rising"} since {start_label}'
    sub = (
        'With seasonal and holiday effects removed, the underlying level moved from '
        f'{_fmt_number(first)} to {_fmt_number(now)} by {now_label}.'
    )
    if trend.index[-1] > hist_end:
        sub += f' If the trend continues it reaches {_fmt_number(end)} by {_fmt_date(trend.index[-1], span)}.'
    return title, sub


def _holiday_columns(m: Prophet, fcst: pd.DataFrame) -> list[str]:
    """Per-holiday effect columns of the forecast (not totals, seasonalities or regressors)."""
    modes = cast('dict[str, list[str]]', m.component_modes)
    excluded = set(m.seasonalities) | set(m.extra_regressors) | {
        'additive_terms', 'multiplicative_terms', 'holidays',
        'extra_regressors_additive', 'extra_regressors_multiplicative',
    }
    names = [c for mode in ('additive', 'multiplicative') for c in modes[mode]]
    return [c for c in dict.fromkeys(names) if c not in excluded and c in fcst]


def _base_holiday_name(column: str) -> str:
    return column[: -len(' (observed)')] if column.endswith(' (observed)') else column


def _biggest_holiday(m: Prophet, fcst: pd.DataFrame) -> tuple[str, pd.Timestamp, float] | None:
    cols = _holiday_columns(m, fcst)
    if not cols:
        return None
    values = fcst[cols].to_numpy(dtype=float)
    if not np.isfinite(values).any() or np.nanmax(np.abs(values)) == 0:
        return None
    row, col = np.unravel_index(np.nanargmax(np.abs(values)), values.shape)
    return _base_holiday_name(cols[col]), fcst['ds'].iloc[row], float(values[row, col])


def _holiday_headline(m: Prophet, fcst: pd.DataFrame) -> tuple[str, str]:
    biggest = _biggest_holiday(m, fcst)
    if biggest is None:
        return 'Holidays: no effect in this period', ''
    name, _, value = biggest
    mult = 'holidays' in cast('dict[str, list[str]]', m.component_modes)['multiplicative']
    return (
        f'Holidays: {name} has the largest effect',
        f'Each stem is one holiday in one year. On {name} the forecast is '
        f'{_above_below(value, mult)} its usual level.',
    )


def _regressor_headline(m: Prophet, name: str, series: pd.Series) -> tuple[str, str]:
    mode = name.rsplit('_', 1)[-1]
    names = [r for r, props in m.extra_regressors.items() if props['mode'] == mode]
    listed = ', '.join(names) if len(names) <= 3 else f'{len(names)} regressors'
    mult = mode == 'multiplicative'
    lo, hi = float(series.min()), float(series.max())
    return (
        f'Extra regressors: {listed}',
        f'Combined effect on the forecast, from {_signed_effect(lo, mult)} to {_signed_effect(hi, mult)}.',
    )


def _phase_label(ts: pd.Timestamp, period: float) -> str:
    """Where in a seasonal cycle a timestamp falls, in words that suit the period."""
    if period <= 2:
        return f'{ts:%H:%M}'
    if period < 14:
        return f'{ts:%A %H:%M}'
    if 360 <= period <= 370:
        return f'{ts:%B} {ts.day}'
    return f'day {(ts - pd.Timestamp("2017-01-01")).total_seconds() / 86400:.0f} of {period:g}'


def _seasonality_headline(
    m: Prophet,
    name: str,
    ds: pd.Series | pd.DatetimeIndex,
    values: np.ndarray | pd.Series,
    kind: str = 'other',
) -> tuple[str, str]:
    """Headline from a single seasonal cycle. kind is 'weekly', 'yearly' or 'other'."""
    mult = m.seasonalities[name]['mode'] == 'multiplicative'
    vals = np.asarray(values, dtype=float)
    days = pd.DatetimeIndex(ds)
    label = _pretty(name)
    tolerance = 0.005 if mult else 0.005 * _y_scale(m)
    if not np.isfinite(vals).any() or np.nanmax(vals) - np.nanmin(vals) < tolerance:
        return f'{label} pattern: little variation', ''
    if kind == 'weekly':
        names = days.day_name().to_numpy()
        hi, lo = int(np.nanargmax(vals)), int(np.nanargmin(vals))
        return (
            f'{label} pattern: {names[hi]}s are highest',
            f'{names[hi]}s are {_above_below(vals[hi], mult)} the usual level and '
            f'{names[lo]}s {_above_below(vals[lo], mult)} it.',
        )
    if kind == 'yearly':
        monthly = pd.Series(vals, index=days.month).groupby(level=0).mean()
        hi, lo = int(monthly.idxmax()), int(monthly.idxmin())
        peak_name, low_name = (pd.Timestamp(2017, k, 1).strftime('%B') for k in (hi, lo))
        return (
            f'{label} pattern: peaks in {peak_name}',
            f'{peak_name} is {_above_below(float(monthly[hi]), mult)} the usual level and '
            f'{low_name} {_above_below(float(monthly[lo]), mult)} it.',
        )
    period = float(m.seasonalities[name]['period'])
    hi, lo = int(np.nanargmax(vals)), int(np.nanargmin(vals))
    return (
        f'{label} pattern: peaks at {_phase_label(days[hi], period)}',
        f'The high point is {_above_below(vals[hi], mult)} the usual level; the low point '
        f'({_phase_label(days[lo], period)}) is {_above_below(vals[lo], mult)} it.',
    )


def _significant_changepoints(m: Prophet, threshold: float = 0.01) -> pd.DataFrame:
    """Changepoints whose mean rate change is at least threshold, as columns ds and delta."""
    empty = pd.DataFrame({'ds': pd.Series(dtype='datetime64[ns]'), 'delta': pd.Series(dtype=float)})
    changepoints = m.changepoints
    params = m.params
    if changepoints is None or len(changepoints) == 0 or not params or 'delta' not in params:
        return empty
    delta = np.nanmean(np.atleast_2d(params['delta']), axis=0)
    if len(delta) != len(changepoints):
        return empty
    frame = pd.DataFrame({'ds': pd.to_datetime(changepoints.to_numpy()), 'delta': delta})
    return frame[frame['delta'].abs() >= threshold].reset_index(drop=True)


def _top_changepoints(m: Prophet, fcst: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The n largest significant changepoints that fall inside the plotted dates."""
    sig = _significant_changepoints(m)
    if sig.empty or 'trend' not in fcst:
        return sig
    inside = sig[(sig['ds'] >= fcst['ds'].min()) & (sig['ds'] <= fcst['ds'].max())]
    order = inside['delta'].abs().sort_values(ascending=False).index
    return inside.loc[order].head(n).reset_index(drop=True)


def _resolve_text(value: str | bool | None, auto: Callable[[], str]) -> str | None:
    """None means automatic, a string overrides, False disables."""
    if value is False:
        return None
    if isinstance(value, str):
        return value or None
    return auto() or None


def _headline_texts(
    title: str | bool | None,
    subtitle: str | bool | None,
    auto: Callable[[], tuple[str, str]],
) -> tuple[str | None, str | None]:
    """Resolve title and subtitle. title=False also drops the automatic subtitle.

    A title given as a string also drops the automatic subtitle unless the
    caller offers a subtitle parameter (only plot() does), because the
    derived sentence was written to go with the derived title.
    """
    cache: list[tuple[str, str]] = []

    def derived() -> tuple[str, str]:
        if not cache:
            cache.append(auto())
        return cache[0]

    title_text = _resolve_text(title, lambda: derived()[0])
    if title is False and not isinstance(subtitle, str):
        return None, None
    return title_text, _resolve_text(subtitle, lambda: derived()[1])


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------

def _wrap(text: str, width_in: float, size_pt: float, max_lines: int | None = None,
          factor: float = 0.56) -> str:
    """Wrap to a width in inches; factor is the average glyph width as a share of the font size."""
    chars = max(24, int(width_in * 72 / (size_pt * factor)))
    if max_lines is not None:
        text = textwrap.shorten(text, width=chars * max_lines, placeholder='...')
    return '\n'.join(textwrap.wrap(text, width=chars)) or text


def _n_lines(text: str | None) -> int:
    return 0 if not text else text.count('\n') + 1


def _style_axes(ax: plt.Axes) -> None:
    """Editorial furniture: horizontal gridlines, baseline rule only, soft tick labels."""
    ax.set_facecolor(PAPER)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color(INK)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis='y', color=GRID, linewidth=0.8, linestyle='-')
    ax.tick_params(axis='x', length=4, color=INK, labelsize=_TICK_PT, labelcolor=INK_SOFT)
    ax.tick_params(axis='y', length=0, labelsize=_TICK_PT, labelcolor=INK_SOFT, pad=4)
    ax.xaxis.get_offset_text().set(color=INK_SOFT, fontsize=_NOTE_PT, fontfamily=_sans())
    # Tick labels are created lazily, so set their family on the axis rather than relying on rcParams.
    if mpl.__version_info__ >= (3, 8):
        ax.tick_params(axis='both', labelfontfamily=_sans())
    else:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(_sans())


def _style_dates(ax: plt.Axes, skip: SkipAxis | None = None) -> None:
    if skip is not None:
        skip.apply(ax)
        return
    locator = AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))


def _axis_label(ax: plt.Axes, axis: str, text: str | None) -> None:
    if not text:
        return
    setter = ax.set_xlabel if axis == 'x' else ax.set_ylabel
    setter(text, fontsize=_LABEL_PT, color=INK_SOFT, fontfamily=_sans())


def _set_ylim(ax: plt.Axes, arrays: Sequence[object], bottom: float = 0.08, top: float = 0.08) -> None:
    """Pad the y range around the data actually drawn (the default autoscale is too tight for labels)."""
    flat = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return
    lo, hi = float(flat.min()), float(flat.max())
    span = hi - lo or max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - bottom * span, hi + top * span)


def _stroke() -> list[object]:
    """A paper-coloured halo so labels stay legible over lines and bands."""
    return [patheffects.withStroke(linewidth=3, foreground=PAPER)]


def _text_at(
    ax: plt.Axes, x: object, y: float, s: str, dx: float = 0.0, dy: float = 0.0,
    transform: object | None = None, **kw: object,
) -> Text:
    """Text anchored at a point plus an offset in points; a plain Text, so its extent is reliable."""
    base = transform if transform is not None else ax.transData
    shifted = base + ScaledTranslation(dx / 72, dy / 72, ax.figure.dpi_scale_trans)  # pyrefly:ignore[unsupported-operation, missing-attribute]
    return ax.text(x, y, s, transform=shifted, **kw)  # pyrefly:ignore[bad-argument-type]


def _series_px(ax: plt.Axes, x: object, y: object) -> np.ndarray:
    """Screen positions of plotted points, so labels can steer clear of the data."""
    pts = np.column_stack([date2num(x), np.asarray(y, dtype=float)])  # pyrefly:ignore[bad-argument-type]
    return ax.transData.transform(pts[np.isfinite(pts).all(axis=1)])


def _box(text: Text, pad: float = 6.0) -> object:
    return text.get_window_extent().padded(pad)


def _free(box: object, taken: Sequence[object], series: Sequence[np.ndarray] = ()) -> bool:
    """True if the box overlaps no earlier label and covers no plotted point."""
    if any(box.overlaps(other) for other in taken):  # pyrefly:ignore[missing-attribute]
        return False
    for pts in series:
        inside = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)  # pyrefly:ignore[missing-attribute]
                  & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))  # pyrefly:ignore[missing-attribute]
        if inside.any():
            return False
    return True


def _claim(text: Text, taken: list[object], series: Sequence[np.ndarray] = (), pad: float = 6.0) -> bool:
    """Reserve the text's screen area unless it overlaps something already placed."""
    box = _box(text, pad)
    if not _free(box, taken, series):
        return False
    taken.append(box)
    return True


def _fit_inside(ax: plt.Axes, text: Text, left: float | None = None) -> bool:
    """Centre the text on its anchor, or hang it off the anchor to keep it inside the axes.

    ``left`` is an optional screen x the text must not cross, for a label that has to stay right of a boundary.
    """
    axes_box = ax.get_window_extent()
    floor = axes_box.x0 if left is None else max(axes_box.x0, left)
    for ha in ('center', 'right', 'left'):
        text.set_horizontalalignment(ha)
        box = text.get_window_extent()
        if box.x0 >= floor and box.x1 <= axes_box.x1:
            return True
    return False


def _dnum(ts: pd.Timestamp) -> float:
    """A single timestamp as a matplotlib date number."""
    return float(date2num(ts))


def _zones(ax: plt.Axes, x0: pd.Timestamp, boundary: pd.Timestamp, x1: pd.Timestamp) -> None:
    """Tint history and forecast differently and mark where the data ends."""
    # Dates go in as matplotlib date numbers: the axis already holds dates, so they convert back correctly.
    ax.axvspan(_dnum(x0), _dnum(boundary), color=HISTORY_TINT, lw=0, zorder=0)
    ax.axvspan(_dnum(boundary), _dnum(x1), color=FORECAST_TINT, lw=0, zorder=0)
    ax.axvline(_dnum(boundary), color=INK_SOFT, lw=0.9, ls=(0, (3, 2)), zorder=2)


def _midpoint(a: pd.Timestamp, b: pd.Timestamp, skip: SkipAxis | None) -> pd.Timestamp:
    """Halfway between two dates; on a skip axis, halfway along the drawn width rather than in elapsed time."""
    if skip is None:
        return a + (b - a) / 2
    return skip.from_x((skip.to_x(a) + skip.to_x(b)) / 2)[0]


def _zone_labels(
    ax: plt.Axes, x0: pd.Timestamp, boundary: pd.Timestamp, x1: pd.Timestamp,
    taken: list[object], note: str | None, series: Sequence[np.ndarray] = (),
    skip: SkipAxis | None = None,
) -> None:
    """Direct labels for the two zones and a note saying when the data ends.

    The three labels share a baseline, which is raised until none of them sits
    on the data (a cap or floor line near the bottom, for example). A label
    wider than its zone is dropped rather than spilling over the boundary.
    """
    tr = ax.get_xaxis_transform()
    edge = float(ax.transData.transform((date2num(boundary), 0.0))[0])
    common = {'fontsize': _LABEL_PT, 'fontweight': 'bold', 'fontfamily': _sans(), 'zorder': 6}

    def build(y: float, va: str) -> list[tuple[Text, tuple[float, float]]]:
        labels = [
            (_text_at(ax, x0, y, 'HISTORY', dx=6, transform=tr, ha='left', va=va, color=OBSERVED_TEXT, **common),
             (-np.inf, edge)),
            (_text_at(ax, _midpoint(boundary, x1, skip), y, 'FORECAST', transform=tr, ha='center', va=va,
                      color=ACCENT, **common), (edge, np.inf)),
        ]
        if note:
            labels.append((_text_at(ax, boundary, y, note, dx=-6, transform=tr, ha='right', va=va,
                                    fontsize=_NOTE_PT, color=INK_SOFT, fontfamily=_sans(), zorder=6),
                           (-np.inf, edge)))
        return labels

    candidates = ((0.025, 'bottom'), (0.09, 'bottom'), (0.155, 'bottom'), (0.975, 'top'))
    for y, va in candidates:
        labels = build(y, va)
        trial: list[object] = list(taken)
        placed = []
        for text, (lo, hi) in labels:
            box = _box(text)
            ok = box.x0 >= lo and box.x1 <= hi and _free(box, trial, series)  # pyrefly:ignore[missing-attribute]
            if ok:
                trial.append(box)
            placed.append(ok)
        if all(placed) or y == candidates[-1][0]:
            taken[:] = trial
            for (text, _), ok in zip(labels, placed):
                if not ok:
                    text.remove()
            return
        for text, _ in labels:
            text.remove()


def _callout_box() -> dict[str, object]:
    return {'boxstyle': 'round,pad=0.35,rounding_size=0.6', 'fc': PAPER, 'ec': INK_SOFT, 'lw': 0.8}


def _trend_marker(ax: plt.Axes, when: pd.Timestamp, y: float) -> None:
    ax.plot([when], [y], 'o', ms=6, mfc=PAPER, mec=INK, mew=1.5, zorder=6)


def _changepoint_callouts(
    ax: plt.Axes, m: Prophet, fcst: pd.DataFrame, taken: list[object], span: pd.Timedelta,
    tiers: Sequence[float], with_text: bool = True,
) -> None:
    """Mark the largest significant trend changepoints and say which way the trend bends."""
    top = _top_changepoints(m, fcst)
    if top.empty:
        return
    xs = date2num(fcst['ds'])
    trend = fcst['trend'].to_numpy(dtype=float)
    for _, row in top.iterrows():
        when = row['ds']
        y = float(np.interp(date2num(when), xs, trend))
        if not with_text:
            _trend_marker(ax, when, y)
            continue
        label = f'{_fmt_date(when, span)}: trend bends {"upwards" if row["delta"] > 0 else "downwards"}'
        for tier in tiers:
            text = ax.text(when, tier, label, transform=ax.get_xaxis_transform(), ha='center', va='center',
                           fontsize=_NOTE_PT, color=INK, fontfamily=_sans(), bbox=_callout_box(), zorder=7)
            if not _fit_inside(ax, text) or not _claim(text, taken):
                text.remove()
                continue
            _trend_marker(ax, when, y)
            ax.annotate('', xy=(when, y), xycoords='data', xytext=(when, tier),
                        textcoords=ax.get_xaxis_transform(),
                        arrowprops={'arrowstyle': '-', 'color': INK_SOFT, 'lw': 0.8, 'ls': (0, (1, 2)),
                                    'shrinkA': 0, 'shrinkB': 4}, zorder=5)
            break


def _extrema_labels(ax: plt.Axes, x: Sequence[object], y: np.ndarray, multiplicative: bool) -> None:
    """Mark the highest and lowest point of a seasonal curve with their signed effect."""
    hi, lo = int(np.nanargmax(y)), int(np.nanargmin(y))
    for i, offset, va in ((hi, 9, 'bottom'), (lo, -9, 'top')):
        if hi == lo and i == lo:
            continue
        ax.plot([x[i]], [y[i]], 'o', ms=6, color=ACCENT, zorder=5)
        # A point at either end hangs its label inward, so it clears the axis tick labels beside it.
        share = i / max(len(y) - 1, 1)
        ha = 'left' if share < 0.06 else 'right' if share > 0.94 else 'center'
        _text_at(ax, x[i], float(y[i]), _signed_effect(float(y[i]), multiplicative), dy=offset, ha=ha,
                 va=va, fontsize=_LABEL_PT, fontweight='bold', color=INK, fontfamily=_sans(),
                 path_effects=_stroke(), zorder=6)


def _frame_layout(
    figsize: tuple[float, float], title: str | None, subtitle: str | None,
) -> tuple[str | None, str | None, float]:
    """Wrap the headline for the figure width and return it with the header height in inches."""
    width_in = figsize[0] - _TEXT_LEFT_IN - _MARGIN_RIGHT_IN
    title_text = _wrap(title, width_in, _TITLE_PT, 2, factor=0.6) if title else None
    sub_text = _wrap(subtitle, width_in, _SUBTITLE_PT, 3) if subtitle else None
    head_pt = _n_lines(title_text) * _TITLE_PT * 1.25 + _n_lines(sub_text) * _SUBTITLE_PT * 1.4
    gap_pt = 8 if title_text and sub_text else 0
    return title_text, sub_text, 0.3 + (head_pt + gap_pt) / 72 + 0.2


def _draw_frame(fig: plt.Figure, title: str | None, subtitle: str | None, source: str | None) -> None:
    """Figure furniture: accent bar, serif headline, sans subtitle and the source line."""
    w, h = fig.get_size_inches()
    x = _TEXT_LEFT_IN / w
    fig.add_artist(Rectangle((0, 1 - 0.1 / h), 0.7 / w, 0.1 / h, transform=fig.transFigure, color=ACCENT,
                             clip_on=False))
    y = 1 - 0.3 / h
    if title:
        fig.text(x, y, title, fontfamily=_serif(), fontweight='bold', fontsize=_TITLE_PT, color=INK,
                 va='top', ha='left', linespacing=1.15)
        y -= (_n_lines(title) * _TITLE_PT * 1.25 + 8) / 72 / h
    if subtitle:
        fig.text(x, y, subtitle, fontfamily=_sans(), fontsize=_SUBTITLE_PT, color=INK_SOFT, va='top',
                 ha='left', linespacing=1.35)
    if source:
        fig.text(x, 0.12 / h, source, fontfamily=_sans(), fontsize=_SOURCE_PT, color=INK_SOFT,
                 va='bottom', ha='left')


def _set_headline(ax: plt.Axes, title: str | None, subtitle: str | None, size: float = _PANEL_TITLE_PT,
                  sub_lines: int = 2) -> None:
    """Headline on the axes itself (user-supplied axes and component panels)."""
    width_in = ax.get_window_extent().width / ax.figure.dpi  # pyrefly:ignore[missing-attribute]
    sub_text = _wrap(subtitle, width_in, _SUBTITLE_PT, sub_lines) if subtitle else None
    title_text = _wrap(title, width_in, size, 2, factor=0.6) if title else None
    sub_pt = _n_lines(sub_text) * _SUBTITLE_PT * 1.4
    if sub_text:
        ax.annotate(sub_text, xy=(0, 1), xycoords='axes fraction', xytext=(0, 6), textcoords='offset points',
                    ha='left', va='bottom', fontsize=_SUBTITLE_PT, color=INK_SOFT, fontfamily=_sans(),
                    linespacing=1.35, annotation_clip=False)
    if title_text:
        ax.set_title(title_text, loc='left', pad=6 + (sub_pt + 7 if sub_text else 0), fontfamily=_serif(),
                     fontsize=size, fontweight='bold', color=INK)


def _open_axes(
    ax: plt.Axes | None,
    figsize: tuple[float, float],
    title: str | None,
    subtitle: str | None,
    source: str | None,
    xlabel_room: bool = False,
    ylabel_room: bool = False,
    sub_lines: int = 2,
) -> tuple[plt.Figure, plt.Axes]:
    """Use the caller's axes (headline on the axes) or build an editorial figure around new ones."""
    if ax is not None:
        _set_headline(ax, title, subtitle, sub_lines=sub_lines)
        return cast('plt.Figure', ax.get_figure()), ax
    left_in = 0.9 + (0.3 if ylabel_room else 0)
    title_text, sub_text, head_in = _frame_layout(figsize, title, subtitle)
    w, h = figsize
    # A long credit line (skip and weather notes) wraps to the page width instead of running off it.
    source = _wrap(source, w - _TEXT_LEFT_IN - 0.3, _SOURCE_PT) if source else source
    foot_in = 0.7 + (0.3 if xlabel_room else 0) + (0.25 + 0.15 * (_n_lines(source) - 1) if source else 0)
    fig = plt.figure(figsize=figsize, facecolor=PAPER)
    ax = fig.add_axes((left_in / w, foot_in / h, (w - left_in - _MARGIN_RIGHT_IN) / w,
                       max(h - head_in - foot_in, 0.5) / h))
    _draw_frame(fig, title_text, sub_text, source)
    return fig, ax


def _source_text(source: str | bool | None, default: str) -> str | None:
    return _resolve_text(source, lambda: default)


def _obs_phrase(m: Prophet) -> str:
    history = cast('pd.DataFrame', m.history)
    span = history['ds'].max() - history['ds'].min()
    return (f'{len(history):,} observations, {_fmt_date(history["ds"].min(), span)} to '
            f'{_fmt_date(history["ds"].max(), span)}')


def _forecast_zone_args(m: Prophet, fcst: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp] | None:
    """Start, boundary and end for the history and forecast zones, or None if nothing is forecast."""
    boundary = cast('pd.DataFrame', m.history)['ds'].max()
    if not (fcst['ds'] > boundary).any():
        return None
    return min(fcst['ds'].min(), boundary), boundary, fcst['ds'].max()


# ----------------------------------------------------------------------------
# Matplotlib
# ----------------------------------------------------------------------------

def _interval_label(
    ax: plt.Axes, m: Prophet, future: pd.DataFrame, taken: list[object], series: Sequence[np.ndarray],
    boundary: pd.Timestamp | None = None,
) -> None:
    """Name the shaded band beside its highest or lowest point nearby, or leave it unnamed if there is no room.

    The label stays inside the forecast zone (right of ``boundary``) and clear of the top and bottom edges of
    the axes, so in a narrow chart it is dropped rather than straddling the boundary or touching the frame.
    """
    upper = future['yhat_upper'].to_numpy(dtype=float)
    lower = future['yhat_lower'].to_numpy(dtype=float)
    n = len(upper)
    window = max(n // 12, 1)
    axes_box = ax.get_window_extent()
    edge = None if boundary is None else float(ax.transData.transform((date2num(boundary), 0.0))[0])
    margin = 6.0
    spots = [int(np.nanargmax(upper))] + [int(n * q) for q in (0.25, 0.5, 0.75)]
    # Above the band first; below its lowest point as the second choice.
    for above in (True, False):
        for spot in spots:
            spot = min(max(spot, 0), n - 1)
            lo, hi = max(spot - window, 0), min(spot + window + 1, n)
            if above:
                top = lo + int(np.nanargmax(upper[lo:hi]))
                y = float(upper[top])
            else:
                top = lo + int(np.nanargmin(lower[lo:hi]))
                y = float(lower[top])
            x = future['ds'].iloc[top]
            sign = 1 if above else -1
            label = _text_at(ax, x, y, f'{m.interval_width:.0%} interval', dy=16 * sign, ha='center',
                             va='bottom' if above else 'top', fontsize=_NOTE_PT, fontweight='bold',
                             color=ACCENT, fontfamily=_sans(), zorder=6)
            if _fit_inside(ax, label, None if edge is None else edge + margin):
                box = label.get_window_extent()
                if box.y1 <= axes_box.y1 - margin and box.y0 >= axes_box.y0 + margin and _claim(label, taken, series):
                    ax.annotate('', xy=(x, y), xytext=(0, 14 * sign), textcoords='offset points',
                                arrowprops={'arrowstyle': '-', 'color': ACCENT, 'lw': 0.7, 'shrinkA': 0,
                                            'shrinkB': 0})
                    return
            label.remove()


def plot(
    m: Prophet,
    fcst: pd.DataFrame,
    ax: plt.Axes | None = None,
    uncertainty: bool = True,
    plot_cap: bool = True,
    xlabel: str = "ds",
    ylabel: str = "y",
    figsize: tuple[int, int] = (10, 6),
    include_legend: bool = False,
    *,
    title: str | bool | None = None,
    subtitle: str | bool | None = None,
    annotate: bool = True,
    source: str | bool | None = None,
    skip_days: _SkipSpec = None,
    skip_hours: _SkipSpec = None,
    source_note: str | None = None,
) -> plt.Figure:
    """Plot the Prophet forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    ax: Optional matplotlib axes on which to plot. The headline then goes on
        the axes and no figure-level furniture is added.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    xlabel: Optional label name on X-axis. The default 'ds' draws no label,
        since the axis is plainly dates.
    ylabel: Optional label name on Y-axis. The default 'y' draws no label.
    figsize: Optional tuple width, height in inches.
    include_legend: Optional boolean to add legend to the plot.
    title: Headline. None writes one from the data, a string replaces it and
        False draws no headline (nor automatic subtitle).
    subtitle: Line under the headline. None writes one from the data, a
        string replaces it and False omits it.
    annotate: Draw the History and Forecast labels, the "data ends" note,
        the interval label, cap and floor labels and callouts on the largest
        significant trend changepoints.
    source: Credit line under the figure, only when this function creates the
        figure. None writes a default, a string replaces it and False omits it.
    skip_days: Weekdays to leave out of the time axis, so that for example
        Friday is drawn next to Monday: names ("Sat", "Saturday"), numbers
        (Monday is 0) or "weekends". Rows on those days are not drawn. See
        :func:`skip_axis`.
    skip_hours: Hours of the day to leave out of the time axis, for example
        ``range(0, 8)`` for sub-daily data that has no data at night.
    source_note: A sentence added to the default credit line, for facts the
        data alone cannot tell, such as where an extra regressor's future
        values came from. Ignored when ``source`` is given.

    Returns
    -------
    A matplotlib figure.
    """
    history = cast('pd.DataFrame', m.history)
    skip = _chart_axis(fcst['ds'], skip_days, skip_hours)
    if skip is not None:
        history = history[skip.keeps(history['ds'])]
        fcst = fcst[skip.keeps(fcst['ds'])]
    show_band = bool(uncertainty and m.uncertainty_samples and 'yhat_lower' in fcst)
    title_text, sub_text = _headline_texts(title, subtitle, lambda: _forecast_headline(m, fcst, show_band))
    sentences = [part.rstrip('.') for part in (f'NeoProphet forecast: {_obs_phrase(m)}',
                                               None if skip is None else _skip_note(skip), source_note) if part]
    default_source = '. '.join(sentences) + ('.' if len(sentences) > 1 else '')
    source_text = _source_text(source, default_source)
    xlabel_text = xlabel if xlabel != 'ds' else None
    ylabel_text = ylabel if ylabel != 'y' else None
    fig, ax = _open_axes(ax, figsize, title_text, sub_text, source_text,
                         xlabel_room=bool(xlabel_text), ylabel_room=bool(ylabel_text), sub_lines=4)
    _style_axes(ax)

    fcst_t = fcst['ds']
    hist_end = history['ds'].max()
    x0, x1 = min(history['ds'].min(), fcst_t.min()), max(hist_end, fcst_t.max())
    future = fcst[fcst_t > hist_end]
    past = fcst[fcst_t <= hist_end]
    n_obs = len(history)
    dots = 5.0 if n_obs <= 200 else 3.5 if n_obs <= 1500 else 2.6 if n_obs <= 5000 else 1.8
    ax.plot(history['ds'], history['y'], ls='-' if n_obs <= 400 else '', lw=0.8, marker='o', ms=dots,
            mfc=OBSERVED, mec='none', color=OBSERVED, alpha=0.7, label='Observed data points', zorder=3,
            rasterized=n_obs > 5000)
    forecast_line = future if not future.empty else fcst
    if not future.empty and not past.empty:
        ax.plot(past['ds'], past['yhat'], ls='-', c=ACCENT, lw=0.9, alpha=0.5, zorder=3)
    ax.plot(forecast_line['ds'], forecast_line['yhat'], ls='-', c=ACCENT, lw=1.8, label='Forecast', zorder=4,
            solid_capstyle='round')
    if 'cap' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['cap'], ls='--', c=INK_SOFT, lw=1.1, label='Maximum capacity', zorder=3)
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c=INK_SOFT, lw=1.1, label='Minimum capacity', zorder=3)
    if show_band:
        ax.fill_between(forecast_line['ds'], forecast_line['yhat_lower'], forecast_line['yhat_upper'],
                        color=ACCENT, alpha=BAND_ALPHA, lw=0, label='Uncertainty interval', zorder=2)

    candidates = _top_changepoints(m, fcst) if annotate else pd.DataFrame()
    if len(candidates):
        ax.plot(fcst_t, fcst['trend'], ls=(0, (1, 2.2)), c=INK_SOFT, lw=1.1, zorder=3, label='Trend')
    # Room above the data for the callouts and below it for the zone labels.
    lo, hi = ax.get_ylim()
    head = 0.05 + 0.08 * len(candidates)
    ax.set_ylim(lo - (0.05 * (hi - lo) if annotate else 0), hi + (hi - lo) * head / (1 - head))
    ax.set_xlim(x0, x1)
    _style_dates(ax, skip)
    _axis_label(ax, 'x', xlabel_text)
    _axis_label(ax, 'y', ylabel_text)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))

    zones = _forecast_zone_args(m, fcst)
    taken: list[object] = []
    span = x1 - x0
    # Labels steer clear of everything drawn as a line or point.
    series = [_series_px(ax, history['ds'], history['y']), _series_px(ax, fcst_t, fcst['yhat'])]
    for column in ('cap', 'floor'):
        if column in fcst and plot_cap:
            series.append(_series_px(ax, fcst_t, fcst[column]))
    if show_band:
        for column in ('yhat_lower', 'yhat_upper'):
            series.append(_series_px(ax, forecast_line['ds'], forecast_line[column]))
    if zones is not None:
        _zones(ax, *zones)
        if annotate:
            _zone_labels(ax, *zones, taken, f'data ends {_fmt_date(hist_end, span)}', series, skip)
    if annotate:
        for column, word in (('cap', 'Maximum capacity'), ('floor', 'Minimum capacity')):
            if column in fcst and plot_cap and (column == 'cap' or m.logistic_floor):
                note = _text_at(ax, x1, float(fcst[column].iloc[-1]), word, dx=-4, dy=4, ha='right', va='bottom',
                                fontsize=_NOTE_PT, color=INK_SOFT, fontfamily=_sans(), path_effects=_stroke(),
                                zorder=6)
                taken.append(_box(note, 2.0))
        if show_band and not future.empty:
            _interval_label(ax, m, future, taken, series, hist_end)
        tiers = [0.945 - 0.08 * i for i in range(len(candidates))]
        _changepoint_callouts(ax, m, fcst, taken, span, tiers)
    if include_legend:
        ax.legend(frameon=False, loc='upper left', prop={'family': _sans(), 'size': _TICK_PT})
    return fig


def plot_components(
    m: Prophet,
    fcst: pd.DataFrame,
    uncertainty: bool = True,
    plot_cap: bool = True,
    weekly_start: int = 0,
    yearly_start: int = 0,
    figsize: tuple[int, int] | None = None,
    *,
    title: str | bool | None = None,
    annotate: bool = True,
    panel_titles: bool = True,
    source: str | bool | None = None,
    skip_days: _SkipSpec = None,
    skip_hours: _SkipSpec = None,
) -> plt.Figure:
    """Plot the Prophet forecast components.

    Will plot whichever are available of: trend, holidays, weekly
    seasonality, yearly seasonality, and additive and multiplicative extra
    regressors.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.
    yearly_start: Optional int specifying the start day of the yearly
        seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
        by 1 day to Jan 2, and so on.
    figsize: Optional tuple width, height in inches.
    title: Figure headline. None writes one from the panels shown, a string
        replaces it and False draws none (nor its subtitle).
    annotate: Draw the highlights inside the panels (zone labels, changepoint
        markers, peaks and lows).
    panel_titles: Give each panel a plain-English headline written from the
        data. If False, panels carry just their component name.
    source: Credit line under the figure. None writes a default, a string
        replaces it and False omits it.
    skip_days: Weekdays to leave out of the time axes (names, numbers with
        Monday as 0, or "weekends"). The weekly profile then shows only the
        remaining days. See :func:`skip_axis`.
    skip_hours: Hours of the day to leave out of the time axes and of the
        daily profile.

    Returns
    -------
    A matplotlib figure.
    """
    skip = _chart_axis(fcst['ds'], skip_days, skip_hours)
    # Identify components to be plotted
    components = ['trend']
    if m.train_holiday_names is not None and 'holidays' in fcst:
        components.append('holidays')
    # Plot weekly seasonality, if present
    if 'weekly' in m.seasonalities and 'weekly' in fcst:
        components.append('weekly')
    # Yearly if present
    if 'yearly' in m.seasonalities and 'yearly' in fcst:
        components.append('yearly')
    # Other seasonalities
    components.extend([
        name for name in sorted(m.seasonalities)
        if name in fcst and name not in ['weekly', 'yearly']
    ])
    regressors = {'additive': False, 'multiplicative': False}
    for name, props in m.extra_regressors.items():
        regressors[props['mode']] = True
    for mode in ['additive', 'multiplicative']:
        if regressors[mode] and 'extra_regressors_{}'.format(mode) in fcst:
            components.append('extra_regressors_{}'.format(mode))
    npanel = len(components)

    w = figsize[0] if figsize else 9
    assert m.component_modes is not None
    multiplicative = any(c in m.component_modes['multiplicative'] for c in components)
    names = [
        f'{c.rsplit("_", 1)[-1]} regressors' if c.startswith('extra_regressors') else c if c in ('trend', 'holidays')
        else f'{c.replace("_", " ")} pattern'
        for c in components
    ]
    listed = names[0] if npanel == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]

    def auto_headline() -> tuple[str, str]:
        unit = ('Multiplicative effects are percentage changes on the trend; the rest are in the units '
                'of the data.' if multiplicative else 'Effects are in the same units as the data.')
        return f'What shapes the forecast: {listed}', f'Each panel isolates one part of the model. {unit}'

    title_text, sub_text = _headline_texts(title, None, auto_headline)
    default_source = f'NeoProphet forecast components: {_obs_phrase(m)}'
    source_text = _source_text(source, default_source if skip is None else f'{default_source}. {_skip_note(skip)}')
    left_in = 0.9
    title_wrapped, sub_wrapped, head_in = _frame_layout((w, 1), title_text, sub_text)
    foot_in = 0.3 + (0.25 if source_text else 0)
    header_in = 0.95 if panel_titles else 0.5
    tick_in = 0.5
    h = figsize[1] if figsize else head_in + foot_in + npanel * (header_in + tick_in + 1.7)
    block = (h - head_in - foot_in) / npanel
    axes_h = max(block - header_in - tick_in, 0.8)
    pitch = axes_h + header_in + tick_in
    fig = plt.figure(figsize=(w, h), facecolor=PAPER)
    _draw_frame(fig, title_wrapped, sub_wrapped, source_text)
    axes = []
    for i in range(npanel):
        bottom = h - head_in - (i + 1) * pitch + tick_in
        axes.append(fig.add_axes((left_in / w, bottom / h, (w - left_in - _MARGIN_RIGHT_IN) / w, axes_h / h)))

    dt = cast('pd.DataFrame', m.history)['ds'].diff()
    min_dt = dt.iloc[cast('np.ndarray', dt.values).nonzero()[0]].min()

    for ax, plot_name in zip(axes, components):
        # A string title is shown as given and suppresses the derived sentence.
        panel_title = None if panel_titles else _pretty(plot_name)
        if plot_name == 'trend':
            plot_forecast_component(
                m=m, fcst=fcst, name='trend', ax=ax, uncertainty=uncertainty,
                plot_cap=plot_cap, title=panel_title, annotate=annotate,
                skip_days=skip_days, skip_hours=skip_hours,
            )
        elif plot_name in m.seasonalities:
            if (
                (plot_name == 'weekly' or m.seasonalities[plot_name]['period'] == 7)
                and (min_dt == pd.Timedelta(days=1))
            ):
                plot_weekly(
                    m=m, name=plot_name, ax=ax, uncertainty=uncertainty, weekly_start=weekly_start,
                    title=panel_title, annotate=annotate, skip_days=skip_days,
                )
            elif plot_name == 'yearly' or m.seasonalities[plot_name]['period'] == 365.25:
                plot_yearly(
                    m=m, name=plot_name, ax=ax, uncertainty=uncertainty, yearly_start=yearly_start,
                    title=panel_title, annotate=annotate,
                )
            else:
                plot_seasonality(
                    m=m, name=plot_name, ax=ax, uncertainty=uncertainty,
                    title=panel_title, annotate=annotate, skip_days=skip_days, skip_hours=skip_hours,
                )
        elif plot_name in [
            'holidays',
            'extra_regressors_additive',
            'extra_regressors_multiplicative',
        ]:
            plot_forecast_component(
                m=m, fcst=fcst, name=plot_name, ax=ax, uncertainty=uncertainty,
                plot_cap=False, title=panel_title, annotate=annotate,
                skip_days=skip_days, skip_hours=skip_hours,
            )
    return fig


def plot_forecast_component(
    m: Prophet,
    fcst: pd.DataFrame,
    name: str,
    ax: plt.Axes | None = None,
    uncertainty: bool = True,
    plot_cap: bool = False,
    figsize: tuple[int, int] = (10, 6),
    *,
    title: str | bool | None = None,
    annotate: bool = True,
    source: str | bool | None = None,
    skip_days: _SkipSpec = None,
    skip_hours: _SkipSpec = None,
) -> Sequence[plt.Artist]:
    """Plot a particular component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    name: Name of the component to plot.
    ax: Optional matplotlib Axes to plot on.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: Optional tuple width, height in inches.
    title: Headline. None writes one from the data, a string replaces it and
        False draws no headline.
    annotate: Draw the History and Forecast zone labels and changepoint markers
        (trend), or the callout on the largest holiday effect.
    source: Credit line, only when this function creates the figure. None
        writes a default, a string replaces it and False omits it.
    skip_days, skip_hours: Weekdays and hours of the day to leave out of the
        time axis, as for :func:`plot`.

    Returns
    -------
    a list of matplotlib artists
    """
    artists: list[plt.Artist] = []
    assert m.component_modes
    mult = name in m.component_modes['multiplicative']
    skip = _chart_axis(fcst['ds'], skip_days, skip_hours)
    if skip is not None:
        fcst = fcst[skip.keeps(fcst['ds'])].reset_index(drop=True)

    def auto() -> tuple[str, str]:
        if name == 'trend':
            return _trend_headline(m, fcst)
        if name == 'holidays':
            return _holiday_headline(m, fcst)
        if name.startswith('extra_regressors_'):
            return _regressor_headline(m, name, fcst[name])
        return _pretty(name), ''

    title_text, sub_text = _headline_texts(title, None, auto)
    _, ax = _open_axes(ax, figsize, title_text, sub_text, _source_text(source, f'NeoProphet forecast: {_obs_phrase(m)}'))
    _style_axes(ax)

    fcst_t = fcst['ds']
    zones = _forecast_zone_args(m, fcst)
    x0, x1 = fcst_t.min(), fcst_t.max()
    if zones is not None:
        _zones(ax, *zones)
    values = fcst[name].to_numpy(dtype=float)
    drawn: list[object] = [values]
    show_band = bool(uncertainty and m.uncertainty_samples and name + '_lower' in fcst)
    # Room for the labels drawn inside the panel: zone labels below the trend, a callout beside a holiday stem.
    pad_bottom, pad_top = (0.22 if annotate and name == 'trend' else 0.12), 0.12
    if name == 'holidays':
        nonzero = values != 0
        ax.axhline(0, color=INK_SOFT, lw=0.8, zorder=2)
        artists.append(ax.vlines(fcst_t[nonzero], 0, values[nonzero], colors=ACCENT, lw=1.6, zorder=3))
        artists += ax.plot(fcst_t[nonzero], values[nonzero], ls='', marker='o', ms=3.5, c=ACCENT, zorder=4)
        biggest = _biggest_holiday(m, fcst) if annotate else None
        if biggest is not None:
            hname, when, value = biggest
            pad_bottom, pad_top = (pad_bottom, 0.4) if value >= 0 else (0.4, pad_top)
            left_half = when < x0 + (x1 - x0) / 2
            _text_at(ax, when, value, f'{hname}: {_signed_effect(value, mult)}', dx=8 if left_half else -8,
                     dy=6 if value >= 0 else -6, ha='left' if left_half else 'right',
                     va='bottom' if value >= 0 else 'top', fontsize=_NOTE_PT, fontweight='bold', color=INK,
                     fontfamily=_sans(), path_effects=_stroke(), zorder=6)
    else:
        artists += ax.plot(fcst_t, values, ls='-', c=ACCENT, lw=1.9, solid_capstyle='round', zorder=4)
        if show_band:
            artists += [ax.fill_between(fcst_t, fcst[name + '_lower'], fcst[name + '_upper'], color=ACCENT,
                                        alpha=BAND_ALPHA, lw=0, zorder=3)]
            drawn += [fcst[name + '_lower'].to_numpy(dtype=float), fcst[name + '_upper'].to_numpy(dtype=float)]
        if name != 'trend':
            ax.axhline(0, color=INK_SOFT, lw=0.8, zorder=2)
    for column, word in (('cap', 'Maximum capacity'), ('floor', 'Minimum capacity')):
        if column in fcst and plot_cap and (column == 'cap' or m.logistic_floor):
            artists += ax.plot(fcst_t, fcst[column], ls='--', c=INK_SOFT, lw=1.1, zorder=3)
            drawn.append(fcst[column].to_numpy(dtype=float))
            if annotate:
                _text_at(ax, x1, float(fcst[column].iloc[-1]), word, dx=-4, dy=4, ha='right', va='bottom',
                         fontsize=_NOTE_PT, color=INK_SOFT, fontfamily=_sans(), path_effects=_stroke(), zorder=6)
    _set_ylim(ax, drawn, bottom=pad_bottom, top=pad_top)
    ax.set_xlim(x0, x1)
    _style_dates(ax, skip)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    if annotate and name == 'trend':
        taken: list[object] = []
        if zones is not None:
            series = [_series_px(ax, fcst_t, arr) for arr in drawn]
            _zone_labels(ax, *zones, taken, None, series, skip)
        _changepoint_callouts(ax, m, fcst, taken, x1 - x0, [], with_text=False)
    if mult:
        ax = set_y_as_percent(ax)
    return artists


def _blank_points(
    ds: pd.Series, name: str, period: float, skip_days: object, skip_hours: object,
) -> np.ndarray:
    """Points of a seasonal cycle that fall on skipped days or hours and so should not be drawn.

    A weekly cycle (period 7) is blanked by both weekday and hour, a daily one
    (period 1) by hour only. Other periods do not line up with the calendar,
    so nothing is blanked.
    """
    days, hours = parse_skip_days(skip_days), parse_skip_hours(skip_hours)
    weekly = name == 'weekly' or period == 7
    daily = name == 'daily' or period == 1
    if not (weekly or daily) or not (days or hours):
        return np.zeros(len(ds), dtype=bool)
    shell = SkipAxis(days if weekly else frozenset(), hours)
    return ~shell.keeps(ds)


def seasonality_plot_df(
    m: Prophet,
    ds: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Prepare dataframe for plotting seasonal components.

    Parameters
    ----------
    m: Prophet model.
    ds: List of dates for column ds.

    Returns
    -------
    A dataframe with seasonal components on ds.
    """
    df_dict = {'ds': ds, 'cap': 1., 'floor': 0.}
    for name in m.extra_regressors:
        df_dict[name] = 0.
    # Activate all conditional seasonality columns
    for props in m.seasonalities.values():
        if props['condition_name'] is not None:
            df_dict[props['condition_name']] = True
    df = pd.DataFrame(df_dict)
    df = m.setup_dataframe(df)
    return df


def _seasonal_axes(
    m: Prophet,
    ax: plt.Axes | None,
    figsize: tuple[int, int],
    title: str | bool | None,
    source: str | bool | None,
    auto: Callable[[], tuple[str, str]],
) -> plt.Axes:
    """Shared opening for the seasonality plots: resolve the headline, create or style the axes."""
    title_text, sub_text = _headline_texts(title, None, auto)
    _, ax = _open_axes(ax, figsize, title_text, sub_text, _source_text(source, f'NeoProphet forecast: {_obs_phrase(m)}'))
    _style_axes(ax)
    return ax


def plot_weekly(
    m: Prophet,
    ax: plt.Axes | None = None,
    uncertainty: bool = True,
    weekly_start: int = 0,
    figsize: tuple[int, int] = (10, 6),
    name: str = 'weekly',
    *,
    title: str | bool | None = None,
    annotate: bool = True,
    source: str | bool | None = None,
    skip_days: _SkipSpec = None,
) -> Sequence[plt.Artist]:
    """Plot the weekly component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    ax: Optional matplotlib Axes to plot on. One will be created if this
        is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.
    figsize: Optional tuple width, height in inches.
    name: Name of seasonality component if changed from default 'weekly'.
    title: Headline. None writes one from the data (for example "Weekly
        pattern: Mondays are highest"), a string replaces it and False draws
        none.
    annotate: Label the highest and lowest day with their effect.
    source: Credit line, only when this function creates the figure. None
        writes a default, a string replaces it and False omits it.
    skip_days: Weekdays to leave out of the profile (names, numbers with
        Monday as 0, or "weekends"), for data that never has them.

    Returns
    -------
    a list of matplotlib artists
    """
    artists: list[plt.Artist] = []
    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=7) +
            pd.Timedelta(days=weekly_start))
    skipped = parse_skip_days(skip_days)
    if skipped:
        days = days[~days.dayofweek.isin(sorted(skipped))]
    df_w = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_w)
    mult = m.seasonalities[name]['mode'] == 'multiplicative'
    values = seas[name].to_numpy(dtype=float)
    ax = _seasonal_axes(m, ax, figsize, title, source, lambda: _seasonality_headline(m, name, days, values, 'weekly'))
    x = list(range(len(days)))
    ax.axhline(0, color=INK_SOFT, lw=0.8, zorder=2)
    drawn: list[object] = [values]
    if uncertainty and m.uncertainty_samples:
        artists += [ax.fill_between(x, seas[name + '_lower'], seas[name + '_upper'], color=ACCENT,
                                    alpha=BAND_ALPHA, lw=0, zorder=3)]
        drawn += [seas[name + '_lower'], seas[name + '_upper']]
    artists += ax.plot(x, values, ls='-', c=ACCENT, lw=1.9, marker='o', ms=4.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([d[:3] for d in days.day_name()], fontfamily=_sans())
    ax.set_xlim(-0.3, len(days) - 0.7)
    _set_ylim(ax, drawn, bottom=0.35 if annotate else 0.1, top=0.35 if annotate else 0.1)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    if annotate:
        _extrema_labels(ax, x, values, mult)
    if mult:
        ax = set_y_as_percent(ax)
    return artists


def plot_yearly(
    m: Prophet,
    ax: plt.Axes | None = None,
    uncertainty: bool = True,
    yearly_start: int = 0,
    figsize: tuple[int, int] = (10, 6),
    name: str = 'yearly',
    *,
    title: str | bool | None = None,
    annotate: bool = True,
    source: str | bool | None = None,
) -> Sequence[plt.Artist]:
    """Plot the yearly component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    ax: Optional matplotlib Axes to plot on. One will be created if
        this is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    yearly_start: Optional int specifying the start day of the yearly
        seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
        by 1 day to Jan 2, and so on.
    figsize: Optional tuple width, height in inches.
    name: Name of seasonality component if previously changed from default 'yearly'.
    title: Headline. None writes one from the data (for example "Yearly
        pattern: peaks in January"), a string replaces it and False draws
        none.
    annotate: Shade and label the strongest and weakest month.
    source: Credit line, only when this function creates the figure. None
        writes a default, a string replaces it and False omits it.

    Returns
    -------
    a list of matplotlib artists
    """
    artists: list[plt.Artist] = []
    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=365) +
            pd.Timedelta(days=yearly_start))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    mult = m.seasonalities[name]['mode'] == 'multiplicative'
    values = seas[name].to_numpy(dtype=float)
    ax = _seasonal_axes(m, ax, figsize, title, source, lambda: _seasonality_headline(m, name, days, values, 'yearly'))
    ax.axhline(0, color=INK_SOFT, lw=0.8, zorder=2)
    drawn: list[object] = [values]
    if uncertainty and m.uncertainty_samples:
        artists += [ax.fill_between(df_y['ds'], seas[name + '_lower'], seas[name + '_upper'], color=ACCENT,
                                    alpha=BAND_ALPHA, lw=0, zorder=3)]
        drawn += [seas[name + '_lower'], seas[name + '_upper']]
    artists += ax.plot(df_y['ds'], values, ls='-', c=ACCENT, lw=1.9, zorder=4)
    ax.set_xlim(df_y['ds'].min(), df_y['ds'].max())
    _set_ylim(ax, drawn, bottom=0.35 if annotate else 0.1, top=0.35 if annotate else 0.1)
    width_in = ax.get_window_extent().width / ax.figure.dpi  # pyrefly:ignore[missing-attribute]
    ax.xaxis.set_major_locator(MonthLocator(range(1, 13), bymonthday=1, interval=1 if width_in >= 6 else 2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos=None: '{dt:%b}'.format(dt=num2date(x))))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    if annotate and np.nanmax(values) - np.nanmin(values) > 0:
        dates = pd.DatetimeIndex(df_y['ds'])
        monthly = pd.Series(values, index=dates.month).groupby(level=0).mean()
        tr = ax.get_xaxis_transform()
        # Labels sit at the far side of the curve from the extreme they name, clear of the line.
        for month, color, word, frac, va in ((int(monthly.idxmax()), ACCENT, 'Peak', 0.05, 'bottom'),
                                              (int(monthly.idxmin()), INK_SOFT, 'Low', 0.95, 'top')):
            hit = dates[dates.month == month]
            start, end = hit.min(), hit.max() + pd.Timedelta(days=1)
            ax.axvspan(_dnum(start), _dnum(end), color=color, alpha=0.12, lw=0, zorder=1)
            near_left = start - dates.min() < pd.Timedelta(days=20)
            near_right = dates.max() - end < pd.Timedelta(days=20)
            ax.text(_dnum(start if near_left else end if near_right else start + (end - start) / 2), frac,
                    f'{word}: {start:%b}', transform=tr, ha='left' if near_left else 'right' if near_right else 'center',
                    va=va, fontsize=_NOTE_PT, fontweight='bold', color=color, fontfamily=_sans(),
                    path_effects=_stroke(), zorder=6)
    if mult:
        ax = set_y_as_percent(ax)
    return artists


def plot_seasonality(
    m: Prophet,
    name: str,
    ax: plt.Axes | None = None,
    uncertainty: bool = True,
    figsize: tuple[int, int] = (10, 6),
    *,
    title: str | bool | None = None,
    annotate: bool = True,
    source: str | bool | None = None,
    skip_days: _SkipSpec = None,
    skip_hours: _SkipSpec = None,
) -> Sequence[plt.Artist]:
    """Plot a custom seasonal component.

    Parameters
    ----------
    m: Prophet model.
    name: Seasonality name, like 'daily', 'weekly'.
    ax: Optional matplotlib Axes to plot on. One will be created if
        this is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    figsize: Optional tuple width, height in inches.
    title: Headline. None writes one from the data (for example "Daily
        pattern: peaks at 14:00"), a string replaces it and False draws none.
    annotate: Label the highest and lowest point with their effect.
    source: Credit line, only when this function creates the figure. None
        writes a default, a string replaces it and False omits it.
    skip_days, skip_hours: Weekdays and hours of the day the data never has.
        The curve is left blank there when the seasonality repeats weekly
        (days and hours) or daily (hours); other periods are drawn in full.

    Returns
    -------
    a list of matplotlib artists
    """
    artists: list[plt.Artist] = []
    # Compute seasonality from Jan 1 through a single period.
    start = pd.to_datetime('2017-01-01 0000')
    period = m.seasonalities[name]['period']
    end = start + pd.Timedelta(days=period)
    plot_points = 200
    # https://github.com/pandas-dev/pandas-stubs/issues/1645
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points))  # pyrefly:ignore[no-matching-overload]
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    mult = m.seasonalities[name]['mode'] == 'multiplicative'
    blank = _blank_points(df_y['ds'], name, period, skip_days, skip_hours)
    seas = seas.mask(pd.Series(blank, index=seas.index), axis=0) if blank.any() else seas
    values = seas[name].to_numpy(dtype=float)
    ax = _seasonal_axes(m, ax, figsize, title, source, lambda: _seasonality_headline(m, name, df_y['ds'], values))
    drawn: list[object] = [values]
    ax.axhline(0, color=INK_SOFT, lw=0.8, zorder=2)
    if uncertainty and m.uncertainty_samples:
        artists += [ax.fill_between(
            df_y['ds'], seas[name + '_lower'], seas[name + '_upper'],
            color=ACCENT, alpha=BAND_ALPHA, lw=0, zorder=3)]
        drawn += [seas[name + '_lower'], seas[name + '_upper']]
    artists += ax.plot(df_y['ds'], values, ls='-', c=ACCENT, lw=1.9, zorder=4)
    ax.set_xlim(df_y['ds'].min(), df_y['ds'].max())
    _set_ylim(ax, drawn, bottom=0.35 if annotate else 0.1, top=0.35 if annotate else 0.1)
    n_ticks = 8
    if period <= 2:
        # Whole-hour steps (every 3 hours for a day) read better than eighths of the period.
        total_hours = period * 24
        step = next(s for s in (1, 2, 3, 4, 6, 12, 24, 48) if total_hours / s <= 9)
        xticks = [(start + pd.Timedelta(hours=k * step)).to_pydatetime() for k in range(int(total_hours // step) + 1)]
    elif name not in ('weekly', 'yearly') and period > 14:
        # Whole days at a round step (every 5 days for a month) read better than eighths of the period.
        step_days = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if period / s <= 8)
        xticks = [(start + pd.Timedelta(days=k * step_days)).to_pydatetime() for k in range(int(period // step_days) + 1)]
    else:
        # https://github.com/pandas-dev/pandas-stubs/issues/1645
        xticks = pd.to_datetime(np.linspace(start.value, end.value, n_ticks)  # pyrefly:ignore[no-matching-overload]
            ).to_pydatetime()
    ax.set_xticks(xticks)
    xlabel: str | None
    if name == 'yearly':
        fmt = FuncFormatter(
            lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x)))
        xlabel = None
    elif name == 'weekly':
        fmt = FuncFormatter(
            lambda x, pos=None: '{dt:%a %H:%M}'.format(dt=num2date(x)))
        xlabel = None
    elif name == 'daily':
        fmt = FuncFormatter(
            lambda x, pos=None: '{dt:%H:%M}'.format(dt=num2date(x)))
        xlabel = 'Hour of day'
    elif period <= 2:
        fmt = FuncFormatter(
            lambda x, pos=None: '{dt:%H:%M}'.format(dt=num2date(x)))
        xlabel = 'Hours'
    else:
        fmt = FuncFormatter(
            lambda x, pos=None: '{:g}'.format(round((num2date(x).replace(tzinfo=None) - start).total_seconds() / 86400, 1)))
        xlabel = 'Days'
    ax.xaxis.set_major_formatter(fmt)
    _axis_label(ax, 'x', xlabel)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    if annotate:
        _extrema_labels(ax, list(df_y['ds']), values, mult)
    if mult:
        ax = set_y_as_percent(ax)
    return artists


def set_y_as_percent(ax: _AxT) -> _AxT:
    # A formatter, not fixed tick labels, so the labels stay right if the limits change later.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos=None: '{0:.4g}%'.format(100 * y + 0.0)))
    return ax


def add_changepoints_to_plot(
    ax: plt.Axes,
    m: Prophet,
    fcst: pd.DataFrame,
    threshold: float = 0.01,
    cp_color: str = INK,
    cp_linestyle: str = '--',
    trend: bool = True,
) -> list[plt.Line2D]:
    """Add markers for significant changepoints to prophet forecast plot.

    Example:
    fig = m.plot(forecast)
    add_changepoints_to_plot(fig.gca(), m, forecast)

    Parameters
    ----------
    ax: axis on which to overlay changepoint markers.
    m: Prophet model.
    fcst: Forecast output from m.predict.
    threshold: Threshold on trend change magnitude for significance.
    cp_color: Color of changepoint markers.
    cp_linestyle: Linestyle for changepoint markers.
    trend: If True, will also overlay the trend.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if trend:
        artists.extend(ax.plot(fcst['ds'], fcst['trend'], c=cp_color, lw=1.2, zorder=5))

    assert m.changepoints is not None
    for cp in _significant_changepoints(m, threshold)['ds']:
        artists.append(ax.axvline(x=cp, c=cp_color, ls=cp_linestyle, lw=0.8, alpha=0.8, zorder=5))
    return artists


_METRIC_WORDS = {
    # metric: (plain-English name, is a fraction shown as a percentage)
    'mape': ('Typical error', True),
    'mdape': ('Typical error (median)', True),
    'smape': ('Typical error (symmetric)', True),
    'mae': ('Typical miss', False),
    'rmse': ('Typical miss', False),
    'mse': ('Squared error', False),
    'coverage': ('Interval coverage', True),
}


def _cv_headline(
    metric: str, markers: Sequence[tuple[str, float]], fmt: Callable[[float], str], clipped: float,
    line_clipped: bool, ratio: float | None,
) -> tuple[str, str]:
    noun = _METRIC_WORDS.get(metric, (metric, False))[0]
    if not markers:
        return f'{noun} by forecast horizon', ''
    first, last = markers[0], markers[-1]
    if metric == 'coverage':
        title = f'The interval caught {fmt(float(np.mean([v for _, v in markers])))} of actual values'
        sub = f'{noun} {first[0]} ahead: {fmt(first[1])}.'
        if len(markers) > 1:
            sub += f' By {last[0]} ahead: {fmt(last[1])}.'
    else:
        if ratio is None:
            title = f'{noun} by forecast horizon'
        elif ratio > 1.15:
            title = 'The further ahead we predict, the bigger the miss'
        elif ratio < 0.87:
            title = 'Error is lower at longer horizons'
        else:
            title = 'Error is similar at every horizon'
        sub = f'{noun} {first[0]} ahead: {fmt(first[1])}.'
        if len(markers) > 1:
            sub += f' By {last[0]} ahead it is {fmt(last[1])}.'
    sub += ' Faint dots are single predictions checked against what really happened; the line is their rolling average.'
    if clipped > 0:
        what = 'single predictions, and part of the rolling average,' if line_clipped else 'single predictions'
        sub += f' The largest {_fmt_pct(clipped)} of {what} run off the top of the chart.'
    return title, sub


def plot_cross_validation_metric(
    df_cv: pd.DataFrame,
    metric: str,
    rolling_window: float = 0.1,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
    color: str = ACCENT,
    point_color: str = OBSERVED,
    *,
    title: str | bool | None = None,
    annotate: bool = True,
    source: str | bool | None = None,
) -> plt.Figure:
    """Plot a performance metric vs. forecast horizon from cross validation.

    Cross validation produces a collection of out-of-sample model predictions
    that can be compared to actual values, at a range of different horizons
    (distance from the cutoff). This computes a specified performance metric
    for each prediction, and aggregated over a rolling window with horizon.

    This uses prophet.diagnostics.performance_metrics to compute the metrics.
    Valid values of metric are 'mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', and 'coverage'.

    rolling_window is the proportion of data included in the rolling window of
    aggregation. The default value of 0.1 means 10% of data are included in the
    aggregation for computing the metric.

    As a concrete example, if metric='mse', then this plot will show the
    squared error for each cross validation prediction, along with the MSE
    averaged over rolling windows of 10% of the data.

    Parameters
    ----------
    df_cv: The output from prophet.diagnostics.cross_validation.
    metric: Metric name, one of ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage'].
    rolling_window: Proportion of data to use for rolling average of metric.
        In [0, 1]. Defaults to 0.1.
    ax: Optional matplotlib axis on which to plot. If not given, a new figure
        will be created.
    figsize: Optional tuple width, height in inches.
    color: Optional color for plot and error points, useful when plotting
        multiple model performances on one axis for comparison.
    point_color: Optional color for the individual prediction points.
    title: Headline. None writes one from the data, a string replaces it and
        False draws none (nor its subtitle).
    annotate: Mark the error at roughly a third, two thirds and the full
        horizon with plain-English labels, and clip extreme outliers off the
        top of the chart (saying so in the subtitle).
    source: Credit line, only when this function creates the figure. None
        writes a default, a string replaces it and False omits it.

    Returns
    -------
    a matplotlib figure.
    """
    # Get the metric at the level of individual predictions, and with the rolling window.
    df_none = performance_metrics(df_cv, metrics=[metric], rolling_window=-1)
    df_h = performance_metrics(df_cv, metrics=[metric], rolling_window=rolling_window)

    assert df_none is not None
    assert df_h is not None

    # Some work because matplotlib does not handle timedelta
    # Target ~10 ticks.
    tick_w = max(df_none['horizon'].astype('timedelta64[ns]')) / 10.
    # Find the largest time resolution that has <1 unit per bin.
    dts: list[Literal["D", "h", "m", "s", "ms", "us", "ns"]]
    dts = ['D', 'h', 'm', 's', 'ms', 'us', 'ns']
    dt_names = [
        'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds',
        'nanoseconds'
    ]
    dt_conversions = [
        24 * 60 * 60 * 10 ** 9,
        60 * 60 * 10 ** 9,
        60 * 10 ** 9,
        10 ** 9,
        10 ** 6,
        10 ** 3,
        1.,
    ]
    for i, dt in enumerate(dts):
        if np.timedelta64(1, dt) < np.timedelta64(tick_w, 'ns'):
            break

    x_plt = np.asarray(df_none['horizon'].astype('timedelta64[ns]')).view(np.int64) / float(dt_conversions[i])
    x_plt_h = np.asarray(df_h['horizon'].astype('timedelta64[ns]')).view(np.int64) / float(dt_conversions[i])
    y_all = df_none[metric].to_numpy(dtype=float)
    y_h = df_h[metric].to_numpy(dtype=float)

    noun, is_fraction = _METRIC_WORDS.get(metric, (metric, False))

    def fmt(v: float) -> str:
        return _fmt_pct(v) if is_fraction else _fmt_number(v)

    # Markers at about a third, two thirds and the end of the horizon, with plain-English horizons.
    markers: list[tuple[str, float, float]] = []
    if len(x_plt_h):
        for share in (1 / 3, 2 / 3, 1.0):
            idx = int(np.argmin(np.abs(x_plt_h - share * x_plt_h.max())))
            value = float(y_h[idx])
            phrase = _duration_phrase(pd.Timedelta(float(x_plt_h[idx]) * float(dt_conversions[i]), unit='ns'))
            if np.isfinite(value) and phrase not in [p for p, _, _ in markers]:
                markers.append((phrase, float(x_plt_h[idx]), value))

    # Single predictions can be wildly off (a tiny actual makes MAPE explode): clip, and say so.
    finite = y_all[np.isfinite(y_all)]
    clipped = 0.0
    top_limit: float | None = None
    line_clipped = False
    if annotate and finite.size:
        ceiling = float(np.quantile(finite, 0.99)) * 1.15
        if ceiling > 0 and float(finite.max()) > 3 * ceiling:
            top_limit = ceiling
            clipped = float((finite > ceiling).mean())
            line_clipped = bool((y_h > ceiling).any())

    # Compare the first and last thirds of the rolling line, so one noisy endpoint cannot set the headline.
    valid = y_h[np.isfinite(y_h)]
    third = max(len(valid) // 3, 1)
    early, late = (float(np.median(valid[:third])), float(np.median(valid[-third:]))) if len(valid) >= 3 else (0.0, 0.0)
    ratio = late / early if early > 0 else None
    title_text, sub_text = _headline_texts(
        title, None, lambda: _cv_headline(metric, [(p, v) for p, _, v in markers], fmt, clipped, line_clipped, ratio))
    fig, ax = _open_axes(ax, figsize, title_text, sub_text, _source_text(source, 'NeoProphet cross-validation'),
                         xlabel_room=True, ylabel_room=True, sub_lines=4)
    _style_axes(ax)

    ax.plot(x_plt, y_all, ls='', marker='o', ms=3.5, mfc=point_color, mec='none', alpha=0.2, rasterized=True,
            zorder=2)
    # Break the line where it leaves the chart instead of drawing a near-vertical stroke off the top.
    ax.plot(x_plt_h, y_h if top_limit is None else np.where(y_h > top_limit, np.nan, y_h), '-', c=color, lw=2.2,
            solid_capstyle='round', zorder=4)
    if top_limit is not None:
        ax.set_ylim(0, top_limit)
    ax.set_xlim(0, float(np.nanmax(x_plt)) * 1.03 if len(x_plt) else 1)
    if is_fraction:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos=None: f'{v * 100 + 0.0:.4g}%'))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    _axis_label(ax, 'x', f'How far ahead we were predicting ({dt_names[i]})')
    _axis_label(ax, 'y', f'{noun} ({metric.upper()})')
    if annotate:
        taken: list[object] = []
        for k, (phrase, x_at, value) in enumerate(markers):
            if top_limit is not None and value > top_limit:
                continue
            ax.plot([x_at], [value], 'o', ms=6.5, mfc=PAPER, mec=color, mew=2.0, zorder=7)
            last = k == len(markers) - 1
            label = _text_at(ax, x_at, value, f'{phrase} ahead: {fmt(value)}', dx=-8 if last else 0, dy=30,
                             ha='right' if last else 'center', va='center', fontsize=_NOTE_PT,
                             fontweight='bold', color=INK, fontfamily=_sans(), bbox=_callout_box(), zorder=8)
            if _claim(label, taken):
                ax.annotate('', xy=(x_at, value), xytext=(0, 30), textcoords='offset points',
                            arrowprops={'arrowstyle': '-', 'color': INK_SOFT, 'lw': 0.8, 'ls': (0, (1, 2)),
                                        'shrinkA': 0, 'shrinkB': 4}, zorder=5)
            else:
                label.remove()
    return fig


# ----------------------------------------------------------------------------
# Plotly
# ----------------------------------------------------------------------------

def _plotly_title(title: str | None, subtitle: str | None, width_px: int) -> dict[str, object] | None:
    """Plotly title block: serif headline over a sans subtitle."""
    if not title:
        return None
    block: dict[str, object] = {
        'text': title, 'x': 0.0, 'xanchor': 'left', 'xref': 'paper',
        'font': {'family': _PLOTLY_SERIF, 'size': 20, 'color': INK},
    }
    if subtitle:
        wrapped = _wrap(subtitle, width_px / 96.0, 12, 3).replace('\n', '<br>')
        block['subtitle'] = {'text': wrapped, 'font': {'family': _PLOTLY_SANS, 'size': 12, 'color': INK_SOFT}}
    return block


def _plotly_headline_height(title: str | None, subtitle: str | None, width_px: int) -> int:
    """Top margin in px that fits the title block."""
    if not title:
        return 30
    lines = _n_lines(_wrap(subtitle, width_px / 96.0, 12, 3)) if subtitle else 0
    return 56 + (22 + 17 * lines if subtitle else 0)


def _plotly_axes_style() -> tuple[dict[str, object], dict[str, object]]:
    """Editorial axes: baseline rule on x, horizontal gridlines on y."""
    tick_font = {'family': _PLOTLY_SANS, 'color': INK_SOFT, 'size': 12}
    x: dict[str, object] = {'showgrid': False, 'showline': True, 'linecolor': INK, 'linewidth': 1, 'ticks': 'outside',
         'tickcolor': INK, 'tickfont': tick_font}
    y: dict[str, object] = {'showgrid': True, 'gridcolor': GRID, 'gridwidth': 1, 'showline': False, 'zeroline': True,
         'zerolinecolor': INK_SOFT, 'zerolinewidth': 1, 'ticks': '', 'tickfont': tick_font}
    return x, y


_PLOTLY_DAYS = ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')


def _plotly_rangebreaks(
    skip_days: object, skip_hours: object, days: bool = True, hours: bool = True,
) -> list[dict[str, object]]:
    """Plotly ``rangebreaks`` that close the gaps left by skipped weekdays and hours.

    Consecutive skipped days (or hours) become one break, wrapping past the end
    of the week (or day) when needed, for example Saturday and Sunday are
    ``bounds=['sat', 'mon']``.
    """
    breaks: list[dict[str, object]] = []
    if days:
        present = parse_skip_days(skip_days)
        for start in sorted(d for d in present if (d - 1) % 7 not in present):
            end = start
            while (end + 1) % 7 in present:
                end = (end + 1) % 7
            breaks.append({'bounds': [_PLOTLY_DAYS[start], _PLOTLY_DAYS[(end + 1) % 7]]})
    if hours:
        for first, last in _hour_runs(sorted(parse_skip_hours(skip_hours))):
            breaks.append({'pattern': 'hour', 'bounds': [first, (last + 1) % 24]})
    return breaks


def _axis_ref(row: int | None, kind: str = 'y') -> str:
    """Plotly axis id for a subplot row ('y', 'y2', ...)."""
    return kind if not row or row == 1 else f'{kind}{row}'


def _add_plotly_zones(
    fig: go.Figure, zones: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp] | None,
    row: int | None = None, labels: bool = True,
) -> None:
    """Tint the history and forecast zones, mark where the data ends and label the zones."""
    if zones is None:
        return
    x0, boundary, x1 = (z.to_pydatetime() for z in zones)
    target: dict[str, int] = {} if row is None else {'row': row, 'col': 1}
    fig.add_vrect(x0=x0, x1=boundary, fillcolor=HISTORY_TINT, opacity=1, layer='below', line_width=0, **target)
    fig.add_vrect(x0=boundary, x1=x1, fillcolor=FORECAST_TINT, opacity=1, layer='below', line_width=0, **target)
    fig.add_vline(x=boundary, line={'color': INK_SOFT, 'width': 1, 'dash': 'dash'}, layer='below', **target)
    if not labels:
        return
    for x, text, color, anchor in ((x0, 'HISTORY', OBSERVED_TEXT, 'left'), (x1, 'FORECAST', ACCENT, 'right')):
        fig.add_annotation(x=x, y=0, text=f'<b>{text}</b>', showarrow=False, xanchor=anchor, yanchor='bottom',
                           font={'family': _PLOTLY_SANS, 'size': 12, 'color': color}, **target)
        # Anchor to the bottom of the plotting area whatever the y data range is.
        fig.layout.annotations[-1].yref = f'{_axis_ref(row)} domain'


def plot_plotly(
    m: Prophet,
    fcst: pd.DataFrame,
    uncertainty: bool = True,
    plot_cap: bool = True,
    trend: bool = False,
    changepoints: bool = False,
    changepoints_threshold: float = 0.01,
    xlabel: str = 'ds',
    ylabel: str = 'y',
    figsize: tuple[int, int] = (900, 600),
    *,
    title: str | bool | None = None,
    subtitle: str | bool | None = None,
    annotate: bool = True,
    skip_days: _SkipSpec = None,
    skip_hours: _SkipSpec = None,
) -> go.Figure:
    """Plot the Prophet forecast with Plotly offline.

    Plotting in Jupyter Notebook requires initializing plotly.offline.init_notebook_mode():
    >>> import plotly.offline as py
    >>> py.init_notebook_mode()
    Then the figure can be displayed using plotly.offline.iplot(...):
    >>> fig = plot_plotly(m, fcst)
    >>> py.iplot(fig)
    see https://plot.ly/python/offline/ for details

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Optional boolean to plot uncertainty intervals.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    trend: Optional boolean to plot trend
    changepoints: Optional boolean to plot changepoints
    changepoints_threshold: Threshold on trend change magnitude for significance.
    xlabel: Optional label name on X-axis. The default 'ds' draws no label.
    ylabel: Optional label name on Y-axis. The default 'y' draws no label.
    figsize: The plot's size (in px).
    title: Headline. None writes one from the data, a string replaces it and
        False draws none (nor an automatic subtitle).
    subtitle: Line under the headline. None writes one from the data, a
        string replaces it and False omits it.
    annotate: Draw the History and Forecast zone labels, the "data ends"
        note, the end-of-line "Forecast" label and callouts on the largest
        significant trend changepoints.
    skip_days: Weekdays to leave out of the time axis (names, numbers with
        Monday as 0, or "weekends"), drawn with Plotly range breaks so Friday
        joins Monday.
    skip_hours: Hours of the day to leave out of the time axis.

    Returns
    -------
    A Plotly Figure.
    """
    breaks = _plotly_rangebreaks(skip_days, skip_hours)
    prediction_color = ACCENT
    error_color = _PLOTLY_BAND
    actual_color = OBSERVED
    cap_color = INK_SOFT
    trend_color = INK
    line_width = 2
    marker_size = 4

    assert m.history is not None
    show_band = bool(uncertainty and m.uncertainty_samples)
    title_text, sub_text = _headline_texts(title, subtitle, lambda: _forecast_headline(m, fcst, show_band))

    data = []
    # Add actual
    data.append(go.Scatter(
        name='Actual',
        x=m.history['ds'],
        y=m.history['y'],
        marker=dict(color=actual_color, size=marker_size, opacity=0.75),
        mode='markers'
    ))
    # Add lower bound
    if show_band:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip'
        ))
    # Add prediction
    data.append(go.Scatter(
        name='Predicted',
        x=fcst['ds'],
        y=fcst['yhat'],
        mode='lines',
        line=dict(color=prediction_color, width=line_width),
        fillcolor=error_color,
        fill='tonexty' if show_band else 'none'
    ))
    # Add upper bound
    if show_band:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            fillcolor=error_color,
            fill='tonexty',
            hoverinfo='skip'
        ))
    # Add caps
    if 'cap' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Cap',
            x=fcst['ds'],
            y=fcst['cap'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Floor',
            x=fcst['ds'],
            y=fcst['floor'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    # Add trend
    if trend:
        data.append(go.Scatter(
            name='Trend',
            x=fcst['ds'],
            y=fcst['trend'],
            mode='lines',
            line=dict(color=trend_color, width=line_width),
        ))
    # Add changepoints
    assert m.changepoints is not None
    if changepoints and len(m.changepoints) > 0:
        signif_changepoints = _significant_changepoints(m, changepoints_threshold)['ds']
        data.append(go.Scatter(
            x=signif_changepoints,
            y=fcst.loc[fcst['ds'].isin(signif_changepoints), 'trend'],
            marker=dict(size=50, symbol='line-ns-open', color=trend_color,
                        line=dict(width=line_width)),
            mode='markers',
            hoverinfo='skip'
        ))

    x_style, y_style = _plotly_axes_style()
    label_font = dict(family=_PLOTLY_SANS, color=INK_SOFT, size=12)
    layout = dict(
        showlegend=False,
        width=figsize[0],
        height=figsize[1],
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family=_PLOTLY_SANS, color=INK_SOFT),
        margin=dict(l=70, r=90 if annotate else 40, b=40,
                    t=_plotly_headline_height(title_text, sub_text, figsize[0])),
        yaxis=dict(**y_style, title=dict(text=ylabel if ylabel != 'y' else None, font=label_font)),
        xaxis=dict(
            **x_style,
            title=dict(text=xlabel if xlabel != 'ds' else None, font=label_font),
            type='date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7,
                         label='1w',
                         step='day',
                         stepmode='backward'),
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                         label='1y',
                         step='year',
                         stepmode='backward'),
                    dict(step='all')
                ]),
                bgcolor=HISTORY_TINT,
                activecolor=FORECAST_TINT,
                font=dict(family=_PLOTLY_SANS, color=INK_SOFT),
            ),
            rangeslider=dict(
                visible=True,
                bgcolor=HISTORY_TINT,
            ),
        ),
    )
    block = _plotly_title(title_text, sub_text, figsize[0])
    if block is not None:
        layout['title'] = block
    fig = go.Figure(data=data, layout=layout)
    if breaks:
        fig.update_xaxes(rangebreaks=breaks)

    zones = _forecast_zone_args(m, fcst)
    _add_plotly_zones(fig, zones, labels=annotate)
    if annotate:
        span_all = fcst['ds'].max() - fcst['ds'].min()
        if zones is not None:
            fig.add_annotation(x=zones[1], y=0, xref='x', yref='y domain', xanchor='right', yanchor='bottom',
                               xshift=-6, showarrow=False, text=f'data ends {_fmt_date(zones[1], span_all)}',
                               font=dict(family=_PLOTLY_SANS, size=11, color=INK_SOFT))
        last = fcst.iloc[-1]
        fig.add_annotation(x=last['ds'], y=last['yhat'], text='<b>Forecast</b>', showarrow=False,
                           xanchor='left', xshift=6, font=dict(family=_PLOTLY_SANS, size=12, color=ACCENT))
        for k, (_, row) in enumerate(_top_changepoints(m, fcst).iterrows()):
            y_at = float(np.interp(date2num(row['ds']), date2num(fcst['ds']), fcst['trend']))
            direction = 'upwards' if row['delta'] > 0 else 'downwards'
            fig.add_annotation(
                x=row['ds'], y=y_at, ax=0, ay=-(46 + 30 * k), showarrow=True, arrowwidth=1, arrowcolor=INK_SOFT,
                text=f'{_fmt_date(row["ds"], span_all)}: trend bends {direction}',
                bgcolor=PAPER, bordercolor=INK_SOFT, borderwidth=1, borderpad=3,
                font=dict(family=_PLOTLY_SANS, size=11, color=INK))
    return fig


def plot_components_plotly(
    m: Prophet,
    fcst: pd.DataFrame,
    uncertainty: bool = True,
    plot_cap: bool = True,
    figsize: tuple[int, int] = (900, 200),
    *,
    title: str | bool | None = None,
    panel_titles: bool = True,
    skip_days: _SkipSpec = None,
    skip_hours: _SkipSpec = None,
) -> go.Figure:
    """Plot the Prophet forecast components using Plotly.
    See plot_plotly() for Plotly setup instructions

    Will plot whichever are available of: trend, holidays, weekly
    seasonality, yearly seasonality, and additive and multiplicative extra
    regressors.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: Set the size for the subplots (in px).
    title: Figure headline. None writes a default, a string replaces it and
        False draws none.
    panel_titles: Give each panel a plain-English headline written from the
        data. If False, panels carry just their component name.
    skip_days, skip_hours: Weekdays and hours of the day to leave out of the
        time axes with Plotly range breaks. The weekly profile also drops the
        skipped days and the daily profile the skipped hours.

    Returns
    -------
    A Plotly Figure.
    """
    parse_skip_days(skip_days), parse_skip_hours(skip_hours)  # fail early on a bad specification

    # Identify components to plot and get their Plotly props
    components = {}
    components['trend'] = get_forecast_component_plotly_props(
        m, fcst, 'trend', uncertainty, plot_cap)
    if m.train_holiday_names is not None and 'holidays' in fcst:
        components['holidays'] = get_forecast_component_plotly_props(
            m, fcst, 'holidays', uncertainty)

    regressors = {'additive': False, 'multiplicative': False}
    for name, props in m.extra_regressors.items():
        regressors[props['mode']] = True
    for mode in ['additive', 'multiplicative']:
        if regressors[mode] and 'extra_regressors_{}'.format(mode) in fcst:
            components['extra_regressors_{}'.format(mode)] = get_forecast_component_plotly_props(
                m, fcst, 'extra_regressors_{}'.format(mode))
    for seasonality in m.seasonalities:
        components[seasonality] = get_seasonality_plotly_props(m, seasonality)

    n = len(components)
    sub_px = 15
    gap_px = (70 if panel_titles else 36) + 36
    total_h = figsize[1] * n + gap_px * (n - 1)
    spacing = min(gap_px / total_h, 0.95 / (n - 1)) if n > 1 else 0.0

    def panel_label(key: str) -> str:
        props = components[key]
        head = props.get('title') or _pretty(key) if panel_titles else _pretty(key)
        sub = props.get('subtitle') if panel_titles else ''
        if not sub:
            return f'<b>{head}</b>'
        wrapped = _wrap(sub, figsize[0] / 96.0, 11, 2).replace('\n', '<br>')
        return f'<b>{head}</b><br><span style="font-size:11px;color:{INK_SOFT}">{wrapped}</span>'

    head_text = _resolve_text(title, lambda: 'What shapes the forecast: ' + ', '.join(
        _pretty(c).lower() for c in components))
    top_px = 30 + (50 if head_text else 20) + (sub_px * 2 if panel_titles else 0)

    # Create Plotly subplot figure and add the components to it
    fig = make_subplots(rows=n, cols=1, print_grid=False,
                        subplot_titles=[panel_label(c) for c in components], vertical_spacing=spacing)
    # Panel headlines sit flush left above their panel, serif like the figure headline.
    for annotation in fig.layout.annotations:
        annotation.update(x=0, xref='paper', xanchor='left', align='left',
                          font={'family': _PLOTLY_SERIF, 'size': 14, 'color': INK})
    fig.update_layout(
        showlegend=False,
        width=figsize[0],
        height=total_h + top_px + 40,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family=_PLOTLY_SANS, color=INK_SOFT),
        margin=dict(l=70, r=40, t=top_px, b=40),
    )
    if head_text:
        fig.update_layout(title={
            'text': head_text, 'x': 0.0, 'xanchor': 'left', 'xref': 'paper',
            'font': {'family': _PLOTLY_SERIF, 'size': 20, 'color': INK}})
    x_style, y_style = _plotly_axes_style()
    zones = _forecast_zone_args(m, fcst)
    for i, name in enumerate(components):
        xaxis = fig.layout['xaxis' if i == 0 else 'xaxis{}'.format(i + 1)]
        yaxis = fig.layout['yaxis' if i == 0 else 'yaxis{}'.format(i + 1)]
        xaxis.update(components[name]['xaxis'])
        yaxis.update(components[name]['yaxis'])
        xaxis.update(**x_style)
        yaxis.update(**y_style)
        yaxis.update(title={'text': None})
        for trace in components[name]['traces']:
            fig.add_trace(trace, row=i + 1, col=1)
        panel_breaks = _panel_rangebreaks(m, name, skip_days, skip_hours)
        if panel_breaks:
            xaxis.update(rangebreaks=panel_breaks)
        if name in ('trend', 'holidays') or name.startswith('extra_regressors_'):
            _add_plotly_zones(fig, zones, row=i + 1, labels=(name == 'trend'))
    return fig


def _panel_rangebreaks(m: Prophet, name: str, skip_days: object, skip_hours: object) -> list[dict[str, object]]:
    """Range breaks for one component panel: timelines skip both, repeating cycles only what lines up."""
    if name in m.seasonalities:
        period = float(m.seasonalities[name]['period'])
        weekly = name == 'weekly' or period == 7
        daily = name == 'daily' or period == 1
        return _plotly_rangebreaks(skip_days, skip_hours, days=weekly, hours=weekly or daily)
    return _plotly_rangebreaks(skip_days, skip_hours)


def _component_figure(
    props: _PlotlyProps,
    title: str | bool | None,
    figsize: tuple[int, int],
    name: str,
    zones: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp] | None = None,
    breaks: list[dict[str, object]] | None = None,
) -> go.Figure:
    """A single-component Plotly figure in the editorial style."""
    title_text, sub_text = _headline_texts(
        title, None, lambda: (props.get('title') or _pretty(name), props.get('subtitle') or ''))
    x_style, y_style = _plotly_axes_style()
    xaxis = props['xaxis'].update(**x_style)
    yaxis = props['yaxis'].update(**y_style)
    yaxis.update(title={'text': None})
    layout = go.Layout(
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family=_PLOTLY_SANS, color=INK_SOFT),
        margin=dict(l=70, r=40, b=50, t=_plotly_headline_height(title_text, sub_text, figsize[0])),
        xaxis=xaxis,
        yaxis=yaxis,
    )
    block = _plotly_title(title_text, sub_text, figsize[0])
    if block is not None:
        layout.update(title=block)
    fig = go.Figure(data=props['traces'], layout=layout)
    if breaks:
        fig.update_xaxes(rangebreaks=breaks)
    _add_plotly_zones(fig, zones, labels=name == 'trend')
    return fig


def plot_forecast_component_plotly(
    m: Prophet,
    fcst: pd.DataFrame,
    name: str,
    uncertainty: bool = True,
    plot_cap: bool = False,
    figsize: tuple[int, int] = (900, 300),
    *,
    title: str | bool | None = None,
    skip_days: _SkipSpec = None,
    skip_hours: _SkipSpec = None,
) -> go.Figure:
    """Plot a particular component of the forecast using Plotly.
    See plot_plotly() for Plotly setup instructions

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    name: Name of the component to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: The plot's size (in px).
    title: Headline. None writes one from the data, a string replaces it and
        False draws none.
    skip_days, skip_hours: Weekdays and hours of the day to leave out of the
        time axis with Plotly range breaks.

    Returns
    -------
    A Plotly Figure.
    """
    props = get_forecast_component_plotly_props(m, fcst, name, uncertainty, plot_cap)
    return _component_figure(props, title, figsize, name, zones=_forecast_zone_args(m, fcst),
                             breaks=_plotly_rangebreaks(skip_days, skip_hours))


def plot_seasonality_plotly(
    m: Prophet,
    name: str,
    uncertainty: bool = True,
    figsize: tuple[int, int] = (900, 300),
    *,
    title: str | bool | None = None,
    skip_days: _SkipSpec = None,
    skip_hours: _SkipSpec = None,
) -> go.Figure:
    """Plot a custom seasonal component using Plotly.
    See plot_plotly() for Plotly setup instructions

    Parameters
    ----------
    m: Prophet model.
    name: Seasonality name, like 'daily', 'weekly'.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    figsize: Set the plot's size (in px).
    title: Headline. None writes one from the data, a string replaces it and
        False draws none.
    skip_days, skip_hours: Weekdays and hours of the day the data never has;
        applied when the seasonality repeats weekly or daily, as for
        :func:`plot_components_plotly`.

    Returns
    -------
    A Plotly Figure.
    """
    props = get_seasonality_plotly_props(m, name, uncertainty)
    return _component_figure(props, title, figsize, name,
                             breaks=_panel_rangebreaks(m, name, skip_days, skip_hours))


def get_forecast_component_plotly_props(
    m: Prophet,
    fcst: pd.DataFrame,
    name: str,
    uncertainty: bool = True,
    plot_cap: bool = False,
) -> _PlotlyProps:
    """Prepares a dictionary for plotting the selected forecast component with Plotly

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    name: Name of the component to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.

    Returns
    -------
    A dictionary with Plotly traces, xaxis and yaxis, plus a plain-English
    title and subtitle written from the data.
    """
    prediction_color = ACCENT
    error_color = _PLOTLY_BAND
    cap_color = INK_SOFT
    zeroline_color = INK_SOFT
    line_width = 2

    range_margin = (fcst['ds'].max() - fcst['ds'].min()) * 0.05
    range_x = [fcst['ds'].min() - range_margin, fcst['ds'].max() + range_margin]

    text = None
    mode = 'lines'
    if name == 'holidays':

        # Combine holidays into one hover text
        holidays = m.construct_holiday_dataframe(fcst['ds'])
        holiday_features, _, _ = m.make_holiday_features(fcst['ds'], holidays)
        holiday_features.columns = holiday_features.columns.str.replace('_delim_', '', regex=False)
        holiday_features.columns = holiday_features.columns.str.replace('+0', '', regex=False)
        text = pd.Series(data='', index=holiday_features.index)
        for holiday_feature, idxs in holiday_features.items():
            # https://github.com/facebook/pyrefly/issues/2248
            # pyrefly:ignore[unsupported-operation]
            text[idxs.astype(bool) & (text != '')] += '<br>'  # Add newline if additional holiday
            text[idxs.astype(bool)] += holiday_feature  # pyrefly:ignore[unsupported-operation]

    traces = []
    traces.append(go.Scatter(
        name=name,
        x=fcst['ds'],
        y=fcst[name],
        mode=mode,
        line=go.scatter.Line(color=prediction_color, width=line_width),
        text=text,
    ))
    if uncertainty and m.uncertainty_samples and (fcst[name + '_upper'] != fcst[name + '_lower']).any():
        if mode == 'markers':
            traces[0].update(
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=fcst[name + '_upper'],
                    arrayminus=fcst[name + '_lower'],
                    width=0,
                    color=error_color
                )
            )
        else:
            traces.append(go.Scatter(
                name=name + '_upper',
                x=fcst['ds'],
                y=fcst[name + '_upper'],
                mode=mode,
                line=go.scatter.Line(width=0, color=error_color)
            ))
            traces.append(go.Scatter(
                name=name + '_lower',
                x=fcst['ds'],
                y=fcst[name + '_lower'],
                mode=mode,
                line=go.scatter.Line(width=0, color=error_color),
                fillcolor=error_color,
                fill='tonexty'
            ))
    if 'cap' in fcst and plot_cap:
        traces.append(go.Scatter(
            name='Cap',
            x=fcst['ds'],
            y=fcst['cap'],
            mode='lines',
            line=go.scatter.Line(color=cap_color, dash='dash', width=line_width),
        ))
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        traces.append(go.Scatter(
            name='Floor',
            x=fcst['ds'],
            y=fcst['floor'],
            mode='lines',
            line=go.scatter.Line(color=cap_color, dash='dash', width=line_width),
        ))

    xaxis = go.layout.XAxis(
        type='date',
        range=range_x)
    yaxis = go.layout.YAxis(rangemode='normal' if name == 'trend' else 'tozero',
                            title=go.layout.yaxis.Title(text=name),
                            zerolinecolor=zeroline_color)
    assert m.component_modes
    if name in m.component_modes['multiplicative']:
        yaxis.update(tickformat=_PERCENT_TICKS, hoverformat='.2%')

    if name == 'trend':
        headline = _trend_headline(m, fcst)
    elif name == 'holidays':
        headline = _holiday_headline(m, fcst)
    elif name.startswith('extra_regressors_'):
        headline = _regressor_headline(m, name, fcst[name])
    else:
        headline = (_pretty(name), '')
    return {'traces': traces, 'xaxis': xaxis, 'yaxis': yaxis, 'title': headline[0], 'subtitle': headline[1]}


def get_seasonality_plotly_props(
    m: Prophet,
    name: str,
    uncertainty: bool = True,
) -> _PlotlyProps:
    """Prepares a dictionary for plotting the selected seasonality with Plotly

    Parameters
    ----------
    m: Prophet model.
    name: Name of the component to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.

    Returns
    -------
    A dictionary with Plotly traces, xaxis and yaxis, plus a plain-English
    title and subtitle written from the data.
    """
    prediction_color = ACCENT
    error_color = _PLOTLY_BAND
    line_width = 2
    zeroline_color = INK_SOFT

    # Compute seasonality from Jan 1 through a single period.
    start = pd.to_datetime('2017-01-01 0000')
    period = m.seasonalities[name]['period']
    end = start + pd.Timedelta(days=period)
    assert m.history is not None
    if (m.history['ds'].dt.hour == 0).all():  # Day Precision
        plot_points = np.floor(period).astype(int)
    elif (m.history['ds'].dt.minute == 0).all():  # Hour Precision
        plot_points = np.floor(period * 24).astype(int)
    else:  # Minute Precision
        plot_points = np.floor(period * 24 * 60).astype(int)
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points, endpoint=False))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)

    traces = []
    traces.append(go.Scatter(
        name=name,
        x=df_y['ds'],
        y=seas[name],
        mode='lines',
        line=go.scatter.Line(color=prediction_color, width=line_width)
    ))
    if uncertainty and m.uncertainty_samples and (seas[name + '_upper'] != seas[name + '_lower']).any():
        traces.append(go.Scatter(
            name=name + '_upper',
            x=df_y['ds'],
            y=seas[name + '_upper'],
            mode='lines',
            line=go.scatter.Line(width=0, color=error_color)
        ))
        traces.append(go.Scatter(
            name=name + '_lower',
            x=df_y['ds'],
            y=seas[name + '_lower'],
            mode='lines',
            line=go.scatter.Line(width=0, color=error_color),
            fillcolor=error_color,
            fill='tonexty'
        ))

    # Set tick formats (examples are based on 2017-01-06 21:15)
    if period <= 2:
        tickformat = '%H:%M'  # "21:15"
    elif period < 7:
        tickformat = '%A %H:%M'  # "Friday 21:15"
    elif period < 14:
        tickformat = '%A'  # "Friday"
    else:
        tickformat = '%B %e'  # "January  6"

    range_margin = (df_y['ds'].max() - df_y['ds'].min()) * 0.05
    xaxis = go.layout.XAxis(
        tickformat=tickformat,
        type='date',
        range=[df_y['ds'].min() - range_margin, df_y['ds'].max() + range_margin]
    )

    yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(text=name),
                            zerolinecolor=zeroline_color)
    if m.seasonalities[name]['mode'] == 'multiplicative':
        yaxis.update(tickformat=_PERCENT_TICKS, hoverformat='.2%')

    if name == 'yearly' or period == 365.25:
        kind = 'yearly'
    elif (name == 'weekly' or period == 7) and (m.history['ds'].dt.hour == 0).all():
        kind = 'weekly'
    else:
        kind = 'other'
    headline = _seasonality_headline(m, name, df_y['ds'], seas[name], kind)
    return {'traces': traces, 'xaxis': xaxis, 'yaxis': yaxis, 'title': headline[0], 'subtitle': headline[1]}
