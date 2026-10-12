"""Dashboard section "Holidays": the effect of each holiday and the holidays inside the forecast window.

The model gives every holiday one 0/1 column per day of its window (``Christmas
Day_delim_-1``, ``Christmas Day_delim_+0``, ...) and one fitted coefficient per
column. Everything here is read from those columns: how often each holiday
occurs in the history, what a day of it adds, and which days ahead it touches.
Nothing is refitted.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from prophet.dashboard import _style
from prophet.dashboard._context import DashboardContext, Section
from prophet.dashboard._palette import DIVERGING, colorscale
from prophet.plot import ACCENT, INK_SOFT, PAPER
from prophet.visuals import (
    _day_month,
    _fmt_number,
    _fmt_pct,
    _plural,
    _signed,
    _y_scale,
)

if TYPE_CHECKING:
    from prophet.forecaster import Prophet

KEY = "holidays"
TITLE = "Holidays"

log = logging.getLogger("prophet.dashboard")

#: The two poles of the diverging ramp: holidays that raise y and holidays that lower it.
RAISES = DIVERGING[-1][1]
LOWERS = DIVERGING[0][1]
#: Most holidays drawn in a chart, and most rows listed in a table.
MAX_BARS = 20
MAX_LISTED = 120
#: Pixels kept clear between the category labels and the axis, so that the value
#: label of a long negative bar never runs into the name of its holiday.
LABEL_GAP = 44
#: Holiday effects belong to whole days, so hover dates never carry a time of day.
_DAY = "%d %b %Y"
_COLUMN = re.compile(r"^(?P<name>.*)_delim_(?P<offset>[+-]\d+)$", re.DOTALL)
_NOT_SEEN = "not seen"


# ----------------------------------------------------------------------------
# Pure parts: parsing and wording
# ----------------------------------------------------------------------------


def parse_column(column: str) -> tuple[str, int] | None:
    """``'Christmas Day_delim_-1'`` is ``('Christmas Day', -1)``; other columns are None."""
    match = _COLUMN.match(column)
    return None if match is None else (match["name"], int(match["offset"]))


def offset_phrase(offset: int) -> str:
    """A day of a holiday window in words: ``the day itself``, ``1 day before``."""
    if offset == 0:
        return "the day itself"
    return f"{_plural(abs(offset), 'day')} {'before' if offset < 0 else 'after'}"


def window_phrase(window: tuple[int, int]) -> str:
    """The configured window of a holiday: ``Single day``, ``Day plus 1 day before and 2 days after``."""
    before, after = -window[0], window[1]
    parts = [
        *([f"{_plural(before, 'day')} before"] if before else []),
        *([f"{_plural(after, 'day')} after"] if after else []),
    ]
    return "Single day" if not parts else "Day plus " + " and ".join(parts)


def effect_phrase(value: float, multiplicative: bool) -> str:
    """What an effect does to y: ``adds 41 to y``, ``takes 18 off y``, ``raises y by 12% of the trend level``."""
    if multiplicative:
        verb = "lowers" if value < 0 else "raises"
        return f"{verb} y by {_fmt_pct(value)} of the trend level"
    if value < 0:
        return f"takes {_fmt_number(abs(value))} off y"
    return f"adds {_fmt_number(value)} to y"


def is_negligible(value: float, multiplicative: bool, scale: float) -> bool:
    """Too small to matter: under half a percent of the trend, or of the typical size of y."""
    return abs(value) < (0.005 if multiplicative else 0.005 * max(scale, 1e-12))


def pct_tickformat(largest: float) -> str:
    """A d3 format for fractions as percentages, with decimals to suit effects of this size.

    Plotly spaces ticks in steps of 1, 2 or 5, at least a tenth of the range
    apart, so the decimals follow the size of the largest effect. The ``~``
    drops trailing zeros: ticks read ``0.2%`` and ``10%``, never ``0.20%``, and
    effects of a fraction of a percent never round to the same label.
    """
    if not np.isfinite(largest) or largest <= 0:
        return ".1~%"
    decimals = int(np.clip(1 - np.floor(np.log10(22 * largest)), 1, 6))
    return f".{decimals}~%"


def bar_range(values: Sequence[float]) -> list[float]:
    """An x range for the bar chart with a little air beyond the longest bars.

    The value labels sit outside the bars, in the right margin and in the gap
    the y axis leaves (see :data:`LABEL_GAP`), so the range needs no room for them.
    """
    low, high = min([0.0, *values]), max([0.0, *values])
    if high == low:
        return [-1.0, 1.0]
    pad = 0.05 * (high - low)
    return [low - (pad if low < 0 else 0.0), high + (pad if high > 0 else 0.0)]


def _weekday_date(ts: pd.Timestamp) -> str:
    """'Mon 26 Dec 2016': the day without a leading zero."""
    return f"{ts:%a} {_day_month(ts)} {ts:%Y}"


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def _escape(text: str) -> str:
    """Escape text for a plotly hover label, which renders a small subset of HTML."""
    return html.escape(text, quote=False)


# ----------------------------------------------------------------------------
# Pure parts: the effect of each holiday
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowDay:
    """One day of a holiday window as seen in the history."""

    offset: int
    effect: float
    #: Calendar days of the history on which this day of the window falls.
    days: int


@dataclass(frozen=True)
class HolidayEffect:
    """One holiday, as measured on the history.

    ``effect`` is the mean contribution over the days the holiday applies, in
    units of y (additive) or as a fraction of the trend (multiplicative). It is
    None when no day of the holiday falls in the history: the model then has
    only its prior to go on and there is nothing honest to report.
    """

    name: str
    effect: float | None
    #: Times the holiday occurs in the history (any day of its window counts).
    #: When the dates are not known this is the most days any one window day
    #: was seen, which can undercount.
    occurrences: int
    #: Calendar days of the history on which it applies.
    days: int
    #: Lowest and highest offset of the configured window; (0, 0) is a single day.
    window: tuple[int, int]
    #: The days of the window that occur in the history, in date order.
    profile: tuple[WindowDay, ...]

    @property
    def multi_day(self) -> bool:
        return self.window != (0, 0)

    @property
    def peak(self) -> WindowDay | None:
        """The day of the window with the largest effect, once there are days to compare."""
        if len(self.profile) < 2:
            return None
        return max(self.profile, key=lambda day: abs(day.effect))


def _effect_value(effect: HolidayEffect) -> float:
    return 0.0 if effect.effect is None else effect.effect


def window_bounds(row: Any) -> tuple[int, int]:
    """The window of one row of a holidays frame, read as the model reads it (no window is 0, 0)."""
    try:
        return int(getattr(row, "lower_window", 0)), int(
            getattr(row, "upper_window", 0)
        )
    except ValueError:
        return 0, 0


def occurrence_counts(holidays: pd.DataFrame, dates: pd.Series) -> dict[str, int]:
    """How often each holiday occurs in ``dates``: its dated rows with a day of their window in range.

    The 0/1 columns cannot say this. A Saturday holiday on weekday-only data
    leaves only its Friday, a Sunday one only its Monday, and the columns then
    show one day of each where the holiday occurred twice.
    """
    if holidays.empty or not {"holiday", "ds"} <= set(holidays.columns):
        return {}
    present = set(pd.DatetimeIndex(dates).normalize())
    counted: set[tuple[str, pd.Timestamp]] = set()
    counts: dict[str, int] = {}
    for row in holidays.itertuples():
        if pd.isna(row.ds):
            continue
        date = cast("pd.Timestamp", row.ds).normalize()
        name = str(row.holiday)
        if (name, date) in counted:
            continue
        counted.add((name, date))
        before, after = window_bounds(row)
        if any(
            date + pd.Timedelta(days=k) in present for k in range(before, after + 1)
        ):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _one_effect(
    name: str,
    columns: Sequence[tuple[str, int]],
    daily: pd.DataFrame,
    coefficients: Mapping[str, float],
    dated: int = 0,
) -> HolidayEffect:
    if not columns:
        return HolidayEffect(name, None, 0, 0, (0, 0), ())
    labels = [column for column, _ in columns]
    offsets = [offset for _, offset in columns]
    on = daily[labels].to_numpy(dtype=float) > 0
    coef = np.array([coefficients.get(column, 0.0) for column in labels])
    per_column = on.sum(axis=0)
    applies = on.any(axis=1)
    days = int(applies.sum())
    contribution = (on * coef).sum(axis=1)
    profile = tuple(
        sorted(
            (
                WindowDay(offset, float(value), int(count))
                for offset, value, count in zip(offsets, coef, per_column)
                if count > 0
            ),
            key=lambda day: day.offset,
        )
    )
    return HolidayEffect(
        name=name,
        effect=float(contribution[applies].mean()) if days else None,
        occurrences=max(int(per_column.max()), dated),
        days=days,
        window=(min(offsets), max(offsets)),
        profile=profile,
    )


def holiday_effects(
    dates: pd.Series,
    indicators: pd.DataFrame,
    coefficients: Mapping[str, float],
    names: Sequence[str],
    occurrences: Mapping[str, int] | None = None,
) -> tuple[HolidayEffect, ...]:
    """The mean contribution of each holiday on the days it applies in the history.

    ``indicators`` holds the model's 0/1 holiday columns for the rows of
    ``dates``; ``coefficients`` maps each column to what one active day of it
    contributes. Rows are folded to calendar days first, so sub-daily data
    counts a holiday once, not once per observation. A name with no column, or
    with columns that are never on, gets ``effect=None``. ``occurrences`` is
    :func:`occurrence_counts` of the holiday dates; without it the count is
    read from the columns.
    """
    days = pd.DatetimeIndex(dates).normalize()
    daily = pd.DataFrame(
        indicators.to_numpy(dtype=float), index=days, columns=indicators.columns
    )
    daily = daily.groupby(level=0).max()
    by_name: dict[str, list[tuple[str, int]]] = {}
    for column in daily.columns:
        parsed = parse_column(str(column))
        if parsed is not None:
            by_name.setdefault(parsed[0], []).append((str(column), parsed[1]))
    dated = occurrences or {}
    return tuple(
        _one_effect(
            name, by_name.get(name, []), daily, coefficients, dated.get(name, 0)
        )
        for name in names
    )


def chart_rows(
    effects: Sequence[HolidayEffect], limit: int = MAX_BARS
) -> tuple[tuple[HolidayEffect, ...], int]:
    """The holidays to chart, smallest effect first, and how many seen ones were left out.

    The ``limit`` largest by absolute effect are kept, so a country calendar
    with dozens of holidays still gives a readable chart. Holidays that never
    occur in the history are not charted.
    """
    seen = [e for e in effects if e.effect is not None]
    largest = sorted(seen, key=lambda e: (-abs(_effect_value(e)), e.name))[:limit]
    return (
        tuple(sorted(largest, key=lambda e: (_effect_value(e), e.name))),
        len(seen) - len(largest),
    )


def leaders(
    effects: Sequence[HolidayEffect], multiplicative: bool, scale: float
) -> tuple[HolidayEffect | None, HolidayEffect | None]:
    """The biggest raising and the biggest lowering holiday, each None when absent or negligible."""
    seen = [e for e in effects if e.effect is not None]
    top = max(seen, key=lambda e: (_effect_value(e), e.name), default=None)
    bottom = min(seen, key=lambda e: (_effect_value(e), e.name), default=None)

    def matters(effect: HolidayEffect | None, sign: int) -> HolidayEffect | None:
        if effect is None or _effect_value(effect) * sign <= 0:
            return None
        return (
            None
            if is_negligible(_effect_value(effect), multiplicative, scale)
            else effect
        )

    return matters(top, 1), matters(bottom, -1)


# ----------------------------------------------------------------------------
# Pure parts: holiday days ahead and through time
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class HolidayDay:
    """A holiday touching one calendar day of the forecast frame."""

    date: pd.Timestamp
    name: str
    #: Days from the holiday itself; negative is before it.
    offset: int
    #: The coefficient of that day of the window, or None when the history never
    #: saw it and the model has nothing to base an effect on.
    effect: float | None

    @property
    def holiday_date(self) -> pd.Timestamp:
        return self.date - pd.Timedelta(days=self.offset)


def holiday_days(
    dates: pd.Series,
    indicators: pd.DataFrame,
    coefficients: Mapping[str, float],
    effects: Sequence[HolidayEffect],
) -> tuple[HolidayDay, ...]:
    """Every calendar day in ``dates`` that a holiday touches, in date order.

    One entry per day, holiday and window day, however many rows the day has.
    """
    parsed = [parse_column(str(c)) for c in indicators.columns]
    seen = {(e.name, d.offset) for e in effects for d in e.profile}
    days = pd.DatetimeIndex(dates).normalize()
    rows, cols = np.nonzero(indicators.to_numpy(dtype=float) > 0)
    hits = {(days[r], c) for r, c in zip(rows.tolist(), cols.tolist())}
    found = [
        HolidayDay(
            date=date,
            name=p[0],
            offset=p[1],
            effect=coefficients.get(str(indicators.columns[c]), 0.0)
            if (p[0], p[1]) in seen
            else None,
        )
        for date, c in hits
        if (p := parsed[c]) is not None
    ]
    return tuple(sorted(found, key=lambda d: (d.date, d.name, d.offset)))


@dataclass(frozen=True)
class ForecastSpan:
    """The calendar days that carry forecast rows, first to last.

    A day counts as soon as one of its rows lies after the history, so when
    sub-daily data ends part-way through a holiday the rest of that day is
    ahead. Days are not compared with the timestamp the history ends on.
    """

    first: pd.Timestamp
    last: pd.Timestamp

    def contains(self, date: pd.Timestamp) -> bool:
        return self.first <= date <= self.last


def forecast_span(future_ds: pd.Series) -> ForecastSpan | None:
    """The span of the forecast rows after the history, or None when there are none."""
    if len(future_ds) == 0:
        return None
    days = pd.DatetimeIndex(future_ds).normalize()
    return ForecastSpan(pd.Timestamp(days.min()), pd.Timestamp(days.max()))


@dataclass(frozen=True)
class Ahead:
    """The holidays that touch the forecast window."""

    #: Distinct holidays dated inside the window.
    count: int
    first_name: str | None
    first_date: pd.Timestamp | None
    #: Holidays dated outside the window, by name in date order, whose window
    #: still reaches a day of it: the tail of one that ended just before the
    #: forecast starts, or the run-up to one just after it ends.
    reaching: tuple[str, ...] = ()


def ahead_summary(
    days: Sequence[HolidayDay], span: ForecastSpan | None
) -> Ahead | None:
    """How many holidays fall in the forecast window, or None when there is no window."""
    if span is None:
        return None
    touching = sorted({(d.holiday_date, d.name) for d in days if span.contains(d.date)})
    inside = [(date, name) for date, name in touching if span.contains(date)]
    reaching = tuple(name for date, name in touching if not span.contains(date))
    if not inside:
        return Ahead(0, None, None, reaching)
    return Ahead(len(inside), inside[0][1], inside[0][0], reaching)


@dataclass(frozen=True)
class NextDate:
    """When a holiday next touches the forecast window."""

    date: pd.Timestamp
    #: False when the holiday is dated outside the window and only a day of its window reaches in.
    inside: bool


def next_dates(
    days: Sequence[HolidayDay], span: ForecastSpan | None
) -> dict[str, NextDate]:
    """The next date of each holiday that touches the forecast window, one dated inside it first."""
    out: dict[str, NextDate] = {}
    if span is None:
        return out
    for d in sorted(days, key=lambda d: d.holiday_date):
        if not span.contains(d.date):
            continue
        inside = span.contains(d.holiday_date)
        current = out.get(d.name)
        if current is None or (inside and not current.inside):
            out[d.name] = NextDate(d.holiday_date, inside)
    return out


def day_line(day: HolidayDay, multiplicative: bool) -> str:
    """One hover line: ``Christmas Day (1 day before): +3``."""
    when = "" if day.offset == 0 else f" ({offset_phrase(day.offset)})"
    value = _NOT_SEEN if day.effect is None else _signed(day.effect, multiplicative)
    return f"{_escape(day.name)}{when}: {value}"


def day_labels(
    days: Sequence[HolidayDay], multiplicative: bool
) -> dict[pd.Timestamp, str]:
    """Hover text by calendar day, each line led by a line break."""
    grouped: dict[pd.Timestamp, list[str]] = {}
    for d in days:
        grouped.setdefault(d.date, []).append(day_line(d, multiplicative))
    return {
        date: "".join(f"<br>{line}" for line in lines)
        for date, lines in grouped.items()
    }


def spike_series(
    ds: pd.Series, total: np.ndarray, labels: Sequence[str]
) -> pd.DataFrame:
    """The rows the line needs: for each day worth drawing, its first and its last row.

    A day is drawn when it carries an effect or sits next to one, which keeps
    each spike's shape, and the first and last days are drawn so the line spans
    the chart. Everything else is a flat stretch at zero. A holiday effect is
    the same on every row of its day, so the first and last row of a day carry
    the same value and the line stays flat through the day, even when the data
    is sub-daily. With one row a day the two are the same row.
    """
    frame = pd.DataFrame(
        {
            "ds": ds.to_numpy(),
            "total": np.asarray(total, dtype=float),
            "label": list(labels),
        }
    )
    frame["day"] = frame["ds"].dt.normalize()
    first = ~frame.duplicated("day", keep="first")
    last = ~frame.duplicated("day", keep="last")
    daily = frame[first].reset_index(drop=True)
    if daily.empty:
        return daily[["ds", "total", "label"]]
    on = daily["total"].to_numpy() != 0
    keep = on | np.r_[on[1:], False] | np.r_[False, on[:-1]]
    keep[[0, -1]] = True
    kept_days = daily["day"].to_numpy()[keep]
    rows = frame[(first | last) & frame["day"].isin(kept_days)]
    return rows[["ds", "total", "label"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Pure parts: sentences
# ----------------------------------------------------------------------------


def _peak_tail(effect: HolidayEffect, multiplicative: bool) -> str:
    peak = effect.peak
    if not effect.multi_day or peak is None:
        return ""
    return f" Its biggest day is {offset_phrase(peak.offset)}, at {_signed(peak.effect, multiplicative)}."


def _once_tail(effect: HolidayEffect) -> str:
    """A holiday seen once rests on a single day of data, so its size is a rough guide."""
    if effect.occurrences != 1:
        return ""
    return " It occurs only once in the history, so its size is a rough guide."


def _size_with_basis(effect: HolidayEffect, multiplicative: bool) -> str:
    """'+1.45', or '-3.58, seen once' when the size rests on a single occurrence."""
    size = _signed(_effect_value(effect), multiplicative)
    return f"{size}, seen once" if effect.occurrences == 1 else size


def reaching_sentence(reaching: Sequence[str]) -> str:
    """Holidays dated outside the forecast window whose own window reaches a day of it."""
    if not reaching:
        return ""
    if len(reaching) == 1:
        return f"{reaching[0]} is dated outside it, but its window reaches in."
    return f"The windows of {len(reaching)} holidays dated outside it reach in."


def ahead_sentence(ahead: Ahead | None) -> str:
    """How many holidays fall in the forecast window, and the first of them."""
    if ahead is None:
        return ""
    if ahead.count == 0 or ahead.first_date is None or ahead.first_name is None:
        head = "No holiday falls inside the forecast window."
    else:
        day = f"{_day_month(ahead.first_date)} {ahead.first_date:%Y}"
        head = (
            f"One holiday falls inside the forecast window: {ahead.first_name} on {day}."
            if ahead.count == 1
            else f"{ahead.count} holidays fall inside the forecast window, the first being {ahead.first_name} on {day}."
        )
    return " ".join(part for part in (head, reaching_sentence(ahead.reaching)) if part)


def holiday_lede(
    effects: Sequence[HolidayEffect],
    multiplicative: bool,
    scale: float,
    ahead: Ahead | None,
) -> str:
    """The plain-English opening of the section."""
    total = len(effects)
    seen = [e for e in effects if e.effect is not None]
    unseen = total - len(seen)
    if not seen:
        opening = (
            f"The model includes {_plural(total, 'holiday')}, but "
            f"{'it does not fall' if total == 1 else 'none of them falls'} on a date in the history, "
            "so there was nothing to measure."
        )
        return " ".join(s for s in (opening, ahead_sentence(ahead)) if s)
    parts = [f"The model includes {_plural(total, 'holiday')}."]
    if unseen:
        parts.append(
            f"{_plural(unseen, 'holiday')} never {'falls' if unseen == 1 else 'fall'} on a date in the "
            f"history, so {'it' if unseen == 1 else 'they'} could not be measured."
        )
    top, bottom = leaders(effects, multiplicative, scale)
    if top is None and bottom is None:
        biggest = max(seen, key=lambda e: (abs(_effect_value(e)), e.name))
        parts.append(
            f"Holidays barely move this forecast: the largest effect, {biggest.name}, "
            f"{effect_phrase(_effect_value(biggest), multiplicative)} on average."
        )
    if top is not None:
        parts.append(
            f"{top.name} lifts y the most: on the days it applies, it "
            f"{effect_phrase(_effect_value(top), multiplicative)} on average.{_peak_tail(top, multiplicative)}{_once_tail(top)}"
        )
    if bottom is not None:
        parts.append(
            f"{bottom.name} pulls y down the most: on the days it applies, it "
            f"{effect_phrase(_effect_value(bottom), multiplicative)} on average.{_peak_tail(bottom, multiplicative)}{_once_tail(bottom)}"
        )
    parts.append(ahead_sentence(ahead))
    return " ".join(p for p in parts if p)


def holiday_summary(
    effects: Sequence[HolidayEffect],
    multiplicative: bool,
    scale: float,
    ahead: Ahead | None,
) -> str:
    """One plain-text sentence for the key findings at the top of the page."""
    total = len(effects)
    seen = [e for e in effects if e.effect is not None]
    if not seen:
        return (
            "The holiday does not fall on a date in the history, so it could not be measured."
            if total == 1
            else f"None of the {total} holidays falls on a date in the history, so none could be measured."
        )
    top, bottom = leaders(effects, multiplicative, scale)
    bits = [
        *(
            [f"{top.name} raises y the most ({_size_with_basis(top, multiplicative)})"]
            if top
            else []
        ),
        *(
            [
                f"{bottom.name} lowers {'it' if top else 'y'} the most ({_size_with_basis(bottom, multiplicative)})"
            ]
            if bottom
            else []
        ),
    ]
    if not bits:
        biggest = max(seen, key=lambda e: (abs(_effect_value(e)), e.name))
        bits = [
            f"Holidays barely move the forecast (largest: {biggest.name}, {_signed(_effect_value(biggest), multiplicative)})"
        ]
    head = " and ".join(bits)
    if ahead is None:
        return f"{head}."
    tail = (
        "no holiday falls in the forecast window"
        if ahead.count == 0
        else f"{ahead.count} {'holiday falls' if ahead.count == 1 else 'holidays fall'} in the forecast window"
    )
    if ahead.reaching:
        n = len(ahead.reaching)
        tail += (
            f"{', though ' if ahead.count == 0 else ', and '}{_plural(n, 'holiday')} dated outside it "
            f"{'has a window that reaches' if n == 1 else 'have windows that reach'} in"
        )
    return f"{head}; {tail}."


def peak_cell(effect: HolidayEffect) -> str:
    """The 'biggest day' cell of the table: only holidays with a window have one."""
    peak = effect.peak
    if not effect.multi_day or peak is None:
        return "-"
    return _capitalise(offset_phrase(peak.offset))


def bar_hover(effect: HolidayEffect, multiplicative: bool) -> str:
    """The lines of a bar's hover label after the average effect, escaped."""
    seen = (
        "once in the history"
        if effect.occurrences == 1
        else f"{effect.occurrences} times in the history"
    )
    lines = [f"Seen {seen}"]
    if effect.multi_day:
        lines.append(f"Window: {window_phrase(effect.window)}")
        shown = effect.profile[:8]
        lines += [
            f"{_capitalise(offset_phrase(day.offset))}: {_signed(day.effect, multiplicative)}"
            for day in shown
        ]
        if len(effect.profile) > len(shown):
            lines.append(f"and {len(effect.profile) - len(shown)} more days")
        if effect.peak is not None:
            lines.append(f"Biggest: {offset_phrase(effect.peak.offset)}")
    return "<br>".join(lines)


# ----------------------------------------------------------------------------
# Reading the model
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInputs:
    """What the model knows about its holidays."""

    names: tuple[str, ...]
    multiplicative: bool
    #: Contribution of one active day of each holiday column, in the units the
    #: forecast uses (y units when additive, a fraction of the trend otherwise).
    coefficients: Mapping[str, float]
    #: The 0/1 holiday columns for the rows of the training history.
    history: pd.DataFrame
    #: Times each holiday occurs in the training history, from its dates.
    occurrences: Mapping[str, int]


def read_model(ctx: DashboardContext) -> ModelInputs | None:
    """The holiday columns and coefficients of the fitted model, or None without holidays."""
    m = ctx.m
    if m.train_holiday_names is None or len(m.train_holiday_names) == 0:
        return None
    if m.holidays is None and m.country_holidays is None:
        log.debug(
            "The model lists holiday names but holds no holidays to rebuild them from."
        )
        return None
    names = tuple(dict.fromkeys(str(n) for n in m.train_holiday_names))
    features = m.make_all_seasonality_features(ctx.history)[0]
    beta = np.asarray(m.params["beta"], dtype=float)
    if beta.ndim != 2 or beta.shape[1] != features.shape[1]:
        raise ValueError(
            f"The model has {beta.shape[-1]} coefficients for {features.shape[1]} regressor columns."
        )
    additive = set((m.component_modes or {}).get("additive", ()))
    wanted = set(names)
    mean_beta = beta.mean(axis=0)
    positions: list[int] = []
    coefficients: dict[str, float] = {}
    for i, column in enumerate(features.columns):
        parsed = parse_column(str(column))
        if parsed is not None and parsed[0] in wanted:
            positions.append(i)
            coefficients[str(column)] = float(
                mean_beta[i] * (m.y_scale if parsed[0] in additive else 1.0)
            )
    return ModelInputs(
        names=names,
        multiplicative=m.holidays_mode == "multiplicative",
        coefficients=coefficients,
        history=features.iloc[:, positions].reset_index(drop=True),
        occurrences=occurrence_counts(
            m.construct_holiday_dataframe(ctx.history["ds"]), ctx.history["ds"]
        ),
    )


def holiday_indicators(m: Prophet, dates: pd.Series) -> pd.DataFrame:
    """The 0/1 holiday columns the model would build for ``dates`` (only ``ds`` is needed)."""
    frame = m.construct_holiday_dataframe(dates)
    if len(frame) == 0:
        return pd.DataFrame(index=range(len(dates)))
    features, _, _ = m.make_holiday_features(dates, frame)
    return features


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def effect_figure(
    ctx: DashboardContext, rows: Sequence[HolidayEffect], multiplicative: bool
) -> Any:
    """Horizontal bars of the average effect per holiday, biggest raiser on top."""
    go = ctx.go
    x_style, y_style = _style.axes()
    names = [_escape(e.name) for e in rows]
    fig = go.Figure()
    groups = (
        ("Raises y", RAISES, [e for e in rows if _effect_value(e) >= 0]),
        ("Lowers y", LOWERS, [e for e in rows if _effect_value(e) < 0]),
    )
    value = "%{x:+.2%}" if multiplicative else "%{x:+,.4g}"
    for label, colour, part in groups:
        if not part:
            continue
        fig.add_trace(
            go.Bar(
                name=label,
                orientation="h",
                x=[_effect_value(e) for e in part],
                y=[_escape(e.name) for e in part],
                marker={"color": colour},
                customdata=[
                    f"<b>{_escape(e.name)}</b><br>{bar_hover(e, multiplicative)}"
                    for e in part
                ],
                text=[_signed(_effect_value(e), multiplicative) for e in part],
                textposition="outside",
                textfont={"color": INK_SOFT, "size": 11},
                cliponaxis=False,
                hovertemplate=f"%{{customdata}}<br>Average effect <b>{value}</b><extra></extra>",
            )
        )
    values = [_effect_value(e) for e in rows]
    largest = max((abs(v) for v in values), default=0.0)
    # Ticks of a set length push the category labels away from the axis. They are
    # paper-coloured and drawn under the traces, so they stay invisible and the
    # value label of a bar that reaches into the gap is drawn over them.
    gap = {
        "ticks": "outside",
        "ticklen": LABEL_GAP,
        "tickcolor": PAPER,
        "layer": "below traces",
    }
    fig.update_layout(
        **_style.layout(
            max(220, 30 * len(rows) + 90),
            margin={"l": 64, "r": 64, "t": 36, "b": 40},
            barmode="overlay",
            bargap=0.35,
            showlegend=len(fig.data) > 1,
            xaxis={
                **y_style,
                "zeroline": True,
                "zerolinecolor": INK_SOFT,
                "zerolinewidth": 1,
                "range": bar_range(values),
                **({"tickformat": pct_tickformat(largest)} if multiplicative else {}),
            },
            yaxis={
                **x_style,
                "type": "category",
                "categoryorder": "array",
                "categoryarray": names,
                "automargin": True,
                # The zero line is the baseline; a rule at the left edge would cut
                # through the value label of the longest negative bar.
                "showline": False,
                **(gap if any(v < 0 for v in values) else {"ticks": ""}),
            },
        )
    )
    return fig


def profile_figure(
    ctx: DashboardContext, rows: Sequence[HolidayEffect], multiplicative: bool
) -> Any:
    """A heatmap of the effect on each day of the window, one row per holiday with a window."""
    go = ctx.go
    x_style, _ = _style.axes()
    offsets = [d.offset for e in rows for d in e.profile]
    axis = list(range(min(offsets), max(offsets) + 1))
    z = [
        [{d.offset: d.effect for d in e.profile}.get(offset) for offset in axis]
        for e in rows
    ]
    reach = max((abs(v) for row in z for v in row if v is not None), default=1.0) or 1.0
    cells = [
        ["" if v is None else _signed(v, multiplicative) for v in row] for row in z
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=axis,
            y=[_escape(e.name) for e in rows],
            colorscale=colorscale(DIVERGING),
            zmin=-reach,
            zmax=reach,
            xgap=2,
            ygap=2,
            hoverongaps=False,
            text=cells,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar={
                "thickness": 12,
                "outlinewidth": 0,
                "tickfont": {"color": INK_SOFT, "size": 11},
                **({"tickformat": pct_tickformat(reach)} if multiplicative else {}),
            },
            customdata=[
                [f"<b>{_escape(e.name)}</b><br>{offset_phrase(o)}" for o in axis]
                for e in rows
            ],
            hovertemplate=(
                "%{customdata}<br>Effect <b>"
                + ("%{z:+.2%}" if multiplicative else "%{z:+,.4g}")
                + "</b><extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_style.layout(
            max(200, 34 * len(rows) + 110),
            margin={"l": 64, "r": 24, "t": 24, "b": 64},
            xaxis={
                **x_style,
                "tickmode": "array",
                "tickvals": axis,
                "ticktext": ["0" if o == 0 else f"{o:+d}" for o in axis],
                "title": {
                    "text": "Days from the holiday",
                    "font": {"color": INK_SOFT, "size": 12},
                },
            },
            yaxis={
                **x_style,
                "type": "category",
                "autorange": "reversed",
                "automargin": True,
                "ticks": "",
            },
        )
    )
    return fig


def timeline_figure(
    ctx: DashboardContext, series: pd.DataFrame, multiplicative: bool
) -> Any:
    """What all holidays add to the forecast over time, with history and forecast tinted."""
    go = ctx.go
    _, y_style = _style.axes()
    value = "%{y:+.2%}" if multiplicative else "%{y:+,.4g}"
    largest = float(series["total"].abs().max()) if len(series) else 0.0
    fig = go.Figure(
        go.Scatter(
            x=series["ds"],
            y=series["total"],
            mode="lines+markers",
            line={"color": ACCENT, "width": 2},
            marker={
                "color": ACCENT,
                "size": [5 if v != 0 else 0 for v in series["total"]],
            },
            customdata=series["label"].tolist(),
            hovertemplate=f"%{{x|{_DAY}}}<br>All holidays <b>{value}</b>%{{customdata}}<extra></extra>",
        )
    )
    fig.update_layout(
        **_style.layout(
            280,
            showlegend=False,
            hovermode="closest",
            hoverdistance=30,
            xaxis=_style.time_axis(ctx),
            yaxis={
                **y_style,
                "zeroline": True,
                "zerolinecolor": INK_SOFT,
                **({"tickformat": pct_tickformat(largest)} if multiplicative else {}),
            },
        )
    )
    _style.zones(ctx, fig)
    return fig


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------


def _effect_cell(effect: float | None, multiplicative: bool) -> str:
    return _NOT_SEEN if effect is None else _signed(effect, multiplicative)


def upcoming_table(
    days: Sequence[HolidayDay], span: ForecastSpan, multiplicative: bool
) -> tuple[list[str], list[list[str]], int]:
    """Headers, rows and the number of rows left out for the holiday days inside the forecast window."""
    future = [d for d in days if span.contains(d.date)]
    shown = future[:MAX_LISTED]
    windowed = any(d.offset != 0 for d in shown)
    headers = [
        "Date",
        "Holiday",
        *(["Window day"] if windowed else []),
        "Predicted effect",
    ]
    rows = [
        [
            _weekday_date(d.date),
            d.name,
            *([_capitalise(offset_phrase(d.offset))] if windowed else []),
            _effect_cell(d.effect, multiplicative),
        ]
        for d in shown
    ]
    return headers, rows, len(future) - len(shown)


def next_cell(next_date: NextDate | None) -> str:
    """When a holiday next touches the forecast window, for the table."""
    if next_date is None:
        return "none"
    text = _weekday_date(next_date.date)
    return text if next_date.inside else f"{text} (window reaches in)"


def all_holidays_table(
    effects: Sequence[HolidayEffect],
    nexts: Mapping[str, NextDate] | None,
    multiplicative: bool,
) -> tuple[list[str], list[list[str]]]:
    """Headers and rows for every holiday, biggest effect first and unseen ones last.

    ``nexts`` is None, or empty, when there is nothing to say about the forecast
    window (it has none, or no holiday reaches it), which drops that column. The
    window and biggest-day columns are dropped when every holiday is a single day.
    The two numeric columns come last: the page pads a numeric cell on its left
    only, so a text column after one would run into its figures.
    """
    ordered = sorted(
        effects,
        key=lambda e: (e.effect is None, -abs(_effect_value(e)), e.name),
    )
    windowed = any(e.multi_day for e in effects)
    upcoming = bool(nexts)
    headers = [
        "Holiday",
        *(["Window", "Biggest day"] if windowed else []),
        *(["Next in forecast"] if upcoming else []),
        "Seen in history",
        "Average effect",
    ]
    rows = [
        [
            e.name,
            *([window_phrase(e.window), peak_cell(e)] if windowed else []),
            *([next_cell((nexts or {}).get(e.name))] if upcoming else []),
            str(e.occurrences),
            _effect_cell(e.effect, multiplicative),
        ]
        for e in ordered
    ]
    return headers, rows


# ----------------------------------------------------------------------------
# The section
# ----------------------------------------------------------------------------


def _unit_note(multiplicative: bool) -> str:
    return (
        "Effects are shares of the trend level, so they grow and shrink with it."
        if multiplicative
        else "Effects are in the units of y."
    )


def _profile_note(shown: int, left_out: int) -> str:
    text = (
        "Each row is a holiday with a window wider than one day; each column is a day of it, "
        "counted from the holiday itself. Blank cells are days the history never saw."
    )
    if left_out:
        text += f" Showing the {shown} largest; the other {left_out} are in the table below."
    return text


def _bars_note(shown: int, left_out: int, multiplicative: bool) -> str:
    parts = [
        "Each bar is the average effect on the days a holiday applies, measured on the history. "
        "Hover for how often it occurred and, for holidays with a window, each day of it.",
        _unit_note(multiplicative),
    ]
    if left_out:
        parts.append(
            f"Showing the {shown} largest effects; the other {left_out} are in the table below."
        )
    return " ".join(parts)


def build(ctx: DashboardContext) -> Section | None:
    """The section, or None when it does not apply to this model."""
    inputs = read_model(ctx)
    if inputs is None:
        return None
    mult = inputs.multiplicative
    scale = _y_scale(ctx.m)
    effects = holiday_effects(
        ctx.history["ds"],
        inputs.history,
        inputs.coefficients,
        inputs.names,
        inputs.occurrences,
    )
    indicators = holiday_indicators(ctx.m, ctx.fcst["ds"])
    days = holiday_days(ctx.fcst["ds"], indicators, inputs.coefficients, effects)
    span = forecast_span(ctx.future["ds"])
    ahead = ahead_summary(days, span)
    nexts = None if span is None else next_dates(days, span)

    parts = [
        _style.lede(holiday_lede(effects, mult, scale, ahead)),
    ]
    rows, left_out = chart_rows(effects)
    if rows:
        parts += [
            _style.h3("Effect of each holiday"),
            ctx.embed(effect_figure(ctx, rows, mult)),
            _style.note(_bars_note(len(rows), left_out, mult)),
        ]
    windowed_all = sorted(
        (e for e in effects if e.effect is not None and e.multi_day and e.profile),
        key=lambda e: (-abs(_effect_value(e)), e.name),
    )
    windowed = windowed_all[:MAX_BARS]
    if windowed:
        parts += [
            _style.h3("Days within a window"),
            ctx.embed(profile_figure(ctx, windowed, mult)),
            _style.note(
                _profile_note(len(windowed), len(windowed_all) - len(windowed))
            ),
        ]

    vector = np.array(
        [inputs.coefficients.get(str(c), 0.0) for c in indicators.columns]
    )
    total = (
        indicators.to_numpy(dtype=float) @ vector
        if len(vector)
        else np.zeros(len(ctx.fcst))
    )
    if np.any(total != 0):
        labels_by_day = day_labels(days, mult)
        day_index = pd.DatetimeIndex(ctx.fcst["ds"]).normalize()
        series = spike_series(
            ctx.fcst["ds"], total, [labels_by_day.get(d, "") for d in day_index]
        )
        parts += [
            _style.h3("Holidays through time"),
            ctx.embed(timeline_figure(ctx, series, mult)),
            _style.note(
                "What all holidays together add to the forecast on each day. Each spike is a holiday; "
                "hover for which ones. " + _unit_note(mult)
            ),
        ]

    listed: list[list[str]] = []
    if span is not None:
        headers, listed, left = upcoming_table(days, span, mult)
        if listed:
            parts += [
                _style.h3("Holidays inside the forecast window"),
                _style.table(
                    headers,
                    listed,
                    numeric=(len(headers) - 1,),
                    scroll=len(listed) > 12,
                    table_id="holidays-upcoming",
                ),
                _style.note(
                    "The effect the forecast gives each day, from the average of that day of the holiday in the history. "
                    f"'{_NOT_SEEN}' marks a day the history never saw, which the model cannot judge."
                    + (
                        f" The first {len(listed)} of {len(listed) + left} are listed."
                        if left
                        else ""
                    )
                ),
            ]
    # The upcoming table already says which holidays reach the forecast, and when.
    headers, table_rows = all_holidays_table(effects, None if listed else nexts, mult)
    parts += [
        _style.h3("All holidays"),
        _style.table(
            headers,
            table_rows,
            numeric=(len(headers) - 2, len(headers) - 1),
            scroll=len(table_rows) > 12,
            table_id="holidays-all",
        ),
        _style.note(
            "The average effect is taken over the days a holiday applies in the history; one seen only once is a rough guide. "
            f"'{_NOT_SEEN}' means no date of that holiday falls in the history, so there was nothing to measure it from."
        ),
    ]
    return Section(
        key=KEY,
        title=TITLE,
        body="".join(parts),
        summary=holiday_summary(effects, mult, scale, ahead),
    )
