"""The component panels of the "Components" section, made to agree with the rest of the page.

:func:`prophet.plot.plot_components_plotly` draws every seasonal curve over a
whole cycle. For a weekly curve that includes the weekdays the data never has,
and for a daily curve the hours it never has. The curve is unconstrained there,
so its values, and the sentence read off its highest and lowest points, are
extrapolations that contradict the Seasonality section. This module trims those
points and re-centres the curve on the days and hours the data has, writes the
sentences again from what is left, gives every trace a hover label and leaves
out the panels that have a section of their own.

Numbers and wording come from small pure functions; :func:`refine` only applies
them to the figures.
"""

from __future__ import annotations

import html
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from prophet.dashboard import _style
from prophet.dashboard._context import DashboardContext
from prophet.dashboard.seasonality import (
    Fact,
    _into_cycle,
    missing_days_phrase,
    observed_calendar,
    seasonal_facts,
    seasonality_kind,
)
from prophet.plot import (
    _above_below,
    _holiday_columns,
    _plotly_rangebreaks,
    _seasonality_headline,
)
from prophet.visuals import _fmt_number, _pretty

#: Where every seasonal curve starts on the x axis of its panel (``prophet.plot`` draws them from here).
_CYCLE_START = pd.Timestamp("2017-01-01")
_WEEKDAYS = frozenset(range(7))
_HOURS = frozenset(range(24))
#: The capacity lines of a logistic trend panel, by trace name.
_LIMITS: Mapping[str, str] = {"Cap": "Maximum capacity", "Floor": "Minimum capacity"}
#: Trace names of the panels that other sections draw in more detail.
LONG_FORM = ("trend", "holidays")


@dataclass(frozen=True)
class Panel:
    """One panel of the section: a heading, a sentence and the figure under them."""

    name: str
    head: str
    sentence: str
    figure: Any


# ----------------------------------------------------------------------------
# Pure: which points of a cycle the data has
# ----------------------------------------------------------------------------


def observed_mask(
    grid: pd.DatetimeIndex,
    *,
    weekly: bool,
    intraday: bool,
    days: frozenset[int],
    hours: frozenset[int],
) -> np.ndarray:
    """True where a point of a cycle falls on a weekday (weekly curves) and an hour (intraday data) the data has."""
    keep = np.ones(len(grid), dtype=bool)
    if weekly:
        keep &= np.isin(np.asarray(grid.dayofweek), sorted(days))
    if intraday:
        keep &= np.isin(np.asarray(grid.hour), sorted(hours))
    return keep


def left_out_phrase(
    *, weekly: bool, intraday: bool, days: frozenset[int], hours: frozenset[int]
) -> str:
    """'Saturdays and Sundays are left out because the data never has them.', and the same for hours."""
    parts = []
    if weekly:
        parts.append(missing_days_phrase(days))
    if intraday and hours != _HOURS:
        parts.append("Hours of the day the data never has are left out.")
    return " ".join(p for p in parts if p)


def missing_breaks(
    *, weekly: bool, intraday: bool, days: frozenset[int], hours: frozenset[int]
) -> list[dict[str, object]]:
    """Plotly range breaks that close the gaps left by the weekdays and hours the data never has."""
    return _plotly_rangebreaks(
        _WEEKDAYS - days, _HOURS - hours, days=weekly, hours=intraday
    )


# ----------------------------------------------------------------------------
# Pure: wording
# ----------------------------------------------------------------------------


def _versus_usual(value: float, multiplicative: bool, *, first: bool) -> str:
    """'8.75 above the usual level' for the first clause, '9 below it' for the second, 'at it' when level."""
    amount = _above_below(value, multiplicative)
    if amount == "at":
        return "at the usual level" if first else "at it"
    return f"{amount} the usual level" if first else f"{amount} it"


def fact_text(fact: Fact, name: str) -> tuple[str, str]:
    """Headline and sentence of a yearly or custom-cycle panel, from the same peak and low as the Seasonality section."""
    label = _pretty(name)
    if fact.flat:
        return f"{label} pattern: little variation", ""
    mult = fact.multiplicative
    high = _versus_usual(fact.high_value, mult, first=True)
    low = _versus_usual(fact.low_value, mult, first=False)
    if fact.kind == "yearly":
        return (
            f"{label} pattern: peaks around {fact.high}",
            f"{fact.high} is {high} and {fact.low} is {low}.",
        )
    return (
        f"{label} pattern: peaks {fact.high}",
        f"The high point ({fact.high}) is {high}; the low point ({fact.low}) is {low}.",
    )


def largest_holiday(
    columns: Sequence[str], values: np.ndarray
) -> tuple[str, int, float] | None:
    """The holiday column and row with the largest absolute effect, or None when no holiday moves the forecast.

    The column name is kept whole: ``Christmas Day (observed)`` is a day of its own with its
    own effect, not another name for ``Christmas Day``.
    """
    if not columns or values.size == 0:
        return None
    absolute = np.abs(values)
    if not np.isfinite(absolute).any() or np.nanmax(absolute) == 0:
        return None
    row, col = np.unravel_index(int(np.nanargmax(absolute)), values.shape)
    return columns[col], int(row), float(values[row, col])


def holiday_text(name: str, value: float, multiplicative: bool) -> tuple[str, str]:
    """Headline and sentence for the holiday panel."""
    return (
        f"Holidays: {name} has the largest effect",
        f"On {name} the forecast is {_above_below(value, multiplicative)} its usual level.",
    )


def flat_trend_text(level: float) -> tuple[str, str]:
    """Headline and sentence for a trend that does not move."""
    return (
        "Trend: broadly flat",
        "With seasonal and holiday effects removed, the underlying level stays at "
        f"about {_fmt_number(level)}.",
    )


_COVERED_NOTES: Mapping[tuple[str, ...], str] = {
    ("trend",): "The trend has a section of its own, so it is not repeated here.",
    (
        "holidays",
    ): "The holidays have a section of their own, so they are not repeated here.",
    ("trend", "holidays"): (
        "The trend and the holidays have sections of their own, so they are not repeated here."
    ),
}


def covered_note(dropped: Sequence[str]) -> str:
    """Why panels are missing: the trend and the holidays are drawn in sections of their own."""
    return _COVERED_NOTES.get(tuple(n for n in LONG_FORM if n in dropped), "")


# ----------------------------------------------------------------------------
# Pure: hover labels
# ----------------------------------------------------------------------------


def cycle_labels(grid: pd.DatetimeIndex, kind: str) -> list[str]:
    """Where in the cycle each point of a yearly or custom curve falls, in words."""
    if kind == "yearly":
        return [f"{d.day} {d:%B}" for d in grid]
    offsets = (grid - _CYCLE_START) / pd.Timedelta(days=1)
    return [_into_cycle(float(o)) for o in offsets]


def hover_template(
    when: str,
    label: str,
    *,
    multiplicative: bool,
    signed: bool,
    interval: str | None,
    prefix: str = "",
) -> str:
    """The hover label of a panel's line: the position, any extra line, the value and the interval.

    The interval and the extra line read ``customdata`` (lower, upper, text).
    """
    spec = f"{'+' if signed else ''}{'.2%' if multiplicative else ',.4g'}"
    bounds = (
        ""
        if interval is None
        else f"<br>{interval} %{{customdata[0]:{spec}}} to %{{customdata[1]:{spec}}}"
    )
    return f"{when}<br>{prefix}{label} <b>%{{y:{spec}}}</b>{bounds}<extra></extra>"


def _escaped_names(text: Sequence[str] | None, n: int) -> list[str]:
    """Holiday names for a hover label, each followed by a line break, or an empty string on days without one."""
    if text is None:
        return [""] * n
    return [f"{html.escape(str(t), quote=False)}<br>" if t else "" for t in text]


# ----------------------------------------------------------------------------
# Applying it to the figures
# ----------------------------------------------------------------------------


def panel_name(figure: Any) -> str:
    """The component a panel draws: the name of its first trace."""
    return cast("str", figure.data[0].name) if len(figure.data) else ""


def _grid_of(trace: Any) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.asarray(trace.x))


def _trim(figure: Any, keep: np.ndarray) -> float:
    """Drop the points outside ``keep`` from every trace and centre the line on what is left; returns the old average."""
    main = np.asarray(figure.data[0].y, dtype=float)[keep]
    level = float(np.nanmean(main)) if np.isfinite(main).any() else 0.0
    for trace in figure.data:
        trace.update(
            x=np.asarray(trace.x)[keep],
            y=np.asarray(trace.y, dtype=float)[keep] - level,
        )
    return level


def _trimmed_cycle(
    ctx: DashboardContext,
    name: str,
    figure: Any,
    kind: str,
    days: frozenset[int],
    hours: frozenset[int],
) -> tuple[str, str] | None:
    """Trim a weekly or daily curve to the days and hours the data has; the new headline and sentence, or None if nothing needed trimming."""
    weekly = kind == "weekly"
    intraday = ctx.step < pd.Timedelta(days=1)
    grid = _grid_of(figure.data[0])
    keep = observed_mask(grid, weekly=weekly, intraday=intraday, days=days, hours=hours)
    if keep.all() or not keep.any():
        return None
    _trim(figure, keep)
    figure.update_xaxes(
        rangebreaks=missing_breaks(
            weekly=weekly, intraday=intraday, days=days, hours=hours
        )
    )
    day_precision = bool((cast("pd.DataFrame", ctx.m.history)["ds"].dt.hour == 0).all())
    head, sentence = _seasonality_headline(
        ctx.m,
        name,
        grid[keep],
        np.asarray(figure.data[0].y, dtype=float),
        "weekly" if weekly and day_precision else "other",
    )
    left_out = left_out_phrase(weekly=weekly, intraday=intraday, days=days, hours=hours)
    return head, " ".join(p for p in (sentence, left_out) if p)


def _seasonal(
    ctx: DashboardContext,
    name: str,
    head: str,
    sentence: str,
    figure: Any,
    days: frozenset[int],
    hours: frozenset[int],
    facts: Mapping[str, Fact],
) -> tuple[str, str, str, list[str] | None]:
    """Headline, sentence, the position part of the hover label and per-point position labels of a seasonal panel."""
    period = float(ctx.m.seasonalities[name]["period"])
    kind = seasonality_kind(period)
    if kind in ("weekly", "daily"):
        trimmed = _trimmed_cycle(ctx, name, figure, kind, days, hours)
        if trimmed is not None:
            head, sentence = trimmed
        intraday = ctx.step < pd.Timedelta(days=1)
        when = (
            "%{x|%A %H:%M}"
            if kind == "weekly" and intraday
            else "%{x|%A}"
            if kind == "weekly"
            else "%{x|%H:%M}"
        )
        return head, sentence, when, None
    fact = facts.get(name)
    if fact is not None and "little variation" not in head:
        head, sentence = fact_text(fact, name)
    labels = cycle_labels(_grid_of(figure.data[0]), kind)
    return head, sentence, "%{customdata[2]}", labels


def _holidays(ctx: DashboardContext, head: str, sentence: str) -> tuple[str, str]:
    columns = _holiday_columns(ctx.m, ctx.fcst)
    if not columns:
        return head, sentence
    found = largest_holiday(columns, ctx.fcst[columns].to_numpy(dtype=float))
    if found is None:
        return head, sentence
    name, _, value = found
    mult = (
        "holidays"
        in cast("dict[str, list[str]]", ctx.m.component_modes)["multiplicative"]
    )
    return holiday_text(name, value, mult)


def _trend(ctx: DashboardContext, head: str, sentence: str) -> tuple[str, str]:
    if not head.startswith("Trend: broadly flat"):
        return head, sentence
    inside = ctx.fcst[ctx.fcst["ds"] <= ctx.history_end]["trend"]
    rows = inside if len(inside) else ctx.fcst["trend"]
    return flat_trend_text(float(rows.mean()))


def add_hover(
    ctx: DashboardContext,
    figure: Any,
    label: str,
    when: str,
    *,
    signed: bool,
    positions: Sequence[str] | None = None,
    holiday_names: bool = False,
) -> None:
    """Give a panel's line a hover label and silence its interval traces.

    ``when`` is the position part of the label (a date format or a reference to
    the position labels in ``customdata``), ``positions`` those labels.
    """
    main = figure.data[0]
    multiplicative = "%" in (figure.layout.yaxis.tickformat or "")
    bands = {t.name: t for t in figure.data[1:]}
    lower = bands.get(f"{main.name}_lower")
    upper = bands.get(f"{main.name}_upper")
    interval = (
        f"{ctx.m.interval_width:.0%} interval"
        if lower is not None and upper is not None
        else None
    )
    template = hover_template(
        when,
        label,
        multiplicative=multiplicative,
        signed=signed,
        interval=interval,
        prefix="%{customdata[2]}" if holiday_names else "",
    )
    n = len(main.y)
    texts = (
        list(positions)
        if positions is not None
        else _escaped_names(cast("Sequence[str] | None", main.text), n)
        if holiday_names
        else [""] * n
    )
    if interval is None and positions is None and not holiday_names:
        main.update(hovertemplate=template)
    else:
        lows = np.asarray(lower.y, dtype=float) if lower is not None else None
        highs = np.asarray(upper.y, dtype=float) if upper is not None else None
        main.update(
            customdata=[
                [
                    None if lows is None else float(lows[i]),
                    None if highs is None else float(highs[i]),
                    texts[i],
                ]
                for i in range(n)
            ],
            text=None,
            hovertemplate=template,
        )
    for trace in figure.data[1:]:
        limit = _LIMITS.get(trace.name)
        if limit is None:
            trace.update(hoverinfo="skip")
        else:
            trace.update(
                hovertemplate=f"{when}<br>{limit} <b>%{{y:,.4g}}</b><extra></extra>"
            )


def refine(
    ctx: DashboardContext,
    raw: Sequence[tuple[str, str, Any]],
    covered: Collection[str] = (),
) -> tuple[list[Panel], list[str]]:
    """The panels to draw, and the names of those left out because ``covered`` sections draw them.

    ``raw`` is ``(headline, sentence, figure)`` per component as split from
    ``plot_components_plotly``. Figures are changed in place.
    """
    intraday = ctx.step < pd.Timedelta(days=1)
    days, hours = observed_calendar(ctx.history["ds"], intraday)
    facts: Mapping[str, Fact] | None = None
    panels: list[Panel] = []
    dropped: list[str] = []
    date = f"%{{x|{_style.hover_date(ctx)}}}"
    for head, sentence, figure in raw:
        name = panel_name(figure)
        if name in covered:
            dropped.append(name)
            continue
        if name in ctx.m.seasonalities:
            if facts is None:
                facts = {
                    f.name: f
                    for f in seasonal_facts(
                        ctx.m,
                        ctx.fcst,
                        ctx.history_end,
                        ctx.step,
                        days,
                        hours,
                    )
                }
            head, sentence, when, positions = _seasonal(
                ctx, name, head, sentence, figure, days, hours, facts
            )
            add_hover(
                ctx, figure, _pretty(name), when, signed=True, positions=positions
            )
        elif name == "trend":
            head, sentence = _trend(ctx, head, sentence)
            add_hover(ctx, figure, "Trend", date, signed=False)
        elif name == "holidays":
            head, sentence = _holidays(ctx, head, sentence)
            add_hover(ctx, figure, "Holidays", date, signed=True, holiday_names=True)
        else:
            add_hover(ctx, figure, "Extra regressors", date, signed=True)
        panels.append(Panel(name, head, sentence, figure))
    return panels, dropped
