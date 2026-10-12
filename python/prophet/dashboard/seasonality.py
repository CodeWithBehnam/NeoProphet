"""Dashboard section "Seasonal patterns": how much each component matters, the peaks and lows of every seasonality, and the observed data by weekday, hour, month and calendar day.

Numbers and wording come from small pure functions (``component_importance``,
``fact_from_profile``, ``observed_grid``, ``daily_values``, ``calendar_layout``
and the sentence builders); :func:`build` only draws them.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import pandas as pd

from prophet.dashboard import _palette, _style
from prophet.dashboard._context import DashboardContext, Section
from prophet.plot import INK, INK_SOFT, seasonality_plot_df
from prophet.visuals import (
    _fmt_number,
    _fmt_pct,
    _plural,
    _pretty,
    _signed,
    _step_unit,
    _y_scale,
)

if TYPE_CHECKING:
    from prophet.forecaster import Prophet
    from prophet.plot import SkipAxis

KEY = "seasonality"
TITLE = "Seasonal patterns"

Kind = Literal["weekly", "daily", "yearly", "cycle"]
Part = Literal["trend", "seasonality", "holidays", "regressor"]

#: A seasonal swing below this share of the typical movement of the data (its standard deviation) is called flat.
FLAT_SHARE = 0.01
#: Days of history needed before month by weekday averages mean anything (about ten months).
MIN_HEATMAP_DAYS = 300
#: Days needed before a calendar is worth drawing.
MIN_CALENDAR_DAYS = 21
#: Days of intraday history needed before weekday by hour averages say anything (two weeks).
MIN_GRID_DAYS = 14
MAX_CALENDAR_YEARS = 3
MAX_BARS = 12
#: A day counts as complete when it has this share of the usual number of observations.
COMPLETE_DAY = 0.8
#: Points sampled across one period of a seasonality that is not weekly, daily or yearly.
CYCLE_POINTS = 120
#: The widest a calendar year can be, in Monday-first weeks.
MAX_WEEKS = 54
#: The narrowest a calendar is drawn, in pixels; narrower screens scroll sideways.
CALENDAR_MIN_WIDTH = 640
#: Pixels of a calendar: height of one weekday row and the margins above and below.
_CAL_ROW = 17
_CAL_TOP = 56
_CAL_BOTTOM = 12

_CALENDAR_PERIODS: tuple[tuple[Kind, float], ...] = (
    ("weekly", 7.0),
    ("daily", 1.0),
    ("yearly", 365.25),
)
_BASE_MONDAY = pd.Timestamp("2017-01-02")
_BASE_YEAR = pd.Timestamp("2017-01-01")
_HOUR = pd.Timedelta(hours=1)
_EPS = 1e-9


# ----------------------------------------------------------------------------
# Pure: wording helpers
# ----------------------------------------------------------------------------


def _lower_first(text: str) -> str:
    """Lower-case the first letter to continue a sentence, unless the first word is a name like 'NFL' or 'Q4'."""
    first_word = text.split(" ", 1)[0]
    if any(c.isupper() or c.isdigit() for c in first_word[1:]):
        return text
    return text[:1].lower() + text[1:]


def _span_words(days: float) -> str:
    """A length of time in the unit that fits it: '12 days', '1 hour', '30 minutes'."""
    if days >= 1 - _EPS:
        n, unit = days, "day"
    elif days * 24 >= 1 - _EPS:
        n, unit = days * 24, "hour"
    else:
        n, unit = days * 1440, "minute"
    text = str(round(n)) if n >= 10 else f"{n:.2g}"
    return f"{text} {unit}" if text == "1" else f"{text} {unit}s"


def _cycle_words(period: float) -> str:
    """'30.5-day', '1-hour': the length of a custom seasonal cycle, ready to precede 'cycle'."""
    if period >= 1 - _EPS:
        return f"{period:.3g}-day"
    if period * 24 >= 1 - _EPS:
        return f"{period * 24:.3g}-hour"
    return f"{period * 1440:.3g}-minute"


def _into_cycle(offset_days: float) -> str:
    return "at the start" if offset_days <= _EPS else f"{_span_words(offset_days)} in"


def _swing_text(swing: float, multiplicative: bool) -> str:
    return f"{_fmt_pct(swing)} of the trend" if multiplicative else _fmt_number(swing)


def missing_days_phrase(days: frozenset[int]) -> str:
    """'Saturdays and Sundays are left out ...' for the weekdays the data never has, or an empty string."""
    names = [f"{calendar.day_name[d]}s" for d in range(7) if d not in days]
    if not names or len(names) == 7:
        return ""
    joined = (
        names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    )
    return f"{joined} are left out because the data never has them."


# ----------------------------------------------------------------------------
# Pure: how much each component matters
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Importance:
    """The typical size of one component's effect on the fitted values."""

    name: str
    label: str
    #: The label as the subject of a sentence: 'The weekly pattern', 'Holidays'.
    noun: str
    part: Part
    #: Standard deviation of its contribution, in the units of the data.
    spread: float
    #: ``spread`` as a share of all the spreads together.
    share: float
    #: The verb that agrees with ``noun``: 'carries' for 'The trend', 'carry' for 'Holidays'.
    verb: str = "carries"


def basis_rows(
    fcst: pd.DataFrame, history_end: pd.Timestamp
) -> tuple[pd.DataFrame, bool]:
    """The rows the comparison is made on, and whether they are in-sample.

    A forecast frame that holds only future dates falls back to all of its
    rows, so that the section still has something to say.
    """
    inside = fcst[fcst["ds"] <= history_end]
    if len(inside) >= 2:
        return inside, True
    return fcst, False


def _is_multiplicative(m: Prophet, name: str) -> bool:
    modes = m.component_modes
    return modes is not None and name in modes["multiplicative"]


class _Spec(NamedTuple):
    """One bar before it is measured: what to call it and which forecast columns make it up."""

    name: str
    label: str
    noun: str
    part: Part
    members: tuple[str, ...]
    verb: str = "carries"


def _effect(
    m: Prophet, rows: pd.DataFrame, members: Sequence[str]
) -> tuple[pd.Series, float]:
    """What some forecast columns add to the movement, in the units of the data, and the constant they add to the trend.

    Additive columns are in the units of the data already, and a constant does
    not change a standard deviation. A multiplicative column is a fraction of
    the trend, so it is centred before it is scaled by the trend: its average
    is a constant multiple of the trend and belongs to the trend, not to the
    component. (A weekly curve that sits at 4.5 on the weekdays the data has
    would otherwise borrow the whole variation of the trend.) The second
    value is the sum of those averages.
    """
    trend = rows["trend"].astype(float)
    total = pd.Series(0.0, index=rows.index)
    fraction = 0.0
    for name in members:
        values = rows[name].astype(float)
        if _is_multiplicative(m, name):
            level = float(values.mean())
            if not np.isfinite(level):
                level = 0.0
            total = total + (values - level) * trend
            fraction += level
        else:
            total = total + values
    return total, fraction


def _specs(m: Prophet, rows: pd.DataFrame, max_regressors: int) -> list[_Spec]:
    """The components other than the trend that the forecast has columns for."""
    specs = [
        _Spec(
            name,
            _pretty(name),
            f"The {_lower_first(name.replace('_', ' '))} pattern",
            "seasonality",
            (name,),
        )
        for name in m.seasonalities
        if name in rows
    ]
    if m.train_holiday_names is not None and "holidays" in rows:
        specs.append(
            _Spec(
                "holidays", "Holidays", "Holidays", "holidays", ("holidays",), "carry"
            )
        )
    regressors = [name for name in m.extra_regressors if name in rows]
    if len(regressors) > max_regressors:
        specs.append(
            _Spec(
                "extra_regressors",
                f"Regressors ({len(regressors)})",
                f"The {len(regressors)} regressors together",
                "regressor",
                tuple(regressors),
                "carry",
            )
        )
    else:
        specs += [
            _Spec(
                name,
                f"{_pretty(name)} (regressor)",
                f"The {name.replace('_', ' ')} regressor",
                "regressor",
                (name,),
            )
            for name in regressors
        ]
    return specs


def component_importance(
    m: Prophet,
    fcst: pd.DataFrame,
    history_end: pd.Timestamp,
    *,
    max_regressors: int = 5,
) -> tuple[Importance, ...]:
    """Trend, every seasonality, holidays and regressors, largest effect first.

    The effect of a component is the standard deviation of its contribution
    over the in-sample rows, in the units of the data, and the share is that
    spread over the sum of all spreads. The constant part of every
    multiplicative component is booked to the trend (see :func:`_effect`).
    Components that do not move at all (a flat trend, a seasonality that never
    switches on) are left out. More than ``max_regressors`` regressors are
    combined into one bar.
    """
    rows, _ = basis_rows(fcst, history_end)
    if len(rows) < 2 or "trend" not in rows:
        return ()
    specs = _specs(m, rows, max_regressors)
    effects = [_effect(m, rows, spec.members) for spec in specs]
    shift = sum(fraction for _, fraction in effects)
    trend = rows["trend"].astype(float) * (1.0 + shift)
    candidates = [(_Spec("trend", "Trend", "The trend", "trend", ()), trend)] + [
        (spec, series) for spec, (series, _) in zip(specs, effects)
    ]
    floor = _EPS * _y_scale(m)
    spreads = [(spec, float(series.std(ddof=0))) for spec, series in candidates]
    kept = [
        (spec, spread)
        for spec, spread in spreads
        if np.isfinite(spread) and spread > floor
    ]
    if not kept:
        return ()
    total_spread = sum(spread for _, spread in kept)
    items = [
        Importance(
            spec.name,
            spec.label,
            spec.noun,
            spec.part,
            spread,
            spread / total_spread,
            spec.verb,
        )
        for spec, spread in kept
    ]
    return tuple(sorted(items, key=lambda item: -item.share))


def _lead_clause(top: Importance, in_sample: bool) -> str:
    """'The trend carries most of the movement in the fitted values (71%)'."""
    amount = "most" if top.share > 0.5 else "the largest share"
    where = "in the fitted values" if in_sample else "in the forecast"
    return f"{top.noun} {top.verb} {amount} of the movement {where} ({_fmt_pct(top.share)})"


def importance_sentence(items: Sequence[Importance], in_sample: bool = True) -> str:
    """'The trend carries most of the movement in the fitted values (71%), followed by ...'."""
    if len(items) < 2:
        return ""
    rest = " and ".join(
        f"{_lower_first(i.noun)} at {_fmt_pct(i.share)}" for i in items[1:3]
    )
    return f"{_lead_clause(items[0], in_sample)}, followed by {rest}."


# ----------------------------------------------------------------------------
# Pure: peaks and lows of each seasonality
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """One seasonal curve sampled at labelled positions: weekdays, hours, days of the year or points in a cycle."""

    kind: Kind
    labels: tuple[str, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class Fact:
    """Where one seasonality peaks and bottoms out, and by how much."""

    name: str
    label: str
    kind: Kind
    multiplicative: bool
    high: str
    low: str
    #: How far the highest and the lowest point lie from the average of the curve (units of the data, or a fraction of the trend if multiplicative).
    high_value: float
    low_value: float
    #: High minus low, in the units of the data (additive) or a fraction of the trend (multiplicative).
    swing: float
    #: ``swing`` in the units of the data, so that additive and multiplicative patterns compare.
    swing_y: float
    flat: bool

    def _where(self, position: str) -> str:
        return {
            "weekly": "on {}",
            "daily": "around {}",
            "yearly": "around {}",
            "cycle": "{}",
        }[self.kind].format(position)

    @property
    def peak(self) -> str:
        """Where the curve is highest, as a phrase: 'on Thursday', 'around 28 February', '21 days in'."""
        return self._where(self.high)

    @property
    def detail(self) -> str:
        if self.flat:
            return "almost no pattern"
        return (
            f"highest {self._where(self.high)}, lowest {self._where(self.low)}, "
            f"a swing of {_swing_text(self.swing, self.multiplicative)}"
        )

    @property
    def sentence(self) -> str:
        return f"{self.label}: {self.detail}"


def seasonality_kind(period: float) -> Kind:
    """Which calendar cycle a seasonal period (in days) follows, if any."""
    for kind, days in _CALENDAR_PERIODS:
        if abs(period - days) < 1e-6:
            return kind
    return "cycle"


def fact_from_profile(
    profile: Profile,
    *,
    name: str,
    label: str,
    multiplicative: bool,
    scale: float,
    level: float,
) -> Fact:
    """The peak, the low and the swing of a profile.

    The levels of the peak and the low are taken from the average of the
    profile: a seasonal curve is only defined up to a constant (and, over just
    the weekdays or hours the data has, is not centred on zero), so its raw
    values would read as effects they are not. ``scale`` is the typical
    movement of the data (see :func:`_data_spread`), for the flatness test, and
    ``level`` the typical trend, which turns a multiplicative swing into the
    units of the data so that both kinds are judged alike.
    """
    values = np.asarray(profile.values, dtype=float)
    hi, lo = int(np.argmax(values)), int(np.argmin(values))
    swing = float(values[hi] - values[lo])
    swing_y = swing * level if multiplicative else swing
    average = float(values.mean())
    return Fact(
        name=name,
        label=label,
        kind=profile.kind,
        multiplicative=multiplicative,
        high=profile.labels[hi],
        low=profile.labels[lo],
        high_value=float(values[hi]) - average,
        low_value=float(values[lo]) - average,
        swing=swing,
        swing_y=swing_y,
        flat=swing_y < FLAT_SHARE * scale,
    )


def _data_spread(m: Prophet) -> float:
    """The typical movement of the data: the standard deviation of ``y``, or its size when ``y`` is constant."""
    y = (
        np.asarray(m.history["y"], dtype=float)
        if m.history is not None
        else np.empty(0)
    )
    y = y[np.isfinite(y)]
    spread = float(y.std()) if y.size else 0.0
    return spread if spread > 0 else _y_scale(m)


def strongest_fact(facts: Sequence[Fact]) -> Fact | None:
    """The seasonality with the biggest swing in the units of the data, ignoring flat ones."""
    return max((f for f in facts if not f.flat), key=lambda f: f.swing_y, default=None)


def observed_range(m: Prophet) -> tuple[float, float] | None:
    """The lowest and highest observed value the model was trained on, or None without any."""
    y = (
        np.asarray(m.history["y"], dtype=float)
        if m.history is not None
        else np.empty(0)
    )
    y = y[np.isfinite(y)]
    return (float(y.min()), float(y.max())) if y.size else None


Excess = Literal["range", "trend"]

#: What the summary says about a swing that needs a caution.
EXCESS_PHRASE: dict[Excess, str] = {
    "range": "wider than the whole range of the data",
    "trend": "larger than the trend itself",
}


def swing_excess(fact: Fact, observed: tuple[float, float] | None) -> Excess | None:
    """Whether a swing is implausibly large: past the whole range of the data, or (multiplicative) past the trend itself.

    A seasonal swing like that is rarely real: another part of the model (a
    regressor, or the trend of a multiplicative fit) is usually cancelling it.
    """
    if fact.flat:
        return None
    if observed is not None and fact.swing_y > observed[1] - observed[0]:
        return "range"
    if fact.multiplicative and fact.swing > 1.0:
        return "trend"
    return None


#: A regressor offsets a seasonal pattern when their contributions correlate at least this negatively ...
OFFSET_CORRELATION = -0.8
#: ... and the regressor moves the fit at least this share as much as the pattern does.
OFFSET_SIZE = 0.5


def offsetting_regressors(m: Prophet, rows: pd.DataFrame, name: str) -> list[str]:
    """Regressors whose contribution largely cancels the seasonal pattern ``name`` over ``rows``.

    Both contributions are taken in the units of the data (see :func:`_effect`).
    """
    if len(rows) < 3 or name not in rows:
        return []
    own = _effect(m, rows, (name,))[0].to_numpy(dtype=float)
    spread = float(own.std())
    if not np.isfinite(spread) or spread <= 0:
        return []
    found = []
    for regressor in m.extra_regressors:
        if regressor not in rows:
            continue
        other = _effect(m, rows, (regressor,))[0].to_numpy(dtype=float)
        size = float(other.std())
        if not np.isfinite(size) or size < OFFSET_SIZE * spread:
            continue
        r = float(np.corrcoef(own, other)[0, 1])
        if np.isfinite(r) and r <= OFFSET_CORRELATION:
            found.append(regressor)
    return found


def swing_caution(
    fact: Fact,
    excess: Excess | None,
    observed: tuple[float, float] | None,
    offsets: Sequence[str],
) -> str:
    """Why a pattern should not be read on its own: its swing is implausibly large, or a regressor cancels it.

    Empty when neither holds.
    """
    names = " and ".join(r.replace("_", " ") for r in offsets)
    noun = "regressor" if len(offsets) == 1 else "regressors"
    subject = f"The {_lower_first(fact.label)} pattern"
    if excess is None:
        return (
            f"{subject} is largely offset by the {names} {noun}, so read the two together."
            if offsets
            else ""
        )
    if excess == "range" and observed is not None:
        low, high = _fmt_number(observed[0]), _fmt_number(observed[1])
        lead = f"{subject} swings by more than the whole range of the data ({low} to {high})"
    else:
        lead = f"{subject} swings by {_swing_text(fact.swing, fact.multiplicative)}, more than the trend itself"
    if offsets:
        return (
            f"{lead}; the {names} {noun} largely offsets it, so read the two together."
        )
    return f"{lead}. Another part of the model is probably cancelling it out, so read the components together."


def _seasonal_values(m: Prophet, grid: pd.DatetimeIndex, name: str) -> np.ndarray:
    components = m.predict_seasonal_components(seasonality_plot_df(m, grid))
    return components[name].to_numpy(dtype=float)


def seasonal_profile(
    m: Prophet,
    name: str,
    *,
    intraday: bool,
    days: frozenset[int],
    hours: frozenset[int],
) -> Profile | None:
    """The seasonal curve of ``name`` on a small grid.

    ``days`` and ``hours`` are the weekdays and hours of the day the data has;
    the rest are left out, because a seasonal curve is unconstrained where there
    is no data (the weekly curve of a weekdays-only series is meaningless on a
    Sunday). Weekly curves are averaged per weekday (over the hours kept when
    the data is intraday), daily curves are read per hour, yearly curves per day
    of the year and any other period at ``CYCLE_POINTS`` points across one cycle.
    """
    period = float(m.seasonalities[name]["period"])
    kind = seasonality_kind(period)
    if kind == "weekly":
        grid = (
            pd.date_range(_BASE_MONDAY, periods=168, freq=_HOUR)
            if intraday
            else pd.date_range(_BASE_MONDAY, periods=7, freq=pd.Timedelta(days=1))
        )
        mask = np.isin(np.asarray(grid.dayofweek), sorted(days))
        if intraday:
            mask &= np.isin(np.asarray(grid.hour), sorted(hours))
        grid = grid[mask]
        if not len(grid):
            return None
        by_day = (
            pd.Series(_seasonal_values(m, grid, name))
            .groupby(np.asarray(grid.dayofweek))
            .mean()
        )
        return Profile(
            kind,
            tuple(calendar.day_name[int(d)] for d in by_day.index),
            tuple(float(v) for v in by_day),
        )
    if kind == "daily":
        kept = [h for h in range(24) if h in hours]
        if not kept:
            return None
        grid = pd.DatetimeIndex([_BASE_YEAR + h * _HOUR for h in kept])
        values = _seasonal_values(m, grid, name)
        return Profile(
            kind, tuple(f"{h:02d}:00" for h in kept), tuple(float(v) for v in values)
        )
    if kind == "yearly":
        year = pd.date_range(_BASE_YEAR, periods=365, freq=pd.Timedelta(days=1))
        values = _seasonal_values(m, year, name)
        return Profile(
            kind,
            tuple(f"{d.day} {d:%B}" for d in year),
            tuple(float(v) for v in values),
        )
    offsets = np.arange(CYCLE_POINTS) * period / CYCLE_POINTS
    grid = pd.DatetimeIndex(_BASE_YEAR + pd.to_timedelta(offsets, unit="D"))
    values = _seasonal_values(m, grid, name)
    return Profile(
        kind,
        tuple(_into_cycle(float(o)) for o in offsets),
        tuple(float(v) for v in values),
    )


def observed_calendar(
    ds: pd.Series, intraday: bool
) -> tuple[frozenset[int], frozenset[int]]:
    """The weekdays, and for intraday data the hours of the day, that occur in ``ds``."""
    days = frozenset(int(d) for d in ds.dt.dayofweek.unique())
    hours = (
        frozenset(int(h) for h in ds.dt.hour.unique())
        if intraday
        else frozenset(range(24))
    )
    return days, hours


def seasonal_facts(
    m: Prophet,
    fcst: pd.DataFrame,
    history_end: pd.Timestamp,
    step: pd.Timedelta,
    days: frozenset[int],
    hours: frozenset[int],
) -> tuple[Fact, ...]:
    """A :class:`Fact` per seasonality, the strongest swing first and flat patterns last."""
    scale = _data_spread(m)
    rows, _ = basis_rows(fcst, history_end)
    level = float(rows["trend"].abs().mean()) if len(rows) and "trend" in rows else 0.0
    if not np.isfinite(level) or level <= 0:
        level = 1.0
    facts: list[Fact] = []
    for name, props in m.seasonalities.items():
        profile = seasonal_profile(
            m, name, intraday=step < pd.Timedelta(days=1), days=days, hours=hours
        )
        if profile is None:
            continue
        label = _pretty(name)
        if profile.kind == "cycle":
            label += f" ({_cycle_words(float(props['period']))} cycle)"
        facts.append(
            fact_from_profile(
                profile,
                name=name,
                label=label,
                multiplicative=_is_multiplicative(m, name),
                scale=scale,
                level=level,
            )
        )
    return tuple(sorted(facts, key=lambda f: (f.flat, -f.swing_y)))


def fact_rows(facts: Sequence[Fact], shares: Mapping[str, float]) -> list[list[str]]:
    """Table rows: pattern, highest, lowest, swing and, when known, the share of the movement.

    The share comes last so that the two numeric columns sit side by side on the right.
    """
    rows: list[list[str]] = []
    for f in facts:
        high = (
            "No clear peak"
            if f.flat
            else f"{f.high} ({_signed(f.high_value, f.multiplicative)})"
        )
        low = (
            "No clear low"
            if f.flat
            else f"{f.low} ({_signed(f.low_value, f.multiplicative)})"
        )
        row = [f.label, high, low, _swing_text(f.swing, f.multiplicative)]
        if shares:
            row.append(_fmt_pct(shares[f.name]) if f.name in shares else "0%")
        rows.append(row)
    return rows


def summary_sentence(
    items: Sequence[Importance], strongest: Fact | None, excess: Excess | None = None
) -> str:
    """One short sentence for the key findings: what carries the movement and where the strongest pattern peaks.

    ``excess`` says why the strongest swing is implausibly large (see
    :func:`swing_excess`), which usually means another part of the model offsets it.
    """
    head = _short_lead(items[0]) if len(items) >= 2 else ""
    if strongest is None:
        return f"{head}." if head else ""
    swing = _swing_text(strongest.swing, strongest.multiplicative)
    extra = f", {EXCESS_PHRASE[excess]}" if excess is not None else ""
    peak = f"peaking {strongest.peak} (a swing of {swing}{extra})"
    if not head:
        return f"The strongest seasonal pattern is {_lower_first(strongest.label)}, {peak}."
    if items[0].part == "seasonality" and strongest.name == items[0].name:
        return f"{head}, {peak}."
    return f"{head}; the strongest seasonal pattern is {_lower_first(strongest.label)}, {peak}."


def _short_lead(top: Importance) -> str:
    """'The trend carries most of the movement (73%)': the lead clause without where it was measured."""
    amount = "most" if top.share > 0.5 else "the largest share"
    return f"{top.noun} {top.verb} {amount} of the movement ({_fmt_pct(top.share)})"


def lede_text(
    items: Sequence[Importance],
    facts: Sequence[Fact],
    in_sample: bool,
    has_charts: bool,
    caution: str = "",
) -> str:
    """The opening of the section: what matters most, then the strongest seasonal swing.

    ``caution`` is :func:`swing_caution` of the strongest swing, when it needs one.
    """
    parts = [importance_sentence(items, in_sample)]
    strongest = strongest_fact(facts)
    if strongest is not None:
        parts.append(
            f"The strongest seasonal pattern is {_lower_first(strongest.label)}: {strongest.detail}."
        )
        parts.append(caution)
    elif facts:
        parts.append("Every seasonal pattern the model fitted is close to flat.")
    else:
        parts.append(
            "The model fitted no seasonal pattern, so the charts below show how the observed data "
            "behave across the calendar instead."
            if has_charts
            else "The model fitted no seasonal pattern."
        )
    return " ".join(p for p in parts if p)


# ----------------------------------------------------------------------------
# Pure: observed averages by weekday and hour, or by month and weekday
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """Averages of the observed values in a table of cells."""

    kind: Literal["weekday_hour", "month_weekday"]
    #: Weekday numbers (Monday is 0).
    row_ids: tuple[int, ...]
    #: Hours (0 to 23) or months (0 for January).
    col_ids: tuple[int, ...]
    mean: np.ndarray
    count: np.ndarray

    @property
    def rows(self) -> tuple[str, ...]:
        return tuple(calendar.day_name[d] for d in self.row_ids)

    @property
    def cols(self) -> tuple[str, ...]:
        if self.kind == "weekday_hour":
            return tuple(f"{h:02d}:00" for h in self.col_ids)
        return tuple(calendar.month_abbr[c + 1] for c in self.col_ids)

    def cell(self, i: int, j: int) -> str:
        """Plain-English name of one cell: 'Friday at 14:00', 'Fridays in March'."""
        day = calendar.day_name[self.row_ids[i]]
        if self.kind == "weekday_hour":
            return f"{day} at {self.col_ids[j]:02d}:00"
        return f"{day}s in {calendar.month_name[self.col_ids[j] + 1]}"


def _tabulate(
    keys: tuple[np.ndarray, np.ndarray],
    y: np.ndarray,
    row_ids: Sequence[int],
    col_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of ``y`` in each (row, column) cell; empty cells are NaN and 0."""
    frame = pd.DataFrame({"r": keys[0], "c": keys[1], "y": y})
    stats = frame.groupby(["r", "c"])["y"].agg(["mean", "size"])
    mean = (
        stats["mean"].unstack("c").reindex(index=list(row_ids), columns=list(col_ids))
    )
    count = (
        stats["size"]
        .unstack("c")
        .reindex(index=list(row_ids), columns=list(col_ids))
        .fillna(0)
    )
    return mean.to_numpy(dtype=float), count.to_numpy(dtype=float)


def observed_grid(
    obs: pd.DataFrame, step: pd.Timedelta, skip: SkipAxis | None
) -> Grid | None:
    """Observed averages by weekday and hour (intraday data) or by month and weekday (daily data).

    Needs columns ``ds`` and ``y``. Weekly or coarser data, daily data
    spanning under ten months and intraday data spanning under two weeks (each
    cell would hold only a handful of observations, which is the series itself
    restated) have no grid. Skipped weekdays and hours, and weekdays or hours that never occur,
    are left out.
    """
    if obs.empty:
        return None
    skip_days = skip.skip_days if skip is not None else frozenset()
    skip_hours = skip.skip_hours if skip is not None else frozenset()
    ds = obs["ds"]
    y = obs["y"].to_numpy(dtype=float)
    weekdays = [d for d in range(7) if d not in skip_days]
    if step <= pd.Timedelta(hours=6):
        if ds.max() - ds.min() < pd.Timedelta(days=MIN_GRID_DAYS):
            return None
        hours = [h for h in range(24) if h not in skip_hours]
        mean, count = _tabulate(
            (ds.dt.dayofweek.to_numpy(), ds.dt.hour.to_numpy()), y, weekdays, hours
        )
        kind: Literal["weekday_hour", "month_weekday"] = "weekday_hour"
        cols = hours
        col_idx = [j for j, n in enumerate(count.sum(axis=0)) if n > 0]
    elif _step_unit(step) == "day" and ds.max() - ds.min() >= pd.Timedelta(
        days=MIN_HEATMAP_DAYS
    ):
        # Months run along the axis and weekdays down it, so rows are weekdays here too.
        months = list(range(12))
        mean, count = _tabulate(
            (ds.dt.dayofweek.to_numpy(), ds.dt.month.to_numpy() - 1),
            y,
            weekdays,
            months,
        )
        kind = "month_weekday"
        cols = months
        col_idx = [j for j, n in enumerate(count.sum(axis=0)) if n > 0]
    else:
        return None
    row_idx = [i for i, n in enumerate(count.sum(axis=1)) if n > 0]
    if len(row_idx) < 2 or len(col_idx) < 3:
        return None
    return Grid(
        kind=kind,
        row_ids=tuple(weekdays[i] for i in row_idx),
        col_ids=tuple(cols[j] for j in col_idx),
        mean=mean[np.ix_(row_idx, col_idx)],
        count=count[np.ix_(row_idx, col_idx)],
    )


def grid_extremes_sentence(grid: Grid) -> str:
    """The best and worst cell with the number of observations behind each."""
    if not np.isfinite(grid.mean).any():
        return ""
    if np.isclose(np.nanmax(grid.mean), np.nanmin(grid.mean), rtol=1e-9, atol=0.0):
        return "Every cell has the same average."
    hi = np.unravel_index(int(np.nanargmax(grid.mean)), grid.mean.shape)
    lo = np.unravel_index(int(np.nanargmin(grid.mean)), grid.mean.shape)

    def describe(idx: tuple[int, ...]) -> str:
        i, j = int(idx[0]), int(idx[1])
        n = int(grid.count[i, j])
        return f"{grid.cell(i, j)} ({_fmt_number(float(grid.mean[i, j]))}, from {_plural(n, 'observation')})"

    return f"The highest average is {describe(hi)} and the lowest is {describe(lo)}."


# ----------------------------------------------------------------------------
# Pure: daily values and the calendar
# ----------------------------------------------------------------------------


def _day_means(ds: pd.Series, values: pd.Series) -> pd.DataFrame:
    """Mean per calendar day and how many rows it averages: columns ``day``, ``value``, ``n``."""
    frame = pd.DataFrame(
        {"day": ds.dt.normalize().to_numpy(), "value": values.to_numpy(dtype=float)}
    )
    out = frame.groupby("day")["value"].agg(["mean", "size"]).reset_index()
    return out.rename(columns={"mean": "value", "size": "n"})


def daily_values(
    obs: pd.DataFrame, fcst: pd.DataFrame, step: pd.Timedelta
) -> pd.DataFrame | None:
    """Observed days from the data and forecast days from ``yhat``: columns ``day``, ``value``, ``forecast``.

    Intraday rows are averaged per day, and a day with well under the usual
    number of rows (the day the history stops part-way) is left out. Forecast
    days are those after the last observed day. Forecast rows on a weekday the
    data never has, or (for intraday data) at an hour it never has, are left
    out: the seasonal curves are unconstrained there, so those values would be
    extrapolations. None for data spaced more than about a day apart or when
    nothing remains.
    """
    if obs.empty or step >= pd.Timedelta(days=1.5):
        return None
    observed = _day_means(obs["ds"], obs["y"])
    usual = float(observed["n"].median())
    observed = observed[observed["n"] >= COMPLETE_DAY * usual]
    last_day = obs["ds"].max().normalize()
    ahead = fcst[(fcst["ds"].dt.normalize() > last_day)].dropna(subset=["yhat"])
    ahead = ahead[ahead["ds"].dt.dayofweek.isin(sorted(set(obs["ds"].dt.dayofweek)))]
    if step < pd.Timedelta(days=1):
        ahead = ahead[ahead["ds"].dt.hour.isin(sorted(set(obs["ds"].dt.hour)))]
    forecast = (
        _day_means(ahead["ds"], ahead["yhat"])
        if len(ahead)
        else pd.DataFrame({"day": [], "value": [], "n": []})
    )
    forecast = forecast[forecast["n"] >= COMPLETE_DAY * usual]
    days = pd.concat(
        [observed.assign(forecast=False), forecast.assign(forecast=True)],
        ignore_index=True,
    )
    if days.empty:
        return None
    days["day"] = pd.to_datetime(days["day"])
    days["forecast"] = days["forecast"].astype(bool)
    return days.sort_values("day")[["day", "value", "forecast"]].reset_index(drop=True)


@dataclass(frozen=True)
class CalendarBlock:
    """One calendar year inside the stacked matrix."""

    year: int
    #: First row of the block.
    top: int
    #: Month abbreviation and the column its first day falls in.
    months: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Calendar:
    """Daily values laid out as a calendar: one block per year, weeks across, weekdays down."""

    weekdays: tuple[int, ...]
    n_cols: int
    blocks: tuple[CalendarBlock, ...]
    observed: np.ndarray
    forecast: np.ndarray
    text_observed: tuple[tuple[str, ...], ...]
    text_forecast: tuple[tuple[str, ...], ...]
    #: Vertices of the line between observed and forecast days, or None.
    cut: tuple[tuple[float, float], ...] | None

    @property
    def n_rows(self) -> int:
        return self.observed.shape[0]

    @property
    def has_forecast(self) -> bool:
        return bool(np.isfinite(self.forecast).any())


def _week_start(year: int) -> pd.Timestamp:
    """The Monday on or before 1 January: where column 0 of a year's block starts."""
    first = pd.Timestamp(year=year, month=1, day=1)
    return first - pd.Timedelta(days=first.dayofweek)


def stair_cut(
    col: int, row: int, n_rows: int, top: int = 0
) -> tuple[tuple[float, float], ...]:
    """The line separating cells up to (col, row) from the cells after it in reading order.

    Cells run down each week column; the cut goes down the right edge of the
    last full column part, steps left under the last observed cell and down
    the left edge of its column. Coordinates are cell centres, so edges lie
    half a cell out.
    """
    points = [(col + 0.5, top - 0.5), (col + 0.5, top + row + 0.5)]
    if row < n_rows - 1:
        points += [(col - 0.5, top + row + 0.5), (col - 0.5, top + n_rows - 0.5)]
    return tuple(points)


def calendar_layout(
    days: pd.DataFrame,
    skip_days: frozenset[int] = frozenset(),
    max_years: int = MAX_CALENDAR_YEARS,
) -> Calendar | None:
    """Lay daily values out as stacked yearly blocks of weeks across and weekdays down.

    ``days`` has columns ``day``, ``value`` and ``forecast`` (see
    :func:`daily_values`). The last ``max_years`` calendar years that have days
    are kept. Weekdays in ``skip_days``, and weekdays that never occur, get
    no row. Returns None for fewer than ``MIN_CALENDAR_DAYS`` days.
    """
    if len(days) < MIN_CALENDAR_DAYS:
        return None
    years = sorted(int(y) for y in days["day"].dt.year.unique())[-max_years:]
    days = days[days["day"].dt.year.isin(years)]
    if len(days) < MIN_CALENDAR_DAYS:
        return None
    weekdays = tuple(
        sorted({int(d) for d in days["day"].dt.dayofweek} - set(skip_days))
    )
    if not weekdays:
        return None
    row_of = {d: i for i, d in enumerate(weekdays)}
    n_rows = len(weekdays)
    stride = n_rows + 1
    height = len(years) * stride - 1
    starts = {y: _week_start(y) for y in years}
    observed = np.full((height, MAX_WEEKS), np.nan)
    forecast = np.full((height, MAX_WEEKS), np.nan)
    text_obs = [[""] * MAX_WEEKS for _ in range(height)]
    text_fc = [[""] * MAX_WEEKS for _ in range(height)]
    widest = 0
    seen_months: dict[int, set[int]] = {y: set() for y in years}
    for day, value, is_forecast in zip(days["day"], days["value"], days["forecast"]):
        if day.dayofweek not in row_of:
            continue
        r = years.index(day.year) * stride + row_of[day.dayofweek]
        c = (day - starts[day.year]).days // 7
        widest = max(widest, c)
        seen_months[day.year].add(day.month)
        if is_forecast:
            forecast[r, c] = value
            text_fc[r][c] = f"{day:%a} {day.day} {day:%b %Y} (forecast)"
        else:
            observed[r, c] = value
            text_obs[r][c] = f"{day:%a} {day.day} {day:%b %Y}"
    n_cols = widest + 1
    blocks = tuple(
        CalendarBlock(
            year=y,
            top=k * stride,
            months=tuple(
                (
                    calendar.month_abbr[month],
                    (pd.Timestamp(year=y, month=month, day=1) - starts[y]).days // 7,
                )
                for month in sorted(seen_months[y])
            ),
        )
        for k, y in enumerate(years)
    )
    cut: tuple[tuple[float, float], ...] | None = None
    ahead = days[days["forecast"]]
    seen = days[~days["forecast"]]
    if len(ahead) and len(seen):
        last = seen["day"].max()
        if last.year == ahead["day"].min().year and last.dayofweek in row_of:
            cut = stair_cut(
                (last - starts[last.year]).days // 7,
                row_of[last.dayofweek],
                n_rows,
                years.index(last.year) * stride,
            )
    return Calendar(
        weekdays=weekdays,
        n_cols=n_cols,
        blocks=blocks,
        observed=observed[:, :n_cols],
        forecast=forecast[:, :n_cols],
        text_observed=tuple(tuple(row[:n_cols]) for row in text_obs),
        text_forecast=tuple(tuple(row[:n_cols]) for row in text_fc),
        cut=cut,
    )


def colour_limits(values: np.ndarray) -> tuple[float, float]:
    """Colour range that ignores the most extreme days, so that a spike does not wash out the rest."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size >= 50:
        low, high = (float(q) for q in np.percentile(finite, [2, 98]))
    else:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        low, high = low - 0.5, high + 0.5
    return low, high


# ----------------------------------------------------------------------------
# Imperative shell: figures
# ----------------------------------------------------------------------------


def observed_frame(ctx: DashboardContext) -> pd.DataFrame:
    """The observed rows (columns ``ds`` and ``y``): the caller's data if given, else the training history."""
    base = (
        ctx.df
        if ctx.df is not None and {"ds", "y"} <= set(ctx.df.columns)
        else ctx.history
    )
    obs = pd.DataFrame(
        {"ds": pd.to_datetime(base["ds"]), "y": base["y"].astype(float)}
    ).dropna()
    if ctx.skip is not None:
        obs = obs[ctx.skip.keeps(obs["ds"])]
    return obs.sort_values("ds").reset_index(drop=True)


def _rounded(values: np.ndarray) -> list[list[float | None]]:
    """Nested lists with NaN as None and six significant digits, to keep the embedded data small."""
    return [[None if np.isnan(v) else float(f"{v:.6g}") for v in row] for row in values]


def _heat_axes() -> tuple[dict[str, Any], dict[str, Any]]:
    x_style, y_style = _style.axes()
    return (
        {**x_style, "showline": False, "ticks": "", "fixedrange": True},
        {**y_style, "showgrid": False, "fixedrange": True},
    )


def _colorbar() -> dict[str, Any]:
    return {
        "thickness": 10,
        "outlinewidth": 0,
        "ticks": "",
        "tickfont": {"family": _style.SANS, "size": 11, "color": INK_SOFT},
    }


def _annotation(text: str, **options: Any) -> dict[str, Any]:
    return {
        "text": text,
        "showarrow": False,
        "font": {
            "family": _style.SANS,
            "size": options.pop("size", 12),
            "color": options.pop("color", INK_SOFT),
        },
        **options,
    }


def importance_figure(ctx: DashboardContext, items: Sequence[Importance]) -> Any:
    """Horizontal bars of each component's share, largest on top, in one colour."""
    shown = list(items)[:MAX_BARS]
    peak = max(i.share for i in shown)
    x_style, y_style = _style.axes()
    fig = ctx.go.Figure(
        ctx.go.Bar(
            x=[i.share for i in shown],
            y=[i.label for i in shown],
            orientation="h",
            marker={"color": _palette.SERIES[0]},
            text=[_fmt_pct(i.share) for i in shown],
            textposition="outside",
            textfont={"family": _style.SANS, "size": 12, "color": INK},
            cliponaxis=False,
            customdata=[_fmt_number(i.spread) for i in shown],
            hovertemplate=(
                "%{y}<br>Share of the movement <b>%{x:.1%}</b>"
                "<br>Typical size of its effect %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_style.layout(
            height=70 + 38 * len(shown),
            margin={"l": 24, "r": 48, "t": 12, "b": 36},
            showlegend=False,
            bargap=0.4,
            xaxis={
                **y_style,
                "tickformat": ".0%",
                "range": [0, peak * 1.2],
                "fixedrange": True,
            },
            yaxis={
                **x_style,
                "type": "category",
                "autorange": "reversed",
                "automargin": True,
                "ticks": "",
                "fixedrange": True,
            },
        )
    )
    return fig


def grid_figure(ctx: DashboardContext, grid: Grid) -> Any:
    """The observed averages as a heatmap: weekdays down, hours or months across."""
    x_axis, y_axis = _heat_axes()
    joiner = "at" if grid.kind == "weekday_hour" else "in"
    fig = ctx.go.Figure(
        ctx.go.Heatmap(
            z=_rounded(grid.mean),
            x=list(grid.cols),
            y=list(grid.rows),
            customdata=grid.count.astype(int).tolist(),
            colorscale=_palette.colorscale(_palette.SEQUENTIAL),
            xgap=2,
            ygap=2,
            hoverongaps=False,
            colorbar=_colorbar(),
            hovertemplate=(
                f"%{{y}} {joiner} %{{x}}<br>Average <b>%{{z:,.4g}}</b>"
                "<br>Observations in this cell: %{customdata:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_style.layout(
            height=96 + 34 * len(grid.rows),
            margin={"l": 24, "r": 16, "t": 12, "b": 40},
            xaxis={
                **x_axis,
                "type": "category",
                "categoryorder": "array",
                "categoryarray": list(grid.cols),
                "dtick": 2 if len(grid.cols) > 12 else 1,
            },
            yaxis={
                **y_axis,
                "type": "category",
                "categoryorder": "array",
                "categoryarray": list(grid.rows),
                "autorange": "reversed",
                "automargin": True,
            },
        )
    )
    return fig


def calendar_figure(ctx: DashboardContext, cal: Calendar) -> Any:
    """Yearly blocks of daily values on one shared colour scale; forecast days are smaller squares."""
    go = ctx.go
    x_axis, y_axis = _heat_axes()
    low, high = colour_limits(
        np.concatenate([cal.observed.ravel(), cal.forecast.ravel()])
    )
    shared: dict[str, Any] = {
        "x": list(range(cal.n_cols)),
        "y": list(range(cal.n_rows)),
        "coloraxis": "coloraxis",
        "hoverongaps": False,
        "hovertemplate": "%{text}<br><b>%{z:,.4g}</b><extra></extra>",
    }
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=_rounded(cal.observed),
            text=[list(r) for r in cal.text_observed],
            xgap=2,
            ygap=2,
            **shared,
        )
    )
    if cal.has_forecast:
        fig.add_trace(
            go.Heatmap(
                z=_rounded(cal.forecast),
                text=[list(r) for r in cal.text_forecast],
                xgap=6,
                ygap=6,
                **shared,
            )
        )
    if cal.cut is not None:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in cal.cut],
                y=[p[1] for p in cal.cut],
                mode="lines",
                name="Forecast starts",
                line={"color": INK_SOFT, "width": 2, "dash": "dash"},
                hovertemplate="Forecast days follow this line<extra></extra>",
            )
        )
    names = [calendar.day_abbr[d] for d in cal.weekdays]
    annotations = [
        _annotation(
            str(b.year),
            xref="paper",
            x=0,
            xanchor="right",
            xshift=-46,
            yref="y",
            y=b.top + (len(cal.weekdays) - 1) / 2,
            color=INK,
            size=13,
        )
        for b in cal.blocks
    ] + [
        _annotation(
            month,
            xref="x",
            x=col,
            xanchor="left",
            yref="y",
            # Just inside the axis range: plotly hides annotations anchored outside it.
            y=b.top - 0.48,
            yanchor="bottom",
            yshift=2,
            size=11,
        )
        for b in cal.blocks
        for month, col in b.months
    ]
    fig.update_layout(
        **_style.layout(
            height=_CAL_TOP + _CAL_BOTTOM + _CAL_ROW * cal.n_rows,
            margin={"l": 92, "r": 16, "t": _CAL_TOP, "b": _CAL_BOTTOM},
            showlegend=cal.cut is not None,
            # Above the month labels of the first block, which sit just over the plot.
            legend={
                "orientation": "h",
                "x": 1,
                "xanchor": "right",
                "y": 1 + 26 / (_CAL_ROW * cal.n_rows),
                "yanchor": "bottom",
                "font": {"family": _style.SANS, "size": 12},
            },
            annotations=annotations,
            coloraxis={
                "colorscale": _palette.colorscale(_palette.SEQUENTIAL),
                "cmin": low,
                "cmax": high,
                "colorbar": _colorbar(),
            },
            xaxis={
                **x_axis,
                "range": [-0.5, cal.n_cols - 0.5],
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
            },
            yaxis={
                **y_axis,
                "range": [cal.n_rows - 0.5, -0.5],
                "tickmode": "array",
                "tickvals": [b.top + i for b in cal.blocks for i in range(len(names))],
                "ticktext": names * len(cal.blocks),
                "zeroline": False,
            },
        )
    )
    return fig


# ----------------------------------------------------------------------------
# Imperative shell: the section
# ----------------------------------------------------------------------------


def _importance_note(in_sample: bool) -> str:
    basis = "the fitted values" if in_sample else "the forecast"
    return (
        f"Each bar is how far a component typically moves {basis} from their average (a standard deviation, in the "
        "units of the data) as a share of all the components together. Multiplicative effects are scaled by the "
        "trend first. Components overlap a little, so read the shares as a guide, not an exact split."
    )


def _grid_title(grid: Grid) -> str:
    return (
        "Average by weekday and hour"
        if grid.kind == "weekday_hour"
        else "Average by month and weekday"
    )


def _grid_note(grid: Grid, has_table: bool) -> str:
    unit = (
        "weekday and hour of the day"
        if grid.kind == "weekday_hour"
        else "weekday and month"
    )
    table = (
        "The table above shows the seasonal effect with the trend taken out. "
        if has_table
        else ""
    )
    pulled = "weeks" if grid.kind == "weekday_hour" else "years or weeks"
    return (
        f"Each cell is the plain average of the observed values for that {unit}, so it also carries the trend: "
        f"if the series rose or fell over the period, the busier {pulled} pull their cells up or down. {table}"
        + grid_extremes_sentence(grid)
    ).strip()


def _calendar_note(cal: Calendar, intraday: bool) -> str:
    daily = "Intraday data are averaged per day. " if intraday else ""
    ahead = (
        "Days after the dashed line are forecast and drawn as smaller squares. "
        if cal.cut is not None
        else (
            "Forecast days are drawn as smaller squares. " if cal.has_forecast else ""
        )
    )
    return (
        "One block per year, one column per week with Monday first. "
        f"{daily}{ahead}Hover for the date and the value."
    )


def _scrolling(figure_html: str) -> str:
    """Wrap a wide figure so that it scrolls sideways on a narrow screen instead of squeezing its cells to slivers."""
    return (
        f'<div class="tbl-wrap"><div style="min-width:{CALENDAR_MIN_WIDTH}px">'
        f"{figure_html}</div></div>"
    )


def build(ctx: DashboardContext) -> Section | None:
    """The section, or None when it does not apply to this model."""
    m = ctx.m
    intraday = ctx.step < pd.Timedelta(days=1)
    items = component_importance(m, ctx.fcst, ctx.history_end)
    _, in_sample = basis_rows(ctx.fcst, ctx.history_end)
    days_seen, hours_seen = observed_calendar(ctx.history["ds"], intraday)
    facts = seasonal_facts(
        m, ctx.fcst, ctx.history_end, ctx.step, days_seen, hours_seen
    )
    obs = observed_frame(ctx)
    grid = observed_grid(obs, ctx.step, ctx.skip)
    days = daily_values(obs, ctx.fcst, ctx.step)
    cal = (
        calendar_layout(
            days, ctx.skip.skip_days if ctx.skip is not None else frozenset()
        )
        if days is not None
        else None
    )
    has_charts = grid is not None or cal is not None
    if not facts and not has_charts:
        return None

    observed = observed_range(m)
    basis, _ = basis_rows(ctx.fcst, ctx.history_end)
    excesses = {f.name: swing_excess(f, observed) for f in facts}
    cautions = {
        f.name: text
        for f in facts
        if (
            text := swing_caution(
                f,
                excesses[f.name],
                observed,
                offsetting_regressors(m, basis, f.name) if not f.flat else [],
            )
        )
    }
    strongest = strongest_fact(facts)
    lead_caution = cautions.get(strongest.name, "") if strongest is not None else ""
    other_cautions = [
        text
        for name, text in cautions.items()
        if strongest is None or name != strongest.name
    ]

    parts = [_style.lede(lede_text(items, facts, in_sample, has_charts, lead_caution))]
    if len(items) >= 2:
        parts += [
            _style.h3(
                "What moves the fitted values"
                if in_sample
                else "What moves the forecast"
            ),
            ctx.embed(importance_figure(ctx, items)),
            _style.note(_importance_note(in_sample)),
        ]
    if facts:
        shares = {i.name: i.share for i in items} if len(items) >= 2 else {}
        headers = ["Pattern", "Highest", "Lowest", "Swing"] + (
            ["Share of movement"] if shares else []
        )
        numeric = list(range(3, len(headers)))
        skipped = (
            missing_days_phrase(days_seen)
            if any(f.kind == "weekly" for f in facts)
            else ""
        )
        parts += [
            _style.h3("Peaks and lows"),
            _style.table(headers, fact_rows(facts, shares), numeric=numeric),
            _style.note(
                "Highest and lowest are read from each seasonal curve as the distance from its own average. Additive "
                "effects are in the units of the data and multiplicative ones are a share of the trend. "
                + " ".join(part for part in (skipped, *other_cautions) if part)
            ),
        ]
    if grid is not None:
        parts += [
            _style.h3(_grid_title(grid)),
            ctx.embed(grid_figure(ctx, grid)),
            _style.note(_grid_note(grid, bool(facts))),
        ]
    if cal is not None:
        parts += [
            _style.h3("Day by day"),
            _scrolling(ctx.embed(calendar_figure(ctx, cal))),
            _style.note(_calendar_note(cal, intraday)),
        ]
    return Section(
        key=KEY,
        title=TITLE,
        body="".join(parts),
        summary=summary_sentence(
            items, strongest, excesses.get(strongest.name) if strongest else None
        ),
    )
