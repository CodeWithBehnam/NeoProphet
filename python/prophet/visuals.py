"""Visual outputs built on top of a fitted model: backtest, regressor effects, report and dashboard.

Four outputs share the editorial default style of :mod:`prophet.plot` (warm
paper, serif headline over a sans subtitle, one claret accent for the
forecast, direct labels, plain-English headlines written from the data):

* :func:`plot_backtest` holds out the end of the data, refits a copy of the
  model on the rest and draws actuals against the forecast.
* :func:`plot_regressor_effects` shows what each extra regressor added to the
  forecast, with its fitted coefficient in plain words.
* :func:`plot_report` is a single A4 page: forecast, key numbers, components
  and regressor effects.
* :func:`dashboard_html` writes an interactive, self-contained plotly page made
  of the sections in :mod:`prophet.dashboard`, with a light and a dark theme
  (plotly is an optional dependency, imported only there).

The numbers and the wording come from small pure functions
(:func:`backtest_metrics`, :func:`backtest_headline`, :func:`regressor_sentence`
and friends); the ``plot_*`` and ``dashboard_html`` functions only draw.
Nothing here changes global matplotlib state.
"""

from __future__ import annotations

import calendar
import html
import importlib
import logging
import re
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable, Collection, Iterable
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import font_manager, patheffects
from matplotlib import pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, date2num
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import ScaledTranslation

from prophet.dashboard import _palette
from prophet.diagnostics import prophet_copy
from prophet.plot import (
    ACCENT,
    BAND_ALPHA,
    FORECAST_TINT,
    GRID,
    HISTORY_TINT,
    INK,
    INK_SOFT,
    OBSERVED,
    OBSERVED_TEXT,
    PAPER,
    SkipAxis,
    _chart_axis,
    _midpoint,
    _skip_note,
    plot,
    plot_forecast_component,
    plot_seasonality,
    seasonality_plot_df,
    plot_weekly,
    plot_yearly,
    set_y_as_percent,
)
from prophet.utilities import regressor_coefficients

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.transforms import Bbox, Transform

    from prophet.dashboard._context import DashboardContext, Section
    from prophet.forecaster import Prophet

logger: logging.Logger = logging.getLogger("prophet.visuals")

#: Weekdays or hours to leave out: names, numbers, "weekends" or a collection of these (see ``skip_axis``).
SkipSpec = str | int | Iterable[str | int] | None

__all__ = [
    "Backtest",
    "BacktestError",
    "BacktestMetrics",
    "NoRegressorsError",
    "RegressorEffect",
    "SourceRun",
    "Tile",
    "backtest_headline",
    "backtest_metrics",
    "dashboard_html",
    "describe_periods",
    "plot_backtest",
    "plot_regressor_effects",
    "plot_report",
    "regressor_effects",
    "regressor_sentence",
    "run_backtest",
    "split_holdout",
    "weather_ahead_note",
    "weather_source_phrase",
    "weather_source_runs",
]


class BacktestError(ValueError):
    """The data, the horizon or the model cannot support a backtest."""


class NoRegressorsError(ValueError):
    """A regressor chart was requested for a model without extra regressors."""


# ----------------------------------------------------------------------------
# Style (palette comes from prophet.plot; fonts and sizes are local)
# ----------------------------------------------------------------------------

_SANS_STACK = ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
_SERIF_STACK = ("Georgia", "Charter", "Times New Roman", "DejaVu Serif")
_CSS_SANS = '"Helvetica Neue", Helvetica, Arial, "DejaVu Sans", sans-serif'
_CSS_SERIF = 'Georgia, Charter, "Times New Roman", "DejaVu Serif", serif'

_TITLE_PT = 17.0
_PANEL_TITLE_PT = 12.0
_SUBTITLE_PT = 10.0
_TICK_PT = 9.5
_LABEL_PT = 10.0
_NOTE_PT = 9.0
_SOURCE_PT = 8.5
_KICKER_PT = 8.5

_LEFT_IN = 0.9
_RIGHT_IN = 0.45
_TEXT_LEFT_IN = 0.4

# Matplotlib draws a lone point as a dot, so single-valued series are still visible.
_WEATHER_LABELS: Mapping[str, str] = {
    "archive": "Observed weather",
    "forecast": "Weather forecast",
    "climatology": "Typical season (climatology)",
}
_DEFAULT_UNITS: Mapping[str, str] = {"temperature": "°C"}


def _blend(foreground: str, background: str, share: float) -> str:
    fg, bg = mcolors.to_rgb(foreground), mcolors.to_rgb(background)
    r, g, b = (share * f + (1 - share) * k for f, k in zip(fg, bg))
    return mcolors.to_hex((r, g, b))


_CLIMATOLOGY_FILL = _blend(ACCENT, PAPER, 0.3)
_WEATHER_FILL: Mapping[str, tuple[str, str]] = {
    "archive": (OBSERVED_TEXT, "white"),
    "forecast": (ACCENT, "white"),
    "climatology": (_CLIMATOLOGY_FILL, INK),
}


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
# Pure helpers: numbers and plain-English wording
# ----------------------------------------------------------------------------


def _fmt_number(v: float) -> str:
    """Compact, unit-free number: 8.52, 1,235, 12.3K, 1.2M."""
    a = abs(v)
    if not np.isfinite(v):
        return "n/a"
    if a >= 1e9:
        return f"{v / 1e9:.3g}B"
    if a >= 1e6:
        return f"{v / 1e6:.3g}M"
    if a >= 1e4:
        return f"{v / 1e3:.3g}K"
    if a >= 100:
        return f"{v:,.0f}"
    return f"{v + 0.0:.3g}"


def _fmt_tick(v: float, pos: int | None = None) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.4g}B"
    if a >= 1e6:
        return f"{v / 1e6:.4g}M"
    if a >= 1e4:
        return f"{v / 1e3:.4g}K"
    if a >= 100:
        return f"{v:,.0f}"
    return f"{v + 0.0:.4g}"


def _fmt_pct(fraction: float) -> str:
    """Percentage of a fraction with sensible precision: 0.45%, 6.5%, 12%."""
    a = abs(fraction) * 100
    return f"{a:.0f}%" if a >= 10 else f"{a:.2g}%"


def _signed(v: float, multiplicative: bool) -> str:
    sign = "-" if v < 0 else "+"
    return f"{sign}{_fmt_pct(v)}" if multiplicative else f"{sign}{_fmt_number(abs(v))}"


def _pretty(name: str) -> str:
    text = name.replace("_", " ")
    return text[:1].upper() + text[1:]


def _day_month(ts: pd.Timestamp) -> str:
    """'5 Mar': the day without a leading zero, as the prose of the page writes it."""
    return f"{ts.day} {ts:%b}"


def _fmt_date(ts: pd.Timestamp, span: pd.Timedelta) -> str:
    """Date wording that fits the span of the data: month for years, time for hours."""
    if span >= pd.Timedelta(days=730):
        return f"{ts:%b %Y}"
    if span >= pd.Timedelta(days=60):
        return f"{_day_month(ts)} {ts:%Y}"
    return f"{_day_month(ts)} {ts:%H:%M}"


def _fmt_day(ts: pd.Timestamp, step: pd.Timedelta) -> str:
    """A point in time at the resolution of the data."""
    day = f"{_day_month(ts)} {ts:%Y}"
    return day if step >= pd.Timedelta(hours=20) else f"{day} {ts:%H:%M}"


# Calendar-ish units with their typical length, for naming a spacing or span.
_UNIT_STEPS: tuple[tuple[str, pd.Timedelta], ...] = (
    ("minute", pd.Timedelta(minutes=1)),
    ("hour", pd.Timedelta(hours=1)),
    ("day", pd.Timedelta(days=1)),
    ("week", pd.Timedelta(days=7)),
    ("month", pd.Timedelta(days=30.4375)),
    ("quarter", pd.Timedelta(days=91.3125)),
    ("year", pd.Timedelta(days=365.25)),
)


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def _duration_parts(td: pd.Timedelta) -> tuple[int, str]:
    """Round a span to a whole number of the most natural unit."""
    seconds = max(td.total_seconds(), 1.0)
    days = seconds / 86400.0
    if seconds < 2 * 3600:
        return max(1, round(seconds / 60)), "minute"
    if days < 2:
        return max(1, round(seconds / 3600)), "hour"
    if days < 14:
        return max(1, round(days)), "day"
    if days < 70:
        return max(1, round(days / 7)), "week"
    if days < 730:
        return max(1, round(days / 30.44)), "month"
    return max(1, round(days / 365.25)), "year"


def _duration_phrase(td: pd.Timedelta) -> str:
    n, unit = _duration_parts(td)
    return _plural(n, unit)


def _lead_phrase(n: int, unit: str, lead: str) -> str:
    """'the next 12 months', 'the last day'."""
    return f"{lead} {unit}" if n == 1 else f"{lead} {n} {unit}s"


def _step_unit(step: pd.Timedelta) -> str | None:
    """Name of the calendar unit a spacing matches within 6%, or None."""
    for unit, size in _UNIT_STEPS:
        if abs(step / size - 1.0) <= 0.06:
            return unit
    return None


def describe_periods(n: int, step: pd.Timedelta, lead: str = "the last") -> str:
    """Name a number of periods in the unit of their spacing: ``the last 90 days``.

    Spacings that match no calendar unit (every 5 minutes, every 3 days) are
    described by the span they cover instead.
    """
    unit = _step_unit(step)
    if unit is not None:
        return _lead_phrase(n, unit, lead)
    count, name = _duration_parts(step * n)
    return _lead_phrase(count, name, lead)


def frequency_words(step: pd.Timedelta) -> tuple[str, str]:
    """Headline and detail for the spacing of the data: ``('Daily', 'one point per day')``."""
    adjectives = {
        "minute": "Every minute",
        "hour": "Hourly",
        "day": "Daily",
        "week": "Weekly",
        "month": "Monthly",
        "quarter": "Quarterly",
        "year": "Yearly",
    }
    unit = _step_unit(step)
    if unit is not None:
        return adjectives[unit], f"one point per {unit}"
    n, name = _duration_parts(step)
    return f"Every {_plural(n, name)}", "typical spacing between points"


def _frequency_with_skips(step: pd.Timedelta, skip: SkipAxis | None) -> tuple[str, str]:
    """Like :func:`frequency_words`, but honest about skipped weekdays.

    A weekday-only series has a one-day step, so the plain wording would call
    it daily even though weekends never appear.
    """
    freq, detail = frequency_words(step)
    if skip is None or not skip.skip_days or _step_unit(step) != "day":
        return freq, detail
    if set(skip.skip_days) == {5, 6}:
        return "Weekdays", "one point per weekday"
    return freq, f"{detail} on the days kept"


def _compare_phrase(new: float, old: float, scale: float) -> str:
    """'about 12% higher than', 'about the same as', '0.9 lower than'.

    Percentages are only used when the base is clearly positive; with a base
    near zero or negative the absolute difference is the honest comparison.
    """
    diff = new - old
    if old > 0 and new >= 0 and old >= 0.2 * scale:
        rel = diff / old
        if abs(rel) < 0.03:
            return "about the same as"
        return f"about {_fmt_pct(rel)} {'lower' if rel < 0 else 'higher'} than"
    if abs(diff) < 0.03 * scale:
        return "about the same as"
    return f"{_fmt_number(abs(diff))} {'lower' if diff < 0 else 'higher'} than"


# ----------------------------------------------------------------------------
# Backtest: pure parts
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestMetrics:
    """Accuracy of a backtest.

    ``mape`` and ``coverage`` are fractions (0.065 is 6.5%). ``mape`` is None
    when the actuals include zeros or values tiny next to their average,
    where a percentage error says nothing. ``coverage`` is None when the
    forecast has no interval.
    """

    n: int
    mae: float
    rmse: float
    mape: float | None
    coverage: float | None
    interval_width: float | None
    mean_actual: float


@dataclass(frozen=True, eq=False)
class Backtest:
    """A finished backtest: the split, the refitted model's forecast and its metrics."""

    cutoff: pd.Timestamp
    span: str
    train: pd.DataFrame
    holdout: pd.DataFrame
    forecast: pd.DataFrame
    metrics: BacktestMetrics
    skip_days: tuple[int, ...] = ()
    skip_hours: tuple[int, ...] = ()


def _mape_is_meaningful(actual: np.ndarray) -> bool:
    mean = float(np.mean(np.abs(actual)))
    return bool(np.all(actual > 0) and mean > 0 and np.min(actual) >= 0.1 * mean)


def backtest_metrics(
    actual: Sequence[float] | np.ndarray | pd.Series,
    yhat: Sequence[float] | np.ndarray | pd.Series,
    lower: Sequence[float] | np.ndarray | pd.Series | None = None,
    upper: Sequence[float] | np.ndarray | pd.Series | None = None,
    *,
    interval_width: float | None = None,
) -> BacktestMetrics:
    """MAE, RMSE, MAPE (where meaningful) and interval coverage of a forecast.

    Pairs where the actual or the forecast is missing are ignored. Raises
    :class:`BacktestError` when the inputs differ in length or no pair is left.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(yhat, dtype=float)
    if a.ndim != 1 or a.shape != f.shape:
        raise BacktestError(
            f"actual and yhat must be 1-D and the same length, got {a.shape} and {f.shape}."
        )
    keep = np.isfinite(a) & np.isfinite(f)
    if not keep.any():
        raise BacktestError("No holdout point has both an actual value and a forecast.")
    err = f[keep] - a[keep]
    coverage: float | None = None
    if lower is not None and upper is not None:
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if lo.shape != a.shape or hi.shape != a.shape:
            raise BacktestError("lower and upper must be the same length as actual.")
        both = keep & np.isfinite(lo) & np.isfinite(hi)
        if both.any():
            coverage = float(np.mean((a[both] >= lo[both]) & (a[both] <= hi[both])))
    mape = (
        float(np.mean(np.abs(err) / np.abs(a[keep])))
        if _mape_is_meaningful(a[keep])
        else None
    )
    return BacktestMetrics(
        n=int(keep.sum()),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        mape=mape,
        coverage=coverage,
        interval_width=interval_width if coverage is not None else None,
        mean_actual=float(np.mean(a[keep])),
    )


_HORIZON = re.compile(r"^\s*(\d+)\s*([A-Za-z]+)\s*$")
_HORIZON_UNITS: Mapping[str, tuple[str, pd.DateOffset]] = {
    "s": ("second", pd.DateOffset(seconds=1)),
    "min": ("minute", pd.DateOffset(minutes=1)),
    "h": ("hour", pd.DateOffset(hours=1)),
    "d": ("day", pd.DateOffset(days=1)),
    "w": ("week", pd.DateOffset(weeks=1)),
    "mo": ("month", pd.DateOffset(months=1)),
    "q": ("quarter", pd.DateOffset(months=3)),
    "y": ("year", pd.DateOffset(years=1)),
}
_HORIZON_ALIASES: Mapping[str, str] = {
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "mins": "min",
    "t": "min",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "day": "d",
    "days": "d",
    "wk": "w",
    "wks": "w",
    "week": "w",
    "weeks": "w",
    "month": "mo",
    "months": "mo",
    "mos": "mo",
    "quarter": "q",
    "quarters": "q",
    "qtr": "q",
    "year": "y",
    "years": "y",
    "yr": "y",
    "yrs": "y",
}


@dataclass(frozen=True)
class _Horizon:
    """A backtest horizon: a count of rows, or an amount of calendar time."""

    rows: int | None
    amount: int
    unit: str
    offset: pd.DateOffset | None


def _parse_horizon(horizon: int | str) -> _Horizon:
    if isinstance(horizon, bool):
        raise BacktestError(
            'horizon must be a number of periods or a duration such as "90D", not a bool.'
        )
    if isinstance(horizon, (int, np.integer)):
        if horizon < 1:
            raise BacktestError(f"horizon must be at least 1 period, got {horizon}.")
        return _Horizon(int(horizon), int(horizon), "period", None)
    if not isinstance(horizon, str):
        raise BacktestError(
            f"horizon must be an int or a duration string, got {type(horizon).__name__}."
        )
    match = _HORIZON.match(horizon)
    if match is None:
        raise BacktestError(
            f'Cannot read horizon {horizon!r}. Use a number of periods (90) or a duration such as "90D".'
        )
    amount, raw = int(match.group(1)), match.group(2)
    # Upper-case M means months and lower-case m is ambiguous, as in prophet.quick.
    if raw == "m":
        raise BacktestError(
            f'Ambiguous unit "m" in horizon {horizon!r}. Use "min" for minutes or "M" for months.'
        )
    key = "mo" if raw == "M" else _HORIZON_ALIASES.get(raw.lower(), raw.lower())
    if key not in _HORIZON_UNITS or amount < 1:
        raise BacktestError(
            f"Cannot read horizon {horizon!r}. Use a positive amount and s, min, h, D, W, M, Q or Y."
        )
    name, base = _HORIZON_UNITS[key]
    return _Horizon(
        None,
        amount,
        name,
        pd.DateOffset(**{k: v * amount for k, v in base.kwds.items()}),
    )


def split_holdout(
    df: pd.DataFrame, horizon: int | str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a ds/y frame into training data and the held-out end.

    An int holds out that many rows; a duration string such as ``"90D"`` holds
    out everything later than that long before the last date.
    """
    if "ds" not in df or "y" not in df:
        raise BacktestError('df needs "ds" and "y" columns.')
    frame = (
        df.assign(ds=pd.to_datetime(df["ds"])).sort_values("ds").reset_index(drop=True)
    )
    spec = _parse_horizon(horizon)
    if spec.rows is not None:
        cut = len(frame) - spec.rows
    else:
        assert spec.offset is not None
        cutoff = frame["ds"].max() - spec.offset
        cut = int((frame["ds"] <= cutoff).sum())
    if cut < 2 or frame["y"].iloc[:cut].notna().sum() < 2:
        raise BacktestError(
            f"A horizon of {horizon!r} leaves fewer than 2 rows to train on ({len(frame)} rows in total)."
        )
    if cut >= len(frame):
        raise BacktestError(f"A horizon of {horizon!r} leaves nothing to hold out.")
    return frame.iloc[:cut].reset_index(drop=True), frame.iloc[cut:].reset_index(
        drop=True
    )


def _horizon_phrase(
    horizon: int | str,
    n_rows: int,
    step: pd.Timedelta,
    calendar: pd.Timedelta | None = None,
) -> str:
    """'the last 90 days': in the caller's own words for a duration, from the spacing for a row count.

    ``calendar`` is the time the held-out rows actually span. It replaces the
    spacing-based wording when days are skipped, where 30 weekday rows are
    six weeks, not 30 days.
    """
    spec = _parse_horizon(horizon)
    if spec.rows is None:
        return _lead_phrase(spec.amount, spec.unit, "the last")
    if calendar is not None:
        return _lead_phrase(*_duration_parts(calendar), "the last")
    return describe_periods(n_rows, step)


def backtest_headline(metrics: BacktestMetrics, span: str) -> str:
    """Plain-English title: ``Backtest: typical error 6.5% over the last 90 days; 74% of actuals inside the 80% band``."""
    if metrics.mape is not None:
        head = f"Backtest: typical error {_fmt_pct(metrics.mape)} over {span}"
    else:
        head = f"Backtest: forecasts missed by {_fmt_number(metrics.mae)} on average over {span}"
    if metrics.coverage is None:
        return head
    band = (
        f"{metrics.interval_width:.0%} band" if metrics.interval_width else "interval"
    )
    return f"{head}; {_fmt_pct(metrics.coverage)} of actuals inside the {band}"


def _backtest_subtitle(bt: Backtest, show_band: bool, step: pd.Timedelta) -> str:
    text = (
        f"The model was refitted on data up to {_fmt_day(bt.cutoff, step)} without seeing {bt.span}, then asked to "
        "forecast them. Dark dots are what actually happened and the claret line is the forecast"
    )
    m = bt.metrics
    if show_band and m.coverage is not None:
        width = f"{m.interval_width:.0%} " if m.interval_width else ""
        text += f", with its {width}interval shaded. Hollow dots fall outside it."
        if m.interval_width:
            text += f" A well-calibrated band should catch about {m.interval_width:.0%} of actuals."
    else:
        text += "."
    return text


# ----------------------------------------------------------------------------
# Regressor effects: pure parts
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressorEffect:
    """One extra regressor: its fitted coefficient and a sentence saying what it means."""

    name: str
    mode: Literal["additive", "multiplicative"]
    coef: float
    coef_lower: float
    coef_upper: float
    unit: str | None
    binary: bool
    sentence: str
    headline: str


def regressor_sentence(
    name: str,
    mode: str,
    coef: float,
    *,
    unit: str | None = None,
    binary: bool = False,
    scale: float = 1.0,
) -> str:
    """The fitted coefficient in plain words.

    Additive: "each extra °C adds 0.8 to y". Multiplicative: "each extra
    unit of ad spend raises y by 2.1% of the trend level". A 0/1 regressor
    is described as on versus off. ``scale`` is the typical size of y, used to
    tell a negligible effect from a real one.
    """
    mult = mode == "multiplicative"
    label = name.replace("_", " ")
    each = f"each extra {unit}" if unit else f"each extra unit of {label}"
    when = f"when {label} is on (1 rather than 0)"
    amount = _fmt_pct(coef) if mult else _fmt_number(abs(coef))
    negligible = abs(coef) < (0.005 if mult else 0.005 * max(scale, 1e-12))
    if negligible:
        return (
            f"{when}, y barely changes"
            if binary
            else f"{each} has almost no effect on y"
        )
    if binary:
        if mult:
            return f"{when}, y is {amount} of the trend level {'lower' if coef < 0 else 'higher'}"
        return f"{when}, y is {amount} {'lower' if coef < 0 else 'higher'}"
    if mult:
        return f"{each} {'lowers' if coef < 0 else 'raises'} y by {amount} of the trend level"
    return f"{each} {'takes' if coef < 0 else 'adds'} {amount} {'off' if coef < 0 else 'to'} y"


def _binary_regressor(m: Prophet, name: str) -> bool:
    history = cast("pd.DataFrame", m.history)
    if name not in history:
        return False
    values = pd.unique(history[name].dropna())
    return len(values) > 0 and set(np.asarray(values, dtype=float).tolist()) <= {
        0.0,
        1.0,
    }


def _y_scale(m: Prophet) -> float:
    y = np.abs(cast("pd.DataFrame", m.history)["y"].to_numpy(dtype=float))
    scale = float(np.nanmax(y)) if y.size else 0.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def _contribution(m: Prophet, fcst: pd.DataFrame, name: str) -> pd.Series:
    """What a regressor added to the forecast: units of y if additive, a fraction of the trend if multiplicative."""
    if name not in fcst:
        raise NoRegressorsError(
            f"The forecast has no column {name!r}. Pass the full output of m.predict() for a model fitted with "
            "that regressor."
        )
    return fcst[name]


def _contribution_range(series: pd.Series, mult: bool, binary: bool = False) -> str:
    if binary and len(series):
        # A 0/1 input is either off (no effect) or on (one fixed effect), so say that rather than a range.
        peak = float(series.iloc[int(np.argmax(np.abs(series.to_numpy(dtype=float))))])
        if mult:
            return f"When it is on, it {'lowers' if peak < 0 else 'raises'} the forecast by {_fmt_pct(peak)} of the trend."
        return f"When it is on, it {'takes' if peak < 0 else 'adds'} {_fmt_number(abs(peak))} {'off' if peak < 0 else 'to'} the forecast."
    lo, hi = float(series.min()), float(series.max())
    if mult:
        return f"Moves the forecast between {_signed(lo, True)} and {_signed(hi, True)} of the trend."
    return f"Moves the forecast between {_signed(lo, False)} and {_signed(hi, False)}."


def regressor_effects(
    m: Prophet,
    fcst: pd.DataFrame,
    *,
    units: Mapping[str, str] | None = None,
) -> tuple[RegressorEffect, ...]:
    """Coefficient and plain-English sentence for each extra regressor, strongest effect first.

    Raises :class:`NoRegressorsError` when the model has none.
    """
    if not m.extra_regressors:
        raise NoRegressorsError(
            "This model has no extra regressors. Add one with m.add_regressor(name) before fitting to chart its effect."
        )
    unit_map = {**_DEFAULT_UNITS, **(units or {})}
    coefs = regressor_coefficients(m).set_index("regressor")
    scale = _y_scale(m)
    out: list[RegressorEffect] = []
    for name in m.extra_regressors:
        row = cast("dict[str, Any]", coefs.loc[name].to_dict())
        mode = cast("Literal['additive', 'multiplicative']", str(row["regressor_mode"]))
        coef = float(row["coef"])
        series = _contribution(m, fcst, name)
        binary = _binary_regressor(m, name)
        out.append(
            RegressorEffect(
                name=name,
                mode=mode,
                coef=coef,
                coef_lower=float(row["coef_lower"]),
                coef_upper=float(row["coef_upper"]),
                unit=unit_map.get(name),
                binary=binary,
                sentence=regressor_sentence(
                    name,
                    mode,
                    coef,
                    unit=unit_map.get(name),
                    binary=binary,
                    scale=scale,
                ),
                headline=_contribution_range(series, mode == "multiplicative", binary),
            )
        )
    spread = {e.name: _reach_in_y(m, fcst, e.name, e.mode) for e in out}
    # A stable sort keeps the order the regressors were added in for ties.
    return tuple(sorted(out, key=lambda e: -spread[e.name]))


def _reach_in_y(m: Prophet, fcst: pd.DataFrame, name: str, mode: str) -> float:
    """How far a regressor moves the forecast at most, in the units of y.

    An additive regressor is in those units already. A multiplicative one is a
    fraction of the trend, so it is scaled by the trend on each date first;
    ranking the raw numbers would put a regressor that moves y by tens below one
    that moves it by tenths. Without a trend column the typical size of y stands in.
    """
    effect = _contribution(m, fcst, name).abs()
    if mode == "multiplicative":
        effect = effect * (fcst["trend"].abs() if "trend" in fcst else _y_scale(m))
    peak = float(effect.max())
    return peak if np.isfinite(peak) else 0.0


def _coef_figure(effect: RegressorEffect) -> str:
    """The big number on a regressor card: '+0.8 per °C'."""
    mult = effect.mode == "multiplicative"
    per = (
        " when on"
        if effect.binary
        else f" per {effect.unit}"
        if effect.unit
        else " per unit"
    )
    return f"{_signed(effect.coef, mult)}{per}"


def _coef_range(effect: RegressorEffect) -> str | None:
    """Credible range, only when sampling gave one that differs from the estimate."""
    mult = effect.mode == "multiplicative"
    if np.isclose(effect.coef_lower, effect.coef_upper, rtol=1e-6, atol=1e-12):
        return None
    return f"Likely range {_signed(effect.coef_lower, mult)} to {_signed(effect.coef_upper, mult)}"


# ----------------------------------------------------------------------------
# Weather sources: pure parts
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRun:
    """A stretch of consecutive rows whose weather came from the same source."""

    source: str
    start: pd.Timestamp
    end: pd.Timestamp
    n: int


def _align_sources(
    fcst: pd.DataFrame, sources: Sequence[object] | pd.Series
) -> pd.Series:
    """Source labels as a Series on the forecast's own index (NaN where unknown)."""
    series = sources if isinstance(sources, pd.Series) else pd.Series(list(sources))
    if series.index.equals(fcst.index):
        aligned = series
    elif (
        isinstance(series.index, pd.DatetimeIndex)
        and pd.DatetimeIndex(fcst["ds"]).isin(series.index).all()
    ):
        aligned = pd.Series(
            series.reindex(pd.DatetimeIndex(fcst["ds"])).to_numpy(), index=fcst.index
        )
    elif len(series) == len(fcst):
        aligned = pd.Series(series.to_numpy(), index=fcst.index)
    else:
        raise ValueError(
            f'weather_sources has {len(series)} values but the forecast has {len(fcst)} rows; align it to fcst["ds"].'
        )
    unknown = sorted({str(v) for v in aligned.dropna().unique()} - set(_WEATHER_LABELS))
    if unknown:
        raise ValueError(
            f'Unknown weather source(s) {unknown}; use "archive", "forecast" or "climatology".'
        )
    return aligned


def weather_source_runs(
    ds: Sequence[pd.Timestamp] | pd.Series, sources: Sequence[object] | pd.Series
) -> tuple[SourceRun, ...]:
    """Group consecutive rows with the same weather source into runs (rows with no source are skipped)."""
    dates = pd.Series(pd.to_datetime(pd.Series(list(ds)))).reset_index(drop=True)
    labels = pd.Series(list(sources)).reset_index(drop=True)
    runs: list[SourceRun] = []
    for i, label in enumerate(labels):
        if pd.isna(label):
            continue
        if runs and runs[-1].source == label and runs[-1].end == dates.iloc[i - 1]:
            last = runs[-1]
            runs[-1] = SourceRun(last.source, last.start, dates.iloc[i], last.n + 1)
        else:
            runs.append(SourceRun(str(label), dates.iloc[i], dates.iloc[i], 1))
    return tuple(runs)


def _runs_after(
    runs: Sequence[SourceRun],
    ds: Sequence[pd.Timestamp] | pd.Series,
    after: pd.Timestamp,
) -> tuple[SourceRun, ...]:
    """The part of each run that lies after ``after``.

    A run that starts before the end of the history but carries on into the
    forecast (the weather forecast usually begins a day or two before the last
    observation) is cut to the rows after it, so describing "the future" does
    not lose it. ``ds`` are the timestamps the runs were built from.
    """
    dates = pd.Series(pd.to_datetime(pd.Series(list(ds)))).reset_index(drop=True)
    later: list[SourceRun] = []
    for run in runs:
        if run.start > after:
            later.append(run)
            continue
        rows = dates[(dates > after) & (dates >= run.start) & (dates <= run.end)]
        if len(rows):
            later.append(SourceRun(run.source, rows.min(), rows.max(), len(rows)))
    return tuple(later)


def _run_length(run: SourceRun, step: pd.Timedelta, calendar: bool) -> str:
    """How long a run lasts: its row count at the data's spacing, or the calendar time it covers."""
    if calendar:
        return _duration_phrase(run.end - run.start + step)
    return describe_periods(run.n, step, lead="").strip()


def weather_source_phrase(
    runs: Sequence[SourceRun],
    after: pd.Timestamp,
    step: pd.Timedelta,
    *,
    calendar: bool = False,
) -> str:
    """Where the future weather came from: 'Ahead, it comes from the weather forecast for 9 days, then typical seasonal values (climatology) for 21 days.'

    Only runs that start after ``after`` (the end of the history) are described.
    With ``calendar=True`` each stretch is measured in elapsed time rather than
    rows, which matters when days are skipped (nine weekday rows span 13 days).
    """
    future = [r for r in runs if r.start > after]
    if not future:
        return ""
    words = {
        "archive": "observed weather",
        "forecast": "the weather forecast",
        "climatology": "typical seasonal values (climatology)",
    }
    if len(future) == 1:
        return f"Ahead, all of it comes from {words[future[0].source]}."
    parts = [f"{words[r.source]} for {_run_length(r, step, calendar)}" for r in future]
    return f"Ahead, it comes from {parts[0]}, then " + ", then ".join(parts[1:]) + "."


def weather_ahead_note(
    runs: Sequence[SourceRun],
    after: pd.Timestamp,
    step: pd.Timedelta,
    *,
    name: str = "temperature",
    calendar: bool = False,
) -> str:
    """A credit-line sentence, only when climatology fills part of the future.

    ``Future temperature: weather forecast for 9 days, then climatology (typical seasonal values) for 51 days.``
    Returns an empty string when the future weather is all measured or forecast (or there is no future),
    since then there is nothing a reader needs warning about.
    """
    future = [r for r in runs if r.start > after]
    if not any(r.source == "climatology" for r in future):
        return ""
    words = {
        "archive": "observed weather",
        "forecast": "weather forecast",
        "climatology": "climatology (typical seasonal values)",
    }
    parts = [f"{words[r.source]} for {_run_length(r, step, calendar)}" for r in future]
    return f"Future {name}: " + ", then ".join(parts) + "."


# ----------------------------------------------------------------------------
# Summary numbers and tiles: pure parts
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """A labelled key number for the report and the dashboard."""

    label: str
    value: str
    detail: str


@dataclass(frozen=True)
class _Summary:
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    step: pd.Timedelta
    last_value: float
    n_future: int
    future_end: pd.Timestamp | None
    forecast_mean: float | None
    recent_mean: float | None
    window: pd.Timedelta | None
    scale: float
    y_min: float = float("nan")
    y_max: float = float("nan")


def _summarise(m: Prophet, fcst: pd.DataFrame, df: pd.DataFrame | None) -> _Summary:
    base = cast("pd.DataFrame", df if df is not None else m.history)
    observed = (
        base.assign(ds=pd.to_datetime(base["ds"]))
        .dropna(subset=["y"])
        .sort_values("ds")
    )
    if observed.empty:
        observed = cast("pd.DataFrame", m.history).sort_values("ds")
    start, end = observed["ds"].min(), observed["ds"].max()
    diffs = observed["ds"].diff().dropna()
    step = pd.Timedelta(diffs.median()) if len(diffs) else pd.Timedelta(days=1)
    future = fcst[fcst["ds"] > end]
    mean = float(future["yhat"].mean()) if len(future) else None
    window: pd.Timedelta | None = None
    recent: float | None = None
    if len(future):
        span = future["ds"].max() - end
        window = min(span, end - start)
        if window > pd.Timedelta(0) and window >= span * 0.5:
            recent = float(observed.loc[observed["ds"] > end - window, "y"].mean())
    return _Summary(
        n_obs=len(observed),
        start=start,
        end=end,
        step=step,
        last_value=float(observed["y"].iloc[-1]),
        n_future=len(future),
        future_end=future["ds"].max() if len(future) else None,
        forecast_mean=mean,
        recent_mean=recent,
        window=window,
        scale=_y_scale(m),
        y_min=float(observed["y"].min()),
        y_max=float(observed["y"].max()),
    )


def _compare_sentence(s: _Summary) -> str:
    if s.forecast_mean is None or s.recent_mean is None or s.window is None:
        return ""
    return f"{_compare_phrase(s.forecast_mean, s.recent_mean, s.scale)} the last {_duration_phrase(s.window)}"


def _summary_tiles(s: _Summary, skip: SkipAxis | None = None) -> list[Tile]:
    span = s.end - s.start
    freq, freq_detail = _frequency_with_skips(s.step, skip)
    tiles = [
        Tile(
            "HISTORY",
            _duration_phrase(span),
            f"{_fmt_date(s.start, span)} to {_fmt_date(s.end, span)}, {s.n_obs:,} points",
        ),
        Tile("FREQUENCY", freq, freq_detail),
    ]
    if s.future_end is not None and s.forecast_mean is not None:
        ahead = s.future_end - s.end
        tiles += [
            Tile(
                "HORIZON",
                _duration_phrase(ahead),
                f"to {_fmt_date(s.future_end, span)}, {s.n_future:,} points ahead",
            ),
            Tile(
                "LAST VALUE", _fmt_number(s.last_value), f"on {_fmt_day(s.end, s.step)}"
            ),
            Tile(
                "FORECAST AVERAGE",
                _fmt_number(s.forecast_mean),
                _compare_sentence(s) or "over the horizon",
            ),
        ]
    else:
        tiles += [
            Tile("HORIZON", "None", "in-sample fit only"),
            Tile(
                "LAST VALUE", _fmt_number(s.last_value), f"on {_fmt_day(s.end, s.step)}"
            ),
        ]
    return tiles


def _metric_tiles(metrics: BacktestMetrics) -> list[Tile]:
    tiles: list[Tile] = []
    if metrics.mape is not None:
        tiles.append(
            Tile(
                "TYPICAL ERROR",
                _fmt_pct(metrics.mape),
                "average miss as a share of the actual (MAPE)",
            )
        )
    tiles += [
        Tile(
            "AVERAGE MISS",
            _fmt_number(metrics.mae),
            "mean absolute error (MAE), in the units of the data",
        ),
        Tile(
            "BIG MISSES",
            _fmt_number(metrics.rmse),
            "RMSE, which weights large misses more",
        ),
    ]
    if metrics.coverage is not None:
        width = f"{metrics.interval_width:.0%} " if metrics.interval_width else ""
        tiles.append(
            Tile(
                "INSIDE THE BAND",
                _fmt_pct(metrics.coverage),
                f"of actuals fell inside the {width}interval",
            )
        )
    tiles.append(
        Tile("HELD BACK", f"{metrics.n:,} points", "forecast without seeing them")
    )
    return tiles


#: A forecast average this far (as a share of the range of the data) outside that range is called out.
_OUTSIDE_RANGE = 0.25


def _range_warning(s: _Summary) -> str:
    """A sentence when the forecast average lies well outside every value the data has shown, else nothing.

    A negative average for data that never goes below zero is called out at any
    distance. A trend can carry a forecast past the old extremes, so the other
    cases need a clear margin before they count.
    """
    mean, low, high = s.forecast_mean, s.y_min, s.y_max
    if mean is None or not (np.isfinite(low) and np.isfinite(high)):
        return ""
    margin = _OUTSIDE_RANGE * (high - low)
    if mean < low - margin or (low >= 0 and mean < 0):
        return (
            f"That average is below every value in the history (the lowest is {_fmt_number(low)}), "
            "so check the model before relying on the forecast."
        )
    if mean > high + margin:
        return (
            f"That average is above every value in the history (the highest is {_fmt_number(high)}), "
            "so check the model before relying on the forecast."
        )
    return ""


def _report_headline(
    m: Prophet, s: _Summary, skip: SkipAxis | None = None
) -> tuple[str, str]:
    freq, _ = _frequency_with_skips(s.step, skip)
    facts = (
        f"Based on {s.n_obs:,} observations from {_fmt_date(s.start, s.end - s.start)} to "
        f"{_fmt_date(s.end, s.end - s.start)}, {freq.lower()}."
    )
    if s.future_end is None or s.forecast_mean is None:
        return "Model report: how the model fits the history", facts
    ahead = _lead_phrase(*_duration_parts(s.future_end - s.end), "the next")
    compare = _compare_sentence(s)
    title = f"Forecast report: {ahead}" + (f" look {compare}" if compare else "")
    sub = f"The forecast averages {_fmt_number(s.forecast_mean)} over {ahead}."
    warning = _range_warning(s)
    if warning:
        sub += f" {warning}"
    if m.uncertainty_samples:
        sub += f" Shaded bands show the range the real value should land in {m.interval_width:.0%} of the time."
    return title, f"{sub} {facts}"


# ----------------------------------------------------------------------------
# Component blurbs: pure parts
# ----------------------------------------------------------------------------


def _effect_pair(lo: float, hi: float, mult: bool) -> str:
    return f"{_signed(lo, mult)} to {_signed(hi, mult)}"


def _trend_blurb(first: float, now: float, scale: float, since: str) -> str:
    diff = now - first
    if abs(diff) < 0.03 * max(abs(first), 0.05 * scale):
        return "Broadly flat over the history"
    if first > 0 and now >= 0 and first >= 0.2 * scale:
        return f"{'Down' if diff < 0 else 'Up'} {_fmt_pct(diff / first)} since {since}"
    return f"{'Falling' if diff < 0 else 'Rising'} since {since}"


def _grouped_blurb(
    values: pd.Series,
    labels: pd.Series,
    mult: bool,
    scale: float,
    names: Mapping[int, str],
    template: str,
) -> str:
    means = values.groupby(labels).mean()
    if means.empty:
        return ""
    lo, hi = float(means.min()), float(means.max())
    if hi - lo < (0.005 if mult else 0.005 * scale):
        return "Almost no pattern here"
    return template.format(
        hi=names[int(means.idxmax())],
        hi_v=_signed(hi, mult),
        lo=names[int(means.idxmin())],
        lo_v=_signed(lo, mult),
    )


def _yearly_blurb(m: Prophet, name: str, mult: bool, scale: float) -> str:
    """Peak and low of the yearly curve as the panel draws it (a daily curve), so text and chart agree."""
    days = pd.date_range("2017-01-01", periods=365)
    values = m.predict_seasonal_components(seasonality_plot_df(m, days))[name].to_numpy(
        dtype=float
    )
    hi, lo = int(np.argmax(values)), int(np.argmin(values))
    if values[hi] - values[lo] < (0.005 if mult else 0.005 * scale):
        return "Almost no pattern here"
    return (
        f"Highest in {days[hi]:%B} ({_signed(float(values[hi]), mult)}), "
        f"lowest in {days[lo]:%B} ({_signed(float(values[lo]), mult)})"
    )


def _component_blurb(m: Prophet, fcst: pd.DataFrame, name: str) -> str:
    """One data-derived line under a component panel title."""
    modes = cast("dict[str, list[str]]", m.component_modes)
    mult = name in modes["multiplicative"]
    scale = _y_scale(m)
    if name not in fcst:
        return ""
    values = fcst[name].astype(float)
    ds = fcst["ds"]
    if name == "trend":
        hist_end = cast("pd.DataFrame", m.history)["ds"].max()
        inside = fcst.loc[ds <= hist_end]
        inside = inside if len(inside) else fcst
        since = _fmt_date(inside["ds"].iloc[0], fcst["ds"].max() - fcst["ds"].min())
        return _trend_blurb(
            float(inside["trend"].iloc[0]),
            float(inside["trend"].iloc[-1]),
            scale,
            since,
        )
    if name in m.seasonalities:
        period = float(m.seasonalities[name]["period"])
        if abs(period - 7) < 1e-6:
            days = {i: f"{calendar.day_name[i]}s" for i in range(7)}
            return _grouped_blurb(
                values,
                ds.dt.dayofweek,
                mult,
                scale,
                days,
                "{hi} are highest ({hi_v}), {lo} lowest ({lo_v})",
            )
        if abs(period - 365.25) < 1e-6:
            return _yearly_blurb(m, name, mult, scale)
        if abs(period - 1) < 1e-6:
            hours = {i: f"{i:02d}:00" for i in range(24)}
            return _grouped_blurb(
                values,
                ds.dt.hour,
                mult,
                scale,
                hours,
                "Highest around {hi} ({hi_v}), lowest around {lo} ({lo_v})",
            )
    return f"Effect ranges from {_effect_pair(float(values.min()), float(values.max()), mult)}"


def _report_components(m: Prophet, fcst: pd.DataFrame) -> list[str]:
    """Component panels for the report: trend, holidays and seasonalities (regressors have their own section)."""
    names = ["trend"]
    if m.train_holiday_names is not None and "holidays" in fcst:
        names.append("holidays")
    for first in ("weekly", "yearly"):
        if first in m.seasonalities and first in fcst:
            names.append(first)
    names += [
        n
        for n in sorted(m.seasonalities)
        if n in fcst and n not in ("weekly", "yearly")
    ]
    return names


def _component_title(name: str) -> str:
    if name in ("trend", "holidays"):
        return _pretty(name)
    return f"{_pretty(name)} pattern"


# ----------------------------------------------------------------------------
# Drawing helpers (matplotlib)
# ----------------------------------------------------------------------------


def _join_sentences(*parts: str | None) -> str:
    """Join non-empty credit-line parts into sentences; a lone part is left exactly as given."""
    kept = [p.rstrip(".") for p in parts if p]
    return ". ".join(kept) + ("." if len(kept) > 1 else "")


def _wrap(
    text: str,
    width_in: float,
    size_pt: float,
    max_lines: int | None = None,
    factor: float = 0.56,
) -> str:
    """Wrap to a width in inches; factor is the average glyph width as a share of the font size."""
    chars = max(12, int(width_in * 72 / (size_pt * factor)))
    if max_lines is not None:
        text = textwrap.shorten(text, width=chars * max_lines, placeholder="...")
        # Wrapping at word boundaries can need one line more than the character budget suggests. The width
        # estimate is cautious, so widen it a little rather than spill past the line limit.
        for extra in range(chars // 4 + 1):
            lines = textwrap.wrap(text, width=chars + extra)
            if len(lines) <= max_lines:
                return "\n".join(lines) or text
    return "\n".join(textwrap.wrap(text, width=chars)) or text


def _n_lines(text: str | None) -> int:
    return 0 if not text else text.count("\n") + 1


@dataclass(frozen=True)
class _Page:
    """A figure measured in inches from its top-left corner."""

    w: float
    h: float

    def fx(self, x_in: float) -> float:
        return x_in / self.w

    def fy(self, top_in: float) -> float:
        return 1 - top_in / self.h

    def rect(
        self, left: float, top: float, width: float, height: float
    ) -> tuple[float, float, float, float]:
        return (
            left / self.w,
            1 - (top + height) / self.h,
            width / self.w,
            height / self.h,
        )


@dataclass(frozen=True)
class _Header:
    title: str | None
    subtitle: str | None
    height_in: float


def _make_header(
    width_in: float,
    title: str | None,
    subtitle: str | None,
    title_pt: float = _TITLE_PT,
) -> _Header:
    """Wrap the headline for a width and measure the header in inches."""
    title_text = _wrap(title, width_in, title_pt, 2, factor=0.6) if title else None
    sub_text = _wrap(subtitle, width_in, _SUBTITLE_PT, 3) if subtitle else None
    head_pt = (
        _n_lines(title_text) * title_pt * 1.25 + _n_lines(sub_text) * _SUBTITLE_PT * 1.4
    )
    gap_pt = 8 if title_text and sub_text else 0
    return _Header(title_text, sub_text, 0.3 + (head_pt + gap_pt) / 72 + 0.2)


def _draw_header(
    fig: Figure,
    page: _Page,
    header: _Header,
    left_in: float = _TEXT_LEFT_IN,
    title_pt: float = _TITLE_PT,
) -> None:
    """Figure furniture: accent bar, serif headline and sans subtitle."""
    fig.add_artist(
        Rectangle(
            (0, page.fy(0.1)),
            0.7 / page.w,
            0.1 / page.h,
            transform=fig.transFigure,
            color=ACCENT,
            clip_on=False,
        )
    )
    y = 0.3
    if header.title:
        fig.text(
            page.fx(left_in),
            page.fy(y),
            header.title,
            fontfamily=_serif(),
            fontweight="bold",
            fontsize=title_pt,
            color=INK,
            va="top",
            ha="left",
            linespacing=1.15,
        )
        y += (_n_lines(header.title) * title_pt * 1.25 + 8) / 72
    if header.subtitle:
        fig.text(
            page.fx(left_in),
            page.fy(y),
            header.subtitle,
            fontfamily=_sans(),
            fontsize=_SUBTITLE_PT,
            color=INK_SOFT,
            va="top",
            ha="left",
            linespacing=1.35,
        )


def _source_extra(text: str | None, page_w: float, left_in: float) -> float:
    """Extra footer height, in inches, for a credit line that wraps onto more than one line."""
    if not text:
        return 0.0
    return 0.15 * (_n_lines(_wrap(text, page_w - left_in - 0.3, _SOURCE_PT)) - 1)


def _draw_source(
    fig: Figure, page: _Page, text: str | None, left_in: float = _TEXT_LEFT_IN
) -> None:
    """The credit line, wrapped to the page width and anchored to the bottom edge."""
    if text:
        fig.text(
            page.fx(left_in),
            page.fy(page.h - 0.12),
            _wrap(text, page.w - left_in - 0.3, _SOURCE_PT),
            fontfamily=_sans(),
            fontsize=_SOURCE_PT,
            color=INK_SOFT,
            va="bottom",
            ha="left",
        )


def _axes_headline(ax: Axes, title: str | None, subtitle: str | None) -> None:
    """Headline on the axes itself, for user-supplied axes."""
    width_in = (
        ax.get_window_extent().width / ax.figure.dpi
    )  # pyrefly:ignore[missing-attribute]
    sub_text = _wrap(subtitle, width_in, _SUBTITLE_PT, 3) if subtitle else None
    title_text = (
        _wrap(title, width_in, _PANEL_TITLE_PT + 1, 2, factor=0.6) if title else None
    )
    sub_pt = _n_lines(sub_text) * _SUBTITLE_PT * 1.4
    if sub_text:
        ax.annotate(
            sub_text,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=_SUBTITLE_PT,
            color=INK_SOFT,
            fontfamily=_sans(),
            linespacing=1.35,
            annotation_clip=False,
        )
    if title_text:
        ax.set_title(
            title_text,
            loc="left",
            pad=6 + (sub_pt + 7 if sub_text else 0),
            fontfamily=_serif(),
            fontsize=_PANEL_TITLE_PT + 1,
            fontweight="bold",
            color=INK,
        )


def _style_axes(ax: Axes) -> None:
    """Editorial furniture: horizontal gridlines, baseline rule only, soft tick labels."""
    ax.set_facecolor(PAPER)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.tick_params(
        axis="x", length=4, color=INK, labelsize=_TICK_PT, labelcolor=INK_SOFT
    )
    ax.tick_params(axis="y", length=0, labelsize=_TICK_PT, labelcolor=INK_SOFT, pad=4)
    ax.xaxis.get_offset_text().set(
        color=INK_SOFT, fontsize=_NOTE_PT, fontfamily=_sans()
    )
    if mpl.__version_info__ >= (3, 8):
        ax.tick_params(axis="both", labelfontfamily=_sans())
    else:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(_sans())


def _style_dates(ax: Axes, skip: SkipAxis | None = None) -> None:
    """Concise date ticks; on a skip axis the scale closes the gaps and the ticks follow it."""
    if skip is not None:
        skip.apply(ax)
        return
    locator = AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))


def _dnum(ts: pd.Timestamp) -> float:
    return float(date2num(ts))


def _tint_zones(
    ax: Axes, x0: pd.Timestamp, boundary: pd.Timestamp, x1: pd.Timestamp
) -> None:
    """Tint the two zones differently and mark the boundary."""
    ax.axvspan(_dnum(x0), _dnum(boundary), color=HISTORY_TINT, lw=0, zorder=0)
    ax.axvspan(_dnum(boundary), _dnum(x1), color=FORECAST_TINT, lw=0, zorder=0)
    ax.axvline(_dnum(boundary), color=INK_SOFT, lw=0.9, ls=(0, (3, 2)), zorder=2)


def _pad_ylim(
    ax: Axes, arrays: Sequence[object], bottom: float = 0.1, top: float = 0.1
) -> None:
    flat = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return
    lo, hi = float(flat.min()), float(flat.max())
    span = hi - lo or max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - bottom * span, hi + top * span)


def _stroke() -> list[Any]:
    """A paper-coloured halo so labels stay legible over lines and bands."""
    return [patheffects.withStroke(linewidth=3, foreground=PAPER)]


def _text_at(
    ax: Axes,
    x: Any,
    y: float,
    s: str,
    dx: float = 0.0,
    dy: float = 0.0,
    transform: Transform | None = None,
    **kw: Any,
) -> Text:
    """Text anchored at a point plus an offset in points; a plain Text, so its extent is reliable."""
    base = transform if transform is not None else ax.transData
    shifted = base + ScaledTranslation(
        dx / 72, dy / 72, ax.figure.dpi_scale_trans
    )  # pyrefly:ignore[unsupported-operation, missing-attribute]
    return ax.text(x, y, s, transform=shifted, **kw)


def _px(ax: Axes, x: object, y: object) -> np.ndarray:
    """Screen positions of plotted points, so labels can steer clear of the data."""
    pts = np.column_stack(
        [date2num(x), np.asarray(y, dtype=float)]
    )  # pyrefly:ignore[bad-argument-type]
    return ax.transData.transform(pts[np.isfinite(pts).all(axis=1)])


def _free(box: Bbox, taken: Sequence[Bbox], series: Sequence[np.ndarray]) -> bool:
    """True if the box overlaps no earlier label and covers no plotted point."""
    if any(box.overlaps(other) for other in taken):
        return False
    for pts in series:
        inside = (
            (pts[:, 0] >= box.x0)
            & (pts[:, 0] <= box.x1)
            & (pts[:, 1] >= box.y0)
            & (pts[:, 1] <= box.y1)
        )
        if inside.any():
            return False
    return True


def _inside_axes(ax: Axes, box: Bbox) -> bool:
    axes_box = ax.get_window_extent()
    return bool(
        box.x0 >= axes_box.x0
        and box.x1 <= axes_box.x1
        and box.y0 >= axes_box.y0
        and box.y1 <= axes_box.y1
    )


@dataclass(frozen=True)
class _Slot:
    """A candidate position for a direct label: anchor, offset in points and alignment."""

    x: object
    y: float
    dx: float
    dy: float
    ha: str
    va: str


def _place_label(
    ax: Axes,
    text: str,
    slots: Sequence[_Slot],
    taken: list[Bbox],
    series: Sequence[np.ndarray],
    **style: Any,
) -> Text | None:
    """Draw the label at the first slot where it fits inside the axes and covers nothing; None if no slot fits."""
    for slot in slots:
        label = _text_at(
            ax,
            slot.x,
            slot.y,
            text,
            dx=slot.dx,
            dy=slot.dy,
            ha=slot.ha,
            va=slot.va,
            fontfamily=_sans(),
            zorder=6,
            **style,
        )
        box = label.get_window_extent().padded(4)
        if _inside_axes(ax, box) and _free(box, taken, series):
            taken.append(box)
            return label
        label.remove()
    return None


def _zone_labels(
    ax: Axes,
    labels: Sequence[
        tuple[str, object, float, str, str, float, bool, tuple[float, float]]
    ],
    taken: list[Bbox],
    series: Sequence[np.ndarray],
) -> None:
    """Direct labels for the zones, on a shared baseline raised until none sits on the data.

    Each label is (text, x, dx, ha, colour, size, bold, allowed range in screen x). A label wider than its zone
    is dropped rather than spilling over the boundary.
    """
    tr = ax.get_xaxis_transform()
    candidates = (
        (0.025, "bottom"),
        (0.09, "bottom"),
        (0.155, "bottom"),
        (0.975, "top"),
    )
    for y, va in candidates:
        trial = list(taken)
        placed: list[Text | None] = []
        for text, x, dx, ha, colour, size, bold, (lo, hi) in labels:
            label = _text_at(
                ax,
                x,
                y,
                text,
                dx=dx,
                transform=tr,
                ha=ha,
                va=va,
                fontsize=size,
                color=colour,
                fontweight="bold" if bold else "normal",
                fontfamily=_sans(),
                zorder=6,
            )
            box = label.get_window_extent().padded(6)
            if box.x0 >= lo and box.x1 <= hi and _free(box, trial, series):
                trial.append(box)
                placed.append(label)
            else:
                label.remove()
                placed.append(None)
        if all(p is not None for p in placed) or y == candidates[-1][0]:
            taken[:] = trial
            return
        for label in placed:
            if label is not None:
                label.remove()


def _mark_extremes(ax: Axes, x: Sequence[object], y: np.ndarray, mult: bool) -> None:
    """Mark the highest and lowest point of a curve with their signed effect."""
    if not np.isfinite(y).any():
        return
    hi, lo = int(np.nanargmax(y)), int(np.nanargmin(y))
    if hi == lo:
        return
    for i, offset, va in ((hi, 9, "bottom"), (lo, -9, "top")):
        ax.plot([x[i]], [y[i]], "o", ms=5, color=ACCENT, zorder=5)
        # Hang the label inward when its point sits at either end, so it is not cut off by the axes edge.
        share = i / max(len(y) - 1, 1)
        ha = "left" if share < 0.06 else "right" if share > 0.94 else "center"
        _text_at(
            ax,
            x[i],
            float(y[i]),
            _signed(float(y[i]), mult),
            dy=offset,
            ha=ha,
            va=va,
            fontsize=_NOTE_PT,
            fontweight="bold",
            color=INK,
            fontfamily=_sans(),
            path_effects=_stroke(),
            zorder=6,
        )


def _kicker(
    fig: Figure, page: _Page, text: str, left_in: float, top_in: float, width_in: float
) -> None:
    """Small-caps section label over a hairline."""
    fig.text(
        page.fx(left_in),
        page.fy(top_in),
        text,
        fontfamily=_sans(),
        fontsize=_KICKER_PT,
        fontweight="bold",
        color=INK_SOFT,
        va="top",
        ha="left",
    )
    fig.add_artist(
        plt.Line2D(
            [page.fx(left_in), page.fx(left_in + width_in)],
            [page.fy(top_in + 0.2)] * 2,
            transform=fig.transFigure,
            color=GRID,
            lw=1.0,
        )
    )


def _draw_tile(
    fig: Figure, page: _Page, tile: Tile, left_in: float, top_in: float, width_in: float
) -> None:
    """A key number: label over a rule, a big serif value and a small wrapped detail."""
    fig.add_artist(
        plt.Line2D(
            [page.fx(left_in), page.fx(left_in + width_in)],
            [page.fy(top_in)] * 2,
            transform=fig.transFigure,
            color=INK,
            lw=1.2,
        )
    )
    fig.text(
        page.fx(left_in),
        page.fy(top_in + 0.07),
        tile.label,
        fontfamily=_sans(),
        fontsize=7.5,
        fontweight="bold",
        color=INK_SOFT,
        va="top",
        ha="left",
    )
    size = 17.0 if len(tile.value) <= 10 else 13.0
    fig.text(
        page.fx(left_in),
        page.fy(top_in + 0.24),
        tile.value,
        fontfamily=_serif(),
        fontsize=size,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )
    fig.text(
        page.fx(left_in),
        page.fy(top_in + 0.55),
        _wrap(tile.detail, width_in - 0.08, _NOTE_PT - 0.5, 3),
        fontfamily=_sans(),
        fontsize=_NOTE_PT - 0.5,
        color=INK_SOFT,
        va="top",
        ha="left",
        linespacing=1.3,
    )


def _tile_row(
    fig: Figure,
    page: _Page,
    tiles: Sequence[Tile],
    left_in: float,
    top_in: float,
    width_in: float,
    gap_in: float = 0.14,
) -> None:
    n = max(len(tiles), 1)
    each = (width_in - gap_in * (n - 1)) / n
    for i, tile in enumerate(tiles):
        _draw_tile(fig, page, tile, left_in + i * (each + gap_in), top_in, each)


# ----------------------------------------------------------------------------
# Backtest (imperative shell)
# ----------------------------------------------------------------------------


def run_backtest(
    m: Prophet,
    df: pd.DataFrame,
    horizon: int | str,
    *,
    skip_days: SkipSpec = None,
    skip_hours: SkipSpec = None,
) -> Backtest:
    """Hold out the end of ``df``, refit a copy of ``m`` on the rest and forecast the holdout.

    ``skip_days`` / ``skip_hours`` name periods the series never has. Rows of
    ``df`` on those days or hours are dropped first, so neither the training
    data nor the holdout contains them, and a row-count horizon counts only
    the rows that remain.

    The copy keeps the model's configuration: growth, seasonalities, extra
    regressors, holidays, priors and interval width. ``df`` needs ``ds`` and
    ``y`` plus every regressor, conditional-seasonality and ``cap``/``floor``
    column the model was fitted with.

    Raises
    ------
    BacktestError: If the model is not fitted, the horizon cannot be read or
        leaves too little to train on, or ``df`` lacks a column the model needs.
    """
    if m.history is None:
        raise BacktestError("The model must be fitted before a backtest.")
    skip = _chart_axis(df["ds"], skip_days, skip_hours) if "ds" in df else None
    frame = df if skip is None else df[skip.keeps(pd.to_datetime(df["ds"]))]
    train, holdout = split_holdout(frame, horizon)
    needed = set(m.extra_regressors)
    needed |= {
        p["condition_name"]
        for p in m.seasonalities.values()
        if p["condition_name"] is not None
    }
    if m.growth == "logistic":
        needed |= {"cap"} | ({"floor"} if m.logistic_floor else set())
    missing = sorted(needed - set(df.columns))
    if missing:
        raise BacktestError(
            f"df lacks the column(s) {missing} that the model was fitted with."
        )
    cutoff = pd.Timestamp(train["ds"].max())
    copy = prophet_copy(m, cutoff)
    # prophet_copy leaves these at their defaults; carry them over so the refit matches the original.
    copy.scaling = m.scaling
    copy.fit_kwargs = dict(m.fit_kwargs)
    logger.info(
        "Backtest: refitting on %d rows up to %s, holding out %d.",
        len(train),
        cutoff,
        len(holdout),
    )
    copy.fit(train)
    forecast = copy.predict(holdout.drop(columns="y"))
    step = (
        pd.Timedelta(train["ds"].diff().dropna().median())
        if len(train) > 1
        else pd.Timedelta(days=1)
    )
    lower = forecast["yhat_lower"] if "yhat_lower" in forecast else None
    upper = forecast["yhat_upper"] if "yhat_upper" in forecast else None
    metrics = backtest_metrics(
        holdout["y"], forecast["yhat"], lower, upper, interval_width=m.interval_width
    )
    calendar = (
        None
        if skip is None
        else pd.Timedelta(holdout["ds"].max() - holdout["ds"].min()) + step
    )
    return Backtest(
        cutoff=cutoff,
        span=_horizon_phrase(horizon, len(holdout), step, calendar),
        train=train,
        holdout=holdout,
        forecast=forecast,
        metrics=metrics,
        skip_days=() if skip is None else tuple(sorted(skip.skip_days)),
        skip_hours=() if skip is None else tuple(sorted(skip.skip_hours)),
    )


def _draw_backtest(
    ax: Axes,
    bt: Backtest,
    *,
    show_band: bool,
    annotate: bool,
    context: int,
    skip: SkipAxis | None = None,
) -> None:
    train = bt.train.tail(context)
    hold = bt.holdout
    fc = bt.forecast
    n_hold = len(hold)
    x0 = min(train["ds"].min() if len(train) else hold["ds"].min(), bt.cutoff)
    x1 = max(hold["ds"].max(), fc["ds"].max())
    _tint_zones(ax, x0, bt.cutoff, x1)

    dots = (
        5.0
        if len(train) + n_hold <= 200
        else 3.5
        if len(train) + n_hold <= 1500
        else 2.6
    )
    ax.plot(
        train["ds"],
        train["y"],
        ls="-" if len(train) <= 400 else "",
        lw=0.8,
        marker="o",
        ms=dots,
        mfc=OBSERVED,
        mec="none",
        color=OBSERVED,
        alpha=0.7,
        zorder=3,
    )
    if show_band:
        ax.fill_between(
            fc["ds"],
            fc["yhat_lower"],
            fc["yhat_upper"],
            color=ACCENT,
            alpha=BAND_ALPHA,
            lw=0,
            zorder=2,
        )
    ax.plot(
        fc["ds"], fc["yhat"], ls="-", c=ACCENT, lw=1.8, solid_capstyle="round", zorder=4
    )
    ax.plot(
        hold["ds"],
        hold["y"],
        ls="-" if n_hold <= 400 else "",
        lw=0.9,
        color=INK,
        alpha=0.6,
        zorder=4,
    )
    ax.plot(
        hold["ds"],
        hold["y"],
        ls="",
        marker="o",
        ms=dots + 0.5,
        mfc=INK,
        mec="none",
        zorder=5,
    )
    if show_band:
        pair = pd.DataFrame(
            {
                "ds": hold["ds"].to_numpy(),
                "y": hold["y"].to_numpy(),
                "lo": fc["yhat_lower"].to_numpy(),
                "hi": fc["yhat_upper"].to_numpy(),
            }
        )
        out = pair[(pair["y"] < pair["lo"]) | (pair["y"] > pair["hi"])]
        if len(out):
            ax.plot(
                out["ds"],
                out["y"],
                ls="",
                marker="o",
                ms=dots + 2.5,
                mfc=PAPER,
                mec=ACCENT,
                mew=1.4,
                zorder=6,
            )

    drawn: list[object] = [
        train["y"].to_numpy(dtype=float),
        hold["y"].to_numpy(dtype=float),
        fc["yhat"].to_numpy(dtype=float),
    ]
    if show_band:
        drawn += [
            fc["yhat_lower"].to_numpy(dtype=float),
            fc["yhat_upper"].to_numpy(dtype=float),
        ]
    _pad_ylim(ax, drawn, bottom=0.22 if annotate else 0.1, top=0.12)
    ax.set_xlim(x0, x1)
    _style_dates(ax, skip)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    if not annotate:
        return

    series = [
        _px(ax, train["ds"], train["y"]),
        _px(ax, hold["ds"], hold["y"]),
        _px(ax, fc["ds"], fc["yhat"]),
    ]
    if show_band:
        series += [
            _px(ax, fc["ds"], fc["yhat_lower"]),
            _px(ax, fc["ds"], fc["yhat_upper"]),
        ]
    taken: list[Bbox] = []
    edge = float(ax.transData.transform((_dnum(bt.cutoff), 0.0))[0])
    span = x1 - x0
    step = (
        pd.Timedelta(train["ds"].diff().dropna().median())
        if len(train) > 1
        else pd.Timedelta(days=1)
    )
    note = f"trained up to {_fmt_day(bt.cutoff, step) if span < pd.Timedelta(days=60) else _fmt_date(bt.cutoff, span)}"
    _zone_labels(
        ax,
        [
            (
                "TRAINING",
                x0,
                6.0,
                "left",
                OBSERVED_TEXT,
                _LABEL_PT,
                True,
                (-np.inf, edge),
            ),
            (
                "HOLDOUT",
                _midpoint(bt.cutoff, x1, skip),
                0.0,
                "center",
                ACCENT,
                _LABEL_PT,
                True,
                (edge, np.inf),
            ),
            (
                note,
                bt.cutoff,
                -6.0,
                "right",
                INK_SOFT,
                _NOTE_PT,
                False,
                (-np.inf, edge),
            ),
        ],
        taken,
        series,
    )

    # Direct labels for the two lines, wherever there is room beside them.
    picks = [
        min(max(int(n_hold * q), 0), n_hold - 1) for q in (0.5, 0.75, 0.25, 0.9, 0.1)
    ]
    actual_slots = [
        _Slot(
            hold["ds"].iloc[i],
            float(hold["y"].iloc[i]),
            0,
            10 * s,
            "center",
            "bottom" if s > 0 else "top",
        )
        for i in picks
        for s in (1, -1)
    ]
    fore_slots = [
        _Slot(
            fc["ds"].iloc[i],
            float(fc["yhat"].iloc[i]),
            0,
            10 * s,
            "center",
            "bottom" if s > 0 else "top",
        )
        for i in picks
        for s in (-1, 1)
    ]
    _place_label(
        ax,
        "Actual",
        actual_slots,
        taken,
        series,
        fontsize=_LABEL_PT,
        fontweight="bold",
        color=INK,
        path_effects=_stroke(),
    )
    _place_label(
        ax,
        "Forecast",
        fore_slots,
        taken,
        series,
        fontsize=_LABEL_PT,
        fontweight="bold",
        color=ACCENT,
        path_effects=_stroke(),
    )


def plot_backtest(
    m: Prophet,
    df: pd.DataFrame,
    horizon: int | str,
    *,
    ax: Axes | None = None,
    title: str | bool | None = None,
    subtitle: str | bool | None = None,
    backtest: Backtest | None = None,
    uncertainty: bool = True,
    annotate: bool = True,
    context: int | None = None,
    figsize: tuple[float, float] = (10, 6),
    source: str | bool | None = None,
    skip_days: SkipSpec = None,
    skip_hours: SkipSpec = None,
) -> Figure:
    """Hold out the end of the data, refit on the rest and draw actuals against the forecast.

    Parameters
    ----------
    m: Fitted Prophet model whose configuration is copied.
    df: The data the model was fitted on (``ds``, ``y`` and any regressor,
        ``cap`` and ``floor`` columns).
    horizon: How much to hold out: a number of rows, or a duration string such
        as ``"90D"``, ``"12W"`` or ``"6M"`` (months).
    ax: Optional axes. The headline then goes on the axes.
    title: Headline. None writes one that states the accuracy, a string
        replaces it and False draws none (nor an automatic subtitle).
    subtitle: Line under the headline. None writes one, a string replaces it,
        False omits it.
    backtest: A result from :func:`run_backtest` to draw without refitting.
    uncertainty: Draw the interval and count coverage when the model has one.
    annotate: Draw the zone labels and the direct line labels.
    context: Training points to show before the cutoff. Defaults to three
        times the holdout (at least 30).
    figsize: Width and height in inches, when this function creates the figure.
    source: Credit line under the figure, only when this function creates the
        figure. None writes a default, a string replaces it and False omits it.
    skip_days, skip_hours: Weekdays (names, numbers with Monday as 0, or
        "weekends") and hours of the day the data never has. Rows on them are
        left out of the training data and the holdout, and the time axis closes
        the gaps. None uses the settings of ``backtest`` when one is given; pass
        ``()`` to draw a continuous axis.

    Returns
    -------
    A matplotlib figure. Use :func:`run_backtest` for the metrics.
    """
    bt = (
        backtest
        if backtest is not None
        else run_backtest(m, df, horizon, skip_days=skip_days, skip_hours=skip_hours)
    )
    skip = _chart_axis(
        pd.concat([bt.train["ds"], bt.holdout["ds"]]),
        bt.skip_days if skip_days is None else skip_days,
        bt.skip_hours if skip_hours is None else skip_hours,
    )
    show_band = bool(
        uncertainty and bt.metrics.coverage is not None and "yhat_lower" in bt.forecast
    )
    step = (
        pd.Timedelta(bt.train["ds"].diff().dropna().median())
        if len(bt.train) > 1
        else pd.Timedelta(days=1)
    )
    title_text = (
        None
        if title is False
        else title
        if isinstance(title, str)
        else backtest_headline(bt.metrics, bt.span)
    )
    if title is False and not isinstance(subtitle, str):
        sub_text = None
    elif isinstance(subtitle, str):
        sub_text = subtitle or None
    else:
        sub_text = (
            None
            if subtitle is False or isinstance(title, str)
            else _backtest_subtitle(bt, show_band, step)
        )
    n_context = context if context is not None else max(3 * len(bt.holdout), 30)
    if ax is not None:
        _axes_headline(ax, title_text, sub_text)
        fig = cast("Figure", ax.get_figure())
    else:
        page = _Page(*figsize)
        header = _make_header(page.w - _TEXT_LEFT_IN - _RIGHT_IN, title_text, sub_text)
        source_text = (
            None
            if source is False
            else source
            if isinstance(source, str)
            else _join_sentences(
                f"NeoProphet backtest: {len(bt.train):,} training and {len(bt.holdout):,} held-back points",
                None if skip is None else _skip_note(skip),
            )
        )
        foot = (
            0.7
            + (0.25 if source_text else 0)
            + _source_extra(source_text, page.w, _TEXT_LEFT_IN)
        )
        fig = plt.figure(figsize=figsize, facecolor=PAPER)
        ax = fig.add_axes(
            page.rect(
                _LEFT_IN,
                header.height_in,
                page.w - _LEFT_IN - _RIGHT_IN,
                max(page.h - header.height_in - foot, 0.5),
            )
        )
        _draw_header(fig, page, header)
        _draw_source(fig, page, source_text)
    _style_axes(ax)
    _draw_backtest(
        ax, bt, show_band=show_band, annotate=annotate, context=n_context, skip=skip
    )
    return fig


# ----------------------------------------------------------------------------
# Regressor effects (imperative shell)
# ----------------------------------------------------------------------------


def _source_strip(
    fig: Figure,
    rect: tuple[float, float, float, float],
    main: Axes,
    runs: Sequence[SourceRun],
    last: pd.Timestamp,
    skip: SkipAxis | None = None,
) -> list[SourceRun]:
    """A thin ribbon above a panel saying where each stretch of weather came from.

    Returns the runs too narrow to carry their own label, which the caller keys separately.
    The ribbon shares the panel's skip scale so both line up.
    """
    strip = fig.add_axes(rect)
    if skip is not None:
        skip.apply(strip)
    strip.set_xlim(main.get_xlim())
    strip.set_ylim(0, 1)
    strip.axis("off")
    unlabelled: list[SourceRun] = []
    for i, run in enumerate(runs):
        end = runs[i + 1].start if i + 1 < len(runs) else last
        fill, ink = _WEATHER_FILL[run.source]
        strip.axvspan(_dnum(run.start), _dnum(end), color=fill, lw=0)
        width_px = (
            strip.transData.transform((_dnum(end), 0))[0]
            - strip.transData.transform((_dnum(run.start), 0))[0]
        )
        fitted = False
        for label in (
            _WEATHER_LABELS[run.source],
            _WEATHER_LABELS[run.source].split(" (")[0],
        ):
            text = strip.text(
                _mid_num(run.start, end, skip),
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=_NOTE_PT - 1,
                fontweight="bold",
                color=ink,
                fontfamily=_sans(),
            )
            if text.get_window_extent().width <= width_px - 6:
                fitted = True
                break
            text.remove()
        if not fitted:
            unlabelled.append(run)
    return unlabelled


def _mid_num(a: pd.Timestamp, b: pd.Timestamp, skip: SkipAxis | None) -> float:
    """Date number halfway between two timestamps, measured along the drawn axis."""
    return _dnum(_midpoint(a, b, skip))


def _source_keys(
    fig: Figure, page: _Page, runs: Sequence[SourceRun], right_in: float, top_in: float
) -> None:
    """Colour keys, right-aligned, for ribbon stretches too narrow to label themselves."""
    x = right_in
    for source in reversed(list(dict.fromkeys(r.source for r in runs))):
        label = _WEATHER_LABELS[source].split(" (")[0]
        text = fig.text(
            page.fx(x),
            page.fy(top_in),
            label,
            ha="right",
            va="top",
            fontsize=_NOTE_PT - 0.5,
            color=INK_SOFT,
            fontfamily=_sans(),
        )
        width_in = text.get_window_extent().width / fig.dpi
        fig.add_artist(
            Rectangle(
                (page.fx(x - width_in - 0.18), page.fy(top_in + 0.12)),
                0.11 / page.w,
                0.11 / page.h,
                transform=fig.transFigure,
                color=_WEATHER_FILL[source][0],
                lw=0,
            )
        )
        x -= width_in + 0.18 + 0.16


@dataclass(frozen=True)
class _RegressorRow:
    effect: RegressorEffect
    series: pd.Series
    lower: pd.Series | None
    upper: pd.Series | None


def _regressor_rows(
    m: Prophet, fcst: pd.DataFrame, units: Mapping[str, str] | None
) -> list[_RegressorRow]:
    rows = []
    for effect in regressor_effects(m, fcst, units=units):
        lower = (
            fcst[effect.name + "_lower"]
            if m.uncertainty_samples and effect.name + "_lower" in fcst
            else None
        )
        upper = (
            fcst[effect.name + "_upper"]
            if m.uncertainty_samples and effect.name + "_upper" in fcst
            else None
        )
        rows.append(
            _RegressorRow(effect, _contribution(m, fcst, effect.name), lower, upper)
        )
    return rows


def _draw_contribution(
    ax: Axes,
    m: Prophet,
    fcst: pd.DataFrame,
    row: _RegressorRow,
    annotate: bool,
    skip: SkipAxis | None = None,
) -> None:
    """The regressor's contribution over time, in the history and forecast zones."""
    mult = row.effect.mode == "multiplicative"
    ds = fcst["ds"]
    values = row.series.to_numpy(dtype=float)
    zones = _zone_args(m, fcst)
    if zones is not None:
        _tint_zones(ax, *zones)
    ax.axhline(0, color=INK_SOFT, lw=0.8, zorder=2)
    drawn: list[object] = [values]
    if row.lower is not None and row.upper is not None and not row.effect.binary:
        ax.fill_between(
            ds, row.lower, row.upper, color=ACCENT, alpha=BAND_ALPHA, lw=0, zorder=3
        )
        drawn += [row.lower.to_numpy(dtype=float), row.upper.to_numpy(dtype=float)]
    if row.effect.binary:
        # An on/off input is a block wherever it is on; a line would be a dense comb.
        ax.fill_between(
            ds,
            0,
            values,
            step="mid",
            color=ACCENT,
            lw=0,
            zorder=4,
            rasterized=len(values) > 5000,
        )
        drawn.append(np.zeros(1))
    else:
        ax.plot(
            ds,
            values,
            ls="-",
            c=ACCENT,
            lw=1.6,
            solid_capstyle="round",
            zorder=4,
            rasterized=len(values) > 5000,
        )
    labelled = annotate and not row.effect.binary
    _pad_ylim(
        ax, drawn, bottom=0.4 if labelled else 0.12, top=0.4 if labelled else 0.12
    )
    ax.set_xlim(ds.min(), ds.max())
    _style_dates(ax, skip)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_tick))
    if labelled:
        _mark_extremes(ax, list(ds), values, mult)
    if mult:
        set_y_as_percent(ax)


def _zone_args(
    m: Prophet, fcst: pd.DataFrame
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp] | None:
    boundary = cast("pd.DataFrame", m.history)["ds"].max()
    if not (fcst["ds"] > boundary).any():
        return None
    return min(fcst["ds"].min(), boundary), boundary, fcst["ds"].max()


def _draw_effect_card(
    fig: Figure,
    page: _Page,
    effect: RegressorEffect,
    left_in: float,
    top_in: float,
    width_in: float,
    extra: str = "",
) -> None:
    """The fitted coefficient: a big signed number and what it means, beside the panel."""
    fig.add_artist(
        plt.Line2D(
            [page.fx(left_in)] * 2,
            [page.fy(top_in), page.fy(top_in + 1.3)],
            transform=fig.transFigure,
            color=GRID,
            lw=1.2,
        )
    )
    x = left_in + 0.14
    inner = width_in - 0.14
    fig.text(
        page.fx(x),
        page.fy(top_in),
        "FITTED EFFECT",
        fontfamily=_sans(),
        fontsize=7.5,
        fontweight="bold",
        color=INK_SOFT,
        va="top",
        ha="left",
    )
    figure = _coef_figure(effect)
    size = 17.0 if len(figure) <= 12 else 13.0 if len(figure) <= 18 else 11.0
    fig.text(
        page.fx(x),
        page.fy(top_in + 0.17),
        _wrap(figure, inner, size, 2, factor=0.6),
        fontfamily=_serif(),
        fontsize=size,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
        linespacing=1.1,
    )
    y = (
        top_in
        + 0.17
        + (_n_lines(_wrap(figure, inner, size, 2, factor=0.6)) * size * 1.2) / 72
        + 0.06
    )
    body = effect.sentence[:1].upper() + effect.sentence[1:] + "."
    if (rng := _coef_range(effect)) is not None:
        body += f" {rng}."
    if extra:
        body += f" {extra}"
    wrapped = _wrap(body, inner, _NOTE_PT, 6)
    fig.text(
        page.fx(x),
        page.fy(y),
        wrapped,
        fontfamily=_sans(),
        fontsize=_NOTE_PT,
        color=INK_SOFT,
        va="top",
        ha="left",
        linespacing=1.3,
    )


def _draw_regressor_row(
    fig: Figure,
    page: _Page,
    m: Prophet,
    fcst: pd.DataFrame,
    row: _RegressorRow,
    *,
    left_in: float,
    top_in: float,
    axes_w: float,
    axes_h: float,
    card_left: float,
    card_w: float,
    runs: Sequence[SourceRun] | None,
    annotate: bool,
    title_pt: float = _PANEL_TITLE_PT,
    headline_lines: int = 1,
    skip: SkipAxis | None = None,
) -> float:
    """One regressor: title, optional source strip, contribution panel and coefficient card. Returns the row height."""
    effect = row.effect
    fig.text(
        page.fx(left_in),
        page.fy(top_in),
        _pretty(effect.name),
        fontfamily=_serif(),
        fontsize=title_pt,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )
    # With a source ribbon the right of the headline row holds colour keys, so the headline wraps narrower.
    lines = headline_lines + (1 if runs else 0)
    fig.text(
        page.fx(left_in),
        page.fy(top_in + title_pt * 1.35 / 72),
        _wrap(
            effect.headline, axes_w - (2.6 if runs else 0.0), _SUBTITLE_PT - 1, lines
        ),
        fontfamily=_sans(),
        fontsize=_SUBTITLE_PT - 1,
        color=INK_SOFT,
        va="top",
        ha="left",
    )
    head = (title_pt * 1.35 + (_SUBTITLE_PT - 1) * 1.4 * lines) / 72 + 0.1
    strip_h = 0.19 if runs else 0.0
    ax = fig.add_axes(
        page.rect(
            left_in, top_in + head + strip_h + (0.06 if runs else 0.0), axes_w, axes_h
        )
    )
    _style_axes(ax)
    _draw_contribution(ax, m, fcst, row, annotate, skip)
    note = ""
    if runs:
        narrow = _source_strip(
            fig,
            page.rect(left_in, top_in + head, axes_w, strip_h),
            ax,
            runs,
            fcst["ds"].max(),
            skip,
        )
        _source_keys(fig, page, narrow, left_in + axes_w, top_in + title_pt * 1.35 / 72)
        hist_end = cast("pd.DataFrame", m.history)["ds"].max()
        step = (
            pd.Timedelta(fcst["ds"].diff().dropna().median())
            if len(fcst) > 1
            else pd.Timedelta(days=1)
        )
        note = weather_source_phrase(
            _runs_after(runs, fcst["ds"], hist_end),
            hist_end,
            step,
            calendar=skip is not None,
        )
    _draw_effect_card(
        fig, page, effect, card_left, top_in + head + strip_h, card_w, note
    )
    return head + strip_h + (0.06 if runs else 0.0) + axes_h


def _weather_runs_for(
    m: Prophet,
    fcst: pd.DataFrame,
    weather_sources: Sequence[object] | pd.Series | None,
    weather_regressor: str,
) -> dict[str, tuple[SourceRun, ...]]:
    """Source runs keyed by the regressor they describe (empty when no sources were given)."""
    if weather_sources is None:
        return {}
    if weather_regressor not in m.extra_regressors:
        raise ValueError(
            f"weather_sources was given but the model has no {weather_regressor!r} regressor to attach it to "
            f"(regressors: {sorted(m.extra_regressors)})."
        )
    aligned = _align_sources(fcst, weather_sources)
    return {weather_regressor: weather_source_runs(fcst["ds"], aligned)}


def _without_skipped(
    fcst: pd.DataFrame,
    sources: Sequence[object] | pd.Series | None,
    skip: SkipAxis | None,
) -> tuple[pd.DataFrame, Sequence[object] | pd.Series | None]:
    """Drop forecast rows on skipped days or hours, and the matching weather-source labels.

    The labels are aligned to the full frame first, so rows and labels stay paired.
    """
    if skip is None:
        return fcst, sources
    keep = skip.keeps(fcst["ds"])
    if sources is not None:
        sources = _align_sources(fcst, sources)[keep].reset_index(drop=True)
    return fcst[keep].reset_index(drop=True), sources


def _regressor_figure_headline(rows: Sequence[_RegressorRow]) -> tuple[str, str]:
    top = rows[0]
    mult = top.effect.mode == "multiplicative"
    peak = float(top.series.abs().max())
    reach = (
        f"up to {_fmt_pct(peak)} of the trend" if mult else f"up to {_fmt_number(peak)}"
    )
    if len(rows) == 1:
        title = f"Extra regressor: {top.effect.name.replace('_', ' ')} moves the forecast {reach}"
    else:
        title = f"Extra regressors: {top.effect.name.replace('_', ' ')} matters most, moving the forecast {reach}"
    sub = (
        "Each panel shows how much one input added to the forecast over time (in the units of the data, or as a "
        "share of the trend for multiplicative regressors). The card beside it gives the fitted effect of one "
        "extra unit."
    )
    return title, sub


def plot_regressor_effects(
    m: Prophet,
    fcst: pd.DataFrame,
    *,
    title: str | bool | None = None,
    subtitle: str | bool | None = None,
    weather_sources: Sequence[object] | pd.Series | None = None,
    weather_regressor: str = "temperature",
    units: Mapping[str, str] | None = None,
    uncertainty: bool = True,
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    source: str | bool | None = None,
    skip_days: SkipSpec = None,
    skip_hours: SkipSpec = None,
) -> Figure:
    """What each extra regressor added to the forecast, with its fitted coefficient in plain words.

    One row per regressor: its contribution over time (from the columns
    ``Prophet.predict`` writes for it) beside a card such as "each extra °C
    adds 0.8 to y" or "+2.1% per unit".

    Parameters
    ----------
    m: Fitted Prophet model with at least one extra regressor.
    fcst: Output of ``m.predict``.
    title, subtitle: None writes them from the data, a string replaces them and
        False omits them (title=False also drops the automatic subtitle).
    weather_sources: Labels ("archive", "forecast" or "climatology") for each
        row of ``fcst``, saying where that row's weather value came from. The
        chart then shows a ribbon above the weather regressor's panel.
    weather_regressor: Name of the weather regressor that ``weather_sources``
        describes.
    units: Unit wording per regressor name, for example ``{"humidity": "%"}``;
        ``temperature`` defaults to °C.
    uncertainty: Shade the interval when the model has one.
    annotate: Mark each panel's highest and lowest point.
    figsize: Width and height in inches. The height defaults to fit the rows.
    source: Credit line. None writes a default, a string replaces it, False omits it.
    skip_days, skip_hours: Weekdays (names, numbers with Monday as 0, or
        "weekends") and hours of the day the data never has. Rows of ``fcst``
        on them are not drawn, the time axes close the gaps and the weather
        sources stay paired with their rows.

    Raises
    ------
    NoRegressorsError: If the model has no extra regressors or ``fcst`` lacks a regressor's column.
    ValueError: If ``weather_sources`` is given but the model has no weather regressor, or holds unknown labels.
    """
    skip = _chart_axis(fcst["ds"], skip_days, skip_hours)
    fcst, weather_sources = _without_skipped(fcst, weather_sources, skip)
    rows = _regressor_rows(m, fcst, units)
    if not uncertainty:
        rows = [_RegressorRow(r.effect, r.series, None, None) for r in rows]
    runs_by_name = _weather_runs_for(m, fcst, weather_sources, weather_regressor)
    auto_title, auto_sub = _regressor_figure_headline(rows)
    title_text = (
        None if title is False else title if isinstance(title, str) else auto_title
    )
    if title is False and not isinstance(subtitle, str):
        sub_text = None
    elif isinstance(subtitle, str):
        sub_text = subtitle or None
    else:
        sub_text = None if subtitle is False or isinstance(title, str) else auto_sub

    width = figsize[0] if figsize else 10.0
    header = _make_header(width - _TEXT_LEFT_IN - _RIGHT_IN, title_text, sub_text)
    source_text = (
        None
        if source is False
        else source
        if isinstance(source, str)
        else _join_sentences(
            f"NeoProphet regressor effects: {_summary_phrase(m)}",
            None if skip is None else _skip_note(skip),
        )
    )
    foot = (
        0.55
        + (0.25 if source_text else 0)
        + _source_extra(source_text, width, _TEXT_LEFT_IN)
    )
    axes_h, tick_h, gap = 1.3, 0.45, 0.25
    row_heights = [
        0.5 + (0.25 if r.effect.name in runs_by_name else 0.0) + axes_h + tick_h + gap
        for r in rows
    ]
    height = figsize[1] if figsize else header.height_in + sum(row_heights) + foot
    page = _Page(width, height)
    fig = plt.figure(figsize=(width, height), facecolor=PAPER)
    _draw_header(fig, page, header)
    card_w = 2.3
    axes_w = width - _LEFT_IN - _RIGHT_IN - card_w - 0.3
    top = header.height_in
    scale = (height - header.height_in - foot) / max(sum(row_heights), 1e-9)
    for row, row_h in zip(rows, row_heights):
        used = _draw_regressor_row(
            fig,
            page,
            m,
            fcst,
            row,
            left_in=_LEFT_IN,
            top_in=top,
            axes_w=axes_w,
            axes_h=max(axes_h * scale, 0.6),
            card_left=_LEFT_IN + axes_w + 0.3,
            card_w=card_w,
            runs=runs_by_name.get(row.effect.name),
            annotate=annotate,
            skip=skip,
        )
        top += max(row_h * scale, used + tick_h + gap)
    _draw_source(fig, page, source_text)
    return fig


def _summary_phrase(m: Prophet) -> str:
    history = cast("pd.DataFrame", m.history)
    span = history["ds"].max() - history["ds"].min()
    return (
        f"{len(history):,} observations, {_fmt_date(history['ds'].min(), span)} to "
        f"{_fmt_date(history['ds'].max(), span)}"
    )


# ----------------------------------------------------------------------------
# Report (imperative shell)
# ----------------------------------------------------------------------------

_A4 = (8.27, 11.69)
_MAX_REPORT_HEIGHT = 16.5
_MARGIN_IN = 0.6


def _distinct_ticks(ax: Axes) -> None:
    """Give y ticks enough decimals to tell neighbours apart.

    A nearly flat trend would otherwise label every tick "282".
    """
    formatter = ax.yaxis.get_major_formatter()
    ticks = np.asarray([t for t in ax.get_yticks() if np.isfinite(t)], dtype=float)
    if not isinstance(formatter, FuncFormatter) or len(ticks) < 2:
        return
    shown = [formatter(t) for t in ticks]
    if len(set(shown)) == len(shown):
        return
    step = float(np.min(np.abs(np.diff(ticks))))
    decimals = int(min(max(0.0, -np.floor(np.log10(step))), 6)) if step > 0 else 2
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.{decimals}f}"))


def _draw_component(
    m: Prophet,
    fcst: pd.DataFrame,
    name: str,
    ax: Axes,
    min_dt: pd.Timedelta,
    uncertainty: bool,
    skip: SkipAxis | None = None,
) -> None:
    """Draw one component panel with the library's own plot functions (no headline: the report writes its own).

    ``skip`` closes skipped days and hours on timelines and drops them from the weekly and daily profiles.
    """
    days = () if skip is None else tuple(sorted(skip.skip_days))
    hours = () if skip is None else tuple(sorted(skip.skip_hours))
    if name in ("trend", "holidays"):
        plot_forecast_component(
            m,
            fcst,
            name,
            ax=ax,
            uncertainty=uncertainty,
            plot_cap=name == "trend",
            title=False,
            skip_days=days,
            skip_hours=hours,
        )
    elif name in m.seasonalities:
        period = m.seasonalities[name]["period"]
        if (name == "weekly" or period == 7) and min_dt == pd.Timedelta(days=1):
            plot_weekly(
                m,
                name=name,
                ax=ax,
                uncertainty=uncertainty,
                title=False,
                skip_days=days,
            )
        elif name == "yearly" or period == 365.25:
            plot_yearly(m, name=name, ax=ax, uncertainty=uncertainty, title=False)
        else:
            plot_seasonality(
                m,
                name,
                ax=ax,
                uncertainty=uncertainty,
                title=False,
                skip_days=days,
                skip_hours=hours,
            )
    # Small panels need smaller tick labels, or neighbouring weekday names run together.
    ax.tick_params(axis="both", labelsize=_TICK_PT - 1.5)
    _distinct_ticks(ax)
    if name not in ("trend", "holidays"):
        # The peak and low labels sit outside the curve; a short panel needs more headroom than the default.
        lo, hi = ax.get_ylim()
        pad = (hi - lo) * 0.1
        ax.set_ylim(lo - pad, hi + pad)


def plot_report(
    m: Prophet,
    fcst: pd.DataFrame,
    *,
    df: pd.DataFrame | None = None,
    metrics: BacktestMetrics | None = None,
    weather_sources: Sequence[object] | pd.Series | None = None,
    weather_regressor: str = "temperature",
    units: Mapping[str, str] | None = None,
    title: str | bool | None = None,
    subtitle: str | bool | None = None,
    max_regressors: int = 3,
    uncertainty: bool = True,
    source: str | bool | None = None,
    skip_days: SkipSpec = None,
    skip_hours: SkipSpec = None,
) -> Figure:
    """One A4 portrait page: the forecast, key numbers, components and regressor effects.

    ``fig.savefig("report.pdf")`` writes a clean single page.

    Parameters
    ----------
    m: Fitted Prophet model.
    fcst: Output of ``m.predict``.
    df: The observed data, used for the key numbers (defaults to the model's own history).
    metrics: Backtest metrics (from :func:`backtest_metrics` or :func:`run_backtest`).
        When given, a second row of tiles reports them; the history, frequency, horizon,
        last value and forecast average tiles are always shown.
    weather_sources: Where each row's weather value came from, as for
        :func:`plot_regressor_effects`.
    weather_regressor: Name of the weather regressor.
    units: Unit wording per regressor name.
    title, subtitle: None writes them from the data, a string replaces them,
        False omits them.
    max_regressors: Regressors charted before the rest are summarised in a line.
    uncertainty: Shade intervals when the model has them.
    source: Footer line. None writes a default, a string replaces it, False omits it.
        The default says which days are skipped and, with ``weather_sources``,
        where the future weather came from (forecast or climatology).
    skip_days, skip_hours: Weekdays (names, numbers with Monday as 0, or
        "weekends") and hours of the day the data never has. Rows on them are
        left out of ``fcst`` and ``df``, every time axis closes the gaps, the
        weekly profile shows only the remaining days, and weather sources stay
        paired with their rows.
    """
    content_w = _A4[0] - 2 * _MARGIN_IN
    skip = _chart_axis(fcst["ds"], skip_days, skip_hours)
    fcst, weather_sources = _without_skipped(fcst, weather_sources, skip)
    if skip is not None and df is not None and "ds" in df:
        df = df[skip.keeps(pd.to_datetime(df["ds"]))]
    summary = _summarise(m, fcst, df)
    auto_title, auto_sub = _report_headline(m, summary, skip)
    title_text = (
        None if title is False else title if isinstance(title, str) else auto_title
    )
    if title is False and not isinstance(subtitle, str):
        sub_text = None
    elif isinstance(subtitle, str):
        sub_text = subtitle or None
    else:
        sub_text = None if subtitle is False or isinstance(title, str) else auto_sub
    header = _make_header(content_w, title_text, sub_text, title_pt=19.0)

    comps = _report_components(m, fcst)
    cols = len(comps) if len(comps) <= 3 else 2 if len(comps) == 4 else 3
    comp_rows = -(-len(comps) // cols) if comps else 0
    effects = _regressor_rows(m, fcst, units) if m.extra_regressors else []
    if not uncertainty:
        effects = [_RegressorRow(r.effect, r.series, None, None) for r in effects]
    shown = effects[:max_regressors]
    runs_by_name = _weather_runs_for(m, fcst, weather_sources, weather_regressor)
    weather_runs = runs_by_name.get(weather_regressor, ())
    hist_end = cast("pd.DataFrame", m.history)["ds"].max()
    source_text = (
        None
        if source is False
        else source
        if isinstance(source, str)
        else _join_sentences(
            f"NeoProphet forecast report: {_summary_phrase(m)}",
            None if skip is None else _skip_note(skip),
            weather_ahead_note(
                _runs_after(weather_runs, fcst["ds"], hist_end),
                hist_end,
                summary.step,
                calendar=skip is not None,
            ),
        )
    )

    # Page height: A4, stretched to fill it when the content is light and grown (up to a cap) when it is dense,
    # so a panel never has to shrink below a readable size. Either way the figure is one page.
    kicker_h, tile_h, gap = 0.34, 1.12, 0.14
    comp_head, tick_h, reg_head = 0.66, 0.34, 0.5
    footer = 0.45 + _source_extra(source_text, _A4[0], _MARGIN_IN)
    strips = sum(0.25 for r in shown if r.effect.name in runs_by_name)
    more_h = 0.35 if len(effects) > len(shown) else 0.0
    overhead = (
        header.height_in
        + 0.05
        + kicker_h
        + tile_h
        + gap
        + (kicker_h + tile_h + gap if metrics else 0)
        + kicker_h
        + tick_h
        + gap
        + (kicker_h + comp_rows * (comp_head + tick_h + gap) if comps else 0)
        + (
            kicker_h + len(shown) * (reg_head + tick_h + gap) + strips + more_h
            if shown
            else 0
        )
        + footer
    )
    natural = 2.9 + comp_rows * 1.3 + len(shown) * 1.15
    page_h = min(max(overhead + natural, _A4[1]), _MAX_REPORT_HEIGHT)
    stretch = max((page_h - overhead) / natural, 0.6)
    chart_h, comp_h, reg_h = 2.9 * stretch, 1.3 * stretch, 1.15 * stretch
    page = _Page(_A4[0], page_h)

    fig = plt.figure(figsize=(page.w, page.h), facecolor=PAPER)
    _draw_header(fig, page, header, left_in=_MARGIN_IN, title_pt=19.0)
    top = header.height_in + 0.05

    # Key numbers
    _kicker(fig, page, "AT A GLANCE", _MARGIN_IN, top, content_w)
    top += kicker_h
    _tile_row(fig, page, _summary_tiles(summary, skip), _MARGIN_IN, top, content_w)
    top += tile_h + gap
    if metrics is not None:
        _kicker(
            fig, page, "HOW WELL IT WORKS ON HELD-BACK DATA", _MARGIN_IN, top, content_w
        )
        top += kicker_h
        _tile_row(fig, page, _metric_tiles(metrics), _MARGIN_IN, top, content_w)
        top += tile_h + gap

    # Forecast chart
    _kicker(fig, page, "THE FORECAST", _MARGIN_IN, top, content_w)
    top += kicker_h
    ax = fig.add_axes(page.rect(_MARGIN_IN + 0.5, top, content_w - 0.5, chart_h))
    plot(
        m,
        fcst,
        ax=ax,
        uncertainty=uncertainty,
        title=False,
        skip_days=None if skip is None else skip.skip_days,
        skip_hours=None if skip is None else skip.skip_hours,
    )
    top += chart_h + tick_h + gap

    # Components
    if comps:
        _kicker(fig, page, "WHAT SHAPES IT", _MARGIN_IN, top, content_w)
        top += kicker_h
        history = cast("pd.DataFrame", m.history)
        dt = history["ds"].diff()
        min_dt = dt.iloc[cast("np.ndarray", dt.values).nonzero()[0]].min()
        col_gap = 0.3
        col_w = (content_w - col_gap * (cols - 1)) / cols
        for i, name in enumerate(comps):
            r, c = divmod(i, cols)
            left = _MARGIN_IN + c * (col_w + col_gap)
            row_top = top + r * (comp_head + comp_h + tick_h + gap)
            fig.text(
                page.fx(left),
                page.fy(row_top),
                _component_title(name),
                fontfamily=_serif(),
                fontsize=_PANEL_TITLE_PT,
                fontweight="bold",
                color=INK,
                va="top",
                ha="left",
            )
            blurb = _component_blurb(m, fcst, name)
            if blurb:
                fig.text(
                    page.fx(left),
                    page.fy(row_top + 0.24),
                    _wrap(blurb, col_w, _NOTE_PT, 2),
                    fontfamily=_sans(),
                    fontsize=_NOTE_PT,
                    color=INK_SOFT,
                    va="top",
                    ha="left",
                )
            cax = fig.add_axes(
                page.rect(left + 0.5, row_top + comp_head, col_w - 0.5, comp_h)
            )
            _draw_component(m, fcst, name, cax, pd.Timedelta(min_dt), uncertainty, skip)
        top += comp_rows * (comp_head + comp_h + tick_h + gap)

    # Regressor effects
    if shown:
        _kicker(fig, page, "EXTRA REGRESSORS", _MARGIN_IN, top, content_w)
        top += kicker_h
        card_w = 2.1
        axes_w = content_w - card_w - 0.3 - 0.5
        for row in shown:
            used = _draw_regressor_row(
                fig,
                page,
                m,
                fcst,
                row,
                left_in=_MARGIN_IN + 0.5,
                top_in=top,
                axes_w=axes_w,
                axes_h=reg_h,
                card_left=_MARGIN_IN + 0.5 + axes_w + 0.3,
                card_w=card_w,
                runs=runs_by_name.get(row.effect.name),
                annotate=True,
                title_pt=_PANEL_TITLE_PT - 1,
                skip=skip,
            )
            top += used + tick_h + gap
        if len(effects) > len(shown):
            rest = ", ".join(
                e.effect.name.replace("_", " ") for e in effects[len(shown) :]
            )
            fig.text(
                page.fx(_MARGIN_IN),
                page.fy(top - gap),
                _wrap(
                    f"Also in the model, with smaller effects: {rest}.",
                    content_w,
                    _NOTE_PT,
                    2,
                ),
                fontfamily=_sans(),
                fontsize=_NOTE_PT,
                color=INK_SOFT,
                va="top",
                ha="left",
            )

    _draw_source(fig, page, source_text, left_in=_MARGIN_IN)
    return fig


# ----------------------------------------------------------------------------
# Dashboard (plotly is optional)
# ----------------------------------------------------------------------------


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "dashboard_html needs plotly, which is an optional dependency. Install it with: pip install plotly"
        ) from exc
    return go


def _hover_date(step: pd.Timedelta) -> str:
    return "%d %b %Y" if step >= pd.Timedelta(hours=20) else "%d %b %Y %H:%M"


def _plotly_axes() -> tuple[dict[str, Any], dict[str, Any]]:
    tick = {"family": _CSS_SANS, "color": INK_SOFT, "size": 12}
    x = {
        "showgrid": False,
        "showline": True,
        "linecolor": INK,
        "linewidth": 1,
        "ticks": "outside",
        "tickcolor": INK,
        "tickfont": tick,
    }
    y = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "showline": False,
        "zeroline": False,
        "ticks": "",
        "tickfont": tick,
    }
    return x, y


def _plotly_base_layout(height: int) -> dict[str, Any]:
    return {
        "height": height,
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PAPER,
        "font": {"family": _CSS_SANS, "color": INK_SOFT},
        "margin": {"l": 64, "r": 24, "t": 36, "b": 40},
        "hoverlabel": {
            "bgcolor": PAPER,
            "bordercolor": INK_SOFT,
            "font": {"family": _CSS_SANS, "color": INK},
        },
        "legend": {
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.0,
            "yanchor": "bottom",
            "font": {"family": _CSS_SANS, "size": 12},
        },
    }


def _plotly_zones(
    go: Any,
    fig: Any,
    zones: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp] | None,
    row: int | None = None,
) -> None:
    if zones is None:
        return
    x0, boundary, x1 = (z.to_pydatetime() for z in zones)
    target: dict[str, int] = {} if row is None else {"row": row, "col": 1}
    fig.add_vrect(
        x0=x0,
        x1=boundary,
        fillcolor=HISTORY_TINT,
        opacity=1,
        layer="below",
        line_width=0,
        **target,
    )
    fig.add_vrect(
        x0=boundary,
        x1=x1,
        fillcolor=FORECAST_TINT,
        opacity=1,
        layer="below",
        line_width=0,
        **target,
    )
    fig.add_vline(
        x=boundary,
        line={"color": INK_SOFT, "width": 1, "dash": "dash"},
        layer="below",
        **target,
    )


def _forecast_figure(
    go: Any,
    m: Prophet,
    fcst: pd.DataFrame,
    uncertainty: bool,
    step: pd.Timedelta,
    breaks: Sequence[Mapping[str, object]] = (),
) -> Any:
    history = cast("pd.DataFrame", m.history)
    hist_end = history["ds"].max()
    show_band = bool(uncertainty and m.uncertainty_samples and "yhat_lower" in fcst)
    future = fcst[fcst["ds"] > hist_end]
    past = fcst[fcst["ds"] <= hist_end]
    date = _hover_date(step)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Observed",
            x=history["ds"],
            y=history["y"],
            mode="markers",
            marker={"color": OBSERVED, "size": 4, "opacity": 0.75},
            hovertemplate=f"%{{x|{date}}}<br>Observed <b>%{{y:,.4g}}</b><extra></extra>",
        )
    )
    if show_band:
        band = future if len(future) else fcst
        label = f"{m.interval_width:.0%} interval"
        fig.add_trace(
            go.Scatter(
                x=band["ds"],
                y=band["yhat_lower"],
                mode="lines",
                line={"width": 0},
                legendgroup="band",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                name=label,
                x=band["ds"],
                y=band["yhat_upper"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(153, 15, 61, 0.25)",
                legendgroup="band",
                hoverinfo="skip",
            )
        )

    def line(name: str, frame: pd.DataFrame, width: float, opacity: float) -> None:
        custom = (
            np.column_stack([frame["yhat_lower"], frame["yhat_upper"]])
            if show_band
            else np.empty((len(frame), 0))
        )
        interval = (
            "<br>Interval %{customdata[0]:,.4g} to %{customdata[1]:,.4g}"
            if show_band
            else ""
        )
        fig.add_trace(
            go.Scatter(
                name=name,
                x=frame["ds"],
                y=frame["yhat"],
                mode="lines",
                line={"color": ACCENT, "width": width},
                opacity=opacity,
                customdata=custom,
                hovertemplate=f"%{{x|{date}}}<br>{name} <b>%{{y:,.4g}}</b>{interval}<extra></extra>",
            )
        )

    if len(future) and len(past):
        line("Fitted", past, 1.2, 0.55)
        line("Forecast", future, 2.4, 1.0)
    else:
        line("Forecast", fcst, 2.4, 1.0)
    for column, name in (("cap", "Maximum capacity"), ("floor", "Minimum capacity")):
        if column in fcst and (column == "cap" or m.logistic_floor):
            fig.add_trace(
                go.Scatter(
                    name=name,
                    x=fcst["ds"],
                    y=fcst[column],
                    mode="lines",
                    line={"color": INK_SOFT, "width": 1.4, "dash": "dash"},
                    hovertemplate=f"%{{x|{date}}}<br>{name} <b>%{{y:,.4g}}</b><extra></extra>",
                )
            )
    fig.add_trace(
        go.Scatter(
            name="Trend",
            x=fcst["ds"],
            y=fcst["trend"],
            mode="lines",
            visible="legendonly",
            line={"color": INK, "width": 1.6, "dash": "dot"},
            hovertemplate=f"%{{x|{date}}}<br>Trend <b>%{{y:,.4g}}</b><extra></extra>",
        )
    )
    x_style, y_style = _plotly_axes()
    span = fcst["ds"].max() - fcst["ds"].min()
    buttons = [
        {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
        {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
        {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
        {"step": "all", "label": "all"},
    ]
    selector = (
        {
            "buttons": buttons,
            "bgcolor": HISTORY_TINT,
            "activecolor": FORECAST_TINT,
            "font": {"family": _CSS_SANS, "color": INK_SOFT},
        }
        if span > pd.Timedelta(days=120)
        else None
    )
    fig.update_layout(
        **_plotly_base_layout(480),
        hovermode="closest",
        hoverdistance=30,
        xaxis={
            **x_style,
            "type": "date",
            **({"rangeselector": selector} if selector else {}),
            **({"rangebreaks": list(breaks)} if breaks else {}),
        },
        yaxis=y_style,
    )
    _plotly_zones(go, fig, _zone_args(m, fcst))
    return fig


def _regressor_figure(
    go: Any,
    m: Prophet,
    fcst: pd.DataFrame,
    rows: Sequence[_RegressorRow],
    runs_by_name: Mapping[str, Sequence[SourceRun]],
    step: pd.Timedelta,
    breaks: Sequence[Mapping[str, object]] = (),
) -> Any:
    from plotly.subplots import make_subplots

    n = len(rows)
    date = _hover_date(step)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.12, 0.5 / max(n, 1)),
        subplot_titles=[_pretty(r.effect.name) for r in rows],
    )
    for annotation in fig.layout.annotations:
        annotation.update(
            x=0,
            xref="paper",
            xanchor="left",
            align="left",
            font={"family": _CSS_SERIF, "size": 15, "color": INK},
        )
    x_style, y_style = _plotly_axes()
    zones = _zone_args(m, fcst)
    for i, row in enumerate(rows, start=1):
        mult = row.effect.mode == "multiplicative"
        fmt = "%{y:+.2%}" if mult else "%{y:+,.4g}"
        name = _pretty(row.effect.name)
        if row.lower is not None and row.upper is not None:
            fig.add_trace(
                go.Scatter(
                    x=fcst["ds"],
                    y=row.lower,
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    legendgroup=row.effect.name + "-band",
                    hoverinfo="skip",
                ),
                row=i,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    name=f"{name} interval",
                    x=fcst["ds"],
                    y=row.upper,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(153, 15, 61, 0.25)",
                    legendgroup=row.effect.name + "-band",
                    hoverinfo="skip",
                ),
                row=i,
                col=1,
            )
        # An on/off input is drawn as blocks wherever it is on; a line would be a dense comb.
        shape: dict[str, Any] = (
            {"line_shape": "hv", "fill": "tozeroy", "fillcolor": ACCENT}
            if row.effect.binary
            else {}
        )
        fig.add_trace(
            go.Scatter(
                name=name,
                x=fcst["ds"],
                y=row.series,
                mode="lines",
                line={"color": ACCENT, "width": 2},
                **shape,
                hovertemplate=f"%{{x|{date}}}<br>{name} adds <b>{fmt}</b><extra></extra>",
            ),
            row=i,
            col=1,
        )
        fig.update_xaxes(**x_style, row=i, col=1)
        fig.update_yaxes(
            **{**y_style, "zeroline": True, "zerolinecolor": INK_SOFT},
            tickformat=".0%" if mult else None,
            row=i,
            col=1,
        )
        _plotly_zones(go, fig, zones, row=i)
        span = (fcst["ds"].max() - fcst["ds"].min()).total_seconds() or 1.0
        for run in runs_by_name.get(row.effect.name, ()):
            if run.source == "archive":
                continue  # the history zone is already tinted; only the future needs its source marked
            wide = (run.end - run.start).total_seconds() / span > 0.12
            label: dict[str, Any] = {}
            if wide:
                label = {
                    "annotation_text": _WEATHER_LABELS[run.source].split(" (")[0],
                    "annotation_position": "top left",
                    "annotation_font": {
                        "family": _CSS_SANS,
                        "size": 11,
                        "color": INK_SOFT,
                    },
                }
            fig.add_vrect(
                x0=run.start.to_pydatetime(),
                x1=run.end.to_pydatetime(),
                fillcolor=_WEATHER_FILL[run.source][0],
                opacity=0.18,
                layer="below",
                line_width=0,
                row=i,
                col=1,
                **label,
            )
    fig.update_layout(
        **{
            **_plotly_base_layout(300 * n + 100),
            "margin": {"l": 64, "r": 24, "t": 90, "b": 40},
        },
        hovermode="closest",
        hoverdistance=30,
    )
    fig.update_layout(legend={"y": 1.0 + 40 / (300 * n + 100)})
    fig.update_xaxes(type="date")
    if breaks:
        fig.update_xaxes(rangebreaks=list(breaks))
    return fig


def _css_tokens(tokens: Mapping[str, str]) -> str:
    return " ".join(f"--{name}: {value};" for name, value in tokens.items())


def _tiles_html(tiles: Sequence[Tile]) -> str:
    cells = "".join(
        f'<div class="tile"><div class="l">{html.escape(t.label)}</div><div class="v">{html.escape(t.value)}</div>'
        f'<div class="d">{html.escape(t.detail)}</div></div>'
        for t in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _cards_html(rows: Sequence[_RegressorRow]) -> str:
    cells = []
    for r in rows:
        e = r.effect
        rng = _coef_range(e)
        body = (
            e.sentence[:1].upper() + e.sentence[1:] + "." + (f" {rng}." if rng else "")
        )
        cells.append(
            f'<div class="card"><div class="t">{html.escape(_pretty(e.name))}</div>'
            f'<div class="n">{html.escape(_coef_figure(e))}</div><div class="t">{html.escape(body)}</div></div>'
        )
    return f'<div class="cards">{"".join(cells)}</div>'


#: Everything below styles through custom properties, so a theme is just a set of tokens.
_PAGE_CSS = f"""
:root {{ {_css_tokens(_palette.LIGHT)} color-scheme: light; --bar: 45px; scroll-padding-top: calc(var(--bar) + 10px); }}
:root[data-theme="dark"] {{ {_css_tokens(_palette.DARK)} color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: {_CSS_SANS}; line-height: 1.45; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 0 20px 48px; }}
a {{ color: inherit; }}
a:focus-visible, button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.skip {{ position: absolute; left: -9999px; top: 0; background: var(--ink); color: var(--paper); padding: 8px 12px; z-index: 30; }}
.skip:focus {{ left: 8px; top: 8px; }}
.topbar {{ position: sticky; top: 0; z-index: 20; background: var(--paper); border-bottom: 1px solid var(--grid); }}
.topbar-in {{ max-width: 1080px; margin: 0 auto; padding: 0 20px; display: flex; align-items: center; gap: 12px; min-height: 44px; }}
.topbar nav {{ position: relative; flex: 1 1 auto; min-width: 0; display: flex; overflow-x: auto; scrollbar-width: none;
              --fade-l: 0px; --fade-r: 0px;
              -webkit-mask-image: linear-gradient(to right, transparent, #000 var(--fade-l), #000 calc(100% - var(--fade-r)), transparent);
              mask-image: linear-gradient(to right, transparent, #000 var(--fade-l), #000 calc(100% - var(--fade-r)), transparent); }}
.topbar nav::-webkit-scrollbar {{ display: none; }}
.topbar nav.more-l {{ --fade-l: 28px; }}
.topbar nav.more-r {{ --fade-r: 28px; }}
.topbar nav a {{ flex: none; padding: 12px 7px 10px; font-size: 12.5px; color: var(--ink-soft); text-decoration: none; border-bottom: 2px solid transparent; }}
.topbar nav a:hover, .topbar nav a[aria-current="true"] {{ color: var(--ink); border-bottom-color: var(--accent); }}
.topbar nav a:focus-visible {{ outline-offset: -2px; }}
.btn {{ flex: none; margin-left: auto; font: inherit; font-size: 12px; font-weight: 700; letter-spacing: .04em; color: var(--ink); background: var(--history-tint);
       border: 1px solid var(--grid); border-radius: 2px; padding: 6px 12px; cursor: pointer; }}
.btn:hover {{ background: var(--forecast-tint); }}
.btn[aria-pressed="true"] {{ background: var(--ink); color: var(--paper); border-color: var(--ink); }}
.bar {{ height: 8px; width: 56px; background: var(--accent); margin: 22px 0; }}
h1 {{ font-family: {_CSS_SERIF}; font-size: 30px; line-height: 1.2; margin: 0 0 10px; }}
h1, h2, h3, .sub, .lede, .note, .foot, .tile, .card, .findings li {{ overflow-wrap: anywhere; }}
.sub {{ color: var(--ink-soft); font-size: 15px; max-width: 760px; margin: 0 0 26px; }}
h2 {{ font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-soft);
     border-bottom: 1px solid var(--grid); padding-bottom: 6px; margin: 34px 0 12px; }}
h3 {{ font-family: {_CSS_SERIF}; font-size: 16px; line-height: 1.3; margin: 22px 0 6px; }}
.lede {{ font-size: 16px; max-width: 760px; margin: 0 0 14px; }}
.note {{ font-size: 13px; color: var(--ink-soft); margin: 4px 0 0; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; }}
.tile {{ border-top: 2px solid var(--ink); padding-top: 8px; }}
.tile .l {{ font-size: 11px; font-weight: 700; letter-spacing: .06em; color: var(--ink-soft); }}
.tile .v {{ font-family: {_CSS_SERIF}; font-size: 24px; font-weight: 700; margin: 2px 0; }}
.tile .d {{ font-size: 13px; color: var(--ink-soft); }}
.findings {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 14px 32px; grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr)); }}
.findings li {{ border-left: 2px solid var(--grid); padding-left: 14px; font-size: 14px; }}
.findings a {{ display: block; font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-soft); text-decoration: none; }}
.findings a:hover {{ color: var(--accent); }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 8px; }}
.card {{ border-left: 2px solid var(--grid); padding-left: 14px; }}
.card .n {{ font-family: {_CSS_SERIF}; font-size: 22px; font-weight: 700; }}
.card .t {{ font-size: 13px; color: var(--ink-soft); }}
.grid2 {{ display: grid; grid-template-columns: 1fr; gap: 16px 28px; }}
.grid2 > div {{ min-width: 0; }}
@media (min-width: 760px) {{ .grid2 {{ grid-template-columns: 1fr 1fr; }} }}
.tbl-wrap {{ overflow-x: auto; margin: 8px 0; }}
.tbl-wrap.scroll {{ max-height: 420px; overflow-y: auto; }}
.tbl {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
.tbl th {{ font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-soft); text-align: left;
          padding: 6px 12px 6px 0; border-bottom: 2px solid var(--ink); white-space: nowrap; }}
.tbl td {{ padding: 6px 12px 6px 0; border-bottom: 1px solid var(--grid); vertical-align: top; }}
.tbl th.num, .tbl td.num {{ text-align: right; font-variant-numeric: tabular-nums; padding-right: 0; padding-left: 12px; }}
.tbl-wrap.scroll .tbl thead th {{ position: sticky; top: 0; background: var(--paper); z-index: 1; }}
.kv {{ margin: 8px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); gap: 0 32px; font-size: 13px; }}
.kv > div {{ display: grid; grid-template-columns: minmax(110px, 40%) 1fr; gap: 12px; padding: 6px 0; border-bottom: 1px solid var(--grid); }}
.kv dt {{ color: var(--ink-soft); }}
.kv dd {{ margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
pre, code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }}
code {{ background: var(--history-tint); padding: 1px 4px; border-radius: 2px; }}
pre {{ background: var(--history-tint); border-left: 2px solid var(--grid); padding: 10px 12px; overflow-x: auto; margin: 8px 0; }}
pre code {{ background: none; padding: 0; }}
.foot {{ margin-top: 40px; font-size: 12px; color: var(--ink-soft); }}
.theme-pending .plotly-graph-div {{ visibility: hidden; }}
@media (max-width: 520px) {{
  h1 {{ font-size: 25px; }}
  .tile .v {{ font-size: 21px; }}
}}
/* Touch screens keep the toolbar showing, where it covers the legend; pinch and double-tap still zoom and reset. */
@media (max-width: 640px) {{ .modebar-container {{ display: none !important; }} }}
@media print {{
  .topbar, .skip, .btn, .modebar-container {{ display: none !important; }}
  body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  main {{ max-width: none; padding: 0; }}
  .js-plotly-plot, .plotly-graph-div, .tbl-wrap, .tile, .card, .findings li, pre {{ break-inside: avoid; page-break-inside: avoid; }}
  .tbl-wrap.scroll {{ max-height: none; overflow: visible; }}
  h2 {{ break-after: avoid; page-break-after: avoid; }}
}}
"""

#: For theme="auto" without scripts: follow the system until the script settles the theme.
_AUTO_CSS = (
    '\n@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { '
    + _css_tokens(_palette.DARK)
    + " color-scheme: dark; } }\n"
)

#: Last in the stylesheet: a page in the dark theme still prints dark ink on light paper.
_PRINT_CSS = (
    '\n@media print { :root, :root[data-theme="dark"], :root:not([data-theme="light"]) { '
    + _css_tokens(_palette.LIGHT)
    + " color-scheme: light; } }\n"
)

_THEMES: tuple[str, ...] = ("light", "dark", "auto")
DashboardTheme = Literal["light", "dark", "auto"]


def _check_theme(theme: str) -> None:
    if theme not in _THEMES:
        raise ValueError(f"theme must be one of {list(_THEMES)}, not {theme!r}.")


def _resolve_sections(
    requested: Sequence[str] | None, valid: Sequence[str]
) -> tuple[str, ...]:
    """The section keys to build, in order: all of ``valid`` for None, else exactly ``requested``."""
    if requested is None:
        return tuple(valid)
    if isinstance(requested, str):
        raise ValueError(
            f"sections must be a list of keys, not the single string {requested!r}; "
            f"valid sections are {list(valid)}."
        )
    keys = tuple(requested)
    unknown = [k for k in keys if k not in valid]
    if unknown:
        raise ValueError(
            f"Unknown dashboard section(s) {unknown}; valid sections are {list(valid)}."
        )
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise ValueError(
            f"Dashboard section(s) {repeated} listed more than once; "
            f"list each of {list(valid)} at most once."
        )
    return keys


def _sorted_by_date(
    fcst: pd.DataFrame, sources: Sequence[object] | pd.Series | None
) -> tuple[pd.DataFrame, Sequence[object] | pd.Series | None]:
    """The forecast in date order, with weather-source labels moved along with their rows."""
    if fcst["ds"].is_monotonic_increasing:
        return fcst, sources
    order = np.argsort(fcst["ds"].to_numpy(), kind="stable")
    if sources is not None:
        sources = _align_sources(fcst, sources).iloc[order].reset_index(drop=True)
    return fcst.iloc[order].reset_index(drop=True), sources


def _regressors_summary(rows: Sequence[_RegressorRow]) -> str:
    """One sentence for the key findings: the strongest regressor and how far it moves the forecast.

    Rows are ranked by how far a regressor moves the forecast, so the sentence says that
    too. The per-unit coefficient is no guide here: a regressor on a large scale can
    move the forecast a lot with a tiny coefficient.
    """
    if not rows:
        return ""
    top = rows[0].effect
    label = top.name.replace("_", " ")
    reach = top.headline[:1].lower() + top.headline[1:]
    if reach.startswith("moves "):
        reach = f"it {reach}"
    if len(rows) == 1:
        return f"Extra regressor {label}: {reach}"
    return f"Of {len(rows)} extra regressors, {label} matters most: {reach}"


def _findings_html(sections: Sequence[Section]) -> str:
    """The key findings list: one line per section that has a summary, linked to its section."""
    items = "".join(
        f'<li><a href="#{html.escape(s.key)}">{html.escape(s.title)}</a>{html.escape(s.summary.strip())}</li>'
        for s in sections
        if s.summary.strip()
    )
    return f'<h2>Key findings</h2><ul class="findings">{items}</ul>' if items else ""


def _nav_html(sections: Sequence[Section]) -> str:
    links = "".join(
        f'<a href="#{html.escape(s.key)}">{html.escape(s.title)}</a>' for s in sections
    )
    return f'<nav aria-label="Sections">{links}</nav>' if links else ""


def _forecast_section(ctx: DashboardContext, step: pd.Timedelta) -> Section:
    from prophet.dashboard._context import Section

    figure = _forecast_figure(
        ctx.go, ctx.m, ctx.fcst, ctx.uncertainty, step, ctx.breaks
    )
    body = (
        ctx.embed(figure)
        + '<p class="note">Hover for the date, value and interval. Click a legend entry to hide or show '
        "it, drag to zoom and double-click to reset.</p>"
    )
    return Section("forecast", "Forecast", body)


#: ``plot_components_plotly`` writes each panel's headline and sentence as a figure
#: annotation, wrapped for the width it is given. An annotation cannot wrap to a
#: phone and plotly clips it at the figure edge, so the page lifts that text into
#: HTML and draws each panel as a figure of its own below it.
_PANEL_LABEL = re.compile(
    r"^<b>(?P<head>.*?)</b>(?:<br><span[^>]*>(?P<sub>.*)</span>)?$", re.S
)
_MARKUP_BREAK = re.compile(r"<br\s*/?>")
_MARKUP_TAG = re.compile(r"<[^>]+>")
#: Wide on purpose: the sentences are then never cut short, and the wrapping is left to the browser.
_PANEL_FIGSIZE = (4000, 200)
_PANEL_MARGIN = {"l": 70, "r": 24, "t": 16, "b": 40}


def _plain_text(markup: str) -> str:
    """Plotly's small HTML subset as plain text: line breaks become spaces, tags go, entities are decoded."""
    text = html.unescape(_MARKUP_TAG.sub("", _MARKUP_BREAK.sub(" ", markup)))
    return " ".join(text.split())


def _panel_text(label: str) -> tuple[str, str]:
    """Headline and sentence of a ``plot_components_plotly`` panel label, as plain text."""
    found = _PANEL_LABEL.match(label)
    if found is None:
        return _plain_text(label), ""
    return _plain_text(found["head"]), _plain_text(found["sub"] or "")


def _panel_id(kind: str, k: int) -> str:
    """Plotly's name for the axis of the k-th panel (1-based): ``x``, ``x2``, ``xaxis3`` and so on."""
    return kind if k == 1 else f"{kind}{k}"


def _panel_ref(ref: str | None, k: int) -> str | None:
    """A shape or annotation reference (``x3``, ``y3 domain``) as panel k alone sees it; None if it points elsewhere."""
    if ref is None:
        return None
    axis, _, rest = ref.partition(" ")
    for kind in ("x", "y"):
        if axis == _panel_id(kind, k):
            return f"{kind} {rest}" if rest else kind
    return None


def _panel_figures(go: Any, combined: Any) -> list[tuple[str, str, Any]]:
    """Split a ``plot_components_plotly`` figure into ``(headline, sentence, figure)``, one per panel.

    Each figure keeps its panel's traces, axes, zone tints and zone labels. The
    combined figure is only read.
    """
    layout = combined.layout
    labels = [a.text for a in layout.annotations if a.xref == "paper"]
    panels = []
    for k, label in enumerate(labels, start=1):
        xaxis = go.layout.XAxis(layout[_panel_id("xaxis", k)])
        yaxis = go.layout.YAxis(layout[_panel_id("yaxis", k)])
        for axis in (xaxis, yaxis):
            axis.anchor = axis.domain = axis.matches = None
        data = [
            type(t)(t).update(xaxis="x", yaxis="y")
            for t in combined.data
            if (t.xaxis or "x") == _panel_id("x", k)
        ]
        shapes = [
            go.layout.Shape(s).update(xref="x", yref=_panel_ref(s.yref, k) or s.yref)
            for s in layout.shapes
            if _panel_ref(s.xref, k)
        ]
        annotations = [
            go.layout.Annotation(a).update(
                xref="x", yref=_panel_ref(a.yref, k) or a.yref
            )
            for a in layout.annotations
            if _panel_ref(a.xref, k)
        ]
        figure = go.Figure(
            data=data,
            layout=go.Layout(
                autosize=True,
                height=_PANEL_FIGSIZE[1] + _PANEL_MARGIN["t"] + _PANEL_MARGIN["b"],
                showlegend=False,
                paper_bgcolor=layout.paper_bgcolor,
                plot_bgcolor=layout.plot_bgcolor,
                font=layout.font,
                margin=_PANEL_MARGIN,
                xaxis=xaxis,
                yaxis=yaxis,
                shapes=shapes,
                annotations=annotations,
            ),
        )
        panels.append((*_panel_text(label), figure))
    return panels


def _components_section(
    ctx: DashboardContext, covered: Collection[str] = ()
) -> Section | None:
    """The component panels, or None when nothing is left to draw.

    ``covered`` names the panels (``"trend"``, ``"holidays"``) that a section
    of their own draws in more detail; they are left out and a line says so.
    """
    from prophet.dashboard import panels
    from prophet.dashboard._context import Section
    from prophet.dashboard._style import h3, lede, note
    from prophet.plot import plot_components_plotly

    skip = ctx.skip
    combined = plot_components_plotly(
        ctx.m,
        ctx.fcst,
        uncertainty=ctx.uncertainty,
        figsize=_PANEL_FIGSIZE,
        title=False,
        skip_days=None if skip is None else skip.skip_days,
        skip_hours=None if skip is None else skip.skip_hours,
    )
    drawn, dropped = panels.refine(ctx, _panel_figures(ctx.go, combined), covered)
    if not drawn:
        return None
    reason = panels.covered_note(dropped)
    body = (lede(reason) if reason else "") + "".join(
        h3(p.head) + (note(p.sentence) if p.sentence else "") + ctx.embed(p.figure)
        for p in drawn
    )
    return Section("components", "Components", body)


def _regressors_section(
    ctx: DashboardContext,
    runs_by_name: Mapping[str, Sequence[SourceRun]],
    weather_regressor: str,
    step: pd.Timedelta,
) -> Section | None:
    from prophet.dashboard._context import Section

    m = ctx.m
    if not m.extra_regressors:
        return None
    rows = _regressor_rows(m, ctx.fcst, ctx.units)
    if not ctx.uncertainty:
        rows = [_RegressorRow(r.effect, r.series, None, None) for r in rows]
    phrase = ""
    if runs_by_name:
        hist_end = cast("pd.DataFrame", m.history)["ds"].max()
        phrase = weather_source_phrase(
            _runs_after(runs_by_name[weather_regressor], ctx.fcst["ds"], hist_end),
            hist_end,
            step,
            calendar=ctx.skip is not None,
        )
    extra = f'<p class="note">{html.escape(phrase)}</p>' if phrase else ""
    figure = _regressor_figure(
        ctx.go, m, ctx.fcst, rows, runs_by_name, step, ctx.breaks
    )
    return Section(
        "regressors",
        "Extra regressors",
        _cards_html(rows) + ctx.embed(figure) + extra,
        _regressors_summary(rows),
    )


def _module_builder(key: str) -> Callable[[DashboardContext], Section | None]:
    """The ``build`` function of the section module for ``key``, looked up when called."""
    module = importlib.import_module(f"prophet.dashboard.{key}")
    return cast("Callable[[DashboardContext], Section | None]", module.build)


def _build_sections(
    keys: Sequence[str],
    ctx: DashboardContext,
    own: Mapping[str, Callable[[DashboardContext], Section | None]],
    after: Mapping[str, Callable[[DashboardContext, frozenset[str]], Section | None]]
    | None = None,
) -> list[Section]:
    """Build the sections of ``keys``, returned in ``keys`` order; those that do not apply to the model are left out.

    A builder in ``after`` runs once every other section is built and is told
    which of them exist, so that it can leave out what they already show.
    """
    late = after or {}
    built: dict[str, Section] = {}

    def keep(key: str, section: Section | None) -> None:
        if section is None:
            logger.info(
                "Dashboard section %r does not apply to this model; left out.", key
            )
            return
        if section.key != key:
            raise ValueError(
                f"The {key!r} section came back with the key {section.key!r}."
            )
        built[key] = section

    for key in keys:
        if key not in late:
            build = own[key] if key in own else _module_builder(key)
            keep(key, build(ctx))
    for key in keys:
        if key in late:
            keep(key, late[key](ctx, frozenset(built)))
    return [built[key] for key in keys if key in built]


def _page_html(
    *,
    title: str,
    subtitle: str,
    tiles: Sequence[Tile],
    metric_tiles: Sequence[Tile],
    sections: Sequence[Section],
    footer: str,
    plotly_js: str,
    theme: DashboardTheme = "light",
) -> str:
    from prophet.dashboard import _assets

    body = [
        f'<div class="bar"></div><h1>{html.escape(title)}</h1><p class="sub">{html.escape(subtitle)}</p>',
        "<h2>At a glance</h2>",
        _tiles_html(tiles),
    ]
    if metric_tiles:
        body += [
            "<h2>How well it works on held-back data</h2>",
            _tiles_html(metric_tiles),
        ]
    body.append(_findings_html(sections))
    body += [
        f'<section id="{html.escape(s.key)}"><h2>{html.escape(s.title)}</h2>{s.body}</section>'
        for s in sections
    ]
    body.append(f'<p class="foot">{html.escape(footer)}</p>')
    initial = "" if theme == "auto" else f' data-theme="{theme}"'
    pressed = "true" if theme == "dark" else "false"
    css = _PAGE_CSS + (_AUTO_CSS if theme == "auto" else "") + _PRINT_CSS
    return (
        f'<!doctype html>\n<html lang="en"{initial} data-theme-default="{theme}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{css}</style>"
        f"<script>{_assets.BOOT_SCRIPT}</script>"
        f'<script type="text/javascript">{plotly_js}</script>'
        f"<script>{_assets.PLOT_TRACKER}</script></head>"
        '<body><a class="skip" href="#main">Skip to the content</a>'
        f'<div class="topbar"><div class="topbar-in">{_nav_html(sections)}'
        '<button type="button" class="btn" id="theme-toggle" aria-pressed="'
        f'{pressed}" title="Switch between the light and dark theme">Dark theme</button></div></div>'
        f'<main id="main">{"".join(body)}</main>'
        f"<script>{_assets.theme_script(_palette.DARK_SWAP)}</script></body></html>\n"
    )


def dashboard_html(
    m: Prophet,
    fcst: pd.DataFrame,
    path: str | Path,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    df: pd.DataFrame | None = None,
    metrics: BacktestMetrics | None = None,
    weather_sources: Sequence[object] | pd.Series | None = None,
    weather_regressor: str = "temperature",
    units: Mapping[str, str] | None = None,
    uncertainty: bool = True,
    skip_days: SkipSpec = None,
    skip_hours: SkipSpec = None,
    sections: Sequence[str] | None = None,
    theme: DashboardTheme = "light",
    backtest: Backtest | None = None,
    cv: pd.DataFrame | None = None,
) -> Path:
    """Write a self-contained interactive page about a fitted model and its forecast.

    The plotly.js bundle is embedded, so the file works offline. The page has
    key numbers and key findings at the top, a navigation bar and one block per
    section. Hover shows the date, value and interval; clicking a legend entry
    hides or shows that series; a button switches between the light and dark
    theme and remembers the choice in the browser.

    The sections, by the key that ``sections`` takes, in their default order:

    * ``forecast``: the interactive forecast chart.
    * ``accuracy``: backtest and cross-validation accuracy by horizon.
    * ``components``: the seasonal curves and the regressor effects in panels,
      worded from the weekdays and hours the data has. The trend and the
      holidays are panels here only when their own sections are not shown.
    * ``trend``: the trend with its changepoints and its rate of change.
    * ``seasonality``: heatmaps and strength of the seasonal patterns.
    * ``holidays``: the effect of each holiday and those in the forecast window.
    * ``regressors``: what each extra regressor added to the forecast.
    * ``diagnostics``: residuals of the fit on the training data.
    * ``data``: data quality and the forecast table.
    * ``settings``: the model settings in plain words.

    Parameters
    ----------
    m: Fitted Prophet model.
    fcst: Output of ``m.predict``.
    path: Where to write the HTML file. Missing parent folders are created.
    title, subtitle: None writes them from the data.
    df: The observed data, for the key numbers and the data section.
    metrics: Backtest metrics to show as a second row of key numbers. None
        takes them from ``backtest`` when that is given.
    weather_sources, weather_regressor, units: As for :func:`plot_regressor_effects`.
    uncertainty: Draw intervals when the model has them.
    skip_days, skip_hours: Weekdays (names, numbers with Monday as 0, or
        "weekends") and hours of the day the data never has. Rows of ``fcst``
        and ``df`` on them are left out and every time axis uses plotly
        ``rangebreaks`` to close the gap; the weekly profile shows only the
        remaining days.
    sections: Keys of the sections to include, in the order to show them. None
        includes every section that applies to the model, in the order below.
        A section that does not apply (``"holidays"`` for a model without
        holidays) is left out and logged at info level. Unknown or repeated
        keys raise ValueError.
    theme: The theme the page opens in: ``"light"``, ``"dark"`` or ``"auto"``
        (follow the system setting, live, until the viewer picks one). A
        choice the viewer made earlier in the same browser takes precedence.
    backtest: A finished :func:`run_backtest`. Its metrics fill ``metrics`` and
        the accuracy section draws it against the held-back data.
    cv: Output of :func:`prophet.diagnostics.cross_validation`, for the
        accuracy section's error by horizon.

    Returns
    -------
    The path written.

    Raises
    ------
    ImportError: If plotly is not installed (``pip install plotly``).
    ValueError: For an unknown ``theme``, unknown or repeated ``sections``, or
        ``weather_sources`` for a model without extra regressors.
    """
    from prophet.dashboard import SECTION_ORDER
    from prophet.dashboard._context import build_context

    _check_theme(theme)
    keys = _resolve_sections(sections, SECTION_ORDER)
    go = _require_plotly()
    from plotly.offline import get_plotlyjs

    if weather_sources is not None and not m.extra_regressors:
        raise ValueError(
            "weather_sources was given but the model has no extra regressors."
        )
    if metrics is None and backtest is not None:
        metrics = backtest.metrics
    skip = _chart_axis(fcst["ds"], skip_days, skip_hours)
    fcst, weather_sources = _without_skipped(fcst, weather_sources, skip)
    fcst, weather_sources = _sorted_by_date(fcst, weather_sources)
    if skip is not None and df is not None and "ds" in df:
        df = df[skip.keeps(pd.to_datetime(df["ds"]))]
    ctx = build_context(
        m,
        fcst,
        df=df,
        skip=skip,
        uncertainty=uncertainty,
        backtest=backtest,
        cv=cv,
        units=units,
        go=go,
    )
    summary = _summarise(m, ctx.fcst, df)
    auto_title, auto_sub = _report_headline(m, summary, skip)
    runs_by_name = _weather_runs_for(m, ctx.fcst, weather_sources, weather_regressor)
    own: dict[str, Callable[[DashboardContext], Section | None]] = {
        "forecast": lambda c: _forecast_section(c, summary.step),
        "regressors": lambda c: _regressors_section(
            c, runs_by_name, weather_regressor, summary.step
        ),
    }
    after: dict[str, Callable[[DashboardContext, frozenset[str]], Section | None]] = {
        "components": lambda c, present: _components_section(
            c, covered=present & {"trend", "holidays"}
        ),
    }
    built = _build_sections(keys, ctx, own, after)
    hist_end = cast("pd.DataFrame", m.history)["ds"].max()
    page = _page_html(
        title=title if title is not None else auto_title,
        subtitle=subtitle if subtitle is not None else auto_sub,
        tiles=_summary_tiles(summary, skip),
        metric_tiles=_metric_tiles(metrics) if metrics else [],
        sections=built,
        footer=_join_sentences(
            f"NeoProphet forecast dashboard: {_summary_phrase(m)}",
            None if skip is None else _skip_note(skip),
            weather_ahead_note(
                _runs_after(
                    runs_by_name.get(weather_regressor, ()), ctx.fcst["ds"], hist_end
                ),
                hist_end,
                summary.step,
                calendar=skip is not None,
            ),
        ),
        plotly_js=get_plotlyjs(),
        theme=theme,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    logger.info("Wrote dashboard to %s (%.1f MB).", out, out.stat().st_size / 1e6)
    return out
