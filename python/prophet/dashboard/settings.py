"""Dashboard section "Model settings": the settings the model was fitted with, in plain words."""

from __future__ import annotations

import html
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from prophet.dashboard import _style
from prophet.dashboard._context import DashboardContext, Section
from prophet.dashboard.data import iso_stamps
from prophet.forecaster import Prophet
from prophet.visuals import _fmt_day, _fmt_number, _fmt_pct, _plural

KEY = "settings"
TITLE = "Model settings"

#: Built-in seasonalities: period in days and the Fourier order Prophet uses by default.
_BUILTIN: Mapping[str, tuple[float, int]] = {
    "yearly": (365.25, 10),
    "weekly": (7, 3),
    "daily": (1, 4),
}
_BACKENDS: Mapping[str, str] = {
    "CMDSTANPY": "CmdStan through CmdStanPy",
    "MLX": "MLX (experimental)",
}
#: Holiday frames longer than this are not written out in the recreate block.
_MAX_HOLIDAY_ROWS = 40
_MAX_LINE = 88


def init_defaults() -> dict[str, Any]:
    """Default value of every ``Prophet.__init__`` parameter, read from its signature."""
    parameters = inspect.signature(Prophet.__init__).parameters
    return {name: p.default for name, p in parameters.items() if name != "self"}


# ----------------------------------------------------------------------------
# Pure: wording
# ----------------------------------------------------------------------------

_PERIOD_UNITS: tuple[tuple[str, float], ...] = (
    ("year", 365.25),
    ("month", 30.4375),
    ("week", 7.0),
    ("day", 1.0),
    ("hour", 1 / 24),
    ("minute", 1 / 1440),
)


def period_words(days: float) -> str:
    """A seasonal period in days as plain words: ``1 year``, ``1 week``, ``12 hours``.

    A period close to a whole number of a unit (within half a percent) takes
    that unit; anything else stays in days.
    """
    for unit, size in _PERIOD_UNITS:
        if size < 1 and days >= 2:
            continue
        ratio = days / size
        n = round(ratio)
        if n >= 1 and abs(ratio - n) <= 0.005 * n:
            return _plural(n, unit)
    return f"{days:.3g} days"


def against_default(value: float, default: float, high: str, low: str) -> str:
    """A number next to the default: ``0.5, more flexible than the default (0.05)``."""
    if math.isclose(value, default):
        return f"{value:g}, the default"
    side = high if value > default else low
    return f"{value:g}, {side} than the default ({default:g})"


def join_words(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def mode_words(mode: str) -> str:
    return {
        "additive": "Additive: effects are added to the trend",
        "multiplicative": "Multiplicative: effects scale with the trend",
    }.get(mode, mode)


def backend_label(m: Prophet) -> str | None:
    """Readable name of the Stan backend, or None when the model has none."""
    backend = m.stan_backend
    if backend is None:
        return None
    kind = backend.get_type()
    return _BACKENDS.get(kind, kind)


def growth_words(growth: str) -> str:
    return {
        "linear": "Linear: a trend that can change slope at changepoints",
        "logistic": "Logistic: growth that levels off at a capacity",
        "flat": "Flat: a constant level with no slope",
    }.get(growth, growth)


def _range_words(series: pd.Series) -> str:
    lo, hi = float(series.min()), float(series.max())
    if lo == hi:
        return _fmt_number(lo)
    return f"{_fmt_number(lo)} to {_fmt_number(hi)}, varying by date"


def floor_words(m: Prophet, history: pd.DataFrame) -> str:
    """The floor of a logistic trend: a column, the lowest y (min-max scaling), or 0."""
    if m.logistic_floor and "floor" in history:
        return _range_words(history["floor"])
    if m.scaling == "minmax" and m.y_min is not None:
        return f"{_fmt_number(float(m.y_min))}, the lowest value of y (min-max scaling)"
    return "0, the default"


def trend_pairs(
    m: Prophet, history: pd.DataFrame | None, step: pd.Timedelta
) -> list[tuple[str, str]]:
    """Growth, capacity and floor, and the changepoints."""
    defaults = init_defaults()
    pairs = [("Growth", growth_words(m.growth))]
    if m.growth == "logistic" and history is not None:
        if "cap" in history:
            pairs.append(("Capacity (cap)", _range_words(history["cap"])))
        pairs.append(("Floor", floor_words(m, history)))
    if m.growth == "flat":
        pairs.append(("Changepoints", "Not used: a flat trend has no slope to change"))
        return pairs
    cps = m.changepoints
    if cps is None or len(cps) == 0:
        text = "None: the trend never changes slope"
    elif m.specified_changepoints:
        text = (
            f"{len(cps)} chosen by hand, from {_fmt_day(cps.min(), step)} "
            f"to {_fmt_day(cps.max(), step)}"
        )
    else:
        text = (
            f"{len(cps)} candidate dates over the first {_fmt_pct(m.changepoint_range)} "
            f"of the history, up to {_fmt_day(cps.max(), step)}"
        )
    pairs.append(("Changepoints", text))
    pairs.append(
        (
            "Changepoint prior scale",
            against_default(
                m.changepoint_prior_scale,
                defaults["changepoint_prior_scale"],
                "more flexible",
                "stiffer",
            ),
        )
    )
    return pairs


def fitting_pairs(m: Prophet) -> list[tuple[str, str]]:
    """How the model was fitted and how its intervals are made."""
    if m.mcmc_samples > 0:
        method = f"MCMC sampling with {m.mcmc_samples:,} samples"
    else:
        method = "MAP optimisation, a single best-fit estimate"
    if m.uncertainty_samples:
        intervals = (
            f"{_fmt_pct(m.interval_width)} intervals from "
            f"{m.uncertainty_samples:,} simulated futures"
        )
    else:
        intervals = "Not estimated: uncertainty samples is 0"
    pairs = [("Fitting method", method), ("Uncertainty", intervals)]
    backend = backend_label(m)
    if backend is not None:
        pairs.append(("Backend", backend))
    pairs.append(("Scaling", scaling_words(m)))
    return pairs


def scaling_words(m: Prophet) -> str:
    """How y is rescaled before fitting; with a floor column the floor is taken off first."""
    scale = None if m.y_scale is None else _fmt_number(float(m.y_scale))
    shifted = "y minus the floor" if m.logistic_floor else "y"
    if m.scaling == "minmax":
        if m.logistic_floor:
            by = "its range" if scale is None else scale
            return f"Min-max: y minus the floor is divided by {by}"
        low = None if m.y_min is None else _fmt_number(float(m.y_min))
        if scale is None or low is None:
            return "Min-max: y is shifted by its minimum and divided by its range"
        return f"Min-max: y is shifted by {low} and divided by {scale}"
    if scale is None:
        return f"Absolute maximum: {shifted} is divided by its largest absolute value"
    return f"Absolute maximum: {shifted} is divided by {scale}"


def seasonality_defaults_pairs(m: Prophet) -> list[tuple[str, str]]:
    defaults = init_defaults()
    return [
        ("Default mode", mode_words(m.seasonality_mode)),
        (
            "Default prior scale",
            against_default(
                m.seasonality_prior_scale,
                defaults["seasonality_prior_scale"],
                "looser",
                "tighter",
            ),
        ),
    ]


SEASONALITY_HEADERS = [
    "Name",
    "Period",
    "Mode",
    "Condition",
    "Fourier order",
    "Prior scale",
]
REGRESSOR_HEADERS = ["Name", "Mode", "Standardised", "Prior scale"]


def seasonality_rows(
    seasonalities: Mapping[str, Mapping[str, Any]],
) -> list[list[str]]:
    """One row per seasonality, in the order of :data:`SEASONALITY_HEADERS` (numbers last)."""
    return [
        [
            name,
            period_words(float(props["period"])),
            str(props["mode"]),
            str(props.get("condition_name") or "none"),
            str(props["fourier_order"]),
            f"{props['prior_scale']:g}",
        ]
        for name, props in seasonalities.items()
    ]


def window_words(lower: pd.Series, upper: pd.Series) -> str:
    """The days a holiday reaches before and after its date.

    One phrase when every holiday has the same window, else the widest window
    on each side, which no single holiday need have.
    """
    before = -pd.to_numeric(lower).fillna(0).astype(int)
    after = pd.to_numeric(upper).fillna(0).astype(int)
    if before.nunique() <= 1 and after.nunique() <= 1:
        b, a = int(before.iloc[0]), int(after.iloc[0])
        if b == 0 and a == 0:
            return "None: only the day itself"
        left = "the day itself" if b == 0 else f"{_plural(b, 'day')} before"
        right = "the day itself" if a == 0 else f"{_plural(a, 'day')} after"
        return f"{left} to {right}"
    widest = [
        f"up to {_plural(int(n), 'day')} {side}"
        for n, side in ((before.max(), "before"), (after.max(), "after"))
        if n
    ]
    return "Varies by holiday: " + " and ".join(widest)


def holiday_pairs(m: Prophet) -> list[tuple[str, str]]:
    """Holiday count, where they come from, and their prior scale and mode."""
    own = m.holidays
    if own is None and m.country_holidays is None:
        return [("Holidays", "None")]
    names = m.train_holiday_names
    pairs = [("Holidays", _plural(0 if names is None else len(names), "holiday name"))]
    if own is not None:
        pairs.append(
            (
                "Your holidays",
                f"{_plural(own['holiday'].nunique(), 'name')}, {_plural(len(own), 'date')}",
            )
        )
        if "lower_window" in own and "upper_window" in own:
            pairs.append(
                (
                    "Days around each date",
                    window_words(own["lower_window"], own["upper_window"]),
                )
            )
        if "prior_scale" in own and own["prior_scale"].notna().any():
            lo, hi = own["prior_scale"].min(), own["prior_scale"].max()
            pairs.append(
                (
                    "Per-holiday prior scales",
                    f"{lo:g}" if lo == hi else f"{lo:g} to {hi:g}",
                )
            )
    pairs.append(("Country holidays", m.country_holidays or "None"))
    defaults = init_defaults()
    pairs.append(
        (
            "Holiday prior scale",
            against_default(
                m.holidays_prior_scale,
                defaults["holidays_prior_scale"],
                "looser",
                "tighter",
            ),
        )
    )
    pairs.append(("Holiday mode", mode_words(m.holidays_mode)))
    return pairs


def standardised_words(props: Mapping[str, Any]) -> str:
    """Whether a regressor was standardised, reading the result Prophet left behind."""
    if props.get("mu", 0.0) != 0.0 or props.get("std", 1.0) != 1.0:
        return "yes"
    return "no (binary or constant)" if props.get("standardize") == "auto" else "no"


def regressor_rows(
    regressors: Mapping[str, Mapping[str, Any]], units: Mapping[str, str]
) -> list[list[str]]:
    """One row per extra regressor, in the order of :data:`REGRESSOR_HEADERS` (numbers last)."""
    return [
        [
            name + (f" ({units[name]})" if name in units else ""),
            str(props["mode"]),
            standardised_words(props),
            f"{props['prior_scale']:g}",
        ]
        for name, props in regressors.items()
    ]


def lede_text(m: Prophet) -> str:
    """Two short sentences that sum the model up."""
    growth = {"linear": "Linear", "logistic": "Logistic", "flat": "Flat"}.get(
        m.growth, m.growth
    )
    n = 0 if m.changepoints is None else len(m.changepoints)
    if m.growth == "flat":
        trend = f"{growth} growth"
    else:
        trend = f"{growth} growth with {_plural(n, 'changepoint') if n else 'no changepoints'}"
    extras: list[str] = []
    names = list(m.seasonalities)
    if names:
        count = _plural(len(names), "seasonal pattern")
        extras.append(f"{count} ({join_words(names)})" if len(names) <= 3 else count)
    if m.train_holiday_names is not None and len(m.train_holiday_names):
        extras.append(_plural(len(m.train_holiday_names), "holiday"))
    if m.extra_regressors:
        extras.append(_plural(len(m.extra_regressors), "extra regressor"))
    plus = (
        f", plus {join_words(extras)}"
        if extras
        else ", with no seasonality, holidays or regressors"
    )
    method = (
        f"MCMC sampling ({m.mcmc_samples:,} samples)"
        if m.mcmc_samples > 0
        else "MAP optimisation"
    )
    return f"{trend}{plus}. It was fitted by {method}."


# ----------------------------------------------------------------------------
# Pure: the code that recreates the model
# ----------------------------------------------------------------------------


def py_literal(value: object) -> str:
    """Python source for a number, string, boolean, None or list of these."""
    if value is None:
        return "None"
    if isinstance(value, (bool, np.bool_)):
        return repr(bool(value))
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, np.integer)):
        return repr(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{value!r} has no Python literal.")
        return repr(float(value))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(py_literal(v) for v in value) + "]"
    raise TypeError(f"No Python literal for {type(value).__name__}.")


def _packed(items: Sequence[str], indent: str) -> list[str]:
    """Items joined by commas into lines no longer than the line limit."""
    lines: list[str] = []
    current = indent
    for item in items:
        piece = f"{item}, "
        if current.strip() and len(current) + len(piece) > _MAX_LINE:
            lines.append(current.rstrip())
            current = indent
        current += piece
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _list_lines(
    prefix: str,
    items: Sequence[str],
    indent: str,
    suffix: str = ",",
    wrap: tuple[str, str] = ("", ""),
) -> list[str]:
    """``prefix[a, b, c]`` on one line when it fits, else packed over several lines.

    ``wrap`` goes around the bracket, for ``pd.to_datetime([...])``.
    """
    head, tail = wrap
    one = f"{indent}{prefix}{head}[{', '.join(items)}]{tail}{suffix}"
    if len(one) <= _MAX_LINE:
        return [one]
    return [
        f"{indent}{prefix}{head}[",
        *_packed(items, indent + "    "),
        f"{indent}]{tail}{suffix}",
    ]


def call_source(callee: str, args: Sequence[tuple[str | None, str]]) -> str:
    """A call on one line when it fits, else one argument per line."""
    items = [value if name is None else f"{name}={value}" for name, value in args]
    line = f"{callee}({', '.join(items)})"
    if len(line) <= _MAX_LINE:
        return line
    body = "".join(f"    {item},\n" for item in items)
    return f"{callee}(\n{body})"


def _native(value: object) -> object:
    """A pandas or numpy scalar as a plain Python value, NaN as None."""
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def holidays_source(holidays: pd.DataFrame) -> tuple[list[str], bool]:
    """Lines that define ``holidays``, and whether they use ``pd``.

    A long frame is not written out: the lines then only explain what to define.
    """
    rows, names = len(holidays), int(holidays["holiday"].nunique())
    if rows > _MAX_HOLIDAY_ROWS:
        return (
            [
                f"# holidays: define the DataFrame you passed to Prophet ({rows} rows, "
                f"{names} holidays) before the next line."
            ],
            False,
        )
    columns = ["holiday", "ds"] + [
        c for c in ("lower_window", "upper_window", "prior_scale") if c in holidays
    ]
    lines = ["holidays = pd.DataFrame(", "    {"]
    indent = "        "
    for column in columns:
        if column == "ds":
            items = [py_literal(stamp) for stamp in iso_stamps(holidays["ds"])]
            lines += _list_lines(
                f'"{column}": ', items, indent, wrap=("pd.to_datetime(", ")")
            )
        else:
            items = [py_literal(_native(v)) for v in holidays[column].tolist()]
            lines += _list_lines(f'"{column}": ', items, indent)
    return [*lines, "    }", ")"], True


def _builtin_like(m: Prophet, name: str, props: Mapping[str, Any]) -> bool:
    """True when the constructor arguments alone would create this seasonality."""
    if name not in _BUILTIN:
        return False
    period, default_order = _BUILTIN[name]
    arg = getattr(m, f"{name}_seasonality")
    if arg == "auto" or arg is True:
        order = default_order
    elif arg is False:
        order = 0
    else:
        order = int(arg)
    return (
        props["period"] == period
        and props["fourier_order"] == order
        and props["prior_scale"] == m.seasonality_prior_scale
        and props["mode"] == m.seasonality_mode
        and props["condition_name"] is None
    )


def changepoints_cut_by_history(m: Prophet) -> bool:
    """Prophet lowered ``n_changepoints`` to fit a short history.

    It does so when there are fewer rows in the changepoint range than
    ``n_changepoints + 1``, and keeps the lower number on the model. That value
    is then a result of the history, not a setting someone chose.
    """
    if m.specified_changepoints or m.history is None:
        return False
    default = init_defaults()["n_changepoints"]
    room = math.floor(len(m.history) * m.changepoint_range)
    return bool(m.n_changepoints < default and m.n_changepoints + 1 == room)


def constructor_args(m: Prophet) -> list[tuple[str, str]]:
    """The ``Prophet(...)`` arguments that differ from the defaults, as name and source."""
    args: list[tuple[str, str]] = []
    for name, default in init_defaults().items():
        if name == "changepoints":
            if m.specified_changepoints:
                args.append((name, "changepoints"))
        elif name == "n_changepoints":
            if (
                not m.specified_changepoints
                and m.n_changepoints != default
                and not changepoints_cut_by_history(m)
            ):
                args.append((name, py_literal(m.n_changepoints)))
        elif name == "holidays":
            if m.holidays is not None:
                args.append((name, "holidays"))
        elif name == "stan_backend":
            if m.stan_backend is not None and m.stan_backend.get_type() == "MLX":
                args.append((name, py_literal("MLX")))
        elif name == "holidays_mode":
            if m.holidays_mode != m.seasonality_mode:
                args.append((name, py_literal(m.holidays_mode)))
        else:
            value = getattr(m, name)
            if value != default:
                args.append((name, py_literal(value)))
    return args


def seasonality_calls(m: Prophet) -> list[str]:
    """``add_seasonality`` calls for every seasonality the constructor would not create."""
    calls = []
    for name, props in m.seasonalities.items():
        if _builtin_like(m, name, props):
            continue
        args: list[tuple[str | None, str]] = [
            ("name", py_literal(name)),
            ("period", py_literal(props["period"])),
            ("fourier_order", py_literal(props["fourier_order"])),
        ]
        if props["prior_scale"] != m.seasonality_prior_scale:
            args.append(("prior_scale", py_literal(props["prior_scale"])))
        if props["mode"] != m.seasonality_mode:
            args.append(("mode", py_literal(props["mode"])))
        if props["condition_name"] is not None:
            args.append(("condition_name", py_literal(props["condition_name"])))
        calls.append(call_source("m.add_seasonality", args))
    return calls


def regressor_calls(m: Prophet) -> list[str]:
    """``add_regressor`` calls, with only the arguments that differ from the defaults."""
    calls = []
    for name, props in m.extra_regressors.items():
        args: list[tuple[str | None, str]] = [(None, py_literal(name))]
        if props["prior_scale"] != m.holidays_prior_scale:
            args.append(("prior_scale", py_literal(props["prior_scale"])))
        if props["standardize"] != "auto":
            args.append(("standardize", py_literal(props["standardize"])))
        if props["mode"] != m.seasonality_mode:
            args.append(("mode", py_literal(props["mode"])))
        calls.append(call_source("m.add_regressor", args))
    return calls


def data_columns(m: Prophet) -> list[str]:
    """Columns the data frame handed to ``fit`` needs."""
    columns = ["ds", "y"]
    if m.growth == "logistic":
        columns.append("cap")
    if m.logistic_floor:
        columns.append("floor")
    columns += list(m.extra_regressors)
    columns += [
        c
        for c in dict.fromkeys(
            props["condition_name"] for props in m.seasonalities.values()
        )
        if c is not None
    ]
    return columns


def recreate_code(m: Prophet) -> str:
    """Python that builds an unfitted ``Prophet`` with the same settings as ``m``.

    Only constructor arguments that differ from the defaults are written, then
    the ``add_*`` calls for custom seasonalities, regressors and country holidays.
    """
    lines: list[str] = []
    setup: list[str] = []
    uses_pandas = False
    if m.holidays is not None:
        setup, uses_pandas = holidays_source(m.holidays)
    args: list[tuple[str | None, str]] = list(constructor_args(m))
    prelude: list[str] = []
    if m.specified_changepoints:
        stamps = [py_literal(s) for s in iso_stamps(pd.Series(m.changepoints))]
        prelude = _list_lines("changepoints = ", stamps, "", suffix="")
    if uses_pandas:
        lines += ["import pandas as pd", ""]
    lines.append("from prophet import Prophet")
    lines.append("")
    if setup:
        lines += [*setup, ""]
    if prelude:
        lines += [*prelude, ""]
    if changepoints_cut_by_history(m):
        lines.append(
            f"# Prophet cut n_changepoints to {m.n_changepoints} to fit the short "
            "history, so it is left out here."
        )
    lines.append(call_source("m = Prophet", args))
    if m.country_holidays is not None:
        lines.append(
            call_source(
                "m.add_country_holidays", [(None, py_literal(m.country_holidays))]
            )
        )
    lines += seasonality_calls(m)
    lines += regressor_calls(m)
    lines += [
        "",
        f"# Fit with m.fit(df), where df has the columns: {', '.join(data_columns(m))}.",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------


def build(ctx: DashboardContext) -> Section | None:
    """The section for a fitted model."""
    m = ctx.m
    history = m.history
    if history is None:
        return None
    parts = [_style.lede(lede_text(m))]
    left = _style.h3("Trend") + _style.kv(trend_pairs(m, history, ctx.step))
    right = _style.h3("Fitting") + _style.kv(fitting_pairs(m))
    parts.append(_style.grid2(left, right))

    parts.append(_style.h3("Seasonality"))
    parts.append(_style.kv(seasonality_defaults_pairs(m)))
    if m.seasonalities:
        parts.append(
            _style.table(
                SEASONALITY_HEADERS,
                seasonality_rows(m.seasonalities),
                numeric=[4, 5],
            )
        )
        parts.append(
            _style.note(
                "A higher Fourier order lets a pattern change shape faster. "
                "A condition is a true or false column that switches the pattern on and off."
            )
        )
    else:
        parts.append(_style.note("The model has no seasonal patterns."))

    parts.append(_style.h3("Holidays"))
    parts.append(_style.kv(holiday_pairs(m)))

    parts.append(_style.h3("Extra regressors"))
    if m.extra_regressors:
        parts.append(
            _style.table(
                REGRESSOR_HEADERS,
                regressor_rows(m.extra_regressors, ctx.units),
                numeric=[3],
            )
        )
    else:
        parts.append(_style.note("The model has no extra regressors."))

    parts.append(_style.h3("Recreate this model"))
    parts.append(
        _style.note(
            "This builds an unfitted model with the same settings. Only values that "
            "differ from the Prophet defaults are written."
        )
    )
    parts.append(
        f'<pre class="code"><code>{html.escape(recreate_code(m))}</code></pre>'
    )
    return Section(key=KEY, title=TITLE, body="".join(parts))
