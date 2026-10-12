"""High-level, one-call forecasting API on top of :class:`prophet.Prophet`.

The functions here are additive: they never change the behaviour of the
existing ``Prophet`` class.  The module follows a functional-core,
imperative-shell layout:

* Pure core: :func:`normalise_input`, :func:`infer_frequency` and
  :func:`horizon_to_periods` validate and transform data without touching
  the model.
* Imperative shell: :func:`forecast` builds, fits and runs the model and
  wraps the output in an immutable :class:`ForecastResult`.

Beyond the basics, :func:`forecast` takes ``regressors`` (extra columns or
series, optionally with per-regressor settings), ``weather`` (a location whose
temperature becomes a regressor named ``temperature``) and ``skip_days`` /
``skip_hours`` (periods the series never has, left out of training, the
forecast and, by default, the charts).

The returned :class:`ForecastResult` also draws and checks itself:
``result.plot()``, ``result.backtest()``, ``result.plot_backtest()``,
``result.plot_regressor_effects()``, ``result.report()`` and
``result.dashboard(path)`` all respect the skipped days and, with ``weather``,
label each future temperature as weather forecast or climatology.

Example
-------
>>> import pandas as pd
>>> from prophet import forecast
>>> index = pd.date_range("2024-01-01", periods=60, freq="D")
>>> series = pd.Series(range(60), index=index, dtype=float)
>>> result = forecast(series, horizon=7)  # doctest: +SKIP
>>> result.forecast.columns.tolist()  # doctest: +SKIP
['ds', 'yhat', 'yhat_lower', 'yhat_upper']
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

from prophet.forecaster import Prophet
from prophet.plot import (
    SkipAxis,
    SkipSpecError,
    _hour_runs,
    parse_skip_days,
    parse_skip_hours,
)

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.pyplot as plt

    from prophet.tuning import TuningResult
    from prophet.visuals import Backtest
    from prophet.weather import WeatherResult

logger = logging.getLogger("prophet.quick")

__all__ = [
    "AllNullValuesError",
    "ColumnSelectionError",
    "DuplicateTimestampsError",
    "ForecastInputError",
    "ForecastResult",
    "ForecastSummary",
    "FrequencyInference",
    "FrequencyInferenceError",
    "HorizonError",
    "InsufficientDataError",
    "MissingRegressorValuesError",
    "NonNumericValuesError",
    "RegressorError",
    "SkipDaysError",
    "UnparseableDatesError",
    "forecast",
    "horizon_to_periods",
    "infer_frequency",
    "infer_frequency_details",
    "normalise_input",
    "weather_source_labels",
]

TimezoneMode = Literal["wall", "utc"]
FrequencyMethod = Literal["infer_freq", "modal_difference"]
SkipSpec = str | int | Iterable[str | int] | None

#: Name of the regressor created by ``forecast(..., weather=...)``.
WEATHER_REGRESSOR = "temperature"


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ForecastInputError(ValueError):
    """Base class for every problem with the data passed to :func:`forecast`."""


class ColumnSelectionError(ForecastInputError):
    """The date or value column is missing or cannot be chosen unambiguously."""


class UnparseableDatesError(ForecastInputError):
    """Some dates are missing or cannot be parsed as timestamps."""


class DuplicateTimestampsError(ForecastInputError):
    """The same timestamp appears more than once."""


class InsufficientDataError(ForecastInputError):
    """Fewer than two usable (non-null) observations are available."""


class AllNullValuesError(InsufficientDataError):
    """Every value in the series is null."""


class NonNumericValuesError(ForecastInputError):
    """The value column holds non-numeric or non-finite values."""


class FrequencyInferenceError(ForecastInputError):
    """The sampling frequency cannot be determined or is invalid."""


class HorizonError(ForecastInputError):
    """The forecast horizon is not a positive integer or a valid duration."""


class SkipDaysError(ForecastInputError):
    """``skip_days`` or ``skip_hours`` is invalid, or leaves no data to use."""


class RegressorError(ForecastInputError):
    """A regressor, its options or the ``weather`` location is badly specified."""


class MissingRegressorValuesError(RegressorError):
    """A regressor has no value for some training or forecast timestamps."""


# --------------------------------------------------------------------------
# Input normalisation (pure)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Normalised:
    frame: pd.DataFrame
    notes: tuple[str, ...]


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(
        s.dtype
    )


def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s.dtype)


def _parses_as_dates(s: pd.Series) -> bool:
    """Whether a text column parses fully as dates (used only to pick columns)."""
    if _is_numeric(s) or _is_datetime(s):
        return False
    non_null = s.dropna()
    if non_null.empty:
        return False
    try:
        pd.to_datetime(non_null)
    except (ValueError, TypeError):
        return False
    return True


def _require_column(df: pd.DataFrame, name: str, role: str) -> str:
    if name not in df.columns:
        raise ColumnSelectionError(
            f"{role}={name!r} is not a column of the data. Available columns: {list(df.columns)}."
        )
    return name


def _pick_unique(
    candidates: list[str], role: str, available: list[str], hint: str
) -> str:
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ColumnSelectionError(
            f"Could not find a {role} column among {available}. {hint}"
        )
    raise ColumnSelectionError(
        f"Found several possible {role} columns {candidates}. {hint}"
    )


def _choose_columns(
    df: pd.DataFrame,
    date_col: str | None,
    value_col: str | None,
    reserved: Sequence[str] = (),
) -> tuple[str | None, str]:
    """Return ``(date column or None for the index, value column)``.

    ``reserved`` columns (regressors) are never chosen automatically.
    """
    cols = [str(c) for c in df.columns]
    if len(set(cols)) != len(cols):
        raise ColumnSelectionError(f"Column names must be unique, got {cols}.")
    has_dt_index = isinstance(df.index, pd.DatetimeIndex)

    if date_col is not None:
        chosen_date: str | None = _require_column(df, date_col, "date_col")
    elif "ds" in df.columns:
        chosen_date = "ds"
    else:
        dt_cols = [c for c in df.columns if c not in reserved and _is_datetime(df[c])]
        if not dt_cols and not has_dt_index:
            dt_cols = [
                c for c in df.columns if c not in reserved and _parses_as_dates(df[c])
            ]
        if has_dt_index and not dt_cols:
            chosen_date = None
        elif has_dt_index:
            raise ColumnSelectionError(
                f"The data has a DatetimeIndex and a datetime column {dt_cols}. "
                "Pass date_col=... to use the column, or drop it to use the index."
            )
        else:
            chosen_date = _pick_unique(
                dt_cols,
                "date",
                cols,
                "Pass date_col=... to choose one, or give the data a DatetimeIndex.",
            )

    if value_col is not None:
        return chosen_date, _require_column(df, value_col, "value_col")
    if "y" in df.columns and chosen_date != "y":
        return chosen_date, "y"
    num_cols = [
        c
        for c in df.columns
        if c != chosen_date and c not in reserved and _is_numeric(df[c])
    ]
    return chosen_date, _pick_unique(
        num_cols, "numeric value", cols, "Pass value_col=... to choose one."
    )


def _to_naive(ds: pd.Series, tz: TimezoneMode, notes: list[str]) -> pd.Series:
    if ds.dt.tz is None:
        return ds
    if tz == "utc":
        notes.append(
            f"Dates are timezone-aware ({ds.dt.tz}); converted to UTC and made "
            "timezone-naive because Prophet needs naive timestamps."
        )
        return ds.dt.tz_convert("UTC").dt.tz_localize(None)
    notes.append(
        f"Dates are timezone-aware ({ds.dt.tz}); kept local wall-clock time and "
        "dropped the timezone because Prophet needs naive timestamps."
    )
    return ds.dt.tz_localize(None)


def _parse_dates(raw: pd.Series) -> pd.Series:
    if _is_datetime(raw):
        return raw
    if _is_numeric(raw):
        raise UnparseableDatesError(
            f"The date column {raw.name!r} holds numbers, which are ambiguous as "
            "timestamps. Convert it first, e.g. pd.to_datetime(col, unit='s')."
        )
    try:
        parsed = pd.to_datetime(raw)
    except (ValueError, TypeError) as exc:
        raise UnparseableDatesError(
            f"Could not parse the date column {raw.name!r} as timestamps: {exc}. "
            "Use ISO 8601 strings (YYYY-MM-DD) or convert with pd.to_datetime(..., format=...)."
        ) from exc
    # Mixed-offset strings yield an object column of datetimes in pandas.
    if not _is_datetime(parsed):
        raise UnparseableDatesError(
            f"Dates in column {raw.name!r} mix different timezone offsets. "
            "Convert with pd.to_datetime(col, utc=True) first."
        )
    return parsed


def _coerce_values(raw: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(raw.dtype):
        raise NonNumericValuesError(
            f"Value column {raw.name!r} is boolean; convert it with .astype(float) if intended."
        )
    if not _is_numeric(raw):
        try:
            raw = pd.to_numeric(raw)
        except (ValueError, TypeError) as exc:
            raise NonNumericValuesError(
                f"Value column {raw.name!r} (dtype {raw.dtype}) is not numeric: {exc}. "
                "Clean or convert it before forecasting."
            ) from exc
    values = raw.astype("float64")
    if bool(np.isinf(values.to_numpy()).any()):
        raise NonNumericValuesError(
            f"Value column {raw.name!r} contains infinite values. Replace them with "
            "NaN (they will be dropped and logged) or remove them."
        )
    return values


def _normalise(
    data: pd.DataFrame | pd.Series,
    date_col: str | None,
    value_col: str | None,
    tz: TimezoneMode,
    extra: Sequence[str] = (),
) -> _Normalised:
    """Clean ``ds``/``y`` frame, plus the ``extra`` regressor columns as floats (NaN allowed)."""
    notes: list[str] = []
    extra_raw: pd.DataFrame | None = None
    if isinstance(data, pd.Series):
        if date_col is not None or value_col is not None:
            raise ColumnSelectionError(
                "date_col and value_col only apply to DataFrame input."
            )
        if extra:
            raise RegressorError(
                f"regressors={list(extra)} name columns of the data, but the data is a "
                "Series. Pass a DataFrame, or give the values as a mapping "
                "{name: pandas.Series}."
            )
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ColumnSelectionError(
                "A Series must have a DatetimeIndex, got "
                f"{type(data.index).__name__}. Use series.index = pd.to_datetime(...) "
                "or pass a DataFrame with a date column."
            )
        raw_ds: pd.Series = pd.Series(data.index, name="ds")
        raw_y: pd.Series = data.reset_index(drop=True)
    elif isinstance(data, pd.DataFrame):
        chosen_date, chosen_value = _choose_columns(data, date_col, value_col, extra)
        if chosen_date is None:
            raw_ds = pd.Series(data.index, name="ds")
        else:
            raw_ds = data[chosen_date].reset_index(drop=True)
        raw_y = data[chosen_value].reset_index(drop=True)
        for name in extra:
            if name not in data.columns:
                raise RegressorError(
                    f"regressor {name!r} is not a column of the data. Available "
                    f"columns: {list(data.columns)}. To supply values from outside "
                    "the data, pass a mapping {name: pandas.Series} instead."
                )
            if name in (chosen_date, chosen_value):
                raise RegressorError(
                    f"{name!r} is the {'date' if name == chosen_date else 'value'} "
                    "column, so it cannot also be a regressor."
                )
        if extra:
            extra_raw = pd.DataFrame(
                {
                    name: _coerce_regressor(name, data[name].reset_index(drop=True))
                    for name in extra
                }
            )
    else:
        raise ForecastInputError(
            f"data must be a pandas DataFrame or Series, got {type(data).__name__}."
        )

    ds = _parse_dates(raw_ds)
    if bool(ds.isna().any()):
        raise UnparseableDatesError(
            f"{int(ds.isna().sum())} date(s) are missing. Remove or fill them before forecasting."
        )
    ds = _to_naive(ds, tz, notes)
    y = _coerce_values(raw_y)

    dup = ds[ds.duplicated(keep=False)]
    if not dup.empty:
        examples = ", ".join(str(t) for t in dup.drop_duplicates().head(3))
        raise DuplicateTimestampsError(
            f"{int(ds.duplicated().sum())} duplicate timestamp(s), e.g. {examples}. "
            "Aggregate first, e.g. df.groupby('date').sum()."
        )

    n_null = int(y.isna().sum())
    if n_null == len(y):
        raise AllNullValuesError(
            "Every value is null, so there is nothing to forecast."
        )
    if n_null:
        notes.append(f"Dropped {n_null} row(s) with null values.")
    base = pd.DataFrame({"ds": ds, "y": y})
    if extra_raw is not None:
        base = pd.concat([base, extra_raw], axis=1)
    frame = base.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)
    if len(frame) < 2:
        raise InsufficientDataError(
            f"Only {len(frame)} non-null observation(s); at least 2 are required."
        )
    return _Normalised(frame, tuple(notes))


def normalise_input(
    data: pd.DataFrame | pd.Series,
    *,
    date_col: str | None = None,
    value_col: str | None = None,
    tz: TimezoneMode = "wall",
) -> pd.DataFrame:
    """Turn user data into the clean ``ds``/``y`` frame Prophet expects.

    Accepted shapes:

    * a DataFrame with ``ds`` and ``y`` columns;
    * a DataFrame with one datetime-like and one numeric column of any name,
      or explicit ``date_col`` / ``value_col``;
    * a DataFrame with a ``DatetimeIndex`` and one numeric column;
    * a Series with a ``DatetimeIndex``.

    Rows with null values are dropped, the result is sorted by date and
    every adjustment is logged at WARNING level on the ``prophet.quick``
    logger.

    Parameters
    ----------
    data: The input data.
    date_col: Name of the date column, if it cannot be inferred.
    value_col: Name of the value column, if it cannot be inferred.
    tz: How to make timezone-aware dates naive. ``"wall"`` (default) keeps
        the local wall-clock time; ``"utc"`` converts to UTC first.

    Returns
    -------
    A new DataFrame with columns ``ds`` (datetime64) and ``y`` (float64).

    Raises
    ------
    ColumnSelectionError, UnparseableDatesError, DuplicateTimestampsError,
    InsufficientDataError, AllNullValuesError, NonNumericValuesError
        All subclasses of :class:`ForecastInputError` (a ``ValueError``).

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1.0, 2.0, 4.0], index=pd.date_range("2024-01-01", periods=3))
    >>> normalise_input(s)
              ds    y
    0 2024-01-01  1.0
    1 2024-01-02  2.0
    2 2024-01-03  4.0
    """
    if tz not in ("wall", "utc"):
        raise ForecastInputError(f"tz must be 'wall' or 'utc', got {tz!r}.")
    result = _normalise(data, date_col, value_col, tz)
    for note in result.notes:
        logger.warning(note)
    return result.frame


# --------------------------------------------------------------------------
# Frequency inference (pure)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyInference:
    """Outcome of :func:`infer_frequency_details`."""

    freq: str
    method: FrequencyMethod

    @property
    def regular(self) -> bool:
        """True when pandas found an exact, gap-free frequency."""
        return self.method == "infer_freq"


_TIME_UNITS: tuple[tuple[str, pd.Timedelta], ...] = (
    ("D", pd.Timedelta(days=1)),
    ("h", pd.Timedelta(hours=1)),
    ("min", pd.Timedelta(minutes=1)),
    ("s", pd.Timedelta(seconds=1)),
    ("ms", pd.Timedelta(milliseconds=1)),
)


def _timedelta_alias(step: pd.Timedelta) -> str:
    for unit, size in _TIME_UNITS:
        if step % size == pd.Timedelta(0):
            n = step // size
            return unit if n == 1 else f"{n}{unit}"
    # Sub-millisecond spacing: fall back to nanoseconds, which pandas accepts.
    return f"{step.value}ns"


def _calendar_alias(dates: pd.DatetimeIndex, step: pd.Timedelta) -> str | None:
    """Month-anchored alias when the modal gap is about a month and dates sit on month edges."""
    if not pd.Timedelta(days=28) <= step <= pd.Timedelta(days=31):
        return None
    if bool((dates.is_month_start).all()):
        return "MS"
    if bool((dates.is_month_end).all()):
        return "ME"
    return None


def infer_frequency_details(dates: pd.Series | pd.DatetimeIndex) -> FrequencyInference:
    """Detect the sampling frequency of a set of timestamps.

    Uses :func:`pandas.infer_freq` first.  For irregular data (gaps, missing
    periods) it falls back to the most common spacing between consecutive
    timestamps, mapped to a pandas alias; monthly-ish spacing on month
    starts or ends gives ``"MS"`` / ``"ME"``.

    Raises
    ------
    FrequencyInferenceError: With fewer than 2 distinct timestamps.
    """
    index = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.drop_duplicates()
    if len(index) < 2:
        raise FrequencyInferenceError(
            "Need at least 2 distinct timestamps to infer a frequency; pass freq=... explicitly."
        )
    if len(index) >= 3:
        found = pd.infer_freq(index)
        if found is not None:
            return FrequencyInference(found, "infer_freq")
    diffs = pd.Series(index[1:] - index[:-1])
    step = pd.Timedelta(diffs.mode().min())
    alias = _calendar_alias(index, step) or _timedelta_alias(step)
    return FrequencyInference(alias, "modal_difference")


def infer_frequency(dates: pd.Series | pd.DatetimeIndex) -> str:
    """Return a pandas offset alias for the series frequency.

    The alias can be passed straight to ``Prophet.make_future_dataframe``.

    Examples
    --------
    >>> import pandas as pd
    >>> infer_frequency(pd.date_range("2024-01-31", periods=6, freq="ME"))
    'ME'
    >>> infer_frequency(pd.date_range("2024-01-01", periods=48, freq="h"))
    'h'
    """
    return infer_frequency_details(dates).freq


# --------------------------------------------------------------------------
# Horizon (pure)
# --------------------------------------------------------------------------

_DURATION = re.compile(r"^\s*(\d+)\s*([A-Za-z]+)\s*$")

# Case matters only for "M" (months) versus "m" (ambiguous, rejected).
_UNIT_ALIASES: Mapping[str, str] = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min", "t": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "d": "d", "day": "d", "days": "d",
    "w": "w", "wk": "w", "wks": "w", "week": "w", "weeks": "w",
    "M": "mo", "mo": "mo", "mos": "mo", "month": "mo", "months": "mo",
    "q": "q", "qtr": "q", "quarter": "q", "quarters": "q",
    "y": "y", "yr": "y", "yrs": "y", "year": "y", "years": "y",
}  # fmt: skip


def _duration_offset(n: int, unit: str) -> pd.DateOffset:
    match unit:
        case "s":
            return pd.DateOffset(seconds=n)
        case "min":
            return pd.DateOffset(minutes=n)
        case "h":
            return pd.DateOffset(hours=n)
        case "d":
            return pd.DateOffset(days=n)
        case "w":
            return pd.DateOffset(weeks=n)
        case "mo":
            return pd.DateOffset(months=n)
        case "q":
            return pd.DateOffset(months=3 * n)
        case "y":
            return pd.DateOffset(years=n)
        case _:
            raise HorizonError(f"Unsupported duration unit {unit!r}.")


def horizon_to_periods(horizon: int | str, last_date: pd.Timestamp, freq: str) -> int:
    """Convert a horizon into a number of periods of ``freq`` after ``last_date``.

    An ``int`` is taken as a number of periods.  A string such as ``"12W"``,
    ``"90D"``, ``"6 months"`` or ``"2y"`` is a duration: the result is the
    number of ``freq`` steps that fall in ``(last_date, last_date + duration]``
    (at least 1).  Use ``"M"`` or ``"mo"`` for months; lowercase ``"m"`` is
    rejected as ambiguous.

    Examples
    --------
    >>> import pandas as pd
    >>> horizon_to_periods(5, pd.Timestamp("2024-01-31"), "D")
    5
    >>> horizon_to_periods("2W", pd.Timestamp("2024-01-01"), "D")
    14
    >>> horizon_to_periods("1y", pd.Timestamp("2024-01-31"), "ME")
    12
    """
    if isinstance(horizon, bool):
        raise HorizonError("horizon must be an int or a duration string, not a bool.")
    if isinstance(horizon, (int, np.integer)):
        if horizon < 1:
            raise HorizonError(f"horizon must be at least 1 period, got {horizon}.")
        return int(horizon)
    if not isinstance(horizon, str):
        raise HorizonError(
            f"horizon must be an int or a duration string like '12W', got {type(horizon).__name__}."
        )
    match = _DURATION.match(horizon)
    if match is None:
        raise HorizonError(
            f"Cannot read horizon {horizon!r}. Use a number of periods (90) or a "
            "duration such as '12W', '90D', '6 months'."
        )
    amount, raw_unit = int(match.group(1)), match.group(2)
    unit = _UNIT_ALIASES.get(raw_unit) or _UNIT_ALIASES.get(raw_unit.lower())
    if raw_unit == "m" or unit is None:
        raise HorizonError(
            f"Unknown or ambiguous unit {raw_unit!r} in horizon {horizon!r}. "
            "Use s, min, h, D, W, M (months), Q, or Y."
        )
    if amount < 1:
        raise HorizonError(f"horizon must be positive, got {horizon!r}.")
    end = last_date + _duration_offset(amount, unit)
    dates = pd.date_range(start=last_date, end=end, freq=freq)
    return max(1, int((dates > last_date).sum()))


# --------------------------------------------------------------------------
# Skipped days and hours (pure)
# --------------------------------------------------------------------------

_MAX_GRID = 1_000_000
_DAY_NANOS = 86_400_000_000_000


def _parse_skip(skip_days: SkipSpec, skip_hours: SkipSpec) -> SkipAxis:
    """Validate ``skip_days`` / ``skip_hours`` into a :class:`SkipAxis` used here only for its masks."""
    try:
        return SkipAxis(parse_skip_days(skip_days), parse_skip_hours(skip_hours))
    except SkipSpecError as exc:
        raise SkipDaysError(str(exc)) from exc


def _is_sub_daily(freq: str) -> bool:
    """True for fixed steps shorter than a day (``h``, ``15min``); calendar offsets are never sub-daily."""
    offset = to_offset(freq)
    # Calendar offsets (and Day in recent pandas) are not Ticks and have no fixed length.
    return isinstance(offset, Tick) and offset.nanos < _DAY_NANOS


def _grid_after(last_date: pd.Timestamp, freq: str, n: int) -> pd.DatetimeIndex:
    """The first ``n`` points of ``freq`` strictly after ``last_date`` (as ``Prophet.make_future_dataframe``)."""
    dates = pd.date_range(start=last_date, periods=n + 1, freq=freq)
    return dates[dates > last_date][:n]


def _future_dates(
    last_date: pd.Timestamp, freq: str, horizon: int | str, skip: SkipAxis
) -> pd.DatetimeIndex:
    """Forecast timestamps after ``last_date``, with skipped days and hours left out.

    Without skipping this is exactly ``horizon_to_periods`` steps of ``freq``.
    With skipping, an ``int`` horizon is the number of rows to forecast
    (skipped periods are not counted), and a duration such as ``"2W"`` is a
    window of calendar time from which the skipped periods are removed.
    """
    wanted = horizon_to_periods(horizon, last_date, freq)
    if not skip.active:
        return _grid_after(last_date, freq, wanted)
    if isinstance(horizon, str):
        grid = _grid_after(last_date, freq, wanted)
        kept = grid[skip.keeps(grid)]
        if kept.empty:
            raise HorizonError(
                f"horizon={horizon!r} only covers days or hours that are skipped, so "
                "there is nothing to forecast. Use a longer horizon."
            )
        return kept
    n = wanted
    try:
        while n <= _MAX_GRID:
            grid = _grid_after(last_date, freq, n)
            kept = grid[skip.keeps(grid)]
            if len(kept) >= wanted:
                return kept[:wanted]
            n *= 4
    except (pd.errors.OutOfBoundsDatetime, OverflowError):
        pass  # the grid ran past the representable dates without finding enough kept rows
    raise SkipDaysError(
        f"skip_days/skip_hours remove every timestamp of freq {freq!r} after "
        f"{last_date}, so there is nothing to forecast. Check that the skipped days "
        "or hours are not the only ones the frequency produces."
    )


def _unexplained_gaps(ds: pd.Series, freq: str, skip: SkipAxis) -> int:
    """Grid timestamps that are neither present nor skipped: the gaps skipping does not account for."""
    grid = pd.date_range(ds.min(), ds.max(), freq=freq)
    expected = grid[skip.keeps(grid)]
    return int(len(expected.difference(pd.DatetimeIndex(ds))))


# --------------------------------------------------------------------------
# Regressors (pure)
# --------------------------------------------------------------------------

_REGRESSOR_OPTION_KEYS = ("mode", "prior_scale", "standardize")


@dataclass(frozen=True)
class _RegressorInput:
    """Regressors as requested: columns of the data, and full series supplied by name."""

    columns: tuple[str, ...] = ()
    external: Mapping[str, pd.Series] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.columns, *(self.external or {}))


def _coerce_regressor(name: str, raw: pd.Series) -> pd.Series:
    """Regressor values as float64: booleans become 0/1, strings are parsed, infinities are rejected."""
    if pd.api.types.is_bool_dtype(raw.dtype) or _is_numeric(raw):
        values = raw.astype("float64")
    else:
        try:
            values = pd.to_numeric(raw).astype("float64")
        except (ValueError, TypeError) as exc:
            raise RegressorError(
                f"Regressor {name!r} (dtype {raw.dtype}) is not numeric: {exc}. "
                "Regressors must be numbers (or booleans for 0/1 flags)."
            ) from exc
    if bool(np.isinf(values.to_numpy()).any()):
        raise RegressorError(
            f"Regressor {name!r} contains infinite values. Replace them or remove those rows."
        )
    return values


def _external_series(
    name: str, raw: object, tz: TimezoneMode, notes: list[str]
) -> pd.Series:
    """A user-supplied regressor series: datetime index made naive and sorted, numeric values."""
    if not isinstance(raw, pd.Series):
        raise RegressorError(
            f"regressors[{name!r}] must be a pandas Series indexed by datetime, "
            f"got {type(raw).__name__}."
        )
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise RegressorError(
            f"regressors[{name!r}] must have a DatetimeIndex, got "
            f"{type(raw.index).__name__}. Use series.index = pd.to_datetime(...)."
        )
    if raw.index.hasnans:
        raise RegressorError(f"regressors[{name!r}] has missing dates in its index.")
    index = pd.DatetimeIndex(_to_naive(pd.Series(raw.index, name="ds"), tz, notes))
    values = _coerce_regressor(name, raw.reset_index(drop=True))
    out = pd.Series(values.to_numpy(), index=index, name=name).sort_index()
    if out.index.has_duplicates:
        raise RegressorError(
            f"regressors[{name!r}] has duplicate timestamps, e.g. "
            f"{_describe_timestamps(out.index[out.index.duplicated()].unique())}."
        )
    return out


def _parse_regressors(
    regressors: object, tz: TimezoneMode
) -> tuple[_RegressorInput, tuple[str, ...]]:
    """Turn the ``regressors`` argument into column names and external series, plus notes to log."""
    notes: list[str] = []
    if regressors is None:
        return _RegressorInput(), ()
    if isinstance(regressors, str):
        regressors = [regressors]
    if isinstance(regressors, pd.DataFrame):
        if "ds" in regressors.columns:
            ds = pd.DatetimeIndex(_to_naive(_parse_dates(regressors["ds"]), tz, notes))
            table = regressors.drop(columns="ds")
        elif isinstance(regressors.index, pd.DatetimeIndex):
            ds, table = regressors.index, regressors
        else:
            raise RegressorError(
                "A regressors DataFrame needs a 'ds' column or a DatetimeIndex "
                f"alongside its value columns; got columns {list(regressors.columns)}."
            )
        regressors = {
            str(c): pd.Series(table[c].to_numpy(), index=ds) for c in table.columns
        }
    if isinstance(regressors, Mapping):
        names = [str(k) for k in regressors]
        series = {
            name: _external_series(name, regressors[key], tz, notes)
            for name, key in zip(names, regressors, strict=True)
        }
        parsed = _RegressorInput(external=series)
    elif isinstance(regressors, (Sequence, pd.Index)):
        names = list(regressors)
        bad = [n for n in names if not isinstance(n, str)]
        if bad:
            raise RegressorError(
                f"regressors must be column names (strings), got {bad[:3]}. To pass "
                "values instead, use a mapping {name: pandas.Series}."
            )
        parsed = _RegressorInput(columns=tuple(names))
    else:
        raise RegressorError(
            "regressors must be a list of column names, a mapping {name: pandas.Series} "
            f"or a DataFrame with a 'ds' column, got {type(regressors).__name__}."
        )
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated or not all(names):
        raise RegressorError(
            f"Regressor names must be unique and non-empty; problem with {duplicated or ['']}."
        )
    return parsed, tuple(notes)


def _check_regressor_names(names: Sequence[str]) -> None:
    """Reject names Prophet reserves (``yhat``, ``trend``, ``ds``...) before any data work is done."""
    probe = Prophet()
    for name in names:
        try:
            probe.validate_column_name(name)
        except ValueError as exc:
            raise RegressorError(
                f"Regressor name {name!r} cannot be used: {exc}"
            ) from exc


def _check_regressor_options(
    options: Mapping[str, Mapping[str, Any]] | None, names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Validate ``regressor_options`` against the regressor names and the settings ``add_regressor`` takes."""
    if not options:
        return {}
    unknown = sorted(set(options) - set(names))
    if unknown:
        raise RegressorError(
            f"regressor_options has entries for {unknown}, which are not regressors. "
            f"Regressors: {list(names)}."
        )
    checked: dict[str, dict[str, Any]] = {}
    for name, opts in options.items():
        if not isinstance(opts, Mapping):
            raise RegressorError(
                f"regressor_options[{name!r}] must be a mapping such as "
                f"{{'mode': 'multiplicative'}}, got {type(opts).__name__}."
            )
        bad = sorted(set(opts) - set(_REGRESSOR_OPTION_KEYS))
        if bad:
            raise RegressorError(
                f"regressor_options[{name!r}] has unknown setting(s) {bad}. "
                f"Allowed: {list(_REGRESSOR_OPTION_KEYS)}."
            )
        checked[name] = dict(opts)
    return checked


def _parse_future(
    future: object, tz: TimezoneMode, names: Sequence[str], notes: list[str]
) -> pd.DataFrame:
    """Future regressor values as a frame indexed by naive ``ds``, one float column per regressor present."""
    if not isinstance(future, pd.DataFrame):
        raise RegressorError(
            f"future must be a DataFrame with a 'ds' column and the regressor columns, "
            f"got {type(future).__name__}."
        )
    if "ds" in future.columns:
        ds = pd.DatetimeIndex(_to_naive(_parse_dates(future["ds"]), tz, notes))
    elif isinstance(future.index, pd.DatetimeIndex):
        ds = future.index
    else:
        raise RegressorError(
            f"future needs a 'ds' column or a DatetimeIndex, got columns {list(future.columns)}."
        )
    columns = {
        name: _coerce_regressor(name, future[name].reset_index(drop=True))
        for name in names
        if name in future.columns
    }
    out = pd.DataFrame({k: v.to_numpy() for k, v in columns.items()}, index=ds)
    if out.index.has_duplicates:
        raise RegressorError(
            "future has duplicate timestamps, e.g. "
            f"{_describe_timestamps(out.index[out.index.duplicated()].unique())}."
        )
    return out


def _describe_timestamps(ts: pd.Index, limit: int = 5) -> str:
    """'2024-03-01, 2024-03-02 and 3 more': dates only when no timestamp has a time of day."""
    ts = pd.DatetimeIndex(ts)
    midnight = bool((ts == ts.normalize()).all())
    shown = [f"{t:%Y-%m-%d}" if midnight else f"{t:%Y-%m-%d %H:%M}" for t in ts[:limit]]
    rest = len(ts) - limit
    return ", ".join(shown) + (f" and {rest} more" if rest > 0 else "")


def _require_values(name: str, values: pd.Series, where: str, hint: str) -> None:
    missing = values.index[values.isna().to_numpy()]
    if len(missing):
        raise MissingRegressorValuesError(
            f"Regressor {name!r} has no value for {len(missing)} of {len(values)} "
            f"{where} timestamps: {_describe_timestamps(pd.DatetimeIndex(missing))}. {hint}"
        )


def _assemble_regressors(
    names: Sequence[str],
    history_ds: pd.DatetimeIndex,
    future_ds: pd.DatetimeIndex,
    from_data: pd.DataFrame,
    external: Mapping[str, pd.Series],
    future: pd.DataFrame | None,
) -> pd.DataFrame:
    """Regressor values for every training and forecast timestamp, or an error naming what is missing.

    ``from_data`` holds the regressors that came from the input data (indexed
    by training timestamp). A regressor's values come from its external series
    or its data column, and ``future`` fills timestamps they do not cover.
    """
    all_ds = history_ds.append(future_ds)
    columns: dict[str, np.ndarray] = {}
    for name in names:
        known = external[name] if name in external else from_data[name]
        if future is not None and name in future.columns:
            known = known.combine_first(future[name])
        in_history = known.reindex(history_ds)
        in_future = known.reindex(future_ds)
        _require_values(
            name,
            in_history,
            "training",
            "Regressors need a value on every row used for fitting; fill or remove those rows.",
        )
        _require_values(
            name,
            in_future,
            "forecast",
            f"Supply them with future=<DataFrame with a 'ds' column and a {name!r} column> "
            "or include them in the regressors mapping.",
        )
        columns[name] = np.concatenate([in_history.to_numpy(), in_future.to_numpy()])
    return pd.DataFrame(columns, index=all_ds)


# --------------------------------------------------------------------------
# Weather (pure planning, lazy fetch)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _WeatherPlan:
    """What to fetch and how each series timestamp maps onto the fetched values."""

    kind: Literal["daily", "hourly"]
    targets: pd.DatetimeIndex
    start: pd.Timestamp
    end: pd.Timestamp
    low: pd.DatetimeIndex
    high: pd.DatetimeIndex


def _period_ends_at_label(freq: str) -> bool:
    """Whether a coarse period is labelled by its last day (``W``, ``ME``, ``QE``) rather than its first."""
    prefix = to_offset(freq).rule_code.split("-")[0]
    return prefix.startswith(
        ("W", "M", "SM", "BM", "Q", "A", "Y", "BQ", "BA", "BY")
    ) and not prefix.endswith("S")


def _plan_weather(ds: pd.DatetimeIndex, freq: str) -> _WeatherPlan:
    """Choose hourly weather for sub-daily series and daily weather otherwise.

    For series coarser than a day, each timestamp averages the daily
    temperatures of its period: the period ending on the label for weekly
    ``W-*`` and month, quarter and year ends, the period starting on the label
    otherwise (for example ``MS``).
    """
    if _is_sub_daily(freq):
        return _WeatherPlan(
            "hourly", ds, ds.min().floor("h"), ds.max().ceil("h"), ds, ds
        )
    day = pd.Timedelta(days=1)
    probe = pd.date_range(ds.min(), periods=3, freq=freq)
    one_day = bool((probe[1:] - probe[:-1]).min() <= day)
    first = ds.normalize()
    if one_day:
        low, high = first, first + day
    else:
        offset = to_offset(freq)
        with warnings.catch_warnings():
            # Non-vectorised offsets (business months) are slow but correct.
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            if _period_ends_at_label(freq):
                low, high = (first - offset) + day, first + day
            else:
                low, high = first, first + offset
    return _WeatherPlan("daily", ds, low.min(), high.max() - day, low, high)


def _temperature_for(plan: _WeatherPlan, frame: pd.DataFrame) -> pd.Series:
    """Temperature at each target timestamp: interpolated hourly values, or period means of daily ones."""
    ordered = frame.sort_values("ds")
    index = pd.DatetimeIndex(ordered["ds"])
    values = ordered["temperature"].to_numpy(dtype=float)
    if plan.kind == "hourly":
        hourly = pd.Series(values, index=index)
        both = hourly.index.union(plan.targets)
        filled = hourly.reindex(both).interpolate(method="time", limit_area="inside")
        return filled.reindex(plan.targets)
    finite = np.isfinite(values)
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0))])
    count = np.concatenate([[0], np.cumsum(finite)])
    lo = index.searchsorted(plan.low, side="left")
    hi = index.searchsorted(plan.high, side="left")
    n = count[hi] - count[lo]
    mean = np.where(n > 0, (total[hi] - total[lo]) / np.maximum(n, 1), np.nan)
    return pd.Series(mean, index=plan.targets)


def weather_source_labels(
    ds: pd.DatetimeIndex, freq: str, frame: pd.DataFrame
) -> pd.Series:
    """The weather source ("archive", "forecast" or "climatology") behind each timestamp's temperature.

    ``frame`` is a :attr:`prophet.weather.WeatherResult.frame` with ``ds`` and
    ``source`` columns. The label of a timestamp follows how its temperature
    was built in :func:`forecast`:

    * Sub-daily series use hourly weather: the label of the same hour, or of
      the nearest earlier hour when the timestamp falls between two.
    * Daily series, and series with several points a day, use the label of
      the day.
    * Coarser series (weekly, monthly, ...) average the days of their period.
      The label is that of the latest day in the period, so a period counts
      as climatology as soon as its last day is, and as measured weather only
      when all of it is.

    A timestamp with no matching weather row gets NaN. The result is indexed
    by position (0 to ``len(ds) - 1``).
    """
    source = frame.drop_duplicates("ds").set_index("ds")["source"].sort_index()
    plan = _plan_weather(ds, freq)
    if plan.kind == "hourly":
        labels = source.reindex(ds, method="ffill")
    else:
        # The period over which a timestamp's temperature is averaged ends the day before ``high``.
        labels = source.reindex(plan.high - pd.Timedelta(days=1))
    return pd.Series(labels.to_numpy(dtype=object), name="weather_source")


def _check_location(weather: object) -> None:
    """A place name or a (latitude, longitude) pair; the weather module checks the ranges."""
    if isinstance(weather, str):
        if weather.strip():
            return
    elif (
        isinstance(weather, Sequence)
        and len(weather) == 2
        and all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
            for v in weather
        )
    ):
        return
    raise RegressorError(
        "weather must be a place name such as 'London' or a (latitude, longitude) "
        f"pair such as (51.5, -0.12), got {weather!r}."
    )


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

_FORECAST_COLUMNS = ("ds", "yhat", "yhat_lower", "yhat_upper")
_DAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class ForecastSummary:
    """Short description of a forecast run (see :meth:`ForecastResult.summary`)."""

    n_rows_used: int
    history_start: pd.Timestamp
    history_end: pd.Timestamp
    freq: str
    freq_method: FrequencyMethod
    horizon_periods: int
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    skip_days: tuple[str, ...] = ()
    skip_hours: tuple[int, ...] = ()
    rows_skipped: int = 0
    regressors: tuple[str, ...] = ()
    weather_location: str | None = None

    def __str__(self) -> str:
        text = (
            f"{self.n_rows_used} rows ({self.history_start:%Y-%m-%d} to "
            f"{self.history_end:%Y-%m-%d}) at freq {self.freq!r}; forecast "
            f"{self.horizon_periods} periods ({self.forecast_start:%Y-%m-%d} to "
            f"{self.forecast_end:%Y-%m-%d})"
        )
        if self.skip_days or self.skip_hours:
            skipped = [
                *self.skip_days,
                *(
                    f"{a:02d}:00 to {(b + 1) % 24:02d}:00"
                    for a, b in _hour_runs(self.skip_hours)
                ),
            ]
            text += (
                f"; skipping {', '.join(skipped)} ({self.rows_skipped} rows dropped)"
            )
        if self.regressors:
            text += f"; regressors: {', '.join(self.regressors)}"
            if self.weather_location:
                text += f" (temperature for {self.weather_location})"
        return text


@dataclass(frozen=True, eq=False)
class ForecastResult:
    """Immutable bundle returned by :func:`forecast`.

    Attributes
    ----------
    model: The fitted :class:`prophet.Prophet`.
    history: The cleaned ``ds``/``y`` training data.
    predictions: Full ``Prophet.predict`` output for history and future
        (all component columns), as needed by the plotting functions.
    freq: Pandas offset alias used to extend the series.
    freq_method: How ``freq`` was obtained (``"infer_freq"``,
        ``"modal_difference"`` or ``"user"``).
    horizon_periods: Number of future periods predicted.
    include_history: Whether :attr:`forecast` also contains in-sample rows.
    tuning: The :class:`prophet.tuning.TuningResult` when ``forecast`` was
        called with a truthy ``tune`` (``True``, ``"grid"`` or ``"optuna"``),
        otherwise ``None``.
    skip_days: Weekdays left out (Monday is 0), sorted; empty when none.
    skip_hours: Hours of the day left out, sorted; empty when none.
    rows_skipped: Input rows dropped because they fell on a skipped day or hour.
    regressors: Names of the extra regressors in the model, including
        ``"temperature"`` when ``weather`` was used.
    regressor_values: ``ds`` plus one column per regressor, for every training
        and forecast row (``None`` without regressors).
    weather: The :class:`prophet.weather.WeatherResult` behind the
        ``temperature`` regressor. Its ``frame`` has a ``source`` column
        (``"archive"``, ``"forecast"`` or ``"climatology"``) saying where each
        value came from, so charts can tell measured from predicted weather.
    """

    model: Prophet
    history: pd.DataFrame
    predictions: pd.DataFrame
    freq: str
    freq_method: FrequencyMethod | Literal["user"]
    horizon_periods: int
    include_history: bool = False
    tuning: TuningResult | None = None
    skip_days: tuple[int, ...] = ()
    skip_hours: tuple[int, ...] = ()
    rows_skipped: int = 0
    regressors: tuple[str, ...] = ()
    regressor_values: pd.DataFrame | None = None
    weather: WeatherResult | None = None

    @property
    def future(self) -> pd.DataFrame:
        """Only the future rows: ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``."""
        cutoff = self.history["ds"].max()
        return self._slim(self.predictions[self.predictions["ds"] > cutoff])

    @property
    def forecast(self) -> pd.DataFrame:
        """``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper`` (with history if requested)."""
        return self.to_frame(include_history=self.include_history)

    def to_frame(self, include_history: bool | None = None) -> pd.DataFrame:
        """Forecast as a compact DataFrame.

        Parameters
        ----------
        include_history: Override whether in-sample rows are included. Defaults
            to the value chosen when calling :func:`forecast`.
        """
        with_history = (
            self.include_history if include_history is None else include_history
        )
        src = self.predictions if with_history else self.future
        return self._slim(src)

    def plot(self, **kwargs: Any) -> plt.Figure:
        """Plot history and forecast via :meth:`Prophet.plot`; kwargs are passed on.

        Skipped days and hours are closed up on the time axis automatically;
        pass ``skip_days=()`` to draw the continuous axis instead.  When
        ``weather`` was used and climatology fills part of the horizon, the
        credit line says how much of it is weather forecast and how much is
        climatology.
        """
        style: dict[str, Any] = {
            **self._skip_kwargs(),
            **self._source_note_kwargs(),
            **kwargs,
        }
        return self.model.plot(self.predictions, **style)

    def plot_components(self, **kwargs: Any) -> plt.Figure:
        """Plot trend and seasonality via :meth:`Prophet.plot_components`.

        The time axes close up skipped days and hours, and the weekly profile
        shows only the days that remain, unless overridden by kwargs.
        """
        style: dict[str, Any] = {**self._skip_kwargs(), **kwargs}
        return self.model.plot_components(self.predictions, **style)

    @property
    def training_frame(self) -> pd.DataFrame:
        """``ds``, ``y`` and one column per regressor for every training row.

        This is the data the model was fitted on, in the form
        :func:`prophet.visuals.run_backtest` and the ``df`` argument of the
        report functions expect.  Without regressors it is a copy of
        :attr:`history`.
        """
        if self.regressor_values is None:
            return self.history.copy()
        columns = ["ds", *self.regressors]
        return self.history.merge(
            self.regressor_values[columns], on="ds", how="left"
        ).reset_index(drop=True)

    @property
    def weather_sources(self) -> pd.Series | None:
        """Where each row's temperature came from, aligned row for row to ``predictions``.

        Labels are ``"archive"`` (measured), ``"forecast"`` (weather forecast)
        or ``"climatology"`` (typical values for the date).  ``None`` when
        ``weather`` was not used.  Sub-daily rows take the label of their
        hour (the nearest earlier hour between two), daily rows that of their
        day, and weekly or monthly rows that of the latest day in the period
        they average.  See :func:`weather_source_labels`.
        """
        if self.weather is None:
            return None
        labels = weather_source_labels(
            pd.DatetimeIndex(self.predictions["ds"]), self.freq, self.weather.frame
        )
        return labels.set_axis(self.predictions.index)

    def backtest(self, horizon: int | str | None = None) -> Backtest:
        """Hold out the end of the history, refit a copy of the model on the rest and score its forecast.

        Parameters
        ----------
        horizon: How much to hold out: a number of rows or a duration such as
            ``"60D"``.  Defaults to the forecast's own horizon
            (:attr:`horizon_periods` rows), capped at half the history so that
            the refit always has data to learn from.  Skipped days and hours
            are never part of the holdout.

        Returns
        -------
        A :class:`prophet.visuals.Backtest` with the split, the refitted
        model's forecast and the accuracy metrics.
        """
        from prophet.visuals import run_backtest

        return run_backtest(
            self.model,
            self.training_frame,
            self._backtest_horizon(horizon),
            **self._skip_kwargs(),
        )

    def plot_backtest(
        self, horizon: int | str | None = None, **kwargs: Any
    ) -> plt.Figure:
        """Draw the backtest (see :meth:`backtest`); kwargs go to :func:`prophet.visuals.plot_backtest`.

        Pass ``backtest=`` with an earlier :meth:`backtest` result to draw it
        without refitting.
        """
        from prophet.visuals import plot_backtest

        chosen = self._backtest_horizon(horizon)
        run = kwargs.pop("backtest", None) or self.backtest(chosen)
        style: dict[str, Any] = {**self._skip_kwargs(), **kwargs}
        return plot_backtest(
            self.model, self.training_frame, chosen, backtest=run, **style
        )

    def plot_regressor_effects(self, **kwargs: Any) -> plt.Figure:
        """What each extra regressor added to the forecast, via :func:`prophet.visuals.plot_regressor_effects`.

        With ``weather`` the temperature panel is labelled by where each value
        came from (measured, weather forecast or climatology).  Raises
        :class:`prophet.visuals.NoRegressorsError` without regressors.
        """
        from prophet.visuals import plot_regressor_effects

        style: dict[str, Any] = {
            **self._skip_kwargs(),
            **self._weather_kwargs(),
            **kwargs,
        }
        return plot_regressor_effects(self.model, self.predictions, **style)

    def report(
        self,
        *,
        backtest: bool | Backtest = True,
        horizon: int | str | None = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """One A4 page with the forecast, key numbers, components and regressor effects.

        Parameters
        ----------
        backtest: ``True`` (default) runs :meth:`backtest` once and adds its
            accuracy to the page; pass a :class:`prophet.visuals.Backtest` you
            already have to reuse it, or ``False`` to skip it, which avoids
            the refit.
        horizon: Holdout for the backtest, as for :meth:`backtest`.
        **kwargs: Passed to :func:`prophet.visuals.plot_report`.
        """
        from prophet.visuals import plot_report

        run = self._resolve_backtest(backtest, horizon)
        style: dict[str, Any] = {
            "df": self.history,
            "metrics": None if run is None else run.metrics,
            **self._skip_kwargs(),
            **self._weather_kwargs(),
            **kwargs,
        }
        return plot_report(self.model, self.predictions, **style)

    def dashboard(
        self,
        path: str | Path,
        *,
        backtest: bool | Backtest = True,
        horizon: int | str | None = None,
        **kwargs: Any,
    ) -> Path:
        """Write the interactive HTML dashboard and return its path.

        Needs plotly (``pip install plotly``).  ``backtest`` and ``horizon``
        work as in :meth:`report`; the backtest feeds both the accuracy numbers
        at the top of the page and the accuracy section's chart.  Other kwargs
        go to :func:`prophet.visuals.dashboard_html`, for example ``sections``
        to choose and order the sections, ``theme`` (``"light"``, ``"dark"`` or
        ``"auto"``) and ``cv`` with the output of
        :func:`prophet.diagnostics.cross_validation`.
        """
        from prophet.visuals import dashboard_html

        run = self._resolve_backtest(backtest, horizon)
        style: dict[str, Any] = {
            "df": self.history,
            "backtest": run,
            **self._skip_kwargs(),
            **self._weather_kwargs(),
            **kwargs,
        }
        return dashboard_html(self.model, self.predictions, path, **style)

    def _backtest_horizon(self, horizon: int | str | None) -> int | str:
        """The caller's horizon, or the forecast's own, kept to half the history."""
        if horizon is not None:
            return horizon
        capped = min(self.horizon_periods, len(self.history) // 2)
        if capped < self.horizon_periods:
            logger.info(
                "Backtest horizon cut from %d to %d rows so that half the history remains for training.",
                self.horizon_periods,
                capped,
            )
        return max(capped, 1)

    def _resolve_backtest(
        self, backtest: bool | Backtest, horizon: int | str | None
    ) -> Backtest | None:
        if backtest is False:
            return None
        if backtest is True:
            return self.backtest(horizon)
        return backtest

    def _weather_kwargs(self) -> dict[str, pd.Series]:
        sources = self.weather_sources
        return {} if sources is None else {"weather_sources": sources}

    def _source_note_kwargs(self) -> dict[str, str]:
        """A credit-line sentence on where future temperatures came from, only when climatology is part of it."""
        sources = self.weather_sources
        if sources is None:
            return {}
        from prophet.visuals import weather_ahead_note, weather_source_runs

        ds = self.predictions["ds"]
        step = (
            pd.Timedelta(ds.diff().dropna().median())
            if len(ds) > 1
            else pd.Timedelta(days=1)
        )
        # Runs are built from the future rows alone: the weather forecast often starts a day or two
        # before the last observation, and that stretch belongs to the future too.
        end = self.history["ds"].max()
        ahead = (ds > end).to_numpy()
        note = weather_ahead_note(
            weather_source_runs(ds[ahead], sources[ahead]),
            end,
            step,
            calendar=bool(self.skip_days or self.skip_hours),
        )
        return {"source_note": note} if note else {}

    def _skip_kwargs(self) -> dict[str, tuple[int, ...]]:
        return {
            key: value
            for key, value in (
                ("skip_days", self.skip_days),
                ("skip_hours", self.skip_hours),
            )
            if value
        }

    def summary(self) -> ForecastSummary:
        """Rows used, frequency and horizon of this run."""
        future = self.future
        return ForecastSummary(
            n_rows_used=len(self.history),
            history_start=self.history["ds"].min(),
            history_end=self.history["ds"].max(),
            freq=self.freq,
            freq_method=cast("FrequencyMethod", self.freq_method),
            horizon_periods=self.horizon_periods,
            forecast_start=future["ds"].min(),
            forecast_end=future["ds"].max(),
            skip_days=tuple(_DAY_LABELS[d] for d in self.skip_days),
            skip_hours=self.skip_hours,
            rows_skipped=self.rows_skipped,
            regressors=self.regressors,
            weather_location=None
            if self.weather is None
            else self.weather.location.name,
        )

    @staticmethod
    def _slim(frame: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in _FORECAST_COLUMNS if c in frame.columns]
        return frame[cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Entry point (imperative shell)
# --------------------------------------------------------------------------


def _resolve_prophet_params(
    history: pd.DataFrame,
    user_params: Mapping[str, Any],
    *,
    tune: bool | Literal["grid", "optuna"] = False,
    country_holidays: str | None = None,
    regressors: Sequence[str] = (),
) -> tuple[dict[str, Any], TuningResult | None]:
    """Choose the keyword arguments for ``Prophet(...)``.

    With ``tune=True`` (or ``"grid"``) the default grid is searched by
    cross-validation, and with ``tune="optuna"`` the default Optuna search
    space, minus any parameter the user set explicitly, so explicit arguments
    always win. ``history`` must then hold a column for each of ``regressors``.
    """
    if not tune:
        return dict(user_params), None
    if tune not in (True, "grid", "optuna"):
        raise ValueError(f"tune must be True, False, 'grid' or 'optuna', got {tune!r}.")
    search: Literal["grid", "optuna"] = "optuna" if tune == "optuna" else "grid"

    # Imported lazily: tuning pulls in process pools and cross-validation,
    # which plain forecasts never need.
    from prophet.make_holidays import make_holidays_df
    from prophet.tuning import default_param_grid, default_search_space
    from prophet.tuning import tune as run_tune

    defaults = default_search_space() if search == "optuna" else default_param_grid()
    grid = {k: v for k, v in defaults.items() if k not in user_params}
    if not grid:
        logger.warning(
            "tune=%r ignored: every tuned parameter was set explicitly.", tune
        )
        return dict(user_params), None

    # The final model adds country holidays after construction, so the tuner
    # gets them as a holidays frame to score the same model it will select.
    tuning_kwargs = dict(user_params)
    if country_holidays is not None and "holidays" not in tuning_kwargs:
        years = list(range(history["ds"].min().year, history["ds"].max().year + 1))
        tuning_kwargs["holidays"] = make_holidays_df(
            year_list=years, country=country_holidays
        )
    tune_extra: dict[str, Any] = {"regressors": tuple(regressors)} if regressors else {}
    result = run_tune(
        history,
        param_grid=grid,
        prophet_kwargs=tuning_kwargs,
        search=search,
        **tune_extra,
    )
    logger.info("tune=%r selected %s", tune, result.best_params)
    return {**result.best_params, **user_params}, result


def _check_freq(freq: str) -> str:
    try:
        to_offset(freq)
    except (ValueError, TypeError) as exc:
        raise FrequencyInferenceError(
            f"freq={freq!r} is not a valid pandas offset alias: {exc}. "
            "Examples: 'D', 'W', 'MS', 'ME', 'h'."
        ) from exc
    return freq


def _fetch_weather(
    location: str | tuple[float, float], plan: _WeatherPlan
) -> WeatherResult:
    """Fetch temperature for the plan. The module is imported here so ``import prophet`` never needs it."""
    try:
        from prophet.weather import fetch_temperature
    except ImportError as exc:
        raise ImportError(
            f"weather=... needs the prophet.weather module, which could not be imported: {exc}"
        ) from exc
    return fetch_temperature(location, plan.start, plan.end, freq=plan.kind)


def _add_regressors(
    model: Prophet, names: Sequence[str], options: Mapping[str, Mapping[str, Any]]
) -> None:
    for name in names:
        try:
            model.add_regressor(name, **options.get(name, {}))
        except ValueError as exc:
            raise RegressorError(f"Cannot add regressor {name!r}: {exc}") from exc


def forecast(
    data: pd.DataFrame | pd.Series,
    horizon: int | str,
    *,
    date_col: str | None = None,
    value_col: str | None = None,
    freq: str | None = None,
    include_history: bool = False,
    country_holidays: str | None = None,
    interval_width: float | None = None,
    tz: TimezoneMode = "wall",
    tune: bool | Literal["grid", "optuna"] = False,
    regressors: Sequence[str] | Mapping[str, pd.Series] | pd.DataFrame | None = None,
    regressor_options: Mapping[str, Mapping[str, Any]] | None = None,
    future: pd.DataFrame | None = None,
    weather: str | tuple[float, float] | None = None,
    skip_days: SkipSpec = None,
    skip_hours: SkipSpec = None,
    **prophet_kwargs: Any,
) -> ForecastResult:
    """Fit a Prophet model and forecast ahead in one call.

    Parameters
    ----------
    data: A DataFrame (``ds``/``y``, any one date plus one numeric column,
        or a DatetimeIndex plus one numeric column) or a Series with a
        DatetimeIndex.  See :func:`normalise_input`.
    horizon: Periods to predict (``90``) or a duration string (``"12W"``,
        ``"6 months"``) converted to periods at the series frequency.  With
        ``skip_days`` / ``skip_hours`` an integer is the number of rows to
        forecast (skipped periods are not counted), and a duration is a window
        of calendar time from which the skipped periods are removed.
    date_col, value_col: Column names, when they cannot be inferred.
    freq: Pandas offset alias overriding frequency inference.
    include_history: Include in-sample rows in ``result.forecast``.
    country_holidays: Country code for built-in holidays, for example ``"GB"``.
    interval_width: Width of the uncertainty interval (Prophet default 0.8).
    tz: ``"wall"`` (default) or ``"utc"``; how timezone-aware dates are
        made naive.
    tune: Choose ``changepoint_prior_scale``, ``seasonality_prior_scale`` and
        ``seasonality_mode`` by cross-validation with
        :func:`prophet.tuning.tune` before fitting (parameters you pass
        explicitly are kept). Typically 5 to 20 seconds for a few thousand
        daily rows. Cross-validation only sees the past, so tuning can hurt
        when the latest data behaves unlike earlier data; check
        ``result.tuning.scores`` and a holdout. In scripts, call it under
        ``if __name__ == "__main__":`` because tuning uses a process pool.
        ``tune=True`` (or ``"grid"``) scores the default grid;
        ``tune="optuna"`` samples continuous prior scale ranges with Optuna
        instead (40 trials, needs ``pip install 'neoprophet[optuna]'``).
        Tuning uses each regressor's default settings, so ``regressor_options``
        only apply to the final model.
    regressors: Extra explanatory variables.  Either a list of column names
        that are already in ``data`` (their future values then come from
        ``future``), or a mapping ``{name: pandas.Series}`` / a DataFrame with a
        ``ds`` column whose series cover both the history and the forecast
        timestamps.  Values must be numeric (booleans count as 0/1) with no gaps.
    regressor_options: Per-regressor settings passed to
        :meth:`Prophet.add_regressor`, for example
        ``{"promo": {"mode": "multiplicative", "prior_scale": 5.0,
        "standardize": False}}``.
    future: A DataFrame with a ``ds`` column and one column per regressor,
        holding values for the forecast timestamps (and any the other sources
        do not cover).
    weather: A place name (``"London"``) or ``(latitude, longitude)``.  The
        daily temperature there (hourly for sub-daily data, averaged over each
        period for weekly or monthly data) is fetched with
        :func:`prophet.weather.fetch_temperature` for the history and the
        horizon and added as a regressor named ``temperature``.  Future values
        beyond the weather forecast are climatology; ``result.weather`` keeps
        which is which.  Combine freely with ``regressors``.
    skip_days: Weekdays the series never has, excluded from training, the
        future frame and ``result.forecast``: names (``"Sat"``, ``"Saturday"``,
        ``"sat"``), numbers (Monday is 0) or ``"weekends"``, singly or as a
        list.  Prophet then fits weekly seasonality on the remaining days.  The
        number of dropped rows is logged, and charts close the gap.
    skip_hours: The same for hours of the day on sub-daily data, for example
        ``range(0, 8)`` to leave out nights.
    **prophet_kwargs: Passed to :class:`prophet.Prophet`, e.g.
        ``seasonality_mode="multiplicative"``.  Logistic growth is not
        supported here because it needs a ``cap`` column; use ``Prophet``
        directly for that.

    Returns
    -------
    A frozen :class:`ForecastResult`.

    Raises
    ------
    ForecastInputError: (a ``ValueError``) for bad data, frequency, horizon,
        skip settings or regressors.  :class:`RegressorError` and its subclass
        :class:`MissingRegressorValuesError` cover regressors, which say which
        timestamps lack values; :class:`SkipDaysError` covers skip settings.

    Examples
    --------
    >>> import pandas as pd
    >>> from prophet import forecast
    >>> idx = pd.date_range("2023-01-01", periods=120, freq="D")
    >>> series = pd.Series(range(120), index=idx, dtype=float)
    >>> result = forecast(series, horizon="2W", country_holidays="GB")  # doctest: +SKIP
    >>> result.summary().horizon_periods  # doctest: +SKIP
    14
    >>> result.forecast.head()  # doctest: +SKIP
    >>> fig = result.plot()  # doctest: +SKIP
    >>> shop = forecast(series, "2W", skip_days="weekends", weather="London")  # doctest: +SKIP
    """
    if "growth" in prophet_kwargs and prophet_kwargs["growth"] == "logistic":
        raise ForecastInputError(
            "growth='logistic' needs a 'cap' column, which forecast() does not take. "
            "Use prophet.Prophet directly."
        )
    if interval_width is not None:
        if (
            "interval_width" in prophet_kwargs
        ):  # pragma: no cover - same name cannot repeat
            raise ForecastInputError("interval_width given twice.")
        prophet_kwargs = {**prophet_kwargs, "interval_width": interval_width}

    skip = _parse_skip(skip_days, skip_hours)
    wanted, regressor_notes = _parse_regressors(regressors, tz)
    if weather is not None:
        _check_location(weather)
        if WEATHER_REGRESSOR in wanted.names:
            raise RegressorError(
                f"weather= adds a regressor named {WEATHER_REGRESSOR!r}, which clashes "
                "with one you passed. Rename yours."
            )
    names = (*wanted.names, *((WEATHER_REGRESSOR,) if weather is not None else ()))
    _check_regressor_names(names)
    options = _check_regressor_options(regressor_options, names)
    if future is not None and not names:
        logger.warning("future= ignored: there are no regressors to supply values for.")
    for note in regressor_notes:
        logger.warning(note)

    normalised = _normalise(data, date_col, value_col, tz, wanted.columns)
    for note in normalised.notes:
        logger.warning(note)
    history = normalised.frame

    if freq is None:
        details = infer_frequency_details(history["ds"])
        if not details.regular:
            if (
                skip.active
                and _unexplained_gaps(history["ds"], details.freq, skip) == 0
            ):
                logger.debug(
                    "Series gaps are all skipped days or hours; freq=%r.", details.freq
                )
            else:
                logger.warning(
                    "Series is irregular (gaps or uneven spacing); using the most common "
                    "spacing, freq=%r. Pass freq=... to override.",
                    details.freq,
                )
        used_freq, method = details.freq, details.method
    else:
        used_freq, method = _check_freq(freq), "user"

    rows_skipped = 0
    if skip.active:
        if skip.skip_hours and not _is_sub_daily(used_freq):
            raise SkipDaysError(
                f"skip_hours needs sub-daily data, but the series frequency is {used_freq!r}. "
                "Use skip_days for daily or coarser data."
            )
        kept = skip.keeps(history["ds"])
        rows_skipped = int((~kept).sum())
        logger.log(
            logging.WARNING if rows_skipped else logging.DEBUG,
            "Dropped %d of %d row(s) that fall on skipped days or hours (skip_days=%s, skip_hours=%s).",
            rows_skipped,
            len(history),
            sorted(skip.skip_days),
            sorted(skip.skip_hours),
        )
        history = history[kept].reset_index(drop=True)
        if len(history) < 2:
            raise InsufficientDataError(
                f"Only {len(history)} row(s) remain after skipping days or hours; at least 2 are required."
            )

    future_ds = _future_dates(history["ds"].max(), used_freq, horizon, skip)
    history_ds = pd.DatetimeIndex(history["ds"])
    all_ds = pd.DatetimeIndex(history_ds.append(future_ds))

    external = dict(wanted.external or {})
    weather_result: WeatherResult | None = None
    if weather is not None:
        plan = _plan_weather(all_ds, used_freq)
        weather_result = _fetch_weather(weather, plan)
        external[WEATHER_REGRESSOR] = _temperature_for(
            plan, weather_result.frame
        ).rename(WEATHER_REGRESSOR)

    values: pd.DataFrame | None = None
    fit_frame = history[["ds", "y"]]
    if names:
        future_notes: list[str] = []
        supplied = (
            _parse_future(future, tz, names, future_notes)
            if future is not None
            else None
        )
        for note in future_notes:
            logger.warning(note)
        from_data = history.set_index("ds")[list(wanted.columns)]
        values = _assemble_regressors(
            names, history_ds, future_ds, from_data, external, supplied
        )
        fit_frame = fit_frame.assign(
            **{name: values[name].to_numpy()[: len(history)] for name in names}
        )

    params, tuning = _resolve_prophet_params(
        fit_frame,
        prophet_kwargs,
        tune=tune,
        country_holidays=country_holidays,
        regressors=names,
    )
    model = Prophet(**params)
    if country_holidays is not None:
        model.add_country_holidays(country_name=country_holidays)
    _add_regressors(model, names, options)
    model.fit(fit_frame)

    future_frame = pd.DataFrame({"ds": all_ds})
    if values is not None:
        future_frame = future_frame.assign(**{n: values[n].to_numpy() for n in names})
    predictions = model.predict(future_frame)
    return ForecastResult(
        model=model,
        history=history[["ds", "y"]],
        predictions=predictions,
        freq=used_freq,
        freq_method=method,
        horizon_periods=len(future_ds),
        include_history=include_history,
        tuning=tuning,
        skip_days=tuple(sorted(skip.skip_days)),
        skip_hours=tuple(sorted(skip.skip_hours)),
        rows_skipped=rows_skipped,
        regressors=names,
        regressor_values=None
        if values is None
        else values.rename_axis("ds").reset_index(),
        weather=weather_result,
    )
