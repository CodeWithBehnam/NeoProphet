"""High-level, one-call forecasting API on top of :class:`prophet.Prophet`.

The functions here are additive: they never change the behaviour of the
existing ``Prophet`` class.  The module follows a functional-core,
imperative-shell layout:

* Pure core: :func:`normalise_input`, :func:`infer_frequency` and
  :func:`horizon_to_periods` validate and transform data without touching
  the model.
* Imperative shell: :func:`forecast` builds, fits and runs the model and
  wraps the output in an immutable :class:`ForecastResult`.

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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from prophet.forecaster import Prophet

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from prophet.tuning import TuningResult

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
    "NonNumericValuesError",
    "UnparseableDatesError",
    "forecast",
    "horizon_to_periods",
    "infer_frequency",
    "infer_frequency_details",
    "normalise_input",
]

TimezoneMode = Literal["wall", "utc"]
FrequencyMethod = Literal["infer_freq", "modal_difference"]


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
    df: pd.DataFrame, date_col: str | None, value_col: str | None
) -> tuple[str | None, str]:
    """Return ``(date column or None for the index, value column)``."""
    cols = [str(c) for c in df.columns]
    if len(set(cols)) != len(cols):
        raise ColumnSelectionError(f"Column names must be unique, got {cols}.")
    has_dt_index = isinstance(df.index, pd.DatetimeIndex)

    if date_col is not None:
        chosen_date: str | None = _require_column(df, date_col, "date_col")
    elif "ds" in df.columns:
        chosen_date = "ds"
    else:
        dt_cols = [c for c in df.columns if _is_datetime(df[c])]
        if not dt_cols and not has_dt_index:
            dt_cols = [c for c in df.columns if _parses_as_dates(df[c])]
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
    num_cols = [c for c in df.columns if c != chosen_date and _is_numeric(df[c])]
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
) -> _Normalised:
    notes: list[str] = []
    if isinstance(data, pd.Series):
        if date_col is not None or value_col is not None:
            raise ColumnSelectionError(
                "date_col and value_col only apply to DataFrame input."
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
        chosen_date, chosen_value = _choose_columns(data, date_col, value_col)
        if chosen_date is None:
            raw_ds = pd.Series(data.index, name="ds")
        else:
            raw_ds = data[chosen_date].reset_index(drop=True)
        raw_y = data[chosen_value].reset_index(drop=True)
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
    frame = (
        pd.DataFrame({"ds": ds, "y": y})
        .dropna(subset=["y"])
        .sort_values("ds")
        .reset_index(drop=True)
    )
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
# Result
# --------------------------------------------------------------------------

_FORECAST_COLUMNS = ("ds", "yhat", "yhat_lower", "yhat_upper")


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

    def __str__(self) -> str:
        return (
            f"{self.n_rows_used} rows ({self.history_start:%Y-%m-%d} to "
            f"{self.history_end:%Y-%m-%d}) at freq {self.freq!r}; forecast "
            f"{self.horizon_periods} periods ({self.forecast_start:%Y-%m-%d} to "
            f"{self.forecast_end:%Y-%m-%d})"
        )


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
        called with ``tune=True``, otherwise ``None``.
    """

    model: Prophet
    history: pd.DataFrame
    predictions: pd.DataFrame
    freq: str
    freq_method: FrequencyMethod | Literal["user"]
    horizon_periods: int
    include_history: bool = False
    tuning: TuningResult | None = None

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
        """Plot history and forecast via :meth:`Prophet.plot`; kwargs are passed on."""
        return self.model.plot(self.predictions, **kwargs)

    def plot_components(self, **kwargs: Any) -> plt.Figure:
        """Plot trend and seasonality via :meth:`Prophet.plot_components`."""
        return self.model.plot_components(self.predictions, **kwargs)

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
    tune: bool = False,
    country_holidays: str | None = None,
) -> tuple[dict[str, Any], TuningResult | None]:
    """Choose the keyword arguments for ``Prophet(...)``.

    With ``tune=True`` the default grid is searched by cross-validation, minus
    any parameter the user set explicitly, so explicit arguments always win.
    """
    if not tune:
        return dict(user_params), None

    # Imported lazily: tuning pulls in process pools and cross-validation,
    # which plain forecasts never need.
    from prophet.make_holidays import make_holidays_df
    from prophet.tuning import default_param_grid
    from prophet.tuning import tune as run_tune

    grid = {k: v for k, v in default_param_grid().items() if k not in user_params}
    if not grid:
        logger.warning("tune=True ignored: every tuned parameter was set explicitly.")
        return dict(user_params), None

    # The final model adds country holidays after construction, so the tuner
    # gets them as a holidays frame to score the same model it will select.
    tuning_kwargs = dict(user_params)
    if country_holidays is not None and "holidays" not in tuning_kwargs:
        years = list(range(history["ds"].min().year, history["ds"].max().year + 1))
        tuning_kwargs["holidays"] = make_holidays_df(
            year_list=years, country=country_holidays
        )
    result = run_tune(history, param_grid=grid, prophet_kwargs=tuning_kwargs)
    logger.info("tune=True selected %s", result.best_params)
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
    tune: bool = False,
    **prophet_kwargs: Any,
) -> ForecastResult:
    """Fit a Prophet model and forecast ahead in one call.

    Parameters
    ----------
    data: A DataFrame (``ds``/``y``, any one date plus one numeric column,
        or a DatetimeIndex plus one numeric column) or a Series with a
        DatetimeIndex.  See :func:`normalise_input`.
    horizon: Periods to predict (``90``) or a duration string (``"12W"``,
        ``"6 months"``) converted to periods at the series frequency.
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
    **prophet_kwargs: Passed to :class:`prophet.Prophet`, e.g.
        ``seasonality_mode="multiplicative"``.  Logistic growth is not
        supported here because it needs a ``cap`` column; use ``Prophet``
        directly for that.

    Returns
    -------
    A frozen :class:`ForecastResult`.

    Raises
    ------
    ForecastInputError: (a ``ValueError``) for bad data, frequency or horizon.

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

    normalised = _normalise(data, date_col, value_col, tz)
    for note in normalised.notes:
        logger.warning(note)
    history = normalised.frame

    if freq is None:
        details = infer_frequency_details(history["ds"])
        if not details.regular:
            logger.warning(
                "Series is irregular (gaps or uneven spacing); using the most common "
                "spacing, freq=%r. Pass freq=... to override.",
                details.freq,
            )
        used_freq, method = details.freq, details.method
    else:
        used_freq, method = _check_freq(freq), "user"

    periods = horizon_to_periods(horizon, history["ds"].max(), used_freq)

    params, tuning = _resolve_prophet_params(
        history, prophet_kwargs, tune=tune, country_holidays=country_holidays
    )
    model = Prophet(**params)
    if country_holidays is not None:
        model.add_country_holidays(country_name=country_holidays)
    model.fit(history)

    future = model.make_future_dataframe(
        periods=periods, freq=used_freq, include_history=True
    )
    predictions = model.predict(future)
    return ForecastResult(
        model=model,
        history=history,
        predictions=predictions,
        freq=used_freq,
        freq_method=method,
        horizon_periods=periods,
        include_history=include_history,
        tuning=tuning,
    )
