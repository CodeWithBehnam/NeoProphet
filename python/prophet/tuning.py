"""Built-in hyperparameter tuning for Prophet.

``tune`` scores every combination of a parameter grid with rolling-origin
cross-validation (:func:`prophet.diagnostics.cross_validation`), in parallel,
and returns an immutable :class:`TuningResult`. With ``search="optuna"`` it
samples continuous ranges (:class:`FloatRange`, :class:`IntRange`) and
categorical choices with Optuna's TPE sampler instead, scored on the same
windows (needs ``pip install 'neoprophet[optuna]'``).

Example
-------
Scripts must guard the entry point, because the process pool uses the
``spawn`` start method on macOS and Windows::

    import pandas as pd
    from prophet.tuning import tune

    if __name__ == "__main__":
        df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
        result = tune(df)                      # about 10 seconds on 12 cores
        print(result.best_params)
        print(result.scores.head())            # one row per combination
        model = result.best_model()            # fitted on the full history
        forecast = model.predict(model.make_future_dataframe(periods=365))

        result = tune(df, search="optuna", n_trials=40)
        result.study                           # the optuna.Study, for its plots

Design notes
------------
The module follows a functional core, imperative shell split. Grid
construction, window derivation, metric ranking and result assembly are pure
functions. Model fitting and the process pool live only in ``_score_params``
and ``tune``. Optuna is imported only by :mod:`prophet.tuning_optuna`, which
``tune`` loads on demand, so ``import prophet`` never needs it.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import inspect
import itertools
import logging
import math
import multiprocessing
import numbers
import os
import sys
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd

from .diagnostics import cross_validation, generate_cutoffs, performance_metrics
from .forecaster import Prophet

if TYPE_CHECKING:
    import optuna

logger: logging.Logger = logging.getLogger("prophet")

METRICS: Final[tuple[str, ...]] = (
    "mse",
    "rmse",
    "mae",
    "mape",
    "mdape",
    "smape",
    "coverage",
)

# A fold trained on fewer points than this says nothing about the parameters.
MIN_TRAIN_POINTS: Final[int] = 10

# Roughly the cost of the default grid (32 combinations plus the baseline).
DEFAULT_N_TRIALS: Final[int] = 40

Search = Literal["grid", "optuna"]
SEARCHES: Final[tuple[str, ...]] = ("grid", "optuna")

_YEAR: Final[pd.Timedelta] = pd.Timedelta(days=365.25)
_WEEK: Final[pd.Timedelta] = pd.Timedelta(days=7)
_DAY: Final[pd.Timedelta] = pd.Timedelta(days=1)


class TuningError(RuntimeError):
    """Raised when scoring a parameter combination fails."""


# ---------------------------------------------------------------------------
# Immutable data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CVWindows:
    """Cross-validation windows used to score every combination.

    Attributes
    ----------
    horizon: Length of each forecast.
    initial: Minimum training span before the first cutoff.
    period: Spacing between consecutive cutoffs.
    cutoffs: Cutoff dates, oldest first.
    """

    horizon: pd.Timedelta
    initial: pd.Timedelta
    period: pd.Timedelta
    cutoffs: tuple[pd.Timestamp, ...]

    @property
    def n_folds(self) -> int:
        return len(self.cutoffs)


@dataclass(frozen=True)
class Trial:
    """Outcome of cross-validating one parameter combination.

    ``number`` is the Optuna trial number (``None`` in grid search, and for a
    baseline outside the search space). A ``pruned`` trial was stopped after
    some folds, so its metrics cover only those folds.
    """

    params: dict[str, Any]
    metrics: dict[str, float]
    seconds: float
    is_default: bool = False
    number: int | None = None
    pruned: bool = False


@dataclass(frozen=True)
class FloatRange:
    """Continuous search range for ``search="optuna"``, both ends included.

    ``log=True`` samples on a log scale, which suits prior scales.
    """

    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        bounds = (self.low, self.high)
        if not all(
            isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)
            for v in bounds
        ):
            raise ValueError(f"FloatRange bounds must be finite numbers, got {self}.")
        if self.low > self.high:
            raise ValueError(f"FloatRange needs low <= high, got {self}.")
        if self.log and self.low <= 0:
            raise ValueError(f"FloatRange(log=True) needs low > 0, got {self}.")


@dataclass(frozen=True)
class IntRange:
    """Integer search range for ``search="optuna"``, both ends included.

    Values are ``low, low + step, ...`` up to ``high``. ``log=True`` needs
    ``step=1``, as in Optuna.
    """

    low: int
    high: int
    log: bool = False
    step: int = 1

    def __post_init__(self) -> None:
        values = (self.low, self.high, self.step)
        if not all(
            isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in values
        ):
            raise ValueError(f"IntRange bounds and step must be integers, got {self}.")
        if self.low > self.high:
            raise ValueError(f"IntRange needs low <= high, got {self}.")
        if self.step < 1:
            raise ValueError(f"IntRange needs step >= 1, got {self}.")
        if self.log and (self.low < 1 or self.step != 1):
            raise ValueError(f"IntRange(log=True) needs low >= 1 and step=1, got {self}.")


# One parameter's candidates: a range, or a sequence of categorical choices.
SearchDimension = Union[FloatRange, IntRange, tuple[Any, ...]]
SearchSpace = dict[str, SearchDimension]


@dataclass(frozen=True)
class TuningResult:
    """Result of :func:`tune`.

    Attributes
    ----------
    best_params: Parameters of the best combination, as passed to ``Prophet``.
    scores: One row per combination, best first. Columns are the tuned
        parameters, every available metric, ``score`` (the value that was
        ranked, see below), ``rank`` (1 is best) and ``seconds`` (wall-clock
        time of that combination's cross-validation).
    metric: Name of the ranking metric. ``score`` equals the metric, except
        for ``coverage`` where it is the distance from ``interval_width``.
    windows: Cross-validation windows that were used.
    prophet_kwargs: Fixed ``Prophet`` arguments supplied by the caller.
    regressors: Names of extra regressors added to every model.
    seed: Seed used for every combination.
    study: The ``optuna.Study`` behind ``search="optuna"`` (``None`` for grid
        search), for Optuna's own analysis such as
        ``optuna.visualization.plot_param_importances(result.study)``. With
        Optuna, ``scores`` also has ``trial`` (the Optuna trial number) and
        ``pruned`` columns; pruned trials rank after all completed ones.

    Why ``best_model()`` and not a fitted attribute: tuning returns plain
    data, and fitting on the full history costs about one extra second per
    call. ``best_model`` lets the caller fit only when needed, and on another
    frame (for example with refreshed data) if they wish.
    """

    best_params: dict[str, Any]
    scores: pd.DataFrame
    metric: str
    windows: CVWindows
    prophet_kwargs: dict[str, Any]
    regressors: tuple[str, ...]
    seed: int
    history: pd.DataFrame = field(repr=False, compare=False)
    study: optuna.Study | None = field(default=None, repr=False, compare=False)

    @property
    def cutoffs(self) -> tuple[pd.Timestamp, ...]:
        return self.windows.cutoffs

    def best_model(self, df: pd.DataFrame | None = None) -> Prophet:
        """Build a Prophet model with the best parameters and fit it.

        Parameters
        ----------
        df: Data to fit. Defaults to the frame given to :func:`tune`.

        Uncertainty intervals are enabled as usual (``uncertainty_samples``
        defaults to Prophet's own value), even though tuning disables them
        for speed.
        """
        model = _build_model(self.prophet_kwargs, self.best_params, self.regressors)
        return model.fit(self.history if df is None else df)


# ---------------------------------------------------------------------------
# Pure functions: grid
# ---------------------------------------------------------------------------


def default_param_grid() -> dict[str, tuple[Any, ...]]:
    """Default grid: 4 x 4 x 2 = 32 combinations.

    The two prior scales follow the Prophet documentation. ``seasonality_mode``
    is added because it was the largest single gain in our measurements
    (about halving RMSE on the monthly air passengers and retail sales series)
    for roughly 30% more runtime. Drop it with
    ``{k: v for k, v in default_param_grid().items() if k != "seasonality_mode"}``
    if the seasonal swings of your series do not scale with the level.
    """
    return {
        "changepoint_prior_scale": (0.001, 0.01, 0.1, 0.5),
        "seasonality_prior_scale": (0.01, 0.1, 1.0, 10.0),
        "seasonality_mode": ("additive", "multiplicative"),
    }


def default_search_space() -> SearchSpace:
    """Default ``search="optuna"`` space: the default grid's parameters as ranges.

    The prior scale ranges are the ones the Prophet documentation suggests,
    sampled on a log scale because both act as regularisation penalties.
    Prophet's defaults (0.05, 10.0, ``"additive"``) lie inside the space.
    """
    return {
        "changepoint_prior_scale": FloatRange(0.001, 0.5, log=True),
        "seasonality_prior_scale": FloatRange(0.01, 10.0, log=True),
        "seasonality_mode": ("additive", "multiplicative"),
    }


def _prophet_parameter_names() -> frozenset[str]:
    names = inspect.signature(Prophet.__init__).parameters
    return frozenset(n for n in names if n != "self")


def _check_names(names: Sequence[str], prophet_kwargs: Mapping[str, Any] | None) -> None:
    if not names:
        raise ValueError("param_grid must contain at least one parameter.")
    valid = _prophet_parameter_names()
    unknown = sorted(set(names) - valid)
    if unknown:
        raise ValueError(
            f"Unknown Prophet parameter(s) in param_grid: {unknown}. "
            f"Valid names are: {sorted(valid)}."
        )
    clash = sorted(set(names) & set(prophet_kwargs or {}))
    if clash:
        raise ValueError(
            f"Parameter(s) {clash} appear in both param_grid and prophet_kwargs. "
            "Keep them in one place."
        )


def _check_choices(name: str, values: object) -> tuple[Any, ...]:
    if isinstance(values, (FloatRange, IntRange)):
        raise ValueError(
            f"param_grid[{name!r}] is a {type(values).__name__}, which only "
            "search='optuna' can sample. Use a list of values for grid search."
        )
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(
            f"param_grid[{name!r}] must be a list or tuple of candidate values, "
            f"got {type(values).__name__}."
        )
    if len(values) == 0:
        raise ValueError(f"param_grid[{name!r}] is empty.")
    return tuple(values)


def validate_param_grid(
    param_grid: Mapping[str, Sequence[Any] | FloatRange | IntRange],
    prophet_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Check a grid against ``Prophet.__init__`` and return a normalised copy.

    Raises
    ------
    ValueError: If the grid is empty, a name is not a ``Prophet`` argument,
        a name is also fixed in ``prophet_kwargs``, or a value list is empty
        or not a sequence.
    """
    _check_names(list(param_grid), prophet_kwargs)
    return {name: _check_choices(name, values) for name, values in param_grid.items()}


def _is_plain_choice(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def validate_search_space(
    space: Mapping[str, Sequence[Any] | FloatRange | IntRange],
    prophet_kwargs: Mapping[str, Any] | None = None,
) -> SearchSpace:
    """Check a ``search="optuna"`` space and return a normalised copy.

    Each value is a :class:`FloatRange`, an :class:`IntRange` or a list of
    categorical choices. Choices must be ``None``, bool, int, float or str,
    which is what Optuna can store; anything else (a holidays frame, say)
    belongs in grid search or ``prophet_kwargs``.

    Raises
    ------
    ValueError: As :func:`validate_param_grid`, or for a choice of another type.
    """
    _check_names(list(space), prophet_kwargs)

    def normalise(name: str, values: object) -> SearchDimension:
        if isinstance(values, (FloatRange, IntRange)):
            return values
        choices = _check_choices(name, values)
        odd = [c for c in choices if not _is_plain_choice(c)]
        if odd:
            raise ValueError(
                f"param_grid[{name!r}] has choices Optuna cannot store: {odd!r}. "
                "Use None, bool, int, float or str, or search='grid'."
            )
        # Optuna stores a choice as its index, found with ==, so choices that
        # compare equal would be learned as one (True == 1, 1 == 1.0).
        equal = [
            (a, b)
            for i, a in enumerate(choices)
            for b in choices[i + 1 :]
            if a == b
        ]
        if equal:
            raise ValueError(
                f"param_grid[{name!r}] has choices that compare equal, which "
                f"Optuna cannot tell apart: {equal!r}. Keep one of each pair, "
                "or use search='grid'."
            )
        return choices

    return {name: normalise(name, values) for name, values in space.items()}


def _same_choice(value: object, choice: object) -> bool:
    # True == 1 and 1 == 1.0 in Python, but Prophet treats them differently.
    return type(value) is type(choice) and value == choice


def _in_dimension(value: object, dim: SearchDimension) -> bool:
    if isinstance(value, bool):
        return not isinstance(dim, (FloatRange, IntRange)) and any(
            _same_choice(value, c) for c in dim
        )
    if isinstance(dim, FloatRange):
        return isinstance(value, numbers.Real) and dim.low <= float(value) <= dim.high
    if isinstance(dim, IntRange):
        if not isinstance(value, numbers.Integral):
            return False
        number = int(value)
        return dim.low <= number <= dim.high and (number - dim.low) % dim.step == 0
    return any(_same_choice(value, c) for c in dim)


def space_contains(space: Mapping[str, SearchDimension], params: Mapping[str, Any]) -> bool:
    """Whether ``params`` names exactly the space's parameters, each within range."""
    return set(params) == set(space) and all(
        _in_dimension(params[name], dim) for name, dim in space.items()
    )


def _dimension_size(dim: SearchDimension) -> int | None:
    if isinstance(dim, FloatRange):
        return None
    if isinstance(dim, IntRange):
        return (dim.high - dim.low) // dim.step + 1
    return len(dim)


def space_size(space: Mapping[str, SearchDimension]) -> int | None:
    """Number of distinct points, or ``None`` if a dimension is continuous."""
    sizes = [_dimension_size(dim) for dim in space.values()]
    if any(s is None for s in sizes):
        return None
    return math.prod(s for s in sizes if s is not None)


def expand_grid(param_grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a grid, in deterministic order."""
    names = list(param_grid)
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(*(param_grid[n] for n in names))
    ]


def prophet_defaults(names: Sequence[str]) -> dict[str, Any]:
    """Default values of the named ``Prophet.__init__`` arguments."""
    params = inspect.signature(Prophet.__init__).parameters
    return {n: params[n].default for n in names}


def with_baseline(
    combos: Sequence[dict[str, Any]], grid_names: Sequence[str]
) -> tuple[list[dict[str, Any]], int]:
    """Make sure Prophet's own defaults are among the combinations.

    Returns the combinations and the index of the baseline. Scoring the
    default configuration alongside the grid means the winner can never be
    worse than untuned Prophet in cross-validation, and the ``scores`` table
    shows how much tuning gained. The default grid omits 0.05, Prophet's
    default ``changepoint_prior_scale``, so this matters.
    """
    baseline = prophet_defaults(grid_names)
    if baseline in combos:
        return list(combos), list(combos).index(baseline)
    return [*combos, baseline], len(combos)


# ---------------------------------------------------------------------------
# Pure functions: windows
# ---------------------------------------------------------------------------


def infer_step(ds: pd.Series) -> pd.Timedelta:
    """Typical spacing of a timestamp column (median of the positive gaps)."""
    gaps = ds.drop_duplicates().sort_values().diff().dropna()
    if gaps.empty:
        raise ValueError(
            "Need at least two distinct timestamps to infer the frequency."
        )
    return pd.Timedelta(gaps.median())


def longest_seasonality(span: pd.Timedelta, step: pd.Timedelta) -> pd.Timedelta:
    """Longest seasonal period Prophet would switch on automatically.

    Mirrors Prophet's ``'auto'`` rules: yearly needs two years of history,
    weekly two weeks and a spacing below a week, daily two days and a spacing
    below a day. Falls back to one step when none applies.
    """
    if span >= 2 * _YEAR:
        return _YEAR
    if span >= 2 * _WEEK and step < _WEEK:
        return _WEEK
    if span >= 2 * _DAY and step < _DAY:
        return _DAY
    return step


def _round_up(value: pd.Timedelta, step: pd.Timedelta) -> pd.Timedelta:
    return step * max(1, math.ceil(value / step))


def _try_cutoffs(
    ds: pd.Series,
    horizon: pd.Timedelta,
    initial: pd.Timedelta,
    period: pd.Timedelta,
) -> tuple[pd.Timestamp, ...]:
    try:
        # generate_cutoffs only reads the 'ds' column.
        cutoffs = generate_cutoffs(pd.DataFrame({"ds": ds}), horizon, initial, period)
    except ValueError:
        return ()
    enough = ds.searchsorted(pd.Series(cutoffs), side="right") >= MIN_TRAIN_POINTS
    return tuple(c for c, ok in zip(cutoffs, enough, strict=True) if ok)


def _candidate_windows(
    span: pd.Timedelta,
    step: pd.Timedelta,
    seasonal: pd.Timedelta,
    horizon: pd.Timedelta | None,
    initial: pd.Timedelta | None,
    period: pd.Timedelta | None,
    max_folds: int,
) -> Iterator[tuple[pd.Timedelta, pd.Timedelta, pd.Timedelta]]:
    """Yield (horizon, initial, period) triples from most to least generous.

    Only the arguments the caller omitted are relaxed.
    """
    h0 = horizon if horizon is not None else _round_up(min(span / 10, _YEAR), step)
    horizons = [h0] if horizon is not None else [h0, h0 / 2, h0 / 4]
    for h_raw in horizons:
        h = _round_up(h_raw, step)
        i_options = (
            [initial]
            if initial is not None
            else [
                max(3 * h, seasonal),
                max(2 * h, seasonal),
                max(h, seasonal),
                seasonal,
            ]
        )
        for i in i_options:
            if period is not None:
                yield h, i, period
                continue
            room = max(span - h - i, step)
            # Cap the number of folds so cost stays bounded on long series.
            p_cap = _round_up(room / max(max_folds - 1, 1), step)
            yield h, i, max(_round_up(h / 2, step), p_cap)
            yield h, i, _round_up(h / 4, step)


def derive_windows(
    ds: pd.Series,
    *,
    horizon: str | pd.Timedelta | None = None,
    initial: str | pd.Timedelta | None = None,
    period: str | pd.Timedelta | None = None,
    min_folds: int = 3,
    max_folds: int = 10,
) -> CVWindows:
    """Derive cross-validation windows from the data span and frequency.

    Rules for arguments the caller omits (``span`` is last minus first
    timestamp, ``step`` the median spacing):

    * ``horizon``: ``span / 10``, capped at one year, rounded up to a whole
      number of steps.
    * ``initial``: ``max(3 * horizon, longest auto seasonality)``, so the
      first model sees at least a full seasonal cycle.
    * ``period``: ``horizon / 2``, widened if needed so that at most
      ``max_folds`` folds are produced (this bounds cost on long series).

    If this yields fewer than ``min_folds`` folds, the omitted arguments are
    relaxed step by step: a smaller ``initial`` (down to one seasonal cycle),
    then a smaller ``period`` (``horizon / 4``), then half and a quarter of
    the horizon. Arguments the caller set are never changed; with explicit
    windows fewer than ``min_folds`` folds only logs a warning.

    Raises
    ------
    ValueError: If not even one fold fits, with the span and the minimum
        span that the chosen windows would need.
    """
    if min_folds < 1 or max_folds < min_folds:
        raise ValueError("Need 1 <= min_folds <= max_folds.")
    unique_ds = ds.drop_duplicates().sort_values().reset_index(drop=True)
    step = infer_step(unique_ds)
    span = pd.Timedelta(unique_ds.iloc[-1] - unique_ds.iloc[0])
    seasonal = longest_seasonality(span, step)
    h_in = None if horizon is None else pd.Timedelta(horizon)
    i_in = None if initial is None else pd.Timedelta(initial)
    p_in = None if period is None else pd.Timedelta(period)
    for value in (h_in, i_in, p_in):
        if value is not None and value <= pd.Timedelta(0):
            raise ValueError("horizon, initial and period must be positive.")

    best: (
        tuple[pd.Timedelta, pd.Timedelta, pd.Timedelta, tuple[pd.Timestamp, ...]] | None
    ) = None
    for h, i, p in _candidate_windows(
        span, step, seasonal, h_in, i_in, p_in, max_folds
    ):
        cutoffs = _try_cutoffs(unique_ds, h, i, p)
        if best is None or len(cutoffs) > len(best[3]):
            best = (h, i, p, cutoffs)
        if len(cutoffs) >= min_folds:
            best = (h, i, p, cutoffs)
            break
    assert best is not None
    h, i, p, cutoffs = best
    if not cutoffs:
        raise ValueError(
            f"History is too short to cross-validate: it spans {span} at a "
            f"{step} step, but horizon={h} and initial={i} need more than "
            f"{h + i}. Supply a longer history or smaller horizon/initial."
        )
    if len(cutoffs) < min_folds:
        logger.warning(
            "Only %d cross-validation fold(s) fit (wanted %d); scores will be noisy.",
            len(cutoffs),
            min_folds,
        )
    return CVWindows(horizon=h, initial=i, period=p, cutoffs=cutoffs)


# ---------------------------------------------------------------------------
# Pure functions: ranking
# ---------------------------------------------------------------------------


def trial_score(
    metrics: Mapping[str, float], metric: str, interval_width: float
) -> float:
    """Value to minimise. Coverage is scored by distance from the nominal width."""
    value = metrics.get(metric, math.nan)
    return abs(value - interval_width) if metric == "coverage" else value


def rank_trials(
    trials: Sequence[Trial], metric: str, interval_width: float = 0.8
) -> list[Trial]:
    """Sort trials best first. Ties keep input order; NaN scores go last.

    Pruned trials follow every completed trial (even one with a NaN score),
    because their metrics cover fewer folds and so cannot be compared.
    """

    def key(indexed: tuple[int, Trial]) -> tuple[bool, bool, float, int]:
        idx, trial = indexed
        score = trial_score(trial.metrics, metric, interval_width)
        nan = math.isnan(score)
        return (trial.pruned, nan, 0.0 if nan else score, idx)

    return [t for _, t in sorted(enumerate(trials), key=key)]


def scores_frame(
    ranked: Sequence[Trial],
    metric: str,
    interval_width: float = 0.8,
    *,
    optuna_columns: bool = False,
) -> pd.DataFrame:
    """Tidy table from already ranked trials.

    ``optuna_columns`` adds ``trial`` (nullable integer) and ``pruned``.
    """
    rows = [
        {
            **t.params,
            **t.metrics,
            "score": trial_score(t.metrics, metric, interval_width),
            "rank": rank,
            "seconds": t.seconds,
            "is_default": t.is_default,
            **({"trial": t.number, "pruned": t.pruned} if optuna_columns else {}),
        }
        for rank, t in enumerate(ranked, start=1)
    ]
    frame = pd.DataFrame(rows)
    if optuna_columns:
        frame["trial"] = frame["trial"].astype("Int64")
    return frame


# ---------------------------------------------------------------------------
# Shell: model fitting and process pool
# ---------------------------------------------------------------------------


def _build_model(
    prophet_kwargs: Mapping[str, Any],
    params: Mapping[str, Any],
    regressors: Sequence[str],
) -> Prophet:
    model = Prophet(**{**prophet_kwargs, **params})
    for name in regressors:
        model.add_regressor(name)
    return model


@dataclass(frozen=True)
class _Task:
    df: pd.DataFrame
    params: dict[str, Any]
    prophet_kwargs: dict[str, Any]
    regressors: tuple[str, ...]
    windows: CVWindows
    seed: int


def _keep_cmdstan_record(record: logging.LogRecord) -> bool:
    """Keep only CRITICAL cmdstanpy records while tuning.

    Every fit logs INFO lines, and L-BFGS fails as expected at extreme grid
    points (cmdstanpy reports exit code 1 as "Operation not permitted").
    Prophet then refits with Newton; if Newton fails too it raises, which
    the tuner reports as a TuningError, so dropping these records loses
    nothing a user needs.
    """
    return record.levelno >= logging.CRITICAL


def _keep_models_record(record: logging.LogRecord) -> bool:
    """Drop Prophet's Newton fallback warning while tuning, for the reason above."""
    return "Falling back to Newton" not in record.getMessage()


@contextlib.contextmanager
def _quiet_cmdstan() -> Iterator[None]:
    """Silence per-fit log lines and expected optimiser fallbacks while tuning.

    Logger filters are used because cmdstanpy resets the logger level on
    every fit, so ``setLevel`` does not stick.
    """
    cmdstan = logging.getLogger("cmdstanpy")
    models = logging.getLogger("prophet.models")
    cmdstan.addFilter(_keep_cmdstan_record)
    models.addFilter(_keep_models_record)
    try:
        yield
    finally:
        cmdstan.removeFilter(_keep_cmdstan_record)
        models.removeFilter(_keep_models_record)


def _overall_metrics(cv: pd.DataFrame) -> dict[str, float]:
    # rolling_window=1 pools every forecast point into a single row.
    table = performance_metrics(cv, metrics=list(METRICS), rolling_window=1)
    if table is None:
        raise ValueError("No performance metric could be computed.")
    last = table.iloc[-1]
    return {m: float(last[m]) for m in METRICS if m in table.columns}


def _score_params(task: _Task) -> Trial:
    """Cross-validate one combination. Module-level so it can be pickled."""
    start = time.perf_counter()
    try:
        with _quiet_cmdstan():
            # Prophet draws uncertainty samples from the global numpy RNG.
            np.random.seed(task.seed)
            model = _build_model(task.prophet_kwargs, task.params, task.regressors)
            model.fit(task.df)
            cv = cross_validation(
                model,
                horizon=task.windows.horizon,
                cutoffs=list(task.windows.cutoffs),
                disable_tqdm=True,
            )
            metrics = _overall_metrics(cv)
    except Exception as exc:
        raise TuningError(f"Cross-validation failed for {task.params}: {exc}") from exc
    return Trial(
        params=task.params, metrics=metrics, seconds=time.perf_counter() - start
    )


def _score_params_by_fold(
    task: _Task, should_stop: Callable[[int, Mapping[str, float]], bool]
) -> Trial:
    """Cross-validate one combination fold by fold, so a pruner can stop it.

    After every fold but the last (stopping then would save nothing),
    ``should_stop(step, metrics)`` receives the 0-based fold index and the
    metrics pooled over the folds so far; ``True`` ends the trial as pruned.
    A trial that runs to the end gets the same metrics as ``_score_params``,
    because the folds run in the same order from the same RNG seed.
    """

    def fail(exc: Exception) -> TuningError:
        return TuningError(f"Cross-validation failed for {task.params}: {exc}")

    start = time.perf_counter()
    try:
        with _quiet_cmdstan():
            np.random.seed(task.seed)
            model = _build_model(task.prophet_kwargs, task.params, task.regressors)
            model.fit(task.df)
    except Exception as exc:
        raise fail(exc) from exc
    folds: list[pd.DataFrame] = []
    metrics: dict[str, float] = {}
    last = task.windows.n_folds - 1
    for step, cutoff in enumerate(task.windows.cutoffs):
        try:
            with _quiet_cmdstan():
                folds.append(
                    cross_validation(
                        model,
                        horizon=task.windows.horizon,
                        cutoffs=[cutoff],
                        disable_tqdm=True,
                    )
                )
                metrics = _overall_metrics(pd.concat(folds, ignore_index=True))
        except Exception as exc:
            raise fail(exc) from exc
        # The callback runs outside the try so its own errors are not
        # reported as cross-validation failures.
        if step < last and should_stop(step, metrics):
            return Trial(
                params=task.params,
                metrics=metrics,
                seconds=time.perf_counter() - start,
                pruned=True,
            )
    return Trial(
        params=task.params, metrics=metrics, seconds=time.perf_counter() - start
    )


def _mp_context() -> multiprocessing.context.BaseContext:
    # Same choice as prophet.diagnostics: fork is unsafe with Stan on macOS.
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return multiprocessing.get_context("spawn")
    return multiprocessing.get_context("forkserver")


def _run_tasks(
    tasks: Sequence[_Task], parallel: bool, max_workers: int | None
) -> list[Trial]:
    workers = min(len(tasks), max_workers or os.cpu_count() or 1)
    if not parallel or workers <= 1:
        return [_score_params(t) for t in tasks]
    logger.info("Tuning %d combinations on %d worker processes.", len(tasks), workers)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=_mp_context()
    ) as pool:
        return list(pool.map(_score_params, tasks))


def _prepare_frame(df: pd.DataFrame, regressors: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in ("ds", "y", *regressors) if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required column(s): {missing}.")
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    if out["ds"].dt.tz is not None:
        raise ValueError(
            "Timestamps in 'ds' must be timezone-naive, as Prophet requires."
        )
    if not out["ds"].is_unique:
        raise ValueError("Timestamps in 'ds' must be unique.")
    return out.sort_values("ds").reset_index(drop=True)


def _check_search_options(
    search: str,
    n_trials: int | None,
    timeout: float | None,
    prune: bool,
    parallel: bool,
) -> int:
    """Validate the Optuna-only arguments and return the number of trials."""
    if search not in SEARCHES:
        raise ValueError(f"Unknown search {search!r}. Valid values: {list(SEARCHES)}.")
    if search == "grid":
        given = [
            name
            for name, is_set in (
                ("n_trials", n_trials is not None),
                ("timeout", timeout is not None),
                ("prune", prune),
            )
            if is_set
        ]
        if given:
            raise ValueError(f"{given} only apply to search='optuna'.")
        return 0
    trials = DEFAULT_N_TRIALS if n_trials is None else n_trials
    if isinstance(trials, bool) or not isinstance(trials, numbers.Integral):
        raise ValueError(f"n_trials must be an integer, got {trials!r}.")
    if trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {trials}.")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, numbers.Real)
        or not timeout > 0
    ):
        raise ValueError(
            f"timeout must be a positive number of seconds, got {timeout!r}."
        )
    if prune and parallel:
        raise ValueError(
            "prune=True scores one trial at a time, because the pruner must see "
            "each fold's score before the next fold runs. Pass parallel=False."
        )
    return int(trials)


def tune(
    df: pd.DataFrame,
    *,
    param_grid: Mapping[str, Sequence[Any] | FloatRange | IntRange] | None = None,
    horizon: str | pd.Timedelta | None = None,
    initial: str | pd.Timedelta | None = None,
    period: str | pd.Timedelta | None = None,
    metric: str = "rmse",
    parallel: bool = True,
    max_workers: int | None = None,
    seed: int = 0,
    prophet_kwargs: Mapping[str, Any] | None = None,
    regressors: Sequence[str] = (),
    min_folds: int = 3,
    max_folds: int = 10,
    include_defaults: bool = True,
    search: Search = "grid",
    n_trials: int | None = None,
    timeout: float | None = None,
    prune: bool = False,
) -> TuningResult:
    """Pick the Prophet hyperparameters with the best cross-validated score.

    With ``search="grid"`` (the default) every combination of ``param_grid``
    is fitted. With ``search="optuna"`` Optuna's TPE sampler proposes
    ``n_trials`` combinations in batches, each batch informed by the scores
    of the batches before it. Either way every candidate is scored on the
    same rolling-origin windows and ranked by ``metric``.

    Parameters
    ----------
    df: History with columns ``ds`` and ``y`` (plus ``cap``/``floor`` for
        logistic growth and any ``regressors``).
    param_grid: Mapping of ``Prophet`` argument name to candidate values.
        Names are validated against ``Prophet.__init__``. For grid search the
        values are lists, and the default is :func:`default_param_grid`. For
        ``search="optuna"`` a value may also be a :class:`FloatRange` or
        :class:`IntRange` (lists become categorical choices), and the default
        is :func:`default_search_space`.
    horizon, initial, period: Cross-validation windows, as in
        :func:`prophet.diagnostics.cross_validation`. Omitted ones are derived
        from the data, see :func:`derive_windows`.
    metric: One of ``mse, rmse, mae, mape, mdape, smape, coverage``. Lower is
        better, except ``coverage`` where closeness to ``interval_width`` wins.
    parallel: Score combinations in a process pool (one process per
        combination, each running its folds sequentially). With
        ``search="optuna"`` the pool scores one batch of trials at a time,
        one trial per worker. Trials in a batch do not see each other's
        scores, so a smaller pool searches better and a larger one faster.
    max_workers: Pool size. Defaults to the CPU count, capped at the number of
        combinations. With Optuna the default is also capped at
        ``n_trials // 4``, leaving at least four informed rounds.
    seed: Seeds numpy's global RNG before every combination so uncertainty
        sampling, and therefore ``coverage``, is reproducible. Also seeds the
        Optuna sampler: with the same ``seed``, ``n_trials`` and pool size,
        Optuna proposes the same trials. The default pool depends on the CPU
        count, so set ``max_workers`` to reproduce a search on other machines.
    prophet_kwargs: Fixed ``Prophet`` arguments (holidays, growth, ...).
    regressors: Columns of ``df`` to add with ``add_regressor`` defaults.
    min_folds, max_folds: Bounds on the number of derived folds.
    include_defaults: Also score Prophet's default values for the tuned
        parameters (flagged in ``scores['is_default']``), so the result can
        be compared against untuned Prophet and never loses to it in
        cross-validation. With Optuna this is one extra evaluation on top of
        ``n_trials``, and it is added to the study when it lies inside the
        search space.
    search: ``"grid"`` or ``"optuna"``. Optuna needs
        ``pip install 'neoprophet[optuna]'``.
    n_trials: Number of Optuna trials, default 40 (about the cost of the
        default grid). Optuna only.
    timeout: Stop proposing trials after this many seconds. Checked between
        batches, so the batch running at the deadline still finishes.
        Optuna only.
    prune: Stop unpromising Optuna trials early. After each fold from the
        second to the second-to-last, a trial whose score pooled over its
        folds so far is worse than the median of completed trials at the same
        fold is stopped. Pruning starts once 5 trials have completed (the
        baseline counts as one) and needs at least 3 folds, so it has no
        effect on very small searches. Needs ``parallel=False``, because the
        pruner must see each fold before the next one runs.

    Returns
    -------
    TuningResult, see its docstring.

    Notes
    -----
    Uncertainty sampling is switched off while scoring (``uncertainty_samples=0``)
    unless ``metric='coverage'``, which makes prediction much cheaper. The
    model from ``result.best_model()`` uses your own setting.

    Example
    -------
    >>> result = tune(df, param_grid={"changepoint_prior_scale": [0.01, 0.1]})  # doctest: +SKIP
    >>> result.best_params  # doctest: +SKIP
    {'changepoint_prior_scale': 0.1}
    >>> space = {"changepoint_prior_scale": FloatRange(0.001, 0.5, log=True)}
    >>> result = tune(df, param_grid=space, search="optuna", n_trials=30)  # doctest: +SKIP
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Valid metrics: {list(METRICS)}.")
    trials_wanted = _check_search_options(search, n_trials, timeout, prune, parallel)
    fixed = dict(prophet_kwargs or {})
    grid = (
        validate_param_grid(
            default_param_grid() if param_grid is None else param_grid, fixed
        )
        if search == "grid"
        else None
    )
    space = (
        validate_search_space(
            default_search_space() if param_grid is None else param_grid, fixed
        )
        if search == "optuna"
        else None
    )
    names = list(grid if grid is not None else space or {})
    frame = _prepare_frame(df, regressors)
    if metric == "mape" and frame["y"].abs().min() < 1e-8:
        raise ValueError(
            "metric='mape' is undefined because y contains values close to 0."
        )

    windows = derive_windows(
        frame["ds"],
        horizon=horizon,
        initial=initial,
        period=period,
        min_folds=min_folds,
        max_folds=max_folds,
    )
    logger.info(
        "Tuning with %d folds (horizon=%s, initial=%s, period=%s).",
        windows.n_folds,
        windows.horizon,
        windows.initial,
        windows.period,
    )

    # Intervals are only worth sampling when the metric reads them.
    scoring_kwargs = dict(fixed)
    if metric != "coverage" and "uncertainty_samples" not in names:
        scoring_kwargs["uncertainty_samples"] = 0
    interval_width = float(fixed.get("interval_width", 0.8))

    def make_task(params: dict[str, Any]) -> _Task:
        return _Task(frame, params, scoring_kwargs, tuple(regressors), windows, seed)

    study: optuna.Study | None = None
    if grid is not None:
        combos = expand_grid(grid)
        baseline_idx = -1
        if include_defaults:
            combos, baseline_idx = with_baseline(combos, list(grid))
        trials = [
            dataclasses.replace(t, is_default=i == baseline_idx)
            for i, t in enumerate(
                _run_tasks([make_task(p) for p in combos], parallel, max_workers)
            )
        ]
    else:
        assert space is not None
        # Imported here: Optuna is an optional dependency.
        from .tuning_optuna import run_study

        trials, study = run_study(
            make_task,
            space,
            n_trials=trials_wanted,
            timeout=timeout,
            seed=seed,
            parallel=parallel,
            max_workers=max_workers,
            prune=prune,
            n_folds=windows.n_folds,
            baseline=prophet_defaults(names) if include_defaults else None,
            score=lambda metrics: trial_score(metrics, metric, interval_width),
        )

    ranked = rank_trials(trials, metric, interval_width)
    best = ranked[0]
    if best.pruned or math.isnan(trial_score(best.metrics, metric, interval_width)):
        raise TuningError(f"No combination produced a finite {metric!r} score.")
    return TuningResult(
        best_params=dict(best.params),
        scores=scores_frame(
            ranked, metric, interval_width, optuna_columns=study is not None
        ),
        metric=metric,
        windows=windows,
        prophet_kwargs=fixed,
        regressors=tuple(regressors),
        seed=seed,
        history=frame,
        study=study,
    )
