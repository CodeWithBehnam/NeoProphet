"""Built-in hyperparameter tuning for Prophet.

``tune`` scores every combination of a parameter grid with rolling-origin
cross-validation (:func:`prophet.diagnostics.cross_validation`), in parallel,
and returns an immutable :class:`TuningResult`.

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

Design notes
------------
The module follows a functional core, imperative shell split. Grid
construction, window derivation, metric ranking and result assembly are pure
functions. Model fitting and the process pool live only in ``_score_params``
and ``tune``.
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
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Final, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .diagnostics import cross_validation, generate_cutoffs, performance_metrics
from .forecaster import Prophet

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
    """Outcome of cross-validating one parameter combination."""

    params: dict[str, Any]
    metrics: dict[str, float]
    seconds: float
    is_default: bool = False


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


def _prophet_parameter_names() -> frozenset[str]:
    names = inspect.signature(Prophet.__init__).parameters
    return frozenset(n for n in names if n != "self")


def validate_param_grid(
    param_grid: Mapping[str, Sequence[Any]],
    prophet_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Check a grid against ``Prophet.__init__`` and return a normalised copy.

    Raises
    ------
    ValueError: If the grid is empty, a name is not a ``Prophet`` argument,
        a name is also fixed in ``prophet_kwargs``, or a value list is empty
        or not a sequence.
    """
    if not param_grid:
        raise ValueError("param_grid must contain at least one parameter.")
    valid = _prophet_parameter_names()
    unknown = sorted(set(param_grid) - valid)
    if unknown:
        raise ValueError(
            f"Unknown Prophet parameter(s) in param_grid: {unknown}. "
            f"Valid names are: {sorted(valid)}."
        )
    clash = sorted(set(param_grid) & set(prophet_kwargs or {}))
    if clash:
        raise ValueError(
            f"Parameter(s) {clash} appear in both param_grid and prophet_kwargs. "
            "Keep them in one place."
        )
    out: dict[str, tuple[Any, ...]] = {}
    for name, values in param_grid.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(
                f"param_grid[{name!r}] must be a list or tuple of candidate values, "
                f"got {type(values).__name__}."
            )
        if len(values) == 0:
            raise ValueError(f"param_grid[{name!r}] is empty.")
        out[name] = tuple(values)
    return out


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
    """Sort trials best first. Ties keep grid order; NaN scores go last."""

    def key(indexed: tuple[int, Trial]) -> tuple[bool, float, int]:
        idx, trial = indexed
        score = trial_score(trial.metrics, metric, interval_width)
        return (math.isnan(score), 0.0 if math.isnan(score) else score, idx)

    return [t for _, t in sorted(enumerate(trials), key=key)]


def scores_frame(
    ranked: Sequence[Trial], metric: str, interval_width: float = 0.8
) -> pd.DataFrame:
    """Tidy table from already ranked trials."""
    rows = [
        {
            **t.params,
            **t.metrics,
            "score": trial_score(t.metrics, metric, interval_width),
            "rank": rank,
            "seconds": t.seconds,
            "is_default": t.is_default,
        }
        for rank, t in enumerate(ranked, start=1)
    ]
    return pd.DataFrame(rows)


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


@contextlib.contextmanager
def _quiet_cmdstan() -> Iterator[None]:
    """Silence per-fit INFO lines, which would otherwise be thousands.

    A logger filter is used because cmdstanpy resets the logger level on
    every fit, so ``setLevel`` does not stick.
    """

    def drop_info(record: logging.LogRecord) -> bool:
        return record.levelno > logging.INFO

    log = logging.getLogger("cmdstanpy")
    log.addFilter(drop_info)
    try:
        yield
    finally:
        log.removeFilter(drop_info)


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


def tune(
    df: pd.DataFrame,
    *,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
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
) -> TuningResult:
    """Pick the Prophet hyperparameters with the best cross-validated score.

    Every combination of ``param_grid`` is fitted and scored on the same
    rolling-origin windows, then ranked by ``metric``.

    Parameters
    ----------
    df: History with columns ``ds`` and ``y`` (plus ``cap``/``floor`` for
        logistic growth and any ``regressors``).
    param_grid: Mapping of ``Prophet`` argument name to candidate values.
        Defaults to :func:`default_param_grid`. Names are validated against
        ``Prophet.__init__``.
    horizon, initial, period: Cross-validation windows, as in
        :func:`prophet.diagnostics.cross_validation`. Omitted ones are derived
        from the data, see :func:`derive_windows`.
    metric: One of ``mse, rmse, mae, mape, mdape, smape, coverage``. Lower is
        better, except ``coverage`` where closeness to ``interval_width`` wins.
    parallel: Score combinations in a process pool (one process per
        combination, each running its folds sequentially).
    max_workers: Pool size. Defaults to the CPU count, capped at the number of
        combinations.
    seed: Seeds numpy's global RNG before every combination so uncertainty
        sampling, and therefore ``coverage``, is reproducible.
    prophet_kwargs: Fixed ``Prophet`` arguments (holidays, growth, ...).
    regressors: Columns of ``df`` to add with ``add_regressor`` defaults.
    min_folds, max_folds: Bounds on the number of derived folds.
    include_defaults: Also score Prophet's default values for the tuned
        parameters (flagged in ``scores['is_default']``), so the result can
        be compared against untuned Prophet and never loses to it in
        cross-validation.

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
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Valid metrics: {list(METRICS)}.")
    fixed = dict(prophet_kwargs or {})
    grid = validate_param_grid(
        default_param_grid() if param_grid is None else param_grid, fixed
    )
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
    if metric != "coverage" and "uncertainty_samples" not in grid:
        scoring_kwargs["uncertainty_samples"] = 0
    combos = expand_grid(grid)
    baseline_idx = -1
    if include_defaults:
        combos, baseline_idx = with_baseline(combos, list(grid))
    tasks = [
        _Task(frame, params, scoring_kwargs, tuple(regressors), windows, seed)
        for params in combos
    ]
    trials = [
        dataclasses.replace(t, is_default=i == baseline_idx)
        for i, t in enumerate(_run_tasks(tasks, parallel, max_workers))
    ]

    interval_width = float(fixed.get("interval_width", 0.8))
    ranked = rank_trials(trials, metric, interval_width)
    if math.isnan(trial_score(ranked[0].metrics, metric, interval_width)):
        raise TuningError(f"No combination produced a finite {metric!r} score.")
    return TuningResult(
        best_params=dict(ranked[0].params),
        scores=scores_frame(ranked, metric, interval_width),
        metric=metric,
        windows=windows,
        prophet_kwargs=fixed,
        regressors=tuple(regressors),
        seed=seed,
        history=frame,
    )
