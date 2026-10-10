"""Benchmark whichever ``prophet`` is importable on speed and holdout accuracy.

The same script is run once inside the NeoProphet project environment and once
inside an isolated environment holding upstream ``prophet`` from PyPI. Both
packages import as ``prophet``, so they cannot share an interpreter, and using
one script keeps the measurement symmetric. Only APIs present in prophet 1.3.0
and 1.5.0 are used.

Holdout design: the final ``horizon`` observations of each series are held out,
the model is fitted on the rest, and ``predict`` is called on the held-out
timestamps with uncertainty intervals. ``np.random.seed`` is set immediately
before every ``predict`` so the sampled intervals are comparable.

One untimed warm-up fit runs first so one-off start-up costs are excluded.

Variants (``--variant``), chosen so fork-only features never run against upstream:

* ``default``: plain ``Prophet(**config)``. Works on upstream and the fork.
* ``tuned``: ``prophet.tuning.tune`` on the training part only (never the
  holdout), then the best model is fitted and the holdout forecast. Tuning
  time is recorded apart from fit and predict time. Fork only.
* ``mlx``: ``Prophet(stan_backend="MLX", **config)``. Datasets the backend
  cannot fit (logistic growth, MCMC) are skipped with a recorded reason. Fork
  only.

``--throughput 10,100`` adds a many-series scenario: N seeded, perturbed copies
of Peyton Manning fitted one Prophet per series with CmdStan, and, for the
``mlx`` variant, also with ``fit_many`` in one batched optimisation.

Scaling for MASE: the mean absolute error of the in-sample seasonal-naive
forecast with period ``season`` (7 for daily, 12 for monthly, 288 for
five-minute data). Series shorter than ``season + 1`` fall back to the naive
(lag 1) error.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("compare_upstream")

DEFAULT_REPEATS = 3
DEFAULT_SEED = 42
DEFAULT_TUNE_REPEATS = 1

VARIANTS = ("default", "tuned", "mlx")
# Modules a variant needs from the installed prophet; upstream 1.5.0 has neither.
VARIANT_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "default": (),
    "tuned": ("prophet.tuning",),
    "mlx": ("prophet.mlx_backend", "mlx"),
}

# Throughput scenario: perturbed copies of one series.
THROUGHPUT_DATASET = "Peyton Manning"
THROUGHPUT_NOISE = 0.05  # noise sd as a fraction of the series' own sd


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset, its Prophet settings and the holdout used to score it."""

    name: str
    file: str
    horizon: int
    season: int
    prophet_kwargs: dict[str, Any] = field(default_factory=dict)


# Settings mirror notebooks/benchmark_all_datasets.ipynb so configs stay
# consistent across the repo. Horizons: 365 days for the long daily series,
# 90 days for the shorter pedestrian series, 24 months for monthly series and
# 576 five-minute steps (two days) for Yosemite.
DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "Peyton Manning",
        "example_wp_log_peyton_manning.csv",
        365,
        7,
        dict(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode="additive",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        ),
    ),
    DatasetSpec(
        "Wikipedia R",
        "example_wp_log_R.csv",
        365,
        7,
        dict(
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10,
            seasonality_mode="additive",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        ),
    ),
    DatasetSpec(
        "R Outliers 1",
        "example_wp_log_R_outliers1.csv",
        365,
        7,
        dict(
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=5,
            seasonality_mode="additive",
        ),
    ),
    DatasetSpec(
        "R Outliers 2",
        "example_wp_log_R_outliers2.csv",
        365,
        7,
        dict(
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=5,
            seasonality_mode="additive",
        ),
    ),
    DatasetSpec(
        "Air Passengers",
        "example_air_passengers.csv",
        24,
        12,
        dict(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        ),
    ),
    DatasetSpec(
        "Retail Sales",
        "example_retail_sales.csv",
        24,
        12,
        dict(
            changepoint_prior_scale=0.5,
            seasonality_prior_scale=15,
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        ),
    ),
    DatasetSpec(
        "Pedestrians COVID",
        "example_pedestrians_covid.csv",
        90,
        7,
        dict(
            changepoint_prior_scale=0.5,
            seasonality_prior_scale=10,
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            n_changepoints=30,
        ),
    ),
    DatasetSpec(
        "Yosemite Temps",
        "example_yosemite_temps.csv",
        576,
        288,
        dict(
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=5,
            seasonality_mode="additive",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=True,
        ),
    ),
)

# Cross-validation timing runs on one mid-sized daily series.
CV_DATASET = "Pedestrians COVID"
CV_INITIAL = "730 days"
CV_PERIOD = "90 days"
CV_HORIZON = "30 days"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def split_holdout(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered frame into train and the final ``horizon`` rows."""
    if not 0 < horizon < len(df):
        raise ValueError(f"horizon {horizon} must be in (0, {len(df)})")
    return df.iloc[:-horizon].reset_index(drop=True), df.iloc[-horizon:].reset_index(
        drop=True
    )


def seasonal_naive_scale(y_train: np.ndarray, season: int) -> float:
    """Mean absolute in-sample error of the seasonal-naive forecast."""
    lag = season if len(y_train) > season else 1
    return float(np.mean(np.abs(y_train[lag:] - y_train[:-lag])))


def compute_metrics(
    y: np.ndarray,
    yhat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scale: float,
) -> dict[str, float]:
    """Point and interval accuracy metrics on a holdout."""
    err = y - yhat
    denom = np.abs(y) + np.abs(yhat)
    smape = np.where(
        denom == 0, 0.0, 2 * np.abs(err) / np.where(denom == 0, 1.0, denom)
    )
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "smape": float(100 * np.mean(smape)),
        "mase": float(np.mean(np.abs(err)) / scale) if scale > 0 else float("nan"),
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "interval_width": float(np.mean(upper - lower)),
    }


def median_timings(runs: list[dict[str, float]]) -> dict[str, float]:
    """Median of each timing key across repeated runs."""
    return {key: statistics.median(run[key] for run in runs) for key in runs[0]}


def missing_modules(variant: str, available: dict[str, bool]) -> list[str]:
    """Modules a variant needs that are not importable, given an availability map."""
    if variant not in VARIANT_REQUIREMENTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    return [m for m in VARIANT_REQUIREMENTS[variant] if not available.get(m, False)]


def parse_throughput(text: str) -> tuple[int, ...]:
    """Parse ``"10,100"`` into series counts, rejecting non-positive values."""
    try:
        sizes = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(
            f"--throughput expects integers like 10,100: {text!r}"
        ) from exc
    if not sizes or any(n < 1 for n in sizes):
        raise ValueError(f"--throughput needs positive integers: {text!r}")
    return sizes


def tuning_fixed_kwargs(
    config: dict[str, Any], grid_names: set[str] | frozenset[str]
) -> dict[str, Any]:
    """Dataset settings that stay fixed while tuning.

    Parameters the grid searches are dropped, because ``tune`` rejects a name
    present in both places. The tuned model therefore ignores the hand-set
    value of those parameters, which is part of what the comparison measures.
    """
    return {k: v for k, v in config.items() if k not in grid_names}


def unsupported_by_mlx(config: dict[str, Any]) -> str | None:
    """Reason the MLX backend cannot fit this configuration, or ``None``."""
    if config.get("growth", "linear") == "logistic":
        return "logistic growth is not supported by the MLX backend"
    if config.get("mcmc_samples", 0) > 0:
        return "MCMC is not supported by the MLX backend (MAP only)"
    return None


def perturbed_copies(
    df: pd.DataFrame, n: int, seed: int, noise: float = THROUGHPUT_NOISE
) -> list[pd.DataFrame]:
    """``n`` copies of ``df`` with seeded Gaussian noise added to ``y``.

    Each copy has its own generator (``seed + i``), so copy ``i`` is identical
    whatever ``n`` is and the 10-series set is a prefix of the 100-series set.
    """
    sd = float(df["y"].std())
    return [
        df.assign(
            y=df["y"] + np.random.default_rng(seed + i).normal(0, noise * sd, len(df))
        )
        for i in range(n)
    ]


def relative_yhat_diff(yhat_a: np.ndarray, yhat_b: np.ndarray, y: np.ndarray) -> float:
    """Max absolute yhat difference divided by the range of ``y``."""
    y_range = float(np.max(y) - np.min(y))
    return float(np.max(np.abs(yhat_a - yhat_b)) / y_range)


def select_datasets(
    specs: tuple[DatasetSpec, ...], names: list[str] | None
) -> tuple[DatasetSpec, ...]:
    """Filter specs by case-insensitive name, failing on unknown names."""
    if not names:
        return specs
    wanted = {n.lower() for n in names}
    chosen = tuple(s for s in specs if s.name.lower() in wanted)
    unknown = wanted - {s.name.lower() for s in chosen}
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    return chosen


# ---------------------------------------------------------------------------
# I/O and measurement
# ---------------------------------------------------------------------------


def load_dataset(examples_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    """Read, clean and time-order a dataset."""
    df = pd.read_csv(examples_dir / spec.file, parse_dates=["ds"])
    return df.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)


def module_available(name: str) -> bool:
    """Whether a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        # find_spec imports the parent package, which may itself be absent.
        return False


def time_fit_predict(
    prophet_cls: type | functools.partial,
    spec: DatasetSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit once and predict the holdout, returning timings and the forecast."""
    model = prophet_cls(**spec.prophet_kwargs)
    start = time.perf_counter()
    model.fit(train)
    fit_s = time.perf_counter() - start

    np.random.seed(seed)
    start = time.perf_counter()
    forecast = model.predict(test[["ds"]])
    predict_s = time.perf_counter() - start
    return {
        "fit_s": fit_s,
        "predict_s": predict_s,
        "total_s": fit_s + predict_s,
    }, forecast


def warm_up(
    prophet_cls: type | functools.partial, spec: DatasetSpec, examples_dir: Path
) -> None:
    """Run one untimed fit so first-call costs (Stan binary load) are not measured."""
    df = load_dataset(examples_dir, spec)
    train, test = split_holdout(df, spec.horizon)
    time_fit_predict(prophet_cls, spec, train, test, seed=0)


def dataset_result(
    spec: DatasetSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    runs: list[dict[str, float]],
    forecast: pd.DataFrame,
    prophet_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Score a holdout forecast and assemble the per-dataset JSON record."""
    y = test["y"].to_numpy(dtype=float)
    yhat = forecast["yhat"].to_numpy(dtype=float)
    lower = forecast["yhat_lower"].to_numpy(dtype=float)
    upper = forecast["yhat_upper"].to_numpy(dtype=float)
    metrics = compute_metrics(
        y,
        yhat,
        lower,
        upper,
        seasonal_naive_scale(train["y"].to_numpy(dtype=float), spec.season),
    )
    holdout = pd.DataFrame(
        {
            "ds": test["ds"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "y": y,
            "yhat": yhat,
            "yhat_lower": lower,
            "yhat_upper": upper,
        }
    )
    return {
        "name": spec.name,
        "n_train": len(train),
        "horizon": spec.horizon,
        "season": spec.season,
        "prophet_kwargs": prophet_kwargs,
        "timings_all": runs,
        "timings_median": median_timings(runs),
        "metrics": metrics,
        "forecast": holdout.to_dict(orient="list"),
    }


def benchmark_dataset(
    prophet_cls: type | functools.partial,
    spec: DatasetSpec,
    examples_dir: Path,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Time and score one dataset, repeating the fit and predict ``repeats`` times."""
    df = load_dataset(examples_dir, spec)
    train, test = split_holdout(df, spec.horizon)
    runs: list[dict[str, float]] = []
    first_forecast: pd.DataFrame | None = None
    for i in range(repeats):
        LOGGER.info("%s: repeat %d/%d", spec.name, i + 1, repeats)
        timings, forecast = time_fit_predict(prophet_cls, spec, train, test, seed)
        runs.append(timings)
        if first_forecast is None:
            first_forecast = forecast
    assert first_forecast is not None
    return dataset_result(spec, train, test, runs, first_forecast, spec.prophet_kwargs)


def benchmark_tuned(
    spec: DatasetSpec,
    examples_dir: Path,
    repeats: int,
    tune_repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Tune on the training part only, then fit the winner and forecast the holdout.

    The holdout never reaches ``tune``, so the tuned scores are as honest as the
    default ones. Tuning is deterministic for a fixed seed, so repeating it only
    refines its wall time; the parameters of the first run are used.
    """
    from prophet import Prophet
    from prophet.tuning import default_param_grid, tune

    df = load_dataset(examples_dir, spec)
    train, test = split_holdout(df, spec.horizon)
    fixed = tuning_fixed_kwargs(spec.prophet_kwargs, frozenset(default_param_grid()))

    tune_durations: list[float] = []
    result = None
    for i in range(tune_repeats):
        LOGGER.info("%s: tuning %d/%d", spec.name, i + 1, tune_repeats)
        start = time.perf_counter()
        run = tune(train, prophet_kwargs=fixed, seed=seed)
        tune_durations.append(time.perf_counter() - start)
        result = result or run
    assert result is not None

    runs: list[dict[str, float]] = []
    first_forecast: pd.DataFrame | None = None
    for i in range(repeats):
        LOGGER.info("%s: tuned fit %d/%d", spec.name, i + 1, repeats)
        start = time.perf_counter()
        model = result.best_model()
        fit_s = time.perf_counter() - start
        np.random.seed(seed)
        start = time.perf_counter()
        forecast = model.predict(test[["ds"]])
        predict_s = time.perf_counter() - start
        runs.append(
            {"fit_s": fit_s, "predict_s": predict_s, "total_s": fit_s + predict_s}
        )
        if first_forecast is None:
            first_forecast = forecast
    assert first_forecast is not None

    # Untuned Prophet (library defaults for the searched parameters, dataset
    # settings for the rest) shows what the search is measured against by the
    # tuning documentation, as opposed to the hand-set settings. Not timed.
    np.random.seed(seed)
    untuned = Prophet(**fixed).fit(train).predict(test[["ds"]])
    y_test = test["y"].to_numpy(dtype=float)
    untuned_metrics = compute_metrics(
        y_test,
        untuned["yhat"].to_numpy(dtype=float),
        untuned["yhat_lower"].to_numpy(dtype=float),
        untuned["yhat_upper"].to_numpy(dtype=float),
        seasonal_naive_scale(train["y"].to_numpy(dtype=float), spec.season),
    )

    scores = result.scores
    baseline = scores[scores["is_default"]]
    record = dataset_result(
        spec, train, test, runs, first_forecast, {**fixed, **result.best_params}
    )
    record["tuning"] = {
        "durations_s": tune_durations,
        "median_s": statistics.median(tune_durations),
        "best_params": result.best_params,
        "metric": result.metric,
        "n_combinations": len(scores),
        "n_folds": result.windows.n_folds,
        "horizon": str(result.windows.horizon),
        "initial": str(result.windows.initial),
        "period": str(result.windows.period),
        "cv_score_best": float(scores["score"].iloc[0]),
        "cv_score_prophet_defaults": (
            float(baseline["score"].iloc[0]) if len(baseline) else None
        ),
        "holdout_metrics_prophet_defaults": untuned_metrics,
    }
    return record


# ---------------------------------------------------------------------------
# Many-series throughput
# ---------------------------------------------------------------------------


def in_sample_yhat(models: list[Any], dfs: list[pd.DataFrame]) -> list[np.ndarray]:
    """In-sample point forecast of each fitted model (not timed)."""
    return [
        m.predict(d[["ds"]])["yhat"].to_numpy(dtype=float)
        for m, d in zip(models, dfs, strict=True)
    ]


def benchmark_throughput(
    prophet_cls: type,
    base: pd.DataFrame,
    config: dict[str, Any],
    sizes: tuple[int, ...],
    repeats: int,
    seed: int,
    include_mlx: bool,
) -> list[dict[str, Any]]:
    """Fit N perturbed series sequentially with CmdStan and, optionally, with ``fit_many``.

    Wall time covers fitting only; the in-sample forecasts used for the
    accuracy check are computed afterwards. ``fit_many`` gets one untimed run on
    the same batch first, so one-off graph compilation is not charged to it.
    """
    # Intervals are irrelevant to fitting speed and would dominate the untimed predicts.
    config = {**config, "uncertainty_samples": 0}
    out: list[dict[str, Any]] = []
    for n in sizes:
        dfs = perturbed_copies(base, n, seed)
        record: dict[str, Any] = {"n_series": n, "n_rows": len(base)}

        durations: list[float] = []
        cmdstan_yhat: list[np.ndarray] = []
        for i in range(repeats):
            LOGGER.info("throughput N=%d: CmdStan sequential %d/%d", n, i + 1, repeats)
            start = time.perf_counter()
            models = [prophet_cls(**config).fit(d) for d in dfs]
            durations.append(time.perf_counter() - start)
            if i == 0:
                cmdstan_yhat = in_sample_yhat(models, dfs)
        record["cmdstan"] = {
            "durations_s": durations,
            "median_s": statistics.median(durations),
            "per_series_s": statistics.median(durations) / n,
        }

        if include_mlx:
            from prophet.mlx_backend import fit_many

            LOGGER.info("throughput N=%d: fit_many warm-up", n)
            fit_many(dfs, **config)
            mlx_durations: list[float] = []
            mlx_yhat: list[np.ndarray] = []
            for i in range(repeats):
                LOGGER.info("throughput N=%d: fit_many %d/%d", n, i + 1, repeats)
                start = time.perf_counter()
                mlx_models = fit_many(dfs, **config)
                mlx_durations.append(time.perf_counter() - start)
                if i == 0:
                    mlx_yhat = in_sample_yhat(mlx_models, dfs)
            rel = [
                relative_yhat_diff(a, b, d["y"].to_numpy(dtype=float))
                for a, b, d in zip(mlx_yhat, cmdstan_yhat, dfs, strict=True)
            ]
            record["mlx"] = {
                "durations_s": mlx_durations,
                "median_s": statistics.median(mlx_durations),
                "per_series_s": statistics.median(mlx_durations) / n,
                "max_rel_yhat_diff": max(rel),
                "median_rel_yhat_diff": statistics.median(rel),
            }
        out.append(record)
    return out


def benchmark_cross_validation(
    prophet_cls: type, spec: DatasetSpec, examples_dir: Path, repeats: int, seed: int
) -> dict[str, Any]:
    """Time a sequential ``cross_validation`` run on one dataset."""
    from prophet.diagnostics import cross_validation

    df = load_dataset(examples_dir, spec)
    durations: list[float] = []
    n_rows = 0
    for i in range(repeats):
        LOGGER.info("cross_validation on %s: repeat %d/%d", spec.name, i + 1, repeats)
        model = prophet_cls(**spec.prophet_kwargs)
        model.fit(df)
        np.random.seed(seed)
        start = time.perf_counter()
        cv = cross_validation(
            model,
            initial=CV_INITIAL,
            period=CV_PERIOD,
            horizon=CV_HORIZON,
            parallel=None,
        )
        durations.append(time.perf_counter() - start)
        n_rows = len(cv)
    return {
        "dataset": spec.name,
        "initial": CV_INITIAL,
        "period": CV_PERIOD,
        "horizon": CV_HORIZON,
        "n_rows": n_rows,
        "durations_s": durations,
        "median_s": statistics.median(durations),
    }


def cpu_description() -> str:
    """Best-effort CPU model name without invoking a shell."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            LOGGER.warning("sysctl failed, falling back to platform.processor()")
    return platform.processor() or platform.machine()


def mlx_version() -> str | None:
    """Installed ``mlx`` version, if any."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mlx")
    except PackageNotFoundError:
        return None


def collect_metadata(
    label: str, variant: str, repeats: int, tune_repeats: int, seed: int
) -> dict[str, Any]:
    """Describe the environment the numbers were measured in."""
    import prophet

    return {
        "label": label,
        "variant": variant,
        "prophet_version": prophet.__version__,
        "prophet_path": str(Path(prophet.__file__).resolve().parent),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "mlx_version": mlx_version(),
        "platform": platform.platform(),
        "cpu": cpu_description(),
        "cpu_count": os.cpu_count(),
        "repeats": repeats,
        "tune_repeats": tune_repeats,
        "seed": seed,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--examples-dir", type=Path, required=True, help="folder with the CSVs"
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON results path")
    parser.add_argument(
        "--label", default="prophet", help="name stored in the metadata"
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="default",
        help="default works anywhere; tuned and mlx need the NeoProphet fork",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--tune-repeats",
        type=int,
        default=DEFAULT_TUNE_REPEATS,
        help="times to repeat tuning per dataset (tuned variant)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--datasets", nargs="*", help="dataset names to run (default: all)"
    )
    parser.add_argument(
        "--skip-cv", action="store_true", help="skip the cross_validation timing run"
    )
    parser.add_argument(
        "--throughput",
        type=parse_throughput,
        default=(),
        help="comma-separated series counts for the many-series scenario, e.g. 10,100",
    )
    parser.add_argument(
        "--throughput-repeats",
        type=int,
        default=1,
        help="repeats per throughput size (sequential CmdStan is slow)",
    )
    return parser.parse_args(argv)


def check_variant(variant: str, throughput: tuple[int, ...]) -> None:
    """Fail fast when the installed prophet cannot run the requested variant."""
    available = {m: module_available(m) for m in VARIANT_REQUIREMENTS.get(variant, ())}
    missing = missing_modules(variant, available)
    if missing:
        raise SystemExit(
            f"--variant {variant} needs {missing}, which the installed prophet "
            "does not provide. Run it in the NeoProphet environment, not upstream."
        )
    if throughput and variant == "tuned":
        raise SystemExit("--throughput supports the default and mlx variants only")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # cmdstanpy is chatty at INFO and would drown the progress messages.
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    # Upstream logs an ERROR on import when plotly is absent; plotting is unused here.
    logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)

    check_variant(args.variant, args.throughput)
    from prophet import Prophet

    model_cls: type | functools.partial = (
        functools.partial(Prophet, stan_backend="MLX")
        if args.variant == "mlx"
        else Prophet
    )

    specs = select_datasets(DATASETS, args.datasets)
    skipped: list[dict[str, str]] = []
    runnable = specs
    if args.variant == "mlx":
        reasons = {s.name: unsupported_by_mlx(s.prophet_kwargs) for s in specs}
        skipped = [{"name": n, "reason": r} for n, r in reasons.items() if r]
        runnable = tuple(s for s in specs if not reasons[s.name])
    for item in skipped:
        LOGGER.warning("skipping %s: %s", item["name"], item["reason"])

    if runnable:
        LOGGER.info("warming up")
        warm_up(model_cls, runnable[0], args.examples_dir)
    if args.variant == "tuned":
        results = [
            benchmark_tuned(
                spec, args.examples_dir, args.repeats, args.tune_repeats, args.seed
            )
            for spec in runnable
        ]
    else:
        results = [
            benchmark_dataset(
                model_cls, spec, args.examples_dir, args.repeats, args.seed
            )
            for spec in runnable
        ]

    cv_spec = next((s for s in specs if s.name == CV_DATASET), None)
    cv = (
        None
        if args.skip_cv or cv_spec is None or args.variant != "default"
        else benchmark_cross_validation(
            Prophet, cv_spec, args.examples_dir, args.repeats, args.seed
        )
    )

    throughput = None
    if args.throughput:
        base_spec = next(s for s in DATASETS if s.name == THROUGHPUT_DATASET)
        throughput = benchmark_throughput(
            Prophet,
            load_dataset(args.examples_dir, base_spec),
            base_spec.prophet_kwargs,
            args.throughput,
            args.throughput_repeats,
            args.seed,
            include_mlx=args.variant == "mlx",
        )

    payload = {
        "metadata": collect_metadata(
            args.label, args.variant, args.repeats, args.tune_repeats, args.seed
        ),
        "datasets": results,
        "skipped": skipped,
        "cross_validation": cv,
        "throughput": throughput,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    LOGGER.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
