"""Benchmark Python Prophet: fit, predict, and cross-validation."""

import time
import json
import sys
import numpy as np
import pandas as pd

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


def generate_dataset(n_days: int = 1095) -> pd.DataFrame:
    """Generate a synthetic daily time series with trend + seasonality + noise."""
    np.random.seed(42)
    ds = pd.date_range("2020-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)

    # Linear trend
    trend = 100 + 0.05 * t

    # Yearly seasonality (Fourier, 2 terms)
    yearly = (
        10 * np.sin(2 * np.pi * t / 365.25)
        + 5 * np.cos(2 * np.pi * t / 365.25)
        + 3 * np.sin(4 * np.pi * t / 365.25)
    )

    # Weekly seasonality
    weekly = 5 * np.sin(2 * np.pi * t / 7)

    noise = np.random.normal(0, 3, n_days)
    y = trend + yearly + weekly + noise

    return pd.DataFrame({"ds": ds, "y": y})


def benchmark_fit(df: pd.DataFrame, n_runs: int = 3) -> dict:
    """Benchmark fit() timing."""
    times = []
    model = None
    for i in range(n_runs):
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        start = time.perf_counter()
        m.fit(df)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  fit run {i+1}/{n_runs}: {elapsed:.3f}s")
        if model is None:
            model = m
    return {"fit_times": times, "fit_mean": np.mean(times), "model": model}


def benchmark_predict(model: Prophet, periods: int = 365, n_runs: int = 5) -> dict:
    """Benchmark predict() timing."""
    future = model.make_future_dataframe(periods=periods)
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        forecast = model.predict(future)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  predict run {i+1}/{n_runs}: {elapsed:.3f}s")
    return {"predict_times": times, "predict_mean": np.mean(times), "n_rows": len(future)}


def benchmark_cv(df: pd.DataFrame) -> dict:
    """Benchmark cross_validation() timing."""
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(df)

    start = time.perf_counter()
    cv_results = cross_validation(
        m, initial="365 days", period="90 days", horizon="90 days"
    )
    cv_time = time.perf_counter() - start

    metrics = performance_metrics(cv_results)
    mape = metrics["mape"].mean()
    n_cutoffs = cv_results["cutoff"].nunique()

    print(f"  cv: {cv_time:.3f}s ({n_cutoffs} cutoffs, MAPE={mape:.4f})")
    return {"cv_time": cv_time, "n_cutoffs": n_cutoffs, "mape": mape}


def export_dataset(df: pd.DataFrame, path: str):
    """Export dataset as JSON for Go benchmark to consume."""
    records = []
    for _, row in df.iterrows():
        records.append({
            "ds": row["ds"].timestamp(),
            "y": float(row["y"]),
        })
    with open(path, "w") as f:
        json.dump(records, f)
    print(f"  exported {len(records)} rows to {path}")


def main():
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 1095
    print(f"=== Python Prophet Benchmark ({n_days} days) ===\n")

    df = generate_dataset(n_days)
    export_dataset(df, "/tmp/prophet_bench_data.json")

    print("\n--- Fit ---")
    fit_result = benchmark_fit(df, n_runs=3)

    print("\n--- Predict (365 days ahead) ---")
    pred_result = benchmark_predict(fit_result["model"], periods=365, n_runs=5)

    print("\n--- Cross-Validation ---")
    cv_result = benchmark_cv(df)

    # Summary
    results = {
        "language": "python",
        "n_days": n_days,
        "fit_mean_s": fit_result["fit_mean"],
        "predict_mean_s": pred_result["predict_mean"],
        "predict_n_rows": pred_result["n_rows"],
        "cv_time_s": cv_result["cv_time"],
        "cv_cutoffs": cv_result["n_cutoffs"],
        "cv_mape": cv_result["mape"],
    }

    print("\n=== Summary ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open("/tmp/prophet_bench_python.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /tmp/prophet_bench_python.json")


if __name__ == "__main__":
    main()
