
# NeoProphet: Prophet with a Go Wrapper Service

A fork of [Facebook Prophet](https://github.com/facebook/prophet) — the additive time series forecasting library — extended with a **high-performance Go wrapper service** for production serving.

The Go service wraps the same compiled Stan binary that Python uses, reimplements data preparation and prediction math in pure Go, and exposes a REST API. Benchmarks on real datasets show **~200x faster prediction** and **~10x faster cross-validation** with identical forecast accuracy.

## Benchmark: Python vs Go

Tested on the Peyton Manning Wikipedia pageviews dataset (2,905 daily observations):

| Metric | Python | Go | Speedup |
|--------|-------:|---:|--------:|
| Fit (mean) | 0.363s | 0.252s | **1.4x** |
| Predict (mean, 3270 rows) | 0.181s | 0.001s | **199x** |
| Cross-validation (24 cutoffs) | 4.958s | 0.482s | **10.3x** |
| CV MAPE | 0.0548 | 0.0547 | **0.2% relative diff** |

Both implementations call the same compiled CmdStan binary for model fitting — the speedup comes from Go's native math for prediction and goroutine parallelism for cross-validation.

## Architecture

```
python/
  prophet/              # Original Python Prophet library
    forecaster.py       # Core Prophet class — fit(), predict()
    models.py           # CmdStanPy backend (shells out to Stan binary)
    diagnostics.py      # cross_validation(), performance_metrics()
  stan/
    prophet.stan        # The Stan model (logistic + linear trend)
go/
  cmd/
    server/             # HTTP service entrypoint
    bench/              # Single-dataset benchmark CLI
    bench_all/          # Multi-dataset benchmark CLI (JSON-driven)
  internal/
    cmdstan/            # exec.CommandContext wrapper + CSV parser
    prophet/            # Go Prophet: Fourier, changepoints, scaling, trend
    diagnostics/        # Parallel cross-validation via errgroup
    pool/               # Bounded-concurrency worker pool
  api/http/             # REST: /v1/fit, /v1/predict, /v1/cv, /v1/health
R/                      # R package (CRAN-published)
notebooks/
  benchmark_python_vs_go.ipynb    # Single-dataset benchmark with visualizations
  benchmark_all_datasets.ipynb    # All 8 datasets with hyperparameter tuning
benchmarks/             # CLI benchmark scripts
examples/               # Example datasets (CSV)
```

## Quick Start

### Python

```bash
cd python
uv pip install -e ".[dev]"
uv run pytest prophet/tests
```

```python
from prophet import Prophet
import pandas as pd

df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
```

### Go Service

```bash
# Compile the Stan model (one-time)
cd python && uv run python -c "from prophet.models import CmdStanPyBackend; CmdStanPyBackend().load_model()"

# Start the Go server
cd go
export PROPHET_STAN_BINARY=../python/prophet/stan_model/prophet_model.bin
go run ./cmd/server --http :8080
```

```bash
# Fit a model
curl -X POST http://localhost:8080/v1/fit \
  -H "Content-Type: application/json" \
  -d '{"ds": [1577836800, ...], "y": [9.59, ...], "config": {}}'

# Predict
curl -X POST http://localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "model-xxx", "periods": 365}'
```

### Run Benchmarks

```bash
# Interactive notebook (recommended)
cd python && uv run jupyter lab ../notebooks/benchmark_all_datasets.ipynb

# CLI benchmark
bash benchmarks/run_benchmark.sh
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `benchmark_python_vs_go.ipynb` | Single-dataset (Peyton Manning) benchmark with forecast visualization, component plots, MAPE comparison |
| `benchmark_all_datasets.ipynb` | All 8 example datasets with per-dataset hyperparameter tuning, forecasts gallery, aggregate speed comparison |

## Go API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/fit` | POST | Fit a Prophet model, returns model ID + parameters |
| `/v1/predict` | POST | Generate forecasts from a fitted model |
| `/v1/cv` | POST | Run cross-validation with parallel SHFs |
| `/v1/health` | GET | Service + Stan binary health check |

## How It Works

Prophet decomposes time series into three components:

```
y(t) = g(t) + s(t) + h(t) + εₜ
```

- **g(t)** — Trend: piecewise linear or logistic growth with automatic changepoints
- **s(t)** — Seasonality: Fourier series for yearly, weekly, and daily patterns
- **h(t)** — Holidays: indicator functions for irregular events
- **εₜ** — Error: normally distributed noise

Both Python and Go share the same Stan model for MAP estimation (L-BFGS optimizer). The Go service reimplements data preparation (Fourier features, changepoint matrices, y-scaling) and prediction math (trend + seasonal components) in native Go for speed.

## Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `changepoint_prior_scale` (τ) | 0.05 | Trend flexibility — higher = more changepoints active |
| `seasonality_prior_scale` (σ) | 10.0 | Seasonality strength — lower = smoother seasonal patterns |
| `seasonality_mode` | additive | `multiplicative` for series where seasonal amplitude scales with trend |
| `n_changepoints` | 25 | Number of potential changepoint locations |
| `growth` | linear | `logistic` for saturating growth, `flat` for no trend |

## References

- **Paper**: Taylor & Letham (2018). *Forecasting at Scale*. The American Statistician 72(1):37-45. [PeerJ Preprint](https://peerj.com/preprints/3190/).
- **Original repo**: [github.com/facebook/prophet](https://github.com/facebook/prophet)
- **Documentation**: [facebook.github.io/prophet](https://facebook.github.io/prophet/)

## License

Prophet is licensed under the [MIT license](LICENSE).
