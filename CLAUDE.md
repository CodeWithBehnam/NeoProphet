# Project: Prophet (NeoProphet fork)

Additive time series forecasting library (trend + seasonality + holidays) backed by Stan, with a Go wrapper service for high-throughput serving.

## Stack

- Python >=3.10 (forecaster library), Go 1.25+ (wrapper service), R (CRAN package)
- Stan via cmdstanpy >=1.0.4 — shared compiled binary used by both Python and Go
- Python build: setuptools + wheel; Go build: `go build`
- Python deps: numpy, pandas, matplotlib, holidays, tqdm
- Type checker: pyrefly 0.50.0; Tests: pytest (Python), `go test` (Go)
- CI: GitHub Actions (`build-and-test.yml`, `wheel.yml`)

## Architecture

```
python/
  prophet/
    forecaster.py       # Core Prophet class — fit(), predict(), trend/seasonality
    models.py           # CmdStanPy backend interface (shells out to Stan binary)
    diagnostics.py      # cross_validation(), performance_metrics()
    plot.py             # Plotting (matplotlib + plotly), NeoProphet default style
    serialize.py        # JSON model serialization
    make_holidays.py    # Holiday calendar generation
    utilities.py        # Regressor and warm-start helpers
    quick.py            # forecast(): one-call API, input normalisation, frequency inference
    tuning.py           # tune(): parallel cross-validated hyperparameter search
    mlx_backend.py      # Experimental MLX backend (MAP only) and batched fit_many()
    tests/              # pytest suite
  stan/
    prophet.stan        # The Stan model (logistic + linear trend)
go/
  cmd/
    server/main.go      # HTTP service entrypoint (--stan, --http flags)
    bench/main.go       # Benchmark: Go vs Python comparison
  internal/
    cmdstan/            # exec.CommandContext wrapper for Stan binary + CSV parser
    prophet/            # Go Prophet: Fourier, changepoints, scaling, trend, fit/predict
    diagnostics/        # Parallel cross-validation via errgroup worker pool
    pool/               # Generic bounded-concurrency worker pool
  api/http/server.go    # REST: POST /v1/fit, /v1/predict, /v1/cv, GET /v1/health
R/                      # R package (separate CRAN-published package)
benchmarks/             # Python benchmark script + comparison runner
```

## Commands

```bash
uv sync                                                    # Root uv workspace: .venv, editable prophet, dev tools, mlx extra
uv run --directory python pytest prophet/tests             # Python tests
uv run --directory python pyrefly check prophet            # Python type check (ignore prophet/tests and prophet/stan_model output)
cd go && go build ./...                                    # Build Go service
cd go && go test ./...                                     # Go tests
cd go && PROPHET_STAN_BINARY=path/to/bin go run ./cmd/server --http :8080  # Run Go server
cd go && PROPHET_STAN_BINARY=path/to/bin go run ./cmd/bench 1095           # Run benchmark
bash benchmarks/run_benchmark.sh 1095                      # Full Python vs Go comparison
```

## Code Conventions

- Python API: `Prophet` class from `prophet.forecaster`; Go API: `prophet.Forecaster` with `Fit()`/`Predict()`
- Both Python and Go shell out to the same compiled Stan binary — the model is never reimplemented, with one deliberate exception: the experimental MLX backend (`python/prophet/mlx_backend.py`) re-implements the MAP objective of `prophet.stan` for batched GPU fitting. It is opt-in only (`Prophet(stan_backend="MLX")` or `fit_many`) and never chosen as a default backend
- High-level API lives in `quick.py` (`forecast`) and `tuning.py` (`tune`); both are additive wrappers over `Prophet` and must not change its behaviour
- Prediction math (trend + Fourier) is reimplemented in Go for speed (~195x faster than Python)
- Cross-validation uses `Fitter` interface in Go — any fit/predict implementation can be swapped in
- Go packages follow one-responsibility-per-package: `cmdstan/`, `prophet/`, `diagnostics/`, `pool/`

## Constraints

NEVER edit `prophet.stan` without running both Python and Go test suites — Stan compilation is slow and errors surface late. Mirror any objective change in `mlx_backend.py` and run `prophet/tests/test_mlx_backend.py` (its parity tests compare against CmdStan).
NEVER run `ruff format` over the whole package. Legacy upstream files keep their original formatting; format only new files you own.
NEVER import `cmdstan` directly in Python — always go through `models.py` backend abstraction.
NEVER pass `data file=path` as a single arg to CmdStan in Go — must be two args: `"data"`, `"file=path"`.
NEVER vendor holiday data — it's generated from the `holidays` package at release time.
ALWAYS run `pyrefly check prophet` before committing Python changes — CI enforces it.
ALWAYS use `os.MkdirTemp` per Stan invocation in Go — concurrent runs share no temp files.
ALWAYS keep Go files under 500 lines — split by concern (trend.go, seasonality.go, etc.).

## When Adding a New Feature to the Forecaster

1. Implement in `python/prophet/forecaster.py` (the `Prophet` class)
2. If it changes the Stan model, edit `python/stan/prophet.stan`
3. Port the data prep / prediction math to `go/internal/prophet/`
4. Add tests: `python/prophet/tests/test_prophet.py` + `go/internal/prophet/*_test.go`
5. Run `cd python && uv run pytest prophet/tests`
6. Run `cd python && uv run pyrefly check prophet`
7. Run `cd go && go test ./...`
