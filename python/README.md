# NeoProphet

**Prophet forecasting that is easier to use, clearer to read and faster at scale.**

NeoProphet is a drop-in fork of [Facebook Prophet](https://github.com/facebook/prophet) with a one-call API, self-explaining charts, built-in tuning and an experimental Apple Silicon GPU backend for fitting many series at once.

![A NeoProphet forecast with history and forecast zones, changepoint callouts and a plain-English title](https://raw.githubusercontent.com/CodeWithBehnam/NeoProphet/main/.github/assets/forecast.png)

## Install

```bash
pip install neoprophet            # or: uv add neoprophet
pip install "neoprophet[mlx]"     # adds the MLX backend on Apple Silicon
```

NeoProphet installs the `prophet` import package, so it replaces upstream Prophet: uninstall `prophet` first and do not install both in one environment. Wheels include the compiled Stan model for macOS (Intel and Apple Silicon), Linux (x86-64 and ARM64) and Windows.

## Quick start

```python
import pandas as pd
from prophet import forecast

df = pd.read_csv("https://raw.githubusercontent.com/CodeWithBehnam/NeoProphet/main/examples/example_wp_log_peyton_manning.csv")
result = forecast(df, horizon="12 months", country_holidays="US")
result.forecast            # ds, yhat, yhat_lower, yhat_upper for the future
result.plot()              # the chart above
result.plot_components()   # trend, holidays, weekly and yearly patterns
```

- `forecast(..., tune=True)` chooses the prior scales and seasonality mode by parallel cross-validation.
- `prophet.mlx_backend.fit_many(dfs)` fits many series in one batch on an Apple Silicon GPU (experimental; linear and flat growth, MAP only).
- The classic API is unchanged: `from prophet import Prophet` works as before.

## Links

- Source, benchmarks and notebooks: https://github.com/CodeWithBehnam/NeoProphet
- Issues: https://github.com/CodeWithBehnam/NeoProphet/issues
- Upstream Prophet documentation: https://facebook.github.io/prophet/
- Paper: Taylor and Letham (2018), *Forecasting at Scale*, The American Statistician 72(1):37-45.

Released under the MIT licence. Prophet was created by Sean J. Taylor and Ben Letham at Facebook.
