#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STAN_BINARY="$PROJECT_DIR/python/prophet/stan_model/prophet_model.bin"
N_DAYS="${1:-1095}"

echo "========================================"
echo "  Prophet Benchmark: Python vs Go"
echo "  Dataset: ${N_DAYS} days synthetic"
echo "========================================"
echo ""

# --- Python ---
echo ">>> Running Python benchmark..."
echo ""
cd "$PROJECT_DIR/python"
uv run python "$SCRIPT_DIR/bench_python.py" "$N_DAYS"
echo ""

# --- Go ---
echo ">>> Running Go benchmark..."
echo ""
cd "$PROJECT_DIR/go"
PROPHET_STAN_BINARY="$STAN_BINARY" go run ./cmd/bench "$N_DAYS"
echo ""

# --- Compare ---
echo "========================================"
echo "  Side-by-Side Comparison"
echo "========================================"
python3 -c "
import json

with open('/tmp/prophet_bench_python.json') as f:
    py = json.load(f)
with open('/tmp/prophet_bench_go.json') as f:
    go = json.load(f)

print(f'')
print(f'{'Metric':<25} {'Python':>12} {'Go':>12} {'Speedup':>10}')
print(f'{\"\":-<25} {\"\":-<12} {\"\":-<12} {\"\":-<10}')

def row(label, pk, gk, unit='s'):
    pv = py[pk]
    gv = go[gk]
    if isinstance(pv, (int,)):
        print(f'{label:<25} {pv:>12} {gv:>12} {\"\":>10}')
    else:
        speedup = pv / gv if gv > 0 else 0
        print(f'{label:<25} {pv:>11.3f}s {gv:>11.3f}s {speedup:>9.1f}x')

row('Fit (mean)',          'fit_mean_s',     'fit_mean_s')
row('Predict (mean)',      'predict_mean_s', 'predict_mean_s')
row('Cross-validation',    'cv_time_s',      'cv_time_s')
print(f'{\"CV cutoffs\":<25} {py[\"cv_cutoffs\"]:>12} {go[\"cv_cutoffs\"]:>12}')
print(f'{\"CV MAPE\":<25} {py[\"cv_mape\"]:>11.4f}  {go[\"cv_mape\"]:>11.4f}')
"
