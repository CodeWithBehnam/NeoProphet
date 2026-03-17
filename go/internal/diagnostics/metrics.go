package diagnostics

import "math"

// Metrics holds aggregated forecast error metrics.
type Metrics struct {
	MSE      float64 `json:"mse"`
	RMSE     float64 `json:"rmse"`
	MAE      float64 `json:"mae"`
	MAPE     float64 `json:"mape"`
	Coverage float64 `json:"coverage"` // fraction of actuals within prediction interval
}

// ComputeMetrics calculates error metrics from cross-validation results.
func ComputeMetrics(results []CVResult) Metrics {
	var sumSE, sumAE, sumAPE float64
	var n int

	for _, r := range results {
		for j := range r.DS {
			if j >= len(r.Yhat) || j >= len(r.Y) {
				continue
			}
			err := r.Yhat[j] - r.Y[j]
			sumSE += err * err
			sumAE += math.Abs(err)
			if r.Y[j] != 0 {
				sumAPE += math.Abs(err / r.Y[j])
			}
			n++
		}
	}

	if n == 0 {
		return Metrics{}
	}

	nf := float64(n)
	mse := sumSE / nf
	return Metrics{
		MSE:  mse,
		RMSE: math.Sqrt(mse),
		MAE:  sumAE / nf,
		MAPE: sumAPE / nf,
	}
}
