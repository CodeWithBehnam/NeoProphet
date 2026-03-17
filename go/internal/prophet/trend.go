package prophet

import "math"

// PiecewiseLinear computes the piecewise linear trend at times t.
// Matches linear_trend() in prophet.stan.
//
//	trend = (k + A*delta) * t + (m + A*gamma)
//	where gamma_j = -s_j * delta_j (to ensure continuity)
//
// Parameters:
//   - k: base growth rate
//   - m: base offset
//   - delta: rate adjustments at each changepoint (length S)
//   - t: normalized time values (length T)
//   - A: changepoint matrix T x S (row-major flat)
//   - tChange: changepoint times (length S)
func PiecewiseLinear(k, m float64, delta, t, A, tChange []float64) []float64 {
	T := len(t)
	S := len(delta)
	trend := make([]float64, T)

	// Compute gamma for continuity: gamma_j = -s_j * delta_j
	gamma := make([]float64, S)
	for j := 0; j < S; j++ {
		gamma[j] = -tChange[j] * delta[j]
	}

	for i := 0; i < T; i++ {
		rate := k
		offset := m
		for j := 0; j < S; j++ {
			aij := A[i*S+j]
			rate += aij * delta[j]
			offset += aij * gamma[j]
		}
		trend[i] = rate*t[i] + offset
	}

	return trend
}

// PiecewiseLogistic computes the piecewise logistic (S-curve) trend.
// Matches logistic_trend() in prophet.stan.
//
//	trend = cap / (1 + exp(-(k + A*delta) * (t - (m + A*gamma))))
//
// Parameters:
//   - k: base growth rate
//   - m: base offset
//   - delta: rate adjustments at each changepoint (length S)
//   - t: normalized time values (length T)
//   - cap: capacity values (length T)
//   - A: changepoint matrix T x S (row-major flat)
//   - tChange: changepoint times (length S)
func PiecewiseLogistic(k, m float64, delta, t, cap, A, tChange []float64) []float64 {
	T := len(t)
	S := len(delta)
	trend := make([]float64, T)

	// Compute gamma for logistic continuity
	// gamma_j = (s_j - m - sum_{l<j} gamma_l) * (1 - (k + sum_{l<j} delta_l) / (k + sum_{l<=j} delta_l))
	gamma := make([]float64, S)
	for j := 0; j < S; j++ {
		var sumGammaPrev float64
		var sumDeltaPrev float64
		for l := 0; l < j; l++ {
			sumGammaPrev += gamma[l]
			sumDeltaPrev += delta[l]
		}
		sumDeltaIncl := sumDeltaPrev + delta[j]
		rateAtJ := k + sumDeltaIncl
		if rateAtJ == 0 {
			gamma[j] = 0
			continue
		}
		gamma[j] = (tChange[j] - m - sumGammaPrev) * (1 - (k+sumDeltaPrev)/rateAtJ)
	}

	for i := 0; i < T; i++ {
		rate := k
		offset := m
		for j := 0; j < S; j++ {
			aij := A[i*S+j]
			rate += aij * delta[j]
			offset += aij * gamma[j]
		}
		trend[i] = cap[i] / (1 + math.Exp(-rate*(t[i]-offset)))
	}

	return trend
}

// FlatTrend returns a constant trend of value m for all T points.
func FlatTrend(m float64, T int) []float64 {
	trend := make([]float64, T)
	for i := range trend {
		trend[i] = m
	}
	return trend
}
