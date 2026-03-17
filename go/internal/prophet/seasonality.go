package prophet

import "math"

// FourierFeatures generates a Fourier series feature matrix for seasonal modeling.
// t values must be in days since Unix epoch (matching Python Prophet's convention).
// For each time value t, it produces 2*order columns:
//
//	sin(2*pi*1*t/period), cos(2*pi*1*t/period), ..., sin(2*pi*N*t/period), cos(2*pi*N*t/period)
//
// Returns a T x (2*order) matrix stored as a flat slice in row-major order.
func FourierFeatures(tDays []float64, period float64, order int) []float64 {
	n := len(tDays)
	cols := 2 * order
	features := make([]float64, n*cols)

	for i, tv := range tDays {
		xT := 2.0 * math.Pi * tv
		for j := 1; j <= order; j++ {
			c := float64(j) / period * xT
			features[i*cols+(j-1)*2] = math.Sin(c)
			features[i*cols+(j-1)*2+1] = math.Cos(c)
		}
	}

	return features
}

// MakeSeasonalityFeatures builds the combined seasonality design matrix X
// from a list of seasonality specs and time values in days since Unix epoch.
// Returns the matrix X (T x K) in row-major, the total number of columns K,
// and parallel slices sA (additive indicators), sM (multiplicative indicators),
// and sigmas (prior scales).
func MakeSeasonalityFeatures(tDays []float64, specs []SeasonalitySpec) (X []float64, K int, sA, sM []float64, sigmas []float64) {
	T := len(tDays)
	if len(specs) == 0 {
		return nil, 0, nil, nil, nil
	}

	for _, s := range specs {
		K += 2 * s.FourierOrder
	}

	X = make([]float64, T*K)
	sA = make([]float64, K)
	sM = make([]float64, K)
	sigmas = make([]float64, K)

	col := 0
	for _, spec := range specs {
		features := FourierFeatures(tDays, spec.Period, spec.FourierOrder)
		ncols := 2 * spec.FourierOrder

		for i := 0; i < T; i++ {
			for j := 0; j < ncols; j++ {
				X[i*K+col+j] = features[i*ncols+j]
			}
		}

		for j := 0; j < ncols; j++ {
			if spec.Mode == ModeMultiplicative {
				sM[col+j] = 1.0
			} else {
				sA[col+j] = 1.0
			}
			sigmas[col+j] = spec.PriorScale
		}

		col += ncols
	}

	return X, K, sA, sM, sigmas
}
