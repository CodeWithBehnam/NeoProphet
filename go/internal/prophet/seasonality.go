package prophet

import "math"

// FourierFeatures generates a Fourier series feature matrix for seasonal modeling.
// For each time value t (in days), it produces 2*order columns:
//
//	cos(2*pi*1*t/period), sin(2*pi*1*t/period), ..., cos(2*pi*N*t/period), sin(2*pi*N*t/period)
//
// Returns a T x (2*order) matrix stored as a flat slice in row-major order.
func FourierFeatures(t []float64, period float64, order int) []float64 {
	n := len(t)
	cols := 2 * order
	features := make([]float64, n*cols)

	for i, tv := range t {
		for j := 1; j <= order; j++ {
			x := 2.0 * math.Pi * float64(j) * tv / period
			features[i*cols+(j-1)*2] = math.Cos(x)
			features[i*cols+(j-1)*2+1] = math.Sin(x)
		}
	}

	return features
}

// FourierFeaturesForDays generates Fourier features where t values are
// already in days (the standard time unit for Prophet).
func FourierFeaturesForDays(tDays []float64, period float64, order int) []float64 {
	return FourierFeatures(tDays, period, order)
}

// MakeSeasonalityFeatures builds the combined seasonality design matrix X
// from a list of seasonality specs and normalized time values.
// tDays contains time values in days.
// Returns the matrix X (T x K) in row-major, the total number of columns K,
// and parallel slices sA (additive indicators) and sM (multiplicative indicators).
func MakeSeasonalityFeatures(tDays []float64, specs []SeasonalitySpec) (X []float64, K int, sA, sM []float64, sigmas []float64) {
	T := len(tDays)
	if len(specs) == 0 {
		return nil, 0, nil, nil, nil
	}

	// Calculate total columns
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

		// Copy features into the correct columns of X
		for i := 0; i < T; i++ {
			for j := 0; j < ncols; j++ {
				X[i*K+col+j] = features[i*ncols+j]
			}
		}

		// Set mode indicators and prior scales
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
