package prophet

// MakeChangepointMatrix builds the binary indicator matrix A where
// A[i][j] = 1 if t[i] >= tChange[j], else 0.
// This matches the get_changepoint_matrix function in prophet.stan.
// Returns a flat T x S slice in row-major order.
func MakeChangepointMatrix(t []float64, tChange []float64) []float64 {
	T := len(t)
	S := len(tChange)
	if S == 0 {
		return nil
	}

	A := make([]float64, T*S)
	for i, tv := range t {
		for j, tc := range tChange {
			if tv >= tc {
				A[i*S+j] = 1.0
			}
		}
	}
	return A
}

// SelectChangepoints places n changepoints uniformly over the first
// changepointRange fraction of the normalized time values t.
// Returns the changepoint times in the normalized t space.
func SelectChangepoints(t []float64, n int, changepointRange float64) []float64 {
	if len(t) == 0 || n <= 0 {
		return nil
	}

	// Find the index that covers changepointRange of the data
	cutIdx := int(float64(len(t)) * changepointRange)
	if cutIdx < 2 {
		cutIdx = 2
	}
	if cutIdx > len(t) {
		cutIdx = len(t)
	}

	tRange := t[:cutIdx]

	// If we have fewer points than changepoints, reduce
	if n >= len(tRange) {
		n = len(tRange) - 1
	}
	if n <= 0 {
		return nil
	}

	// Place changepoints at uniform quantiles
	changepoints := make([]float64, n)
	for i := 0; i < n; i++ {
		// Evenly spaced indices within tRange
		idx := int(float64(i+1) * float64(len(tRange)-1) / float64(n+1))
		if idx >= len(tRange) {
			idx = len(tRange) - 1
		}
		changepoints[i] = tRange[idx]
	}

	return changepoints
}
