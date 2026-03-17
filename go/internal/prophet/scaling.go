package prophet

import "math"

// ScaleParams holds the scaling factors computed during preprocessing.
type ScaleParams struct {
	YScale float64 `json:"y_scale"`
	YMin   float64 `json:"y_min"`
	TStart float64 `json:"t_start"` // min timestamp in seconds
	TScale float64 `json:"t_scale"` // range of timestamps in seconds
	Floor  float64 `json:"floor"`   // logistic floor (0 if none)
}

// ScaleY normalizes y values using either absmax or minmax scaling.
// Returns the scaled values and the computed ScaleParams.
func ScaleY(y []float64, scaling string) ([]float64, ScaleParams) {
	if len(y) == 0 {
		return nil, ScaleParams{}
	}

	params := ScaleParams{}
	scaled := make([]float64, len(y))

	switch scaling {
	case "minmax":
		minY, maxY := y[0], y[0]
		for _, v := range y[1:] {
			if v < minY {
				minY = v
			}
			if v > maxY {
				maxY = v
			}
		}
		params.YScale = maxY - minY
		params.YMin = minY
		if params.YScale == 0 {
			params.YScale = 1.0
		}
		for i, v := range y {
			scaled[i] = (v - minY) / params.YScale
		}

	default: // "absmax"
		maxAbs := 0.0
		for _, v := range y {
			a := math.Abs(v)
			if a > maxAbs {
				maxAbs = a
			}
		}
		params.YScale = maxAbs
		params.YMin = 0.0
		if params.YScale == 0 {
			params.YScale = 1.0
		}
		for i, v := range y {
			scaled[i] = v / params.YScale
		}
	}

	return scaled, params
}

// NormalizeTime converts unix timestamps (seconds) to the [0, 1] range.
// Returns the normalized values and updates the ScaleParams with tStart and tScale.
func NormalizeTime(ds []float64) ([]float64, float64, float64) {
	if len(ds) == 0 {
		return nil, 0, 0
	}

	minT, maxT := ds[0], ds[0]
	for _, v := range ds[1:] {
		if v < minT {
			minT = v
		}
		if v > maxT {
			maxT = v
		}
	}

	tScale := maxT - minT
	if tScale == 0 {
		tScale = 1.0
	}

	t := make([]float64, len(ds))
	for i, v := range ds {
		t[i] = (v - minT) / tScale
	}

	return t, minT, tScale
}

// ScaleLogisticCap normalizes cap values for logistic growth.
// cap_scaled = (cap - floor) / y_scale
func ScaleLogisticCap(cap []float64, floor, yScale float64) []float64 {
	scaled := make([]float64, len(cap))
	for i, c := range cap {
		scaled[i] = (c - floor) / yScale
	}
	return scaled
}
