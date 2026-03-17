package prophet

import (
	"math"
	"testing"
)

func TestPiecewiseLinear_NoChangepoints(t *testing.T) {
	// Simple linear trend: y = 2*t + 1
	k := 2.0
	m := 1.0
	tvals := []float64{0, 0.25, 0.5, 0.75, 1.0}
	var delta, A, tChange []float64

	trend := PiecewiseLinear(k, m, delta, tvals, A, tChange)

	expected := []float64{1.0, 1.5, 2.0, 2.5, 3.0}
	for i, v := range trend {
		assertNear(t, v, expected[i], "PiecewiseLinear no cp")
	}
}

func TestPiecewiseLinear_OneChangepoint(t *testing.T) {
	k := 1.0
	m := 0.0
	delta := []float64{1.0}  // rate doubles at changepoint
	tChange := []float64{0.5} // changepoint at midpoint
	tvals := []float64{0, 0.25, 0.5, 0.75, 1.0}

	A := MakeChangepointMatrix(tvals, tChange)
	trend := PiecewiseLinear(k, m, delta, tvals, A, tChange)

	// Before cp (t<0.5): rate=1, offset=0 → trend = t
	assertNear(t, trend[0], 0.0, "t=0")
	assertNear(t, trend[1], 0.25, "t=0.25")
	// At cp (t=0.5): rate=2, gamma = -0.5*1 = -0.5, offset = -0.5
	// trend = 2*0.5 + (-0.5) = 0.5
	assertNear(t, trend[2], 0.5, "t=0.5")
	// After cp: trend = 2*0.75 + (-0.5) = 1.0
	assertNear(t, trend[3], 1.0, "t=0.75")
	assertNear(t, trend[4], 1.5, "t=1.0")
}

func TestPiecewiseLogistic_NoChangepoints(t *testing.T) {
	k := 1.0
	m := 0.0
	cap := []float64{10, 10, 10, 10, 10}
	tvals := []float64{-2, -1, 0, 1, 2}
	var delta, A, tChange []float64

	trend := PiecewiseLogistic(k, m, delta, tvals, cap, A, tChange)

	// Standard logistic: 10 / (1 + exp(-t))
	for i, tv := range tvals {
		expected := 10.0 / (1.0 + math.Exp(-tv))
		assertNear(t, trend[i], expected, "logistic")
	}
}

func TestFlatTrend(t *testing.T) {
	trend := FlatTrend(5.0, 3)
	for i, v := range trend {
		assertNear(t, v, 5.0, "flat")
		_ = i
	}
}
